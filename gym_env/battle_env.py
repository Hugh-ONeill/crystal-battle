# CrystalBattleEnv: Gymnasium environment for Gen 2 Pokemon battles

from __future__ import annotations

import random
from typing import Any, Callable

import gymnasium
import numpy as np
from gymnasium import spaces

from engine.actions import Action, Struggle, Switch, UseMove
from engine.battle_state import BattleState
from engine.data_loader import DataStore
from engine.player_state import PlayerState
from engine.stat_stages import MOVE_STAT_EFFECTS
from engine.turn_engine import resolve_forced_switches, resolve_turn
from engine.types import TypeChart

# status moves that boost the user's stats
_SELF_BOOST_MOVES = frozenset(
    name for name, effects in MOVE_STAT_EFFECTS.items()
    if any(target == "self" and stages > 0 for _, stages, target in effects)
)
# hazard / screen move IDs
_SETUP_MOVES = frozenset(["Spikes", "Reflect", "Light Screen", "Safeguard"])

from .obs_builder import OBS_SIZE, build_observation
from .reward import _matchup_score, compute_reward
from .team_builder import build_team

MAX_TURNS = 200


def random_opponent_policy(state: PlayerState, opp: PlayerState, rng: random.Random, **kwargs) -> Action:
    """Pick a random valid action."""
    actions = state.valid_actions(opp)
    return rng.choice(actions)


class CrystalBattleEnv(gymnasium.Env):
    """Gen 2 Pokemon Crystal 6v6 singles battle environment."""

    metadata = {"render_modes": ["ansi"]}

    def __init__(
        self,
        opponent_policy: Callable | None = None,
        render_mode: str | None = None,
        seed: int | None = None,
        reward_mode: str = "shaped",
        tier: str = "ou",
        opp_team_strategy: str | None = None,
    ):
        super().__init__()

        self.observation_space = spaces.Box(
            low=-1.0, high=np.inf, shape=(OBS_SIZE,), dtype=np.float32,
        )
        self.action_space = spaces.Discrete(10)

        self._data = DataStore()
        self._type_chart = TypeChart.load()
        self._opponent_policy = opponent_policy or random_opponent_policy
        self.render_mode = render_mode
        self._reward_mode = reward_mode
        self._tier = tier
        self._opp_team_strategy = opp_team_strategy

        self._rng = random.Random(seed)
        self._battle: BattleState | None = None
        self._prev_hp_diff = 0.0
        self._events_log: list[str] = []
        self._pending_p2_switch: Switch | None = None

    def action_masks(self) -> np.ndarray:
        """Return valid action mask for MaskablePPO."""
        if self._battle is None or self._battle.is_over:
            return np.ones(10, dtype=bool)
        return np.array(self._battle.p1.valid_action_mask(
            self._battle.p2, type_chart=self._type_chart), dtype=bool)

    def reset(
        self, *, seed: int | None = None, options: dict | None = None
    ) -> tuple[np.ndarray, dict]:
        if seed is not None:
            self._rng = random.Random(seed)

        # derive independent RNGs for each team to avoid P1/P2 bias
        p1_rng = random.Random(self._rng.randint(0, 2**32))
        p2_rng = random.Random(self._rng.randint(0, 2**32))

        # allow overriding p1 team via options
        if options and "p1_team" in options:
            team1 = options["p1_team"]
        else:
            team1 = build_team(self._data, rng=p1_rng, tier=self._tier)

        # opponent policy can provide its own team builder (e.g. MixedOpponent)
        if hasattr(self._opponent_policy, "build_team"):
            team2 = self._opponent_policy.build_team(self._data, p2_rng)
        else:
            team2 = build_team(self._data, rng=p2_rng, tier=self._tier,
                               strategy=self._opp_team_strategy)

        self._battle = BattleState(
            p1=PlayerState(team=team1),
            p2=PlayerState(team=team2),
            rng=random.Random(self._rng.randint(0, 2**32)),
        )
        self._prev_hp_diff = 0.0
        self._events_log = []
        self._pending_p2_switch = None

        obs = build_observation(
            self._battle.p1, self._battle.p2,
            self._type_chart, self._battle.turn,
            weather=self._battle.weather,
            weather_turns=self._battle.weather_turns,
        )
        info = {"action_mask": self.action_masks()}
        return obs, info

    def step(self, action: int) -> tuple[np.ndarray, float, bool, bool, dict]:
        assert self._battle is not None and not self._battle.is_over

        p1_action = self._decode_action(action, self._battle.p1, self._battle.p2)

        # ---- Forced switch phase (P1 mon fainted last step) ----
        if self._battle.p1.must_switch:
            if not isinstance(p1_action, Switch):
                p1_action = self._pick_forced_switch(self._battle.p1)
            switch_events = resolve_forced_switches(
                self._battle, p1_action, self._pending_p2_switch,
            )
            self._pending_p2_switch = None
            return self._build_step_result(
                switch_events, action_type="forced_switch",
                switch_matchup_delta=0.0,
            )

        # ---- Normal turn phase ----

        # classify action for diagnostics
        if isinstance(p1_action, Struggle):
            action_type = "struggle"
        elif isinstance(p1_action, Switch):
            action_type = "switch"
        elif isinstance(p1_action, UseMove):
            tmpl = self._battle.p1.active.move_slots[p1_action.slot_index].template
            if tmpl.damage_class != "status":
                action_type = "damage"
            elif tmpl.name in _SELF_BOOST_MOVES or tmpl.name in _SETUP_MOVES:
                action_type = "setup"
            elif tmpl.meta and tmpl.meta.get("ailment_id", 0) > 0:
                action_type = "status"
            else:
                action_type = "other"  # weather, recovery, misc
        else:
            action_type = "damage"

        # snapshot matchup score before the turn resolves (for switch reward)
        pre_matchup = _matchup_score(
            self._battle.p1.active, self._battle.p2.active, self._type_chart,
        )

        p2_action = self._opponent_policy(
            self._battle.p2, self._battle.p1, self._battle.rng,
            turn=self._battle.turn,
            weather=self._battle.weather,
            weather_turns=self._battle.weather_turns,
        )

        events = resolve_turn(
            self._battle, p1_action, p2_action, self._type_chart,
        )

        # handle forced switches after faints
        # P2 decides immediately; P1 gets to choose on the next step() call
        p2_switch = None
        if self._battle.p2.must_switch:
            p2_switch_action = self._opponent_policy(
                self._battle.p2, self._battle.p1, self._battle.rng,
                turn=self._battle.turn,
                weather=self._battle.weather,
                weather_turns=self._battle.weather_turns,
            )
            if isinstance(p2_switch_action, Switch):
                p2_switch = p2_switch_action
            else:
                p2_switch = self._pick_forced_switch(self._battle.p2)

        if self._battle.p1.must_switch:
            # P1 needs to choose — defer to next step() call
            # resolve P2's switch now if only P2 needs one
            if p2_switch and not self._battle.p1.must_switch:
                switch_events = resolve_forced_switches(
                    self._battle, None, p2_switch,
                )
                events.extend(switch_events)
            else:
                # both need to switch, or just P1: stash P2's for later
                self._pending_p2_switch = p2_switch
        elif p2_switch:
            # only P2 needs to switch
            switch_events = resolve_forced_switches(
                self._battle, None, p2_switch,
            )
            events.extend(switch_events)

        # matchup improvement from switching (only count voluntary switches)
        switch_matchup_delta = 0.0
        if action_type == "switch" and not self._battle.is_over:
            post_matchup = _matchup_score(
                self._battle.p1.active, self._battle.p2.active, self._type_chart,
            )
            switch_matchup_delta = post_matchup - pre_matchup

        return self._build_step_result(
            events, action_type=action_type,
            switch_matchup_delta=switch_matchup_delta,
        )

    def _build_step_result(
        self, events: list, action_type: str, switch_matchup_delta: float,
    ) -> tuple[np.ndarray, float, bool, bool, dict]:
        terminated = self._battle.is_over
        truncated = not terminated and self._battle.turn >= MAX_TURNS

        reward, self._prev_hp_diff = compute_reward(
            self._battle.p1, self._battle.p2,
            self._prev_hp_diff, self._battle.winner,
            is_p1=True, truncated=truncated,
            events=events, opp_player=2,
            switch_matchup_delta=switch_matchup_delta,
            action_type=action_type,
            mode=self._reward_mode,
        )

        obs = build_observation(
            self._battle.p1, self._battle.p2,
            self._type_chart, self._battle.turn,
            weather=self._battle.weather,
            weather_turns=self._battle.weather_turns,
        )

        info: dict[str, Any] = {"action_mask": self.action_masks(), "action_type": action_type}
        if terminated or truncated:
            info["winner"] = self._battle.winner
            info["turns"] = self._battle.turn

        return obs, reward, terminated, truncated, info

    def _decode_action(self, action: int, player: PlayerState,
                       opponent: PlayerState | None = None) -> Action:
        """Convert gym action int (0-9) to engine Action.

        0-3: use move slot 0-3
        4-9: switch to team[0]-team[5] (active slot masked)
        """
        mask = player.valid_action_mask(opponent, type_chart=self._type_chart)

        # if chosen action is masked, pick first valid
        if not mask[action]:
            for i in range(10):
                if mask[i]:
                    action = i
                    break

        if action < 4:
            active = player.active
            if not active.has_any_pp():
                return Struggle()
            if action < len(active.move_slots) and active.move_slots[action].has_pp:
                return UseMove(slot_index=action)
            # fallback: first valid move
            for i, slot in enumerate(active.move_slots):
                if slot.has_pp:
                    return UseMove(slot_index=i)
            return Struggle()
        else:
            team_index = action - 4
            return Switch(team_index=team_index)

    def _pick_forced_switch(self, player: PlayerState) -> Switch:
        """Pick first alive non-active teammate for forced switch."""
        for i, p in enumerate(player.team):
            if i != player.active_index and not p.is_fainted:
                return Switch(team_index=i)
        raise RuntimeError("No valid switch target but must_switch is true")

    def render(self) -> str | None:
        if self.render_mode != "ansi" or self._battle is None:
            return None

        b = self._battle
        lines = [
            f"=== Turn {b.turn} ===",
            f"P1: {b.p1.active.name} HP:{b.p1.active.current_hp}/{b.p1.active.max_hp} "
            f"({b.p1.alive_count} alive)",
            f"P2: {b.p2.active.name} HP:{b.p2.active.current_hp}/{b.p2.active.max_hp} "
            f"({b.p2.alive_count} alive)",
        ]
        if b.winner:
            lines.append(f"Winner: P{b.winner}")
        return "\n".join(lines)
