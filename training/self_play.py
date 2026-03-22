# OpponentPool + SelfPlayCallback + neural self-play opponent

from __future__ import annotations

import os
import random
import tempfile
from collections import deque
from typing import Any

import numpy as np
import torch as th

from engine.actions import Action, Struggle, Switch, UseMove
from engine.player_state import PlayerState
from engine.types import TypeChart
from gym_env.obs_builder import build_observation


class OpponentPool:
    """Stores frozen policy snapshots for self-play."""

    def __init__(self, max_size: int = 20):
        self._policies: deque[Any] = deque(maxlen=max_size)

    def add(self, policy_params: dict) -> None:
        """Add a snapshot of policy parameters."""
        snapshot = {k: v.clone() for k, v in policy_params.items()}
        self._policies.append(snapshot)

    def sample(self, rng: random.Random | None = None) -> dict | None:
        if not self._policies:
            return None
        if rng is None:
            rng = random.Random()
        return rng.choice(self._policies)

    def __len__(self) -> int:
        return len(self._policies)


def decode_action(action: int, player: PlayerState,
                  opponent: PlayerState | None = None) -> Action:
    """Convert gym action int (0-9) to engine Action."""
    mask = player.valid_action_mask(opponent)

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
        for i, slot in enumerate(active.move_slots):
            if slot.has_pp:
                return UseMove(slot_index=i)
        return Struggle()
    else:
        team_index = action - 4
        return Switch(team_index=team_index)


def make_neural_opponent(
    model,
    pool: OpponentPool,
    type_chart: TypeChart,
    rng: random.Random,
    deterministic: bool = False,
):
    """Create opponent policy that runs a frozen neural network from the pool.

    Samples a random snapshot from the pool, loads it into a copy of the
    model's policy network, and returns a callable that builds observations
    from the opponent's perspective and runs inference.
    """
    frozen_params = pool.sample(rng)
    if frozen_params is None:
        def fallback(opp_state, p1_state, battle_rng, **kwargs):
            return rng.choice(opp_state.valid_actions(p1_state))
        return fallback

    # avoid deepcopy -- fails on non-leaf tensors in Python 3.14+
    # save/load through temp file to get a clean policy copy
    tmp = tempfile.mktemp(suffix=".zip")
    try:
        model.save(tmp)
        frozen_model = type(model).load(tmp, device=model.device)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)
    frozen_policy = frozen_model.policy
    frozen_policy.load_state_dict(frozen_params)
    frozen_policy.eval()

    is_recurrent = hasattr(frozen_policy, "lstm_actor")
    lstm_state = [None]
    last_turn = [0]

    def policy(opp_state, p1_state, battle_rng, turn=0, weather=None, weather_turns=0):
        # reset LSTM state on new episode (turn went backwards)
        if is_recurrent and turn <= last_turn[0]:
            lstm_state[0] = None
        last_turn[0] = turn

        obs = build_observation(
            opp_state, p1_state, type_chart, turn,
            weather=weather, weather_turns=weather_turns,
        )
        mask = np.array(opp_state.valid_action_mask(p1_state), dtype=bool)

        with th.no_grad():
            if is_recurrent:
                episode_start = np.array([True]) if lstm_state[0] is None else np.array([False])
                action, new_state = frozen_policy.predict(
                    obs, state=lstm_state[0],
                    episode_start=episode_start,
                    deterministic=deterministic,
                    action_masks=mask,
                )
                lstm_state[0] = new_state
            else:
                action, _ = frozen_policy.predict(
                    obs, deterministic=deterministic, action_masks=mask,
                )

        return decode_action(int(action), opp_state, p1_state)

    return policy


def make_mixed_neural_opponent(
    model,
    pool: OpponentPool,
    type_chart: TypeChart,
    max_damage_agent,
    neural_weight: float = 0.5,
    rng: random.Random | None = None,
):
    """Mix neural self-play with MaxDamage opponent.

    Each episode randomly selects neural or MaxDamage as opponent for the
    full episode (not per-turn), so strategies stay consistent within games.

    If neural_weight < 0, uses adaptive ramping: starts at 0.2 and ramps to
    0.6 as the pool fills (proportional to pool size / max_size).
    """
    if rng is None:
        rng = random.Random()

    adaptive = neural_weight < 0
    neural = [make_neural_opponent(model, pool, type_chart, rng)]
    last_turn = [0]
    use_neural = [True]
    refresh_counter = [0]

    def _effective_weight() -> float:
        if adaptive:
            # ramp from 0.2 to 0.6 as pool fills
            fill = min(len(pool) / max(pool._policies.maxlen, 1), 1.0)
            return 0.2 + 0.4 * fill
        return neural_weight

    def policy(opp_state, p1_state, battle_rng, turn=0, weather=None, weather_turns=0):
        # pick opponent type at episode start
        if turn <= last_turn[0]:
            w = _effective_weight()
            use_neural[0] = rng.random() < w and len(pool) > 0
            # refresh frozen opponent every ~50 episodes to use newer snapshots
            refresh_counter[0] += 1
            if use_neural[0] and refresh_counter[0] % 50 == 0:
                neural[0] = make_neural_opponent(model, pool, type_chart, rng)
        last_turn[0] = turn

        if use_neural[0]:
            return neural[0](opp_state, p1_state, battle_rng,
                             turn=turn, weather=weather, weather_turns=weather_turns)
        else:
            return max_damage_agent.act(opp_state, p1_state)

    return policy


try:
    from stable_baselines3.common.callbacks import BaseCallback

    class SelfPlayCallback(BaseCallback):
        """Snapshots the policy into OpponentPool every N steps."""

        def __init__(
            self,
            pool: OpponentPool,
            snapshot_freq: int = 50_000,
            verbose: int = 0,
        ):
            super().__init__(verbose)
            self._pool = pool
            self._snapshot_freq = snapshot_freq
            self._last_snapshot = 0

        def _on_step(self) -> bool:
            if self.num_timesteps - self._last_snapshot >= self._snapshot_freq:
                params = self.model.policy.state_dict()
                self._pool.add(params)
                self._last_snapshot = self.num_timesteps
                if self.verbose:
                    print(f"[SelfPlay] Snapshot at step {self.num_timesteps}, "
                          f"pool size: {len(self._pool)}")
            return True

except ImportError:
    pass
