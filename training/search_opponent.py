# search-based opponent for PPO training
# wraps the Rust 2-ply search as a gym opponent policy
# the RL agent trains against search-level play, forcing it to improve

from __future__ import annotations

import random
from pathlib import Path

import numpy as np
import torch

from engine.types import TypeChart
from training.opponent_model import OpponentPredictor


class SearchOpponent:
    """Opponent that uses Rust 2-ply search to pick actions.

    Used as opponent_policy in the gym env. The RL agent trains against
    search-level play instead of heuristic bots.
    """

    def __init__(
        self,
        type_chart: TypeChart | None = None,
        opp_model_path: str = "opp_model.pt",
        depth: int = 2,
        n_samples: int = 5,
        search_prob: float = 1.0,
    ):
        import crystal_engine_rs as ce
        self.ce = ce
        self.tc = type_chart or TypeChart.load()
        self.depth = depth
        self.n_samples = n_samples
        self.search_prob = search_prob

        # opponent model for predicting what the RL agent will do
        self.opp_model = OpponentPredictor()
        self.opp_model.load_state_dict(
            torch.load(opp_model_path, map_location="cpu", weights_only=True))
        self.opp_model.eval()
        self.opp_hidden = None

        self.rs_data = ce.DataStore(str(Path(__file__).parent.parent / "data"))
        self._rs_battle = None
        self._last_turn = -1

    def reset(self):
        """Reset for new game."""
        self.opp_hidden = None
        self._rs_battle = None
        self._last_turn = -1

    def __call__(self, opp_state, p1_state, rng, **kwargs):
        """Gym opponent_policy interface.

        opp_state: this opponent's PlayerState (P2)
        p1_state: the RL agent's PlayerState (P1)
        """
        from engine.actions import Switch, UseMove, Struggle

        # sometimes use simple heuristic instead of search (variety)
        if random.random() > self.search_prob:
            from training.baselines import SmartAgent
            smart = SmartAgent(self.tc, seed=random.randint(0, 10000))
            return smart.act(opp_state, p1_state)

        # get valid actions
        valid_mask = opp_state.valid_action_mask(p1_state, type_chart=self.tc)
        valid = [i for i in range(10) if valid_mask[i]]

        if not valid:
            return Struggle()
        if len(valid) == 1:
            action_int = valid[0]
        else:
            # build Rust battle state for search
            # note: the search opponent plays as P2, but the search evaluates
            # from the searcher's perspective, so we need to swap P1/P2
            try:
                action_int = self._search_action(opp_state, p1_state, valid, kwargs)
            except Exception:
                # fallback to Smart on any error
                from training.baselines import SmartAgent
                smart = SmartAgent(self.tc, seed=random.randint(0, 10000))
                return smart.act(opp_state, p1_state)

        if action_int < 4:
            active = opp_state.active
            if not active.has_any_pp():
                return Struggle()
            if action_int < len(active.move_slots) and active.move_slots[action_int].has_pp:
                return UseMove(slot_index=action_int)
            for j, slot in enumerate(active.move_slots):
                if slot.has_pp:
                    return UseMove(slot_index=j)
            return Struggle()
        else:
            return Switch(team_index=action_int - 4)

    def _search_action(self, opp_state, p1_state, valid, kwargs):
        """Run Rust search from P2's perspective."""
        from gym_env.obs_builder import build_observation

        turn = kwargs.get("turn", 0)
        weather = kwargs.get("weather")
        weather_turns = kwargs.get("weather_turns", 0)

        # build obs from P2's perspective (P2 is the searcher)
        obs = build_observation(
            opp_state, p1_state, self.tc, turn,
            weather=weather, weather_turns=weather_turns,
        )

        # predict what P1 (the RL agent) will do
        opp_probs, self.opp_hidden = self.opp_model.predict_single(obs, self.opp_hidden)

        # mask to P1's valid actions
        p1_mask = p1_state.valid_action_mask(opp_state, type_chart=self.tc)
        opp_probs = opp_probs * np.array(p1_mask, dtype=np.float32)
        s = opp_probs.sum()
        if s > 0:
            opp_probs /= s
        else:
            return valid[0]

        top_idx = np.argsort(opp_probs)[::-1][:self.n_samples]
        p1_actions = [(int(i), float(opp_probs[i])) for i in top_idx if opp_probs[i] > 0.01]
        if not p1_actions:
            p1_actions = [(int(np.argmax(opp_probs)), 1.0)]

        # build Rust battle state with P2 as the searcher
        try:
            rs_t2 = [self.rs_data.build_pokemon(
                p.species.id, [s.template.id for s in p.move_slots])
                for p in opp_state.team]
            rs_t1 = [self.rs_data.build_pokemon(
                p.species.id, [s.template.id for s in p.move_slots])
                for p in p1_state.team]

            # create battle with P2 as P1 (the searcher is always P1 in the engine)
            rs_battle = self.ce.create_battle(rs_t2, rs_t1, seed=turn * 1000)

            # sync HP and status
            for py_mon, rs_mon in zip(opp_state.team, rs_battle.p1.team):
                rs_mon.set_hp(py_mon.current_hp)
                if py_mon.status:
                    rs_mon.set_status(py_mon.status)
            for py_mon, rs_mon in zip(p1_state.team, rs_battle.p2.team):
                rs_mon.set_hp(py_mon.current_hp)
                if py_mon.status:
                    rs_mon.set_status(py_mon.status)

            rs_battle.set_active(0, opp_state.active_index)
            rs_battle.set_active(1, p1_state.active_index)

        except Exception:
            return valid[0]

        # run search
        if self.depth >= 2:
            ranked = self.ce.search_2ply(
                rs_battle, valid, p1_actions, p1_actions,
                base_seed=turn * 1000,
            )
        else:
            ranked = self.ce.search_1ply(
                rs_battle, valid, p1_actions,
                base_seed=turn * 1000,
            )

        return ranked[0][0] if ranked else valid[0]


def make_search_opponent(type_chart=None, search_prob=0.5, depth=2):
    """Create a mixed opponent: search_prob% search, rest Smart/MaxDmg.

    Designed as a drop-in for make_mixed_opponent in train.py.
    """
    from training.baselines import SmartAgent, MaxDamageAgent
    tc = type_chart or TypeChart.load()

    search_opp = SearchOpponent(tc, depth=depth, search_prob=1.0)
    smart = SmartAgent(tc, seed=0)
    maxdmg = MaxDamageAgent(tc)

    def opponent_policy(opp_state, p1_state, rng, **kwargs):
        roll = random.random()
        if roll < search_prob:
            return search_opp(opp_state, p1_state, rng, **kwargs)
        elif roll < search_prob + (1 - search_prob) * 0.5:
            return smart.act(opp_state, p1_state)
        else:
            return maxdmg.act(opp_state, p1_state)

    # attach reset method for env resets
    opponent_policy.reset = search_opp.reset
    return opponent_policy
