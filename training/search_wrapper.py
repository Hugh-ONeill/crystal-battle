# gym wrapper that uses Rust 2-ply search to improve actions during PPO rollouts
# the policy network picks an action, but the wrapper overrides it with the
# search-best action. PPO trains on the actual rewards from the search actions,
# with clipping preventing catastrophic updates.
#
# this gives PPO the benefit of search (better actions -> better rewards)
# while keeping training stable (clipping prevents forgetting).

from __future__ import annotations

import random
from pathlib import Path

import gymnasium
import numpy as np
import torch

from gym_env.obs_builder import build_observation, OBS_SIZE
from training.opponent_model import OpponentPredictor


class SearchActionWrapper(gymnasium.Wrapper):
    """Wraps CrystalBattle env to override actions with search-improved ones.

    The PPO agent proposes an action, but this wrapper runs Rust 2-ply search
    and substitutes the search-best action before the env executes it.
    PPO sees the search action as "what it chose" and trains on the resulting
    rewards, gradually learning to match the search decisions naturally.
    """

    def __init__(self, env, opp_model_path: str = "opp_model.pt",
                 depth: int = 2, n_opp_samples: int = 3,
                 search_prob: float = 0.5):
        super().__init__(env)
        self.depth = depth
        self.n_opp_samples = n_opp_samples
        self.search_prob = search_prob  # probability of using search vs raw policy

        # load opponent model
        self.opp_model = OpponentPredictor()
        self.opp_model.load_state_dict(
            torch.load(opp_model_path, map_location="cpu", weights_only=True))
        self.opp_model.eval()
        self.opp_hidden = None

        # Rust engine
        import crystal_engine_rs as ce
        self.ce = ce
        data_path = str(Path(__file__).parent.parent / "data")
        self.rs_data = ce.DataStore(data_path)

        # type chart for obs building
        from engine.types import TypeChart
        self.tc = TypeChart.load()

    def reset(self, **kwargs):
        self.opp_hidden = None
        return super().reset(**kwargs)

    def step(self, action: int):
        battle = self.env.unwrapped._battle
        if battle is None or battle.is_over:
            return super().step(action)

        # only use search some of the time (mix of search + raw policy)
        if random.random() > self.search_prob:
            # update opp model state even when not searching
            obs = build_observation(
                battle.p1, battle.p2, self.tc, turn=battle.turn,
                weather=battle.weather, weather_turns=battle.weather_turns,
            )
            _, self.opp_hidden = self.opp_model.predict_single(obs, self.opp_hidden)
            return super().step(action)

        # get valid actions
        mask = battle.p1.valid_action_mask(battle.p2, type_chart=self.tc)
        valid = [i for i in range(10) if mask[i]]

        if len(valid) <= 1:
            obs = build_observation(
                battle.p1, battle.p2, self.tc, turn=battle.turn,
                weather=battle.weather, weather_turns=battle.weather_turns,
            )
            _, self.opp_hidden = self.opp_model.predict_single(obs, self.opp_hidden)
            return super().step(action)

        # build Rust battle state from Python state
        try:
            rs_battle = self._py_to_rust_battle(battle)
        except Exception:
            return super().step(action)

        # get opponent predictions
        obs = build_observation(
            battle.p1, battle.p2, self.tc, turn=battle.turn,
            weather=battle.weather, weather_turns=battle.weather_turns,
        )
        opp_probs, self.opp_hidden = self.opp_model.predict_single(obs, self.opp_hidden)

        # mask opponent actions
        opp_mask = battle.p2.valid_action_mask(battle.p1, type_chart=self.tc)
        opp_probs = opp_probs * np.array(opp_mask, dtype=np.float32)
        s = opp_probs.sum()
        if s > 0:
            opp_probs /= s
        else:
            return super().step(action)

        # get top opponent actions as weighted list
        top_idx = np.argsort(opp_probs)[::-1][:self.n_opp_samples]
        opp_actions = [(int(i), float(opp_probs[i])) for i in top_idx if opp_probs[i] > 0.01]
        if not opp_actions:
            return super().step(action)

        # run Rust search
        try:
            if self.depth >= 2:
                ranked = self.ce.search_2ply(
                    rs_battle, valid, opp_actions, opp_actions,
                    base_seed=battle.turn * 1000,
                )
            else:
                ranked = self.ce.search_1ply(
                    rs_battle, valid, opp_actions,
                    base_seed=battle.turn * 1000,
                )
            search_action = ranked[0][0] if ranked else action
        except Exception:
            search_action = action

        return super().step(search_action)

    def _py_to_rust_battle(self, battle):
        """Convert Python BattleState to Rust BattleState."""
        rs_t1 = [self.rs_data.build_pokemon(
            p.species.id, [s.template.id for s in p.move_slots])
            for p in battle.p1.team]
        rs_t2 = [self.rs_data.build_pokemon(
            p.species.id, [s.template.id for s in p.move_slots])
            for p in battle.p2.team]

        rs_battle = self.ce.create_battle(rs_t1, rs_t2, seed=0)

        # sync HP, status, etc. from Python state
        # this is approximate -- the Rust battle won't have identical RNG state
        # but the position evaluation should be close enough for search
        for i, (py_mon, rs_mon) in enumerate(zip(battle.p1.team, rs_battle.p1.team)):
            rs_mon.set_hp(py_mon.current_hp)
            if py_mon.status:
                rs_mon.set_status(py_mon.status)
        for i, (py_mon, rs_mon) in enumerate(zip(battle.p2.team, rs_battle.p2.team)):
            rs_mon.set_hp(py_mon.current_hp)
            if py_mon.status:
                rs_mon.set_status(py_mon.status)

        rs_battle.set_active(0, battle.p1.active_index)
        rs_battle.set_active(1, battle.p2.active_index)

        return rs_battle
