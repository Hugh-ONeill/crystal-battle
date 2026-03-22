# lookahead search: simulate future turns to pick the best action
# uses opponent model for prediction + value network for position evaluation
#
# Usage:
#   evaluate:  python training/lookahead.py --model imitation_ppo --opp-model opp_model.pt

from __future__ import annotations

import argparse
import copy
import random
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

from engine.actions import Action, Switch, UseMove, Struggle
from engine.battle_state import BattleState
from engine.player_state import PlayerState
from engine.turn_engine import resolve_turn, resolve_forced_switches
from engine.types import TypeChart
from gym_env.obs_builder import build_observation, OBS_SIZE
from training.opponent_model import OpponentPredictor


class LookaheadAgent:
    """Agent that uses lookahead search for action selection.

    Combines a base policy (for LSTM state + value estimates) with an
    opponent model (for predicting opponent actions) and a battle simulator
    (for evaluating future positions).
    """

    def __init__(
        self,
        policy_model,
        opponent_model: OpponentPredictor,
        type_chart: TypeChart,
        depth: int = 1,
        n_opponent_samples: int = 3,
        device: str = "cpu",
    ):
        self.policy = policy_model
        self.opp_model = opponent_model
        self.tc = type_chart
        self.depth = depth
        self.n_opp_samples = n_opponent_samples
        self.device = device

        # LSTM state for the policy (tracked across turns)
        self.lstm_state = None
        self.episode_start = np.array([True])

        # opponent model hidden state
        self.opp_hidden = None

    def reset(self):
        """Reset for a new game."""
        self.lstm_state = None
        self.episode_start = np.array([True])
        self.opp_hidden = None

    def act(self, battle: BattleState) -> Action:
        """Pick the best action using lookahead search."""
        obs = build_observation(
            battle.p1, battle.p2, self.tc, turn=battle.turn,
            weather=battle.weather, weather_turns=battle.weather_turns,
        )
        mask = np.array(
            battle.p1.valid_action_mask(battle.p2, type_chart=self.tc),
            dtype=bool,
        )
        valid_actions = [i for i in range(10) if mask[i]]

        if not valid_actions:
            return Struggle()

        if len(valid_actions) == 1:
            # only one option, no search needed
            action_int = valid_actions[0]
            # still update LSTM state
            _, self.lstm_state = self.policy.predict(
                obs, state=self.lstm_state, episode_start=self.episode_start,
                deterministic=True, action_masks=mask,
            )
            self.episode_start = np.array([False])
            # update opponent model state
            obs_t = torch.tensor(obs, dtype=torch.float32)
            _, self.opp_hidden = self.opp_model.predict_single(obs, self.opp_hidden)
            return _int_to_action(action_int, battle.p1)

        # ---- Get opponent action predictions ----
        obs_t = torch.tensor(obs, dtype=torch.float32)
        opp_probs, new_opp_hidden = self.opp_model.predict_single(obs, self.opp_hidden)

        # get valid opponent actions
        opp_mask = np.array(
            battle.p2.valid_action_mask(battle.p1, type_chart=self.tc),
            dtype=bool,
        )
        # mask and renormalize opponent predictions
        opp_probs = opp_probs * opp_mask
        opp_sum = opp_probs.sum()
        if opp_sum > 0:
            opp_probs = opp_probs / opp_sum
        else:
            # fallback: uniform over valid actions
            opp_probs = opp_mask.astype(np.float32)
            opp_probs /= opp_probs.sum()

        # sample top opponent actions
        opp_actions = _sample_top_actions(opp_probs, self.n_opp_samples)

        # ---- Evaluate each of our actions ----
        action_values = self._search(battle, opp_probs, opp_mask,
                                     valid_actions, depth=self.depth)

        # pick best action
        best_action_int = max(action_values, key=action_values.get)

        # update LSTM state with the chosen action
        _, self.lstm_state = self.policy.predict(
            obs, state=self.lstm_state, episode_start=self.episode_start,
            deterministic=True, action_masks=mask,
        )
        self.episode_start = np.array([False])

        # update opponent model state
        self.opp_hidden = new_opp_hidden

        return _int_to_action(best_action_int, battle.p1)

    def _search(self, battle: BattleState, opp_probs: np.ndarray,
                 opp_mask: np.ndarray, valid_actions: list[int],
                 depth: int) -> dict[int, float]:
        """Recursive search: evaluate each action by simulating ahead."""
        opp_actions = _sample_top_actions(opp_probs, self.n_opp_samples)
        action_values = {}

        for my_action_int in valid_actions:
            my_action = _int_to_action(my_action_int, battle.p1)

            total_value = 0.0
            total_weight = 0.0

            for opp_action_int, opp_weight in opp_actions:
                opp_action = _int_to_action(opp_action_int, battle.p2)

                sim_battle = _copy_battle(battle)

                try:
                    resolve_turn(sim_battle, my_action, opp_action, self.tc)
                    _handle_forced_switches_simple(sim_battle, self.tc)
                except Exception:
                    continue

                if depth <= 1 or sim_battle.is_over:
                    value = self._evaluate_position(sim_battle)
                else:
                    # recurse: get the best value from the next depth
                    sim_obs = build_observation(
                        sim_battle.p1, sim_battle.p2, self.tc,
                        turn=sim_battle.turn,
                        weather=sim_battle.weather,
                        weather_turns=sim_battle.weather_turns,
                    )
                    sim_mask = np.array(
                        sim_battle.p1.valid_action_mask(sim_battle.p2, type_chart=self.tc),
                        dtype=bool,
                    )
                    sim_valid = [i for i in range(10) if sim_mask[i]]

                    if not sim_valid:
                        value = self._evaluate_position(sim_battle)
                    else:
                        # predict opponent for next depth
                        sim_opp_probs, _ = self.opp_model.predict_single(sim_obs, None)
                        sim_opp_mask = np.array(
                            sim_battle.p2.valid_action_mask(sim_battle.p1, type_chart=self.tc),
                            dtype=bool,
                        )
                        sim_opp_probs = sim_opp_probs * sim_opp_mask
                        s = sim_opp_probs.sum()
                        if s > 0:
                            sim_opp_probs /= s
                        else:
                            sim_opp_probs = sim_opp_mask.astype(np.float32)
                            sim_opp_probs /= sim_opp_probs.sum()

                        child_values = self._search(
                            sim_battle, sim_opp_probs, sim_opp_mask,
                            sim_valid, depth=depth - 1,
                        )
                        value = max(child_values.values()) if child_values else 0.0

                total_value += value * opp_weight
                total_weight += opp_weight

            if total_weight > 0:
                action_values[my_action_int] = total_value / total_weight
            else:
                action_values[my_action_int] = -1.0

        return action_values

    def _evaluate_position(self, battle: BattleState) -> float:
        """Evaluate a battle position using the value network + heuristics."""
        if battle.is_over:
            if battle.winner == 1:
                return 1.0
            elif battle.winner == 2:
                return -1.0
            return 0.0

        # use HP differential as primary signal
        my_hp = battle.p1.total_hp_frac
        opp_hp = battle.p2.total_hp_frac
        hp_diff = my_hp - opp_hp

        # alive count matters too
        my_alive = battle.p1.alive_count / len(battle.p1.team)
        opp_alive = battle.p2.alive_count / len(battle.p2.team)
        alive_diff = my_alive - opp_alive

        # combine
        return hp_diff * 0.6 + alive_diff * 0.4


def _copy_battle(battle: BattleState) -> BattleState:
    """Deep copy a battle state for simulation."""
    return copy.deepcopy(battle)


def _int_to_action(action_int: int, player: PlayerState) -> Action:
    """Convert action int to engine Action."""
    if action_int < 4:
        active = player.active
        if not active.has_any_pp():
            return Struggle()
        if action_int < len(active.move_slots) and active.move_slots[action_int].has_pp:
            return UseMove(slot_index=action_int)
        # fallback to first usable move
        for j, slot in enumerate(active.move_slots):
            if slot.has_pp:
                return UseMove(slot_index=j)
        return Struggle()
    else:
        return Switch(team_index=action_int - 4)


def _sample_top_actions(probs: np.ndarray, n: int) -> list[tuple[int, float]]:
    """Return top-n most likely actions with their probabilities."""
    top_indices = np.argsort(probs)[::-1][:n]
    result = []
    for idx in top_indices:
        if probs[idx] > 0.01:  # skip negligible probabilities
            result.append((int(idx), float(probs[idx])))
    if not result:
        # fallback
        best = int(np.argmax(probs))
        result = [(best, 1.0)]
    return result


def _handle_forced_switches_simple(battle: BattleState, tc: TypeChart):
    """Handle forced switches after simulation with simple heuristic."""
    sw1 = sw2 = None
    if battle.p1.must_switch:
        # pick first alive bench mon
        for i, p in enumerate(battle.p1.team):
            if i != battle.p1.active_index and not p.is_fainted:
                sw1 = Switch(team_index=i)
                break
    if battle.p2.must_switch:
        for i, p in enumerate(battle.p2.team):
            if i != battle.p2.active_index and not p.is_fainted:
                sw2 = Switch(team_index=i)
                break
    if sw1 or sw2:
        resolve_forced_switches(battle, sw1, sw2)


# ============================================================
# EVALUATION
# ============================================================

def evaluate_lookahead(
    policy_path: str,
    opp_model_path: str,
    n_games: int = 100,
    depth: int = 1,
    device: str = "cpu",
):
    """Evaluate the lookahead agent vs baselines."""
    from training.maskable_recurrent_ppo import MaskableRecurrentPPO
    from training.baselines import SmartAgent, MaxDamageAgent
    from engine.data_loader import DataStore
    from gym_env.team_builder import build_team

    tc = TypeChart.load()
    data = DataStore()

    print(f"Loading policy from {policy_path}...")
    policy = MaskableRecurrentPPO.load(policy_path, device=device)

    print(f"Loading opponent model from {opp_model_path}...")
    opp_model = OpponentPredictor()
    opp_model.load_state_dict(torch.load(opp_model_path, map_location=device, weights_only=True))
    opp_model.eval()

    agent = LookaheadAgent(policy, opp_model, tc, depth=depth, device=device)

    # also test the base policy without lookahead for comparison
    baselines = {
        "max_damage": MaxDamageAgent(tc),
        "smart": SmartAgent(tc, seed=99),
    }

    for bl_name, bl_agent in baselines.items():
        wins = 0
        for i in range(n_games):
            rng = random.Random(i)
            t1 = build_team(data, rng=rng, tier="ou")
            t2 = build_team(data, rng=random.Random(i + 1000), tier="ou")
            battle = BattleState(
                p1=PlayerState(team=t1), p2=PlayerState(team=t2),
                rng=random.Random(i + 2000),
            )

            agent.reset()

            for turn in range(100):
                if battle.is_over:
                    break

                a1 = agent.act(battle)
                a2 = bl_agent.act(battle.p2, battle.p1)

                resolve_turn(battle, a1, a2, tc)

                # forced switches
                sw1 = sw2 = None
                if battle.p1.must_switch:
                    # use simple heuristic for forced switches
                    for j, p in enumerate(battle.p1.team):
                        if j != battle.p1.active_index and not p.is_fainted:
                            sw1 = Switch(team_index=j)
                            break
                if battle.p2.must_switch:
                    sw2_a = bl_agent.act(battle.p2, battle.p1)
                    sw2 = sw2_a if isinstance(sw2_a, Switch) else None
                    if sw2 is None:
                        for j, p in enumerate(battle.p2.team):
                            if j != battle.p2.active_index and not p.is_fainted:
                                sw2 = Switch(team_index=j)
                                break
                if sw1 or sw2:
                    resolve_forced_switches(battle, sw1, sw2)

            if battle.winner == 1:
                wins += 1

        print(f"  vs {bl_name:12s}: {wins}/{n_games} ({wins/n_games*100:.1f}%)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Lookahead search agent")
    parser.add_argument("--model", type=str, default="imitation_ppo")
    parser.add_argument("--opp-model", type=str, default="opp_model.pt")
    parser.add_argument("--n-games", type=int, default=100)
    parser.add_argument("--depth", type=int, default=1)
    parser.add_argument("--device", type=str, default="cpu")
    args = parser.parse_args()

    evaluate_lookahead(
        args.model, args.opp_model,
        n_games=args.n_games, depth=args.depth, device=args.device,
    )
