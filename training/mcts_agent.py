# MCTS agent using Rust search + neural net leaf evaluation
# Phase 3 of the Rust engine plan
#
# Usage:
#   .venv/bin/python training/mcts_agent.py --policy imitation_ppo --sims 200

from __future__ import annotations

import argparse
import random
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from engine.actions import Action, Struggle, Switch, UseMove
from engine.data_loader import DataStore
from engine.types import TypeChart
from gym_env.obs_builder import build_observation, OBS_SIZE
from gym_env.team_builder import build_team
from training.mcts_evaluator import MctsEvaluator
from training.rust_search_agent import (
    _RustPlayerAdapter,
    build_obs_from_rust,
)

import crystal_engine_rs as ce

DATA_DIR = str(Path(__file__).parent.parent / "data")


class MctsAgent:
    """Agent that uses MCTS with neural net evaluation for action selection.

    Each call to act():
    1. Creates an MctsContext from the current battle state
    2. Runs MCTS iterations, batching leaf states for NN evaluation
    3. Returns the action with highest visit count
    """

    def __init__(
        self,
        evaluator: MctsEvaluator,
        type_chart: TypeChart,
        n_simulations: int = 200,
        c_puct: float = 1.5,
        max_batch: int = 64,
        temperature: float = 0.1,
    ):
        self.evaluator = evaluator
        self.tc = type_chart
        self.n_sims = n_simulations
        self.c_puct = c_puct
        self.max_batch = max_batch
        self.temperature = temperature
        self._seed = 0

    def act_on_rust_battle(self, rs_battle) -> int:
        """Pick best action for P1 using MCTS. Returns action int (0-9)."""
        self._seed += 1
        mcts = ce.MctsContext(
            rs_battle,
            n_simulations=self.n_sims,
            seed=self._seed * 7919,
            c_puct=self.c_puct,
        )

        while True:
            n_pending = mcts.run_until_eval_needed(max_batch=self.max_batch)
            if n_pending == 0:
                break

            # get leaf states, build observations, run model
            leaf_states = mcts.get_pending_leaf_states()

            obs_batch = []
            mask_batch = []
            for ls in leaf_states:
                # use Rust obs builder if available, else Python adapter
                obs = ce.build_observation(ls)
                mask = ls.p1.valid_action_mask(ls.p2, filter_immune=True)
                obs_batch.append(obs)
                mask_batch.append(mask)

            obs_np = np.array(obs_batch, dtype=np.float32)
            mask_np = np.array(mask_batch, dtype=np.float32).reshape(len(obs_batch), 10)

            values, priors = self.evaluator.evaluate_batch(obs_np, mask_np)

            # convert to the format MCTS expects
            priors_arr = [list(p) for p in priors]
            mcts.supply_evaluations(values, priors_arr)

        probs = mcts.get_action_probs(temperature=self.temperature)
        return int(np.argmax(probs))

    def act(self, py_battle, rs_data) -> Action:
        """Convenience: act from a Python BattleState (converts to Rust internally)."""
        rs_t1 = [rs_data.build_pokemon(m.species.id, [s.template.id for s in m.move_slots])
                 for m in py_battle.p1.team]
        rs_t2 = [rs_data.build_pokemon(m.species.id, [s.template.id for s in m.move_slots])
                 for m in py_battle.p2.team]
        rs_battle = ce.create_battle(rs_t1, rs_t2, seed=self._seed)
        # NOTE: this doesn't sync mid-game state -- use act_on_rust_battle for live games
        action_int = self.act_on_rust_battle(rs_battle)
        return _int_to_action(action_int, py_battle.p1)


def _int_to_action(action_int, py_ps):
    if action_int < 4:
        if not py_ps.active.has_any_pp():
            return Struggle()
        if action_int < len(py_ps.active.move_slots) and py_ps.active.move_slots[action_int].has_pp:
            return UseMove(slot_index=action_int)
        for j, slot in enumerate(py_ps.active.move_slots):
            if slot.has_pp:
                return UseMove(slot_index=j)
        return Struggle()
    return Switch(team_index=action_int - 4)


# ============================================================
# EVALUATION
# ============================================================

def evaluate_mcts(
    evaluator: MctsEvaluator,
    tc: TypeChart,
    rs_data,
    data: DataStore,
    opponent_policy,
    n_games: int = 100,
    n_sims: int = 200,
    seed: int = 42,
):
    """Evaluate MCTS agent vs an opponent. Returns win rate for P1 (MCTS)."""
    agent = MctsAgent(evaluator, tc, n_simulations=n_sims)

    wins = 0
    losses = 0
    draws = 0
    total_turns = 0
    rng = random.Random(seed)

    for g in range(n_games):
        game_seed = rng.randint(0, 2**31)
        t1 = build_team(data, rng=random.Random(game_seed + 100))
        t2 = build_team(data, rng=random.Random(game_seed + 200))

        rs_t1 = [rs_data.build_pokemon(m.species.id, [s.template.id for s in m.move_slots]) for m in t1]
        rs_t2 = [rs_data.build_pokemon(m.species.id, [s.template.id for s in m.move_slots]) for m in t2]
        rs_battle = ce.create_battle(rs_t1, rs_t2, seed=game_seed + 300)

        for turn in range(200):
            if rs_battle.is_over:
                break

            # P1: MCTS
            a1 = agent.act_on_rust_battle(rs_battle)

            # P2: opponent policy
            p2 = _RustPlayerAdapter(rs_battle.p2)
            p1 = _RustPlayerAdapter(rs_battle.p1)
            p2_action = opponent_policy(p2, p1)
            a2 = _py_action_to_int(p2_action)

            rs_battle.resolve_turn(a1, a2)

            # forced switches
            if rs_battle.p2.must_switch:
                sw = _first_alive_bench(rs_battle.p2)
                if sw is not None:
                    rs_battle.resolve_forced_switches(None, sw)
            if rs_battle.p1.must_switch:
                sw = _first_alive_bench(rs_battle.p1)
                if sw is not None:
                    rs_battle.resolve_forced_switches(sw, None)

        total_turns += rs_battle.turn
        if rs_battle.winner == 1:
            wins += 1
        elif rs_battle.winner == 2:
            losses += 1
        else:
            draws += 1

        if (g + 1) % 20 == 0:
            print(f"  [{g+1}/{n_games}] W={wins} L={losses} D={draws} ({wins/(g+1):.0%})")

    return {
        "wins": wins,
        "losses": losses,
        "draws": draws,
        "win_rate": wins / n_games,
        "avg_turns": total_turns / n_games,
    }


def _py_action_to_int(action) -> int:
    if isinstance(action, UseMove):
        return action.slot_index
    elif isinstance(action, Switch):
        return action.team_index + 4
    return 10

def _first_alive_bench(rs_ps):
    for i, p in enumerate(rs_ps.team):
        if i != rs_ps.active_index and not p.is_fainted:
            return i
    return None


# ============================================================
# MAIN
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="MCTS agent evaluation")
    parser.add_argument("--policy", default="imitation_ppo")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--sims", type=int, default=200)
    parser.add_argument("--games", type=int, default=50)
    parser.add_argument("--opponent", default="smart", choices=["smart", "max_damage", "random"])
    args = parser.parse_args()

    data = DataStore()
    tc = TypeChart.load()
    rs_data = ce.DataStore(DATA_DIR)

    print(f"Loading model from {args.policy}...")
    evaluator = MctsEvaluator(args.policy, device=args.device)
    print(f"MCTS: {args.sims} sims, c_puct=1.5")

    from training.baselines import SmartAgent, MaxDamageAgent, RandomAgent
    if args.opponent == "smart":
        opp = SmartAgent(tc, seed=0)
    elif args.opponent == "max_damage":
        opp = MaxDamageAgent(tc)
    else:
        opp = RandomAgent(seed=0)

    opp_policy = lambda my, their: opp.act(my, their)

    # benchmark: MCTS with heuristic eval
    print(f"\n--- MCTS (heuristic eval) vs {args.opponent} ---")
    agent_h = MctsAgent(MctsEvaluator.__new__(MctsEvaluator), tc, n_simulations=args.sims)
    # monkey-patch to use heuristic
    class HeuristicEval:
        def evaluate_batch(self, obs, masks=None):
            n = len(obs)
            return [0.0] * n, [[0.1]*10] * n  # dummy, supply_heuristic will override
    agent_h.evaluator = HeuristicEval()

    # actually just run with heuristic directly
    heur_wins = 0
    t0 = time.time()
    rng = random.Random(42)
    for g in range(args.games):
        game_seed = rng.randint(0, 2**31)
        t1 = build_team(data, rng=random.Random(game_seed + 100))
        t2 = build_team(data, rng=random.Random(game_seed + 200))
        rs_t1 = [rs_data.build_pokemon(m.species.id, [s.template.id for s in m.move_slots]) for m in t1]
        rs_t2 = [rs_data.build_pokemon(m.species.id, [s.template.id for s in m.move_slots]) for m in t2]
        rs_battle = ce.create_battle(rs_t1, rs_t2, seed=game_seed + 300)
        for _ in range(200):
            if rs_battle.is_over: break
            # MCTS with heuristic
            mcts = ce.MctsContext(rs_battle, n_simulations=args.sims, seed=g*1000+_, c_puct=1.5)
            while True:
                n = mcts.run_until_eval_needed(32)
                if n == 0: break
                mcts.supply_heuristic_evaluations()
            a1 = int(np.argmax(mcts.get_action_probs(0.1)))
            p2 = _RustPlayerAdapter(rs_battle.p2)
            p1 = _RustPlayerAdapter(rs_battle.p1)
            a2 = _py_action_to_int(opp.act(p2, p1))
            rs_battle.resolve_turn(a1, a2)
            if rs_battle.p2.must_switch:
                sw = _first_alive_bench(rs_battle.p2)
                if sw is not None: rs_battle.resolve_forced_switches(None, sw)
            if rs_battle.p1.must_switch:
                sw = _first_alive_bench(rs_battle.p1)
                if sw is not None: rs_battle.resolve_forced_switches(sw, None)
        if rs_battle.winner == 1: heur_wins += 1
    heur_time = time.time() - t0
    print(f"  Win rate: {heur_wins}/{args.games} = {heur_wins/args.games:.0%} ({heur_time:.1f}s)")

    # benchmark: MCTS with NN eval
    print(f"\n--- MCTS (NN eval) vs {args.opponent} ---")
    t0 = time.time()
    results = evaluate_mcts(evaluator, tc, rs_data, data, opp_policy,
                            n_games=args.games, n_sims=args.sims)
    nn_time = time.time() - t0
    print(f"  Win rate: {results['wins']}/{args.games} = {results['win_rate']:.0%} "
          f"(avg {results['avg_turns']:.0f} turns, {nn_time:.1f}s)")


if __name__ == "__main__":
    main()
