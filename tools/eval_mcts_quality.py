# Compare MCTS quality improvements: baseline vs dirichlet + opp model
#
# Usage:
#   .venv/bin/python tools/eval_mcts_quality.py --games 50

from __future__ import annotations

import argparse
import random
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

import crystal_engine_rs as ce
from engine.data_loader import DataStore
from engine.types import TypeChart
from engine.actions import UseMove, Switch, Struggle
from gym_env.team_builder import build_team
from training.mcts_evaluator import MctsEvaluator
from training.rust_search_agent import _RustPlayerAdapter

DATA_DIR = str(Path(__file__).parent.parent / "data")


def build_game(data, rs_data, seed):
    rng = random.Random(seed)
    t1 = build_team(data, rng=random.Random(rng.randint(0, 2**32)))
    t2 = build_team(data, rng=random.Random(rng.randint(0, 2**32)))
    rs_t1 = [rs_data.build_pokemon(m.species.id, [s.template.id for s in m.move_slots],
              dvs=[m.dvs.get('attack',15), m.dvs.get('defense',15),
                   m.dvs.get('speed',15), m.dvs.get('special',15)]) for m in t1]
    rs_t2 = [rs_data.build_pokemon(m.species.id, [s.template.id for s in m.move_slots],
              dvs=[m.dvs.get('attack',15), m.dvs.get('defense',15),
                   m.dvs.get('speed',15), m.dvs.get('special',15)]) for m in t2]
    return ce.create_battle(rs_t1, rs_t2, seed=rng.randint(0, 2**62))


def opp_act(opp, b):
    p2 = _RustPlayerAdapter(b.p2)
    p1 = _RustPlayerAdapter(b.p1)
    a = opp.act(p2, p1)
    if isinstance(a, UseMove): return a.slot_index
    elif isinstance(a, Switch): return a.team_index + 4
    return 10


def do_forced_switches(b):
    for pnum in [2, 1]:
        ps = b.p1 if pnum == 1 else b.p2
        if ps.must_switch:
            for i, p in enumerate(ps.team):
                if i != ps.active_index and not p.is_fainted:
                    if pnum == 1: b.resolve_forced_switches(i, None)
                    else: b.resolve_forced_switches(None, i)
                    break


def run_eval(agent_fn, opp, data, rs_data, n_games, label):
    wins = 0
    rng = random.Random(42)
    t0 = time.time()
    for g in range(n_games):
        b = build_game(data, rs_data, rng.randint(0, 2**31))
        for _ in range(200):
            if b.is_over: break
            a1 = agent_fn(b, g * 1000 + _)
            a2 = opp_act(opp, b)
            b.resolve_turn(a1, a2)
            do_forced_switches(b)
        if b.winner == 1: wins += 1
    elapsed = time.time() - t0
    rate = wins / n_games
    print(f'  {label:50s} {wins:3d}/{n_games} = {rate:.0%}  ({elapsed:.0f}s)')
    return rate


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--games", type=int, default=50)
    parser.add_argument("--policy", default="imitation_ppo")
    parser.add_argument("--sims", type=int, default=200)
    args = parser.parse_args()

    data = DataStore()
    tc = TypeChart.load()
    rs_data = ce.DataStore(DATA_DIR)

    from training.baselines import SmartAgent
    smart = SmartAgent(tc, seed=0)

    # try loading opponent model
    opp_model = None
    opp_model_path = Path(__file__).parent.parent / "opp_model.pt"
    if opp_model_path.exists():
        import torch
        from training.opponent_model import OpponentPredictor
        opp_model = OpponentPredictor()
        opp_model.load_state_dict(torch.load(str(opp_model_path), map_location='cpu', weights_only=True))
        opp_model.eval()
        print("Loaded opponent model")

    evaluator = MctsEvaluator(args.policy, device='cpu')

    def nn_eval_leaves(mcts):
        leaves = mcts.get_pending_leaf_states()
        obs = [ce.build_observation(ls) for ls in leaves]
        masks = [ls.p1.valid_action_mask(ls.p2, filter_immune=True) for ls in leaves]
        obs_np = np.array(obs, dtype=np.float32)
        mask_np = np.array(masks, dtype=np.float32).reshape(len(obs), 10)
        values, priors = evaluator.evaluate_batch(obs_np, mask_np)
        mcts.supply_evaluations(values, priors)

    def get_opp_weights(b):
        if opp_model is None:
            return None
        obs = ce.build_observation(b)
        obs_np = np.array(obs, dtype=np.float32).reshape(1, -1)
        probs, _ = opp_model.predict_single(np.array(obs, dtype=np.float32), None)
        mask = b.p2.valid_action_mask(b.p1, filter_immune=True)
        probs = probs * np.array(mask, dtype=np.float32)
        s = probs.sum()
        if s > 0: probs /= s
        top = np.argsort(probs)[::-1][:5]
        return [(int(i), float(probs[i])) for i in top if probs[i] > 0.01]

    ns = args.sims
    print(f'\nMCTS {ns} sims vs Smart ({args.games} games)')
    print('=' * 70)

    # 1. heuristic baseline
    def heur_agent(b, seed):
        mcts = ce.MctsContext(b, n_simulations=ns, seed=seed, c_puct=1.5)
        while True:
            n = mcts.run_until_eval_needed(64)
            if n == 0: break
            mcts.supply_heuristic_evaluations()
        return int(np.argmax(mcts.get_action_probs(0.1)))
    run_eval(heur_agent, smart, data, rs_data, args.games, 'Heuristic')

    # 2. NN eval baseline
    def nn_agent(b, seed):
        mcts = ce.MctsContext(b, n_simulations=ns, seed=seed, c_puct=1.5)
        while True:
            n = mcts.run_until_eval_needed(64)
            if n == 0: break
            nn_eval_leaves(mcts)
        return int(np.argmax(mcts.get_action_probs(0.1)))
    run_eval(nn_agent, smart, data, rs_data, args.games, 'NN eval')

    # 3. NN eval + opp model weights
    if opp_model is not None:
        def nn_opp_agent(b, seed):
            mcts = ce.MctsContext(b, n_simulations=ns, seed=seed, c_puct=1.5)
            weights = get_opp_weights(b)
            if weights:
                mcts.set_opp_weights(weights)
            while True:
                n = mcts.run_until_eval_needed(64)
                if n == 0: break
                nn_eval_leaves(mcts)
            return int(np.argmax(mcts.get_action_probs(0.1)))
        run_eval(nn_opp_agent, smart, data, rs_data, args.games, 'NN eval + opp model')

    # 4. NN eval + opp model + temperature=1.0 (more exploration)
    if opp_model is not None:
        def nn_opp_explore_agent(b, seed):
            mcts = ce.MctsContext(b, n_simulations=ns, seed=seed, c_puct=1.5)
            weights = get_opp_weights(b)
            if weights:
                mcts.set_opp_weights(weights)
            while True:
                n = mcts.run_until_eval_needed(64)
                if n == 0: break
                nn_eval_leaves(mcts)
            # higher temperature for first 10 turns, low after
            probs = mcts.get_action_probs(0.5)
            return int(np.argmax(probs))
        run_eval(nn_opp_explore_agent, smart, data, rs_data, args.games, 'NN + opp + temp=0.5')

    # 5. higher c_puct (more exploration in tree)
    def nn_opp_cpuct_agent(b, seed):
        mcts = ce.MctsContext(b, n_simulations=ns, seed=seed, c_puct=2.5)
        if opp_model is not None:
            weights = get_opp_weights(b)
            if weights:
                mcts.set_opp_weights(weights)
        while True:
            n = mcts.run_until_eval_needed(64)
            if n == 0: break
            nn_eval_leaves(mcts)
        return int(np.argmax(mcts.get_action_probs(0.1)))
    run_eval(nn_opp_cpuct_agent, smart, data, rs_data, args.games, 'NN + opp + c_puct=2.5')


if __name__ == "__main__":
    main()
