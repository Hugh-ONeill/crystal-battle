# Profile MCTS to find where time is spent
#
# Usage:
#   .venv/bin/python tools/profile_mcts.py

from __future__ import annotations

import random
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from engine.data_loader import DataStore
from engine.types import TypeChart
from gym_env.obs_builder import build_observation, OBS_SIZE
from gym_env.team_builder import build_team
from training.mcts_evaluator import MctsEvaluator
from training.rust_search_agent import build_obs_from_rust

import crystal_engine_rs as ce

DATA_DIR = str(Path(__file__).parent.parent / "data")


def build_position(data, rs_data, seed):
    rng = random.Random(seed)
    t1 = build_team(data, rng=random.Random(rng.randint(0, 2**32)))
    t2 = build_team(data, rng=random.Random(rng.randint(0, 2**32)))
    rs_t1 = [rs_data.build_pokemon(m.species.id, [s.template.id for s in m.move_slots]) for m in t1]
    rs_t2 = [rs_data.build_pokemon(m.species.id, [s.template.id for s in m.move_slots]) for m in t2]
    b = ce.create_battle(rs_t1, rs_t2, seed=rng.randint(0, 2**62))
    # advance a few turns
    for _ in range(random.randint(3, 8)):
        if b.is_over:
            break
        m1 = b.p1.valid_action_mask(b.p2)
        m2 = b.p2.valid_action_mask(b.p1)
        v1 = [i for i in range(10) if m1[i]]
        v2 = [i for i in range(10) if m2[i]]
        if not v1 or not v2:
            break
        b.resolve_turn(random.choice(v1), random.choice(v2))
    return b


def profile_components(battle, evaluator, tc, n_sims=200, max_batch=32):
    """Profile each component of MCTS separately."""
    results = {}

    # ---- 1. Pure Rust MCTS tree ops (heuristic, no Python callbacks) ----
    t0 = time.perf_counter()
    for _ in range(10):
        mcts = ce.MctsContext(battle, n_simulations=n_sims, seed=42, c_puct=1.5)
        while True:
            n = mcts.run_until_eval_needed(max_batch)
            if n == 0:
                break
            mcts.supply_heuristic_evaluations()
    results["rust_tree_heuristic"] = (time.perf_counter() - t0) / 10

    # ---- 2. Rust tree ops only (measure run_until_eval + supply separately) ----
    tree_time = 0
    heur_time = 0
    for _ in range(10):
        mcts = ce.MctsContext(battle, n_simulations=n_sims, seed=42, c_puct=1.5)
        while True:
            t0 = time.perf_counter()
            n = mcts.run_until_eval_needed(max_batch)
            tree_time += time.perf_counter() - t0
            if n == 0:
                break
            t0 = time.perf_counter()
            mcts.supply_heuristic_evaluations()
            heur_time += time.perf_counter() - t0
    results["rust_expand_select"] = tree_time / 10
    results["rust_heuristic_eval"] = heur_time / 10

    # ---- 3. Obs building cost (Rust native) ----
    mcts = ce.MctsContext(battle, n_simulations=n_sims, seed=42, c_puct=1.5)
    mcts.run_until_eval_needed(max_batch)
    leaves = mcts.get_pending_leaf_states()

    t0 = time.perf_counter()
    for _ in range(10):
        obs_batch = []
        mask_batch = []
        for ls in leaves:
            obs = ce.build_observation(ls)
            mask = ls.p1.valid_action_mask(ls.p2, filter_immune=True)
            obs_batch.append(obs)
            mask_batch.append(mask)
        obs_np = np.array(obs_batch, dtype=np.float32)
        mask_np = np.array(mask_batch, dtype=np.float32).reshape(len(obs_batch), 10)
    results["obs_building"] = (time.perf_counter() - t0) / 10
    results["obs_per_leaf"] = results["obs_building"] / len(leaves)
    results["obs_n_leaves"] = len(leaves)

    # ---- 4. get_pending_leaf_states cost (PyO3 clone) ----
    t0 = time.perf_counter()
    for _ in range(100):
        mcts2 = ce.MctsContext(battle, n_simulations=n_sims, seed=42, c_puct=1.5)
        mcts2.run_until_eval_needed(max_batch)
        _ = mcts2.get_pending_leaf_states()
    results["get_leaf_states"] = (time.perf_counter() - t0) / 100

    # ---- 5. Torch inference only ----
    obs_np = np.stack(obs_batch).astype(np.float32)
    mask_np = np.array(mask_batch, dtype=np.float32).reshape(len(obs_batch), 10)
    # warmup
    evaluator.evaluate_batch(obs_np, mask_np)
    t0 = time.perf_counter()
    for _ in range(50):
        evaluator.evaluate_batch(obs_np, mask_np)
    results["torch_inference"] = (time.perf_counter() - t0) / 50
    results["torch_batch_size"] = len(obs_batch)

    # ---- 6. supply_evaluations cost ----
    values_dummy = [0.0] * len(leaves)
    priors_dummy = [[0.1] * 10] * len(leaves)
    # need a fresh mcts each time since supply advances the state
    t0 = time.perf_counter()
    for _ in range(100):
        m3 = ce.MctsContext(battle, n_simulations=n_sims, seed=42, c_puct=1.5)
        m3.run_until_eval_needed(max_batch)
        m3.supply_evaluations(values_dummy, priors_dummy)
    results["supply_evals"] = (time.perf_counter() - t0) / 100

    # ---- 7. Full NN MCTS end-to-end ----
    t0 = time.perf_counter()
    for _ in range(3):
        mcts = ce.MctsContext(battle, n_simulations=n_sims, seed=42, c_puct=1.5)
        while True:
            n = mcts.run_until_eval_needed(max_batch)
            if n == 0:
                break
            leaves = mcts.get_pending_leaf_states()
            obs_batch = []
            mask_batch = []
            for ls in leaves:
                obs = ce.build_observation(ls)
                mask = ls.p1.valid_action_mask(ls.p2, filter_immune=True)
                obs_batch.append(obs)
                mask_batch.append(mask)
            obs_np = np.array(obs_batch, dtype=np.float32)
            mask_np = np.array(mask_batch, dtype=np.float32).reshape(len(obs_batch), 10)
            values, priors = evaluator.evaluate_batch(obs_np, mask_np)
            mcts.supply_evaluations(values, priors)
    results["full_nn_mcts"] = (time.perf_counter() - t0) / 3
    results["full_nn_batches"] = 0
    # count batches
    mcts = ce.MctsContext(battle, n_simulations=n_sims, seed=42, c_puct=1.5)
    batches = 0
    while True:
        n = mcts.run_until_eval_needed(max_batch)
        if n == 0:
            break
        mcts.supply_heuristic_evaluations()
        batches += 1
    results["full_nn_batches"] = batches

    return results


def main():
    data = DataStore()
    tc = TypeChart.load()
    rs_data = ce.DataStore(DATA_DIR)
    evaluator = MctsEvaluator("imitation_ppo", device="cpu")

    n_positions = 10
    n_sims = 200
    max_batch = 32

    print(f"Profiling MCTS: {n_sims} sims, max_batch={max_batch}, {n_positions} positions")
    print("=" * 65)

    all_results = []
    for i in range(n_positions):
        b = build_position(data, rs_data, seed=i * 17 + 3)
        if b.is_over:
            continue
        r = profile_components(b, evaluator, tc, n_sims=n_sims, max_batch=max_batch)
        all_results.append(r)

    # average
    keys = all_results[0].keys()
    avg = {k: np.mean([r[k] for r in all_results]) for k in keys}

    print(f"\n{'Component':<30s} {'Time':>10s}  {'% of total':>10s}")
    print("-" * 55)

    total = avg["full_nn_mcts"]

    rows = [
        ("Rust tree (expand+select)", avg["rust_expand_select"]),
        ("Rust heuristic eval", avg["rust_heuristic_eval"]),
        ("get_pending_leaf_states", avg["get_leaf_states"]),
        ("Obs building (Python)", avg["obs_building"]),
        ("Torch inference", avg["torch_inference"]),
        ("supply_evaluations", avg["supply_evals"]),
    ]

    component_total = sum(t for _, t in rows)
    for name, t in rows:
        pct = t / total * 100 if total > 0 else 0
        print(f"  {name:<28s} {t*1000:>8.1f}ms  {pct:>8.1f}%")

    print(f"  {'---':<28s} {'---':>8s}  {'---':>8s}")
    print(f"  {'Component sum':<28s} {component_total*1000:>8.1f}ms  {component_total/total*100:>8.1f}%")
    print(f"  {'Full NN MCTS (measured)':<28s} {total*1000:>8.1f}ms  {'100.0':>8s}%")

    print(f"\n{'Metric':<35s} {'Value':>10s}")
    print("-" * 48)
    print(f"  {'Batches per search':<33s} {avg['full_nn_batches']:>8.1f}")
    print(f"  {'Leaves per batch':<33s} {avg['obs_n_leaves']:>8.1f}")
    print(f"  {'Obs build per leaf':<33s} {avg['obs_per_leaf']*1000:>7.2f}ms")
    print(f"  {'Torch batch size':<33s} {avg['torch_batch_size']:>8.1f}")
    print(f"  {'Heuristic-only MCTS':<33s} {avg['rust_tree_heuristic']*1000:>7.1f}ms")
    print(f"  {'NN MCTS overhead vs heuristic':<33s} {(total - avg['rust_tree_heuristic'])*1000:>7.1f}ms")
    print(f"  {'Overhead factor':<33s} {total / avg['rust_tree_heuristic']:>7.1f}x")


if __name__ == "__main__":
    main()
