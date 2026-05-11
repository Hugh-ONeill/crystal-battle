#!/usr/bin/env python3
"""Generate MCTS-distillation targets for value-net training.

Replaces per-game outcome labels (0/1) with per-state MCTS-derived V(s) values
in [0, 1]. The model trained on these targets sees continuous values that
distinguish 0.55 mid-game positions from 0.95 dominant ones — outcome labels
collapse all of those into 0 or 1, leaving the model nothing to learn from.

Reads the existing replay pickle (state_strs + outcome label) and writes a new
pickle with the same per-game grouping but an `mcts_v` value per state.

Schema:
  input:  [(winner_int, [(state_str,), ...]), ...]
  output: [(winner_int, [(state_str, mcts_v_p1), ...]), ...]

Usage:
  .venv/bin/python showdown/distill_mcts_targets.py \\
      --input showdown/gen9ou_replay_data.pkl \\
      --output showdown/gen9ou_distill_data.pkl \\
      --search-ms 2000 --workers 12
"""

from __future__ import annotations

import argparse
import multiprocessing as mp
import os
import pickle
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))


def _mcts_value(state_str: str, search_ms: int) -> float:
    """Run plain MCTS, return visit-weighted V(s) from p1's perspective.

    Imported inside the worker so each forked process binds its own pe handle.
    """
    import poke_engine as pe
    state = pe.State.from_string(state_str)
    r = pe.monte_carlo_tree_search(state, duration_ms=search_ms)
    total_score = sum(x.total_score for x in r.side_one)
    total_visits = sum(x.visits for x in r.side_one)
    if total_visits <= 0:
        return 0.5
    return total_score / total_visits


def _worker(task: tuple[int, int, str, int]) -> tuple[int, int, float]:
    """Pool worker. Args: (game_idx, turn_idx, state_str, search_ms).
    Returns: (game_idx, turn_idx, mcts_v)."""
    game_idx, turn_idx, state_str, search_ms = task
    try:
        v = _mcts_value(state_str, search_ms)
    except Exception:
        v = 0.5  # fall back to neutral on engine error
    return game_idx, turn_idx, v


def main() -> int:
    default_workers = max(1, (os.cpu_count() or 4) - 2)
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", type=str, default="showdown/gen9ou_replay_data.pkl")
    ap.add_argument("--output", type=str, default="showdown/gen9ou_distill_data.pkl")
    ap.add_argument("--search-ms", type=int, default=2000,
                    help="MCTS budget per state. 5–10× the inference-time budget "
                         "is the rule of thumb (bench uses 200–500 ms).")
    ap.add_argument("--workers", type=int, default=default_workers,
                    help=f"parallel MCTS workers (default: cpu_count-2 = {default_workers})")
    ap.add_argument("--limit-games", type=int, default=0,
                    help="process at most N games (0 = all). Useful for smoke tests.")
    args = ap.parse_args()

    print(f"loading {args.input}")
    with open(args.input, "rb") as f:
        data: list[tuple[int, list]] = pickle.load(f)
    if args.limit_games > 0:
        data = data[: args.limit_games]
    n_games = len(data)
    n_turns = sum(len(turns) for _, turns in data)
    print(f"  {n_games} games, {n_turns} states to evaluate at {args.search_ms} ms each")
    print(f"  {args.workers} workers — est. wall: "
          f"~{n_turns * args.search_ms / args.workers / 1000 / 60:.0f} min")

    # build task list: (game_idx, turn_idx, state_str, search_ms)
    tasks = []
    for g_idx, (_winner, turns) in enumerate(data):
        for t_idx, turn in enumerate(turns):
            state_str = turn[0]
            tasks.append((g_idx, t_idx, state_str, args.search_ms))

    # fan out
    results: dict[tuple[int, int], float] = {}
    t0 = time.time()
    done = 0
    progress_step = max(1, n_turns // 50)

    if args.workers <= 1:
        for task in tasks:
            g, t, v = _worker(task)
            results[(g, t)] = v
            done += 1
            if done % progress_step == 0:
                dt = time.time() - t0
                rate = done / dt if dt > 0 else 0
                eta = (n_turns - done) / rate if rate > 0 else 0
                print(f"  [{done}/{n_turns}] {rate:.1f}/s  eta {eta/60:.1f} min")
    else:
        with mp.Pool(processes=args.workers) as pool:
            for g, t, v in pool.imap_unordered(_worker, tasks, chunksize=4):
                results[(g, t)] = v
                done += 1
                if done % progress_step == 0:
                    dt = time.time() - t0
                    rate = done / dt if dt > 0 else 0
                    eta = (n_turns - done) / rate if rate > 0 else 0
                    print(f"  [{done}/{n_turns}] {rate:.1f}/s  eta {eta/60:.1f} min",
                          flush=True)

    elapsed = time.time() - t0
    print()
    print(f"done in {elapsed:.0f}s ({elapsed/60:.1f} min)")

    # quick stats on the V distribution
    all_v = list(results.values())
    if all_v:
        import statistics
        print(f"  V distribution: mean={statistics.mean(all_v):.3f}, "
              f"stdev={statistics.stdev(all_v) if len(all_v) > 1 else 0:.3f}, "
              f"min={min(all_v):.3f}, max={max(all_v):.3f}")
        n_decisive = sum(1 for v in all_v if v < 0.05 or v > 0.95)
        n_uncertain = sum(1 for v in all_v if 0.4 < v < 0.6)
        print(f"  decisive (<0.05 or >0.95): {n_decisive} ({100*n_decisive/len(all_v):.1f}%)")
        print(f"  uncertain (0.4–0.6):       {n_uncertain} ({100*n_uncertain/len(all_v):.1f}%)")

    # write output: same structure, but turns now have (state_str, mcts_v)
    out_data = []
    for g_idx, (winner, turns) in enumerate(data):
        new_turns = []
        for t_idx, turn in enumerate(turns):
            state_str = turn[0]
            v = results.get((g_idx, t_idx), 0.5)
            new_turns.append((state_str, v))
        out_data.append((winner, new_turns))

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "wb") as f:
        pickle.dump(out_data, f)
    print(f"  wrote {out_path} ({out_path.stat().st_size / 1e6:.1f} MB)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
