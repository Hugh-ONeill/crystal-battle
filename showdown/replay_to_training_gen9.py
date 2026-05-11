#!/usr/bin/env python3
"""Bulk-process gen9ou replays into a training pickle for value_train.py.

Reads every replay in showdown/replays/gen9ou/, runs replay_to_trajectory on
each to build engine-stepped (state, label) sequences, and writes a pickle in
the schema value_train.py / policy_train.py expect:

    [(winner_int, [(state_str,), (state_str,), ...]), ...]

where winner_int is +1 if side_one won, -1 if side_two, 0 for draw.

Usage:
  .venv/bin/python showdown/replay_to_training_gen9.py \\
      --replays showdown/replays/gen9ou \\
      --out showdown/gen9ou_replay_data.pkl
"""

from __future__ import annotations

import argparse
import json
import pickle
import random
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from showdown.chaos_stats import ChaosStats
from showdown.replay_to_trajectory import replay_to_trajectory


def _label_to_winner(label: float) -> int:
    if label >= 0.99:
        return 1
    if label <= 0.01:
        return -1
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--replays", type=str, default="showdown/replays/gen9ou")
    ap.add_argument("--out", type=str, default="showdown/gen9ou_replay_data.pkl")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--limit", type=int, default=0,
                    help="Process at most N replays (0 = all)")
    ap.add_argument("--min-turns", type=int, default=2,
                    help="Drop replays whose engine trajectory has fewer than N turns")
    args = ap.parse_args()

    random.seed(args.seed)
    chaos = ChaosStats(format="gen9ou")

    replay_dir = Path(args.replays)
    files = sorted(replay_dir.glob("*.json"))
    if args.limit > 0:
        files = files[: args.limit]
    print(f"processing {len(files)} replays from {replay_dir}")

    all_results: list[tuple[int, list[tuple[str]]]] = []
    n_full = n_partial = n_empty = n_short = n_err = 0
    n_turns_total = 0

    t0 = time.time()
    for i, f in enumerate(files):
        try:
            with open(f) as fh:
                data = json.load(fh)
            traj = replay_to_trajectory(data, chaos)
        except Exception as e:
            n_err += 1
            if n_err <= 10:
                print(f"  err {f.name}: {type(e).__name__}: {e}")
            continue

        if not traj:
            n_empty += 1
            continue

        if len(traj) < args.min_turns:
            n_short += 1
            continue

        # All states in a trajectory share the same outcome label.
        winner = _label_to_winner(traj[0][1])
        turns = [(state_str,) for state_str, _label, _tr in traj]
        all_results.append((winner, turns))
        n_turns_total += len(turns)

        # crude full-vs-partial heuristic: treat ≥80% of source turns as "full".
        # We don't have the source-turn count cheaply here, so just bucket by length.
        if len(traj) >= 25:
            n_full += 1
        else:
            n_partial += 1

        if (i + 1) % 200 == 0:
            dt = time.time() - t0
            print(f"  [{i+1}/{len(files)}] {n_turns_total} turns kept, "
                  f"{dt:.1f}s elapsed")

    dt = time.time() - t0
    print()
    print(f"done in {dt:.1f}s")
    print(f"  kept: {len(all_results)} replays, {n_turns_total} turns")
    print(f"  full(≥25 turns)={n_full}  partial<25={n_partial}")
    print(f"  dropped: empty={n_empty} short(<{args.min_turns})={n_short} errs={n_err}")

    # outcome distribution
    if all_results:
        wins_p1 = sum(1 for w, _ in all_results if w > 0)
        wins_p2 = sum(1 for w, _ in all_results if w < 0)
        draws = sum(1 for w, _ in all_results if w == 0)
        print(f"  outcomes: p1={wins_p1} p2={wins_p2} draw={draws}")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "wb") as fh:
        pickle.dump(all_results, fh)
    print(f"  wrote {out_path} ({out_path.stat().st_size / 1e6:.1f} MB)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
