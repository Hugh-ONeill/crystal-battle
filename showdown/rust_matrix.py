# full matchup matrix using Rust game driver
# MCTS(500ms) vs MCTS(50ms) across all team combinations
#
# Usage:
#   .venv/bin/python showdown/rust_matrix.py --games-per 3

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import poke_engine as pe
from showdown.local_battle import build_pe_state
from showdown.sample_teams import SAMPLE_TEAMS

NAMES = [
    "NidoGengar", "NidoMachamp", "JynxGengar", "VapExplosion",
    "ParaMarowak", "HeraExplosion", "DblElectric", "ThiefExegg",
    "AlaExplosion", "MoltExegg", "MCTSStrat",
]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--games-per", type=int, default=3)
    parser.add_argument("--s1-ms", type=int, default=500)
    parser.add_argument("--s2-ms", type=int, default=50)
    parser.add_argument("--max-turns", type=int, default=100)
    parser.add_argument("--teams", type=int, default=10,
                        help="Number of teams to test (max 11)")
    args = parser.parse_args()

    n_teams = min(args.teams, len(SAMPLE_TEAMS))
    t0 = time.time()

    print(f"MCTS({args.s1_ms}ms) vs MCTS({args.s2_ms}ms), "
          f"{args.games_per} games per matchup, {args.max_turns} max turns\n")

    # header
    print(f"{'':14s}", end="")
    for j in range(n_teams):
        print(f" {NAMES[j]:>12s}", end="")
    print("   AVG")
    print("-" * (14 + 13 * n_teams + 6))

    all_results = {}

    for i in range(n_teams):
        row_wins = 0
        row_total = 0
        print(f"{NAMES[i]:14s}", end="", flush=True)

        for j in range(n_teams):
            state = build_pe_state(SAMPLE_TEAMS[i], SAMPLE_TEAMS[j])
            w, l, d = pe.run_games(
                state, n_games=args.games_per,
                s1_search_ms=args.s1_ms, s2_search_ms=args.s2_ms,
                max_turns=args.max_turns,
            )
            all_results[(i, j)] = (w, l, d)
            total = w + l
            row_wins += w
            row_total += total

            if total > 0:
                pct = w / total * 100
                print(f" {pct:11.0f}%", end="", flush=True)
            else:
                print(f" {'draw':>12s}", end="", flush=True)

        avg = row_wins / row_total * 100 if row_total > 0 else 0
        print(f" {avg:5.0f}%")

    # column averages
    print(f"\n{'Opp AVG':14s}", end="")
    total_w = total_g = 0
    for j in range(n_teams):
        col_w = sum(all_results.get((i, j), (0, 0, 0))[0] for i in range(n_teams))
        col_t = sum(all_results.get((i, j), (0, 0, 0))[0] + all_results.get((i, j), (0, 0, 0))[1]
                    for i in range(n_teams))
        total_w += col_w
        total_g += col_t
        if col_t > 0:
            print(f" {col_w / col_t * 100:11.0f}%", end="")
        else:
            print(f" {'?':>12s}", end="")
    if total_g > 0:
        print(f" {total_w / total_g * 100:5.0f}%")
    else:
        print()

    elapsed = time.time() - t0
    n_games = sum(sum(v) for v in all_results.values())
    print(f"\n{elapsed:.0f}s total, {n_games} games played")


if __name__ == "__main__":
    main()
