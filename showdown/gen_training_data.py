#!/usr/bin/env python3
# generate training data: self-play + diverse matchups
#
# Usage:
#   PYTHONUNBUFFERED=1 .venv/bin/python showdown/gen_training_data.py \
#     --team 13 --self-play 1000 --diverse 200 --out hypnosis_mcts_data.pkl

import argparse
import pickle
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
import poke_engine as pe
from showdown.local_battle import build_pe_state
from showdown.sample_teams import SAMPLE_TEAMS

OPPONENT_INDICES = list(range(10))  # teams 0-9

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--team", type=int, required=True, help="our team index")
    parser.add_argument("--self-play", type=int, default=1000,
                        help="number of self-play games (500ms/500ms)")
    parser.add_argument("--diverse", type=int, default=200,
                        help="games per opponent")
    parser.add_argument("--s1-ms", type=int, default=500)
    parser.add_argument("--s2-ms", type=int, default=500)
    parser.add_argument("--max-turns", type=int, default=500,
                        help="hard turn cap; 200 was too short and produced 70 percent draws")
    parser.add_argument("--out", type=str, required=True)
    args = parser.parse_args()

    all_results = []

    # ---- self-play (both sides at 500ms, both sides recorded) ----
    if args.self_play > 0:
        state = build_pe_state(SAMPLE_TEAMS[args.team], SAMPLE_TEAMS[args.team])
        print(f"Self-play: {args.self_play} games, {args.s1_ms}ms/{args.s2_ms}ms...")
        results = pe.run_games_recorded(
            state, n_games=args.self_play,
            s1_search_ms=args.s1_ms, s2_search_ms=args.s2_ms, max_turns=args.max_turns,
        )
        turns = sum(len(r[1]) for r in results)
        print(f"  {len(results)} games, {turns} turns")
        all_results.extend(results)

    # ---- diverse matchups ----
    if args.diverse > 0:
        for opp_idx in OPPONENT_INDICES:
            state = build_pe_state(SAMPLE_TEAMS[args.team], SAMPLE_TEAMS[opp_idx])
            print(f"vs team {opp_idx}: {args.diverse} games, {args.s1_ms}ms/{args.s2_ms}ms...")
            results = pe.run_games_recorded(
                state, n_games=args.diverse,
                s1_search_ms=args.s1_ms, s2_search_ms=args.s2_ms, max_turns=args.max_turns,
            )
            turns = sum(len(r[1]) for r in results)
            print(f"  {len(results)} games, {turns} turns")
            all_results.extend(results)

    total_turns = sum(len(r[1]) for r in all_results)
    print(f"\nTotal: {len(all_results)} games, {total_turns} turns")

    with open(args.out, "wb") as f:
        pickle.dump(all_results, f)
    print(f"Saved to {args.out}")


if __name__ == "__main__":
    main()
