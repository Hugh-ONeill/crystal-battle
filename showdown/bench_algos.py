#!/usr/bin/env python3
"""
Algorithm-vs-algorithm bench: MCTS vs iterative-deepening expectiminimax,
both running on the same engine (so eval is identical). Counterbalanced halves.

Usage:
  .venv/bin/python showdown/bench_algos.py --gen 9 --games 30 --search-ms 300 --seed 1000
"""

from __future__ import annotations

import argparse
import multiprocessing as mp
import os
import random
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import poke_engine as pe

from showdown.local_battle import build_pe_state, build_pe_state_gen9
from showdown.sample_teams import SAMPLE_TEAMS
from showdown.sample_teams_gen9 import SAMPLE_TEAMS_GEN9


def _strip_switch_prefix(m: str) -> str:
    return m[7:] if m.startswith("switch ") else m


def _normalize_no_move(m: str) -> str:
    return "none" if m == "No Move" else m


def _mcts_pick(state, duration_ms: int, side: str) -> str:
    r = pe.monte_carlo_tree_search(state, duration_ms=duration_ms)
    moves = r.side_one if side == "s1" else r.side_two
    return max(moves, key=lambda x: x.visits).move_choice


def _emm_pick(state, duration_ms: int, side: str) -> str:
    """Safest move from the EMM payoff matrix.
    Matrix is row-major over s1, scores from s1's perspective.
    s1 picks max-over-rows of min-over-cols (safest for self).
    s2 picks min-over-cols of max-over-rows (safest for self, since
    higher matrix values are better for s1 / worse for s2).
    """
    r = pe.iterative_deepening_expectiminimax(state, duration_ms=duration_ms)
    n1 = len(r.side_one)
    n2 = len(r.side_two)

    def _score(i: int, j: int) -> float:
        v = r.matrix[i * n2 + j]
        return v if v is not None else float("-inf")

    if side == "s1":
        best_i, best_val = 0, float("-inf")
        for i in range(n1):
            row_min = min(_score(i, j) for j in range(n2))
            if row_min > best_val:
                best_val, best_i = row_min, i
        return r.side_one[best_i]
    else:
        best_j, best_val = 0, float("inf")
        for j in range(n2):
            col_max = max(_score(i, j) for i in range(n1))
            if col_max < best_val:
                best_val, best_j = col_max, j
        return r.side_two[best_j]


def _pick_for(algo: str, state, duration_ms: int, side: str) -> str:
    if algo == "mcts":
        return _mcts_pick(state, duration_ms, side)
    elif algo == "emm":
        return _emm_pick(state, duration_ms, side)
    raise ValueError(f"unknown algo: {algo}")


def play_algo_game(team1_str: str, team2_str: str, search_ms: int,
                   p1_algo: str, p2_algo: str, max_turns: int = 120,
                   gen: int = 2) -> int:
    builder = build_pe_state_gen9 if gen == 9 else build_pe_state
    state = builder(team1_str, team2_str)
    prev_str = ""
    stuck_turns = 0

    for _ in range(max_turns):
        s1_alive = sum(1 for p in state.side_one.pokemon if p.hp > 0)
        s2_alive = sum(1 for p in state.side_two.pokemon if p.hp > 0)
        if s1_alive == 0:
            return 2
        if s2_alive == 0:
            return 1

        s_str = state.to_string()

        try:
            p1_state = pe.State.from_string(s_str)
            p1_move = _pick_for(p1_algo, p1_state, search_ms, "s1")
        except Exception as e:
            print(f"\n  P1 ({p1_algo}) error: {e}")
            break

        try:
            p2_state = pe.State.from_string(s_str)
            p2_move = _pick_for(p2_algo, p2_state, search_ms, "s2")
        except Exception as e:
            print(f"\n  P2 ({p2_algo}) error: {e}")
            break

        p1_move = _normalize_no_move(p1_move)
        p2_move = _normalize_no_move(p2_move)
        if p1_move == "No Move" and p2_move == "No Move":
            break
        p1_clean = _strip_switch_prefix(p1_move)
        p2_clean = _strip_switch_prefix(p2_move)

        try:
            instructions = pe.generate_instructions(state, p1_clean, p2_clean)
        except Exception as e:
            print(f"\n  resolve error: {e}")
            break
        if not instructions:
            break

        roll = random.random() * 100
        cum = 0.0
        chosen = instructions[0]
        for inst in instructions:
            cum += inst.percentage
            if roll <= cum:
                chosen = inst
                break

        state = state.apply_instructions(chosen)

        cur_str = state.to_string()
        if cur_str == prev_str:
            stuck_turns += 1
            if stuck_turns >= 3:
                return 0
        else:
            stuck_turns = 0
        prev_str = cur_str

    return 0


def _worker_play(task):
    half_label, team1, team2, search_ms, p1_algo, p2_algo, seed, gen = task
    if seed is not None:
        random.seed(seed)
    r = play_algo_game(team1, team2, search_ms, p1_algo, p2_algo, gen=gen)
    return half_label, r


def run_both_halves(n_games: int, search_ms: int, team1: str, team2: str,
                    seed_base: int | None, workers: int, algo_a: str,
                    algo_b: str, gen: int = 2):
    """Half A: algo_a as P1, algo_b as P2. Half B: swapped."""
    tasks = []
    for i in range(n_games):
        seed = (seed_base + i) if seed_base is not None else None
        tasks.append(("A", team1, team2, search_ms, algo_a, algo_b, seed, gen))
        tasks.append(("B", team1, team2, search_ms, algo_b, algo_a, seed, gen))

    stats = {"A": [0, 0, 0], "B": [0, 0, 0]}

    def record(label, r):
        idx = 0 if r == 1 else (1 if r == 2 else 2)
        stats[label][idx] += 1
        print({0: "W", 1: "L", 2: "D"}[idx], end="", flush=True)

    if workers <= 1:
        for task in tasks:
            label, r = _worker_play(task)
            record(label, r)
    else:
        with mp.Pool(processes=workers) as pool:
            for label, r in pool.imap_unordered(_worker_play, tasks):
                record(label, r)
    print()
    return stats


def main():
    default_workers = max(1, (os.cpu_count() or 4) - 2)
    parser = argparse.ArgumentParser(description="Counterbalanced algorithm-vs-algorithm bench")
    parser.add_argument("--algo-a", choices=["mcts", "emm"], default="mcts",
                        help="primary algorithm (the reported win-rate side)")
    parser.add_argument("--algo-b", choices=["mcts", "emm"], default="emm",
                        help="opponent algorithm")
    parser.add_argument("--games", type=int, default=20)
    parser.add_argument("--search-ms", type=int, default=300)
    parser.add_argument("--team1", type=int, default=0)
    parser.add_argument("--team2", type=int, default=0)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--workers", type=int, default=default_workers)
    parser.add_argument("--gen", type=int, default=9, choices=[2, 9])
    args = parser.parse_args()

    teams = SAMPLE_TEAMS_GEN9 if args.gen == 9 else SAMPLE_TEAMS
    team1 = teams[args.team1]
    team2 = teams[args.team2]

    print(f"=== gen{args.gen}: {args.algo_a.upper()} vs {args.algo_b.upper()}, "
          f"{2 * args.games} games ({args.games}/half), "
          f"{args.search_ms}ms, {args.workers} workers ===")
    t0 = time.time()
    stats = run_both_halves(args.games, args.search_ms, team1, team2,
                            args.seed, args.workers, args.algo_a, args.algo_b,
                            gen=args.gen)
    elapsed = time.time() - t0

    a_w, a_l, a_d = stats["A"]
    b_w, b_l, b_d = stats["B"]
    print(f"  {args.algo_a}(P1) vs {args.algo_b}(P2): {a_w}W {a_l}L {a_d}D")
    print(f"  {args.algo_b}(P1) vs {args.algo_a}(P2): {b_w}W {b_l}L {b_d}D")

    a_wins = a_w + b_l
    a_losses = a_l + b_w
    draws = a_d + b_d
    total = a_wins + a_losses
    pct = a_wins / total * 100 if total > 0 else 0
    print()
    print(f"{args.algo_a.upper()} vs {args.algo_b.upper()} (counterbalanced, "
          f"{2 * args.games} games, {elapsed:.0f}s): "
          f"{a_wins}W {a_losses}L {draws}D ({pct:.1f}%)")


if __name__ == "__main__":
    main()
