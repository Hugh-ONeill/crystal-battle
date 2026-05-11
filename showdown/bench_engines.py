#!/usr/bin/env python3
"""
Engine-vs-engine MCTS bench: pe (dev) vs pe_ref (frozen reference).
Runs counterbalanced halves so any first-mover bias cancels.

Usage:
  .venv/bin/python showdown/bench_engines.py --games 60 --search-ms 300 --seed 1000
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

import poke_engine as pe_dev
import poke_engine_ref as pe_ref

from showdown.local_battle import build_pe_state, build_pe_state_gen9
from showdown.sample_teams import SAMPLE_TEAMS
from showdown.sample_teams_gen9 import SAMPLE_TEAMS_GEN9


def _engine_for(eid: str):
    """Module lookup so workers can resolve engines from a string id."""
    return pe_dev if eid == "dev" else pe_ref


def _strip_switch_prefix(m: str) -> str:
    return m[7:] if m.startswith("switch ") else m


def _normalize_no_move(m: str) -> str:
    return "none" if m == "No Move" else m


def play_engine_game(team1_str: str, team2_str: str, search_ms: int,
                     p1_engine, p2_engine, max_turns: int = 120,
                     gen: int = 2) -> int:
    """One game with each engine controlling its own side. State is owned by
    pe_dev (canonical resolver); both engines round-trip from string for their
    own search to keep types compatible. Returns 1 if P1 wins, 2 if P2 wins,
    0 on draw/timeout."""
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
            p1_state = p1_engine.State.from_string(s_str)
            r1 = p1_engine.monte_carlo_tree_search(p1_state, duration_ms=search_ms)
            p1_move = max(r1.side_one, key=lambda x: x.visits).move_choice
        except Exception as e:
            print(f"\n  P1 error: {e}")
            break

        try:
            p2_state = p2_engine.State.from_string(s_str)
            r2 = p2_engine.monte_carlo_tree_search(p2_state, duration_ms=search_ms)
            p2_move = max(r2.side_two, key=lambda x: x.visits).move_choice
        except Exception as e:
            print(f"\n  P2 error: {e}")
            break

        p1_move = _normalize_no_move(p1_move)
        p2_move = _normalize_no_move(p2_move)
        if p1_move == "No Move" and p2_move == "No Move":
            break
        p1_clean = _strip_switch_prefix(p1_move)
        p2_clean = _strip_switch_prefix(p2_move)

        try:
            instructions = pe_dev.generate_instructions(state, p1_clean, p2_clean)
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

        # Engine has a no-op edge case where both sides' moves resolve to zero
        # instructions (e.g. failed Sucker Punch + maxed-stage boost move). The
        # state never advances. Bail with draw after a few stuck turns.
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
    """Pool worker: resolve engines from string ids, seed, run one game.
    Returns (half_label, result) so the caller can demux per-half stats."""
    half_label, team1, team2, search_ms, p1_id, p2_id, seed, gen = task
    if seed is not None:
        random.seed(seed)
    r = play_engine_game(team1, team2, search_ms,
                         _engine_for(p1_id), _engine_for(p2_id), gen=gen)
    return half_label, r


def run_both_halves(n_games: int, search_ms: int, team1: str, team2: str,
                    seed_base: int | None, workers: int, gen: int = 2):
    """Submit both halves' games into a single pool to overlap tail latency.
    Returns dict mapping half_label -> (wins, losses, draws)."""
    tasks = []
    for i in range(n_games):
        seed = (seed_base + i) if seed_base is not None else None
        tasks.append(("A", team1, team2, search_ms, "dev", "ref", seed, gen))
        tasks.append(("B", team1, team2, search_ms, "ref", "dev", seed, gen))

    stats = {"A": [0, 0, 0], "B": [0, 0, 0]}  # [W, L, D] per half

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
    parser = argparse.ArgumentParser(description="Counterbalanced engine-vs-engine MCTS bench")
    parser.add_argument("--games", type=int, default=30, help="games per half (total = 2x)")
    parser.add_argument("--search-ms", type=int, default=300)
    parser.add_argument("--team1", type=int, default=0)
    parser.add_argument("--team2", type=int, default=0)
    parser.add_argument("--seed", type=int, default=None,
                        help="per-game seed base; same seed across halves for paired comparison")
    parser.add_argument("--workers", type=int, default=default_workers,
                        help=f"parallel game workers (default: cpu_count-2 = {default_workers})")
    parser.add_argument("--gen", type=int, default=2, choices=[2, 9],
                        help="generation: 2 for gen2 SAMPLE_TEAMS, 9 for gen9 SAMPLE_TEAMS_GEN9")
    args = parser.parse_args()

    teams = SAMPLE_TEAMS_GEN9 if args.gen == 9 else SAMPLE_TEAMS
    team1 = teams[args.team1]
    team2 = teams[args.team2]

    print(f"=== gen{args.gen}: {2 * args.games} games ({args.games}/half), "
          f"{args.search_ms}ms, {args.workers} workers ===")
    t0 = time.time()
    stats = run_both_halves(args.games, args.search_ms, team1, team2,
                            args.seed, args.workers, gen=args.gen)
    elapsed = time.time() - t0

    a_w, a_l, a_d = stats["A"]
    b_w, b_l, b_d = stats["B"]
    print(f"  dev(P1) vs ref(P2): {a_w}W {a_l}L {a_d}D")
    print(f"  ref(P1) vs dev(P2): {b_w}W {b_l}L {b_d}D")

    # dev's record: half A wins (dev=P1) + half B losses (dev=P2)
    dev_wins = a_w + b_l
    dev_losses = a_l + b_w
    draws = a_d + b_d
    total = dev_wins + dev_losses
    pct = dev_wins / total * 100 if total > 0 else 0
    print()
    print(f"DEV vs REF (counterbalanced, {2 * args.games} games, {elapsed:.0f}s): "
          f"{dev_wins}W {dev_losses}L {draws}D ({pct:.1f}%)")


if __name__ == "__main__":
    main()
