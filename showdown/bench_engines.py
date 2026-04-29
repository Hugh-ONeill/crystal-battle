#!/usr/bin/env python3
"""
Engine-vs-engine MCTS bench: pe (dev) vs pe_ref (frozen reference).
Runs counterbalanced halves so any first-mover bias cancels.

Usage:
  .venv/bin/python showdown/bench_engines.py --games 60 --search-ms 300 --seed 1000
"""

from __future__ import annotations

import argparse
import random
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import poke_engine as pe_dev
import poke_engine_ref as pe_ref

from showdown.local_battle import build_pe_state
from showdown.sample_teams import SAMPLE_TEAMS


def _strip_switch_prefix(m: str) -> str:
    return m[7:] if m.startswith("switch ") else m


def _normalize_no_move(m: str) -> str:
    return "none" if m == "No Move" else m


def play_engine_game(team1_str: str, team2_str: str, search_ms: int,
                     p1_engine, p2_engine, max_turns: int = 250) -> int:
    """One game with each engine controlling its own side. State is owned by
    pe_dev (canonical resolver); both engines round-trip from string for their
    own search to keep types compatible. Returns 1 if P1 wins, 2 if P2 wins,
    0 on draw/timeout."""
    state = build_pe_state(team1_str, team2_str)

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

    return 0


def run_half(p1_engine, p2_engine, n_games: int, search_ms: int,
             team1: str, team2: str, seed_base: int | None) -> tuple[int, int, int, float]:
    wins = losses = draws = 0
    t0 = time.time()
    for i in range(n_games):
        if seed_base is not None:
            random.seed(seed_base + i)
        r = play_engine_game(team1, team2, search_ms, p1_engine, p2_engine)
        if r == 1:
            wins += 1
            print("W", end="", flush=True)
        elif r == 2:
            losses += 1
            print("L", end="", flush=True)
        else:
            draws += 1
            print("D", end="", flush=True)
    print()
    return wins, losses, draws, time.time() - t0


def main():
    parser = argparse.ArgumentParser(description="Counterbalanced engine-vs-engine MCTS bench")
    parser.add_argument("--games", type=int, default=30, help="games per half (total = 2x)")
    parser.add_argument("--search-ms", type=int, default=300)
    parser.add_argument("--team1", type=int, default=0)
    parser.add_argument("--team2", type=int, default=0)
    parser.add_argument("--seed", type=int, default=None,
                        help="per-game seed base; same seed across halves for paired comparison")
    args = parser.parse_args()

    team1 = SAMPLE_TEAMS[args.team1]
    team2 = SAMPLE_TEAMS[args.team2]

    print(f"=== half A: dev=P1 vs ref=P2 ({args.games} games, {args.search_ms}ms) ===")
    a_w, a_l, a_d, a_t = run_half(pe_dev, pe_ref, args.games, args.search_ms,
                                   team1, team2, args.seed)
    print(f"  dev(P1) vs ref(P2): {a_w}W {a_l}L {a_d}D in {a_t:.0f}s")

    print(f"=== half B: ref=P1 vs dev=P2 ({args.games} games, {args.search_ms}ms) ===")
    b_w, b_l, b_d, b_t = run_half(pe_ref, pe_dev, args.games, args.search_ms,
                                   team1, team2, args.seed)
    # in half B, "wins" mean ref won, so flip for dev's perspective
    print(f"  ref(P1) vs dev(P2): {b_w}W {b_l}L {b_d}D in {b_t:.0f}s")

    # dev's total record: half A wins (dev=P1) + half B losses (dev=P2)
    dev_wins = a_w + b_l
    dev_losses = a_l + b_w
    draws = a_d + b_d
    total = dev_wins + dev_losses
    pct = dev_wins / total * 100 if total > 0 else 0
    print()
    print(f"DEV vs REF (counterbalanced, {2 * args.games} games): "
          f"{dev_wins}W {dev_losses}L {draws}D ({pct:.1f}%)")


if __name__ == "__main__":
    main()
