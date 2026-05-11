#!/usr/bin/env python3
"""
Round-robin team tournament: every team vs every other team using the same
engine + MCTS on both sides. Counterbalanced (each pair runs games each
direction). Aggregates per-team win rate to surface which teams are strong
in absolute terms within this team pool.

Usage:
  .venv/bin/python showdown/bench_roundrobin.py --games 10 --search-ms 200 --seed 1000
"""

from __future__ import annotations

import argparse
import multiprocessing as mp
import os
import random
import sys
import time
from itertools import combinations_with_replacement
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import poke_engine as pe

from showdown.local_battle import build_pe_state_gen9
from showdown.sample_teams_gen9 import (
    SAMPLE_TEAMS_GEN9,
    SPECIALTY_TEAM_INDICES,
    UBERS_TEAMS_GEN9,
)


def _strip_switch_prefix(m: str) -> str:
    return m[7:] if m.startswith("switch ") else m


def _normalize_no_move(m: str) -> str:
    return "none" if m == "No Move" else m


def play_one(team1_str: str, team2_str: str, search_ms: int,
             max_turns: int = 120) -> int:
    state = build_pe_state_gen9(team1_str, team2_str)
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
            ps1 = pe.State.from_string(s_str)
            r1 = pe.monte_carlo_tree_search(ps1, duration_ms=search_ms)
            p1_move = max(r1.side_one, key=lambda x: x.visits).move_choice

            ps2 = pe.State.from_string(s_str)
            r2 = pe.monte_carlo_tree_search(ps2, duration_ms=search_ms)
            p2_move = max(r2.side_two, key=lambda x: x.visits).move_choice
        except Exception:
            return 0

        p1_move = _normalize_no_move(p1_move)
        p2_move = _normalize_no_move(p2_move)
        if p1_move == "No Move" and p2_move == "No Move":
            return 0
        p1_clean = _strip_switch_prefix(p1_move)
        p2_clean = _strip_switch_prefix(p2_move)

        try:
            instructions = pe.generate_instructions(state, p1_clean, p2_clean)
        except Exception:
            return 0
        if not instructions:
            return 0

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


def _resolve_team(idx_with_pool: tuple[str, int]) -> str:
    pool, i = idx_with_pool
    return UBERS_TEAMS_GEN9[i] if pool == "ubers" else SAMPLE_TEAMS_GEN9[i]


def _worker(task):
    team_a, team_b, p1, p2, search_ms, seed = task
    if seed is not None:
        random.seed(seed)
    r = play_one(_resolve_team(p1), _resolve_team(p2), search_ms)
    return (team_a, team_b, p1, p2, r)


def main():
    default_workers = max(1, (os.cpu_count() or 4) - 2)
    parser = argparse.ArgumentParser(description="Round-robin team tournament")
    parser.add_argument("--games", type=int, default=10,
                        help="games per direction per pair (total per pair = 2*games)")
    parser.add_argument("--search-ms", type=int, default=200)
    parser.add_argument("--workers", type=int, default=default_workers)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--include-mirrors", action="store_true",
                        help="include team_i vs team_i pairs (sanity check, always near 50)")
    parser.add_argument("--include-specialty", action="store_true",
                        help="include specialty teams (e.g. Stall) — distorts standings via draws")
    parser.add_argument("--teams", type=str, default=None,
                        help="comma-separated list of OU team indices (default: all non-specialty)")
    parser.add_argument("--mode", choices=["roundrobin", "ubers-gauntlet"],
                        default="roundrobin",
                        help="roundrobin: OU teams play each other. "
                             "ubers-gauntlet: each OU team plays every Ubers team")
    args = parser.parse_args()

    n = len(SAMPLE_TEAMS_GEN9)
    if args.teams:
        team_ids = [int(x) for x in args.teams.split(",")]
    else:
        team_ids = [i for i in range(n)
                    if args.include_specialty or i not in SPECIALTY_TEAM_INDICES]

    pairs: list[tuple[tuple[str, int], tuple[str, int]]] = []
    if args.mode == "roundrobin":
        for a, b in combinations_with_replacement(team_ids, 2):
            if a == b and not args.include_mirrors:
                continue
            pairs.append((("ou", a), ("ou", b)))
        header = (f"round-robin: {len(team_ids)} OU teams, {len(pairs)} pairs")
    else:
        for ou_i in team_ids:
            for ub_i in range(len(UBERS_TEAMS_GEN9)):
                pairs.append((("ou", ou_i), ("ubers", ub_i)))
        header = (f"ubers-gauntlet: {len(team_ids)} OU teams x "
                  f"{len(UBERS_TEAMS_GEN9)} Ubers teams = {len(pairs)} pairs")

    tasks = []
    for (a, b) in pairs:
        for i in range(args.games):
            seed = (args.seed + i) if args.seed is not None else None
            # half A: team_a as P1
            tasks.append((a, b, a, b, args.search_ms, seed))
            # half B: team_b as P1 (swap)
            tasks.append((a, b, b, a, args.search_ms, seed))

    print(f"=== {header}, {2*args.games} games/pair, {args.search_ms}ms, "
          f"{args.workers} workers, {len(tasks)} total games ===")

    # pair_stats[(a,b)] -> {a_wins, b_wins, draws}
    pair_stats: dict[tuple[int, int], dict[str, int]] = {
        pair: {"a_wins": 0, "b_wins": 0, "draws": 0} for pair in pairs
    }

    t0 = time.time()
    completed = 0
    if args.workers <= 1:
        results = (_worker(t) for t in tasks)
    else:
        pool = mp.Pool(processes=args.workers)
        results = pool.imap_unordered(_worker, tasks)

    for (a, b, p1, p2, r) in results:
        completed += 1
        s = pair_stats[(a, b)]
        if r == 0:
            s["draws"] += 1
        else:
            winner_team = p1 if r == 1 else p2
            if winner_team == a:
                s["a_wins"] += 1
            else:
                s["b_wins"] += 1
        if completed % max(1, len(tasks) // 20) == 0:
            print(f"  [{completed}/{len(tasks)}] {time.time()-t0:.0f}s",
                  flush=True)

    if args.workers > 1:
        pool.close()
        pool.join()

    elapsed = time.time() - t0
    print(f"\nDone in {elapsed:.0f}s")

    ou_labels = {
        0: "Sun", 1: "BO", 2: "Balance", 3: "Stall", 4: "Rain",
        5: "TR", 6: "Screens HO", 7: "Sand", 8: "Webs",
        9: "Kingambit HO",
    }
    ubers_labels = {
        0: "Miraidon Bal", 1: "Kyogre+Don Bal", 2: "Double Prio HO",
    }

    def name_of(team: tuple[str, int]) -> str:
        pool, i = team
        return (f"OU{i}:{ou_labels.get(i, '?')}" if pool == "ou"
                else f"UB{i}:{ubers_labels.get(i, '?')}")

    # Aggregate per-team stats. Use the (pool, idx) tuple as the team key.
    all_teams = sorted({a for a, _ in pairs} | {b for _, b in pairs})
    team_wins = {t: 0 for t in all_teams}
    team_losses = {t: 0 for t in all_teams}
    team_draws = {t: 0 for t in all_teams}

    print("\n=== Per-pair results ===")
    for (a, b), s in sorted(pair_stats.items()):
        decided = s["a_wins"] + s["b_wins"]
        a_pct = (s["a_wins"] / decided * 100) if decided else 0
        total = decided + s["draws"]
        print(f"  {name_of(a):>16} vs {name_of(b):<16}: "
              f"{s['a_wins']:3d}W {s['b_wins']:3d}L {s['draws']:3d}D "
              f"({a_pct:.1f}% for {name_of(a)}, {decided}/{total} decided)")
        if a == b:
            continue
        team_wins[a] += s["a_wins"]
        team_losses[a] += s["b_wins"]
        team_draws[a] += s["draws"]
        team_wins[b] += s["b_wins"]
        team_losses[b] += s["a_wins"]
        team_draws[b] += s["draws"]

    rows = []
    for t in all_teams:
        w, l, d = team_wins[t], team_losses[t], team_draws[t]
        decided = w + l
        pct = (w / decided * 100) if decided else 0
        rows.append((pct, w, l, d, t))
    rows.sort(reverse=True)

    print("\n=== Per-team standings (decided games only) ===")
    print(f"  {'team':>18}  {'win%':>6}  {'W':>4}/{'L':>4}/{'D':>4}  decided%")
    for pct, w, l, d, t in rows:
        total = w + l + d
        decided_frac = ((w + l) / total * 100) if total else 0
        print(f"  {name_of(t):>18}  {pct:5.1f}%  "
              f"{w:4d}/{l:4d}/{d:4d}  {decided_frac:5.1f}%")


if __name__ == "__main__":
    main()
