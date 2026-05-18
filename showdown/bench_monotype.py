#!/usr/bin/env python3
"""
Round-robin tournament over Gen 9 Monotype teams from a Showdown paste file.

Mirrors showdown/bench_roundrobin.py but:
  - loads teams from a `=== [gen9monotype] NAME ===`-delimited text file
  - suppresses Tera in the MCTS agent's chosen move (monotype rule: no Tera)
  - reports per-team win-rate with a Wilson 95% CI so outliers are obvious

Usage:
  .venv/bin/python showdown/bench_monotype.py \\
      --teams-file ~/teams_v3.txt --games 4 --search-ms 200
"""

from __future__ import annotations

import argparse
import math
import multiprocessing as mp
import os
import random
import re
import sys
import time
from itertools import combinations_with_replacement
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import poke_engine as pe

from showdown.local_battle import build_pe_state_gen9


_HEADER_RE = re.compile(r"^=== \[gen9monotype\] (.+?) ===\s*$", re.M)


def load_teams(path: Path) -> list[tuple[str, str]]:
    """Parse a Showdown paste file into [(team_name, team_body), ...]."""
    text = Path(path).expanduser().read_text()
    parts = _HEADER_RE.split(text)
    # parts: [preamble, name1, body1, name2, body2, ...]
    teams = []
    for i in range(1, len(parts), 2):
        name = parts[i].strip()
        body = parts[i + 1].strip()
        if body:
            teams.append((name, body))
    return teams


def _best_non_tera(side_results) -> str:
    """Pick the move_choice with the most visits, excluding '-tera' variants.

    Falls back to the overall-best move if every option is tera (shouldn't
    happen in practice but guards against an empty filter).
    """
    non_tera = [x for x in side_results if not x.move_choice.endswith("-tera")]
    pool = non_tera if non_tera else list(side_results)
    return max(pool, key=lambda x: x.visits).move_choice


def _strip_switch_prefix(m: str) -> str:
    return m[7:] if m.startswith("switch ") else m


def _normalize_no_move(m: str) -> str:
    return "none" if m == "No Move" else m


def play_one(team1_str: str, team2_str: str, search_ms: int,
             max_turns: int = 120) -> int:
    """Play one game; return 1 if side_one wins, 2 if side_two, 0 on draw/error."""
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
            p1_move = _best_non_tera(r1.side_one)

            ps2 = pe.State.from_string(s_str)
            r2 = pe.monte_carlo_tree_search(ps2, duration_ms=search_ms)
            p2_move = _best_non_tera(r2.side_two)
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


# multiprocessing payload: (a_idx, b_idx, p1_idx, p2_idx, search_ms, seed)
_TEAMS: list[tuple[str, str]] = []  # set per-process in _init_worker


def _init_worker(teams):
    global _TEAMS
    _TEAMS = teams


def _worker(task):
    a, b, p1, p2, search_ms, seed = task
    if seed is not None:
        random.seed(seed)
    r = play_one(_TEAMS[p1][1], _TEAMS[p2][1], search_ms)
    return (a, b, p1, p2, r)


def wilson_ci(wins: int, decided: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson score 95% CI for a binomial proportion."""
    if decided == 0:
        return (0.0, 0.0)
    n = decided
    p = wins / n
    denom = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    margin = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return (max(0.0, center - margin), min(1.0, center + margin))


def main():
    default_workers = max(1, (os.cpu_count() or 4) - 2)
    parser = argparse.ArgumentParser(description="Monotype round-robin bench")
    parser.add_argument("--teams-file", type=str, default="~/teams_v3.txt",
                        help="path to Showdown paste file with monotype teams")
    parser.add_argument("--games", type=int, default=4,
                        help="games per direction per pair (total per pair = 2*games)")
    parser.add_argument("--search-ms", type=int, default=200)
    parser.add_argument("--workers", type=int, default=default_workers)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--include-mirrors", action="store_true")
    parser.add_argument("--max-turns", type=int, default=120)
    args = parser.parse_args()

    teams = load_teams(Path(args.teams_file))
    if len(teams) < 2:
        print(f"only {len(teams)} teams parsed from {args.teams_file}", file=sys.stderr)
        sys.exit(1)

    n = len(teams)
    team_ids = list(range(n))

    pairs: list[tuple[int, int]] = []
    for a, b in combinations_with_replacement(team_ids, 2):
        if a == b and not args.include_mirrors:
            continue
        pairs.append((a, b))

    tasks = []
    for (a, b) in pairs:
        for i in range(args.games):
            seed = (args.seed + i) if args.seed is not None else None
            # counterbalance: half with a as P1, half with b as P1
            tasks.append((a, b, a, b, args.search_ms, seed))
            tasks.append((a, b, b, a, args.search_ms, seed))

    print(f"=== monotype round-robin: {n} teams, {len(pairs)} pairs, "
          f"{2*args.games} games/pair, {args.search_ms}ms search, "
          f"{args.workers} workers, {len(tasks)} total games ===")
    for i, (name, _) in enumerate(teams):
        print(f"  [{i:2d}] {name}")

    pair_stats: dict[tuple[int, int], dict[str, int]] = {
        pair: {"a_wins": 0, "b_wins": 0, "draws": 0} for pair in pairs
    }

    t0 = time.time()
    completed = 0
    if args.workers <= 1:
        _init_worker(teams)
        results = (_worker(t) for t in tasks)
        pool = None
    else:
        pool = mp.Pool(processes=args.workers,
                       initializer=_init_worker, initargs=(teams,))
        results = pool.imap_unordered(_worker, tasks)

    progress_step = max(1, len(tasks) // 20)
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
        if completed % progress_step == 0:
            print(f"  [{completed}/{len(tasks)}] {time.time()-t0:.0f}s",
                  flush=True)

    if pool is not None:
        pool.close()
        pool.join()

    elapsed = time.time() - t0
    print(f"\nDone in {elapsed:.0f}s")

    def short(name: str, w: int = 22) -> str:
        return (name[:w-1] + "…") if len(name) > w else name

    # aggregate per-team
    team_wins = [0] * n
    team_losses = [0] * n
    team_draws = [0] * n
    print("\n=== Per-pair results ===")
    for (a, b), s in sorted(pair_stats.items()):
        decided = s["a_wins"] + s["b_wins"]
        a_pct = (s["a_wins"] / decided * 100) if decided else 0
        total = decided + s["draws"]
        print(f"  {short(teams[a][0]):>22} vs {short(teams[b][0]):<22}: "
              f"{s['a_wins']:3d}W {s['b_wins']:3d}L {s['draws']:3d}D "
              f"({a_pct:5.1f}% for {short(teams[a][0])}, {decided}/{total} decided)")
        if a == b:
            continue
        team_wins[a] += s["a_wins"]
        team_losses[a] += s["b_wins"]
        team_draws[a] += s["draws"]
        team_wins[b] += s["b_wins"]
        team_losses[b] += s["a_wins"]
        team_draws[b] += s["draws"]

    rows = []
    for i in range(n):
        w, l, d = team_wins[i], team_losses[i], team_draws[i]
        decided = w + l
        pct = (w / decided * 100) if decided else 0.0
        lo, hi = wilson_ci(w, decided)
        rows.append((pct, w, l, d, lo, hi, i))
    rows.sort(reverse=True)

    print("\n=== Per-team standings (Wilson 95% CI on decided games) ===")
    print(f"  {'team':>22}  {'win%':>6}  {'95% CI':>15}  "
          f"{'W':>4}/{'L':>4}/{'D':>4}  decided%")
    for pct, w, l, d, lo, hi, i in rows:
        total = w + l + d
        decided_frac = ((w + l) / total * 100) if total else 0
        ci_str = f"[{lo*100:4.1f},{hi*100:5.1f}]"
        print(f"  {short(teams[i][0]):>22}  {pct:5.1f}%  {ci_str:>15}  "
              f"{w:4d}/{l:4d}/{d:4d}  {decided_frac:5.1f}%")

    print("\nNote: Tera-suppressed (monotype rule). Move choices ending in "
          "'-tera' were filtered before each turn.")


if __name__ == "__main__":
    main()
