#!/usr/bin/env python3
# watch recorded games turn-by-turn
#
# Usage:
#   PYTHONUNBUFFERED=1 .venv/bin/python showdown/watch_game.py \
#     --team 13 --opponent 3 --s1-ms 500 --s2-ms 500

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
import poke_engine as pe
from showdown.local_battle import build_pe_state
from showdown.sample_teams import SAMPLE_TEAMS

TEAM_NAMES = [
    "NidoGengar", "NidoMachamp", "JynxGengar", "VapExplosion",
    "ParaMarowak", "HeraExplosion", "DblElectric", "ThiefExegg",
    "AlaExplosion", "MoltExegg", "NidoGengarTbolt", "MCTSStrat",
    "Hybrid", "Hypnosis",
]


def parse_side_summary(state_str: str, side: int) -> list[str]:
    """Parse HP/status for all pokemon on a side."""
    major = state_str.split("/")
    parts = major[side].split("=")
    active_idx = int(parts[6]) if len(parts) > 6 else 0
    lines = []
    for i in range(6):
        fields = parts[i].split(",")
        name = fields[0]
        if name == "NONE":
            continue
        hp = int(fields[6])
        maxhp = int(fields[7])
        status = fields[18] if len(fields) > 18 else "NONE"
        marker = ">" if i == active_idx else " "
        status_str = f" [{status.lower()[:3]}]" if status != "NONE" else ""
        if hp <= 0:
            lines.append(f"  {marker} {name:12s} fainted")
        else:
            pct = hp * 100 // max(maxhp, 1)
            lines.append(f"  {marker} {name:12s} {hp:3d}/{maxhp} ({pct:3d}%){status_str}")
    return lines


def show_game(game_result, game_num=1):
    """Print a recorded game turn by turn."""
    winner_val, turns = game_result
    winner = "P1" if winner_val > 0 else ("P2" if winner_val < 0 else "Draw")

    print(f"\n{'=' * 60}")
    print(f"Game {game_num} -- {len(turns)} turns, winner: {winner}")
    print(f"{'=' * 60}")

    for i, turn_data in enumerate(turns):
        state_str = turn_data[0]
        s1_move = turn_data[1]
        s1_visits = turn_data[2]
        s2_move = turn_data[3]
        s2_visits = turn_data[4] if len(turn_data) > 4 else []

        # active pokemon
        p1_lines = parse_side_summary(state_str, 0)
        p2_lines = parse_side_summary(state_str, 1)

        # find active names
        major = state_str.split("/")
        p1_parts = major[0].split("=")
        p2_parts = major[1].split("=")
        p1_active_idx = int(p1_parts[6]) if len(p1_parts) > 6 else 0
        p2_active_idx = int(p2_parts[6]) if len(p2_parts) > 6 else 0
        p1_active = p1_parts[p1_active_idx].split(",")[0]
        p2_active = p2_parts[p2_active_idx].split(",")[0]

        print(f"\n---- Turn {i + 1} ----")
        print(f"  {p1_active} vs {p2_active}")

        # s1 visit distribution
        s1_total = sum(v for _, v in s1_visits)
        s1_sorted = sorted(s1_visits, key=lambda x: x[1], reverse=True)
        print(f"  P1 -> {s1_move}")
        for name, visits in s1_sorted[:4]:
            pct = visits * 100 // max(s1_total, 1)
            bar = "#" * (pct // 4)
            print(f"    {name:20s} {pct:3d}% {bar}")

        # s2 visit distribution
        if s2_visits:
            s2_total = sum(v for _, v in s2_visits)
            s2_sorted = sorted(s2_visits, key=lambda x: x[1], reverse=True)
            print(f"  P2 -> {s2_move}")
            for name, visits in s2_sorted[:4]:
                pct = visits * 100 // max(s2_total, 1)
                bar = "#" * (pct // 4)
                print(f"    {name:20s} {pct:3d}% {bar}")
        else:
            print(f"  P2 -> {s2_move}")

    # show final state (last turn's state + one more turn of resolution)
    print(f"\n{'=' * 60}")
    print(f"Result: {winner} in {len(turns)} turns")


def main():
    parser = argparse.ArgumentParser(description="Watch recorded games")
    parser.add_argument("--team", type=int, default=13)
    parser.add_argument("--opponent", type=int, default=0)
    parser.add_argument("--s1-ms", type=int, default=500)
    parser.add_argument("--s2-ms", type=int, default=500)
    parser.add_argument("--max-turns", type=int, default=100)
    parser.add_argument("--games", type=int, default=2)
    args = parser.parse_args()

    t1 = TEAM_NAMES[args.team] if args.team < len(TEAM_NAMES) else f"Team{args.team}"
    t2 = TEAM_NAMES[args.opponent] if args.opponent < len(TEAM_NAMES) else f"Team{args.opponent}"

    print(f"{t1} ({args.s1_ms}ms) vs {t2} ({args.s2_ms}ms), {args.games} game(s)")

    state = build_pe_state(SAMPLE_TEAMS[args.team], SAMPLE_TEAMS[args.opponent])
    results = pe.run_games_recorded(
        state, n_games=args.games,
        s1_search_ms=args.s1_ms, s2_search_ms=args.s2_ms,
        max_turns=args.max_turns,
    )

    for i, r in enumerate(results):
        show_game(r, i + 1)


if __name__ == "__main__":
    main()
