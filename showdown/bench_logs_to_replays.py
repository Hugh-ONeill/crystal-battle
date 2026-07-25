#!/usr/bin/env python3
"""Convert par_series bench worker logs into replay-shaped JSONL.

Every `*_ours.log` contains the full Showdown protocol stream of each game
the lane played (poke-env echoes received chunks after `<<<`), delimited by
the `=== lane L game G/TOT team: ... ===` headers par_series writes. Re-join
the protocol lines per game and emit metamon-shaped records
(`{id, formatid, log}`) that replay_to_policy_gen9 --jsonl consumes — which
turns months of foul-play sparring into an fp-specific opponent-policy
corpus (~1.1k fp decisions per lane-log) with zero new games played.

Player-view streams carry |request| lines replays lack; the replay parser
ignores unknown line types, and the public |move|/|switch|/|win| lines it
keys on are identical.
"""

from __future__ import annotations

import argparse
import glob
import json
import re

GAME_HEADER = re.compile(r"=== lane \d+ game (\d+)/\d+ team: (\S+) ")
CHUNK = re.compile(r"<<<\s(.*)$")


def games_from_log(path: str):
    """Yield (game_index, team, [protocol lines]) per game in one log."""
    game = team = None
    lines: list[str] = []
    for raw in open(path, errors="replace"):
        m = GAME_HEADER.search(raw)
        if m:
            if game is not None and lines:
                yield game, team, lines
            game, team = int(m.group(1)), m.group(2)
            lines = []
            continue
        if game is None:
            continue
        c = CHUNK.search(raw)
        if c:
            lines.append(c.group(1))
        elif raw.startswith("|"):
            lines.append(raw.rstrip("\n"))
    if game is not None and lines:
        yield game, team, lines


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--logs", required=True,
                    help="glob of worker logs, e.g. 'showdown/bench/*_ours.log'")
    ap.add_argument("--out", required=True)
    ap.add_argument("--min-lines", type=int, default=50,
                    help="Drop games with fewer protocol lines (aborted/kicked)")
    args = ap.parse_args()

    n_games = n_short = 0
    with open(args.out, "w") as out:
        for path in sorted(glob.glob(args.logs)):
            base = path.rsplit("/", 1)[-1].replace("_ours.log", "")
            for game, team, lines in games_from_log(path):
                if len(lines) < args.min_lines:
                    n_short += 1
                    continue
                rec = {"id": f"{base}:G{game}", "formatid": "gen9ou",
                       "team": team, "log": "\n".join(lines)}
                out.write(json.dumps(rec) + "\n")
                n_games += 1
    print(f"wrote {n_games} games to {args.out} ({n_short} short/aborted dropped)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
