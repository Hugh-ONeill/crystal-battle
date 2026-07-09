# Replay-driven validation for the gen9 monotype translator.
#
# Streams scraped gen9monotype replay logs (showdown/replays/gen9monotype/)
# through poke-env's protocol parser from the p1 perspective, translating the
# battle at every turn boundary. This exercises the full protocol surface --
# formes, items, abilities, volatiles, weather -- far beyond what synthetic
# unit tests cover.
#
# Replays are spectator logs (no |request| messages), so own-side stats fall
# back to the neutral-85 estimates; that's fine, the target here is coverage:
# no translate() exceptions, and (optionally) every translated state accepted
# by MCTS.
#
# Usage:
#   .venv/bin/python showdown/validate_gen9_translator.py --n-replays 200
#   .venv/bin/python showdown/validate_gen9_translator.py --n-replays 50 --search-ms 20

from __future__ import annotations

import argparse
import json
import logging
import sys
import traceback
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import poke_engine as pe
from poke_env.battle.battle import Battle

from showdown.gen9_translator import Gen9Translator

REPLAYS_DIR = Path(__file__).parent / "replays" / "gen9monotype"


def validate_replay(path: Path, search_ms: int = 0) -> dict:
    """Replay one log; return counters of parse/translate/search failures."""
    data = json.loads(path.read_text())
    log_lines = data["log"].split("\n")

    logger = logging.getLogger("validate")
    logger.setLevel(logging.CRITICAL)  # spectator logs trip player-name warnings
    battle = Battle(f"battle-{data['id']}", "p1-player", logger, gen=9)
    battle._player_role = "p1"

    translator = Gen9Translator()
    stats = Counter()
    errors: list[tuple[str, str]] = []

    for line in log_lines:
        if not line.startswith("|"):
            continue
        event = line.split("|")
        if len(event) < 2 or not event[1]:
            continue
        # spectator-log lines poke-env's player-perspective parser doesn't model
        if event[1] in ("t:", "win", "raw", "j", "l", "player"):
            continue
        try:
            battle.parse_message(event)
        except Exception as e:
            stats[f"parse_fail:{event[1]}"] += 1
            continue

        if event[1] == "turn":
            stats["turns"] += 1
            try:
                state = translator.translate(battle)
            except Exception:
                stats["translate_fail"] += 1
                errors.append((f"{path.name} turn {event[2]}",
                               traceback.format_exc(limit=3)))
                continue
            if search_ms > 0:
                try:
                    result = pe.monte_carlo_tree_search(state, search_ms)
                    if not result.side_one:
                        stats["search_empty"] += 1
                except BaseException as e:  # pyo3 panics subclass BaseException
                    if isinstance(e, KeyboardInterrupt):
                        raise
                    stats["search_fail"] += 1
                    errors.append((f"{path.name} turn {event[2]} (search)",
                                   traceback.format_exc(limit=3)))

    return {"stats": stats, "errors": errors}


def main():
    parser = argparse.ArgumentParser(description="Validate gen9 translator vs real replays")
    parser.add_argument("--n-replays", type=int, default=100)
    parser.add_argument("--search-ms", type=int, default=0,
                        help="if >0, also run MCTS on every translated state")
    parser.add_argument("--show-errors", type=int, default=5,
                        help="max distinct tracebacks to print")
    args = parser.parse_args()

    files = sorted(REPLAYS_DIR.glob("gen9monotype-*.json"))[: args.n_replays]
    print(f"=== validating translator on {len(files)} replays "
          f"(search_ms={args.search_ms}) ===")

    totals = Counter()
    all_errors: list[tuple[str, str]] = []
    for i, path in enumerate(files):
        try:
            result = validate_replay(path, search_ms=args.search_ms)
        except Exception:
            totals["replay_fail"] += 1
            all_errors.append((path.name, traceback.format_exc(limit=3)))
            continue
        totals.update(result["stats"])
        all_errors.extend(result["errors"])
        if (i + 1) % 50 == 0:
            print(f"  [{i + 1}/{len(files)}] turns={totals['turns']} "
                  f"translate_fail={totals['translate_fail']} "
                  f"search_fail={totals['search_fail']}")

    print("\n=== totals ===")
    for key, count in sorted(totals.items()):
        print(f"  {key:30s} {count}")

    turns = totals["turns"] or 1
    bad = totals["translate_fail"] + totals["search_fail"]
    print(f"\n  {turns - bad}/{turns} turns translated cleanly "
          f"({100 * (turns - bad) / turns:.2f}%)")

    if all_errors:
        seen: set[str] = set()
        shown = 0
        print(f"\n=== sample errors (first {args.show_errors} distinct) ===")
        for where, tb in all_errors:
            key = tb.strip().split("\n")[-1]
            if key in seen:
                continue
            seen.add(key)
            print(f"\n--- {where} ---\n{tb}")
            shown += 1
            if shown >= args.show_errors:
                break


if __name__ == "__main__":
    main()
