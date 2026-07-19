#!/usr/bin/env python3
"""Calibration report over the desk-read ledger (showdown/desk_reads.jsonl).

Every searched decision logs the top line's avg MCTS score as a win
probability; every game resolves to 1/0 (ties 0.5). Brier = mean squared
error between the two — 0 is a prophet, 0.25 is always-saying-50/50, and
confident-and-wrong is punished hardest. Reads within a game share one
outcome, so they are correlated samples: fine for the aggregate score and
the calibration table, just don't treat the read count as independent n.

Bands mirror gen9_player._read_phrase so each row of the calibration table
is literally "when the desk says X, how often do we actually win".

Usage:  python showdown/brier_report.py [--log PATH] [--per-persona]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

DEFAULT_LOG = Path(__file__).parent / "desk_reads.jsonl"

# (floor, label) — mirrors gen9_player._read_phrase's bands, high to low
BANDS = [
    (0.85, "all but sealed"),
    (0.70, "clearly ahead"),
    (0.58, "real edge"),
    (0.45, "dead even"),
    (0.32, "behind"),
    (0.15, "deep trouble"),
    (0.00, "nearly gone"),
]

TURN_BUCKETS = [(0, 9, "opening (T1-9)"), (10, 25, "midgame (T10-25)"),
                (26, 10**9, "lategame (T26+)")]


def band_label(v: float) -> str:
    for floor, label in BANDS:
        if v >= floor:
            return label
    return BANDS[-1][1]


def load(path: Path) -> list[dict]:
    games = []
    with open(path) as f:
        for ln in f:
            ln = ln.strip()
            if not ln:
                continue
            try:
                games.append(json.loads(ln))
            except json.JSONDecodeError:
                continue
    return games


def brier(pairs: list[tuple[float, float]]) -> float:
    return sum((p - o) ** 2 for p, o in pairs) / len(pairs)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    ap.add_argument("--log", type=Path, default=DEFAULT_LOG)
    args = ap.parse_args()

    if not args.log.exists():
        raise SystemExit(f"no ledger at {args.log} — play a game first")
    games = load(args.log)
    if not games:
        raise SystemExit(f"{args.log} holds no parseable games")

    w = sum(1 for g in games if g["result"] == "win")
    l = sum(1 for g in games if g["result"] == "loss")
    t = len(games) - w - l
    pairs = [(read, g["outcome"])
             for g in games for _, read in g["reads"]]
    turns = [(turn, read, g["outcome"])
             for g in games for turn, read in g["reads"]]

    print(f"Desk-read calibration — {len(games)} games "
          f"({w}W-{l}L-{t}T), {len(pairs)} reads")
    overall = brier(pairs)
    base_rate = (w + 0.5 * t) / len(games)
    base_brier = base_rate * (1 - base_rate)  # constant base-rate predictor
    print(f"Overall Brier: {overall:.4f}   "
          f"(always-0.5: 0.2500, base-rate {base_rate:.2f}: {base_brier:.4f})")
    print()

    print(f"{'desk says':<16}{'reads':>7}{'mean pred':>11}{'actual win':>12}")
    for floor, label in BANDS:
        rows = [(p, o) for p, o in pairs if band_label(p) == label]
        if not rows:
            continue
        mean_p = sum(p for p, _ in rows) / len(rows)
        emp = sum(o for _, o in rows) / len(rows)
        print(f"{label:<16}{len(rows):>7}{mean_p:>11.3f}{emp:>12.3f}")
    print()

    print(f"{'phase':<18}{'reads':>7}{'Brier':>9}")
    for lo, hi, label in TURN_BUCKETS:
        rows = [(p, o) for turn, p, o in turns if lo <= turn <= hi]
        if not rows:
            continue
        print(f"{label:<18}{len(rows):>7}{brier(rows):>9.4f}")


if __name__ == "__main__":
    main()
