#!/usr/bin/env python3
"""Belief accuracy BY TURN: does our opponent model refine as fp's does?

fp's trajectory about our sets is measured: 57.6% (move, preview) climbing
to 81.1% late, because it re-draws worlds every turn filtered through
reveals. Our preview number is now measured too (62.8-64.1% about fp's
suite sets, 2026-08-04) and BEATS fp's start — so if we still lose the
modeling war, it must be the CLIMB, not the start. This scores the climb.

Instrument: a par_series run with `--dump-states` appends one record per
move decision — world-0 state (the search's belief), battle tag, turn.
The fp lane logs carry the join: each game's marker line names the fp team
file (truth), and the battle tag follows it. Scoring is per DECISION per
alive opponent mon: fraction of the true 4 moves present in the believed
set, exact item, exact tera. Reveals flowing into the state (a revealed
move, a Knocked-Off item, a fired tera) ARE the refinement being measured,
same construction as fp's number.

Caveats stated up front: fainted mons drop out (no identity in the engine
dummy), so late buckets score surviving mons only; decisions weight the
average, so long games contribute more; world-0 is one sample of the
belief, unbiased across games.

Usage:
  .venv/bin/python showdown/belief_refinement.py refine128 \
      [--truth showdown/teams/suite_v1]
"""

from __future__ import annotations

import argparse
import glob
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE.parent))

from showdown.belief_accuracy_truth import parse_paste
from showdown.belief_accuracy import norm

MARKER = re.compile(r"=== lane \d+ game \d+/\d+ team: G\d+_(.+)_vs_(.+?) "
                    r"\(\d\d:\d\d:\d\d\) ===")
TAG = re.compile(r"(battle-gen9ou-\d+)")

BUCKETS = [(1, 1), (2, 5), (6, 10), (11, 15), (16, 20), (21, 30), (31, 999)]


def tag_to_fp_team(name: str) -> dict[str, str]:
    """battle tag -> fp team stem, from the lane logs' marker/tag sequence."""
    out: dict[str, str] = {}
    for path in sorted(glob.glob(str(HERE / "bench" / f"{name}_L*_foulplay.log"))):
        cur = None
        for line in open(path, errors="replace"):
            m = MARKER.search(line)
            if m:
                cur = m.group(2)
                continue
            t = TAG.search(line)
            if t and cur and t.group(1) not in out:
                out[t.group(1)] = cur
    return out


def bucket(turn: int) -> str:
    for lo, hi in BUCKETS:
        if lo <= turn <= hi:
            return f"T{lo}" if lo == hi else f"T{lo}-{hi}" if hi < 999 else f"T{lo}+"
    return "T?"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("name")
    ap.add_argument("--truth", default=str(HERE / "teams" / "suite_v1"))
    args = ap.parse_args()

    import poke_engine as pe

    truth_dir = Path(args.truth)
    truths = {f.stem: parse_paste(f) for f in truth_dir.glob("*.txt")}
    tag_team = tag_to_fp_team(args.name)
    print(f"{len(tag_team)} battles joined to fp truth teams")

    acc = defaultdict(lambda: dict(mv=0.0, mvt=0, it=0.0, itt=0, tr=0.0,
                                   trt=0, recs=0))
    bad_lines = 0
    unmatched_tags = set()
    dump = HERE / "bench" / f"{args.name}_states.jsonl"
    for line in open(dump, errors="replace"):
        try:
            r = json.loads(line)
        except Exception:
            bad_lines += 1
            continue
        team = tag_team.get(r.get("tag"))
        if not team:
            unmatched_tags.add(r.get("tag"))
            continue
        truth = truths.get(team)
        if not truth:
            continue
        try:
            st = pe.State.from_string(r["state"])
        except Exception:
            bad_lines += 1
            continue
        b = bucket(int(r.get("turn") or 0))
        a = acc[b]
        a["recs"] += 1
        for p in st.side_two.pokemon:
            if getattr(p, "hp", 0) <= 0:
                continue
            tr = truth.get(norm(p.id))
            if not tr:
                continue
            believed = {norm(m.id) for m in p.moves} - {"", "none"}
            a["mv"] += len(believed & tr["moves"])
            a["mvt"] += len(tr["moves"])
            a["it"] += 1 if norm(p.item) == tr["item"] else 0
            a["itt"] += 1
            a["tr"] += 1 if norm(p.tera_type) == tr["tera"] else 0
            a["trt"] += 1
    if bad_lines:
        print(f"({bad_lines} unparseable dump lines skipped)")
    if unmatched_tags:
        print(f"({len(unmatched_tags)} battle tags had no truth join)")

    pct = lambda a, b: 100 * a / b if b else 0.0
    print(f"\n  {'turns':8s} {'decisions':>9s} | {'move':>7s} | {'item':>7s} "
          f"| {'tera':>7s}")
    order = [f"T{lo}" if lo == hi else f"T{lo}-{hi}" if hi < 999 else f"T{lo}+"
             for lo, hi in BUCKETS]
    for b in order:
        a = acc.get(b)
        if not a:
            continue
        print(f"  {b:8s} {a['recs']:9d} | {pct(a['mv'], a['mvt']):6.1f}% "
              f"| {pct(a['it'], a['itt']):6.1f}% | {pct(a['tr'], a['trt']):6.1f}%")
    print("\n  reference — fp about our sets: move 57.6% preview -> 81.1% late "
          "(tera 25.5 -> 42)")


if __name__ == "__main__":
    main()
