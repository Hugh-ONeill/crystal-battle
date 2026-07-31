#!/usr/bin/env python3
"""Offline advocate sweep: is the weather-setter starvation an EXPLORATION
problem or an EVAL problem?

margin_screen (2026-07-31) showed the search declines the roles-demanded
setter switch at median 44.8pp confidence, and in 184/455 disagreements the
switch drew so few visits it never registered. Two rival explanations, each
with a different fix:

  EXPLORATION — the switch is actually good, the search just never gave its
  subtree real depth. Fix: the advocate world (overlay.py), no eval change.
  EVAL — even a deep, forced subtree scores the post-switch states poorly,
  because the field's value lies beyond the leaf horizon. Fix: the
  weather-mode eval term (TODO), and the advocate world alone cannot help.

This sweep replays every weather-down disagreement from a --dump-states pool
through OverlayShadow._advocate (0.75 our-side root prior on the demanded
switch, CB_ADVOCATE_MS deep search) and tallies overturns vs confirms.
Overturn = the demanded switch's deep Q beats the engine's chosen action's Q
in the same tree.

  .venv/bin/python showdown/advocate_study.py \
      --pool showdown/bench/marginpool_states.jsonl \
      --logs 'showdown/bench/marginpool_L*_ours.log'
"""

from __future__ import annotations

import argparse
import glob
import json
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent))
from roles_screen import games, opinions
from showdown.overlay import OverlayShadow


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pool", required=True)
    ap.add_argument("--logs", required=True)
    ap.add_argument("--limit", type=int, default=0,
                    help="cap the number of advocate searches (0 = all)")
    args = ap.parse_args()

    index = {}
    with open(args.pool) as f:
        for line in f:
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            index[(r["tag"], r["turn"])] = r

    o = OverlayShadow()
    cases = []
    for g in games(sorted(glob.glob(args.logs))):
        for op in opinions(g):
            if not op["rule"].startswith("our weather") or not op["disagreed"]:
                continue
            rec = index.get((g["tag"], op["turn"]))
            if rec:
                cases.append((op, rec))
    if args.limit:
        cases = cases[: args.limit]
    print(f"{len(cases)} weather-down disagreements with a pooled state")

    overturn, confirm, miss, err = [], [], 0, 0
    for i, (op, rec) in enumerate(cases):
        total = sum(v for _, v in rec["ranked"]) or 1
        share = {c: v / total for c, v in rec["ranked"]}
        # first demanded setter switch; margin_screen's no-alternative cases
        # simply have share 0 here and are exactly the ones worth forcing
        act = op["targets"][0]
        out = o._advocate(rec["state"], act, rec["choice"])
        if "error" in out:
            err += 1
            continue
        if out.get("skip"):
            miss += 1              # not a root option (trapped/fainted-late)
            continue
        dq = out.get("deep_q")
        eq = out.get("engine_q_same_tree")
        if dq is None or eq is None:
            err += 1
            continue
        row = dict(turn=op["turn"], action=act, prior_share=share.get(act, 0.0),
                   deep_q=dq, engine_q=eq, delta=round(dq - eq, 4))
        (overturn if dq > eq else confirm).append(row)
        if (i + 1) % 50 == 0:
            print(f"  {i+1}/{len(cases)}: overturn {len(overturn)} "
                  f"confirm {len(confirm)}", file=sys.stderr)

    n = len(overturn) + len(confirm)
    print(f"\nsearched {n}  (not-a-root-option {miss}, errors {err})")
    if not n:
        return
    print(f"  OVERTURNS (deep dive prefers the setter switch): "
          f"{len(overturn)} = {100*len(overturn)/n:.0f}%")
    print(f"  CONFIRMS  (engine choice still better, deeply): "
          f"{len(confirm)} = {100*len(confirm)/n:.0f}%")
    for name, rows in (("overturn", overturn), ("confirm", confirm)):
        if rows:
            ds = [r["delta"] for r in rows]
            print(f"  {name} deltaQ median {statistics.median(ds):+.3f} "
                  f"(p25 {sorted(ds)[len(ds)//4]:+.3f}, "
                  f"p75 {sorted(ds)[3*len(ds)//4]:+.3f}, n={len(ds)})")
    zero = [r for r in overturn + confirm if r["prior_share"] < 0.001]
    if zero:
        zo = sum(1 for r in zero if r["delta"] > 0)
        print(f"  previously-unregistered switches (share~0): {len(zero)}, "
          f"of which {zo} overturn")
    print("\n  Overturn-heavy => exploration problem: the advocate world is "
          "the fix.\n  Confirm-heavy => eval problem: the weather-mode term "
          "is the fix.")


if __name__ == "__main__":
    main()
