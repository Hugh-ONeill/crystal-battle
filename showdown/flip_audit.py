#!/usr/bin/env python3
"""Audit the shadow overlay's hypothetical flips against a deep-search oracle.

For every consult where the LLM's world reweighting would have changed the
move (flips at --lam), re-search the position's stored world-0 state at
oracle depth and score which side deep search takes:

  ENGINE  oracle top move == the move actually played
  LLM     oracle top move == the reweighted merge's move
  NEITHER oracle prefers a third option

FRAMING (do not over-read): the oracle runs the SAME eval and world the
engine used, so it measures whether the engine's choice was depth-robust —
it is biased toward the engine. An oracle-ENDORSED flip is therefore strong
evidence the LLM caught something real; an oracle-rejected flip is only weak
evidence against the LLM (shared-eval blindness stays shared at depth). The
K-world caveat also applies: the live choice merged K worlds, the oracle
sees world 0 alone.

  .venv/bin/python showdown/flip_audit.py \
      --shadow showdown/overlay_shadow.jsonl \
      --pool showdown/bench/ladder_states.jsonl --ms 2500
"""

from __future__ import annotations

import argparse
import json
from concurrent.futures import ThreadPoolExecutor


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shadow", default="showdown/overlay_shadow.jsonl")
    ap.add_argument("--pool", default="showdown/bench/ladder_states.jsonl")
    ap.add_argument("--lam", default="1.0")
    ap.add_argument("--ms", type=int, default=2500)
    ap.add_argument("--threads", type=int, default=4)
    args = ap.parse_args()

    index = {}
    for line in open(args.pool):
        try:
            r = json.loads(line)
            index[(r["tag"], r["turn"])] = r
        except (json.JSONDecodeError, KeyError):
            continue

    flips = []
    for line in open(args.shadow):
        try:
            rec = json.loads(line)
        except json.JSONDecodeError:
            continue
        f = (rec.get("flips") or {}).get(args.lam) or {}
        if not f.get("flip"):
            continue
        st = index.get((rec["tag"], rec["turn"]))
        flips.append((rec, f["top"], st))
    print(f"{len(flips)} flips at lam={args.lam}; "
          f"{sum(1 for *_, s in flips if s)} with a stored state")

    import poke_engine as pe

    def oracle(item):
        rec, flip_to, st = item
        if st is None:
            return None
        try:
            res = pe.monte_carlo_tree_search(
                pe.State.from_string(st["state"]), args.ms)
            side = sorted(res.side_one, key=lambda r: -r.visits)
            return side[0].move_choice if side else None
        except Exception:
            return None

    with ThreadPoolExecutor(max_workers=args.threads) as ex:
        oracles = list(ex.map(oracle, flips))

    counts = {"ENGINE": 0, "LLM": 0, "NEITHER": 0, "no-state": 0}
    print(f"\n  {'turn':>4s} {'gates':24s} {'engine -> llm':32s} "
          f"{'margin':>6s} {'oracle':8s} weights")
    for (rec, flip_to, st), otop in zip(flips, oracles):
        if otop is None:
            counts["no-state"] += 1
            v = "?"
        elif otop == rec["engine_choice"]:
            counts["ENGINE"] += 1
            v = "ENGINE"
        elif otop == flip_to:
            counts["LLM"] += 1
            v = "LLM"
        else:
            counts["NEITHER"] += 1
            v = f"3rd:{otop[:10]}"
        gates = "+".join(sorted(set(x.split(":")[0] for x in rec["reasons"])))
        ww = (rec.get("flips") or {}).get("llm_weights")
        print(f"  {rec['turn']:4d} {gates[:24]:24s} "
              f"{(rec['engine_choice'] + ' -> ' + flip_to)[:32]:32s} "
              f"{rec.get('engine_margin', 0):6.2f} {v:8s} {ww}")
        worry = (rec.get("llm") or {}).get("worry")
        if worry:
            print(f"       worry: {worry[:110]}")
    n = counts["ENGINE"] + counts["LLM"] + counts["NEITHER"]
    print(f"\n  oracle verdicts over {n}: ENGINE {counts['ENGINE']}, "
          f"LLM {counts['LLM']}, NEITHER {counts['NEITHER']} "
          f"(no-state {counts['no-state']})")
    print("  Endorsed flips are the strong signal (deep search sides with "
          "the LLM against\n  the live choice). Rejections are weak evidence "
          "— the oracle shares the\n  engine's eval and world.")


if __name__ == "__main__":
    main()
