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

ACCRUAL: verdicts persist to --out keyed (tag, turn, lam); already-audited
flips are skipped (--redo re-audits) so repeated runs only pay for new
flips, and the summary tallies the WHOLE file — the lam-channel roadmap
gates on n >= 20 audited flips at confident margins (>= --min-margin).
States come from the flip's embedded w0_state (post-2026-08-01 records)
with the --pool dump as fallback for the older join-era flips; no-state and
oracle-error flips are NOT marked done, so they retry if states appear.

CONTROL (--control N): the oracle is a NOISY referee — measured 2026-08-01,
it reproduces the engine's confident choice on NON-flip turns only ~52-57%
(n=40 x 2 draws), landing on the merged runner-up 30%. Single verdicts are
therefore weak; the interpretable statistic is the AGGREGATE comparison:
oracle->flip-target rate on flip turns vs oracle->runner-up rate on control
turns (flip targets were the runner-up in 20/20 confident cases). First
reading: 60% vs 30%, Fisher one-sided p=0.026.

  .venv/bin/python showdown/flip_audit.py --ms 2500
  .venv/bin/python showdown/flip_audit.py --control 40   # baseline only
"""

from __future__ import annotations

import argparse
import json
import time
from concurrent.futures import ThreadPoolExecutor


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shadow", default="showdown/overlay_shadow.jsonl")
    ap.add_argument("--pool", default="showdown/bench/ladder_states.jsonl")
    ap.add_argument("--out", default="showdown/bench/flip_audit.jsonl")
    ap.add_argument("--lam", default="1.0")
    ap.add_argument("--ms", type=int, default=2500)
    ap.add_argument("--min-margin", type=float, default=0.10)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--redo", action="store_true")
    ap.add_argument("--control", type=int, default=0,
                    help="instead of auditing flips, oracle N random "
                         "confident NON-flip consults and report the "
                         "engine/runner-up/other baseline")
    args = ap.parse_args()

    done = {}
    try:
        for line in open(args.out):
            try:
                r = json.loads(line)
                done[(r["tag"], r["turn"], r["lam"])] = r
            except (json.JSONDecodeError, KeyError):
                continue
    except FileNotFoundError:
        pass

    index, ranked = {}, {}
    try:
        for line in open(args.pool):
            try:
                r = json.loads(line)
                index[(r["tag"], r["turn"])] = r["state"]
                ranked[(r["tag"], r["turn"])] = [c for c, _ in
                                                 r.get("ranked") or []]
            except (json.JSONDecodeError, KeyError):
                continue
    except FileNotFoundError:
        pass

    if args.control:
        import random
        items = []
        for line in open(args.shadow):
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            if ((rec.get("flips") or {}).get(args.lam) or {}).get("flip"):
                continue
            if (rec.get("engine_margin") or 0) < args.min_margin:
                continue
            key = (rec["tag"], rec["turn"])
            st = rec.get("w0_state") or index.get(key)
            if st and len(ranked.get(key) or []) > 1:
                items.append((rec["engine_choice"], ranked[key][1], st))
        random.seed(7)
        random.shuffle(items)
        items = items[:args.control]

        import poke_engine as pe
        def ctl(item):
            try:
                res = pe.monte_carlo_tree_search(
                    pe.State.from_string(item[2]), args.ms)
                side = sorted(res.side_one, key=lambda r: -r.visits)
                return side[0].move_choice if side else None
            except Exception:
                return None
        with ThreadPoolExecutor(max_workers=args.threads) as ex:
            tops = list(ex.map(ctl, items))
        eng = ru = other = 0
        for (choice, second, _), t in zip(items, tops):
            if t is None:
                continue
            if t == choice:
                eng += 1
            elif t == second:
                ru += 1
            else:
                other += 1
        n = eng + ru + other
        print(f"control baseline over {n} confident non-flip consults at "
              f"{args.ms}ms:\n  oracle -> engine {eng} ({eng / n:.0%}), "
              f"runner-up {ru} ({ru / n:.0%}), other {other}")
        print("  Compare the flip audit's LLM rate against the RUNNER-UP "
              "rate here, not 50%.")
        return

    flips, skipped = [], 0
    for line in open(args.shadow):
        try:
            rec = json.loads(line)
        except json.JSONDecodeError:
            continue
        f = (rec.get("flips") or {}).get(args.lam) or {}
        if not f.get("flip"):
            continue
        if not args.redo and (rec["tag"], rec["turn"], args.lam) in done:
            skipped += 1
            continue
        st = rec.get("w0_state") or index.get((rec["tag"], rec["turn"]))
        flips.append((rec, f["top"], st))
    print(f"{len(flips)} unaudited flips at lam={args.lam} "
          f"({skipped} already in {args.out}); "
          f"{sum(1 for *_, s in flips if s)} with a state")

    import poke_engine as pe

    def oracle(item):
        rec, flip_to, st = item
        if st is None:
            return None
        try:
            res = pe.monte_carlo_tree_search(pe.State.from_string(st), args.ms)
            side = sorted(res.side_one, key=lambda r: -r.visits)
            return side[0].move_choice if side else None
        except Exception:
            return None

    with ThreadPoolExecutor(max_workers=args.threads) as ex:
        oracles = list(ex.map(oracle, flips))

    fresh = {"ENGINE": 0, "LLM": 0, "NEITHER": 0, "no-state": 0}
    out = open(args.out, "a")
    print(f"\n  {'turn':>4s} {'gates':24s} {'engine -> llm':32s} "
          f"{'margin':>6s} {'oracle':8s} weights")
    for (rec, flip_to, st), otop in zip(flips, oracles):
        if otop is None:
            fresh["no-state"] += 1
            v = "?"
        else:
            if otop == rec["engine_choice"]:
                v = "ENGINE"
            elif otop == flip_to:
                v = "LLM"
            else:
                v = "NEITHER"
            fresh[v] += 1
            row = {"tag": rec["tag"], "turn": rec["turn"], "lam": args.lam,
                   "engine_choice": rec["engine_choice"], "flip_to": flip_to,
                   "oracle_top": otop, "verdict": v,
                   "margin": rec.get("engine_margin"),
                   "reasons": rec.get("reasons"), "ms": args.ms,
                   "ts": int(time.time())}
            out.write(json.dumps(row) + "\n")
            done[(rec["tag"], rec["turn"], args.lam)] = row
        gates = "+".join(sorted(set(x.split(":")[0] for x in rec["reasons"])))
        show = v if v != "NEITHER" else f"3rd:{otop[:10]}"
        print(f"  {rec['turn']:4d} {gates[:24]:24s} "
              f"{(rec['engine_choice'] + ' -> ' + flip_to)[:32]:32s} "
              f"{rec.get('engine_margin', 0):6.2f} {show:8s} "
              f"{(rec.get('flips') or {}).get('llm_weights')}")
        worry = (rec.get("llm") or {}).get("worry")
        if worry:
            print(f"       worry: {worry[:110]}")
    out.close()

    n = sum(fresh[k] for k in ("ENGINE", "LLM", "NEITHER"))
    print(f"\n  this run over {n}: ENGINE {fresh['ENGINE']}, "
          f"LLM {fresh['LLM']}, NEITHER {fresh['NEITHER']} "
          f"(no-state {fresh['no-state']})")

    rows = [r for r in done.values() if r["lam"] == args.lam]
    conf = [r for r in rows if (r.get("margin") or 0) >= args.min_margin]

    def tally(rs):
        c = {"ENGINE": 0, "LLM": 0, "NEITHER": 0}
        for r in rs:
            c[r["verdict"]] += 1
        return f"ENGINE {c['ENGINE']}, LLM {c['LLM']}, NEITHER {c['NEITHER']}"

    print(f"\n== CUMULATIVE ({args.out}, lam={args.lam}) ==")
    print(f"  all {len(rows)}: {tally(rows)}")
    print(f"  confident (margin >= {args.min_margin:.2f}) "
          f"{len(conf)}/20 toward the lam-channel gate: {tally(conf)}")
    print("  Endorsed flips are the strong signal (deep search sides with "
          "the LLM against\n  the live choice). Rejections are weak evidence "
          "— the oracle shares the\n  engine's eval and world.")


if __name__ == "__main__":
    main()
