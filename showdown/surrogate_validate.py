#!/usr/bin/env python3
"""Validate surrogate endpoints against the historical A/B library.

The test that matters is not in-sample correlation with the win label — it
is: known nulls must stay null (no false levers) and known effects must
resolve at smaller n (subsample power curves). See surrogate_extract.py for
the record format and TODO NEXT UP for the design.

Findings of record (2026-08-09, first run):
  - TERMINAL MARGIN: cohen-d 0.269 vs winrate's 0.211 on the Ditto effect
    (~x1.6 efficiency: 80% power at n~200/arm vs ~340), null-safe on
    k1commit (p=0.72). ADOPTED as the screen endpoint alongside winrate.
  - Raw checkpoint differentials DILUTE late-landing levers (cp12 d=0.128
    on Ditto — turn-12 adjudication would have missed the +10.5pp): keep
    checkpoints as future model features, not standalone endpoints.
  - --ab pairing adds no variance reduction on margin (rho=+0.023): MCTS
    stochasticity decorrelates pairs from turn 1. Pairing keeps its
    confound-cancelling job only.

Usage:
  .venv/bin/python showdown/surrogate_validate.py \
      --effect dittoab3_0806 --arm-regex 'G\\d+_([a-z]+)\\d' --arms dt ct \
      --null k1commit --null k1commit2
"""

from __future__ import annotations

import argparse
import json
import math
import random
import re
import statistics as st
from collections import defaultdict
from pathlib import Path

ENDS = ("win", "margin", "hp_margin", "cp8", "cp12")


def load(surdir: Path, corpus: str):
    return [json.loads(l) for l in open(surdir / f"{corpus}.jsonl")]


def cp_feat(r, t):
    c = r["cp"].get(t)
    if not c:
        return None
    u, m = c["us"], c["them"]
    return (u["hp"] - m["hp"]) + 0.5 * (u["alive"] - m["alive"])


def endpoints(r):
    return {"win": 1.0 if r["winner_us"] else 0.0,
            "margin": float(r["margin"]),
            "hp_margin": r["hp_margin"],
            "cp8": cp_feat(r, "8"),
            "cp12": cp_feat(r, "12")}


def welch_p(xs, ys):
    nx, ny = len(xs), len(ys)
    se = math.sqrt(st.variance(xs) / nx + st.variance(ys) / ny)
    if se == 0:
        return 1.0
    return math.erfc(abs((st.mean(xs) - st.mean(ys)) / se) / math.sqrt(2))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sur-dir", type=Path,
                    default=Path("showdown/bench/surrogate"))
    ap.add_argument("--effect", help="corpus with a known real effect")
    ap.add_argument("--arm-regex", default=r"G\d+_([a-z]+)\d",
                    help="regex over `pairing` whose group 1 is the arm")
    ap.add_argument("--arms", nargs=2, default=["dt", "ct"],
                    help="the two arm labels to compare (treatment control)")
    ap.add_argument("--null", action="append", default=[],
                    help="--ab corpus with a known null (odd/even arms); "
                         "repeatable, pooled")
    ap.add_argument("--grid", default="100,200,400,700,1000")
    ap.add_argument("--reps", type=int, default=400)
    ap.add_argument("--seed", type=int, default=48151623)
    args = ap.parse_args()
    random.seed(args.seed)

    if args.effect:
        arms = defaultdict(list)
        rx = re.compile(args.arm_regex)
        for r in load(args.sur_dir, args.effect):
            if r["winner_us"] is None or r["tie"]:
                continue
            m = rx.match(r["pairing"])
            if m:
                arms[m.group(1)].append(endpoints(r))
        t_arm, c_arm = arms[args.arms[0]], arms[args.arms[1]]
        print(f"{args.effect} {args.arms[0]}(n={len(t_arm)}) vs "
              f"{args.arms[1]}(n={len(c_arm)}):")
        for e in ENDS:
            xs = [d[e] for d in t_arm if d[e] is not None]
            ys = [d[e] for d in c_arm if d[e] is not None]
            d_eff = (st.mean(xs) - st.mean(ys)) / math.sqrt(
                (st.variance(xs) + st.variance(ys)) / 2)
            print(f"  {e:10s} diff {st.mean(xs)-st.mean(ys):+.3f}  "
                  f"cohen-d {d_eff:+.3f}  p {welch_p(xs, ys):.2e}")
        grid = [int(n) for n in args.grid.split(",")]
        grid = [n for n in grid if n <= min(len(t_arm), len(c_arm))]
        print(f"\npower (frac of {args.reps} subsamples with p<0.05):")
        print(f"{'n/arm':>6} " + " ".join(f"{e:>9}" for e in ENDS))
        for n in grid:
            row = []
            for e in ENDS:
                hits = 0
                for _ in range(args.reps):
                    xs = [d[e] for d in random.sample(t_arm, n)
                          if d[e] is not None]
                    ys = [d[e] for d in random.sample(c_arm, n)
                          if d[e] is not None]
                    hits += welch_p(xs, ys) < 0.05
                row.append(hits / args.reps)
            print(f"{n:>6} " + " ".join(f"{v:>9.0%}" for v in row))

    if args.null:
        a, b = [], []
        for c in args.null:
            for r in load(args.sur_dir, c):
                if r["winner_us"] is None or r["tie"]:
                    continue
                (a if r["game"] % 2 else b).append(endpoints(r))
        print(f"\nnull {'+'.join(args.null)}: A(n={len(a)}) vs B(n={len(b)})"
              " — every p must be >0.05:")
        for e in ENDS:
            xs = [d[e] for d in a if d[e] is not None]
            ys = [d[e] for d in b if d[e] is not None]
            print(f"  {e:10s} diff {st.mean(xs)-st.mean(ys):+.3f}  "
                  f"p {welch_p(xs, ys):.3f}")


if __name__ == "__main__":
    main()
