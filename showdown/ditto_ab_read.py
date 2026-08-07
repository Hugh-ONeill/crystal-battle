#!/usr/bin/env python3
"""Preregistered read for the Ditto slot-value A/B
(showdown/bench/dittoab0806_REGISTER.txt).

Attribution is taken from foul-play's OWN log, which stamps every game with
`team: G<n>_<ourteam>_vs_<fpteam>` and then `Winner: <username>`. That is
authoritative and avoids inferring the team from which species happened to
appear — the inference route silently mislabels any ditto-arm game where
Ditto never switched in (~4% of them).

Primary estimand: POOLED ditto-arm minus control-arm winrate, with a 95% CI
on the difference (Newcombe/Wilson). Per-pair rows are secondary and only
resolve ~9pp at this n; they never override the pooled read.
"""

import glob
import math
import re
import sys
from collections import defaultdict

PAIRS = {"dt1": "ct1", "dt2": "ct2", "dt3": "ct3"}
ARMS = ("ct", "dt", "bb")


def wilson(k, n, z=1.96):
    if not n:
        return (0.0, 0.0, 0.0)
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * ((p * (1 - p) / n + z * z / (4 * n * n)) ** 0.5) / d
    return p, max(0.0, c - h), min(1.0, c + h)


def newcombe(k1, n1, k2, n2, z=1.96):
    """95% CI for p1 - p2 from the two Wilson intervals (Newcombe method 10):
    robust near 0/1 and at unequal n, unlike the naive Wald difference."""
    p1, l1, u1 = wilson(k1, n1, z)
    p2, l2, u2 = wilson(k2, n2, z)
    lo = (p1 - p2) - z * math.sqrt((p1 - l1) ** 2 + (u2 - p2) ** 2) / z
    hi = (p1 - p2) + z * math.sqrt((u1 - p1) ** 2 + (p2 - l2) ** 2) / z
    d1 = math.sqrt((p1 - l1) ** 2 + (u2 - p2) ** 2)
    d2 = math.sqrt((u1 - p1) ** 2 + (p2 - l2) ** 2)
    return p1 - p2, (p1 - p2) - d1, (p1 - p2) + d2


def read(paths):
    games = []
    for path in sorted(paths):
        team = None
        for line in open(path, errors="replace"):
            m = re.search(r"team: G\d+_([a-z0-9]+)_[^ ]*_vs_", line)
            if m and "=== lane" in line:
                team = m.group(1)
                continue
            m2 = re.search(r"Winner: (\S+)", line)
            if m2 and team:
                games.append((team, m2.group(1).startswith("CBGen9")))
                team = None
    return games


def main():
    pattern = sys.argv[1] if len(sys.argv) > 1 else \
        "showdown/bench/dittoab0806_L*_foulplay.log"
    games = read(glob.glob(pattern))
    tally = defaultdict(lambda: [0, 0])          # team -> [wins, n]
    for team, won in games:
        tally[team][1] += 1
        tally[team][0] += won
    print(f"decided games attributed: {len(games)}\n")

    dw = dn = cw = cn = 0
    print("PER PAIR (secondary — resolves only ~9pp at this n)")
    for d, c in PAIRS.items():
        kd, nd = tally[d]
        kc, nc = tally[c]
        dw, dn, cw, cn = dw + kd, dn + nd, cw + kc, cn + nc
        pd, ld, ud = wilson(kd, nd)
        pc, lc, uc = wilson(kc, nc)
        diff, lo, hi = newcombe(kd, nd, kc, nc)
        print(f"  {d}: {kd:3d}/{nd:3d} = {pd:5.1%} [{ld:.1%},{ud:.1%}]  vs  "
              f"{c}: {kc:3d}/{nc:3d} = {pc:5.1%} [{lc:.1%},{uc:.1%}]  "
              f"-> {diff:+.1%} [{lo:+.1%}, {hi:+.1%}]")

    # three-arm mode: if a bb arm is present, report all pairwise contrasts
    if any(k.startswith("bb") for k in tally):
        print("\nTHREE-ARM (pooled)")
        agg = {a: [sum(tally[f"{a}{i}"][0] for i in (1, 2, 3)),
                   sum(tally[f"{a}{i}"][1] for i in (1, 2, 3))] for a in ARMS}
        for a in ARMS:
            k, n = agg[a]
            p_, l_, u_ = wilson(k, n)
            label = {"ct": "control (original)", "dt": "ditto (Imposter)",
                     "bb": "band breaker (Dnite)"}[a]
            print(f"  {label:22s} {k:4d}/{n:4d} = {p_:5.1%} [{l_:.1%}, {u_:.1%}]")
        for a, b in (("dt", "ct"), ("bb", "ct"), ("dt", "bb")):
            d, lo_, hi_ = newcombe(agg[a][0], agg[a][1], agg[b][0], agg[b][1])
            verdict = "excludes 0" if (lo_ > 0 or hi_ < 0) else "SPANS 0"
            print(f"  {a} - {b}: {d:+.2%}  95% CI [{lo_:+.2%}, {hi_:+.2%}]  {verdict}")
        return

    pd, ld, ud = wilson(dw, dn)
    pc, lc, uc = wilson(cw, cn)
    diff, lo, hi = newcombe(dw, dn, cw, cn)
    print("\nPRIMARY (pooled)")
    print(f"  ditto arm   {dw}/{dn} = {pd:.1%} [{ld:.1%}, {ud:.1%}]")
    print(f"  control arm {cw}/{cn} = {pc:.1%} [{lc:.1%}, {uc:.1%}]")
    print(f"  DIFFERENCE  {diff:+.2%}  95% CI [{lo:+.2%}, {hi:+.2%}]")
    verdict = ("FAVOURS DITTO" if lo > 0 else
               "AGAINST DITTO" if hi < 0 else
               "SPANS ZERO — report as an upper bound, do not re-run")
    print(f"  verdict: {verdict}")


if __name__ == "__main__":
    main()
