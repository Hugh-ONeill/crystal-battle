#!/usr/bin/env python3
"""Belief calibration: HOW CONFIDENTLY have the set beliefs been wrong?

belief_accuracy.py measures WHETHER the beliefs are right (recall of a drawn /
best candidate). This measures the CONFIDENCE attached to being right or
wrong, which is the quantity the campaign keeps paying for — the recorded
failure mode is never "knew nothing", it is "confidently wrong beats
calibrated vague" (six instances). Wilson intervals appear here only as error
bars on measured rates; the wrongness metrics are proper scoring rules.

PARTIAL CREDIT IS STRUCTURAL, not a fudge factor. Beliefs are scored as
MARGINALS over observable atoms — each move, the item, the tera — never as
set labels. A wrong set that still carried 3 of the 4 revealed moves priced
those 3 marginals high and correct; only the miss scores badly. "Inferred the
wrong set but got 75% of the moves" therefore counts mostly right by
construction, exactly proportionally to how right it was.

Three views, three different honesty constraints:

1. REVEALED-ATOM CONFIDENCE (log loss / Brier on what showed up). For every
   atom reality revealed, the probability the preview-time belief had
   assigned it. exp(-logloss) reads as "average probability we gave the
   truth". BLINDSIDED = revealed atoms we had priced <=5%: the inverse form
   of confidently wrong (reality did the thing we had dismissed). Caveat:
   revealed moves are usage-biased (clicked moves reveal), so this view
   over-samples common moves. It is a valid per-event proper score, not a
   calibration curve.

2. CALIBRATION TABLE, built ONLY from fully-observed events, where censoring
   cannot fake it: item categoricals (a reveal settles every candidate item
   at once), tera categoricals, and complete 4-move reveals (settles the
   whole move support). Bin by stated probability; compare to the empirical
   rate; Wilson 95% on each bin. CONFIDENT-WRONG RATE = P(wrong | claimed
   >= 80%), the direct number.

3. MODAL-SET VIEW — the question as asked. The single most probable
   candidate (deduped by content; its probability = its weight share) versus
   that game's reveals: how often is it exactly consistent, and when it is
   NOT, what fraction of the revealed atoms did it still get right?

Scored at TEAM PREVIEW against each game's own reveals, like
belief_accuracy: preview-time is prequentially clean (the belief has seen
nothing of the game it is scored on; book tiers stay leave-one-game-out).
In-game belief sharpening is a different trajectory and is not measured here.

ACCURACY IS NOT UTILITY, and neither is calibration: this diagnoses the
beliefs the overlay's world-reweight channel would consume (LLM_OVERLAY.md
gates that channel on beating these numbers offline), it does not ship
anything by itself.

Usage:
  .venv/bin/python showdown/belief_calibration.py --opponent richwoman
"""

from __future__ import annotations

import argparse
import math
import sys
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))
from belief_accuracy import (ArchiveTier, BookWeightedArchiveTier, PSTier,
                             norm, parse_games)

EPS = 0.01          # probability clip: a support miss costs -ln(.01) ~ 4.6
                    # nats instead of infinity; a claimed certainty likewise
BINS = [0.0, 0.2, 0.4, 0.6, 0.8, 0.9, 1.0001]


def clip(p: float) -> float:
    return min(1 - EPS, max(EPS, p))


def wilson(k: int, n: int) -> tuple[float, float]:
    if n == 0:
        return (0.0, 1.0)
    z, p = 1.96, k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (max(0.0, c - h), min(1.0, c + h))


class ChaosMarginalTier:
    """The field prior, scored as the marginals it actually is. Moves are
    normalized so a mon's move mass sums to 4 slots (same arithmetic as the
    roles prevalence work: usage / 25% per slot), item/tera to 1."""

    name = "chaos-marg"
    marginal = True

    def __init__(self):
        import json
        d = json.loads((HERE / "gen9ou_chaos.json").read_text())
        self.map = {norm(k): v for k, v in d.get("data", d).items()}

    def species_marginals(self, sp: str):
        e = self.map.get(sp)
        if not e:
            return None
        mv_raw = {norm(m): v for m, v in (e.get("Moves") or {}).items() if m}
        it_raw = {norm(i): v for i, v in (e.get("Items") or {}).items() if i}
        tr_raw = {norm(t): v for t, v in (e.get("Tera Types") or {}).items() if t}
        ms = sum(mv_raw.values()) / 4 or 1
        i_s = sum(it_raw.values()) or 1
        t_s = sum(tr_raw.values()) or 1
        return ({m: min(1.0, v / ms) for m, v in mv_raw.items()},
                {i: v / i_s for i, v in it_raw.items()},
                {t: v / t_s for t, v in tr_raw.items()})


def per_species_options(cands, n):
    """[(moves, item, tera, weight)] for species n, from either tier shape."""
    if not cands:
        return []
    if isinstance(cands[0][0], dict) and cands[0][1] is None:
        return cands[0][0].get(n) or []
    out = []
    for t, w in cands:
        s = t.get(n)
        if s:
            out.append((s[0], s[1], s[2], w or 1.0))
    return out


def marginals_and_modal(opts):
    wsum = sum(o[3] for o in opts) or 1.0
    mv, it, tr = defaultdict(float), defaultdict(float), defaultdict(float)
    dedup = defaultdict(float)
    for m_, i_, t_, w in opts:
        for m in m_:
            mv[m] += w
        if i_:
            it[i_] += w
        if t_:
            tr[t_] += w
        dedup[(frozenset(m_), i_, t_)] += w
    key, w = max(dedup.items(), key=lambda kv: kv[1])
    modal = (set(key[0]), key[1], key[2], w / wsum)
    return ({m: v / wsum for m, v in mv.items()},
            {i: v / wsum for i, v in it.items()},
            {t: v / wsum for t, v in tr.items()}, modal)


class Agg:
    def __init__(self):
        self.ll = self.br = 0.0
        self.n = self.blind5 = self.blind20 = 0

    def hit(self, p):        # outcome 1 with stated probability p
        c = clip(p)
        self.ll += -math.log(c)
        self.br += (1 - c) ** 2
        self.n += 1
        self.blind5 += p <= 0.05
        self.blind20 += p <= 0.20


def score_tier(tier, games):
    rev_mv, rev_it, rev_tr = Agg(), Agg(), Agg()
    cal = [[0, 0, 0.0] for _ in range(len(BINS) - 1)]   # [n, hits, sum_p]
    conf_wrong = {0.8: [0, 0], 0.9: [0, 0]}             # claims, wrong
    modal_rows = []                                     # (conf, exact, graded)
    covered = 0

    for gi, g in enumerate(games):
        if hasattr(tier, "hist"):                       # LOO book history
            h = defaultdict(lambda: {"moves": set(), "items": set(),
                                     "teras": set()})
            for j, o in enumerate(games):
                if j == gi:
                    continue
                for sp, rv in o["rev"].items():
                    e = h[norm(sp)]
                    e["moves"] |= rv["moves"]
                    if rv["item"]:
                        e["items"].add(rv["item"])
                    if rv["tera"]:
                        e["teras"].add(rv["tera"])
            tier.hist = h
        cands = None if getattr(tier, "marginal", False) \
            else tier.candidates(g["preview"])
        if cands is not None and not cands:
            continue
        covered += 1

        def observe(p, hit):
            b = next(i for i in range(len(BINS) - 1)
                     if BINS[i] <= p < BINS[i + 1])
            cal[b][0] += 1
            cal[b][1] += hit
            cal[b][2] += p
            for thr, cw in conf_wrong.items():
                if p >= thr:
                    cw[0] += 1
                    cw[1] += not hit

        for sp, rv in g["rev"].items():
            n = norm(sp)
            if getattr(tier, "marginal", False):
                got = tier.species_marginals(n)
                if not got:
                    continue
                mv, it, tr = got
                modal = None
            else:
                opts = per_species_options(cands, n)
                if not opts:
                    continue
                mv, it, tr, modal = marginals_and_modal(opts)

            for m in rv["moves"]:                      # view 1: what showed up
                rev_mv.hit(mv.get(m, 0.0))
            if rv["item"]:
                rev_it.hit(it.get(rv["item"], 0.0))
            if rv["tera"]:
                rev_tr.hit(tr.get(rv["tera"], 0.0))

            if rv["item"]:                             # view 2: fully observed
                for i in set(it) | {rv["item"]}:
                    observe(it.get(i, 0.0), i == rv["item"])
            if rv["tera"]:
                for t in set(tr) | {rv["tera"]}:
                    observe(tr.get(t, 0.0), t == rv["tera"])
            if len(rv["moves"]) >= 4:
                for m in set(mv) | rv["moves"]:
                    observe(mv.get(m, 0.0), m in rv["moves"])

            if modal is not None and len(rv["moves"]) >= 2:   # view 3: as asked
                m_, i_, t_, conf = modal
                atoms = [(m in m_) for m in rv["moves"]]
                if rv["item"]:
                    atoms.append(i_ == rv["item"])
                if rv["tera"]:
                    atoms.append(t_ == rv["tera"])
                graded = sum(atoms) / len(atoms)
                modal_rows.append((conf, graded >= 1.0, graded))

    return dict(covered=covered, rev=dict(moves=rev_mv, item=rev_it,
                tera=rev_tr), cal=cal, cw=conf_wrong, modal=modal_rows)


def report(tier, r):
    print(f"\n=== {tier.name}  ({r['covered']} games)")
    print("  confidence on what reality revealed (partial credit inherent):")
    for ax, a in r["rev"].items():
        if not a.n:
            continue
        print(f"    {ax:5s} n={a.n:<5d} logloss {a.ll/a.n:5.2f} "
              f"(avg prob given the truth ~{100*math.exp(-a.ll/a.n):.0f}%)  "
              f"brier {a.br/a.n:.3f}  "
              f"blindsided p<=5%: {100*a.blind5/a.n:4.1f}%  "
              f"p<=20%: {100*a.blind20/a.n:4.1f}%")
    print("  calibration on fully-observed events "
          "(item/tera categoricals + complete 4-move reveals):")
    print(f"    {'claimed':>12s} {'n':>6s} {'avg claim':>9s} {'actual':>7s}"
          f"  [Wilson 95%]")
    for i, (n, hits, sump) in enumerate(r["cal"]):
        if not n:
            continue
        lo, hi = wilson(hits, n)
        lab = f"{100*BINS[i]:.0f}-{100*min(1,BINS[i+1]):.0f}%"
        flag = " OVER" if sump / n > hi else (" under" if sump / n < lo else "")
        print(f"    {lab:>12s} {n:6d} {100*sump/n:8.1f}% {100*hits/n:6.1f}%"
              f"  [{100*lo:.1f}, {100*hi:.1f}]{flag}")
    for thr, (claims, wrong) in sorted(r["cw"].items()):
        if claims:
            lo, hi = wilson(wrong, claims)
            print(f"    confident-wrong: claims >={100*thr:.0f}% were wrong "
                  f"{100*wrong/claims:.1f}% of the time "
                  f"(n={claims}, Wilson [{100*lo:.1f}, {100*hi:.1f}])")
    if r["modal"]:
        print("  modal-set view (top candidate vs reveals; >=2 revealed moves; "
              f"n={len(r['modal'])} mon-games):")
        print(f"    {'stated conf':>12s} {'n':>5s} {'exactly right':>13s} "
              f"{'partial credit when wrong':>26s}")
        for lo_b, hi_b in [(0, .25), (.25, .5), (.5, .75), (.75, 1.0001)]:
            rows = [x for x in r["modal"] if lo_b <= x[0] < hi_b]
            if not rows:
                continue
            wrong = [g for _, ex, g in rows if not ex]
            wr = (f"{100*sum(wrong)/len(wrong):5.1f}% of revealed atoms"
                  if wrong else "    —")
            print(f"    {f'{100*lo_b:.0f}-{100*min(1,hi_b):.0f}%':>12s} "
                  f"{len(rows):5d} {100*sum(x[1] for x in rows)/len(rows):12.1f}% "
                  f"{wr:>26s}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--opponent", default="richwoman")
    ap.add_argument("--max-cands", type=int, default=80)
    args = ap.parse_args()

    games = parse_games(args.opponent)
    print(f"{len(games)} games vs {args.opponent} with preview + reveals")
    if not games:
        return
    tiers = [ArchiveTier(args.max_cands),
             BookWeightedArchiveTier(args.max_cands),
             PSTier(), ChaosMarginalTier()]
    for t in tiers:
        report(t, score_tier(t, games))
    print("\n  Blindsided = reality revealed an atom we had priced <=5% — the"
          "\n  inverse confident-wrong. OVER/under flags a bin whose average"
          "\n  claim falls outside its own Wilson band. Preview-time beliefs,"
          "\n  LOO for book tiers. Calibration is diagnosis, not utility.")


if __name__ == "__main__":
    main()
