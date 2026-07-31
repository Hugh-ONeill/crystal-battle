#!/usr/bin/env python3
"""How OFF-META is each opponent's pool, and does off-meta-ness beat us?

Hypothesis (user, 2026-07-31): richwoman's edge could be a curated
just-off-meta team pool — sets deviating from chaos priors / Smogon dex just
enough that engines leaning on those priors mismodel her, while the teams
stay strong. Specimen already in hand: her Ursaluna is a Bulk Up variant
with ZERO Facade observed against a meta that is 91% Flame Orb Guts.

Two measurements, per opponent, from finished ladder logs:

  TYPICALITY   For every atom the opponent revealed, the probability the
               field priors assigned it: the chaos move marginal (usage/25%
               per slot, the belief_calibration normalization), plus dex
               coverage — whether ANY curated PS candidate for that species
               carries the move. Low typicality = off-meta pool. Comparing
               ACROSS opponents separates "she is curated-deviant" from
               "every bot is a bit weird".

  CONVERSION   Within one opponent, our per-game winrate split by that
               game's typicality tercile. If we lose MORE against the weird
               teams, the confusion mechanism is real and belief-side levers
               against her specifically are live again (the archive nulls
               were measured on the bench vs local fp, never against her
               pool). If weird and normal teams beat us alike, her edge is
               piloting/attrition, not our priors.

  .venv/bin/python showdown/meta_typicality.py
  .venv/bin/python showdown/meta_typicality.py --opponents richwoman LLM-gem3f
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
from pathlib import Path

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))
from belief_accuracy import parse_games, norm


def wilson(k: int, n: int) -> tuple[float, float]:
    if n == 0:
        return (0.0, 1.0)
    z, p = 1.96, k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (max(0.0, c - h), min(1.0, c + h))


def chaos_marginals():
    d = json.loads((HERE / "gen9ou_chaos.json").read_text())
    data = d.get("data", d)
    out = {}
    for k, v in data.items():
        mv = {norm(m): c for m, c in (v.get("Moves") or {}).items() if m}
        tot = sum(mv.values()) / 4 or 1
        out[norm(k)] = {m: min(1.0, c / tot) for m, c in mv.items()}
    return out


def ps_move_pool():
    """species -> set of moves appearing in ANY curated PS candidate."""
    try:
        from ps_sets import get_index
        idx = get_index("gen9ou")
        pool = {}
        for sp, cands in (idx.candidates or {}).items():
            pool[norm(sp)] = {norm(m) for c in cands
                              for slot in c["moves"]
                              for m in (slot if isinstance(slot, list) else [slot])}
        return pool
    except Exception:
        return {}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--opponents", nargs="*", default=None)
    args = ap.parse_args()
    opps = args.opponents
    if not opps:
        book = json.loads((HERE / "scouting_book.json").read_text())
        opps = sorted(book.keys())

    chaos = chaos_marginals()
    dex = ps_move_pool()

    print(f"  {'opponent':20s} {'games':>5s} {'atoms':>6s} {'typicality':>10s} "
          f"{'p<=5%':>6s} {'dex-miss':>8s}   winrate by typicality tercile "
          f"(low|mid|high)")
    for opp in opps:
        games = parse_games(opp)
        per_game = []
        for g in games:
            ps, dexmiss, natoms = [], 0, 0
            for sp, rv in g["rev"].items():
                n = norm(sp)
                for m in rv["moves"]:
                    natoms += 1
                    ps.append(chaos.get(n, {}).get(m, 0.0))
                    if dex and n in dex and m not in dex[n]:
                        dexmiss += 1
            if natoms:
                per_game.append(dict(
                    typ=sum(ps) / len(ps),
                    blind=sum(1 for p in ps if p <= 0.05) / len(ps),
                    dexmiss=dexmiss / natoms, atoms=natoms,
                    won=g.get("won")))
        if not per_game:
            print(f"  {opp:20s} {'0':>5s}")
            continue
        atoms = sum(r["atoms"] for r in per_game)
        typ = sum(r["typ"] * r["atoms"] for r in per_game) / atoms
        blind = sum(r["blind"] * r["atoms"] for r in per_game) / atoms
        dm = sum(r["dexmiss"] * r["atoms"] for r in per_game) / atoms

        decided = [r for r in per_game if r["won"] is not None]
        decided.sort(key=lambda r: r["typ"])
        terc = ""
        if len(decided) >= 9:
            k = len(decided) // 3
            groups = [decided[:k], decided[k:2 * k], decided[2 * k:]]
            bits = []
            for grp in groups:
                w = sum(1 for r in grp if r["won"])
                lo, hi = wilson(w, len(grp))
                bits.append(f"{100*w/len(grp):3.0f}% [{100*lo:.0f},{100*hi:.0f}] n={len(grp)}")
            terc = " | ".join(bits)
        print(f"  {opp:20s} {len(per_game):5d} {atoms:6d} {typ:10.3f} "
              f"{100*blind:5.1f}% {100*dm:7.1f}%   {terc}")
    print("\n  typicality = mean chaos-marginal of the opponent's revealed "
          "moves (1.0 = pure meta).\n  p<=5% = share of revealed moves the "
          "chaos prior all but ruled out.\n  dex-miss = revealed moves absent "
          "from EVERY curated PS candidate for that species.\n  Tercile "
          "winrates: OUR winrate when their team was least/mid/most typical.")


if __name__ == "__main__":
    main()
