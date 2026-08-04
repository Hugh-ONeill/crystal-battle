#!/usr/bin/env python3
"""Preview-time belief accuracy scored against GROUND-TRUTH team files.

The missing cell of the opponent-modeling comparison (2026-08-04). fp's
preview accuracy about OUR suite sets is measured: 57.6% move / 41.0% item
/ 25.5% tera, zero reveals (2026-07-23, truth-known mirror instrumentation).
The reverse direction — what OUR deployed pipeline believes about fp's
suite teams at preview — was never scored, because belief_accuracy.py
scores ladder games against partial reveals. On the bench both sides play
from known team FILES, so the truth is total: every mon's 4 moves, item
and tera. Same basis as fp's number (truth-known, zero reveals).

Adds a tier the component scorer cannot express: TRANSLATOR = the real
`Gen9Translator._opp_set` stack exactly as the bench deploys it (curated-PS
preferred, chaos fallback, archive off, no book) — scored both as its modal
set (rng off; what world-0 believes) and as sampled draws (what a sampled
world actually gets, the MEAN the search experiences).

Usage:
  .venv/bin/python showdown/belief_accuracy_truth.py [truth_dir]
  (default truth_dir: showdown/teams/suite_v1)
"""

from __future__ import annotations

import random
import sys
from pathlib import Path

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE.parent))

from showdown.belief_accuracy import (ArchiveTier, ChaosTier, PSTier, norm,
                                      score)


def parse_paste(path: Path) -> dict:
    """species -> (moves set, item, tera), the scorer's rev shape."""
    out, cur, moves, item, tera = {}, None, set(), None, None
    for line in path.read_text(errors="replace").split("\n"):
        t = line.strip()
        if not t:
            if cur:
                out[norm(cur)] = {"moves": moves, "item": item, "tera": tera}
            cur, moves, item, tera = None, set(), None, None
            continue
        if cur is None:
            cur = t.split("@")[0].split("(")[0].strip()
            item = norm(t.split("@")[1]) if "@" in t else None
        elif t.startswith("- "):
            moves.add(norm(t[2:]))
        elif t.lower().startswith("tera type:"):
            tera = norm(t.split(":", 1)[1])
    if cur:
        out[norm(cur)] = {"moves": moves, "item": item, "tera": tera}
    return out


def truth_games(truth_dir: Path) -> list[dict]:
    games = []
    for f in sorted(truth_dir.glob("*.txt")):
        rev = parse_paste(f)
        if len(rev) == 6:
            games.append({"preview": list(rev), "rev": rev, "team": f.stem})
    return games


class TranslatorTier:
    """The deployed set pipeline itself, zero reveals, per species."""

    def __init__(self, samples: int = 0):
        from showdown.gen9_translator import Gen9Translator
        self.tr = Gen9Translator(set_source="gen9ou")
        self.samples = samples
        self.name = f"translator~{samples}" if samples else "translator@mode"

    def _one(self, sp: str):
        s = self.tr._opp_set(sp)
        if not s:
            return None
        return ({norm(m) for m in (s.get("moves") or [])},
                norm(s.get("item")) or None,
                norm(s.get("tera_type")) or None, 1.0)

    def candidates(self, preview):
        per = {}
        for sp in preview:
            outs = []
            if self.samples:
                for i in range(self.samples):
                    self.tr._rng = random.Random(1000 + i)
                    o = self._one(sp)
                    if o:
                        outs.append(o)
            else:
                self.tr._rng = None
                o = self._one(sp)
                if o:
                    outs.append(o)
            if outs:
                per[norm(sp)] = outs
        return [(per, None)] if per else []


def main():
    truth_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else \
        HERE / "teams" / "suite_v1"
    games = truth_games(truth_dir)
    n_mons = sum(len(g["rev"]) for g in games)
    print(f"{len(games)} ground-truth teams from {truth_dir.name} "
          f"({n_mons} mons; every move/item/tera known)\n")
    if not games:
        return

    tiers = [TranslatorTier(0), TranslatorTier(20), PSTier(), ChaosTier()]
    try:
        tiers.append(ArchiveTier())      # index is gitignored; optional
    except Exception:
        pass
    print(f"  {'tier':16s} {'teams':>5s} | {'move mean':>10s} {'move BEST':>10s} "
          f"| {'item mean':>10s} {'item BEST':>10s} | {'tera mean':>10s} "
          f"{'tera BEST':>10s}")
    for t in tiers:
        r = score(t, games)
        print(f"  {t.name:16s} {r['games']:5d} | {r['move']:9.1f}% "
              f"{r['move_best']:9.1f}% | {r['item']:9.1f}% {r['item_best']:9.1f}% "
              f"| {r['tera']:9.1f}% {r['tera_best']:9.1f}%")
    print("\n  reference (fp about OUR suite sets at preview, 2026-07-23): "
          "move 57.6% / item 41.0% / tera 25.5%")
    print("  move mean = fraction of the true 4 moves a drawn candidate "
          "recovers; BEST = ceiling over candidates.")


if __name__ == "__main__":
    main()
