#!/usr/bin/env python3
"""Per-phase recalibration map for the engine's position read.

WHY (2026-07-30): the desk-read ledger says the raw MCTS root value is a BADLY
calibrated win probability, and phase-dependently so — opening (T1-9) Brier
0.2496-0.2559 across four sessions, i.e. at or worse than always-saying-50/50,
while the lategame reaches 0.10. The calibration table is systematically
optimistic in the middle: positions the engine calls a "real edge" (~0.63)
convert at 0.45-0.57. Anything that REPORTS that number as a probability —
the commentary's position phrases above all — is therefore making confident
claims the data does not support, most of all early.

This fits a monotone map raw-value -> empirical win rate, separately per phase,
by isotonic regression (pool-adjacent-violators). Monotone matters: it can only
rescale confidence, never reorder positions, so a calibrated read still agrees
with the engine about which of two positions is better.

SCOPE — reporting only. This must never touch search or move choice: MCTS ranks
moves by RELATIVE value, a monotone rescale changes no ranking, and feeding a
squashed value back into the tree would corrupt the value scale for no gain.

Validation is BY GAME, not by read: reads within a game share one outcome, so a
read-level split leaks the label between folds and flatters the fit.

Usage:
  .venv/bin/python showdown/read_calibration.py --fit      # refit + report
  .venv/bin/python showdown/read_calibration.py --report   # validate only
"""

from __future__ import annotations

import argparse
import glob
import json
import re
from pathlib import Path

import numpy as np

HERE = Path(__file__).parent
MAP_PATH = HERE / "read_calibration.json"

# phase edges mirror brier_report.TURN_BUCKETS
PHASES = [(0, 9, "opening"), (10, 25, "midgame"), (26, 10**9, "lategame")]

# mirrors beat_director._read_phrase / brier_report.BANDS, high to low
BANDS = [
    (0.85, "all but sealed"),
    (0.70, "clearly ahead"),
    (0.58, "real edge"),
    (0.45, "dead even"),
    (0.32, "behind"),
    (0.15, "deep trouble"),
    (0.00, "nearly gone"),
]


def phase_of(turn: int) -> str:
    for lo, hi, name in PHASES:
        if lo <= turn <= hi:
            return name
    return "lategame"


# ---------------------------------------------------------------- fitting


def _pav(x: np.ndarray, y: np.ndarray, w: np.ndarray):
    """Pool-adjacent-violators: the isotonic (non-decreasing) least-squares fit
    of y on sorted x. Returns (knot_x, knot_y) defining a step function."""
    order = np.argsort(x, kind="mergesort")
    x, y, w = x[order], y[order], w[order]
    # blocks of (weighted mean, weight, right-edge x)
    vals, wts, edges = [], [], []
    for xi, yi, wi in zip(x, y, w):
        vals.append(yi)
        wts.append(wi)
        edges.append(xi)
        # merge while the sequence decreases
        while len(vals) > 1 and vals[-2] > vals[-1]:
            v2, w2 = vals.pop(), wts.pop()
            e2 = edges.pop()
            v1, w1 = vals.pop(), wts.pop()
            edges.pop()
            tw = w1 + w2
            vals.append((v1 * w1 + v2 * w2) / tw)
            wts.append(tw)
            edges.append(e2)
    return np.asarray(edges, float), np.asarray(vals, float)


def apply_map(grid_x, grid_y, v):
    """Interpolate the calibration curve at v (clamped to the fitted range)."""
    return float(np.interp(float(v), np.asarray(grid_x, float),
                           np.asarray(grid_y, float)))


def fit_phase(vals, ys, grid=101):
    """Isotonic (PAV) fit, resampled onto a fixed 0..1 grid.

    Resampling — rather than thinning the PAV blocks by index — is what makes
    the artifact both compact and faithful: an early version dropped blocks
    evenly across the index range, which coarsened the TAILS, and the tails are
    precisely where the lategame read earns its accuracy (a confident 0.02 or
    0.97 is usually right). That version scored WORSE than raw (lategame Brier
    0.159 -> 0.230). Keep the full-resolution fit, sample it densely.
    """
    vals, ys = np.asarray(vals, float), np.asarray(ys, float)
    bx, by = _pav(vals, ys, np.ones_like(ys))
    gx = np.linspace(0.0, 1.0, grid)
    # step-evaluate the PAV blocks (bx = right edges) on the grid
    idx = np.clip(np.searchsorted(bx, gx, side="left"), 0, len(by) - 1)
    gy = by[idx]
    gy = np.maximum.accumulate(gy)  # keep it monotone after resampling
    # never emit certainty: the extreme PAV blocks are the sparsest (a handful
    # of opening reads above 0.9 that all happened to win pinned that block to
    # 1.000), and "certain" is the same overclaiming this map exists to fix
    gy = np.clip(gy, 0.02, 0.98)
    return gx.tolist(), gy.tolist()


# ---------------------------------------------------------------- data


def results_by_tag(me="PAC-Crystal"):
    out = {}
    for path in glob.glob(str(HERE / "bench" / "overnight_*_ladder.log")):
        cur = None
        for raw in open(path, errors="replace"):
            m = re.search(r">(battle-gen9oulongtimer-\d+)", raw)
            if m:
                cur = m.group(1)
            if cur is None:
                continue
            s = raw[raw.find("|"):].rstrip("\n") if "|" in raw else ""
            if s.startswith("|win|"):
                out[cur] = 1.0 if s[5:].strip() == me else 0.0
            elif s.startswith("|tie|"):
                out[cur] = 0.5
    return out


def load_reads():
    """(game_id, turn, raw_value, outcome) over every desk-read ledger."""
    res = results_by_tag()
    rows = []
    for path in glob.glob(str(HERE / "desk_reads_*.jsonl")):
        for line in open(path, errors="replace"):
            try:
                d = json.loads(line)
            except Exception:
                continue
            tag = d.get("battle_tag")
            if tag not in res:
                continue
            for turn, val in d.get("reads", []):
                rows.append((tag, int(turn), float(val), res[tag]))
    return rows


# ---------------------------------------------------------------- validate


def brier(p, y):
    return float(np.mean((np.asarray(p, float) - np.asarray(y, float)) ** 2))


def validate(rows, folds=5, seed=0):
    """Group k-fold BY GAME: fit on train games, score held-out games."""
    tags = sorted({r[0] for r in rows})
    rng = np.random.default_rng(seed)
    order = np.array(tags, dtype=object)
    rng.shuffle(order)
    fold_of = {t: i % folds for i, t in enumerate(order)}
    per_phase = {name: {"raw": [], "cal": [], "y": []} for _, _, name in PHASES}
    for f in range(folds):
        tr = [r for r in rows if fold_of[r[0]] != f]
        te = [r for r in rows if fold_of[r[0]] == f]
        for _, _, name in PHASES:
            trp = [r for r in tr if phase_of(r[1]) == name]
            tep = [r for r in te if phase_of(r[1]) == name]
            if len(trp) < 50 or not tep:
                continue
            kx, ky = fit_phase([r[2] for r in trp], [r[3] for r in trp])
            for r in tep:
                per_phase[name]["raw"].append(r[2])
                per_phase[name]["cal"].append(apply_map(kx, ky, r[2]))
                per_phase[name]["y"].append(r[3])
    return per_phase


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fit", action="store_true", help="refit and write the map")
    ap.add_argument("--report", action="store_true", help="validation only")
    args = ap.parse_args()

    rows = load_reads()
    if not rows:
        print("no joined desk-reads found"); return
    games = len({r[0] for r in rows})
    print(f"{len(rows)} reads over {games} finished games\n")

    per_phase = validate(rows)
    print("held-out (group k-fold BY GAME) Brier — raw vs calibrated")
    for _, _, name in PHASES:
        d = per_phase[name]
        if not d["y"]:
            continue
        r, c = brier(d["raw"], d["y"]), brier(d["cal"], d["y"])
        flag = "improves" if c < r else "NO GAIN"
        print(f"  {name:9s} n={len(d['y']):5d}  raw {r:.4f} -> calibrated {c:.4f}"
              f"   ({flag}; always-0.5 = 0.2500)")

    # The consumer emits LABELS, not probabilities, so the metric that matters
    # is whether a band's name matches what actually happens from there.
    print("\nband honesty — does the phrase match reality? (held-out, all phases)")
    allraw = np.concatenate([per_phase[n]["raw"] for _, _, n in PHASES
                             if per_phase[n]["y"]])
    allcal = np.concatenate([per_phase[n]["cal"] for _, _, n in PHASES
                             if per_phase[n]["y"]])
    ally = np.concatenate([per_phase[n]["y"] for _, _, n in PHASES
                           if per_phase[n]["y"]])
    print(f"  {'band (by raw value)':22s} {'n':>5s} {'nominal':>8s} "
          f"{'actual':>7s} {'gap':>7s}  |  {'calibrated says':>15s} {'gap':>7s}")
    for floor, label in BANDS:
        sel = allraw >= floor
        if floor < 0.85:
            sel &= allraw < prev_floor
        prev_floor = floor
        if sel.sum() < 20:
            continue
        actual = float(ally[sel].mean())
        nominal = float(allraw[sel].mean())
        calsays = float(allcal[sel].mean())
        print(f"  {label:22s} {int(sel.sum()):5d} {nominal:8.3f} {actual:7.3f} "
              f"{nominal-actual:+7.3f}  |  {calsays:15.3f} {calsays-actual:+7.3f}")

    if args.fit or not args.report:
        out = {"_comment": "raw MCTS root value -> empirical win rate, per phase "
                           "(isotonic/PAV). REPORTING ONLY — never feed back "
                           "into search; see read_calibration.py.",
               "phases": {}}
        for _, _, name in PHASES:
            sub = [r for r in rows if phase_of(r[1]) == name]
            if len(sub) < 50:
                continue
            kx, ky = fit_phase([r[2] for r in sub], [r[3] for r in sub])
            out["phases"][name] = {"n": len(sub), "x": kx, "y": ky}
        MAP_PATH.write_text(json.dumps(out, indent=1))
        print(f"\nwrote {MAP_PATH}")
        print("  sample: raw 0.63 ->", {
            name: round(calibrate(0.63, turn), 3)
            for name, turn in (("opening", 3), ("midgame", 15), ("lategame", 40))})


# ---------------------------------------------------------------- consumer API

_CACHE = None


def calibrate(value: float, turn: int, path: Path | None = None) -> float:
    """Raw root value -> calibrated win probability for REPORTING.

    Falls back to the raw value when no map is available, so a missing or
    unreadable artifact degrades to today's behaviour rather than failing.
    """
    global _CACHE
    if _CACHE is None:
        try:
            _CACHE = json.loads((path or MAP_PATH).read_text()).get("phases", {})
        except Exception:
            _CACHE = {}
    ph = _CACHE.get(phase_of(int(turn)))
    if not ph:
        return float(value)
    try:
        return apply_map(ph["x"], ph["y"], float(value))
    except Exception:
        return float(value)


if __name__ == "__main__":
    main()
