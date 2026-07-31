#!/usr/bin/env python3
"""Build the (eval-term-vector, outcome) dataset for residual recalibration.

The hand eval's ~50 constants have only ever been tuned one paired A/B at a
time (~15pp noise floor, days per constant). This dataset inverts that: every
--dump-states pool row becomes pe.evaluate_terms(state) — the eval's own
13-group decomposition (poke-engine EVAL_TERM_LABELS) — labeled with whether
WE won that game. A logistic fit over it recalibrates all the groups' relative
weights at once against outcomes, with the current constants as the prior
(all-1.0 weights reproduce today's eval exactly).

Honesty notes baked into the format:
  - rows within a game share one outcome -> `tag` is saved per row so any fit
    validates GROUP-K-FOLD BY GAME (the read_calibration lesson);
  - `turn` is saved so opening/mid/late can be fit or gated separately (the
    eval is measured blind in the opening — the fit should get the chance to
    say so);
  - states are ROOT states from our own games (the search's visit
    distribution starts here); rollout leaves are deeper — same manifold,
    not the same distribution. That gap is rung-3's problem, not this one's.

  .venv/bin/python showdown/eval_dataset.py \
      --pool showdown/bench/marginpool_states.jsonl \
      --logs 'showdown/bench/marginpool_L*_ours.log' \
      --out showdown/bench/eval_dataset.npz
Repeat --pool/--logs in pairs to merge sources (bench pools + ladder states).
"""

from __future__ import annotations

import argparse
import glob
import json
import re
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from roles_screen import TAG_RE, US_RE


def outcomes_from_logs(paths: list[str]) -> dict[str, int]:
    """battle tag -> 1 if OUR side won, 0 if lost; undecided tags absent."""
    out: dict[str, int] = {}
    us_by_tag: dict[str, str] = {}
    for path in paths:
        cur = None
        for line in open(path, errors="replace"):
            m = TAG_RE.search(line)
            if m:
                cur = m.group(1)
            if cur is None or "|" not in line:
                continue
            s = line.strip()
            pm = re.match(r"\|player\|(p[12])\|([^|]+)", s)
            if pm and US_RE.fullmatch(pm.group(2).strip()):
                us_by_tag[cur] = pm.group(2).strip()
            wm = re.match(r"\|win\|(.+)$", s)
            if wm:
                winner = wm.group(1).strip()
                if cur in us_by_tag:
                    out[cur] = int(winner == us_by_tag[cur])
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pool", action="append", required=True)
    ap.add_argument("--logs", action="append", required=True,
                    help="one glob per --pool, same order (quote them)")
    ap.add_argument("--out", default="showdown/bench/eval_dataset.npz")
    args = ap.parse_args()
    if len(args.pool) != len(args.logs):
        sys.exit("need one --logs glob per --pool")

    import poke_engine as pe
    m = pe.poke_engine
    labels = m.eval_term_labels()

    X, y, tags, turns, srcs = [], [], [], [], []
    for pool, logglob in zip(args.pool, args.logs):
        outcomes = outcomes_from_logs(sorted(glob.glob(logglob)))
        n_rows = n_undecided = n_bad = 0
        for line in open(pool):
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            won = outcomes.get(rec["tag"])
            if won is None:
                n_undecided += 1
                continue
            try:
                terms = m.evaluate_terms(pe.State.from_string(rec["state"]))
            except Exception:
                n_bad += 1
                continue
            X.append(terms)
            y.append(won)
            tags.append(rec["tag"])
            turns.append(rec["turn"])
            srcs.append(Path(pool).stem)
            n_rows += 1
        print(f"  {pool}: {n_rows} rows ({n_undecided} undecided-game rows "
              f"dropped, {n_bad} unparseable)")

    if not X:
        sys.exit("no rows")
    X = np.asarray(X, dtype=np.float32)
    y = np.asarray(y, dtype=np.int8)
    np.savez_compressed(
        args.out, X=X, y=y, tags=np.asarray(tags), turns=np.asarray(turns,
        dtype=np.int16), sources=np.asarray(srcs), labels=np.asarray(labels))
    games = len(set(tags))
    print(f"\nwrote {args.out}: {len(X)} rows / {games} games, "
          f"winrate {y.mean():.3f}")
    print(f"  {'term':16s} {'mean':>8s} {'std':>8s} {'|corr(outcome)|':>16s}")
    for i, lab in enumerate(labels):
        col = X[:, i]
        c = np.corrcoef(col, y)[0, 1] if col.std() > 0 else 0.0
        print(f"  {lab:16s} {col.mean():8.2f} {col.std():8.2f} {abs(c):16.3f}")
    print("  (corr is a sanity peek, not the fit — rows within a game are "
          "correlated;\n   any real fit must group-k-fold by tag)")


if __name__ == "__main__":
    main()
