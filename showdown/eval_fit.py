#!/usr/bin/env python3
"""Fit the eval's 13 term-group weights against game outcomes — rung 1 of the
learning axis, MEASUREMENT ONLY.

What this answers: (a) does an outcome-fitted reweighting of the hand eval's
own term groups predict wins better than the hand eval's uniform sum — in
particular in the OPENING, where the hand eval is measured at coin-flip
(Brier >= 0.25 T1-9)? (b) WHICH terms does the data want amplified, damped,
or inverted? The weight vector is informative even if nothing deploys.

What this does NOT answer (pre-registered cautions):
  - deployment: the branch-invariance law says leaf-value changes may not
    couple to decisions — any live use must pass an offline behavior-coupling
    sweep (weather_scale_sweep pattern) BEFORE a paired A/B;
  - generality: rows are ~95% bench-vs-stock-fp states; the fit learns "what
    converts against this opponent pool with our pilot".

Method: hand-rolled weighted logistic IRLS (no sklearn on the box), L2 on
standardized features, GROUP-5-FOLD BY GAME (rows within a game share one
outcome — the read_calibration lesson), rows weighted 1/game_length so long
losses don't teach "long game = losing". Baselines fit on the same folds:
base rate, and the HAND EVAL ITSELF as a one-parameter logistic (temperature
on sum(terms)) — the fair same-family comparison.

  .venv/bin/python showdown/eval_fit.py --data showdown/bench/eval_dataset.npz
"""

from __future__ import annotations

import argparse
import json

import numpy as np


def irls(X, y, w_row, lam=1.0, iters=50):
    """Weighted L2 logistic; X standardized, intercept unpenalized."""
    n, d = X.shape
    Xb = np.hstack([np.ones((n, 1)), X])
    beta = np.zeros(d + 1)
    pen = np.eye(d + 1) * lam
    pen[0, 0] = 0.0
    for _ in range(iters):
        z = Xb @ beta
        p = 1.0 / (1.0 + np.exp(-np.clip(z, -30, 30)))
        W = w_row * p * (1 - p) + 1e-9
        g = Xb.T @ (w_row * (y - p)) - pen @ beta
        H = (Xb * W[:, None]).T @ Xb + pen
        step = np.linalg.solve(H, g)
        beta += step
        if np.abs(step).max() < 1e-8:
            break
    return beta


def predict(beta, X):
    z = beta[0] + X @ beta[1:]
    return 1.0 / (1.0 + np.exp(-np.clip(z, -30, 30)))


def brier(p, y, w):
    return float(np.average((p - y) ** 2, weights=w))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="showdown/bench/eval_dataset.npz")
    ap.add_argument("--lam", type=float, default=1.0)
    ap.add_argument("--out", default="showdown/bench/eval_fit_weights.json")
    args = ap.parse_args()

    d = np.load(args.data, allow_pickle=True)
    X, y = d["X"].astype(np.float64), d["y"].astype(np.float64)
    tags, turns, labels = d["tags"], d["turns"], list(d["labels"])

    # per-game row weight: every game contributes equally
    uniq, inv, counts = np.unique(tags, return_inverse=True,
                                  return_counts=True)
    w_row = 1.0 / counts[inv]
    w_row *= len(w_row) / w_row.sum()

    keep = X.std(axis=0) > 0            # drop dead columns (pending etc.)
    Xk, lk = X[:, keep], [l for l, k in zip(labels, keep) if k]
    mu, sd = Xk.mean(axis=0), Xk.std(axis=0)
    Xs = (Xk - mu) / sd
    s = X.sum(axis=1)                    # the hand eval's own score
    ss = (s - s.mean()) / s.std()

    rng = np.random.default_rng(42)
    order = rng.permutation(len(uniq))
    folds = np.array_split(order, 5)

    phases = {"opening(T<=9)": turns <= 9,
              "midgame(10-19)": (turns >= 10) & (turns <= 19),
              "lategame(20+)": turns >= 20}
    acc = {m: {ph: [[], [], []] for ph in phases} for m in ("fit", "hand", "base")}

    for f in folds:
        te = np.isin(inv, f)
        tr = ~te
        beta = irls(Xs[tr], y[tr], w_row[tr], lam=args.lam)
        bh = irls(ss[tr, None], y[tr], w_row[tr], lam=0.0)
        pb = np.average(y[tr], weights=w_row[tr])
        preds = {"fit": predict(beta, Xs[te]),
                 "hand": predict(bh, ss[te, None]),
                 "base": np.full(te.sum(), pb)}
        for ph, mask in phases.items():
            m = mask[te]
            if not m.any():
                continue
            for name, p in preds.items():
                acc[name][ph][0].append(p[m])
                acc[name][ph][1].append(y[te][m])
                acc[name][ph][2].append(w_row[te][m])

    print(f"{len(y)} rows / {len(uniq)} games, {Xs.shape[1]} live terms, "
          f"lam={args.lam}, group-5-fold by game, rows weighted 1/game")
    print(f"\n  out-of-fold Brier (lower is better)")
    print(f"  {'phase':16s} {'fitted':>8s} {'hand-eval':>10s} {'base-rate':>10s}")
    for ph in phases:
        row = []
        for name in ("fit", "hand", "base"):
            p = np.concatenate(acc[name][ph][0])
            yy = np.concatenate(acc[name][ph][1])
            ww = np.concatenate(acc[name][ph][2])
            row.append(brier(p, yy, ww))
        print(f"  {ph:16s} {row[0]:8.4f} {row[1]:10.4f} {row[2]:10.4f}")

    beta_full = irls(Xs, y, w_row, lam=args.lam)
    bh_full = irls(ss[:, None], y, w_row, lam=0.0)
    # per-term multiplier RELATIVE to the hand eval's fitted temperature:
    # 1.0 = keep the hand constant, >1 amplify, <1 damp, <0 INVERTED
    temp = bh_full[1] / s.std()
    mult = (beta_full[1:] / sd) / temp
    print(f"\n  fitted per-term multipliers (1.0 = hand constant is right)")
    for l, m_ in sorted(zip(lk, mult), key=lambda kv: -abs(kv[1])):
        print(f"  {l:16s} {m_:7.2f}")
    json.dump({"labels": lk, "multipliers": [round(float(m), 4) for m in mult],
               "temperature": float(temp), "lam": args.lam,
               "n_rows": int(len(y)), "n_games": int(len(uniq))},
              open(args.out, "w"), indent=1)
    print(f"\n  weights -> {args.out}")
    print("  CAUTION: measurement only — deployment needs the behavior-"
          "coupling sweep first\n  (branch-invariance law), then the paired "
          "A/B. Distribution: bench-vs-stock-fp.")


if __name__ == "__main__":
    main()
