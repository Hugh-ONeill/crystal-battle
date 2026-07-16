#!/usr/bin/env python3
"""
Offline eval-calibration harness: compare position scorers on held-out
replay positions WITHOUT playing live games.

Scorers compared, all through their real runtime paths:
  static  — poke-engine's hand-tuned eval (pe.evaluate)
  mlp     — the trained value net (pe.ValueNet.predict, Rust featurization)
  linear  — a calibrated LINEAR logistic model on v3 features, trained here
            on the training pickle and exported to ONNX (the "calibrated
            hand-eval" hypothesis: same feature richness, learned weights)

Metrics: outcome-prediction AUC per game-phase stratum, Brier/calibration
for the probabilistic scorers, and — the question that motivated this —
AUC on MCTS-FLAT positions specifically (top move <45% of visits at a
short probe), where the live-play evidence says the static eval is blind.

Holdout must be replays the training pickle never saw (fresh scrape).

Usage:
  .venv/bin/python showdown/eval_calibration.py \
      --train showdown/gen9ou_replay_data_v3.pkl \
      --holdout <scratch>/holdout_data.pkl \
      --linear-out showdown/value_net_gen9_linear.onnx
"""

from __future__ import annotations

import argparse
import pickle
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

import poke_engine as pe
from showdown.featurizer_v3 import parse_state_v3, STATE_V3_FEATURES


def load_rows(path):
    """[(state_str, label, turn_idx, game_len)] from a training pickle."""
    games = pickle.load(open(path, "rb"))
    rows = []
    for winner, turns in games:
        if winner == 0:
            continue
        label = 1.0 if winner > 0 else 0.0
        n = len(turns)
        for i, t in enumerate(turns):
            rows.append((t[0], label, i, n))
    return rows


def auc(scores, labels):
    """Mann-Whitney AUC; None if a class is missing."""
    scores, labels = np.asarray(scores, float), np.asarray(labels, float)
    pos, neg = scores[labels == 1], scores[labels == 0]
    if len(pos) == 0 or len(neg) == 0:
        return None
    order = np.argsort(np.concatenate([pos, neg]), kind="mergesort")
    ranks = np.empty(len(order), float)
    ranks[order] = np.arange(1, len(order) + 1)
    # average ties
    allv = np.concatenate([pos, neg])
    for v in np.unique(allv):
        m = allv == v
        if m.sum() > 1:
            ranks[m] = ranks[m].mean()
    r_pos = ranks[: len(pos)].sum()
    return (r_pos - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg))


def brier(probs, labels):
    p, y = np.asarray(probs, float), np.asarray(labels, float)
    return float(np.mean((p - y) ** 2))


def featurize(rows, cache: Path | None = None):
    if cache is not None and cache.exists():
        return np.load(cache)["x"]
    x = np.zeros((len(rows), STATE_V3_FEATURES), dtype=np.float32)
    t0 = time.time()
    for i, (s, *_rest) in enumerate(rows):
        x[i] = parse_state_v3(s)
        if i % 20000 == 0 and i:
            print(f"  featurized {i}/{len(rows)} ({time.time()-t0:.0f}s)")
    if cache is not None:
        np.savez_compressed(cache, x=x)
    return x


def train_linear(train_rows, cache_dir: Path, out_onnx: str,
                 epochs: int = 6, weight_decay: float = 1e-4):
    import torch
    import torch.nn as nn

    x = featurize(train_rows, cache_dir / "train_feats.npz")
    y = np.array([r[1] for r in train_rows], dtype=np.float32)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    model = nn.Linear(STATE_V3_FEATURES, 1).to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3,
                           weight_decay=weight_decay)
    lossf = nn.BCEWithLogitsLoss()
    xt = torch.from_numpy(x)
    yt = torch.from_numpy(y)
    n = len(y)
    for ep in range(epochs):
        perm = torch.randperm(n)
        tot = 0.0
        for i in range(0, n, 4096):
            idx = perm[i:i + 4096]
            xb, yb = xt[idx].to(dev), yt[idx].to(dev)
            opt.zero_grad()
            loss = lossf(model(xb).squeeze(-1), yb)
            loss.backward()
            opt.step()
            tot += loss.item() * len(idx)
        print(f"  linear epoch {ep+1}: loss={tot/n:.4f}")

    # export with value_train's conventions: logits out, sigmoid on Rust side
    class Wrap(nn.Module):
        def __init__(self, lin):
            super().__init__()
            self.lin = lin
        def forward(self, s):
            return self.lin(s).squeeze(-1)

    wrap = Wrap(model.cpu()).eval()
    dummy = torch.randn(1, STATE_V3_FEATURES)
    torch.onnx.export(wrap, dummy, out_onnx,
                      input_names=["state"], output_names=["value"],
                      dynamic_axes={"state": {0: "batch"},
                                    "value": {0: "batch"}},
                      dynamo=False)
    print(f"  linear model exported to {out_onnx}")
    return model


def score_holdout(rows, mlp_path: str, linear_path: str):
    """Score every holdout row through the REAL runtime paths."""
    vn_mlp = pe.ValueNet(mlp_path)
    vn_lin = pe.ValueNet(linear_path)
    static, mlp, lin = [], [], []
    t0 = time.time()
    for i, (s, *_rest) in enumerate(rows):
        st = pe.State.from_string(s)
        static.append(pe.evaluate(st))
        mlp.append(vn_mlp.predict(st))
        lin.append(vn_lin.predict(st))
        if i % 1000 == 0 and i:
            print(f"  scored {i}/{len(rows)} ({time.time()-t0:.0f}s)")
    return np.array(static), np.array(mlp), np.array(lin)


def flat_mask(rows, sample: int, probe_ms: int, flat_share: float, rng):
    """Probe a sample of LATE positions with short MCTS; flat = top move
    below flat_share of visits. Returns (indices_probed, is_flat)."""
    late = [i for i, r in enumerate(rows) if r[2] >= 15]
    idx = rng.choice(late, size=min(sample, len(late)), replace=False)
    flags = np.zeros(len(idx), bool)
    for j, i in enumerate(idx):
        st = pe.State.from_string(rows[i][0])
        res = pe.monte_carlo_tree_search(st, probe_ms)
        tot = sum(m.visits for m in res.side_one) or 1
        top = max((m.visits for m in res.side_one), default=0)
        flags[j] = (top / tot) < flat_share
    return idx, flags


def stratum_table(rows, scores_by_name, labels):
    rows_np = np.array([(r[2], r[3]) for r in rows])
    turn, glen = rows_np[:, 0], rows_np[:, 1]
    strata = {
        "all": np.ones(len(rows), bool),
        "early (t<8)": turn < 8,
        "mid (8-19)": (turn >= 8) & (turn < 20),
        "late (t>=20)": turn >= 20,
        "grind (len>=40,t>=20)": (glen >= 40) & (turn >= 20),
    }
    print(f"\n{'stratum':<24} {'n':>6} " +
          " ".join(f"{k:>8}" for k in scores_by_name))
    for name, mask in strata.items():
        line = f"{name:<24} {mask.sum():>6} "
        for k, v in scores_by_name.items():
            a = auc(v[mask], labels[mask])
            line += f"{a:>8.3f} " if a is not None else f"{'--':>8} "
        print(line)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", default="showdown/gen9ou_replay_data_v3.pkl")
    ap.add_argument("--holdout", required=True)
    ap.add_argument("--mlp", default="showdown/value_net_gen9_v3.onnx")
    ap.add_argument("--linear-out", default="showdown/value_net_gen9_linear.onnx")
    ap.add_argument("--cache-dir", default=None)
    ap.add_argument("--skip-linear-train", action="store_true")
    ap.add_argument("--flat-sample", type=int, default=240)
    ap.add_argument("--flat-probe-ms", type=int, default=200)
    args = ap.parse_args()

    cache = Path(args.cache_dir) if args.cache_dir else Path(args.holdout).parent
    rng = np.random.default_rng(7)

    holdout = load_rows(args.holdout)
    print(f"holdout: {len(holdout)} positions "
          f"(mean label {np.mean([r[1] for r in holdout]):.3f})")

    if not args.skip_linear_train:
        print("training linear model on training pickle...")
        train_rows = load_rows(args.train)
        train_linear(train_rows, cache, args.linear_out)

    print("scoring holdout through runtime paths...")
    static, mlp, lin = score_holdout(holdout, args.mlp, args.linear_out)
    labels = np.array([r[1] for r in holdout])

    scorers = {"static": static, "mlp": mlp, "linear": lin}
    stratum_table(holdout, scorers, labels)

    print(f"\nBrier (probabilistic scorers): "
          f"mlp={brier(mlp, labels):.4f}  linear={brier(lin, labels):.4f}")
    early = np.array([r[2] < 3 for r in holdout])
    print(f"mean prediction on near-openings (want ~0.50): "
          f"mlp={mlp[early].mean():.3f}  linear={lin[early].mean():.3f}")

    print(f"\nprobing {args.flat_sample} late positions at "
          f"{args.flat_probe_ms}ms for MCTS-flatness...")
    idx, flags = flat_mask(holdout, args.flat_sample, args.flat_probe_ms,
                           0.45, rng)
    sub_labels = labels[idx]
    print(f"flat: {flags.sum()}  decisive: {(~flags).sum()}")
    print(f"{'subset':<12} {'n':>5} {'static':>8} {'mlp':>8} {'linear':>8}")
    for name, m in (("FLAT", flags), ("decisive", ~flags)):
        line = f"{name:<12} {m.sum():>5} "
        for v in (static, mlp, lin):
            a = auc(v[idx][m], sub_labels[m])
            line += f"{a:>8.3f} " if a is not None else f"{'--':>8} "
        print(line)


if __name__ == "__main__":
    main()
