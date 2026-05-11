#!/usr/bin/env python3
"""Train a policy net on per-turn MCTS visit distributions (gen9 / v3 features).

Reads a self-play pickle in the new 4-tuple schema:
    [(winner_int, [(state_str, v_p1, s1_pi9, s2_pi9), ...]), ...]

Each turn produces two training samples (state, π):
    1. (parse_state_v3(state_str),         s1_pi9)
    2. (parse_state_v3(side-flipped state), s2_pi9)

Loss: cross-entropy with soft targets, equivalent to KL up to a constant.

Architecture matches value_train.py: n_layers × hidden trunk, output 9 logits
(matches Rust policy::N_ACTIONS). Exports to ONNX so it can be loaded as
priors for `pe.mcts_with_value`.
"""

from __future__ import annotations

import argparse
import pickle
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

sys.path.insert(0, str(Path(__file__).parent.parent))

from showdown.features_v3 import parse_state_v3, STATE_V3_FEATURES


N_ACTIONS = 9


def prepare_policy_data(data_path: str):
    print(f"Loading {data_path}...")
    with open(data_path, "rb") as f:
        data = pickle.load(f)

    states: list[np.ndarray] = []
    targets: list[np.ndarray] = []
    n_skipped = 0

    for game in data:
        if not (isinstance(game, tuple) and len(game) == 2):
            n_skipped += 1
            continue
        _, turns = game
        for turn_data in turns:
            if len(turn_data) < 4:
                n_skipped += 1
                continue
            state_str = turn_data[0]
            s1_pi = np.asarray(turn_data[2], dtype=np.float32)
            s2_pi = np.asarray(turn_data[3], dtype=np.float32)
            if s1_pi.shape != (N_ACTIONS,) or s2_pi.shape != (N_ACTIONS,):
                n_skipped += 1
                continue
            # side-one perspective
            states.append(parse_state_v3(state_str))
            targets.append(s1_pi)
            # side-two perspective via state flip
            major = state_str.split("/")
            if len(major) >= 2:
                flipped = "/".join([major[1], major[0]] + major[2:])
                states.append(parse_state_v3(flipped))
                targets.append(s2_pi)

    states_arr = np.asarray(states, dtype=np.float32)
    targets_arr = np.asarray(targets, dtype=np.float32)
    print(f"  {len(states)} samples, state_dim={states_arr.shape[1]}, "
          f"skipped={n_skipped}")
    flat = targets_arr.flatten()
    print(f"  target stats: mean={targets_arr.mean():.4f} (uniform=1/9≈0.111), "
          f"zero-frac={(flat == 0).sum() / flat.size:.1%}, "
          f"one-frac={(flat == 1).sum() / flat.size:.2%}")
    return states_arr, targets_arr


class PolicyNet(nn.Module):
    def __init__(self, state_dim: int, hidden: int = 256, n_layers: int = 3):
        super().__init__()
        layers = []
        in_dim = state_dim
        for _ in range(n_layers):
            layers.extend([nn.Linear(in_dim, hidden), nn.ReLU()])
            in_dim = hidden
        layers.append(nn.Linear(hidden, N_ACTIONS))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


def soft_cross_entropy(logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    log_probs = F.log_softmax(logits, dim=-1)
    return -(target * log_probs).sum(dim=-1).mean()


def train(data_path: str, save_path: str = "policy_net.pt",
          epochs: int = 30, lr: float = 1e-3, hidden: int = 256,
          n_layers: int = 3, batch_size: int = 512, device: str = "cpu"):
    states, targets = prepare_policy_data(data_path)
    n = len(states)
    rng = np.random.default_rng(0)
    perm = rng.permutation(n)
    states, targets = states[perm], targets[perm]
    cut = int(n * 0.95)
    X_train = torch.from_numpy(states[:cut]).to(device)
    Y_train = torch.from_numpy(targets[:cut]).to(device)
    X_val = torch.from_numpy(states[cut:]).to(device)
    Y_val = torch.from_numpy(targets[cut:]).to(device)
    print(f"  train={len(X_train)}, val={len(X_val)}")

    model = PolicyNet(state_dim=states.shape[1], hidden=hidden,
                      n_layers=n_layers).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  {n_params:,} parameters")
    print(f"  uniform-policy CE baseline: {-np.log(1 / N_ACTIONS):.4f}")

    opt = optim.Adam(model.parameters(), lr=lr)
    sched = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)

    best_val = float("inf")
    t0 = time.time()
    for epoch in range(epochs):
        model.train()
        perm = torch.randperm(len(X_train))
        total = 0.0
        nb = 0
        for i in range(0, len(X_train), batch_size):
            idx = perm[i:i + batch_size]
            xb, yb = X_train[idx], Y_train[idx]
            opt.zero_grad()
            loss = soft_cross_entropy(model(xb), yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            total += loss.item()
            nb += 1
        sched.step()

        model.eval()
        with torch.no_grad():
            val_logits = model(X_val)
            val_loss = soft_cross_entropy(val_logits, Y_val).item()
            top1 = (val_logits.argmax(-1) == Y_val.argmax(-1)).float().mean().item()

        marker = ""
        if val_loss < best_val:
            best_val = val_loss
            torch.save({
                "model": model.state_dict(),
                "state_dim": states.shape[1],
                "hidden": hidden,
                "n_layers": n_layers,
            }, save_path)
            marker = " *best*"
        print(f"  Epoch {epoch + 1:2d}: train={total / max(nb, 1):.4f} "
              f"val={val_loss:.4f} top1={top1:.3f}{marker}")

    print(f"\nBest val CE: {best_val:.4f}, saved to {save_path} "
          f"({time.time() - t0:.0f}s)")

    onnx_path = save_path.replace(".pt", ".onnx")
    ckpt = torch.load(save_path, weights_only=True)
    export_model = PolicyNet(state_dim=ckpt["state_dim"], hidden=ckpt["hidden"],
                             n_layers=ckpt.get("n_layers", 3))
    export_model.load_state_dict(ckpt["model"])
    export_model.eval()
    dummy = torch.randn(1, ckpt["state_dim"])
    torch.onnx.export(export_model, dummy, onnx_path,
                      input_names=["state"], output_names=["logits"],
                      dynamic_axes={"state": {0: "batch"}, "logits": {0: "batch"}},
                      dynamo=False)
    print(f"ONNX exported to {onnx_path}")


def main():
    ap = argparse.ArgumentParser(description="Policy net training (gen9 / v3)")
    ap.add_argument("--data", required=True)
    ap.add_argument("--model", default="policy_net.pt")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--hidden", type=int, default=256)
    ap.add_argument("--n-layers", type=int, default=3)
    ap.add_argument("--batch-size", type=int, default=512)
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()
    train(args.data, args.model, args.epochs, args.lr, args.hidden,
          args.n_layers, args.batch_size, args.device)


if __name__ == "__main__":
    main()
