#!/usr/bin/env python3
"""Joint value+policy training (gen9 / v3 features).

One network with a shared trunk feeding two heads:
  trunk:        state → MLP (n_layers × hidden)
  value head:   trunk → 1   (BCE on V)
  policy head:  trunk → 9   (soft cross-entropy on π)

Loss = BCE(v, V) + λ · CE(p, π)

Reads the new self-play schema:
    [(winner_int, [(state_str, v_p1, s1_pi9, s2_pi9), ...]), ...]

Side-flip augmentation produces both (state, V, s1_pi) and
(flipped, 1-V, s2_pi) per turn.

Outputs:
  - <model>.value.pt + <model>.value.onnx  (loadable as pe.ValueNet)
  - <model>.policy.pt                      (loadable via PolicyOnnx wrapper)
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


# ============================================================
# DATA
# ============================================================

def prepare_joint_data(data_path: str):
    print(f"Loading {data_path}...")
    with open(data_path, "rb") as f:
        data = pickle.load(f)

    states: list[np.ndarray] = []
    v_labels: list[float] = []
    p_labels: list[np.ndarray] = []
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
            state_str, v_p1, s1_pi, s2_pi = turn_data[:4]
            s1_pi = np.asarray(s1_pi, dtype=np.float32)
            s2_pi = np.asarray(s2_pi, dtype=np.float32)
            if s1_pi.shape != (N_ACTIONS,) or s2_pi.shape != (N_ACTIONS,):
                n_skipped += 1
                continue

            # side one perspective
            states.append(parse_state_v3(state_str))
            v_labels.append(float(v_p1))
            p_labels.append(s1_pi)

            # side two perspective via state flip
            major = state_str.split("/")
            if len(major) >= 2:
                flipped = "/".join([major[1], major[0]] + major[2:])
                states.append(parse_state_v3(flipped))
                v_labels.append(1.0 - float(v_p1))
                p_labels.append(s2_pi)

    states_arr = np.asarray(states, dtype=np.float32)
    v_arr = np.asarray(v_labels, dtype=np.float32)
    p_arr = np.asarray(p_labels, dtype=np.float32)
    print(f"  {len(states)} samples, state_dim={states_arr.shape[1]}, "
          f"skipped={n_skipped}")
    print(f"  V mean={v_arr.mean():.3f} std={v_arr.std():.3f}")
    print(f"  π top-1 mass mean={p_arr.max(axis=1).mean():.3f}")
    return states_arr, v_arr, p_arr


# ============================================================
# MODEL
# ============================================================

class JointNet(nn.Module):
    def __init__(self, state_dim: int, hidden: int = 256, n_layers: int = 3):
        super().__init__()
        layers = []
        in_dim = state_dim
        for _ in range(n_layers):
            layers.extend([nn.Linear(in_dim, hidden), nn.ReLU()])
            in_dim = hidden
        self.trunk = nn.Sequential(*layers)
        self.value_head = nn.Linear(hidden, 1)
        self.policy_head = nn.Linear(hidden, N_ACTIONS)

    def forward(self, x):
        h = self.trunk(x)
        return self.value_head(h).squeeze(-1), self.policy_head(h)


def soft_cross_entropy(logits, target):
    return -(target * F.log_softmax(logits, dim=-1)).sum(dim=-1).mean()


# ============================================================
# CHECKPOINT FORMATTING
# ============================================================

def save_value_pt(joint: JointNet, save_path: Path, state_dim: int, hidden: int):
    """Save in the format value_train.py uses (loadable as pe.ValueNet via ONNX)."""
    # Reconstruct an equivalent Sequential matching value_train.py's ValueNet
    # (trunk linears + ReLU + final linear → 1).
    seq = nn.Sequential(*list(joint.trunk), joint.value_head)
    sd = {f"net.{k}": v.clone() for k, v in seq.state_dict().items()}
    torch.save({"model": sd, "state_dim": state_dim, "hidden": hidden, "gamma": 1.0},
               save_path)
    return seq


def save_policy_pt(joint: JointNet, save_path: Path, state_dim: int, hidden: int,
                   n_layers: int):
    """Save in the format policy_train.py uses (loadable via PolicyOnnx wrapper)."""
    seq = nn.Sequential(*list(joint.trunk), joint.policy_head)
    sd = {f"net.{k}": v.clone() for k, v in seq.state_dict().items()}
    torch.save({"model": sd, "state_dim": state_dim, "hidden": hidden, "n_layers": n_layers},
               save_path)


# ============================================================
# TRAINING
# ============================================================

def train(data_path: str, save_prefix: str, epochs: int, lr: float, hidden: int,
          n_layers: int, batch_size: int, device: str, lambda_p: float):
    states, v_targets, p_targets = prepare_joint_data(data_path)
    n = len(states)
    rng = np.random.default_rng(0)
    perm = rng.permutation(n)
    states, v_targets, p_targets = states[perm], v_targets[perm], p_targets[perm]
    cut = int(n * 0.95)
    X_train = torch.from_numpy(states[:cut]).to(device)
    V_train = torch.from_numpy(v_targets[:cut]).to(device)
    P_train = torch.from_numpy(p_targets[:cut]).to(device)
    X_val = torch.from_numpy(states[cut:]).to(device)
    V_val = torch.from_numpy(v_targets[cut:]).to(device)
    P_val = torch.from_numpy(p_targets[cut:]).to(device)
    print(f"  train={len(X_train)}, val={len(X_val)}")

    state_dim = states.shape[1]
    model = JointNet(state_dim=state_dim, hidden=hidden, n_layers=n_layers).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  {n_params:,} parameters, λ_policy={lambda_p}")
    print(f"  V baseline (BCE on 0.5 prediction): {-np.log(0.5):.4f}")
    print(f"  π baseline (CE on uniform):         {-np.log(1/N_ACTIONS):.4f}")

    opt = optim.Adam(model.parameters(), lr=lr)
    sched = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)

    best_val = float("inf")
    save_prefix_path = Path(save_prefix)
    value_pt = save_prefix_path.with_suffix(".value.pt")
    policy_pt = save_prefix_path.with_suffix(".policy.pt")
    value_onnx = save_prefix_path.with_suffix(".value.onnx")

    t0 = time.time()
    for epoch in range(epochs):
        model.train()
        perm = torch.randperm(len(X_train))
        v_total = p_total = total = 0.0
        nb = 0
        for i in range(0, len(X_train), batch_size):
            idx = perm[i:i + batch_size]
            xb, vb, pb = X_train[idx], V_train[idx], P_train[idx]
            opt.zero_grad()
            v_pred, p_logits = model(xb)
            v_loss = F.binary_cross_entropy_with_logits(v_pred, vb)
            p_loss = soft_cross_entropy(p_logits, pb)
            loss = v_loss + lambda_p * p_loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            v_total += v_loss.item(); p_total += p_loss.item()
            total += loss.item(); nb += 1
        sched.step()

        model.eval()
        with torch.no_grad():
            v_pred_val, p_logits_val = model(X_val)
            v_val = F.binary_cross_entropy_with_logits(v_pred_val, V_val).item()
            p_val = soft_cross_entropy(p_logits_val, P_val).item()
            top1 = (p_logits_val.argmax(-1) == P_val.argmax(-1)).float().mean().item()
            total_val = v_val + lambda_p * p_val

        marker = ""
        if total_val < best_val:
            best_val = total_val
            save_value_pt(model, value_pt, state_dim, hidden)
            save_policy_pt(model, policy_pt, state_dim, hidden, n_layers)
            marker = " *best*"
        print(f"  Epoch {epoch + 1:2d}: train={total/max(nb,1):.4f} "
              f"(v={v_total/max(nb,1):.4f}, p={p_total/max(nb,1):.4f}) "
              f"val_v={v_val:.4f} val_p={p_val:.4f} top1={top1:.3f}{marker}")

    print(f"\nBest val (combined): {best_val:.4f} ({time.time()-t0:.0f}s)")
    print(f"  saved: {value_pt}, {policy_pt}")

    # ONNX export of value head
    ckpt = torch.load(value_pt, weights_only=True)
    layers = []
    in_dim = ckpt["state_dim"]
    for _ in range(n_layers):
        layers.extend([nn.Linear(in_dim, ckpt["hidden"]), nn.ReLU()])
        in_dim = ckpt["hidden"]
    layers.append(nn.Linear(ckpt["hidden"], 1))
    seq = nn.Sequential(*layers)
    # The saved keys are 'net.X.weight'. Strip prefix:
    sd_net = {k[len("net."):]: v for k, v in ckpt["model"].items() if k.startswith("net.")}
    seq.load_state_dict(sd_net)
    seq.eval()
    dummy = torch.randn(1, ckpt["state_dim"])

    class ValueOnly(nn.Module):
        def __init__(self, s): super().__init__(); self.s = s
        def forward(self, x): return self.s(x).squeeze(-1)
    export_model = ValueOnly(seq)
    torch.onnx.export(export_model, dummy, value_onnx,
                      input_names=["state"], output_names=["value"],
                      dynamic_axes={"state": {0: "batch"}, "value": {0: "batch"}},
                      dynamo=False)
    print(f"  ONNX: {value_onnx}")


def main():
    ap = argparse.ArgumentParser(description="Joint value+policy training (gen9/v3)")
    ap.add_argument("--data", required=True)
    ap.add_argument("--prefix", default="joint_net",
                    help="Output path prefix (writes <prefix>.value.{pt,onnx} + .policy.pt)")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--hidden", type=int, default=256)
    ap.add_argument("--n-layers", type=int, default=3)
    ap.add_argument("--batch-size", type=int, default=512)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--lambda-p", type=float, default=1.0,
                    help="weight on the policy CE loss term")
    args = ap.parse_args()
    train(args.data, args.prefix, args.epochs, args.lr, args.hidden,
          args.n_layers, args.batch_size, args.device, args.lambda_p)


if __name__ == "__main__":
    main()
