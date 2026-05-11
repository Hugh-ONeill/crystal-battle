#!/usr/bin/env python3
"""DeepSet value-net training (gen9 / v3 features).

The flat-MLP value nets we tried memorize positional interactions across the
2738-dim feature vector. Pokemon teams are permutation-invariant by
construction — swapping mon-3 and mon-5 in the team list shouldn't change V.
A DeepSet encoder bakes that in:

  per-mon encoder (shared 224 → emb)  applied to each of 12 mons
  per-side pool   (mean over 6 mons)
  head            (concat both pools + extras + globals → MLP → V)

Trained as a single-output value net so it slots into the existing
`pe.ValueNet` ONNX path; pair with `policy_net_az.pt` at bench time.

The per-mon encoder also sees an is_active flag (derived from side extras'
one-hot active_idx) so it can distinguish the front-line mon from bench mons.
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
from showdown.features_v3 import (
    parse_state_v3,
    POKEMON_V3_FEATURES,  # 223
    SIDE_V3_EXTRAS,        # 26
    N_GLOBAL,              # 10
    STATE_V3_FEATURES,     # 2738
)


SIDE_FLAT = 6 * POKEMON_V3_FEATURES + SIDE_V3_EXTRAS  # 1364


class DeepSetValueNet(nn.Module):
    """Permutation-invariant value net over 6+6 mons.

    Input: flat 2738-dim feature vector (matches Rust/Python featurizer).
    Output: scalar logit (sigmoid for BCE training).
    """

    def __init__(self, d_emb: int = 128, hidden: int = 256, mon_hidden: int = 256):
        super().__init__()
        # +1 for the is_active flag we splice into each mon vector.
        self.mon_encoder = nn.Sequential(
            nn.Linear(POKEMON_V3_FEATURES + 1, mon_hidden), nn.ReLU(),
            nn.Linear(mon_hidden, mon_hidden), nn.ReLU(),
            nn.Linear(mon_hidden, d_emb),
        )
        # Per-side representation = concat(active_emb, mean(bench_emb)) so the
        # active mon's signal isn't diluted to 1/6 of the team. Bench remains
        # permutation-invariant.
        head_in = 2 * (2 * d_emb) + 2 * SIDE_V3_EXTRAS + N_GLOBAL
        self.head = nn.Sequential(
            nn.Linear(head_in, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, 1),
        )

    @staticmethod
    def _split(x: torch.Tensor):
        """Split flat features into structured pieces.

        x: [B, 2738]
        returns: s1_mons[B,6,223], s1_extras[B,26],
                 s2_mons[B,6,223], s2_extras[B,26], global[B,10]
        """
        s1 = x[:, :SIDE_FLAT]              # [B, 1364]
        s2 = x[:, SIDE_FLAT:2*SIDE_FLAT]   # [B, 1364]
        glb = x[:, 2*SIDE_FLAT:]           # [B, 10]
        s1_mons = s1[:, :6 * POKEMON_V3_FEATURES].reshape(-1, 6, POKEMON_V3_FEATURES)
        s1_extras = s1[:, 6 * POKEMON_V3_FEATURES:]
        s2_mons = s2[:, :6 * POKEMON_V3_FEATURES].reshape(-1, 6, POKEMON_V3_FEATURES)
        s2_extras = s2[:, 6 * POKEMON_V3_FEATURES:]
        return s1_mons, s1_extras, s2_mons, s2_extras, glb

    @staticmethod
    def _inject_active_flag(mons: torch.Tensor, extras: torch.Tensor) -> torch.Tensor:
        """Append is_active bit derived from active_idx one-hot in extras[0:6]."""
        # extras[:, 0:6] is one-hot over the 6 mons.
        active_flag = extras[:, :6].unsqueeze(-1)  # [B, 6, 1]
        return torch.cat([mons, active_flag], dim=-1)

    def _encode_side(self, mons: torch.Tensor, extras: torch.Tensor) -> torch.Tensor:
        flagged = self._inject_active_flag(mons, extras)  # [B, 6, 224]
        emb = self.mon_encoder(flagged)                    # [B, 6, d_emb]
        # Split active out so its features go through unblended.
        active_one_hot = extras[:, :6].unsqueeze(-1)       # [B, 6, 1]
        active_emb = (emb * active_one_hot).sum(dim=1)     # [B, d_emb]
        # Mean over the 5 bench mons (dead or alive — encoder sees the alive bit).
        bench_mask = 1.0 - active_one_hot                  # [B, 6, 1]
        bench_count = bench_mask.sum(dim=1).clamp(min=1.0) # [B, 1]
        bench_emb = (emb * bench_mask).sum(dim=1) / bench_count
        return torch.cat([active_emb, bench_emb], dim=-1)  # [B, 2*d_emb]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        s1_mons, s1_extras, s2_mons, s2_extras, glb = self._split(x)
        s1_pool = self._encode_side(s1_mons, s1_extras)
        s2_pool = self._encode_side(s2_mons, s2_extras)
        feat = torch.cat([s1_pool, s2_pool, s1_extras, s2_extras, glb], dim=-1)
        return self.head(feat).squeeze(-1)


def prepare_value_data(data_path: str, residual: bool = False,
                        filter_draws: bool = False,
                        outcome_labels: bool = False,
                        gamma: float = 1.0):
    """Data prep for value-net training.

    Default: per-state target = MCTS-derived V from the pickle (turn_data[1]).
    outcome_labels=True: per-state target = per-game winner (1/0/0.5 from p1's
    POV), optionally pulled toward 0.5 by gamma^turns_remaining so far-from-
    terminal states aren't pinned to ±1 (anti-saturation).
    """
    print(f"Loading {data_path}...")
    with open(data_path, "rb") as f:
        data = pickle.load(f)

    states: list[np.ndarray] = []
    labels: list[float] = []
    n_draw_games_skipped = 0

    if residual:
        # mirror value_train.py's residual baseline
        from showdown.value_train import _h_baseline

    for game in data:
        if not (isinstance(game, tuple) and len(game) == 2):
            continue
        winner, turns = game
        if filter_draws and winner == 0:
            n_draw_games_skipped += 1
            continue
        # per-game outcome label from p1's perspective
        s1_outcome = 1.0 if winner > 0 else (0.0 if winner < 0 else 0.5)
        n_turns = len(turns)
        for i, turn_data in enumerate(turns):
            if len(turn_data) < 2:
                continue
            state_str = turn_data[0]
            turns_remaining = n_turns - i  # at least 1 at the last turn
            if outcome_labels:
                # gamma-discount pulls mid-game labels toward 0.5 prior
                v = 0.5 + (s1_outcome - 0.5) * (gamma ** turns_remaining)
            else:
                v = turn_data[1] if isinstance(turn_data[1], float) else s1_outcome
            # side one perspective
            states.append(parse_state_v3(state_str))
            labels.append(v if not residual
                          else (v - _h_baseline(state_str) + 1.0) / 2.0)
            # side flip
            major = state_str.split("/")
            if len(major) >= 2:
                flipped = "/".join([major[1], major[0]] + major[2:])
                states.append(parse_state_v3(flipped))
                v_f = 1.0 - v
                labels.append(v_f if not residual
                              else (v_f - _h_baseline(flipped) + 1.0) / 2.0)

    states_arr = np.asarray(states, dtype=np.float32)
    labels_arr = np.asarray(labels, dtype=np.float32)
    print(f"  {len(states)} samples, state_dim={states_arr.shape[1]}")
    if filter_draws:
        print(f"  filtered {n_draw_games_skipped} draw games")
    if outcome_labels:
        print(f"  labels: per-game outcome × γ^turns_remaining (γ={gamma})")
    print(f"  V mean={labels_arr.mean():.3f} std={labels_arr.std():.3f}")
    return states_arr, labels_arr


def train(data_path: str, save_path: str, epochs: int, lr: float,
          hidden: int, mon_hidden: int, d_emb: int,
          batch_size: int, device: str, filter_draws: bool = False,
          outcome_labels: bool = False, gamma: float = 1.0):
    states, labels = prepare_value_data(
        data_path, filter_draws=filter_draws,
        outcome_labels=outcome_labels, gamma=gamma)
    n = len(states)
    rng = np.random.default_rng(0)
    perm = rng.permutation(n)
    states, labels = states[perm], labels[perm]
    cut = int(n * 0.95)
    X_train = torch.from_numpy(states[:cut]).to(device)
    Y_train = torch.from_numpy(labels[:cut]).to(device)
    X_val = torch.from_numpy(states[cut:]).to(device)
    Y_val = torch.from_numpy(labels[cut:]).to(device)
    print(f"  train={len(X_train)}, val={len(X_val)}")

    model = DeepSetValueNet(d_emb=d_emb, hidden=hidden,
                            mon_hidden=mon_hidden).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  {n_params:,} parameters (mon_hidden={mon_hidden}, d_emb={d_emb}, "
          f"head_hidden={hidden})")
    print(f"  V baseline (BCE on 0.5): {-np.log(0.5):.4f}")

    opt = optim.Adam(model.parameters(), lr=lr)
    sched = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)

    best_val = float("inf")
    save_path = Path(save_path)
    t0 = time.time()
    for epoch in range(epochs):
        model.train()
        perm_e = torch.randperm(len(X_train))
        total = 0.0; nb = 0
        for i in range(0, len(X_train), batch_size):
            idx = perm_e[i:i + batch_size]
            xb, yb = X_train[idx], Y_train[idx]
            opt.zero_grad()
            logits = model(xb)
            loss = F.binary_cross_entropy_with_logits(logits, yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            total += loss.item(); nb += 1
        sched.step()

        model.eval()
        with torch.no_grad():
            val_logits = model(X_val)
            val_loss = F.binary_cross_entropy_with_logits(val_logits, Y_val).item()
            probs = torch.sigmoid(val_logits)
            sat = ((probs > 0.95) | (probs < 0.05)).float().mean().item()

        marker = ""
        if val_loss < best_val:
            best_val = val_loss
            torch.save({
                "model": model.state_dict(),
                "state_dim": STATE_V3_FEATURES,
                "hidden": hidden,
                "mon_hidden": mon_hidden,
                "d_emb": d_emb,
            }, save_path)
            marker = " *best*"
        print(f"  Epoch {epoch + 1:2d}: train={total/max(nb,1):.4f} "
              f"val={val_loss:.4f} sat={sat:.1%}{marker}")

    print(f"\nBest val: {best_val:.4f} ({time.time()-t0:.0f}s)")
    print(f"  saved: {save_path}")

    # ONNX export: keep input/output API identical to existing ValueNet ONNXs
    onnx_path = save_path.with_suffix(".onnx")
    ckpt = torch.load(save_path, weights_only=True)
    export_model = DeepSetValueNet(d_emb=ckpt["d_emb"],
                                    hidden=ckpt["hidden"],
                                    mon_hidden=ckpt["mon_hidden"])
    export_model.load_state_dict(ckpt["model"])
    export_model.eval()
    dummy = torch.randn(1, ckpt["state_dim"])
    torch.onnx.export(export_model, dummy, onnx_path,
                      input_names=["state"], output_names=["value"],
                      dynamic_axes={"state": {0: "batch"}, "value": {0: "batch"}},
                      dynamo=False)
    print(f"  ONNX: {onnx_path}")


def main():
    ap = argparse.ArgumentParser(description="DeepSet value-net training (gen9/v3)")
    ap.add_argument("--data", required=True)
    ap.add_argument("--model", default="value_net_deepset.pt")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--hidden", type=int, default=256, help="head MLP width")
    ap.add_argument("--mon-hidden", type=int, default=256, help="per-mon encoder width")
    ap.add_argument("--d-emb", type=int, default=128, help="per-mon embedding dim")
    ap.add_argument("--batch-size", type=int, default=512)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--filter-draws", action="store_true",
                    help="drop games with winner=0 before training (timeouts)")
    ap.add_argument("--outcome-labels", action="store_true",
                    help="use per-game winner as label instead of per-turn MCTS V")
    ap.add_argument("--gamma", type=float, default=1.0,
                    help="discount: label = 0.5 + (winner-0.5) * gamma^turns_remaining; "
                         "only takes effect with --outcome-labels")
    args = ap.parse_args()
    train(args.data, args.model, args.epochs, args.lr,
          args.hidden, args.mon_hidden, args.d_emb,
          args.batch_size, args.device,
          filter_draws=args.filter_draws,
          outcome_labels=args.outcome_labels,
          gamma=args.gamma)


if __name__ == "__main__":
    main()
