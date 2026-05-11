#!/usr/bin/env python3
"""Cross-side self-attention value-net training (gen9 / v3 features).

Extends the DeepSet idea by adding attention across the full 12-mon set,
so each mon's contextualized representation can incorporate cross-side
matchup information (my-active vs opp-active is the most important
interaction for V; pure pooling makes that implicit at best).

Architecture:
  per-mon encoder    (224 → mon_hidden → d_emb)         shared for all 12 mons
  side embedding     (2-dim → d_emb)                    + my/opp flag
  attn block         n_attn_layers × TransformerEncoder over 12 tokens
  split pool         (active, mean(bench)) per side
  head               concat(both sides, extras, global) → MLP → V

Trains as a single-output value net so it slots into the existing
`pe.ValueNet` ONNX path; pair with `policy_net_az.pt` at bench time.
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
    POKEMON_V3_FEATURES,
    SIDE_V3_EXTRAS,
    N_GLOBAL,
    STATE_V3_FEATURES,
)


SIDE_FLAT = 6 * POKEMON_V3_FEATURES + SIDE_V3_EXTRAS  # 1364


class AttnValueNet(nn.Module):
    def __init__(self, d_emb: int = 128, hidden: int = 256, mon_hidden: int = 256,
                 n_heads: int = 4, n_attn_layers: int = 2):
        super().__init__()
        self.d_emb = d_emb
        self.mon_encoder = nn.Sequential(
            nn.Linear(POKEMON_V3_FEATURES + 1, mon_hidden), nn.ReLU(),
            nn.Linear(mon_hidden, mon_hidden), nn.ReLU(),
            nn.Linear(mon_hidden, d_emb),
        )
        # side bit injected as a learned vector added to the per-mon embedding
        self.side_emb = nn.Embedding(2, d_emb)
        self.attn_layers = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=d_emb, nhead=n_heads,
                dim_feedforward=2 * d_emb, dropout=0.0,
                batch_first=True, norm_first=True,
            )
            for _ in range(n_attn_layers)
        ])
        # head input: per-side (active + bench) + raw extras + global
        head_in = 2 * (2 * d_emb) + 2 * SIDE_V3_EXTRAS + N_GLOBAL
        self.head = nn.Sequential(
            nn.Linear(head_in, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, 1),
        )

    @staticmethod
    def _split(x: torch.Tensor):
        s1 = x[:, :SIDE_FLAT]
        s2 = x[:, SIDE_FLAT:2 * SIDE_FLAT]
        glb = x[:, 2 * SIDE_FLAT:]
        s1_mons = s1[:, :6 * POKEMON_V3_FEATURES].reshape(-1, 6, POKEMON_V3_FEATURES)
        s1_extras = s1[:, 6 * POKEMON_V3_FEATURES:]
        s2_mons = s2[:, :6 * POKEMON_V3_FEATURES].reshape(-1, 6, POKEMON_V3_FEATURES)
        s2_extras = s2[:, 6 * POKEMON_V3_FEATURES:]
        return s1_mons, s1_extras, s2_mons, s2_extras, glb

    @staticmethod
    def _inject_active_flag(mons: torch.Tensor, extras: torch.Tensor) -> torch.Tensor:
        active_flag = extras[:, :6].unsqueeze(-1)
        return torch.cat([mons, active_flag], dim=-1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        s1_mons, s1_extras, s2_mons, s2_extras, glb = self._split(x)
        s1_emb = self.mon_encoder(self._inject_active_flag(s1_mons, s1_extras))
        s2_emb = self.mon_encoder(self._inject_active_flag(s2_mons, s2_extras))
        # add side embeddings (broadcasts: lookup 1 vector, add to all 6)
        zero_idx = torch.zeros(1, dtype=torch.long, device=s1_emb.device)
        one_idx = torch.ones(1, dtype=torch.long, device=s1_emb.device)
        s1_emb = s1_emb + self.side_emb(zero_idx)
        s2_emb = s2_emb + self.side_emb(one_idx)
        # 12-token sequence; full self-attention each layer
        tokens = torch.cat([s1_emb, s2_emb], dim=1)        # [B, 12, d_emb]
        for layer in self.attn_layers:
            tokens = layer(tokens)
        s1_post = tokens[:, :6]
        s2_post = tokens[:, 6:]
        # split pool per side
        s1_oh = s1_extras[:, :6].unsqueeze(-1)
        s2_oh = s2_extras[:, :6].unsqueeze(-1)
        s1_active = (s1_post * s1_oh).sum(dim=1)
        s2_active = (s2_post * s2_oh).sum(dim=1)
        s1_bench_mask = 1.0 - s1_oh
        s2_bench_mask = 1.0 - s2_oh
        s1_bench = (s1_post * s1_bench_mask).sum(dim=1) \
                    / s1_bench_mask.sum(dim=1).clamp(min=1.0)
        s2_bench = (s2_post * s2_bench_mask).sum(dim=1) \
                    / s2_bench_mask.sum(dim=1).clamp(min=1.0)
        feat = torch.cat([s1_active, s1_bench, s2_active, s2_bench,
                          s1_extras, s2_extras, glb], dim=-1)
        return self.head(feat).squeeze(-1)


def prepare_value_data(data_path: str):
    print(f"Loading {data_path}...")
    with open(data_path, "rb") as f:
        data = pickle.load(f)
    states: list[np.ndarray] = []
    labels: list[float] = []
    for game in data:
        if not (isinstance(game, tuple) and len(game) == 2):
            continue
        winner, turns = game
        for turn_data in turns:
            if len(turn_data) < 2:
                continue
            state_str = turn_data[0]
            v = turn_data[1] if isinstance(turn_data[1], float) else (
                1.0 if winner > 0 else (0.0 if winner < 0 else 0.5))
            states.append(parse_state_v3(state_str))
            labels.append(v)
            major = state_str.split("/")
            if len(major) >= 2:
                flipped = "/".join([major[1], major[0]] + major[2:])
                states.append(parse_state_v3(flipped))
                labels.append(1.0 - v)
    states_arr = np.asarray(states, dtype=np.float32)
    labels_arr = np.asarray(labels, dtype=np.float32)
    print(f"  {len(states)} samples, state_dim={states_arr.shape[1]}")
    return states_arr, labels_arr


def train(data_path, save_path, epochs, lr, hidden, mon_hidden, d_emb,
          n_heads, n_attn_layers, batch_size, device):
    states, labels = prepare_value_data(data_path)
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

    model = AttnValueNet(d_emb=d_emb, hidden=hidden, mon_hidden=mon_hidden,
                         n_heads=n_heads, n_attn_layers=n_attn_layers).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  {n_params:,} params  (mon_hidden={mon_hidden}, d_emb={d_emb}, "
          f"head={hidden}, heads={n_heads}, attn_layers={n_attn_layers})")

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
                "hidden": hidden, "mon_hidden": mon_hidden, "d_emb": d_emb,
                "n_heads": n_heads, "n_attn_layers": n_attn_layers,
            }, save_path)
            marker = " *best*"
        print(f"  Epoch {epoch + 1:2d}: train={total/max(nb,1):.4f} "
              f"val={val_loss:.4f} sat={sat:.1%}{marker}")

    print(f"\nBest val: {best_val:.4f} ({time.time()-t0:.0f}s)")
    print(f"  saved: {save_path}")

    onnx_path = save_path.with_suffix(".onnx")
    ckpt = torch.load(save_path, weights_only=True)
    export_model = AttnValueNet(
        d_emb=ckpt["d_emb"], hidden=ckpt["hidden"],
        mon_hidden=ckpt["mon_hidden"],
        n_heads=ckpt["n_heads"], n_attn_layers=ckpt["n_attn_layers"])
    export_model.load_state_dict(ckpt["model"])
    export_model.eval()
    dummy = torch.randn(1, ckpt["state_dim"])
    torch.onnx.export(export_model, dummy, onnx_path,
                      input_names=["state"], output_names=["value"],
                      dynamic_axes={"state": {0: "batch"}, "value": {0: "batch"}},
                      dynamo=False)
    print(f"  ONNX: {onnx_path}")


def main():
    ap = argparse.ArgumentParser(description="Cross-side attention value net (gen9/v3)")
    ap.add_argument("--data", required=True)
    ap.add_argument("--model", default="value_net_attn.pt")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--hidden", type=int, default=256)
    ap.add_argument("--mon-hidden", type=int, default=256)
    ap.add_argument("--d-emb", type=int, default=128)
    ap.add_argument("--n-heads", type=int, default=4)
    ap.add_argument("--n-attn-layers", type=int, default=2)
    ap.add_argument("--batch-size", type=int, default=512)
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()
    train(args.data, args.model, args.epochs, args.lr,
          args.hidden, args.mon_hidden, args.d_emb,
          args.n_heads, args.n_attn_layers,
          args.batch_size, args.device)


if __name__ == "__main__":
    main()
