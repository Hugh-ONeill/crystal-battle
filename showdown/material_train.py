#!/usr/bin/env python3
"""Material-only value-net (gen9 / v3 features).

Sanity check: drop every move-level feature (PP + per-move features) from the
per-mon vector. The encoder only sees static board material — HP, alive,
types, stats, status, ability flags, item flags — plus side-level boosts /
hazards / weather. The model is structurally incapable of learning
spurious move-level patterns (Rapid Spin spam, type-immunity misplays) because
it can't see what moves exist.

Per mon vector: 95 features (1 hp + 1 alive + 60 types + 1 terad + 5 stats +
7 status + 1 sleep + 8 ability + 11 item).

Same DeepSet split-pool architecture as deepset_train.py (active mon
embedding kept separate from mean-pooled bench), but tiny — this should
work or nothing will.
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
    POKEMON_V3_FEATURES,  # 223 (full mon block)
    SIDE_V3_EXTRAS,        # 26
    N_GLOBAL,              # 10
    STATE_V3_FEATURES,     # 2738
)


SIDE_FLAT = 6 * POKEMON_V3_FEATURES + SIDE_V3_EXTRAS  # 1364
# core per-mon block (hp/alive/types/stats/status/abilities/items), drops PP+moves
N_MON_CORE = 1 + 1 + 20 + 20 + 20 + 1 + 5 + 7 + 1 + 8 + 11  # = 95
assert N_MON_CORE == 95


class MaterialValueNet(nn.Module):
    """Material-only value net. Move features sliced out before encoding."""

    def __init__(self, d_emb: int = 64, hidden: int = 128, mon_hidden: int = 128):
        super().__init__()
        self.mon_encoder = nn.Sequential(
            nn.Linear(N_MON_CORE + 1, mon_hidden), nn.ReLU(),  # +1: is_active flag
            nn.Linear(mon_hidden, mon_hidden), nn.ReLU(),
            nn.Linear(mon_hidden, d_emb),
        )
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
        s1_full = s1[:, :6 * POKEMON_V3_FEATURES].reshape(-1, 6, POKEMON_V3_FEATURES)
        s2_full = s2[:, :6 * POKEMON_V3_FEATURES].reshape(-1, 6, POKEMON_V3_FEATURES)
        # Slice off PP + move features per mon.
        s1_mons = s1_full[:, :, :N_MON_CORE]
        s2_mons = s2_full[:, :, :N_MON_CORE]
        s1_extras = s1[:, 6 * POKEMON_V3_FEATURES:]
        s2_extras = s2[:, 6 * POKEMON_V3_FEATURES:]
        return s1_mons, s1_extras, s2_mons, s2_extras, glb

    def _encode_side(self, mons: torch.Tensor, extras: torch.Tensor) -> torch.Tensor:
        active_one_hot = extras[:, :6].unsqueeze(-1)       # [B, 6, 1]
        flagged = torch.cat([mons, active_one_hot], dim=-1)
        emb = self.mon_encoder(flagged)                    # [B, 6, d_emb]
        active_emb = (emb * active_one_hot).sum(dim=1)
        bench_mask = 1.0 - active_one_hot
        bench_count = bench_mask.sum(dim=1).clamp(min=1.0)
        bench_emb = (emb * bench_mask).sum(dim=1) / bench_count
        return torch.cat([active_emb, bench_emb], dim=-1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        s1_mons, s1_extras, s2_mons, s2_extras, glb = self._split(x)
        s1_pool = self._encode_side(s1_mons, s1_extras)
        s2_pool = self._encode_side(s2_mons, s2_extras)
        feat = torch.cat([s1_pool, s2_pool, s1_extras, s2_extras, glb], dim=-1)
        return self.head(feat).squeeze(-1)


def prepare_value_data(data_path: str, filter_draws: bool = False,
                        outcome_labels: bool = False, gamma: float = 1.0):
    print(f"Loading {data_path}...")
    with open(data_path, "rb") as f:
        data = pickle.load(f)

    states: list[np.ndarray] = []
    labels: list[float] = []
    n_draw_games_skipped = 0

    for game in data:
        if not (isinstance(game, tuple) and len(game) == 2):
            continue
        winner, turns = game
        if filter_draws and winner == 0:
            n_draw_games_skipped += 1
            continue
        s1_outcome = 1.0 if winner > 0 else (0.0 if winner < 0 else 0.5)
        n_turns = len(turns)
        for i, turn_data in enumerate(turns):
            if len(turn_data) < 1:
                continue
            state_str = turn_data[0]
            turns_remaining = n_turns - i
            if outcome_labels:
                # With --outcome-labels we ignore any per-turn V and label
                # purely from the game outcome — so replay-derived data
                # (which has only state_str per turn, no V) works too.
                v = 0.5 + (s1_outcome - 0.5) * (gamma ** turns_remaining)
            elif len(turn_data) >= 2 and isinstance(turn_data[1], float):
                v = turn_data[1]
            else:
                v = s1_outcome
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

    model = MaterialValueNet(d_emb=d_emb, hidden=hidden,
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

    onnx_path = save_path.with_suffix(".onnx")
    ckpt = torch.load(save_path, weights_only=True)
    export_model = MaterialValueNet(d_emb=ckpt["d_emb"],
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


def export_material_bin(pt_path: Path, bin_path: Path):
    """Write trained MaterialValueNet weights as a packed binary file.

    Format:
      [4 bytes] magic = b"MAT1"
      [u32 LE × 6] mon_hidden, d_emb, hidden, n_mon_core(=95),
                    side_extras(=26), n_global(=10)
      6 layer blocks, each:
        [u32 LE × 2] out_dim, in_dim
        [f32 LE × (out*in)] weights, row-major (matches PyTorch nn.Linear)
        [f32 LE × out]      bias
      Layer order:
        mon_encoder.0, mon_encoder.2, mon_encoder.4,
        head.0,        head.2,        head.4
    """
    import struct
    ckpt = torch.load(pt_path, weights_only=True)
    sd = ckpt["model"]
    mon_hidden = ckpt["mon_hidden"]
    d_emb = ckpt["d_emb"]
    hidden = ckpt["hidden"]

    layer_keys = [
        ("mon_encoder.0.weight", "mon_encoder.0.bias"),
        ("mon_encoder.2.weight", "mon_encoder.2.bias"),
        ("mon_encoder.4.weight", "mon_encoder.4.bias"),
        ("head.0.weight",        "head.0.bias"),
        ("head.2.weight",        "head.2.bias"),
        ("head.4.weight",        "head.4.bias"),
    ]
    with open(bin_path, "wb") as f:
        f.write(b"MAT1")
        f.write(struct.pack("<6I", mon_hidden, d_emb, hidden,
                            N_MON_CORE, SIDE_V3_EXTRAS, N_GLOBAL))
        for w_key, b_key in layer_keys:
            W = sd[w_key].cpu().numpy().astype(np.float32)
            b = sd[b_key].cpu().numpy().astype(np.float32)
            out_dim, in_dim = W.shape
            f.write(struct.pack("<2I", out_dim, in_dim))
            f.write(W.tobytes(order="C"))
            f.write(b.tobytes())
    print(f"  exported: {bin_path} ({bin_path.stat().st_size} bytes)")


def main():
    ap = argparse.ArgumentParser(description="Material-only value-net (gen9/v3)")
    ap.add_argument("--data", required=False)
    ap.add_argument("--model", default="value_net_material.pt")
    ap.add_argument("--export-only", action="store_true",
                    help="skip training; just export an existing .pt to .material")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--hidden", type=int, default=128)
    ap.add_argument("--mon-hidden", type=int, default=128)
    ap.add_argument("--d-emb", type=int, default=64)
    ap.add_argument("--batch-size", type=int, default=512)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--filter-draws", action="store_true")
    ap.add_argument("--outcome-labels", action="store_true")
    ap.add_argument("--gamma", type=float, default=1.0)
    args = ap.parse_args()
    if not args.export_only:
        if not args.data:
            ap.error("--data required unless --export-only")
        train(args.data, args.model, args.epochs, args.lr,
              args.hidden, args.mon_hidden, args.d_emb,
              args.batch_size, args.device,
              filter_draws=args.filter_draws,
              outcome_labels=args.outcome_labels,
              gamma=args.gamma)
    pt_path = Path(args.model)
    bin_path = pt_path.with_suffix(".material")
    export_material_bin(pt_path, bin_path)


if __name__ == "__main__":
    main()
