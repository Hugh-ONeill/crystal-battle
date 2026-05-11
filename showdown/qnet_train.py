#!/usr/bin/env python3
"""Q-net training (gen9 / v3 features). Action-conditioned value learning.

Each turn in the polidist pickle has:
  - state_str
  - V (scalar MCTS-derived V; unused here)
  - s1_pi9: side-one MCTS visit distribution over 9 actions
  - s2_pi9: side-two MCTS visit distribution over 9 actions

The Q-net predicts P(win | state, action) for each of the 9 actions. To
train per-action, we sample one action per turn from the side's visit
distribution (matching what self-play actually did at that turn) and apply
a masked BCE loss only at that action's output index.

This fixes the failure mode we've seen across V-net architectures: V was
learning team-level "this team won → all its moves are good" patterns and
couldn't condition on state. With masked Q labels, each action's Q only
gets gradient when that action was actually played — so e.g. (no-hazard
state, Rapid Spin played, lost) directly pulls Q[spin_idx] down for that
state without forcing Q[other_idx] to also drop.

Action layout (matches Rust policy::map_priors_to_options):
  [0..3] move slots 0,1,2,3
  [4..8] switches in dense ordering over 5 non-active bench mons

At inference, the Rust MaterialNet auto-detects N_OUT=9 from the .material
file's final layer and returns max(Q) — the value of playing the
estimated-best move.
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


SIDE_FLAT = 6 * POKEMON_V3_FEATURES + SIDE_V3_EXTRAS
N_MON_CORE = 1 + 1 + 20 + 20 + 20 + 1 + 5 + 7 + 1 + 8 + 11  # = 95
N_ACTIONS = 9
assert N_MON_CORE == 95


class QNet(nn.Module):
    """Material-style backbone with a 9-output Q head.

    Identical structure to MaterialValueNet (material_train.py) except the
    final Linear emits 9 logits instead of 1. The .material export format
    handles this transparently — only the last layer's out_dim changes.
    """

    def __init__(self, d_emb: int = 64, hidden: int = 128, mon_hidden: int = 128):
        super().__init__()
        self.mon_encoder = nn.Sequential(
            nn.Linear(N_MON_CORE + 1, mon_hidden), nn.ReLU(),
            nn.Linear(mon_hidden, mon_hidden), nn.ReLU(),
            nn.Linear(mon_hidden, d_emb),
        )
        head_in = 2 * (2 * d_emb) + 2 * SIDE_V3_EXTRAS + N_GLOBAL
        self.head = nn.Sequential(
            nn.Linear(head_in, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, N_ACTIONS),
        )

    @staticmethod
    def _split(x: torch.Tensor):
        s1 = x[:, :SIDE_FLAT]
        s2 = x[:, SIDE_FLAT:2 * SIDE_FLAT]
        glb = x[:, 2 * SIDE_FLAT:]
        s1_full = s1[:, :6 * POKEMON_V3_FEATURES].reshape(-1, 6, POKEMON_V3_FEATURES)
        s2_full = s2[:, :6 * POKEMON_V3_FEATURES].reshape(-1, 6, POKEMON_V3_FEATURES)
        s1_mons = s1_full[:, :, :N_MON_CORE]
        s2_mons = s2_full[:, :, :N_MON_CORE]
        s1_extras = s1[:, 6 * POKEMON_V3_FEATURES:]
        s2_extras = s2[:, 6 * POKEMON_V3_FEATURES:]
        return s1_mons, s1_extras, s2_mons, s2_extras, glb

    def _encode_side(self, mons: torch.Tensor, extras: torch.Tensor) -> torch.Tensor:
        active_one_hot = extras[:, :6].unsqueeze(-1)
        flagged = torch.cat([mons, active_one_hot], dim=-1)
        emb = self.mon_encoder(flagged)
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
        return self.head(feat)  # [B, 9]


def prepare_qnet_data(data_path: str, filter_draws: bool = False, gamma: float = 0.97,
                       rng_seed: int = 0):
    """Per-turn samples with action, outcome AND full visit distribution.

    The visit distribution is kept so we can apply an AlphaZero-style policy
    auxiliary loss: forces softmax(Q) to match visit_dist, which prevents
    the 9 Q outputs from collapsing to the same value (a problem we observed
    with value-only masked-BCE training — Q-std stuck at ~0.02).
    """
    print(f"Loading {data_path}...")
    with open(data_path, "rb") as f:
        data = pickle.load(f)

    rng = np.random.default_rng(rng_seed)
    states: list[np.ndarray] = []
    actions: list[int] = []
    targets: list[float] = []
    pi_dists: list[np.ndarray] = []
    n_draw_games_skipped = 0
    n_turns_skipped = 0

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
            if len(turn_data) < 4:
                n_turns_skipped += 1
                continue
            state_str = turn_data[0]
            s1_pi = np.asarray(turn_data[2], dtype=np.float32)
            s2_pi = np.asarray(turn_data[3], dtype=np.float32)
            if s1_pi.shape != (N_ACTIONS,) or s2_pi.shape != (N_ACTIONS,):
                n_turns_skipped += 1
                continue
            turns_remaining = n_turns - i
            v_s1 = 0.5 + (s1_outcome - 0.5) * (gamma ** turns_remaining)
            v_s2 = 1.0 - v_s1

            s1_sum = s1_pi.sum()
            if s1_sum > 0:
                pi_norm = s1_pi / s1_sum
                p1_a = int(rng.choice(N_ACTIONS, p=pi_norm))
                states.append(parse_state_v3(state_str))
                actions.append(p1_a)
                targets.append(v_s1)
                pi_dists.append(pi_norm)

            s2_sum = s2_pi.sum()
            if s2_sum > 0:
                major = state_str.split("/")
                if len(major) >= 2:
                    flipped = "/".join([major[1], major[0]] + major[2:])
                    pi_norm = s2_pi / s2_sum
                    p2_a = int(rng.choice(N_ACTIONS, p=pi_norm))
                    states.append(parse_state_v3(flipped))
                    actions.append(p2_a)
                    targets.append(v_s2)
                    pi_dists.append(pi_norm)

    states_arr = np.asarray(states, dtype=np.float32)
    actions_arr = np.asarray(actions, dtype=np.int64)
    targets_arr = np.asarray(targets, dtype=np.float32)
    pi_arr = np.asarray(pi_dists, dtype=np.float32)
    print(f"  {len(states)} samples, state_dim={states_arr.shape[1]}")
    if filter_draws:
        print(f"  filtered {n_draw_games_skipped} draw games, {n_turns_skipped} bad turns")
    counts = np.bincount(actions_arr, minlength=N_ACTIONS)
    print(f"  action counts (idx 0-8): {counts.tolist()}")
    print(f"  Q target mean={targets_arr.mean():.3f} std={targets_arr.std():.3f}")
    print(f"  visit-dist mean entropy: "
          f"{(-(pi_arr * np.log(pi_arr + 1e-9)).sum(axis=1)).mean():.3f} "
          f"(uniform = {np.log(N_ACTIONS):.3f})")
    return states_arr, actions_arr, targets_arr, pi_arr


def train(data_path: str, save_path: str, epochs: int, lr: float,
          hidden: int, mon_hidden: int, d_emb: int,
          batch_size: int, device: str, filter_draws: bool, gamma: float,
          policy_loss_weight: float = 1.0):
    states, actions, targets, pi_dists = prepare_qnet_data(
        data_path, filter_draws=filter_draws, gamma=gamma)
    n = len(states)
    rng = np.random.default_rng(0)
    perm = rng.permutation(n)
    states, actions, targets, pi_dists = (
        states[perm], actions[perm], targets[perm], pi_dists[perm])
    cut = int(n * 0.95)
    X_tr = torch.from_numpy(states[:cut]).to(device)
    A_tr = torch.from_numpy(actions[:cut]).to(device)
    Y_tr = torch.from_numpy(targets[:cut]).to(device)
    P_tr = torch.from_numpy(pi_dists[:cut]).to(device)
    X_va = torch.from_numpy(states[cut:]).to(device)
    A_va = torch.from_numpy(actions[cut:]).to(device)
    Y_va = torch.from_numpy(targets[cut:]).to(device)
    P_va = torch.from_numpy(pi_dists[cut:]).to(device)
    print(f"  train={len(X_tr)}, val={len(X_va)}, policy_loss_weight={policy_loss_weight}")

    model = QNet(d_emb=d_emb, hidden=hidden, mon_hidden=mon_hidden).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  {n_params:,} parameters (mon_hidden={mon_hidden}, d_emb={d_emb}, "
          f"head_hidden={hidden})")
    print(f"  Q baseline (BCE on 0.5): {-np.log(0.5):.4f}")

    opt = optim.Adam(model.parameters(), lr=lr)
    sched = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)

    best_val = float("inf")
    save_path = Path(save_path)
    t0 = time.time()
    arange_bs = None  # cached arange tensor for the largest seen batch
    for epoch in range(epochs):
        model.train()
        perm_e = torch.randperm(len(X_tr))
        total_v = 0.0; total_p = 0.0; nb = 0
        for i in range(0, len(X_tr), batch_size):
            idx = perm_e[i:i + batch_size]
            xb, ab, yb, pb = X_tr[idx], A_tr[idx], Y_tr[idx], P_tr[idx]
            bs = xb.shape[0]
            opt.zero_grad()
            logits = model(xb)                     # [B, 9]
            # Value loss: BCE on played action's sigmoid(Q) vs outcome.
            played_logits = logits[torch.arange(bs, device=device), ab]
            v_loss = F.binary_cross_entropy_with_logits(played_logits, yb)
            # Policy aux loss: cross-entropy between softmax(Q) and visit_dist.
            # Forces the 9 Q outputs to spread according to the empirical
            # policy — prevents the collapse where all Q outputs converge.
            log_pi = F.log_softmax(logits, dim=-1)
            p_loss = -(pb * log_pi).sum(dim=-1).mean()
            loss = v_loss + policy_loss_weight * p_loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            total_v += v_loss.item(); total_p += p_loss.item(); nb += 1
        sched.step()

        model.eval()
        with torch.no_grad():
            val_logits = model(X_va)
            val_played = val_logits[torch.arange(len(A_va), device=device), A_va]
            val_v_loss = F.binary_cross_entropy_with_logits(val_played, Y_va).item()
            val_log_pi = F.log_softmax(val_logits, dim=-1)
            val_p_loss = -(P_va * val_log_pi).sum(dim=-1).mean().item()
            val_loss = val_v_loss + policy_loss_weight * val_p_loss
            probs = torch.sigmoid(val_played)
            sat = ((probs > 0.95) | (probs < 0.05)).float().mean().item()
            all_q = torch.sigmoid(val_logits)
            q_std = all_q.std(dim=1).mean().item()

        marker = ""
        if val_loss < best_val:
            best_val = val_loss
            torch.save({
                "model": model.state_dict(),
                "state_dim": STATE_V3_FEATURES,
                "hidden": hidden,
                "mon_hidden": mon_hidden,
                "d_emb": d_emb,
                "n_actions": N_ACTIONS,
            }, save_path)
            marker = " *best*"
        print(f"  Epoch {epoch + 1:2d}: train_v={total_v/max(nb,1):.4f} "
              f"train_p={total_p/max(nb,1):.4f} val_v={val_v_loss:.4f} "
              f"val_p={val_p_loss:.4f} Q-std={q_std:.3f} sat={sat:.1%}{marker}")

    print(f"\nBest val: {best_val:.4f} ({time.time()-t0:.0f}s)")
    print(f"  saved: {save_path}")
    return save_path


def export_material_bin(pt_path: Path, bin_path: Path):
    """Same .material binary format as MaterialValueNet, but the final
    head layer has 9 outputs instead of 1. The Rust loader detects this
    from layer headers and switches to Q-net inference mode (max over 9)."""
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
        ("head.4.weight",        "head.4.bias"),  # out_dim = 9 for Q-net
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
    ap = argparse.ArgumentParser(description="Q-net training (gen9 / v3)")
    ap.add_argument("--data", required=False)
    ap.add_argument("--model", default="value_net_qnet.pt")
    ap.add_argument("--export-only", action="store_true")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--hidden", type=int, default=128)
    ap.add_argument("--mon-hidden", type=int, default=128)
    ap.add_argument("--d-emb", type=int, default=64)
    ap.add_argument("--batch-size", type=int, default=512)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--filter-draws", action="store_true")
    ap.add_argument("--gamma", type=float, default=0.97)
    ap.add_argument("--policy-loss-weight", type=float, default=1.0,
                    help="weight on the policy aux loss (CE between softmax(Q) "
                         "and visit_dist). 0 disables. 1.0 = same scale as V loss.")
    args = ap.parse_args()
    if not args.export_only:
        if not args.data:
            ap.error("--data required unless --export-only")
        train(args.data, args.model, args.epochs, args.lr,
              args.hidden, args.mon_hidden, args.d_emb,
              args.batch_size, args.device,
              filter_draws=args.filter_draws, gamma=args.gamma,
              policy_loss_weight=args.policy_loss_weight)
    pt_path = Path(args.model)
    bin_path = pt_path.with_suffix(".material")
    export_material_bin(pt_path, bin_path)


if __name__ == "__main__":
    main()
