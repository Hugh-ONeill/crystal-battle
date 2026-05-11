#!/usr/bin/env python3
"""Move-attention value net (gen9 / v3 features).

Builds on the cross-side attn architecture by ALSO encoding moves with a
shared per-move encoder, and letting each move CROSS-ATTEND to the opponent's
mon embeddings to score situational utility ("Earthquake against your team
specifically" rather than "Earthquake in the abstract").

Pipeline:
  1. Slice per-mon: 99-dim mon-state + 4 × 32-dim moves (31 move-feats + PP).
  2. Pre-encode each mon (mon-state + active flag) → mon_pre_emb [B, 12, d_pre].
  3. Encode each move (32 dims) → move_emb [B, 12, 4, d_move].
  4. Cross-attention:
        s1 moves attend to s2 mon_pre_emb
        s2 moves attend to s1 mon_pre_emb
     → move_ctx_emb (matchup-aware move scoring).
  5. Pool 4 moves per mon (mean) → move_set_emb [B, 12, d_move].
  6. Combine mon_pre_emb + move_set_emb → mon_emb [B, 12, d_emb].
  7. Add side embeddings + cross-side self-attention layers (same as attn).
  8. Split pool (active vs bench) per side → head MLP → V.

Pair with `policy_net_az.pt` at bench time via existing infra.
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
    POKEMON_V3_FEATURES,        # 223
    SIDE_V3_EXTRAS,              # 26
    N_GLOBAL,                    # 10
    STATE_V3_FEATURES,           # 2738
    N_MOVE_FEATS,                # 31
)


SIDE_FLAT = 6 * POKEMON_V3_FEATURES + SIDE_V3_EXTRAS  # 1364

# Per-mon layout (matches features_v3._parse_pokemon order):
#   [0..95):     mon-state (hp, alive, cur_types, base_types, tera_type, terad,
#                stats, status, sleep, ability_flags, item_flags)
#   [95..99):    4 PP fractions (one per move slot)
#   [99..223):   4 × 31 move features
# core = 1+1+20+20+20+1+5+7+1+8+11 = 95
N_MON_CORE = 95                        # mon-state without PP & moves
N_PP = 4
N_MOVE_FULL = 1 + N_MOVE_FEATS         # 32 = PP fraction + 31 features
assert N_MON_CORE + N_PP + 4 * N_MOVE_FEATS == POKEMON_V3_FEATURES, \
    f"layout mismatch: {N_MON_CORE + N_PP + 4 * N_MOVE_FEATS} != {POKEMON_V3_FEATURES}"


class MoveAttnValueNet(nn.Module):
    def __init__(self, d_pre: int = 64, d_move: int = 64, d_emb: int = 128,
                 hidden: int = 256, mon_hidden: int = 128,
                 n_heads: int = 4, n_attn_layers: int = 2):
        super().__init__()
        self.d_pre = d_pre
        self.d_move = d_move
        self.d_emb = d_emb

        # Pre-encoder: per-mon "static" state (no moves) → d_pre
        self.state_encoder = nn.Sequential(
            nn.Linear(N_MON_CORE + 1, mon_hidden), nn.ReLU(),  # +1 for active flag
            nn.Linear(mon_hidden, d_pre),
        )
        # Move encoder: per-move features + PP + OWN side extras → d_move
        # The own-side extras (26 dims) directly inform the move encoder about
        # hazard counts, screens, force flags etc. — fixes the "Rapid Spin
        # on empty field" bug where the model couldn't see its own hazards.
        self.move_encoder = nn.Sequential(
            nn.Linear(N_MOVE_FULL + SIDE_V3_EXTRAS, mon_hidden), nn.ReLU(),
            nn.Linear(mon_hidden, d_move),
        )
        # Cross-attention: separate K/V projections for opp vs own mons; the
        # move queries against BOTH sides so it can score for type matchup
        # (opp-mon cross-attention) and team synergy (own-mon cross-attention).
        self.kv_proj_opp = nn.Linear(d_pre, d_move)
        self.kv_proj_own = nn.Linear(d_pre, d_move)
        self.move_xattn_opp = nn.MultiheadAttention(
            embed_dim=d_move, num_heads=n_heads,
            batch_first=True, dropout=0.0,
        )
        self.move_xattn_own = nn.MultiheadAttention(
            embed_dim=d_move, num_heads=n_heads,
            batch_first=True, dropout=0.0,
        )
        # Combiner: mon_pre_emb + (opp move-ctx) + (own move-ctx) → d_emb
        self.mon_combiner = nn.Sequential(
            nn.Linear(d_pre + 2 * d_move, mon_hidden), nn.ReLU(),
            nn.Linear(mon_hidden, d_emb),
        )
        # Side embeddings + cross-side self-attention (same as attn arch)
        self.side_emb = nn.Embedding(2, d_emb)
        self.attn_layers = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=d_emb, nhead=n_heads,
                dim_feedforward=2 * d_emb, dropout=0.0,
                batch_first=True, norm_first=True,
            )
            for _ in range(n_attn_layers)
        ])
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
    def _split_per_mon(mons: torch.Tensor):
        """Split [B, 6, 223] into (core[B,6,99], moves[B,6,4,32])."""
        core = mons[..., :N_MON_CORE]                               # [B,6,99]
        pp = mons[..., N_MON_CORE:N_MON_CORE + N_PP]                # [B,6,4]
        move_feats = mons[..., N_MON_CORE + N_PP:].reshape(
            *mons.shape[:-1], N_PP, N_MOVE_FEATS)                   # [B,6,4,31]
        # attach PP to its move so the move-encoder sees it
        moves = torch.cat([pp.unsqueeze(-1), move_feats], dim=-1)   # [B,6,4,32]
        return core, moves

    def _encode_mons(self, my_mons: torch.Tensor, my_extras: torch.Tensor,
                     opp_mons: torch.Tensor):
        """Encode 6 mons with move-cross-attention to opp's pre-embeddings.

        my_mons: [B, 6, 223], my_extras: [B, 26], opp_mons: [B, 6, 223]
        returns: [B, 6, d_emb]
        """
        B = my_mons.size(0)
        # 1. core + active flag
        my_core, my_moves = self._split_per_mon(my_mons)            # [B,6,99] / [B,6,4,32]
        active_flag = my_extras[:, :6].unsqueeze(-1)                # [B, 6, 1]
        my_state = torch.cat([my_core, active_flag], dim=-1)        # [B,6,96]
        my_pre = self.state_encoder(my_state)                       # [B, 6, d_pre]

        # 2. Move encoder: per-move features + own side extras (so each move
        # encoder sees hazards/screens/force-flags directly).
        extras_per_move = my_extras.unsqueeze(1).unsqueeze(1) \
                                   .expand(B, 6, 4, SIDE_V3_EXTRAS)
        moves_with_ctx = torch.cat([my_moves, extras_per_move], dim=-1)  # [B,6,4,58]
        move_emb = self.move_encoder(moves_with_ctx)                # [B, 6, 4, d_move]

        # 3. Build OPP pre-embeddings (used as K/V for opp cross-attention)
        opp_core, _ = self._split_per_mon(opp_mons)
        opp_state = torch.cat(
            [opp_core, torch.zeros(B, 6, 1, device=my_mons.device)],  # opp active flag unknown here
            dim=-1)
        opp_pre = self.state_encoder(opp_state)                     # [B, 6, d_pre]

        # 4. Two cross-attentions: moves query against opp mons (type matchup)
        # and own mons (team synergy: pivot targets, hazard recipients, etc.).
        opp_kv = self.kv_proj_opp(opp_pre)                          # [B, 6, d_move]
        own_kv = self.kv_proj_own(my_pre)                           # [B, 6, d_move]
        moves_q = move_emb.reshape(B * 6, 4, self.d_move)           # [B*6, 4, d_move]
        # Each mon's 4 moves attend to ALL 6 opp / own mons.
        opp_kv_rep = opp_kv.unsqueeze(1).expand(B, 6, 6, self.d_move) \
                            .reshape(B * 6, 6, self.d_move)
        own_kv_rep = own_kv.unsqueeze(1).expand(B, 6, 6, self.d_move) \
                            .reshape(B * 6, 6, self.d_move)
        ctx_opp, _ = self.move_xattn_opp(moves_q, opp_kv_rep, opp_kv_rep,
                                          need_weights=False)
        ctx_own, _ = self.move_xattn_own(moves_q, own_kv_rep, own_kv_rep,
                                          need_weights=False)
        ctx_opp = ctx_opp.reshape(B, 6, 4, self.d_move)
        ctx_own = ctx_own.reshape(B, 6, 4, self.d_move)

        # 5. Mean-pool over the 4 moves per mon (both contexts)
        move_set_opp = ctx_opp.mean(dim=2)                          # [B, 6, d_move]
        move_set_own = ctx_own.mean(dim=2)                          # [B, 6, d_move]

        # 6. Combine pre-emb + opp move-ctx + own move-ctx → mon emb
        mon_emb = self.mon_combiner(
            torch.cat([my_pre, move_set_opp, move_set_own], dim=-1))
        return mon_emb

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        s1_mons, s1_extras, s2_mons, s2_extras, glb = self._split(x)
        s1_emb = self._encode_mons(s1_mons, s1_extras, s2_mons)
        s2_emb = self._encode_mons(s2_mons, s2_extras, s1_mons)

        zero_idx = torch.zeros(1, dtype=torch.long, device=s1_emb.device)
        one_idx = torch.ones(1, dtype=torch.long, device=s1_emb.device)
        s1_emb = s1_emb + self.side_emb(zero_idx)
        s2_emb = s2_emb + self.side_emb(one_idx)

        tokens = torch.cat([s1_emb, s2_emb], dim=1)
        for layer in self.attn_layers:
            tokens = layer(tokens)
        s1_post = tokens[:, :6]
        s2_post = tokens[:, 6:]
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


def prepare_value_data(data_path: str, filter_draws: bool = False,
                        outcome_labels: bool = False, gamma: float = 1.0):
    """Data prep for moveattn value-net training.

    Default: per-state target = MCTS-derived V (turn_data[1]).
    outcome_labels=True: per-state target = per-game winner (1/0/0.5 from p1),
    optionally γ-discounted by turns_remaining (anti-saturation).
    """
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
            if len(turn_data) < 2:
                continue
            state_str = turn_data[0]
            turns_remaining = n_turns - i
            if outcome_labels:
                v = 0.5 + (s1_outcome - 0.5) * (gamma ** turns_remaining)
            else:
                v = turn_data[1] if isinstance(turn_data[1], float) else s1_outcome
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
        print(f"  labels: per-game outcome × gamma^turns_remaining (gamma={gamma})")
    print(f"  V mean={labels_arr.mean():.3f} std={labels_arr.std():.3f}")
    return states_arr, labels_arr


def train(data_path, save_path, epochs, lr, hidden, mon_hidden, d_emb, d_pre, d_move,
          n_heads, n_attn_layers, batch_size, device,
          filter_draws=False, outcome_labels=False, gamma=1.0):
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

    model = MoveAttnValueNet(
        d_pre=d_pre, d_move=d_move, d_emb=d_emb,
        hidden=hidden, mon_hidden=mon_hidden,
        n_heads=n_heads, n_attn_layers=n_attn_layers).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  {n_params:,} params  (d_pre={d_pre}, d_move={d_move}, d_emb={d_emb}, "
          f"heads={n_heads}, attn_layers={n_attn_layers})")

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
                "hidden": hidden, "mon_hidden": mon_hidden,
                "d_emb": d_emb, "d_pre": d_pre, "d_move": d_move,
                "n_heads": n_heads, "n_attn_layers": n_attn_layers,
            }, save_path)
            marker = " *best*"
        print(f"  Epoch {epoch + 1:2d}: train={total/max(nb,1):.4f} "
              f"val={val_loss:.4f} sat={sat:.1%}{marker}")

    print(f"\nBest val: {best_val:.4f} ({time.time()-t0:.0f}s)")
    print(f"  saved: {save_path}")

    onnx_path = save_path.with_suffix(".onnx")
    ckpt = torch.load(save_path, weights_only=True)
    export_model = MoveAttnValueNet(
        d_pre=ckpt["d_pre"], d_move=ckpt["d_move"], d_emb=ckpt["d_emb"],
        hidden=ckpt["hidden"], mon_hidden=ckpt["mon_hidden"],
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
    ap = argparse.ArgumentParser(description="Move-attention value net (gen9/v3)")
    ap.add_argument("--data", required=True)
    ap.add_argument("--model", default="value_net_moveattn.pt")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--hidden", type=int, default=256)
    ap.add_argument("--mon-hidden", type=int, default=128)
    ap.add_argument("--d-emb", type=int, default=128)
    ap.add_argument("--d-pre", type=int, default=64)
    ap.add_argument("--d-move", type=int, default=64)
    ap.add_argument("--n-heads", type=int, default=4)
    ap.add_argument("--n-attn-layers", type=int, default=2)
    ap.add_argument("--batch-size", type=int, default=512)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--filter-draws", action="store_true")
    ap.add_argument("--outcome-labels", action="store_true",
                    help="use per-game winner as label instead of per-turn MCTS V")
    ap.add_argument("--gamma", type=float, default=1.0,
                    help="discount for outcome labels: 0.5 + (winner-0.5) * gamma^turns_remaining")
    args = ap.parse_args()
    train(args.data, args.model, args.epochs, args.lr,
          args.hidden, args.mon_hidden, args.d_emb, args.d_pre, args.d_move,
          args.n_heads, args.n_attn_layers,
          args.batch_size, args.device,
          filter_draws=args.filter_draws,
          outcome_labels=args.outcome_labels,
          gamma=args.gamma)


if __name__ == "__main__":
    main()
