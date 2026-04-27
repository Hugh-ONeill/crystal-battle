#!/usr/bin/env python3
# value network training: predict win probability from game state
# trained on recorded games with known outcomes
#
# Usage:
#   .venv/bin/python showdown/value_train.py --data hypnosis_diverse_data.pkl \
#     --model value_net.pt --epochs 30 --features-v2

import argparse
import pickle
import random
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).parent.parent))
from showdown.policy_train import parse_state_string
from showdown.features_v2 import parse_state_v2, STATE_FEATURES_V2


# ============================================================
# DATA PREPARATION
# ============================================================

def prepare_value_data(data_path: str, use_v2: bool = False,
                       filter_draws: bool = False,
                       ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load recorded games and prepare (state_features, win_label, turns_remaining).

    Each turn gets labeled with the game outcome from side_one's perspective:
    - winner > 0 (side_one won): label = 1.0
    - winner < 0 (side_two won): label = 0.0
    - draw: label = 0.5

    turns_remaining[i] = number of turn-positions after the i-th turn until
    game ends. Used at training time for gamma-discount of far-from-terminal
    labels (anti-saturation).
    """
    feature_fn = parse_state_v2 if use_v2 else parse_state_string
    feat_name = "v2 (579)" if use_v2 else "v1 (587)"
    print(f"Using {feat_name} features")

    print(f"Loading {data_path}...")
    with open(data_path, "rb") as f:
        results = pickle.load(f)

    # detect format
    if isinstance(results, tuple) and len(results) == 2:
        print("ERROR: pre-extracted format has no game outcomes. Need raw MCTS data.")
        sys.exit(1)

    states = []
    labels = []
    turns_remaining = []

    n_draw_skipped = 0
    for winner, turns in results:
        if filter_draws and winner == 0:
            n_draw_skipped += 1
            continue
        # side_one label
        if winner > 0:
            s1_label = 1.0
        elif winner < 0:
            s1_label = 0.0
        else:
            s1_label = 0.5

        n_turns = len(turns)
        for i, turn_data in enumerate(turns):
            state_str = turn_data[0]
            # i-th turn-position has (n_turns - i) more positions after it
            # before the game ends. The last position has turns_remaining=1.
            tr = n_turns - i

            # side one perspective
            features = feature_fn(state_str)
            states.append(features)
            labels.append(s1_label)
            turns_remaining.append(tr)

            # side two perspective (flip sides, flip label)
            if len(turn_data) >= 5 and turn_data[4]:
                major_parts = state_str.split("/")
                if len(major_parts) >= 2:
                    flipped = "/".join([major_parts[1], major_parts[0]]
                                       + major_parts[2:])
                    features_s2 = feature_fn(flipped)
                    states.append(features_s2)
                    labels.append(1.0 - s1_label)
                    turns_remaining.append(tr)

    states = np.array(states, dtype=np.float32)
    labels = np.array(labels, dtype=np.float32)
    turns_remaining = np.array(turns_remaining, dtype=np.int32)

    if filter_draws:
        print(f"  filtered {n_draw_skipped} draw games")
    print(f"  {len(states)} samples, state_dim={states.shape[1]}")
    print(f"  label distribution: {(labels > 0.5).sum()} wins, "
          f"{(labels < 0.5).sum()} losses, {(labels == 0.5).sum()} draws")
    print(f"  turns_remaining: min={turns_remaining.min()}, "
          f"max={turns_remaining.max()}, mean={turns_remaining.mean():.1f}")
    return states, labels, turns_remaining


# ============================================================
# MODEL
# ============================================================

class ValueNet(nn.Module):
    """MLP value network.

    Input: state features
    Output: scalar win probability (sigmoid)
    """

    def __init__(self, state_dim=579, hidden=256, n_layers=3):
        super().__init__()
        layers = []
        in_dim = state_dim
        for _ in range(n_layers):
            layers.extend([nn.Linear(in_dim, hidden), nn.ReLU()])
            in_dim = hidden
        layers.append(nn.Linear(hidden, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x).squeeze(-1)

    def predict(self, state_features: np.ndarray) -> float:
        """Predict win probability from state features."""
        with torch.no_grad():
            x = torch.tensor(state_features, dtype=torch.float32)
            if x.dim() == 1:
                x = x.unsqueeze(0)
            logit = self.forward(x)
            return torch.sigmoid(logit).item()


# ============================================================
# TRAINING
# ============================================================

def train(data_path: str, save_path: str = "value_net.pt",
          epochs: int = 30, lr: float = 1e-3, hidden: int = 256,
          batch_size: int = 256, device: str = "cpu", use_v2: bool = False,
          gamma: float = 1.0, filter_draws: bool = False):
    """Train the value net.

    gamma: discount applied to the win/loss target via gamma^turns_remaining.
    target = 0.5 + (raw_label - 0.5) * gamma^turns_remaining, so far-from-
    terminal labels are pulled toward the 0.5 prior. gamma<1 fixes the
    saturation pathology seen with raw ±1 outcome labels.
    """
    states, raw_labels, turns_remaining = prepare_value_data(
        data_path, use_v2=use_v2, filter_draws=filter_draws)

    # gamma-discount: pull mid-game labels toward 0.5 (uncertainty)
    if gamma != 1.0:
        delta = raw_labels - 0.5
        discount = gamma ** turns_remaining.astype(np.float32)
        labels = 0.5 + delta * discount
        print(f"  Gamma: {gamma} (mean discount={discount.mean():.3f}, "
              f"min={discount.min():.3f}, max={discount.max():.3f})")
        win_mask = raw_labels > 0.5
        if win_mask.any():
            print(f"  Win-side targets after discount: "
                  f"mean={labels[win_mask].mean():.3f}, "
                  f"min={labels[win_mask].min():.3f}, "
                  f"max={labels[win_mask].max():.3f}")
    else:
        labels = raw_labels

    # shuffle and split
    n = len(states)
    indices = list(range(n))
    random.seed(42)
    random.shuffle(indices)
    val_size = n // 10
    val_idx = indices[:val_size]
    train_idx = indices[val_size:]

    train_states = torch.tensor(states[train_idx], device=device)
    train_labels = torch.tensor(labels[train_idx], device=device)
    val_states = torch.tensor(states[val_idx], device=device)
    val_labels = torch.tensor(labels[val_idx], device=device)
    raw_val_labels = torch.tensor(raw_labels[val_idx], device=device)

    model = ValueNet(state_dim=states.shape[1], hidden=hidden).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  {n_params:,} parameters\n")

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    best_val = float("inf")

    for epoch in range(epochs):
        model.train()
        perm = torch.randperm(len(train_states), device=device)
        total_loss = 0.0
        n_batches = 0

        for start in range(0, len(train_states), batch_size):
            batch_idx = perm[start:start + batch_size]
            batch_states = train_states[batch_idx]
            batch_labels = train_labels[batch_idx]

            logits = model(batch_states)
            loss = F.binary_cross_entropy_with_logits(logits, batch_labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            n_batches += 1

        # validation
        model.eval()
        with torch.no_grad():
            val_logits = model(val_states)
            val_loss = F.binary_cross_entropy_with_logits(val_logits, val_labels).item()
            val_probs = torch.sigmoid(val_logits)
            val_preds = (val_probs > 0.5).float()
            # sign accuracy uses original ±1 labels (un-discounted)
            non_draw = raw_val_labels != 0.5
            if non_draw.sum() > 0:
                accuracy = (val_preds[non_draw] == raw_val_labels[non_draw]).float().mean().item()
            else:
                accuracy = 0.0
            # saturation: predictions hugging the rails
            sat = ((val_probs > 0.95) | (val_probs < 0.05)).float().mean().item()

        marker = ""
        if val_loss < best_val:
            best_val = val_loss
            torch.save({
                "model": model.state_dict(),
                "state_dim": states.shape[1],
                "hidden": hidden,
                "gamma": gamma,
            }, save_path)
            marker = " *best*"

        print(f"  Epoch {epoch + 1:2d}: train_loss={total_loss / n_batches:.4f} "
              f"val_loss={val_loss:.4f} acc={accuracy:.3f} "
              f"sat={sat:.1%}{marker}")

    print(f"\nBest val loss: {best_val:.4f}, saved to {save_path}")

    # export to ONNX
    onnx_path = save_path.replace(".pt", ".onnx")
    ckpt = torch.load(save_path, weights_only=True)
    export_model = ValueNet(state_dim=ckpt["state_dim"], hidden=ckpt["hidden"])
    export_model.load_state_dict(ckpt["model"])
    export_model.eval()
    dummy = torch.randn(1, ckpt["state_dim"])
    torch.onnx.export(export_model, dummy, onnx_path,
                      input_names=["state"], output_names=["value"],
                      dynamic_axes={"state": {0: "batch"}, "value": {0: "batch"}},
                      dynamo=False)
    print(f"ONNX exported to {onnx_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Value net training")
    parser.add_argument("--data", type=str, required=True)
    parser.add_argument("--model", type=str, default="value_net.pt")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--hidden", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--features-v2", action="store_true")
    parser.add_argument("--gamma", type=float, default=1.0,
                        help="discount: target_prob = 0.5 + (raw - 0.5) * "
                             "gamma^turns_remaining. <1 fights saturation; "
                             "0.97 was the sweet spot for crystal_engine.")
    parser.add_argument("--filter-draws", action="store_true",
                        help="drop games with winner=0 (timeouts) before "
                             "training. Useful when data has many max_turns "
                             "timeouts that would dominate BCE as label=0.5.")
    args = parser.parse_args()

    train(args.data, args.model, args.epochs, args.lr, args.hidden,
          args.batch_size, args.device, use_v2=args.features_v2,
          gamma=args.gamma, filter_draws=args.filter_draws)
