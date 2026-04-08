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

def prepare_value_data(data_path: str, use_v2: bool = False
                       ) -> tuple[np.ndarray, np.ndarray]:
    """Load recorded games and prepare (state_features, win_label) arrays.

    Each turn gets labeled with the game outcome from side_one's perspective:
    - winner > 0 (side_one won): label = 1.0
    - winner < 0 (side_two won): label = 0.0
    - draw: label = 0.5
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

    for winner, turns in results:
        # side_one label
        if winner > 0:
            s1_label = 1.0
        elif winner < 0:
            s1_label = 0.0
        else:
            s1_label = 0.5

        for turn_data in turns:
            state_str = turn_data[0]

            # side one perspective
            features = feature_fn(state_str)
            states.append(features)
            labels.append(s1_label)

            # side two perspective (flip sides, flip label)
            if len(turn_data) >= 5 and turn_data[4]:
                major_parts = state_str.split("/")
                if len(major_parts) >= 2:
                    flipped = "/".join([major_parts[1], major_parts[0]]
                                       + major_parts[2:])
                    features_s2 = feature_fn(flipped)
                    states.append(features_s2)
                    labels.append(1.0 - s1_label)

    states = np.array(states, dtype=np.float32)
    labels = np.array(labels, dtype=np.float32)

    print(f"  {len(states)} samples, state_dim={states.shape[1]}")
    print(f"  label distribution: {(labels > 0.5).sum()} wins, "
          f"{(labels < 0.5).sum()} losses, {(labels == 0.5).sum()} draws")
    return states, labels


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
          batch_size: int = 256, device: str = "cpu", use_v2: bool = False):

    states, labels = prepare_value_data(data_path, use_v2=use_v2)

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
            # only count non-draw samples for accuracy
            non_draw = val_labels != 0.5
            if non_draw.sum() > 0:
                accuracy = (val_preds[non_draw] == val_labels[non_draw]).float().mean().item()
            else:
                accuracy = 0.0

        marker = ""
        if val_loss < best_val:
            best_val = val_loss
            torch.save({
                "model": model.state_dict(),
                "state_dim": states.shape[1],
                "hidden": hidden,
            }, save_path)
            marker = " *best*"

        print(f"  Epoch {epoch + 1:2d}: train_loss={total_loss / n_batches:.4f} "
              f"val_loss={val_loss:.4f} acc={accuracy:.3f}{marker}")

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
    args = parser.parse_args()

    train(args.data, args.model, args.epochs, args.lr, args.hidden,
          args.batch_size, args.device, use_v2=args.features_v2)
