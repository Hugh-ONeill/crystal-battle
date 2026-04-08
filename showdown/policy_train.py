#!/usr/bin/env python3
# policy net training on MCTS self-play data
# trains a small net to predict MCTS visit distributions from game state
#
# Usage:
#   .venv/bin/python showdown/policy_train.py --data policy_training_data.pkl --epochs 30

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


# ============================================================
# STATE PARSING
# ============================================================

# gen2 types for one-hot encoding
TYPES = ["bug", "dark", "dragon", "electric", "fighting", "fire", "flying",
         "ghost", "grass", "ground", "ice", "normal", "poison", "psychic",
         "rock", "steel", "water", "typeless"]
TYPE_IDX = {t.upper(): i for i, t in enumerate(TYPES)}

# known items
ITEMS = ["leftovers", "thickclub", "lightball", "miracleberry", "mintberry",
         "charcoal", "mysticwater", "magnet", "nevermeltice", "scopelens", "none"]
ITEM_IDX = {i.upper(): idx for idx, i in enumerate(ITEMS)}

# status
STATUSES = ["none", "burn", "sleep", "freeze", "paralyze", "poison", "toxic"]
STATUS_IDX = {s.upper(): i for i, s in enumerate(STATUSES)}

# moves: we'll use a fixed vocabulary of known Gen 2 moves
# build this dynamically from the training data


def parse_pokemon(fields: list[str]) -> np.ndarray:
    """Parse 28 fields of a pokemon into a feature vector."""
    features = []

    # hp fraction
    hp = int(fields[6])
    maxhp = int(fields[7])
    features.append(hp / max(maxhp, 1))  # hp_frac
    features.append(1.0 if hp > 0 else 0.0)  # alive

    # types (one-hot, 18 dims)
    type_vec = [0.0] * len(TYPES)
    t1 = fields[2].upper()
    t2 = fields[3].upper()
    if t1 in TYPE_IDX:
        type_vec[TYPE_IDX[t1]] = 1.0
    if t2 in TYPE_IDX:
        type_vec[TYPE_IDX[t2]] = 1.0
    features.extend(type_vec)

    # stats (normalized by 500)
    features.append(int(fields[13]) / 500.0)  # attack
    features.append(int(fields[14]) / 500.0)  # defense
    features.append(int(fields[15]) / 500.0)  # spa
    features.append(int(fields[16]) / 500.0)  # spd
    features.append(int(fields[17]) / 500.0)  # speed

    # status (one-hot, 7 dims)
    status_vec = [0.0] * len(STATUSES)
    status = fields[18].upper()
    if status in STATUS_IDX:
        status_vec[STATUS_IDX[status]] = 1.0
    features.extend(status_vec)

    # item (one-hot, 11 dims)
    item_vec = [0.0] * len(ITEMS)
    item = fields[10].upper()
    if item in ITEM_IDX:
        item_vec[ITEM_IDX[item]] = 1.0
    else:
        item_vec[ITEM_IDX["NONE"]] = 1.0
    features.extend(item_vec)

    # move pp fractions (4 moves)
    for i in range(4):
        move_field = fields[22 + i]
        parts = move_field.split(";")
        if len(parts) >= 3:
            pp = int(parts[2])
            features.append(min(pp / 32.0, 1.0))  # pp fraction
        else:
            features.append(0.0)

    return np.array(features, dtype=np.float32)


# per pokemon: 1 (hp) + 1 (alive) + 18 (types) + 5 (stats) + 7 (status) + 11 (item) + 4 (pp) = 47
POKEMON_FEATURES = 47

# side extras: 7 (boosts) + 3 (spikes/reflect/light_screen) = 10
SIDE_EXTRAS = 10


def parse_side(pokemon_strs: list[str], side_parts: list[str] = None) -> np.ndarray:
    """Parse a side (6 pokemon + boosts + hazards) into features."""
    features = []
    for pstr in pokemon_strs:
        fields = pstr.split(",")
        if len(fields) >= 28:
            features.append(parse_pokemon(fields))
        else:
            features.append(np.zeros(POKEMON_FEATURES, dtype=np.float32))

    # pad to 6 pokemon
    while len(features) < 6:
        features.append(np.zeros(POKEMON_FEATURES, dtype=np.float32))

    extras = np.zeros(SIDE_EXTRAS, dtype=np.float32)

    if side_parts and len(side_parts) >= 18:
        # boosts at =-split indices 11-17: atk/def/spa/spd/spe/acc/eva
        # normalize to [-1, 1] range (max boost is +/-6)
        for i in range(7):
            try:
                extras[i] = int(side_parts[11 + i]) / 6.0
            except (ValueError, IndexError):
                pass

        # side_conditions at =-split index 7, semicolon-separated
        # spikes is field 12 in the semicolon list (0-3 layers)
        # reflect is field 10, light_screen is field 3
        try:
            sc_fields = side_parts[7].split(";")
            if len(sc_fields) >= 13:
                extras[7] = int(sc_fields[12]) / 3.0   # spikes (0-3)
                extras[8] = float(int(sc_fields[10]) > 0)  # reflect
                extras[9] = float(int(sc_fields[3]) > 0)   # light screen
        except (ValueError, IndexError):
            pass

    return np.concatenate(features[:6] + [extras])


# side features: 6 * 47 + 10 = 292
SIDE_FEATURES = 6 * POKEMON_FEATURES + SIDE_EXTRAS


def parse_state_string(state_str: str) -> np.ndarray:
    """Parse a full state string into a feature vector."""
    # format: side_one/side_two/weather/terrain/trick_room/team_preview
    major_parts = state_str.split("/")
    if len(major_parts) < 2:
        return np.zeros(SIDE_FEATURES * 2 + 3, dtype=np.float32)

    # each side: p0=p1=p2=p3=p4=p5=active_idx=side_conditions=...=boosts...
    s1_parts = major_parts[0].split("=")
    s2_parts = major_parts[1].split("=")

    # first 6 are pokemon
    s1_features = parse_side(s1_parts[:6], s1_parts)
    s2_features = parse_side(s2_parts[:6], s2_parts)

    # weather (gen2: none/sun/rain/sand -- 3 one-hot, skip none)
    weather_vec = np.zeros(3, dtype=np.float32)  # sun/rain/sand
    if len(major_parts) > 2:
        weather_str = major_parts[2].upper()
        if "SUN" in weather_str:
            weather_vec[0] = 1.0
        elif "RAIN" in weather_str:
            weather_vec[1] = 1.0
        elif "SAND" in weather_str:
            weather_vec[2] = 1.0

    return np.concatenate([s1_features, s2_features, weather_vec])


# total features: 292 + 292 + 3 = 587
STATE_FEATURES = SIDE_FEATURES * 2 + 3

# action space: 4 moves + 5 switches = 9
# but poke-engine returns move names, need to map to indices
N_ACTIONS = 9


def visits_to_policy(visits: list[tuple[str, int]], side_pokemon: list[str]) -> np.ndarray:
    """Convert MCTS visit counts to a policy probability vector.

    Maps move names to action indices:
      0-3: moves (in order of the active pokemon's moveset)
      4-8: switches (to pokemon 1-5, i.e. non-active)
    """
    policy = np.zeros(N_ACTIONS, dtype=np.float32)

    # get active pokemon's move names from first pokemon string
    active_fields = side_pokemon[0].split(",") if side_pokemon else []
    move_names = []
    for i in range(4):
        if 22 + i < len(active_fields):
            parts = active_fields[22 + i].split(";")
            move_names.append(parts[0].upper())
        else:
            move_names.append("NONE")

    # get bench pokemon names
    bench_names = []
    for i in range(1, 6):
        if i < len(side_pokemon):
            fields = side_pokemon[i].split(",")
            if fields:
                bench_names.append(fields[0].upper())
            else:
                bench_names.append("")
        else:
            bench_names.append("")

    total_visits = sum(v for _, v in visits)
    if total_visits == 0:
        return policy

    for move_name, visit_count in visits:
        name_upper = move_name.upper()
        prob = visit_count / total_visits

        # check if it's a move
        matched = False
        for i, mn in enumerate(move_names):
            if name_upper == mn:
                policy[i] = prob
                matched = True
                break

        if not matched:
            # check if it's a switch
            for i, bn in enumerate(bench_names):
                if name_upper == bn:
                    policy[4 + i] = prob
                    matched = True
                    break

    return policy


# ============================================================
# DATA PREPARATION
# ============================================================

def prepare_data(data_path: str, use_v2: bool = False) -> tuple[np.ndarray, np.ndarray]:
    """Load training data in either format:
    - MCTS: list of (winner, [(state_str, s1_move, s1_visits, s2_move[, s2_visits]), ...])
    - Human: (features_array, policies_array) tuple
    """
    if use_v2:
        from showdown.features_v2 import parse_state_v2 as feature_fn
        print(f"Using v2 features (579 dims)")
    else:
        feature_fn = parse_state_string
        print(f"Using v1 features (587 dims)")

    print(f"Loading {data_path}...")
    with open(data_path, "rb") as f:
        results = pickle.load(f)

    # detect format
    if isinstance(results, tuple) and len(results) == 2:
        # human replay format: (features, policies) arrays
        states, policies = results
        if use_v2 and states.shape[1] != 579:
            print(f"  WARNING: pre-computed features are dim {states.shape[1]}, "
                  f"not 579. Re-extracting not supported for human format.")
        print(f"  {len(states)} samples (human replay format), "
              f"state_dim={states.shape[1]}, action_dim={policies.shape[1]}")
        return states, policies

    # MCTS format: list of (winner, turns)
    # turns are either 4-tuples (old) or 5-tuples (new, with s2_visits)
    states = []
    policies = []

    for winner, turns in results:
        for turn_data in turns:
            state_str = turn_data[0]
            s1_visits = turn_data[2]

            # side one: parse state as-is
            features = feature_fn(state_str)
            states.append(features)

            side1_parts = state_str.split("/")[0].split("=")
            side1_pokemon = side1_parts[:6]
            policy = visits_to_policy(s1_visits, side1_pokemon)
            policies.append(policy)

            # side two: if s2_visits present, flip sides for s2 perspective
            if len(turn_data) >= 5 and turn_data[4]:
                s2_visits = turn_data[4]
                major_parts = state_str.split("/")
                if len(major_parts) >= 2:
                    # flip: s2 becomes s1 from their perspective
                    flipped = "/".join([major_parts[1], major_parts[0]]
                                       + major_parts[2:])
                    features_s2 = feature_fn(flipped)
                    states.append(features_s2)

                    side2_parts = major_parts[1].split("=")
                    side2_pokemon = side2_parts[:6]
                    policy_s2 = visits_to_policy(s2_visits, side2_pokemon)
                    policies.append(policy_s2)

    states = np.array(states, dtype=np.float32)
    policies = np.array(policies, dtype=np.float32)

    print(f"  {len(states)} samples (MCTS format), state_dim={states.shape[1]}, "
          f"action_dim={policies.shape[1]}")
    return states, policies


# ============================================================
# MODEL
# ============================================================

class PolicyNet(nn.Module):
    """Simple MLP policy network.

    Input: state features (587 dim)
    Output: action probabilities (9 dim)
    """

    def __init__(self, state_dim=STATE_FEATURES, action_dim=N_ACTIONS,
                 hidden=256, n_layers=3):
        super().__init__()
        layers = []
        in_dim = state_dim
        for _ in range(n_layers):
            layers.extend([nn.Linear(in_dim, hidden), nn.ReLU()])
            in_dim = hidden
        layers.append(nn.Linear(hidden, action_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)

    def predict(self, state_features: np.ndarray) -> np.ndarray:
        """Predict action probabilities from state features."""
        with torch.no_grad():
            x = torch.tensor(state_features, dtype=torch.float32)
            if x.dim() == 1:
                x = x.unsqueeze(0)
            logits = self.forward(x)
            probs = F.softmax(logits, dim=-1)
            return probs.numpy()


# ============================================================
# TRAINING
# ============================================================

def train(data_path: str, save_path: str = "policy_net.pt",
          epochs: int = 30, lr: float = 1e-3, hidden: int = 256,
          batch_size: int = 256, device: str = "cpu", use_v2: bool = False):
    """Train policy net on MCTS self-play data."""

    states, policies = prepare_data(data_path, use_v2=use_v2)

    # shuffle and split
    n = len(states)
    indices = list(range(n))
    random.seed(42)
    random.shuffle(indices)
    val_size = n // 10
    val_idx = indices[:val_size]
    train_idx = indices[val_size:]

    train_states = torch.tensor(states[train_idx], device=device)
    train_policies = torch.tensor(policies[train_idx], device=device)
    val_states = torch.tensor(states[val_idx], device=device)
    val_policies = torch.tensor(policies[val_idx], device=device)

    model = PolicyNet(state_dim=states.shape[1], hidden=hidden).to(device)
    params = sum(p.numel() for p in model.parameters())
    print(f"  {params:,} parameters\n")

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    best_val = float("inf")
    for epoch in range(epochs):
        model.train()
        perm = torch.randperm(len(train_states))
        total_loss = 0
        n_batches = 0

        for i in range(0, len(train_states), batch_size):
            batch_idx = perm[i:i + batch_size]
            batch_s = train_states[batch_idx]
            batch_p = train_policies[batch_idx]

            logits = model(batch_s)
            # KL divergence loss: train on soft MCTS targets
            log_probs = F.log_softmax(logits, dim=-1)
            # avoid log(0) in targets
            batch_p_safe = batch_p.clamp(min=1e-8)
            loss = F.kl_div(log_probs, batch_p_safe, reduction="batchmean",
                            log_target=False)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            total_loss += loss.item()
            n_batches += 1

        # validate
        model.eval()
        with torch.no_grad():
            val_logits = model(val_states)
            val_log_probs = F.log_softmax(val_logits, dim=-1)
            val_p_safe = val_policies.clamp(min=1e-8)
            val_loss = F.kl_div(val_log_probs, val_p_safe, reduction="batchmean",
                                log_target=False).item()

            # top-1 accuracy: does the predicted argmax match the MCTS choice?
            pred_actions = val_logits.argmax(dim=-1)
            true_actions = val_policies.argmax(dim=-1)
            accuracy = (pred_actions == true_actions).float().mean().item()

        scheduler.step()

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
    export_model = PolicyNet(state_dim=ckpt["state_dim"], hidden=ckpt["hidden"])
    export_model.load_state_dict(ckpt["model"])
    export_model.eval()
    dummy = torch.randn(1, ckpt["state_dim"])
    torch.onnx.export(export_model, dummy, onnx_path,
                      input_names=["state"], output_names=["logits"],
                      dynamic_axes={"state": {0: "batch"}, "logits": {0: "batch"}},
                      dynamo=False)
    print(f"ONNX exported to {onnx_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Policy net training")
    parser.add_argument("--data", type=str, default="policy_training_data.pkl")
    parser.add_argument("--model", type=str, default="policy_net.pt")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--hidden", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--features-v2", action="store_true",
                        help="use v2 feature extraction (579 dims, move-aware)")
    args = parser.parse_args()

    train(args.data, args.model, args.epochs, args.lr, args.hidden,
          args.batch_size, args.device, use_v2=args.features_v2)
