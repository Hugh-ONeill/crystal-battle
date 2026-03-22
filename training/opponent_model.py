# opponent action prediction model
# trained to predict P2's action given P1's observation
# used for lookahead search at inference time
#
# Usage:
#   generate:  python training/opponent_model.py --generate --n-games 10000
#   train:     python training/opponent_model.py --train --data opp_data.pkl
#   evaluate:  python training/opponent_model.py --evaluate --model opp_model.pt

from __future__ import annotations

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

from engine.types import TypeChart
from engine.data_loader import DataStore
from engine.battle_state import BattleState
from engine.player_state import PlayerState
from engine.turn_engine import resolve_turn, resolve_forced_switches
from engine.actions import Switch, UseMove, Struggle, Action
from training.baselines import SmartAgent, MaxDamageAgent, RandomAgent
from gym_env.team_builder import build_team
from gym_env.obs_builder import build_observation, OBS_SIZE


def _action_to_int(action: Action, player_state: PlayerState) -> int | None:
    """Convert an engine Action to an int (0-9)."""
    if isinstance(action, UseMove):
        return action.slot_index
    elif isinstance(action, Switch):
        return 4 + action.team_index
    elif isinstance(action, Struggle):
        return 0
    return None


# ============================================================
# DATA GENERATION
# ============================================================

def generate_opponent_data(
    n_games: int = 10000,
    seed: int = 42,
    out_path: str = "opp_data.pkl",
):
    """Generate (p1_obs, p2_action) pairs from diverse opponent games.

    Records what the opponent does from P1's perspective.
    Uses mixed opponents (Smart, MaxDmg, Random) to learn a general model.
    """
    tc = TypeChart.load()
    data = DataStore()
    rng = random.Random(seed)

    sequences = []  # list of (obs_seq, opp_action_seq, opp_mask_seq)

    for game_idx in range(n_games):
        game_seed = rng.randint(0, 2**31)
        game_rng = random.Random(game_seed)

        # P1 plays Smart (generates realistic game states)
        p1_agent = SmartAgent(tc, seed=game_seed)

        # P2 is Smart or MaxDmg (no random -- unpredictable = junk data)
        if game_rng.random() < 0.5:
            p2_agent = SmartAgent(tc, seed=game_seed + 1000)
        else:
            p2_agent = MaxDamageAgent(tc)

        t1 = build_team(data, rng=random.Random(game_seed + 100), tier="ou")
        t2 = build_team(data, rng=random.Random(game_seed + 200), tier="ou")
        battle = BattleState(
            p1=PlayerState(team=t1), p2=PlayerState(team=t2),
            rng=random.Random(game_seed + 300),
        )

        obs_seq = []
        opp_action_seq = []

        for turn in range(100):
            if battle.is_over:
                break

            # P1's observation (what the agent sees)
            obs = build_observation(
                battle.p1, battle.p2, tc, turn=battle.turn,
                weather=battle.weather, weather_turns=battle.weather_turns,
            )

            # both agents pick actions
            a1 = p1_agent.act(battle.p1, battle.p2)
            a2 = p2_agent.act(battle.p2, battle.p1)

            # convert P2's action to int from P2's perspective
            a2_int = _action_to_int(a2, battle.p2)
            if a2_int is None:
                resolve_turn(battle, a1, a2, tc)
                continue

            obs_seq.append(obs)
            opp_action_seq.append(a2_int)

            resolve_turn(battle, a1, a2, tc)

            # forced switches
            sw1 = sw2 = None
            if battle.p1.must_switch:
                sw1_a = p1_agent.act(battle.p1, battle.p2)
                sw1 = sw1_a if isinstance(sw1_a, Switch) else None
                if sw1 is None:
                    for i, p in enumerate(battle.p1.team):
                        if i != battle.p1.active_index and not p.is_fainted:
                            sw1 = Switch(team_index=i); break
            if battle.p2.must_switch:
                sw2_a = p2_agent.act(battle.p2, battle.p1)
                sw2 = sw2_a if isinstance(sw2_a, Switch) else None
                if sw2 is None:
                    for i, p in enumerate(battle.p2.team):
                        if i != battle.p2.active_index and not p.is_fainted:
                            sw2 = Switch(team_index=i); break

                # record forced switch prediction too
                if sw2 is not None:
                    obs_sw = build_observation(
                        battle.p1, battle.p2, tc, turn=battle.turn,
                        weather=battle.weather, weather_turns=battle.weather_turns,
                    )
                    obs_seq.append(obs_sw)
                    opp_action_seq.append(4 + sw2.team_index)

            if sw1 or sw2:
                resolve_forced_switches(battle, sw1, sw2)

        if len(obs_seq) >= 3:
            sequences.append((
                np.array(obs_seq, dtype=np.float32),
                np.array(opp_action_seq, dtype=np.int64),
            ))

        if (game_idx + 1) % 1000 == 0:
            total = sum(len(s[0]) for s in sequences)
            print(f"  {game_idx + 1}/{n_games} games, {total} samples")

    total = sum(len(s[0]) for s in sequences)
    with open(out_path, "wb") as f:
        pickle.dump(sequences, f)
    print(f"Saved {len(sequences)} sequences ({total} samples) to {out_path}")


# ============================================================
# OPPONENT MODEL
# ============================================================

class OpponentPredictor(nn.Module):
    """Predicts opponent's action from P1's observation sequence.

    Uses a small LSTM to capture opponent behavior patterns over the game.
    Output is a distribution over 10 actions (4 moves + 6 switches).
    """

    def __init__(self, obs_dim: int = OBS_SIZE, hidden: int = 128, n_actions: int = 10):
        super().__init__()
        self.obs_encoder = nn.Sequential(
            nn.Linear(obs_dim, 256),
            nn.ReLU(),
            nn.Linear(256, hidden),
            nn.ReLU(),
        )
        self.lstm = nn.LSTM(hidden, hidden, num_layers=1, batch_first=True)
        self.head = nn.Linear(hidden, n_actions)

    def forward(self, obs_seq: torch.Tensor, hidden=None):
        """Forward pass.

        Args:
            obs_seq: (batch, seq_len, obs_dim) or (seq_len, obs_dim)
            hidden: optional LSTM hidden state

        Returns:
            logits: (batch, seq_len, 10) action logits
            hidden: updated LSTM state
        """
        if obs_seq.dim() == 2:
            obs_seq = obs_seq.unsqueeze(0)

        batch, seq_len, _ = obs_seq.shape
        # encode each timestep
        encoded = self.obs_encoder(obs_seq.reshape(-1, obs_seq.shape[-1]))
        encoded = encoded.view(batch, seq_len, -1)

        lstm_out, hidden = self.lstm(encoded, hidden)
        logits = self.head(lstm_out)
        return logits, hidden

    def predict_single(self, obs: np.ndarray, hidden=None):
        """Predict opponent's action for a single observation.

        Returns (action_probs, hidden_state).
        """
        with torch.no_grad():
            obs_t = torch.tensor(obs, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
            logits, hidden = self.forward(obs_t, hidden)
            probs = F.softmax(logits.squeeze(0).squeeze(0), dim=0)
            return probs.numpy(), hidden


# ============================================================
# TRAINING
# ============================================================

def train_opponent_model(
    data_path: str = "opp_data.pkl",
    save_path: str = "opp_model.pt",
    epochs: int = 20,
    lr: float = 1e-3,
    device: str = "cpu",
):
    """Train the opponent prediction model."""
    print(f"Loading data from {data_path}...")
    with open(data_path, "rb") as f:
        sequences = pickle.load(f)

    total = sum(len(s[0]) for s in sequences)
    print(f"  {len(sequences)} sequences, {total} samples")

    rng = random.Random(42)
    rng.shuffle(sequences)
    val_size = len(sequences) // 10
    val_seqs = sequences[:val_size]
    train_seqs = sequences[val_size:]

    model = OpponentPredictor().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    def run_epoch(seqs, train=True):
        model.train() if train else model.eval()
        total_loss = 0
        total_correct = 0
        total_count = 0

        indices = list(range(len(seqs)))
        if train:
            random.shuffle(indices)

        batch_size = 32
        optimizer.zero_grad()

        for batch_start in range(0, len(indices), batch_size):
            batch_idx = indices[batch_start:batch_start + batch_size]
            batch_loss = 0

            for idx in batch_idx:
                obs_seq, act_seq = seqs[idx]
                obs_t = torch.tensor(obs_seq, dtype=torch.float32, device=device)
                act_t = torch.tensor(act_seq, dtype=torch.long, device=device)

                with torch.set_grad_enabled(train):
                    logits, _ = model(obs_t)
                    logits = logits.squeeze(0)  # (seq_len, 10)
                    loss = F.cross_entropy(logits, act_t)
                    batch_loss = batch_loss + loss / len(batch_idx)

                    preds = logits.argmax(dim=1)
                    total_correct += (preds == act_t).sum().item()
                    total_count += len(act_t)
                    total_loss += loss.item() * len(act_t)

            if train:
                batch_loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                optimizer.zero_grad()

        return total_loss / total_count, total_correct / total_count

    best_val_acc = 0
    print(f"Training for {epochs} epochs...")
    for epoch in range(epochs):
        train_loss, train_acc = run_epoch(train_seqs, train=True)
        with torch.no_grad():
            val_loss, val_acc = run_epoch(val_seqs, train=False)
        scheduler.step()

        marker = ""
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), save_path)
            marker = " *best*"

        print(f"  Epoch {epoch+1:2d}: train_loss={train_loss:.4f} train_acc={train_acc:.3f} "
              f"val_loss={val_loss:.4f} val_acc={val_acc:.3f}{marker}")

    print(f"Best val accuracy: {best_val_acc:.3f}")


# ============================================================
# EVALUATION
# ============================================================

def evaluate_model(model_path: str, data_path: str = "opp_data.pkl", device: str = "cpu"):
    """Evaluate opponent model and show per-action accuracy."""
    with open(data_path, "rb") as f:
        sequences = pickle.load(f)

    # use last 10% as test
    rng = random.Random(42)
    rng.shuffle(sequences)
    test_seqs = sequences[:len(sequences) // 10]

    model = OpponentPredictor().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    model.eval()

    # per-action stats
    action_names = ["Move0", "Move1", "Move2", "Move3",
                    "Switch0", "Switch1", "Switch2", "Switch3", "Switch4", "Switch5"]
    correct_per_action = [0] * 10
    total_per_action = [0] * 10

    with torch.no_grad():
        for obs_seq, act_seq in test_seqs:
            obs_t = torch.tensor(obs_seq, dtype=torch.float32, device=device)
            act_t = torch.tensor(act_seq, dtype=torch.long, device=device)

            logits, _ = model(obs_t)
            preds = logits.squeeze(0).argmax(dim=1)

            for pred, true in zip(preds, act_t):
                t = true.item()
                total_per_action[t] += 1
                if pred.item() == t:
                    correct_per_action[t] += 1

    print("Per-action accuracy:")
    for i, name in enumerate(action_names):
        if total_per_action[i] > 0:
            acc = correct_per_action[i] / total_per_action[i]
            print(f"  {name:8s}: {acc:.1%} ({correct_per_action[i]}/{total_per_action[i]})")

    total_correct = sum(correct_per_action)
    total_count = sum(total_per_action)
    print(f"  Overall: {total_correct/total_count:.1%}")


# ============================================================
# CLI
# ============================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Opponent prediction model")
    parser.add_argument("--generate", action="store_true")
    parser.add_argument("--train", action="store_true")
    parser.add_argument("--evaluate", action="store_true")
    parser.add_argument("--n-games", type=int, default=10000)
    parser.add_argument("--data", type=str, default="opp_data.pkl")
    parser.add_argument("--model", type=str, default="opp_model.pt")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--device", type=str, default="cpu")
    args = parser.parse_args()

    if args.generate:
        generate_opponent_data(n_games=args.n_games, out_path=args.data)
    elif args.train:
        train_opponent_model(data_path=args.data, save_path=args.model,
                            epochs=args.epochs, device=args.device)
    elif args.evaluate:
        evaluate_model(args.model, data_path=args.data, device=args.device)
