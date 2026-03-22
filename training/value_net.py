# standalone feedforward value network for search leaf evaluation
# trained on (obs, game_outcome) pairs -- no LSTM, works on single positions
#
# Usage:
#   generate:  python training/value_net.py --generate --n-games 5000
#   train:     python training/value_net.py --train
#   evaluate:  python training/value_net.py --evaluate

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
from torch.utils.data import DataLoader, TensorDataset

sys.path.insert(0, str(Path(__file__).parent.parent))

from gym_env.obs_builder import OBS_SIZE


class FeedforwardValueNet(nn.Module):
    """Predicts game outcome from a single observation (no LSTM context).

    Input: obs vector (OBS_SIZE)
    Output: scalar in [-1, 1] (win probability: +1 = P1 wins, -1 = P2 wins)
    """

    def __init__(self, obs_dim: int = OBS_SIZE, hidden: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1),
            nn.Tanh(),
        )

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.net(obs).squeeze(-1)

    def predict(self, obs: np.ndarray) -> float:
        with torch.no_grad():
            obs_t = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)
            return self.forward(obs_t).item()

    def predict_batch(self, obs_batch: np.ndarray) -> np.ndarray:
        with torch.no_grad():
            obs_t = torch.tensor(obs_batch, dtype=torch.float32)
            return self.forward(obs_t).numpy()


# ============================================================
# DATA GENERATION
# ============================================================

def generate_value_data(
    n_games: int = 5000,
    seed: int = 42,
    out_path: str = "value_data.npz",
):
    """Generate (obs, outcome) pairs from search games.

    Uses Rust 2-ply search for P1 (strong play), mixed opponents for P2.
    Each obs in a game gets the game's final outcome as its target.
    """
    from engine.types import TypeChart
    from engine.data_loader import DataStore
    from training.baselines import SmartAgent, MaxDamageAgent
    from training.opponent_model import OpponentPredictor
    from gym_env.team_builder import build_team
    import crystal_engine_rs as ce
    from training.rust_search_agent import (
        build_obs_from_rust, _RustPlayerAdapter,
        _get_opp_actions, _handle_forced_switches_rs,
        _first_alive_bench_idx, _py_action_to_int
    )

    tc = TypeChart.load()
    data = DataStore()
    rs_data = ce.DataStore(str(Path(__file__).parent.parent / "data"))

    opp_model = OpponentPredictor()
    opp_model.load_state_dict(
        torch.load("opp_model.pt", map_location="cpu", weights_only=True))
    opp_model.eval()

    smart = SmartAgent(tc, seed=0)
    maxdmg = MaxDamageAgent(tc)
    rng = random.Random(seed)

    all_obs = []
    all_outcomes = []

    for game_idx in range(n_games):
        game_seed = rng.randint(0, 2**31)
        game_rng = random.Random(game_seed)

        # mixed opponent
        if game_rng.random() < 0.5:
            opp = smart
        else:
            opp = maxdmg

        def opp_policy(my, op):
            return opp.act(my, op)

        t1 = build_team(data, rng=random.Random(game_seed + 100), tier="ou")
        t2 = build_team(data, rng=random.Random(game_seed + 200), tier="ou")

        rs_t1 = [rs_data.build_pokemon(m.species.id,
                  [s.template.id for s in m.move_slots]) for m in t1]
        rs_t2 = [rs_data.build_pokemon(m.species.id,
                  [s.template.id for s in m.move_slots]) for m in t2]
        rs_battle = ce.create_battle(rs_t1, rs_t2, seed=game_seed + 300)

        game_obs = []
        opp_hidden = None

        for turn in range(100):
            if rs_battle.is_over:
                break

            obs = build_obs_from_rust(rs_battle, tc)
            game_obs.append(obs)

            p1_mask = rs_battle.p1.valid_action_mask(rs_battle.p2, filter_immune=True)
            valid = [j for j in range(10) if p1_mask[j]]

            if not valid:
                break

            if len(valid) == 1:
                best = valid[0]
                _, opp_hidden = opp_model.predict_single(obs, opp_hidden)
            else:
                opp_actions = _get_opp_actions(
                    rs_battle, obs, opp_model, opp_hidden, 5)
                _, opp_hidden = opp_model.predict_single(obs, opp_hidden)

                ranked = ce.search_2ply(
                    rs_battle, valid, opp_actions, opp_actions,
                    base_seed=turn * 1000)
                best = ranked[0][0] if ranked else valid[0]

            p2_adapter = _RustPlayerAdapter(rs_battle.p2)
            p1_adapter = _RustPlayerAdapter(rs_battle.p1)
            p2_action = opp_policy(p2_adapter, p1_adapter)
            a2_int = _py_action_to_int(p2_action)

            rs_battle.resolve_turn(best, a2_int)
            _handle_forced_switches_rs(rs_battle, opp_policy)

            if rs_battle.p1.must_switch:
                sw = _first_alive_bench_idx(rs_battle.p1)
                if sw is not None:
                    rs_battle.resolve_forced_switches(sw, None)

        # assign outcome to every obs in this game
        if rs_battle.winner == 1:
            outcome = 1.0
        elif rs_battle.winner == 2:
            outcome = -1.0
        else:
            outcome = 0.0

        for obs in game_obs:
            all_obs.append(obs)
            all_outcomes.append(outcome)

        if (game_idx + 1) % 500 == 0:
            print(f"  {game_idx + 1}/{n_games} games, {len(all_obs)} samples")

    obs_arr = np.array(all_obs, dtype=np.float32)
    outcome_arr = np.array(all_outcomes, dtype=np.float32)

    np.savez_compressed(out_path, obs=obs_arr, outcomes=outcome_arr)
    print(f"Saved {len(obs_arr)} samples to {out_path}")


# ============================================================
# TRAINING
# ============================================================

def train_value_net(
    data_path: str = "value_data.npz",
    save_path: str = "value_net.pt",
    epochs: int = 30,
    batch_size: int = 1024,
    lr: float = 1e-3,
    device: str = "cpu",
):
    """Train the feedforward value network."""
    print(f"Loading data from {data_path}...")
    npz = np.load(data_path)
    obs = torch.tensor(npz["obs"], dtype=torch.float32)
    outcomes = torch.tensor(npz["outcomes"], dtype=torch.float32)
    print(f"  {len(obs)} samples")

    # check outcome distribution
    wins = (outcomes > 0).sum().item()
    losses = (outcomes < 0).sum().item()
    draws = (outcomes == 0).sum().item()
    print(f"  Outcomes: {wins} wins, {losses} losses, {draws} draws")

    # train/val split
    n = len(obs)
    perm = torch.randperm(n)
    val_size = n // 10
    train_ds = TensorDataset(obs[perm[val_size:]], outcomes[perm[val_size:]])
    val_ds = TensorDataset(obs[perm[:val_size]], outcomes[perm[:val_size]])
    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_dl = DataLoader(val_ds, batch_size=batch_size)

    model = FeedforwardValueNet().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    best_val_loss = float("inf")
    print(f"Training for {epochs} epochs...")

    for epoch in range(epochs):
        # train
        model.train()
        train_loss = 0
        train_n = 0
        for batch_obs, batch_out in train_dl:
            batch_obs = batch_obs.to(device)
            batch_out = batch_out.to(device)
            pred = model(batch_obs)
            loss = F.mse_loss(pred, batch_out)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * len(batch_obs)
            train_n += len(batch_obs)

        # validate
        model.eval()
        val_loss = 0
        val_n = 0
        val_correct = 0
        with torch.no_grad():
            for batch_obs, batch_out in val_dl:
                batch_obs = batch_obs.to(device)
                batch_out = batch_out.to(device)
                pred = model(batch_obs)
                loss = F.mse_loss(pred, batch_out)
                val_loss += loss.item() * len(batch_obs)
                val_n += len(batch_obs)
                # accuracy: does the sign match?
                val_correct += ((pred > 0) == (batch_out > 0)).sum().item()

        scheduler.step()

        marker = ""
        if val_loss / val_n < best_val_loss:
            best_val_loss = val_loss / val_n
            torch.save(model.state_dict(), save_path)
            marker = " *best*"

        val_acc = val_correct / val_n
        print(f"  Epoch {epoch+1:2d}: train_loss={train_loss/train_n:.4f} "
              f"val_loss={val_loss/val_n:.4f} val_sign_acc={val_acc:.3f}{marker}")

    print(f"Best val loss: {best_val_loss:.4f}")
    print(f"Saved to {save_path}")


# ============================================================
# EVALUATION
# ============================================================

def evaluate_value_net(model_path: str = "value_net.pt", data_path: str = "value_data.npz",
                       device: str = "cpu"):
    """Evaluate the value network's predictions."""
    npz = np.load(data_path)
    obs = torch.tensor(npz["obs"], dtype=torch.float32)
    outcomes = torch.tensor(npz["outcomes"], dtype=torch.float32)

    model = FeedforwardValueNet().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    model.eval()

    # test on last 10%
    n = len(obs)
    test_obs = obs[n * 9 // 10:]
    test_out = outcomes[n * 9 // 10:]

    with torch.no_grad():
        preds = model(test_obs.to(device)).cpu()

    # metrics
    mse = F.mse_loss(preds, test_out).item()
    sign_acc = ((preds > 0) == (test_out > 0)).float().mean().item()

    # calibration: average prediction for wins vs losses
    win_mask = test_out > 0
    loss_mask = test_out < 0
    avg_win_pred = preds[win_mask].mean().item() if win_mask.any() else 0
    avg_loss_pred = preds[loss_mask].mean().item() if loss_mask.any() else 0

    print(f"Test metrics ({len(test_obs)} samples):")
    print(f"  MSE: {mse:.4f}")
    print(f"  Sign accuracy: {sign_acc:.1%}")
    print(f"  Avg prediction for wins: {avg_win_pred:+.3f} (should be +1)")
    print(f"  Avg prediction for losses: {avg_loss_pred:+.3f} (should be -1)")
    print(f"  Heuristic comparison: hp_diff*0.6 + alive_diff*0.4 has no learning")

    # check a fresh game position
    print(f"\n  Fresh game (turn 0) prediction: {model.predict(obs[0].numpy()):+.3f} (should be ~0)")


# ============================================================
# CLI
# ============================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Feedforward value network for search")
    parser.add_argument("--generate", action="store_true")
    parser.add_argument("--train", action="store_true")
    parser.add_argument("--evaluate", action="store_true")
    parser.add_argument("--n-games", type=int, default=5000)
    parser.add_argument("--data", type=str, default="value_data.npz")
    parser.add_argument("--model", type=str, default="value_net.pt")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--device", type=str, default="cpu")
    args = parser.parse_args()

    if args.generate:
        generate_value_data(n_games=args.n_games, out_path=args.data)
    elif args.train:
        train_value_net(data_path=args.data, save_path=args.model,
                        epochs=args.epochs, device=args.device)
    elif args.evaluate:
        evaluate_value_net(args.model, data_path=args.data, device=args.device)
