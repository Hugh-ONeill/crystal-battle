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

    def __init__(self, obs_dim: int = OBS_SIZE, hidden: int = 256,
                 n_layers: int = 4, dropout: float = 0.2):
        super().__init__()
        layers: list[nn.Module] = [nn.Linear(obs_dim, hidden), nn.ReLU()]
        if dropout > 0:
            layers.append(nn.Dropout(dropout))
        for _ in range(n_layers - 1):
            layers.extend([nn.Linear(hidden, hidden), nn.ReLU()])
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
        layers.extend([nn.Linear(hidden, 1), nn.Tanh()])
        self.net = nn.Sequential(*layers)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.net(obs).squeeze(-1)

    def predict(self, obs: np.ndarray) -> float:
        with torch.no_grad():
            device = next(self.parameters()).device
            obs_t = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
            return self.forward(obs_t).item()

    def predict_batch(self, obs_batch: np.ndarray) -> np.ndarray:
        with torch.no_grad():
            device = next(self.parameters()).device
            obs_t = torch.tensor(obs_batch, dtype=torch.float32, device=device)
            return self.forward(obs_t).cpu().numpy()


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
    all_turns_remaining = []
    all_q_values = []
    p1_wins = p2_wins = draws = 0

    for game_idx in range(n_games):
        game_seed = rng.randint(0, 2**31)
        game_rng = random.Random(game_seed)

        # mixed opponent
        if game_rng.random() < 0.5:
            opp = smart
        else:
            opp = maxdmg

        # P1 picks its policy: 60% strong (2-ply search), 20% smart, 20% maxdmg.
        # Diversifying P1's strength gives a healthier mix of P1-wins and
        # P1-loses labels (P1=2-ply usually wins, P1=maxdmg often loses) without
        # needing a P2-perspective search (search_2ply is P1-only on the Rust
        # side).
        r = game_rng.random()
        p1_strong = r < 0.6
        p1_baseline = None if p1_strong else (smart if r < 0.8 else maxdmg)

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
        game_q_values = []  # 2-ply search V(s) per turn, in [-1, 1]
        n_resolved = 0  # count of resolve_turn calls; turns_remaining = total - i
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

            p2_adapter = _RustPlayerAdapter(rs_battle.p2)
            p1_adapter = _RustPlayerAdapter(rs_battle.p1)

            # Always compute V(s) via 2-ply search for Q-target labeling.
            # Reuse the search when p1 is the strong player; otherwise run
            # a separate labeling-only search.
            if len(valid) == 1:
                # No real decision; use heuristic position eval as label.
                q_value = float(ce.evaluate_position(rs_battle))
                _, opp_hidden = opp_model.predict_single(obs, opp_hidden)
                if p1_strong:
                    a1 = valid[0]
                else:
                    p1_action = p1_baseline.act(p1_adapter, p2_adapter)
                    a1 = _py_action_to_int(p1_action)
            else:
                opp_actions = _get_opp_actions(
                    rs_battle, obs, opp_model, opp_hidden, 5)
                _, opp_hidden = opp_model.predict_single(obs, opp_hidden)
                ranked = ce.search_2ply(
                    rs_battle, valid, opp_actions, opp_actions,
                    base_seed=turn * 1000)
                q_value = float(ranked[0][1]) if ranked else 0.0
                if p1_strong:
                    a1 = ranked[0][0] if ranked else valid[0]
                else:
                    p1_action = p1_baseline.act(p1_adapter, p2_adapter)
                    a1 = _py_action_to_int(p1_action)
            game_q_values.append(q_value)

            p2_action = opp_policy(p2_adapter, p1_adapter)
            a2 = _py_action_to_int(p2_action)

            rs_battle.resolve_turn(a1, a2)
            n_resolved += 1
            _handle_forced_switches_rs(rs_battle, opp_policy)

            if rs_battle.p1.must_switch:
                sw = _first_alive_bench_idx(rs_battle.p1)
                if sw is not None:
                    rs_battle.resolve_forced_switches(sw, None)

        # assign outcome to every obs in this game (P1 perspective)
        if rs_battle.winner == 1:
            outcome = 1.0
            p1_wins += 1
        elif rs_battle.winner == 2:
            outcome = -1.0
            p2_wins += 1
        else:
            outcome = 0.0
            draws += 1

        # turns_remaining[i] = number of turn-resolutions after the i-th obs
        # until the game ends. The last obs has turns_remaining=1 if the final
        # turn was resolved, or 0 if we broke without resolving (no valid acts).
        for i, obs in enumerate(game_obs):
            tr = max(0, n_resolved - i)
            all_obs.append(obs)
            all_outcomes.append(outcome)
            all_turns_remaining.append(tr)
            all_q_values.append(game_q_values[i])

        if (game_idx + 1) % 500 == 0:
            print(f"  {game_idx + 1}/{n_games} games "
                  f"(P1 {p1_wins} / P2 {p2_wins} / D {draws}), "
                  f"{len(all_obs)} samples")

    obs_arr = np.array(all_obs, dtype=np.float32)
    outcome_arr = np.array(all_outcomes, dtype=np.float32)
    turns_remaining_arr = np.array(all_turns_remaining, dtype=np.int32)
    q_value_arr = np.array(all_q_values, dtype=np.float32)

    np.savez_compressed(out_path, obs=obs_arr, outcomes=outcome_arr,
                        turns_remaining=turns_remaining_arr,
                        q_values=q_value_arr)
    print(f"Saved {len(obs_arr)} samples to {out_path}")
    print(f"  turns_remaining: min={turns_remaining_arr.min()}, "
          f"max={turns_remaining_arr.max()}, "
          f"mean={turns_remaining_arr.mean():.1f}")
    print(f"  q_values: min={q_value_arr.min():.3f}, "
          f"max={q_value_arr.max():.3f}, "
          f"mean={q_value_arr.mean():.3f}, "
          f"|q| mean={np.abs(q_value_arr).mean():.3f}")


# ============================================================
# TRAINING
# ============================================================

def train_value_net(
    data_path: str = "value_data.npz",
    save_path: str = "value_net.pt",
    epochs: int = 30,
    batch_size: int = 4096,
    lr: float = 1e-3,
    device: str | None = None,
    hidden: int = 256,
    n_layers: int = 4,
    dropout: float = 0.2,
    target_clamp: float = 0.9,
    weight_decay: float = 1e-4,
    gamma: float = 1.0,
    q_target: bool = False,
):
    """Train the feedforward value network.

    target_clamp: clamps targets to ±clamp before the tanh head,
    preventing the saturation that broke the previous integration attempt.

    gamma: discount applied to outcome via gamma^turns_remaining. Ignored
    when q_target=True. gamma<1 softens labels for far-from-terminal
    positions; requires turns_remaining in the npz.

    q_target: if True, use the saved 2-ply search V(s) as the regression
    target instead of game outcomes. Naturally calibrated (mid-game values
    are mid-range), more informative gradient than raw outcomes. Requires
    q_values in the npz.
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    print(f"Loading data from {data_path}...")
    npz = np.load(data_path)
    obs = torch.tensor(npz["obs"], dtype=torch.float32)
    outcomes = torch.tensor(npz["outcomes"], dtype=torch.float32)
    print(f"  {len(obs)} samples, obs_dim={obs.shape[1]}")

    # check outcome distribution
    wins = (outcomes > 0).sum().item()
    losses = (outcomes < 0).sum().item()
    draws = (outcomes == 0).sum().item()
    print(f"  Outcomes: {wins} wins, {losses} losses, {draws} draws")

    if q_target:
        if "q_values" not in npz:
            raise RuntimeError(
                "q_target requires q_values in the npz; regenerate the data "
                "with the latest value_net.py.")
        targets = torch.tensor(npz["q_values"], dtype=torch.float32)
        print(f"  Target: 2-ply search V(s) "
              f"(mean={targets.mean():.3f}, std={targets.std():.3f}, "
              f"min={targets.min():.3f}, max={targets.max():.3f})")
    elif gamma != 1.0:
        # discount: target = outcome * gamma^turns_remaining
        if "turns_remaining" not in npz:
            raise RuntimeError(
                "gamma<1 requires turns_remaining in the npz; regenerate the "
                "data with the latest value_net.py.")
        tr = torch.tensor(npz["turns_remaining"], dtype=torch.float32)
        discount = gamma ** tr
        targets = outcomes * discount
        print(f"  Gamma: {gamma} (mean discount={discount.mean():.3f}, "
              f"min={discount.min():.3f}, max={discount.max():.3f})")
    else:
        targets = outcomes.clone()

    # clamp targets — keeps tanh head out of the rails
    targets = targets.clamp(-target_clamp, target_clamp)
    print(f"  Target clamp: ±{target_clamp} "
          f"(target std={targets.std():.3f}, |target| mean={targets.abs().mean():.3f})")

    # train/val split
    n = len(obs)
    perm = torch.randperm(n)
    val_size = n // 10
    train_ds = TensorDataset(obs[perm[val_size:]], targets[perm[val_size:]])
    val_ds = TensorDataset(obs[perm[:val_size]], targets[perm[:val_size]])
    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                          pin_memory=(device == "cuda"))
    val_dl = DataLoader(val_ds, batch_size=batch_size,
                        pin_memory=(device == "cuda"))

    model = FeedforwardValueNet(obs_dim=obs.shape[1], hidden=hidden,
                                n_layers=n_layers, dropout=dropout).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  Model: {hidden}x{n_layers}, dropout={dropout}, {n_params:,} params")

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    best_val_loss = float("inf")
    print(f"Training for {epochs} epochs...")

    for epoch in range(epochs):
        # train
        model.train()
        train_loss = 0
        train_n = 0
        for batch_obs, batch_out in train_dl:
            batch_obs = batch_obs.to(device, non_blocking=True)
            batch_out = batch_out.to(device, non_blocking=True)
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
        val_sat_high = 0  # predictions in (target_clamp, 1)
        val_sat_low = 0   # predictions in (-1, -target_clamp)
        with torch.no_grad():
            for batch_obs, batch_out in val_dl:
                batch_obs = batch_obs.to(device, non_blocking=True)
                batch_out = batch_out.to(device, non_blocking=True)
                pred = model(batch_obs)
                loss = F.mse_loss(pred, batch_out)
                val_loss += loss.item() * len(batch_obs)
                val_n += len(batch_obs)
                val_correct += ((pred > 0) == (batch_out > 0)).sum().item()
                val_sat_high += (pred > target_clamp).sum().item()
                val_sat_low += (pred < -target_clamp).sum().item()

        scheduler.step()

        marker = ""
        if val_loss / val_n < best_val_loss:
            best_val_loss = val_loss / val_n
            torch.save({
                "model": model.state_dict(),
                "obs_dim": obs.shape[1],
                "hidden": hidden,
                "n_layers": n_layers,
                "dropout": dropout,
                "target_clamp": target_clamp,
                "gamma": gamma,
                "q_target": q_target,
            }, save_path)
            marker = " *best*"

        val_acc = val_correct / val_n
        sat_frac = (val_sat_high + val_sat_low) / val_n
        print(f"  Epoch {epoch+1:2d}: train_loss={train_loss/train_n:.4f} "
              f"val_loss={val_loss/val_n:.4f} sign_acc={val_acc:.3f} "
              f"sat={sat_frac:.1%}{marker}")

    print(f"Best val loss: {best_val_loss:.4f}")
    print(f"Saved to {save_path}")


def load_value_net(path: str, device: str = "cpu") -> "FeedforwardValueNet":
    """Load a value net checkpoint with embedded architecture metadata.

    Falls back to defaults for old (raw state_dict) checkpoints.
    """
    ckpt = torch.load(path, map_location=device, weights_only=True)
    if isinstance(ckpt, dict) and "model" in ckpt:
        model = FeedforwardValueNet(
            obs_dim=ckpt.get("obs_dim", OBS_SIZE),
            hidden=ckpt.get("hidden", 256),
            n_layers=ckpt.get("n_layers", 4),
            dropout=ckpt.get("dropout", 0.2),
        )
        model.load_state_dict(ckpt["model"])
    else:
        model = FeedforwardValueNet()
        model.load_state_dict(ckpt)
    model.to(device).eval()
    return model


# ============================================================
# EVALUATION
# ============================================================

def evaluate_value_net(model_path: str = "value_net.pt", data_path: str = "value_data.npz",
                       device: str | None = None):
    """Evaluate the value network's predictions."""
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    npz = np.load(data_path)
    obs = torch.tensor(npz["obs"], dtype=torch.float32)
    outcomes = torch.tensor(npz["outcomes"], dtype=torch.float32)

    model = load_value_net(model_path, device=device)

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
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--data", type=str, default="value_data.npz")
    parser.add_argument("--model", type=str, default="value_net.pt")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--hidden", type=int, default=256)
    parser.add_argument("--n-layers", type=int, default=4)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--target-clamp", type=float, default=0.9)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--gamma", type=float, default=1.0,
                        help="discount: target = outcome * gamma^turns_remaining")
    parser.add_argument("--q-target", action="store_true",
                        help="use 2-ply search V(s) as target instead of "
                             "discounted game outcome")
    parser.add_argument("--device", type=str, default=None,
                        help="cpu / cuda; auto-detects if omitted")
    args = parser.parse_args()

    if args.generate:
        generate_value_data(n_games=args.n_games, seed=args.seed,
                            out_path=args.data)
    elif args.train:
        train_value_net(data_path=args.data, save_path=args.model,
                        epochs=args.epochs, batch_size=args.batch_size,
                        lr=args.lr, device=args.device, hidden=args.hidden,
                        n_layers=args.n_layers, dropout=args.dropout,
                        target_clamp=args.target_clamp,
                        weight_decay=args.weight_decay,
                        gamma=args.gamma,
                        q_target=args.q_target)
    elif args.evaluate:
        evaluate_value_net(args.model, data_path=args.data, device=args.device)
