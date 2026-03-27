#!/usr/bin/env python3
# offline RL training for Pokemon battles
# trains a transformer on pre-collected game data without environment interaction
# inspired by "Human-Level Competitive Pokemon via Offline RL with Transformers"
#
# key difference from pretrain_simple.py:
#   - filters/weights by game outcome (learn from winning play)
#   - reward-to-go conditioning (each position knows how much future reward remains)
#   - scales to large datasets (50k+ games)
#
# Usage:
#   python offline_rl.py --train --data search_3ply_data.pkl --device cuda --epochs 30
#   python offline_rl.py --evaluate --device cuda --n-games 200

import argparse
import pickle
import random

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from gym_env.obs_builder import OBS_SIZE


class OfflineTransformerPolicy(nn.Module):
    """Transformer policy trained via offline RL.

    Optionally conditions on reward-to-go: each timestep gets an
    extra scalar input indicating how much future reward is expected.
    At inference, we set reward-to-go to +1 (asking for winning play).
    """

    def __init__(self, obs_dim=OBS_SIZE, d_model=256, n_heads=4, n_layers=3,
                 max_seq=64, use_rtg=True):
        super().__init__()
        self.use_rtg = use_rtg
        input_dim = obs_dim + (1 if use_rtg else 0)

        self.embed = nn.Sequential(
            nn.Linear(input_dim, d_model),
            nn.ReLU(),
            nn.Linear(d_model, d_model),
            nn.ReLU(),
        )
        self.pos = nn.Embedding(max_seq, d_model)
        layer = nn.TransformerEncoderLayer(
            d_model, n_heads, dim_feedforward=512,
            batch_first=True, dropout=0.1,
        )
        self.transformer = nn.TransformerEncoder(layer, n_layers)
        self.policy_head = nn.Sequential(
            nn.Linear(d_model, 256), nn.ReLU(), nn.Linear(256, 10),
        )
        self.value_head = nn.Sequential(
            nn.Linear(d_model, 256), nn.ReLU(), nn.Linear(256, 1), nn.Tanh(),
        )
        self.max_seq = max_seq

    def forward(self, obs_seq, rtg=None, mask=None):
        """
        Args:
            obs_seq: (batch, seq_len, obs_dim)
            rtg: (batch, seq_len, 1) reward-to-go per timestep, or None
            mask: (batch, seq_len, 10) action masks, or None
        Returns:
            logits: (batch, seq_len, 10)
            values: (batch, seq_len, 1)
        """
        b, t, _ = obs_seq.shape

        if self.use_rtg and rtg is not None:
            x = torch.cat([obs_seq, rtg], dim=-1)
        else:
            # at inference without rtg, append +1 (ask for winning play)
            if self.use_rtg:
                ones = torch.ones(b, t, 1, device=obs_seq.device)
                x = torch.cat([obs_seq, ones], dim=-1)
            else:
                x = obs_seq

        x = self.embed(x)
        x = x + self.pos(torch.arange(t, device=x.device))
        causal = torch.triu(torch.ones(t, t, device=x.device), diagonal=1).bool()
        x = self.transformer(x, mask=causal, is_causal=True)
        logits = self.policy_head(x)
        if mask is not None:
            logits = logits + (1 - mask) * -1e9
        values = self.value_head(x)
        return logits, values

    def predict_action(self, obs_buffer, action_mask, deterministic=True):
        """Predict action from observation buffer (for inference)."""
        with torch.no_grad():
            obs_t = torch.tensor(np.array(obs_buffer), dtype=torch.float32).unsqueeze(0)
            if next(self.parameters()).is_cuda:
                obs_t = obs_t.cuda()
            mask_t = None
            if action_mask is not None:
                mask_t = torch.tensor(action_mask, dtype=torch.float32)
                mask_t = mask_t.unsqueeze(0).unsqueeze(0).expand(-1, obs_t.shape[1], -1)
                if next(self.parameters()).is_cuda:
                    mask_t = mask_t.cuda()
            # rtg=None triggers the +1 conditioning (ask for winning play)
            logits, _ = self.forward(obs_t, rtg=None, mask=mask_t)
            last_logits = logits[0, -1, :]
            if deterministic:
                return last_logits.argmax().item()
            probs = F.softmax(last_logits, dim=0)
            return torch.multinomial(probs, 1).item()


def compute_rtg_for_game(n_steps, outcome, gamma=0.99):
    """Compute discounted reward-to-go for each timestep.

    Terminal reward is the game outcome (+1 win, -1 loss).
    Intermediate rewards are 0 (all reward comes at the end).
    RTG at step t = gamma^(T-t) * outcome
    """
    rtg = np.zeros(n_steps, dtype=np.float32)
    for t in range(n_steps):
        rtg[t] = (gamma ** (n_steps - 1 - t)) * outcome
    return rtg


def prepare_data(data_path, win_only=False, min_outcome=0.0):
    """Load and prepare training data.

    Args:
        data_path: path to pickle file with sequences
        win_only: if True, only include games the search agent won
        min_outcome: minimum outcome to include (0.0 = include draws, 0.5 = wins only)
    """
    print(f"Loading {data_path}...")
    with open(data_path, "rb") as f:
        sequences = pickle.load(f)
    print(f"  {len(sequences)} total sequences")

    # detect format: 3-tuple (obs, act, mask) or 4-tuple (obs, act, mask, outcome)
    has_outcomes = len(sequences[0]) >= 4
    if not has_outcomes:
        print("  WARNING: no outcomes in data, treating all as wins")

    # filter and compute RTG
    processed = []
    wins = losses = draws = 0
    for seq in sequences:
        obs_seq = seq[0]
        act_seq = seq[1]
        mask_seq = seq[2] if len(seq) >= 3 else np.ones((len(act_seq), 10), dtype=np.float32)
        outcome = seq[3] if len(seq) >= 4 else 1.0

        if outcome > 0:
            wins += 1
        elif outcome < 0:
            losses += 1
        else:
            draws += 1

        # filter by outcome
        if win_only and outcome <= 0:
            continue
        if outcome < min_outcome:
            continue

        n = min(len(obs_seq), len(act_seq), len(mask_seq))
        if n < 2:
            continue

        rtg = compute_rtg_for_game(n, outcome)

        processed.append({
            "obs": obs_seq[:n],
            "actions": act_seq[:n],
            "masks": mask_seq[:n],
            "rtg": rtg,
            "outcome": outcome,
        })

    print(f"  Outcomes: {wins} wins, {losses} losses, {draws} draws")
    print(f"  After filtering: {len(processed)} sequences")
    return processed


def train(data_path="search_3ply_data.pkl", device="cpu", epochs=30,
          lr=1e-3, max_seq=64, n_layers=3, save_path="offline_policy.pt",
          win_only=True, use_rtg=True, accum_steps=16):
    """Train offline RL transformer."""

    data = prepare_data(data_path, win_only=win_only)

    random.seed(42)
    random.shuffle(data)
    val_size = max(1, len(data) // 10)
    val = data[:val_size]
    train_data = data[val_size:]

    model = OfflineTransformerPolicy(
        d_model=256, n_heads=4, n_layers=n_layers,
        max_seq=max_seq, use_rtg=use_rtg,
    ).to(device)
    params = sum(p.numel() for p in model.parameters())
    print(f"  {params:,} parameters, use_rtg={use_rtg}")

    opt = torch.optim.Adam(model.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)

    best_val = 0
    for epoch in range(epochs):
        model.train()
        random.shuffle(train_data)
        tc = tt = 0
        tl = 0.0
        opt.zero_grad()

        for i, sample in enumerate(train_data):
            obs = sample["obs"]
            acts = sample["actions"]
            masks = sample["masks"]
            rtg = sample["rtg"]

            n = min(len(obs), max_seq)
            obs_t = torch.tensor(obs[:n], dtype=torch.float32, device=device).unsqueeze(0)
            act_t = torch.tensor(acts[:n], dtype=torch.long, device=device)
            mask_t = torch.tensor(masks[:n], dtype=torch.float32, device=device).unsqueeze(0)

            rtg_t = None
            if use_rtg:
                rtg_t = torch.tensor(rtg[:n], dtype=torch.float32, device=device)
                rtg_t = rtg_t.unsqueeze(0).unsqueeze(-1)  # (1, n, 1)

            logits, values = model(obs_t, rtg=rtg_t, mask=mask_t)
            logits = logits.squeeze(0)

            # policy loss: cross-entropy on expert actions
            policy_loss = F.cross_entropy(logits, act_t)

            # value loss: predict outcome
            value_target = torch.full((n,), sample["outcome"], device=device)
            value_loss = F.mse_loss(values.squeeze(0).squeeze(-1), value_target)

            loss = (policy_loss + 0.5 * value_loss) / accum_steps
            loss.backward()

            tc += (logits.argmax(1) == act_t).sum().item()
            tt += n
            tl += policy_loss.item() * n

            if (i + 1) % accum_steps == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
                opt.zero_grad()

        # final step
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        opt.zero_grad()

        # validate
        model.eval()
        vc = vt = 0
        with torch.no_grad():
            for sample in val:
                obs = sample["obs"]
                acts = sample["actions"]
                masks = sample["masks"]
                rtg = sample["rtg"]
                n = min(len(obs), max_seq)

                obs_t = torch.tensor(obs[:n], dtype=torch.float32, device=device).unsqueeze(0)
                act_t = torch.tensor(acts[:n], dtype=torch.long, device=device)
                mask_t = torch.tensor(masks[:n], dtype=torch.float32, device=device).unsqueeze(0)
                rtg_t = None
                if use_rtg:
                    rtg_t = torch.tensor(rtg[:n], dtype=torch.float32, device=device)
                    rtg_t = rtg_t.unsqueeze(0).unsqueeze(-1)

                logits, _ = model(obs_t, rtg=rtg_t, mask=mask_t)
                logits = logits.squeeze(0)
                vc += (logits.argmax(1) == act_t).sum().item()
                vt += n

        sched.step()
        val_acc = vc / vt
        marker = ""
        if val_acc > best_val:
            best_val = val_acc
            torch.save(model.state_dict(), save_path)
            marker = " *best*"
        print(f"Epoch {epoch+1:2d}: train={tc/tt:.3f} loss={tl/tt:.3f} "
              f"val={val_acc:.3f}{marker}")

    print(f"Best val: {best_val:.3f}, saved to {save_path}")


def evaluate(model_path="offline_policy.pt", device="cpu", n_games=100,
             max_seq=64, n_layers=3, use_rtg=True):
    """Evaluate offline RL policy vs baselines."""
    from engine.types import TypeChart
    from engine.data_loader import DataStore
    from engine.battle_state import BattleState
    from engine.player_state import PlayerState
    from engine.turn_engine import resolve_turn, resolve_forced_switches
    from engine.actions import Switch, UseMove, Struggle
    from training.baselines import SmartAgent, MaxDamageAgent
    from gym_env.team_builder import build_team
    from gym_env.obs_builder import build_observation

    tc = TypeChart.load()
    data = DataStore()

    model = OfflineTransformerPolicy(
        d_model=256, n_heads=4, n_layers=n_layers,
        max_seq=max_seq, use_rtg=use_rtg,
    ).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    model.eval()

    for bl_name, bl_cls in [("max_damage", MaxDamageAgent), ("smart", SmartAgent)]:
        bl = bl_cls(tc, seed=99) if bl_name == "smart" else bl_cls(tc)
        wins = 0
        for i in range(n_games):
            rng = random.Random(i)
            t1 = build_team(data, rng=rng, tier="ou")
            t2 = build_team(data, rng=random.Random(i + 1000), tier="ou")
            battle = BattleState(
                p1=PlayerState(team=t1), p2=PlayerState(team=t2),
                rng=random.Random(i + 2000))

            obs_buffer = []
            for turn in range(100):
                if battle.is_over:
                    break
                obs = build_observation(battle.p1, battle.p2, tc, turn=battle.turn,
                                        weather=battle.weather,
                                        weather_turns=battle.weather_turns)
                mask = np.array(battle.p1.valid_action_mask(battle.p2, type_chart=tc),
                                dtype=np.float32)
                obs_buffer.append(obs)
                if len(obs_buffer) > max_seq:
                    obs_buffer = obs_buffer[-max_seq:]

                action_int = model.predict_action(obs_buffer, mask, deterministic=True)

                if action_int < 4:
                    active = battle.p1.active
                    if not active.has_any_pp():
                        p1_action = Struggle()
                    elif action_int < len(active.move_slots) and active.move_slots[action_int].has_pp:
                        p1_action = UseMove(slot_index=action_int)
                    else:
                        for j, slot in enumerate(active.move_slots):
                            if slot.has_pp:
                                p1_action = UseMove(slot_index=j)
                                break
                        else:
                            p1_action = Struggle()
                else:
                    p1_action = Switch(team_index=action_int - 4)

                p2_action = bl.act(battle.p2, battle.p1)
                resolve_turn(battle, p1_action, p2_action, tc)

                sw1 = sw2 = None
                if battle.p1.must_switch:
                    sw_obs = build_observation(battle.p1, battle.p2, tc, turn=battle.turn,
                                                weather=battle.weather,
                                                weather_turns=battle.weather_turns)
                    sw_mask = np.array(battle.p1.valid_action_mask(battle.p2, type_chart=tc),
                                        dtype=np.float32)
                    obs_buffer.append(sw_obs)
                    if len(obs_buffer) > max_seq:
                        obs_buffer = obs_buffer[-max_seq:]
                    sw_int = model.predict_action(obs_buffer, sw_mask, deterministic=True)
                    if sw_int >= 4:
                        sw1 = Switch(team_index=sw_int - 4)
                    if sw1 is None:
                        for j, p in enumerate(battle.p1.team):
                            if j != battle.p1.active_index and not p.is_fainted:
                                sw1 = Switch(team_index=j)
                                break
                if battle.p2.must_switch:
                    sw2_a = bl.act(battle.p2, battle.p1)
                    sw2 = sw2_a if isinstance(sw2_a, Switch) else None
                    if sw2 is None:
                        for j, p in enumerate(battle.p2.team):
                            if j != battle.p2.active_index and not p.is_fainted:
                                sw2 = Switch(team_index=j)
                                break
                if sw1 or sw2:
                    resolve_forced_switches(battle, sw1, sw2)

            if battle.winner == 1:
                wins += 1
        print(f"  vs {bl_name:12s}: {wins}/{n_games} ({wins/n_games*100:.1f}%)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Offline RL for Pokemon battles")
    parser.add_argument("--train", action="store_true")
    parser.add_argument("--evaluate", action="store_true")
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--n-games", type=int, default=200)
    parser.add_argument("--data", type=str, default="search_3ply_data.pkl")
    parser.add_argument("--model", type=str, default="offline_policy.pt")
    parser.add_argument("--n-layers", type=int, default=3)
    parser.add_argument("--seq-len", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--win-only", action="store_true",
                        help="Only train on games the search agent won")
    parser.add_argument("--no-rtg", action="store_true",
                        help="Disable reward-to-go conditioning")
    args = parser.parse_args()

    if args.train:
        train(args.data, args.device, args.epochs, lr=args.lr,
              max_seq=args.seq_len, n_layers=args.n_layers,
              save_path=args.model, win_only=args.win_only,
              use_rtg=not args.no_rtg)
    elif args.evaluate:
        evaluate(args.model, args.device, args.n_games,
                 max_seq=args.seq_len, n_layers=args.n_layers,
                 use_rtg=not args.no_rtg)
    else:
        print("Use --train or --evaluate")
