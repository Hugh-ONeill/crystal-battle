#!/usr/bin/env python3
# simple transformer pre-training -- uses PyTorch's built-in TransformerEncoder
# bypasses our custom TransformerBattlePolicy to isolate training issues
#
# Usage:
#   .venv/bin/python pretrain_simple.py [--device cuda] [--epochs 30]

import argparse
import pickle
import random

import torch
import torch.nn as nn
import torch.nn.functional as F

from gym_env.obs_builder import OBS_SIZE


class SimpleTransformerPolicy(nn.Module):
    """Minimal transformer policy for Pokemon battles.

    Simple embedding + PyTorch TransformerEncoder + policy/value heads.
    No custom attention extractor -- the transformer handles all attention.
    """

    def __init__(self, obs_dim=OBS_SIZE, d_model=256, n_heads=4, n_layers=3,
                 max_seq=32):
        super().__init__()
        self.embed = nn.Sequential(
            nn.Linear(obs_dim, d_model),
            nn.ReLU(),
            nn.Linear(d_model, d_model),
            nn.ReLU(),
        )
        self.pos = nn.Embedding(max_seq, d_model)
        layer = nn.TransformerEncoderLayer(
            d_model, n_heads, dim_feedforward=512,
            batch_first=True, dropout=0.0,
        )
        self.transformer = nn.TransformerEncoder(layer, n_layers)
        self.policy_head = nn.Sequential(
            nn.Linear(d_model, 256), nn.ReLU(), nn.Linear(256, 10),
        )
        self.value_head = nn.Sequential(
            nn.Linear(d_model, 256), nn.ReLU(), nn.Linear(256, 1), nn.Tanh(),
        )
        self.max_seq = max_seq

    def forward(self, obs_seq, mask=None):
        """Forward pass.

        Args:
            obs_seq: (batch, seq_len, obs_dim)
            mask: (batch, seq_len, 10) action masks, or None

        Returns:
            logits: (batch, seq_len, 10)
            values: (batch, seq_len, 1)
        """
        b, t, _ = obs_seq.shape
        x = self.embed(obs_seq)
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
            obs_t = torch.tensor(obs_buffer, dtype=torch.float32).unsqueeze(0)
            if next(self.parameters()).is_cuda:
                obs_t = obs_t.cuda()
            mask_t = None
            if action_mask is not None:
                mask_t = torch.tensor(action_mask, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
                if next(self.parameters()).is_cuda:
                    mask_t = mask_t.cuda()
                # expand mask to seq_len
                mask_t = mask_t.expand(-1, obs_t.shape[1], -1)
            logits, _ = self.forward(obs_t, mask_t)
            # take last timestep
            last_logits = logits[0, -1, :]
            if deterministic:
                return last_logits.argmax().item()
            probs = F.softmax(last_logits, dim=0)
            return torch.multinomial(probs, 1).item()


def pretrain(data_path="expert_sequences.pkl", device="cpu", epochs=30,
             lr=1e-3, max_seq=32, n_layers=3, save_path="simple_transformer.pt"):
    """Pre-train on SmartAgent sequences."""
    print(f"Loading {data_path}...")
    with open(data_path, "rb") as f:
        seqs = pickle.load(f)
    print(f"  {len(seqs)} sequences")

    random.seed(42)
    random.shuffle(seqs)
    val = seqs[:400]
    train = seqs[400:]

    model = SimpleTransformerPolicy(
        d_model=256, n_heads=4, n_layers=n_layers, max_seq=max_seq,
    ).to(device)
    params = sum(p.numel() for p in model.parameters())
    print(f"  {params:,} parameters")

    opt = torch.optim.Adam(model.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)

    best_val = 0
    for epoch in range(epochs):
        model.train()
        random.shuffle(train)
        tc = tt = 0
        for seq in train:
            obs, acts = seq[0], seq[1]
            n = min(len(obs), len(acts), max_seq)
            if n < 2:
                continue
            obs_t = torch.tensor(obs[:n], dtype=torch.float32, device=device).unsqueeze(0)
            act_t = torch.tensor(acts[:n], dtype=torch.long, device=device)
            mask_t = None
            if len(seq) >= 3:
                mask_t = torch.tensor(
                    seq[2][:n], dtype=torch.float32, device=device,
                ).unsqueeze(0)
            logits, _ = model(obs_t, mask_t)
            logits = logits.squeeze(0)
            loss = F.cross_entropy(logits, act_t)
            opt.zero_grad()
            loss.backward()
            opt.step()
            tc += (logits.argmax(1) == act_t).sum().item()
            tt += n

        model.eval()
        vc = vt = 0
        with torch.no_grad():
            for seq in val:
                obs, acts = seq[0], seq[1]
                n = min(len(obs), len(acts), max_seq)
                if n < 2:
                    continue
                obs_t = torch.tensor(obs[:n], dtype=torch.float32, device=device).unsqueeze(0)
                act_t = torch.tensor(acts[:n], dtype=torch.long, device=device)
                mask_t = None
                if len(seq) >= 3:
                    mask_t = torch.tensor(
                        seq[2][:n], dtype=torch.float32, device=device,
                    ).unsqueeze(0)
                logits, _ = model(obs_t, mask_t)
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
        print(f"Epoch {epoch + 1:2d}: train={tc / tt:.3f} val={val_acc:.3f}{marker}")

    print(f"Best val: {best_val:.3f}, saved to {save_path}")


def evaluate(model_path="simple_transformer.pt", device="cpu", n_games=100):
    """Evaluate pre-trained transformer vs baselines."""
    import numpy as np
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

    model = SimpleTransformerPolicy().to(device)
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
                if len(obs_buffer) > 32:
                    obs_buffer = obs_buffer[-32:]

                action_int = model.predict_action(
                    np.array(obs_buffer), mask, deterministic=True)

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
                    sw_int = model.predict_action(
                        np.array(obs_buffer[-32:]), sw_mask, deterministic=True)
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
        print(f"  vs {bl_name:12s}: {wins}/{n_games} ({wins / n_games * 100:.1f}%)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--pretrain", action="store_true")
    parser.add_argument("--evaluate", action="store_true")
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--n-games", type=int, default=100)
    parser.add_argument("--data", type=str, default="expert_sequences.pkl")
    parser.add_argument("--model", type=str, default="simple_transformer.pt")
    parser.add_argument("--n-layers", type=int, default=3)
    parser.add_argument("--seq-len", type=int, default=32)
    args = parser.parse_args()

    if args.pretrain:
        pretrain(args.data, args.device, args.epochs, max_seq=args.seq_len,
                 n_layers=args.n_layers, save_path=args.model)
    elif args.evaluate:
        evaluate(args.model, args.device, args.n_games)
    else:
        print("Use --pretrain or --evaluate")
