# imitation learning: generate sequential expert data, pre-train full LSTM policy
# Usage:
#   generate:   python training/imitation.py --generate --n-games 5000
#   pretrain:   python training/imitation.py --pretrain --data expert_sequences.pkl
#   init-ppo:   python training/imitation.py --init-ppo --model imitation_seq

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
from engine.actions import Switch, UseMove, Struggle
from training.baselines import SmartAgent, MaxDamageAgent
from gym_env.team_builder import build_team
from gym_env.obs_builder import build_observation, OBS_SIZE


# ============================================================
# SEQUENTIAL DATA GENERATION
# ============================================================

def generate_sequences(
    n_games: int = 5000,
    seed: int = 42,
    out_path: str = "expert_sequences.pkl",
    opponent: str = "mixed",
):
    """Generate full game sequences from SmartAgent."""
    tc = TypeChart.load()
    data = DataStore()
    rng = random.Random(seed)

    sequences = []  # list of (obs_seq, action_seq, mask_seq)

    for game_idx in range(n_games):
        game_seed = rng.randint(0, 2**31)
        game_rng = random.Random(game_seed)

        smart = SmartAgent(tc, seed=game_seed)

        if opponent == "mixed":
            if game_rng.random() < 0.5:
                opp = SmartAgent(tc, seed=game_seed + 1000)
            else:
                opp = MaxDamageAgent(tc)
        elif opponent == "smart":
            opp = SmartAgent(tc, seed=game_seed + 1000)
        else:
            opp = MaxDamageAgent(tc)

        t1 = build_team(data, rng=random.Random(game_seed + 100), tier="ou")
        t2 = build_team(data, rng=random.Random(game_seed + 200), tier="ou")
        battle = BattleState(
            p1=PlayerState(team=t1), p2=PlayerState(team=t2),
            rng=random.Random(game_seed + 300),
        )

        obs_seq = []
        action_seq = []
        mask_seq = []

        for turn in range(100):
            if battle.is_over:
                break

            obs = build_observation(
                battle.p1, battle.p2, tc, turn=battle.turn,
                weather=battle.weather, weather_turns=battle.weather_turns,
            )
            mask = np.array(
                battle.p1.valid_action_mask(battle.p2, type_chart=tc),
                dtype=np.float32,
            )
            action = smart.act(battle.p1, battle.p2)

            if isinstance(action, UseMove):
                action_int = action.slot_index
            elif isinstance(action, Switch):
                action_int = 4 + action.team_index
            elif isinstance(action, Struggle):
                action_int = 0
            else:
                a2 = opp.act(battle.p2, battle.p1)
                resolve_turn(battle, action, a2, tc)
                continue

            if not mask[action_int]:
                continue

            obs_seq.append(obs)
            action_seq.append(action_int)
            mask_seq.append(mask)

            a2 = opp.act(battle.p2, battle.p1)
            resolve_turn(battle, action, a2, tc)

            # forced switches
            sw1 = sw2 = None
            if battle.p1.must_switch:
                sw1_a = smart.act(battle.p1, battle.p2)
                sw1 = sw1_a if isinstance(sw1_a, Switch) else None
                if sw1 is None:
                    for i, p in enumerate(battle.p1.team):
                        if i != battle.p1.active_index and not p.is_fainted:
                            sw1 = Switch(team_index=i); break
                if sw1 is not None:
                    obs_sw = build_observation(
                        battle.p1, battle.p2, tc, turn=battle.turn,
                        weather=battle.weather, weather_turns=battle.weather_turns,
                    )
                    mask_sw = np.array(
                        battle.p1.valid_action_mask(battle.p2, type_chart=tc),
                        dtype=np.float32,
                    )
                    obs_seq.append(obs_sw)
                    action_seq.append(4 + sw1.team_index)
                    mask_seq.append(mask_sw)

            if battle.p2.must_switch:
                sw2_a = opp.act(battle.p2, battle.p1)
                sw2 = sw2_a if isinstance(sw2_a, Switch) else None
                if sw2 is None:
                    for i, p in enumerate(battle.p2.team):
                        if i != battle.p2.active_index and not p.is_fainted:
                            sw2 = Switch(team_index=i); break
            if sw1 or sw2:
                resolve_forced_switches(battle, sw1, sw2)

        if len(obs_seq) >= 3:  # skip very short games
            sequences.append((
                np.array(obs_seq, dtype=np.float32),
                np.array(action_seq, dtype=np.int64),
                np.array(mask_seq, dtype=np.float32),
            ))

        if (game_idx + 1) % 500 == 0:
            total_steps = sum(len(s[0]) for s in sequences)
            print(f"  {game_idx + 1}/{n_games} games, {len(sequences)} sequences, "
                  f"{total_steps} total steps")

    total_steps = sum(len(s[0]) for s in sequences)
    with open(out_path, "wb") as f:
        pickle.dump(sequences, f)
    print(f"Saved {len(sequences)} sequences ({total_steps} steps) to {out_path}")


# ============================================================
# SEQUENCE PRE-TRAINING (full LSTM)
# ============================================================

def pretrain_sequential(
    data_path: str = "expert_sequences.pkl",
    save_path: str = "imitation_seq",
    epochs: int = 30,
    lr: float = 1e-3,
    device: str = "cpu",
    features_dim: int = 256,
    lstm_hidden: int = 256,
    net_arch: list[int] | None = None,
):
    """Pre-train the full model (extractor + LSTM + head) on game sequences."""
    if net_arch is None:
        net_arch = [256, 256]

    print(f"Loading sequences from {data_path}...")
    with open(data_path, "rb") as f:
        sequences = pickle.load(f)

    total_steps = sum(len(s[0]) for s in sequences)
    print(f"  {len(sequences)} sequences, {total_steps} total steps")

    # shuffle and split
    rng = random.Random(42)
    rng.shuffle(sequences)
    val_size = len(sequences) // 10
    val_seqs = sequences[:val_size]
    train_seqs = sequences[val_size:]

    # build model: extractor + LSTM + policy head
    from training.attention_extractor import AttentionFeatureExtractor
    import gymnasium
    obs_space = gymnasium.spaces.Box(low=-10, high=10, shape=(OBS_SIZE,), dtype=np.float32)

    extractor = AttentionFeatureExtractor(obs_space, features_dim=features_dim)
    lstm = nn.LSTM(features_dim, lstm_hidden, num_layers=1, batch_first=True)

    head_layers = []
    in_dim = lstm_hidden
    for h in net_arch:
        head_layers.extend([nn.Linear(in_dim, h), nn.ReLU()])
        in_dim = h
    head_layers.append(nn.Linear(in_dim, 10))
    policy_head = nn.Sequential(*head_layers)

    extractor = extractor.to(device)
    lstm = lstm.to(device)
    policy_head = policy_head.to(device)

    all_params = list(extractor.parameters()) + list(lstm.parameters()) + list(policy_head.parameters())
    optimizer = torch.optim.Adam(all_params, lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    def run_epoch(seqs, train=True):
        if train:
            extractor.train(); lstm.train(); policy_head.train()
        else:
            extractor.eval(); lstm.eval(); policy_head.eval()

        total_loss = 0
        total_correct = 0
        total_count = 0

        # process sequences one at a time (variable length)
        indices = list(range(len(seqs)))
        if train:
            random.shuffle(indices)

        # mini-batch: accumulate gradients over N sequences
        batch_size = 16
        optimizer.zero_grad()

        for batch_start in range(0, len(indices), batch_size):
            batch_indices = indices[batch_start:batch_start + batch_size]
            batch_loss = 0

            for idx in batch_indices:
                obs_seq, act_seq, mask_seq = seqs[idx]
                seq_len = len(obs_seq)

                obs_t = torch.tensor(obs_seq, dtype=torch.float32, device=device)
                act_t = torch.tensor(act_seq, dtype=torch.long, device=device)
                mask_t = torch.tensor(mask_seq, dtype=torch.float32, device=device)

                # extract features for each step
                with torch.set_grad_enabled(train):
                    features = extractor(obs_t)  # (seq_len, features_dim)
                    features = features.unsqueeze(0)  # (1, seq_len, features_dim)

                    lstm_out, _ = lstm(features)  # (1, seq_len, lstm_hidden)
                    lstm_out = lstm_out.squeeze(0)  # (seq_len, lstm_hidden)

                    logits = policy_head(lstm_out)  # (seq_len, 10)
                    logits = logits + (1 - mask_t) * -1e9

                    loss = F.cross_entropy(logits, act_t)
                    batch_loss = batch_loss + loss / len(batch_indices)

                    preds = logits.argmax(dim=1)
                    total_correct += (preds == act_t).sum().item()
                    total_count += seq_len
                    total_loss += loss.item() * seq_len

            if train:
                batch_loss.backward()
                torch.nn.utils.clip_grad_norm_(all_params, 1.0)
                optimizer.step()
                optimizer.zero_grad()

        return total_loss / total_count, total_correct / total_count

    print(f"Training for {epochs} epochs ({len(train_seqs)} train, {len(val_seqs)} val)...")
    best_val_acc = 0
    for epoch in range(epochs):
        train_loss, train_acc = run_epoch(train_seqs, train=True)
        with torch.no_grad():
            val_loss, val_acc = run_epoch(val_seqs, train=False)
        scheduler.step()

        marker = ""
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save({
                "extractor": extractor.state_dict(),
                "lstm": lstm.state_dict(),
                "policy_head": policy_head.state_dict(),
                "features_dim": features_dim,
                "lstm_hidden": lstm_hidden,
                "net_arch": net_arch,
            }, f"{save_path}.pt")
            marker = " *best*"

        print(f"  Epoch {epoch+1:2d}: train_loss={train_loss:.4f} train_acc={train_acc:.3f} "
              f"val_loss={val_loss:.4f} val_acc={val_acc:.3f}{marker}")

    print(f"Best val accuracy: {best_val_acc:.3f}")
    print(f"Saved to {save_path}.pt")


def load_seq_pretrained_into_ppo(
    pretrained_path: str,
    save_path: str = "imitation_ppo",
    device: str = "cpu",
    n_envs: int = 8,
    n_steps: int = 4096,
    md_weight: float = 0.5,
    smart_weight: float = 0.5,
):
    """Load sequence-pretrained weights into MaskableRecurrentPPO."""
    import gymnasium
    import gym_env  # noqa: F401
    from stable_baselines3.common.vec_env import DummyVecEnv
    from training.train import make_env, make_mixed_opponent
    from training.maskable_recurrent_ppo import MaskableRecurrentPPO
    from training.attention_extractor import AttentionFeatureExtractor

    print(f"Loading pre-trained weights from {pretrained_path}...")
    checkpoint = torch.load(pretrained_path, map_location=device, weights_only=True)
    features_dim = checkpoint["features_dim"]
    net_arch = checkpoint["net_arch"]

    vec_env = DummyVecEnv([
        make_env(seed=i,
                 opponent_policy=make_mixed_opponent(md_weight, smart_weight),
                 reward_mode="shaped")
        for i in range(n_envs)
    ])

    policy_kwargs = {
        "features_extractor_class": AttentionFeatureExtractor,
        "features_extractor_kwargs": {"features_dim": features_dim},
        "net_arch": net_arch,
    }

    model = MaskableRecurrentPPO(
        "MlpLstmPolicy", vec_env,
        learning_rate=3e-4, n_steps=n_steps, batch_size=n_steps,
        n_epochs=10, gamma=0.99, gae_lambda=0.95, clip_range=0.2,
        ent_coef=0.02, verbose=1,
        tensorboard_log="./tb_logs", device=device,
        policy_kwargs=policy_kwargs,
    )

    # ---- copy extractor weights ----
    extractor_state = checkpoint["extractor"]
    model.policy.features_extractor.load_state_dict(extractor_state)
    model.policy.pi_features_extractor.load_state_dict(extractor_state)
    model.policy.vf_features_extractor.load_state_dict(extractor_state)

    # ---- copy LSTM weights ----
    lstm_state = checkpoint["lstm"]
    # SB3 recurrent policy has lstm_actor (shared for policy+value in our setup)
    sb3_lstm = model.policy.lstm_actor
    # map our single-layer LSTM to SB3's LSTM
    for key, param in lstm_state.items():
        if hasattr(sb3_lstm, key.split(".")[0]):
            target = dict(sb3_lstm.named_parameters())
            if key in target:
                target[key].data.copy_(param)
            else:
                # try named buffers
                bufs = dict(sb3_lstm.named_buffers())
                if key in bufs:
                    bufs[key].data.copy_(param)

    # ---- copy policy head weights ----
    head_state = checkpoint["policy_head"]
    policy_net = model.policy.mlp_extractor.policy_net
    policy_params = dict(policy_net.named_parameters())

    for key, param in head_state.items():
        layer_idx = int(key.split(".")[0])
        # layers 0,1 = first hidden (Linear+ReLU), 2,3 = second hidden, 4 = output
        # policy_net has the hidden layers, action_net has the output
        n_hidden_layers = len(net_arch) * 2  # each has Linear + ReLU
        if layer_idx < n_hidden_layers:
            if key in policy_params:
                policy_params[key].data.copy_(param)
        else:
            # output layer -> action_net
            param_type = key.split(".")[1]
            if param_type == "weight":
                model.policy.action_net.weight.data.copy_(param)
            else:
                model.policy.action_net.bias.data.copy_(param)

    model.save(save_path)
    print(f"Saved PPO model with pre-trained weights to {save_path}")
    vec_env.close()


# ============================================================
# CLI
# ============================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Imitation learning pipeline")
    parser.add_argument("--generate", action="store_true")
    parser.add_argument("--pretrain", action="store_true")
    parser.add_argument("--init-ppo", action="store_true")
    parser.add_argument("--n-games", type=int, default=5000)
    parser.add_argument("--data", type=str, default="expert_sequences.pkl")
    parser.add_argument("--model", type=str, default="imitation_seq")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--opponent", type=str, default="mixed",
                        choices=["mixed", "smart", "maxdmg"])
    args = parser.parse_args()

    if args.generate:
        generate_sequences(
            n_games=args.n_games, out_path=args.data, opponent=args.opponent,
        )
    elif args.pretrain:
        pretrain_sequential(
            data_path=args.data, save_path=args.model,
            epochs=args.epochs, device=args.device,
        )
    elif args.init_ppo:
        load_seq_pretrained_into_ppo(
            pretrained_path=f"{args.model}.pt", device=args.device,
        )
