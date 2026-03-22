# PPO training loop for the transformer policy
# custom rollout collection that maintains obs history buffers per env
#
# Usage:
#   python training/train_transformer.py --total-steps 10000000

from __future__ import annotations

import argparse
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).parent.parent))

import gymnasium
import gym_env  # noqa: F401

from engine.types import TypeChart
from training.transformer_policy import TransformerBattlePolicy, TransformerAgent
from training.baselines import SmartAgent, MaxDamageAgent
from training.evaluate import evaluate_vs_baseline
from training.train import make_env, make_mixed_opponent
from gym_env.obs_builder import OBS_SIZE


# ============================================================
# PPO ROLLOUT BUFFER
# ============================================================

class TransformerRolloutBuffer:
    """Stores rollout data for PPO with observation sequences."""

    def __init__(self):
        self.obs_sequences = []  # list of (seq_len, obs_dim) arrays
        self.actions = []
        self.rewards = []
        self.dones = []
        self.values = []
        self.log_probs = []
        self.masks = []

    def add(self, obs_seq, action, reward, done, value, log_prob, mask):
        self.obs_sequences.append(obs_seq)
        self.actions.append(action)
        self.rewards.append(reward)
        self.dones.append(done)
        self.values.append(value)
        self.log_probs.append(log_prob)
        self.masks.append(mask)

    def compute_returns(self, last_value: float, gamma: float = 0.99,
                        gae_lambda: float = 0.95):
        """Compute GAE advantages and returns."""
        n = len(self.rewards)
        self.advantages = np.zeros(n, dtype=np.float32)
        self.returns = np.zeros(n, dtype=np.float32)

        last_gae = 0
        for t in reversed(range(n)):
            if t == n - 1:
                next_value = last_value
                next_done = 0
            else:
                next_value = self.values[t + 1]
                next_done = self.dones[t + 1]

            delta = self.rewards[t] + gamma * next_value * (1 - next_done) - self.values[t]
            last_gae = delta + gamma * gae_lambda * (1 - next_done) * last_gae
            self.advantages[t] = last_gae

        self.returns = self.advantages + np.array(self.values, dtype=np.float32)

    def get_batches(self, batch_size: int):
        """Yield mini-batches for PPO update."""
        n = len(self.rewards)
        indices = np.arange(n)
        np.random.shuffle(indices)

        for start in range(0, n, batch_size):
            batch_idx = indices[start:start + batch_size]
            yield {
                "obs_sequences": [self.obs_sequences[i] for i in batch_idx],
                "actions": torch.tensor([self.actions[i] for i in batch_idx], dtype=torch.long),
                "old_log_probs": torch.tensor([self.log_probs[i] for i in batch_idx], dtype=torch.float32),
                "advantages": torch.tensor(self.advantages[batch_idx], dtype=torch.float32),
                "returns": torch.tensor(self.returns[batch_idx], dtype=torch.float32),
                "masks": torch.tensor(np.array([self.masks[i] for i in batch_idx]), dtype=torch.float32),
            }

    def clear(self):
        self.__init__()


# ============================================================
# PPO UPDATE
# ============================================================

def ppo_update(model: TransformerBattlePolicy, buffer: TransformerRolloutBuffer,
               optimizer, device: str, n_epochs: int = 10, batch_size: int = 64,
               clip_range: float = 0.2, ent_coef: float = 0.02,
               max_grad_norm: float = 0.5):
    """Run PPO updates on the collected rollout data."""
    # normalize advantages
    adv = buffer.advantages
    adv = (adv - adv.mean()) / (adv.std() + 1e-8)
    buffer.advantages = adv

    total_loss_sum = 0
    n_updates = 0

    for epoch in range(n_epochs):
        for batch in buffer.get_batches(batch_size):
            # pad sequences to same length for batching
            max_len = max(len(s) for s in batch["obs_sequences"])
            padded = np.zeros((len(batch["obs_sequences"]), max_len, OBS_SIZE), dtype=np.float32)
            for i, seq in enumerate(batch["obs_sequences"]):
                padded[i, :len(seq)] = seq

            obs_t = torch.tensor(padded, dtype=torch.float32, device=device)
            actions = batch["actions"].to(device)
            old_log_probs = batch["old_log_probs"].to(device)
            advantages = batch["advantages"].to(device)
            returns = batch["returns"].to(device)
            masks = batch["masks"].to(device)

            # forward pass
            logits, values = model(obs_t, action_masks=masks)
            values = values.squeeze(-1)

            # policy loss (PPO clipping)
            log_probs = F.log_softmax(logits, dim=-1)
            action_log_probs = log_probs.gather(1, actions.unsqueeze(1)).squeeze(1)
            ratio = torch.exp(action_log_probs - old_log_probs)
            surr1 = ratio * advantages
            surr2 = torch.clamp(ratio, 1 - clip_range, 1 + clip_range) * advantages
            policy_loss = -torch.min(surr1, surr2).mean()

            # value loss
            value_loss = F.mse_loss(values, returns)

            # entropy bonus
            probs = F.softmax(logits, dim=-1)
            entropy = -(probs * log_probs).sum(dim=-1).mean()

            # total loss
            loss = policy_loss + 0.5 * value_loss - ent_coef * entropy

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            optimizer.step()

            total_loss_sum += loss.item()
            n_updates += 1

    return total_loss_sum / max(n_updates, 1)


# ============================================================
# ROLLOUT COLLECTION
# ============================================================

def collect_rollouts(envs, agent: TransformerAgent, model: TransformerBattlePolicy,
                     buffer: TransformerRolloutBuffer, n_steps: int,
                     device: str, obs_buffers: list[list]):
    """Collect n_steps of experience from parallel envs."""
    n_envs = len(envs)
    step = 0

    while step < n_steps:
        for env_idx, env in enumerate(envs):
            if env.unwrapped._battle is None or env.unwrapped._battle.is_over:
                # reset env
                obs, info = env.reset()
                obs_buffers[env_idx] = [obs]
                continue

            obs = env.unwrapped._last_obs if hasattr(env.unwrapped, '_last_obs') else None
            if obs is None:
                obs, info = env.reset()
                obs_buffers[env_idx] = [obs]
                continue

            # build observation sequence
            obs_seq = np.array(obs_buffers[env_idx], dtype=np.float32)
            obs_t = torch.tensor(obs_seq, dtype=torch.float32, device=device).unsqueeze(0)

            mask = np.array(info.get("action_mask",
                   env.unwrapped._battle.p1.valid_action_mask(
                       env.unwrapped._battle.p2)), dtype=np.float32)
            mask_t = torch.tensor(mask, dtype=torch.float32, device=device).unsqueeze(0)

            # get action, value, log_prob
            with torch.no_grad():
                logits, value = model(obs_t, action_masks=mask_t)
                probs = F.softmax(logits, dim=-1)
                dist = torch.distributions.Categorical(probs)
                action = dist.sample()
                log_prob = dist.log_prob(action)

            action_int = action.item()
            value_float = value.item()
            log_prob_float = log_prob.item()

            # step env
            next_obs, reward, terminated, truncated, info = env.step(action_int)
            done = terminated or truncated

            # store in buffer
            buffer.add(
                obs_seq=obs_seq.copy(),
                action=action_int,
                reward=reward,
                done=float(done),
                value=value_float,
                log_prob=log_prob_float,
                mask=mask.copy(),
            )

            # update obs buffer
            obs_buffers[env_idx].append(next_obs)
            if len(obs_buffers[env_idx]) > agent.max_seq_len:
                obs_buffers[env_idx] = obs_buffers[env_idx][-agent.max_seq_len:]

            if done:
                obs, info = env.reset()
                obs_buffers[env_idx] = [obs]

            step += 1
            if step >= n_steps:
                break

    # compute last value for GAE
    last_values = []
    for env_idx, env in enumerate(envs):
        if obs_buffers[env_idx]:
            obs_seq = np.array(obs_buffers[env_idx], dtype=np.float32)
            obs_t = torch.tensor(obs_seq, dtype=torch.float32, device=device).unsqueeze(0)
            with torch.no_grad():
                _, value = model(obs_t)
            last_values.append(value.item())
        else:
            last_values.append(0.0)

    buffer.compute_returns(np.mean(last_values))


# ============================================================
# EVALUATION
# ============================================================

def evaluate_transformer(model: TransformerBattlePolicy, n_games: int = 100,
                          device: str = "cpu"):
    """Evaluate the transformer agent against baselines."""
    from engine.types import TypeChart
    from engine.data_loader import DataStore
    from engine.battle_state import BattleState
    from engine.player_state import PlayerState
    from engine.turn_engine import resolve_turn, resolve_forced_switches
    from engine.actions import Switch, UseMove, Struggle
    from gym_env.team_builder import build_team
    from gym_env.obs_builder import build_observation

    tc = TypeChart.load()
    data = DataStore()

    agent = TransformerAgent(model, device=device)

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

            agent.reset()

            for turn in range(100):
                if battle.is_over:
                    break

                obs = build_observation(
                    battle.p1, battle.p2, tc, turn=battle.turn,
                    weather=battle.weather, weather_turns=battle.weather_turns)
                mask = np.array(battle.p1.valid_action_mask(battle.p2, type_chart=tc),
                               dtype=np.float32)

                action_int = agent.act(obs, mask, deterministic=True)

                # convert to engine action
                if action_int < 4:
                    active = battle.p1.active
                    if not active.has_any_pp():
                        p1_action = Struggle()
                    elif action_int < len(active.move_slots) and active.move_slots[action_int].has_pp:
                        p1_action = UseMove(slot_index=action_int)
                    else:
                        for j, slot in enumerate(active.move_slots):
                            if slot.has_pp:
                                p1_action = UseMove(slot_index=j); break
                        else:
                            p1_action = Struggle()
                else:
                    p1_action = Switch(team_index=action_int - 4)

                p2_action = bl.act(battle.p2, battle.p1)
                resolve_turn(battle, p1_action, p2_action, tc)

                sw1 = sw2 = None
                if battle.p1.must_switch:
                    sw_obs = build_observation(
                        battle.p1, battle.p2, tc, turn=battle.turn,
                        weather=battle.weather, weather_turns=battle.weather_turns)
                    sw_mask = np.array(battle.p1.valid_action_mask(battle.p2, type_chart=tc),
                                       dtype=np.float32)
                    sw_int = agent.act(sw_obs, sw_mask, deterministic=True)
                    if sw_int >= 4:
                        sw1 = Switch(team_index=sw_int - 4)
                    if sw1 is None:
                        for j, p in enumerate(battle.p1.team):
                            if j != battle.p1.active_index and not p.is_fainted:
                                sw1 = Switch(team_index=j); break

                if battle.p2.must_switch:
                    sw2_a = bl.act(battle.p2, battle.p1)
                    sw2 = sw2_a if isinstance(sw2_a, Switch) else None
                    if sw2 is None:
                        for j, p in enumerate(battle.p2.team):
                            if j != battle.p2.active_index and not p.is_fainted:
                                sw2 = Switch(team_index=j); break

                if sw1 or sw2:
                    resolve_forced_switches(battle, sw1, sw2)

            if battle.winner == 1:
                wins += 1

        print(f"  vs {bl_name:12s}: {wins}/{n_games} ({wins/n_games*100:.1f}%)")


# ============================================================
# MAIN
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="Transformer policy training")
    parser.add_argument("--total-steps", type=int, default=10000000)
    parser.add_argument("--n-envs", type=int, default=8)
    parser.add_argument("--n-steps", type=int, default=2048)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--n-epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--ent-coef", type=float, default=0.02)
    parser.add_argument("--eval-freq", type=int, default=50000)
    parser.add_argument("--eval-games", type=int, default=100)
    parser.add_argument("--save-path", type=str, default="transformer_policy")
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--seq-len", type=int, default=32)
    parser.add_argument("--n-layers", type=int, default=3)
    parser.add_argument("--n-heads", type=int, default=4)
    parser.add_argument("--md-weight", type=float, default=0.5)
    parser.add_argument("--smart-weight", type=float, default=0.5)
    args = parser.parse_args()

    print("=" * 60)
    print("  Transformer Policy Training")
    print("=" * 60)
    print(f"  Total steps: {args.total_steps:,}")
    print(f"  Seq length:  {args.seq_len}")
    print(f"  Layers:      {args.n_layers}")
    print(f"  Heads:       {args.n_heads}")
    print(f"  LR:          {args.lr}")
    print()

    obs_space = gymnasium.spaces.Box(low=-10, high=10, shape=(OBS_SIZE,), dtype=np.float32)

    model = TransformerBattlePolicy(
        obs_space=obs_space,
        features_dim=256,
        d_model=256,
        n_heads=args.n_heads,
        n_layers=args.n_layers,
        ff_dim=512,
        max_seq_len=args.seq_len,
        net_arch=[256, 256],
    ).to(args.device)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"  Parameters:  {total_params:,}")

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    # create envs
    envs = []
    for i in range(args.n_envs):
        env = make_env(
            seed=i,
            opponent_policy=make_mixed_opponent(args.md_weight, args.smart_weight),
            reward_mode="shaped",
        )()
        envs.append(env)

    agent = TransformerAgent(model, max_seq_len=args.seq_len, device=args.device)
    obs_buffers = [[] for _ in range(args.n_envs)]

    # initialize envs
    for i, env in enumerate(envs):
        obs, info = env.reset()
        obs_buffers[i] = [obs]

    total_steps = 0
    last_eval = 0
    start_time = time.time()

    print("Training...")
    while total_steps < args.total_steps:
        buffer = TransformerRolloutBuffer()
        collect_rollouts(envs, agent, model, buffer, args.n_steps * args.n_envs,
                         args.device, obs_buffers)

        total_steps += len(buffer.rewards)
        avg_loss = ppo_update(model, buffer, optimizer, args.device,
                              n_epochs=args.n_epochs, batch_size=args.batch_size,
                              ent_coef=args.ent_coef)
        buffer.clear()

        # eval
        if total_steps - last_eval >= args.eval_freq:
            elapsed = time.time() - start_time
            fps = total_steps / elapsed
            print(f"\n  Step {total_steps:>10,} | loss={avg_loss:.4f} | "
                  f"{fps:.0f} steps/s")
            model.eval()
            evaluate_transformer(model, n_games=args.eval_games, device=args.device)
            model.train()
            last_eval = total_steps

            # save
            torch.save(model.state_dict(), f"{args.save_path}.pt")

    # final save
    torch.save(model.state_dict(), f"{args.save_path}.pt")
    print("\nTraining complete!")

    for env in envs:
        env.close()


if __name__ == "__main__":
    main()
