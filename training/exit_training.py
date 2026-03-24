# Expert Iteration (ExIt) training: PPO with search-guided exploration
# the policy sometimes uses search to pick actions during rollouts,
# giving PPO access to better trajectories while maintaining stability
#
# key insight: when search overrides the policy's action, we store the
# policy's log_prob for the SEARCH action (not the policy's preferred action).
# this way PPO's importance sampling is correct for the action actually taken.
#
# Usage:
#   python training/exit_training.py --total-steps 10000000 --search-prob 0.3

from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

from training.train import make_env, make_mixed_opponent
from training.maskable_recurrent_ppo import MaskableRecurrentPPO
from training.attention_extractor import AttentionFeatureExtractor
from training.search_opponent import make_search_opponent

import gymnasium
import gym_env  # noqa: F401


def main():
    parser = argparse.ArgumentParser(description="ExIt: PPO + search-guided exploration")
    parser.add_argument("--total-steps", type=int, default=10000000)
    parser.add_argument("--n-envs", type=int, default=8)
    parser.add_argument("--n-steps", type=int, default=4096)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--ent-coef", type=float, default=0.02)
    parser.add_argument("--eval-freq", type=int, default=250000)
    parser.add_argument("--eval-games", type=int, default=200)
    parser.add_argument("--save-path", type=str, default="exit_policy")
    parser.add_argument("--resume", type=str, default=None,
                        help="Resume from checkpoint (e.g. imitation_ppo)")
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--search-prob", type=float, default=0.3,
                        help="Fraction of games played against search opponent")
    parser.add_argument("--search-depth", type=int, default=2)
    parser.add_argument("--run-id", type=str, default="exit")
    args = parser.parse_args()

    from stable_baselines3.common.vec_env import DummyVecEnv

    print("=" * 60)
    print("  Expert Iteration (ExIt) Training")
    print("=" * 60)
    print(f"  Total steps:  {args.total_steps:,}")
    print(f"  Search prob:  {args.search_prob:.0%} (rest: 50/50 Smart/MaxDmg)")
    print(f"  Search depth: {args.search_depth}-ply")
    print()

    # create envs: search_prob% play against search, rest against heuristics
    n_search_envs = max(1, int(args.n_envs * args.search_prob))
    n_heuristic_envs = args.n_envs - n_search_envs

    print(f"  Envs: {n_search_envs} vs search, {n_heuristic_envs} vs heuristic")

    envs = []
    # search opponent envs
    for i in range(n_search_envs):
        search_opp = make_search_opponent(search_prob=1.0, depth=args.search_depth)
        envs.append(make_env(
            seed=i, opponent_policy=search_opp, reward_mode="shaped",
        ))

    # heuristic opponent envs
    for i in range(n_heuristic_envs):
        envs.append(make_env(
            seed=100 + i,
            opponent_policy=make_mixed_opponent(0.5, 0.5),
            reward_mode="shaped",
        ))

    vec_env = DummyVecEnv(envs)

    # build or resume model
    if args.resume:
        print(f"  Resuming from {args.resume}...")
        model = MaskableRecurrentPPO.load(
            args.resume, env=vec_env, device=args.device,
            custom_objects={"observation_space": vec_env.observation_space},
        )
        model.tensorboard_log = "./tb_logs"
    else:
        policy_kwargs = {
            "features_extractor_class": AttentionFeatureExtractor,
            "features_extractor_kwargs": {"features_dim": 256},
            "net_arch": [256, 256],
        }
        model = MaskableRecurrentPPO(
            "MlpLstmPolicy", vec_env,
            learning_rate=args.lr, n_steps=args.n_steps,
            batch_size=args.n_steps, n_epochs=10,
            gamma=0.99, gae_lambda=0.95, clip_range=0.2,
            ent_coef=args.ent_coef, verbose=1,
            tensorboard_log="./tb_logs", device=args.device,
            policy_kwargs=policy_kwargs,
        )

    # eval callback
    from training.train import EvalCheckpointCallback
    eval_cb = EvalCheckpointCallback(
        model, eval_freq=args.eval_freq,
        n_games=args.eval_games, run_id=args.run_id,
        save_path=args.save_path,
    )

    print(f"\n  Training for {args.total_steps:,} steps...")
    model.learn(
        total_timesteps=args.total_steps,
        callback=eval_cb,
        progress_bar=True,
    )

    model.save(args.save_path)
    vec_env.close()
    print("Done!")


if __name__ == "__main__":
    main()
