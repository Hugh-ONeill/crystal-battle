#!/usr/bin/env python3
# Population-Based Training for Crystal Battle
# Usage: python training/pbt.py --resume-from crystal_battle_ppo_step2500000 --device cuda

from __future__ import annotations

import argparse
import copy
import csv
import json
import random
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import gym_env  # noqa: F401
import gymnasium
import numpy as np
from sb3_contrib.common.wrappers import ActionMasker
from stable_baselines3.common.vec_env import DummyVecEnv

from training.attention_extractor import AttentionFeatureExtractor
from training.baselines import MaxDamageAgent, SmartAgent
from training.maskable_recurrent_ppo import MaskableRecurrentPPO
from training.evaluate import evaluate_vs_baseline
from training.replay import record_games
from training.train import make_env, make_mixed_opponent

PROJECT = Path(__file__).parent.parent
PBT_DIR = PROJECT / "pbt_runs"
PBT_LOG = PROJECT / "pbt_history.csv"

EVAL_BASELINES = ("max_damage", "smart", "crystal_ai")


# ============================================================
# HYPERPARAMETER SPACE
# ============================================================
@dataclass
class HyperParams:
    lr: float = 3e-4
    ent_coef: float = 0.02
    gamma: float = 0.99
    clip_range: float = 0.2
    gae_lambda: float = 0.95
    n_epochs: int = 10

    def to_dict(self) -> dict:
        return {
            "lr": self.lr,
            "ent_coef": self.ent_coef,
            "gamma": self.gamma,
            "clip_range": self.clip_range,
            "gae_lambda": self.gae_lambda,
            "n_epochs": self.n_epochs,
        }

    def perturb(self, rng: random.Random, factor: float = 0.2) -> HyperParams:
        """Create a mutated copy with random perturbations."""
        def jitter(val: float, lo: float, hi: float) -> float:
            mult = rng.choice([1.0 - factor, 1.0, 1.0 + factor])
            return max(lo, min(hi, val * mult))

        return HyperParams(
            lr=jitter(self.lr, 1e-5, 1e-3),
            ent_coef=jitter(self.ent_coef, 0.005, 0.15),
            gamma=jitter(self.gamma, 0.95, 0.999),
            clip_range=jitter(self.clip_range, 0.05, 0.4),
            gae_lambda=jitter(self.gae_lambda, 0.8, 0.99),
            n_epochs=rng.choice([max(3, self.n_epochs - 2), self.n_epochs,
                                 min(20, self.n_epochs + 2)]),
        )


# ============================================================
# POPULATION MEMBER
# ============================================================
@dataclass
class Member:
    id: int
    hp: HyperParams
    seed: int
    model: MaskableRecurrentPPO | None = None
    total_steps: int = 0
    best_score: float = 0.0
    last_scores: dict = field(default_factory=dict)
    history: list[dict] = field(default_factory=list)


def build_vec_env(seed: int, n_envs: int, md_weight: float = 0.5,
                  smart_weight: float = 0.5) -> DummyVecEnv:
    """Create vectorized env with mixed opponents."""
    return DummyVecEnv([
        make_env(
            seed=seed * 100 + i,
            opponent_policy=make_mixed_opponent(
                max_damage_weight=md_weight,
                smart_weight=smart_weight,
                seed=seed * 100 + i,
            ),
            tier="ou",
        )
        for i in range(n_envs)
    ])


def create_model(member: Member, n_envs: int = 4, device: str = "cpu",
                 md_weight: float = 0.5, smart_weight: float = 0.5,
                 n_steps: int = 2048) -> MaskableRecurrentPPO:
    """Create a fresh model for a population member."""
    vec_env = build_vec_env(member.seed, n_envs, md_weight, smart_weight)

    policy_kwargs = {
        "features_extractor_class": AttentionFeatureExtractor,
        "features_extractor_kwargs": {"features_dim": 192},
    }

    return MaskableRecurrentPPO(
        "MlpLstmPolicy", vec_env,
        learning_rate=member.hp.lr,
        n_steps=n_steps,
        batch_size=n_steps,
        n_epochs=member.hp.n_epochs,
        gamma=member.hp.gamma,
        gae_lambda=member.hp.gae_lambda,
        clip_range=member.hp.clip_range,
        ent_coef=member.hp.ent_coef,
        verbose=0,
        device=device,
        policy_kwargs=policy_kwargs,
        seed=member.seed,
    )


def load_warm_start(checkpoint: str, member: Member, n_envs: int = 4,
                    device: str = "cpu", md_weight: float = 0.5,
                    smart_weight: float = 0.5,
                    n_steps: int = 2048) -> MaskableRecurrentPPO:
    """Load a checkpoint and apply member's hyperparams."""
    vec_env = build_vec_env(member.seed, n_envs, md_weight, smart_weight)
    model = MaskableRecurrentPPO.load(checkpoint, env=vec_env, device=device)
    model.num_timesteps = 0
    model._num_timesteps_at_start = 0
    model._last_obs = None
    model._last_episode_starts = None
    apply_hyperparams(model, member.hp, n_steps)
    # rebuild only the rollout buffer (not the policy) for new env/n_steps
    from gymnasium import spaces
    from training.maskable_recurrent_ppo import MaskableRecurrentRolloutBuffer
    action_dim = model.action_space.n if isinstance(model.action_space, spaces.Discrete) else 0
    lstm = model.policy.lstm_actor
    hidden_state_buffer_shape = (model.n_steps, lstm.num_layers, model.n_envs, lstm.hidden_size)
    model.rollout_buffer = MaskableRecurrentRolloutBuffer(
        buffer_size=model.n_steps,
        observation_space=model.observation_space,
        action_space=model.action_space,
        hidden_state_shape=hidden_state_buffer_shape,
        device=model.device,
        gamma=model.gamma,
        gae_lambda=model.gae_lambda,
        n_envs=model.n_envs,
        action_dim=action_dim,
    )
    return model


def apply_hyperparams(model: MaskableRecurrentPPO, hp: HyperParams,
                      n_steps: int | None = None):
    """Apply mutable hyperparams to an existing model."""
    model.learning_rate = hp.lr
    model.ent_coef = hp.ent_coef
    model.gamma = hp.gamma
    model.clip_range = lambda _: hp.clip_range
    model.gae_lambda = hp.gae_lambda
    model.n_epochs = hp.n_epochs
    if n_steps is not None:
        model.n_steps = n_steps
        model.batch_size = n_steps


def evaluate_member(member: Member, n_games: int = 400) -> dict[str, float]:
    """Evaluate vs all baselines. Fixed seed so all members face the same teams."""
    scores = {}
    for bl in EVAL_BASELINES:
        result = evaluate_vs_baseline(member.model, baseline=bl, n_games=n_games, seed=42)
        scores[bl] = result["win_rate"]
    return scores


def composite_score(scores: dict[str, float]) -> float:
    """Weighted composite for ranking: 50% maxdmg + 30% smart + 20% crystal."""
    return (0.5 * scores.get("max_damage", 0.0)
            + 0.3 * scores.get("smart", 0.0)
            + 0.2 * scores.get("crystal_ai", 0.0))


def copy_weights(source: Member, target: Member):
    """Copy network weights from source to target."""
    source_state = source.model.policy.state_dict()
    target.model.policy.load_state_dict(copy.deepcopy(source_state))
    if hasattr(source.model, "_last_lstm_states") and source.model._last_lstm_states is not None:
        target.model._last_lstm_states = copy.deepcopy(source.model._last_lstm_states)


# ============================================================
# PBT MAIN LOOP
# ============================================================
def main():
    parser = argparse.ArgumentParser(description="Population-Based Training")
    parser.add_argument("--population", type=int, default=6,
                        help="Population size")
    parser.add_argument("--generations", type=int, default=20,
                        help="Number of generations")
    parser.add_argument("--steps-per-gen", type=int, default=500_000,
                        help="Training steps per generation per member")
    parser.add_argument("--n-envs", type=int, default=4,
                        help="Envs per member")
    parser.add_argument("--n-steps", type=int, default=2048,
                        help="PPO rollout steps")
    parser.add_argument("--eval-games", type=int, default=400,
                        help="Games per baseline per evaluation")
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--exploit-frac", type=float, default=0.25,
                        help="Bottom fraction to replace each generation")
    parser.add_argument("--no-exploit", action="store_true",
                        help="Disable exploit/explore -- pure parallel HP search")
    parser.add_argument("--resume-from", type=str, default=None,
                        help="Checkpoint path to warm-start all members from")
    parser.add_argument("--md-weight", type=float, default=0.5,
                        help="MaxDamage opponent weight")
    parser.add_argument("--smart-weight", type=float, default=0.5,
                        help="SmartAgent opponent weight")
    parser.add_argument("--replay-games", type=int, default=3,
                        help="Replay games to record for best member each gen")
    args = parser.parse_args()

    PBT_DIR.mkdir(exist_ok=True)
    rng = random.Random(42)

    # ---- Init CSV log ----
    csv_header = ["gen", "member", "steps", "composite",
                  "vs_maxdmg", "vs_smart", "vs_crystal",
                  "lr", "ent_coef", "gamma", "clip_range", "gae_lambda",
                  "n_epochs", "action"]
    with open(PBT_LOG, "w", newline="") as f:
        csv.writer(f).writerow(csv_header)

    # ---- Create population ----
    print(f"Initializing population of {args.population}...")
    if args.resume_from:
        print(f"  Warm-starting from: {args.resume_from}")

    population: list[Member] = []
    for i in range(args.population):
        hp = HyperParams(
            lr=rng.uniform(1e-4, 5e-4),
            ent_coef=rng.uniform(0.01, 0.08),
            gamma=rng.choice([0.98, 0.99, 0.995]),
            clip_range=rng.choice([0.1, 0.15, 0.2, 0.3]),
            gae_lambda=rng.choice([0.9, 0.95, 0.98]),
            n_epochs=rng.choice([6, 8, 10, 12]),
        )
        member = Member(id=i, hp=hp, seed=rng.randint(0, 99999))

        if args.resume_from:
            member.model = load_warm_start(
                args.resume_from, member,
                n_envs=args.n_envs, device=args.device,
                md_weight=args.md_weight, smart_weight=args.smart_weight,
                n_steps=args.n_steps,
            )
        else:
            member.model = create_model(
                member, n_envs=args.n_envs, device=args.device,
                md_weight=args.md_weight, smart_weight=args.smart_weight,
                n_steps=args.n_steps,
            )

        population.append(member)
        print(f"  Member {i}: lr={hp.lr:.2e} ent={hp.ent_coef:.3f} "
              f"gamma={hp.gamma} clip={hp.clip_range} "
              f"gae={hp.gae_lambda} epochs={hp.n_epochs}")

    n_replace = max(1, int(args.population * args.exploit_frac))

    total_member_steps = args.generations * args.steps_per_gen
    print(f"\nPBT: {args.population} members x {args.generations} generations "
          f"x {args.steps_per_gen:,} steps/gen")
    print(f"  Total steps per member: {total_member_steps:,}")
    print(f"  Opponents: {args.md_weight:.0%} MaxDmg + {args.smart_weight:.0%} Smart")
    print(f"  Replace bottom {n_replace} each generation")
    print()

    # ---- Generation loop ----
    for gen in range(args.generations):
        gen_start = time.time()

        # ---- Train each member ----
        for m in population:
            t0 = time.time()

            m.model.learn(
                total_timesteps=args.steps_per_gen,
                reset_num_timesteps=True,
                progress_bar=False,
            )
            m.total_steps += args.steps_per_gen
            dt = time.time() - t0
            fps = args.steps_per_gen / dt if dt > 0 else 0

            # ---- Evaluate ----
            scores = evaluate_member(m, n_games=args.eval_games)
            m.last_scores = scores
            comp = composite_score(scores)
            m.best_score = max(m.best_score, comp)

            print(f"  Gen {gen} M{m.id}: {args.steps_per_gen:,} steps ({fps:.0f} fps) "
                  f"| MaxD {scores['max_damage']:.1%} Smart {scores['smart']:.1%} "
                  f"Crystal {scores['crystal_ai']:.1%} [composite {comp:.3f}]")

            m.history.append({"gen": gen, "scores": scores, "hp": m.hp.to_dict()})

        # ---- Rank by composite score ----
        ranked = sorted(population,
                        key=lambda m: composite_score(m.last_scores),
                        reverse=True)
        best = ranked[0]
        worst = ranked[-n_replace:]
        top = ranked[:max(n_replace, 2)]

        best_comp = composite_score(best.last_scores)
        print(f"\n  Gen {gen} ranking:")
        for rank, m in enumerate(ranked):
            c = composite_score(m.last_scores)
            marker = " *" if m in worst else ""
            print(f"    #{rank+1} M{m.id}: {c:.3f}{marker}")

        # ---- Record replays for best member ----
        if args.replay_games > 0:
            replay_dir = PBT_DIR / f"gen{gen}_M{best.id}"
            for bl in ("max_damage", "smart"):
                bl_dir = replay_dir / bl
                try:
                    record_games(best.model, n_games=args.replay_games,
                                 baseline=bl, out_dir=bl_dir)
                except Exception as e:
                    print(f"  [replay vs {bl}] failed: {e}")

        # ---- Exploit + Explore ----
        if not args.no_exploit:
            for bad in worst:
                donor = rng.choice(top)

                # copy weights from donor
                copy_weights(donor, bad)

                # perturb donor's hyperparams
                bad.hp = donor.hp.perturb(rng)
                apply_hyperparams(bad.model, bad.hp)

                print(f"    M{bad.id} <- weights from M{donor.id}, "
                      f"mutated: lr={bad.hp.lr:.2e} ent={bad.hp.ent_coef:.3f} "
                      f"gamma={bad.hp.gamma} clip={bad.hp.clip_range} "
                      f"epochs={bad.hp.n_epochs}")

        # ---- Log all members ----
        with open(PBT_LOG, "a", newline="") as f:
            w = csv.writer(f)
            for m in ranked:
                action = "kept" if args.no_exploit else ("replaced" if m in worst else "survived")
                comp = composite_score(m.last_scores)
                w.writerow([
                    gen, m.id, m.total_steps, f"{comp:.4f}",
                    f"{m.last_scores.get('max_damage', 0):.3f}",
                    f"{m.last_scores.get('smart', 0):.3f}",
                    f"{m.last_scores.get('crystal_ai', 0):.3f}",
                    f"{m.hp.lr:.2e}", f"{m.hp.ent_coef:.4f}",
                    m.hp.gamma, m.hp.clip_range, m.hp.gae_lambda,
                    m.hp.n_epochs, action,
                ])

        # ---- Save best model ----
        best_path = PBT_DIR / f"best_gen{gen}"
        best.model.save(str(best_path))

        gen_time = time.time() - gen_start
        print(f"    Gen {gen} done in {gen_time/60:.1f}m. "
              f"Best: M{best.id} = {best_comp:.3f} "
              f"(MaxD {best.last_scores['max_damage']:.1%} "
              f"Smart {best.last_scores['smart']:.1%} "
              f"Crystal {best.last_scores['crystal_ai']:.1%})")
        print()

    # ---- Final summary ----
    print("=" * 60)
    print("PBT COMPLETE")
    print("=" * 60)
    overall_best = max(population, key=lambda m: m.best_score)
    best_scores = overall_best.last_scores
    print(f"Best member: M{overall_best.id}")
    print(f"Best composite: {overall_best.best_score:.3f}")
    print(f"  MaxDmg:  {best_scores.get('max_damage', 0):.1%}")
    print(f"  Smart:   {best_scores.get('smart', 0):.1%}")
    print(f"  Crystal: {best_scores.get('crystal_ai', 0):.1%}")
    print(f"Final HP: {overall_best.hp.to_dict()}")

    final_path = PBT_DIR / "best_final"
    overall_best.model.save(str(final_path))
    print(f"Saved to: {final_path}")

    with open(PBT_DIR / "best_params.json", "w") as f:
        json.dump({
            "member_id": overall_best.id,
            "best_score": overall_best.best_score,
            "hyperparams": overall_best.hp.to_dict(),
            "history": overall_best.history,
        }, f, indent=2)


if __name__ == "__main__":
    main()
