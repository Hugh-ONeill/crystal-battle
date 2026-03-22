#!/usr/bin/env python3
# MaskablePPO / MaskableRecurrentPPO training with mixed opponent scheduling
# Usage: python training/train.py [--total-steps N] [--device cpu] [--lstm]

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import gym_env  # noqa: F401 -- registers the env
import gymnasium
import numpy as np
import random as _random

from sb3_contrib import MaskablePPO
from sb3_contrib.common.wrappers import ActionMasker
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import DummyVecEnv

from engine.types import TypeChart
from training.action_tracker import ActionTrackerCallback
from training.baselines import MaxDamageAgent, SmartAgent
from training.crystal_ai import CrystalAIAgent, AI_CHAMPION, AI_EXECUTIVE, AI_TRAINER
from training.maskable_recurrent_ppo import MaskableRecurrentPPO
from training.self_play import (
    OpponentPool, SelfPlayCallback, make_neural_opponent, make_mixed_neural_opponent,
)


EVAL_LOG = Path(__file__).parent.parent / "eval_log.txt"
MASTER_CSV = Path(__file__).parent.parent / "eval_history.csv"
LIVE_CSV = Path(__file__).parent.parent / "live_metrics.csv"

LIVE_COLS = ("run,steps,fps,lr,entropy,kl,clip_frac,exvar,"
             "loss,value_loss,pg_loss,"
             "dmg_pct,status_pct,setup_pct,other_pct,switch_pct,"
             "reward_mean,reward_std")


class LiveMetricsCallback(BaseCallback):
    """Write training metrics to CSV once per PPO update for dashboard."""

    def __init__(self, run_id: str = "", action_tracker=None):
        super().__init__(verbose=0)
        self.run_id = run_id
        self.action_tracker = action_tracker
        self._last_update = -1
        self._last_steps = 0
        self._last_time = 0.0
        # init CSV with header (overwrite per run)
        with open(LIVE_CSV, "w") as f:
            f.write(LIVE_COLS + "\n")

    def _on_step(self) -> bool:
        try:
            nv = self.model.logger.name_to_value
        except Exception:
            return True

        # only write when n_updates changes (once per PPO update)
        n_updates = nv.get("train/n_updates", self._last_update)
        if n_updates == self._last_update:
            return True
        self._last_update = n_updates

        # skip if no training data yet
        if "train/loss" not in nv:
            return True

        # compute fps ourselves since SB3 clears it before we can read
        import time as _time
        now = _time.time()
        fps = ""
        if self._last_time > 0:
            dt = now - self._last_time
            if dt > 0:
                fps = f"{(self.num_timesteps - self._last_steps) / dt:.0f}"
        self._last_steps = self.num_timesteps
        self._last_time = now

        def g(key, default=""):
            v = nv.get(key)
            return f"{v:.6g}" if v is not None else default

        # get action/reward data directly from tracker (logger clears too fast)
        dmg = status = setup = other = switch = rew_mean = rew_std = ""
        at = self.action_tracker
        if at and len(at._window) > 0:
            total = len(at._window)
            counts = {}
            for a in at._window:
                counts[a] = counts.get(a, 0) + 1
            dmg = f"{counts.get('damage', 0) / total:.6g}"
            status = f"{counts.get('status', 0) / total:.6g}"
            setup = f"{counts.get('setup', 0) / total:.6g}"
            other = f"{counts.get('other', 0) / total:.6g}"
            switch = f"{counts.get('switch', 0) / total:.6g}"
        if at and len(at._ep_rewards) > 0:
            rews = np.array(at._ep_rewards)
            rew_mean = f"{float(rews.mean()):.6g}"
            rew_std = f"{float(rews.std()):.6g}"

        row = ",".join([
            self.run_id,
            str(self.num_timesteps),
            fps,
            g("train/learning_rate"),
            g("train/entropy_loss"),
            g("train/approx_kl"),
            g("train/clip_fraction"),
            g("train/explained_variance"),
            g("train/loss"),
            g("train/value_loss"),
            g("train/policy_gradient_loss"),
            dmg,
            status,
            setup,
            other,
            switch,
            rew_mean,
            rew_std,
        ])
        with open(LIVE_CSV, "a") as f:
            f.write(row + "\n")
        return True


class EvalCheckpointCallback(BaseCallback):
    """Periodically evaluate vs baselines and save checkpoints."""

    def __init__(self, eval_freq: int, save_path: str, run_id: str = "",
                 n_games: int = 50, action_tracker=None, replay_games: int = 3,
                 verbose: int = 1):
        super().__init__(verbose)
        self.eval_freq = eval_freq
        self.save_path = save_path
        self.run_id = run_id
        self.n_games = n_games
        self.action_tracker = action_tracker
        self.replay_games = replay_games
        self._last_eval = 0
        # overwrite per-run log
        with open(EVAL_LOG, "w") as f:
            f.write(f"{'steps':>12}  {'vs_random':>10}  {'vs_maxdmg':>10}  {'vs_smart':>10}  {'vs_crystal':>10}\n")
            f.write("-" * 62 + "\n")
        # init master CSV if missing
        if not MASTER_CSV.exists():
            with open(MASTER_CSV, "w") as f:
                _cats = ("d", "s", "e", "o", "w", "f")
                _seq_hdr = ",".join(a + b for a in _cats for b in _cats)
                f.write("run,steps,vs_random,vs_maxdmg,vs_smart,vs_crystal,"
                        "rand_turns,md_turns,smart_turns,crystal_turns,"
                        "dmg_pct,status_pct,setup_pct,other_pct,switch_pct,fsw_pct,"
                        f"{_seq_hdr},"
                        "entropy,kl\n")

    def _on_step(self) -> bool:
        if self.num_timesteps - self._last_eval >= self.eval_freq:
            self._last_eval = self.num_timesteps
            ckpt = f"{self.save_path}_step{self.num_timesteps}"
            self.model.save(ckpt)
            from training.evaluate import evaluate_vs_baseline
            rates = {}
            avg_turns = {}
            for baseline in ["random", "max_damage", "smart", "crystal_ai"]:
                results = evaluate_vs_baseline(
                    self.model, baseline=baseline, n_games=self.n_games,
                )
                rates[baseline] = results["win_rate"]
                avg_turns[baseline] = results["avg_turns"]
                self.logger.record(f"eval/{baseline}_win_rate", results["win_rate"])
                self.logger.record(f"eval/{baseline}_avg_turns", results["avg_turns"])
                if self.verbose:
                    print(f"  [{self.num_timesteps:>10,}] vs {baseline}: "
                          f"{results['win_rate']:.1%} "
                          f"({results['wins']}W/{results['losses']}L) "
                          f"avg {results['avg_turns']:.0f}t")
            with open(EVAL_LOG, "a") as f:
                f.write(f"{self.num_timesteps:>12,}  "
                        f"{rates['random']:>9.1%}  "
                        f"{rates['max_damage']:>9.1%}  "
                        f"{rates['smart']:>9.1%}  "
                        f"{rates['crystal_ai']:>9.1%}\n")

            # record replays
            if self.replay_games > 0:
                from training.replay import record_games
                step_label = f"{self.num_timesteps // 1000}k"
                replay_dir = Path(__file__).parent.parent / "replays" / f"{self.run_id}_{step_label}"
                for bl in ("max_damage", "smart"):
                    bl_dir = replay_dir / bl
                    try:
                        record_games(self.model, n_games=self.replay_games,
                                     baseline=bl, out_dir=bl_dir)
                    except Exception as e:
                        print(f"  [replay vs {bl}] failed: {e}")

            # append to master CSV with action dist + sequences + training metrics
            dmg = status = setup = other = switch = fsw = entropy = kl = ""
            _cats = ("d", "s", "e", "o", "w", "f")
            seq_keys = tuple(a + b for a in _cats for b in _cats)
            seq_vals = {s: "" for s in seq_keys}
            if self.action_tracker and len(self.action_tracker._window) > 0:
                total = len(self.action_tracker._window)
                counts = {}
                for a in self.action_tracker._window:
                    counts[a] = counts.get(a, 0) + 1
                dmg = f"{counts.get('damage', 0) / total:.3f}"
                status = f"{counts.get('status', 0) / total:.3f}"
                setup = f"{counts.get('setup', 0) / total:.3f}"
                other = f"{counts.get('other', 0) / total:.3f}"
                switch = f"{counts.get('switch', 0) / total:.3f}"
                fsw = f"{counts.get('forced_switch', 0) / total:.3f}"
                # sequence pairs from action tracker
                st = self.action_tracker._seq_total or 1
                sc = self.action_tracker._seq_counts
                _KEY_CAT = {"d": "damage", "s": "status", "e": "setup",
                            "o": "other", "w": "switch", "f": "forced_switch"}
                for key in seq_keys:
                    pair = (_KEY_CAT[key[0]], _KEY_CAT[key[1]])
                    seq_vals[key] = f"{sc.get(pair, 0) / st:.3f}"
            if hasattr(self.model, "logger") and self.model.logger is not None:
                try:
                    name_to_val = self.model.logger.name_to_value
                    entropy = f"{abs(name_to_val.get('train/entropy_loss', 0)):.3f}"
                    kl = f"{name_to_val.get('train/approx_kl', 0):.4f}"
                except Exception:
                    pass
            with open(MASTER_CSV, "a") as f:
                seq_str = ",".join(seq_vals[k] for k in seq_keys)
                f.write(f"{self.run_id},{self.num_timesteps},"
                        f"{rates['random']:.3f},{rates['max_damage']:.3f},"
                        f"{rates['smart']:.3f},{rates['crystal_ai']:.3f},"
                        f"{avg_turns['random']:.1f},{avg_turns['max_damage']:.1f},"
                        f"{avg_turns['smart']:.1f},{avg_turns['crystal_ai']:.1f},"
                        f"{dmg},{status},{setup},{other},{switch},{fsw},"
                        f"{seq_str},"
                        f"{entropy},{kl}\n")
        return True


def make_env(seed: int = 0, opponent_policy=None, use_action_masker: bool = True,
             reward_mode: str = "shaped", tier: str = "ou",
             opp_team_strategy: str | None = None):
    """Create a wrapped CrystalBattle env."""
    def _init():
        env = gymnasium.make("CrystalBattle-v1", opponent_policy=opponent_policy,
                             reward_mode=reward_mode, tier=tier,
                             opp_team_strategy=opp_team_strategy)
        if use_action_masker:
            env = ActionMasker(env, lambda e: e.unwrapped.action_masks())
        return env
    return _init


def linear_schedule(initial_lr: float, final_lr: float):
    """Linear LR decay from initial to final over training."""
    def schedule(progress_remaining: float) -> float:
        return final_lr + (initial_lr - final_lr) * progress_remaining
    return schedule


class MixedOpponent:
    """Opponent that mixes MaxDamage, SmartAgent, CrystalAI, SelfPlay, and Random.

    Rolls which agent to use per episode. Builds appropriate teams:
    - MaxDamage: offensive teams (4 damaging moves)
    - Crystal AI: boosted trainer teams
    - SelfPlay: frozen neural policy from opponent pool
    - Smart / Random: standard OU teams

    If weights_ref is provided, reads live weights from it (for curriculum).
    """

    def __init__(self, max_damage_weight: float = 0.5, smart_weight: float = 0.0,
                 crystal_weight: float = 0.0, crystal_layers: int = AI_CHAMPION,
                 seed: int = 0, weights_ref: dict | None = None,
                 model=None, pool: OpponentPool | None = None,
                 type_chart: TypeChart | None = None):
        self._rng = _random.Random(seed)
        self._md = MaxDamageAgent()
        self._smart = SmartAgent(seed=seed)
        self._crystal = CrystalAIAgent(layers=crystal_layers, seed=seed)
        self._w_md = max_damage_weight
        self._w_sm = smart_weight
        self._w_cr = crystal_weight
        self._weights_ref = weights_ref
        self._current_agent: str = "random"  # set per episode

        # self-play support
        self._model = model
        self._pool = pool
        self._type_chart = type_chart or TypeChart.load()
        self._neural_opponent = None
        self._neural_refresh_counter = 0

    def _roll_agent(self) -> str:
        """Roll which agent type to use this episode."""
        if self._weights_ref:
            w_md = self._weights_ref["maxdmg"]
            w_sm = self._weights_ref["smart"]
            w_cr = self._weights_ref["crystal"]
            w_sp = self._weights_ref.get("selfplay", 0.0)
        else:
            w_md = self._w_md
            w_sm = self._w_sm
            w_cr = self._w_cr
            w_sp = 0.0

        roll = self._rng.random()
        if roll < w_md:
            return "maxdmg"
        roll -= w_md
        if roll < w_sm:
            return "smart"
        roll -= w_sm
        if roll < w_cr:
            return "crystal"
        roll -= w_cr
        if roll < w_sp and self._pool and len(self._pool) > 0:
            return "selfplay"
        return "random"

    def _get_neural_opponent(self):
        """Get or refresh the frozen neural opponent."""
        if self._pool is None or len(self._pool) == 0 or self._model is None:
            return None
        # refresh every 50 episodes to pick up newer snapshots
        if self._neural_opponent is None or self._neural_refresh_counter % 50 == 0:
            self._neural_opponent = make_neural_opponent(
                self._model, self._pool, self._type_chart, self._rng,
            )
        self._neural_refresh_counter += 1
        return self._neural_opponent

    def build_team(self, data, rng):
        """Called by the env during reset() to build P2's team."""
        from gym_env.team_builder import build_team
        from training.crystal_ai import build_trainer_team, ALL_TRAINER_NAMES

        self._current_agent = self._roll_agent()

        if self._current_agent == "maxdmg":
            return build_team(data, rng=rng, tier="ou", strategy="offensive")
        elif self._current_agent == "crystal":
            trainer = self._rng.choice(ALL_TRAINER_NAMES)
            return build_trainer_team(data, trainer, rng=rng, boosted=True)
        elif self._current_agent == "selfplay":
            self._get_neural_opponent()  # warm up the frozen policy
            return build_team(data, rng=rng, tier="ou")
        else:
            return build_team(data, rng=rng, tier="ou")

    def __call__(self, opp_state, p1_state, battle_rng, **kwargs):
        agent = self._current_agent
        if agent == "maxdmg":
            return self._md.act(opp_state, p1_state)
        elif agent == "smart":
            return self._smart.act(opp_state, p1_state)
        elif agent == "crystal":
            return self._crystal.act(opp_state, p1_state)
        elif agent == "selfplay" and self._neural_opponent:
            return self._neural_opponent(opp_state, p1_state, battle_rng, **kwargs)
        actions = opp_state.valid_actions(p1_state)
        return self._rng.choice(actions)


def make_mixed_opponent(max_damage_weight: float = 0.5, smart_weight: float = 0.0,
                        crystal_weight: float = 0.0, crystal_layers: int = AI_CHAMPION,
                        seed: int = 0, weights_ref: dict | None = None,
                        model=None, pool: OpponentPool | None = None,
                        type_chart: TypeChart | None = None):
    """Create a MixedOpponent (convenience wrapper)."""
    return MixedOpponent(
        max_damage_weight=max_damage_weight, smart_weight=smart_weight,
        crystal_weight=crystal_weight, crystal_layers=crystal_layers,
        seed=seed, weights_ref=weights_ref,
        model=model, pool=pool, type_chart=type_chart,
    )


# ============================================================
# CURRICULUM SCHEDULE
# ============================================================

# each keyframe: (progress_frac, {weight_dict})
# weights interpolated linearly between keyframes
DEFAULT_CURRICULUM = [
    # aggressive self-play: heuristics for 500k, self-play ramps in fast
    (0.000, {"random": 0.0, "crystal": 0.0, "smart": 0.50, "maxdmg": 0.50, "selfplay": 0.0}),
    (0.025, {"random": 0.0, "crystal": 0.0, "smart": 0.40, "maxdmg": 0.40, "selfplay": 0.20}),
    (0.075, {"random": 0.0, "crystal": 0.0, "smart": 0.30, "maxdmg": 0.30, "selfplay": 0.40}),
    (0.15,  {"random": 0.0, "crystal": 0.0, "smart": 0.20, "maxdmg": 0.25, "selfplay": 0.55}),
    (0.50,  {"random": 0.0, "crystal": 0.0, "smart": 0.15, "maxdmg": 0.20, "selfplay": 0.65}),
    (1.00,  {"random": 0.0, "crystal": 0.0, "smart": 0.15, "maxdmg": 0.20, "selfplay": 0.65}),
]

GENTLE_CURRICULUM = [
    # original gentle ramp (no self-play)
    (0.00, {"random": 1.0, "crystal": 0.0, "smart": 0.0, "maxdmg": 0.0, "selfplay": 0.0}),
    (0.05, {"random": 0.5, "crystal": 0.3, "smart": 0.1, "maxdmg": 0.1, "selfplay": 0.0}),
    (0.15, {"random": 0.2, "crystal": 0.3, "smart": 0.2, "maxdmg": 0.3, "selfplay": 0.0}),
    (0.35, {"random": 0.1, "crystal": 0.1, "smart": 0.3, "maxdmg": 0.5, "selfplay": 0.0}),
    (0.60, {"random": 0.05, "crystal": 0.1, "smart": 0.2, "maxdmg": 0.65, "selfplay": 0.0}),
    (1.00, {"random": 0.05, "crystal": 0.1, "smart": 0.2, "maxdmg": 0.65, "selfplay": 0.0}),
]


def _lerp_weights(schedule: list, progress: float) -> dict:
    """Interpolate curriculum weights at a given progress (0.0 - 1.0)."""
    progress = max(0.0, min(1.0, progress))

    # find surrounding keyframes
    for i in range(len(schedule) - 1):
        t0, w0 = schedule[i]
        t1, w1 = schedule[i + 1]
        if progress <= t1:
            if t1 == t0:
                return dict(w0)
            alpha = (progress - t0) / (t1 - t0)
            return {k: w0[k] + alpha * (w1[k] - w0[k]) for k in w0}

    return dict(schedule[-1][1])


class CurriculumCallback(BaseCallback):
    """Gradually ramp opponent mix weights over training."""

    def __init__(
        self, weights_ref: dict, total_steps: int,
        schedule: list | None = None, verbose: int = 0,
    ):
        super().__init__(verbose)
        self._ref = weights_ref
        self._total = total_steps
        self._schedule = schedule or DEFAULT_CURRICULUM
        self._last_print = -1

    def _on_step(self) -> bool:
        progress = self.num_timesteps / self._total
        w = _lerp_weights(self._schedule, progress)
        self._ref["maxdmg"] = w["maxdmg"]
        self._ref["smart"] = w["smart"]
        self._ref["crystal"] = w["crystal"]
        self._ref["selfplay"] = w.get("selfplay", 0.0)
        # random is the remainder

        # print every 5% progress
        pct5 = int(progress * 20)
        if pct5 > self._last_print:
            self._last_print = pct5
            parts = [f"MaxDmg={w['maxdmg']:.0%}", f"Smart={w['smart']:.0%}",
                     f"Crystal={w['crystal']:.0%}"]
            sp = w.get("selfplay", 0.0)
            if sp > 0.001:
                parts.append(f"SelfPlay={sp:.0%}")
            parts.append(f"Random={w.get('random', 0):.0%}")
            print(f"  Curriculum {progress:.0%}: {' '.join(parts)}")
        return True


def main():
    parser = argparse.ArgumentParser(description="Train Crystal Battle agent")
    parser.add_argument("--total-steps", type=int, default=1_000_000,
                        help="Total training timesteps")
    parser.add_argument("--phase1-steps", type=int, default=None,
                        help="Phase 1 (vs random) steps (default: 20%% of total)")
    parser.add_argument("--phase2-steps", type=int, default=None,
                        help="Phase 2 (vs mixed) steps (default: 40%% of total)")
    parser.add_argument("--md-weight", type=float, default=0.5,
                        help="MaxDamage mix ratio in phase 2/3 (0.0-1.0)")
    parser.add_argument("--smart-weight", type=float, default=0.0,
                        help="SmartAgent mix ratio in phase 2/3 (0.0-1.0)")
    parser.add_argument("--n-envs", type=int, default=8,
                        help="Number of parallel environments")
    parser.add_argument("--lr", type=float, default=3e-4,
                        help="Learning rate (initial)")
    parser.add_argument("--lr-end", type=float, default=None,
                        help="Final learning rate (linear decay). None = constant LR")
    parser.add_argument("--gamma", type=float, default=0.99,
                        help="Discount factor (0.99 default, try 0.999 for long-horizon)")
    parser.add_argument("--ent-coef", type=float, default=0.02,
                        help="Entropy coefficient (higher = more exploration)")
    parser.add_argument("--n-steps", type=int, default=2048,
                        help="Rollout steps per env per update (default: 2048)")
    parser.add_argument("--batch-size", type=int, default=None,
                        help="Minibatch size (default: same as n_steps for LSTM, 256 for MLP)")
    parser.add_argument("--net-arch", type=int, nargs="+", default=[256, 256],
                        help="Hidden layer sizes (e.g. --net-arch 256 256)")
    parser.add_argument("--save-path", type=str, default="crystal_battle_ppo",
                        help="Model save path")
    parser.add_argument("--log-dir", type=str, default="./tb_logs",
                        help="TensorBoard log directory")
    parser.add_argument("--device", type=str, default="auto",
                        help="Torch device (cpu, cuda, auto)")
    parser.add_argument("--eval-freq", type=int, default=None,
                        help="Evaluate every N steps (default: total/10)")
    parser.add_argument("--eval-games", type=int, default=400,
                        help="Games per baseline per eval checkpoint")
    parser.add_argument("--resume", action="store_true",
                        help="Resume training from --save-path checkpoint")
    parser.add_argument("--skip-to-phase", type=int, default=None, choices=[1, 2, 3],
                        help="Skip earlier phases (useful with --resume)")
    parser.add_argument("--neural-weight", type=float, default=-1,
                        help="Self-play neural weight in phase 3 "
                             "(-1=adaptive ramp 0.2->0.6, 0.0-1.0=fixed)")
    parser.add_argument("--opp-team-strategy", type=str, default=None,
                        choices=["offensive"],
                        help="Opponent team building strategy (offensive=4 damaging moves)")
    parser.add_argument("--crystal-weight", type=float, default=0.0,
                        help="Crystal AI mix ratio in phase 2 (0.0-1.0)")
    parser.add_argument("--crystal-tier", type=str, default="champion",
                        choices=["champion", "executive", "trainer"],
                        help="Crystal AI difficulty tier")
    parser.add_argument("--reward-mode", type=str, default="shaped",
                        choices=["shaped", "sparse", "blended"],
                        help="Reward mode: shaped (HP diff), sparse (win/loss only), "
                             "blended (terminal + light HP diff)")
    parser.add_argument("--lstm", action="store_true",
                        help="Use recurrent (LSTM) policy instead of MLP")
    parser.add_argument("--attention", nargs="?", const="v1", default=None,
                        choices=["v1", "v2", "v3", "v4"],
                        help="Attention: v1 (single-head), v2 (multi-head + self-attn), v4 (global-cond + FFN), "
                             "v3 (+ cross-team + opp self-attn)")
    parser.add_argument("--run-id", type=str, default="",
                        help="Run identifier for eval_history.csv")
    parser.add_argument("--curriculum", action="store_true",
                        help="Use graduated curriculum blend instead of 3-phase training")
    parser.add_argument("--schedule", type=str, default="default",
                        choices=["default", "gentle"],
                        help="Curriculum schedule: default (steep + selfplay) or gentle (no selfplay)")
    args = parser.parse_args()

    if args.curriculum:
        _run_curriculum(args)
    else:
        _run_phased(args)


def _build_model(args, vec_env):
    """Create or resume a MaskablePPO / MaskableRecurrentPPO model."""
    if args.resume:
        print(f"Resuming from {args.save_path}...")
        model_cls = MaskableRecurrentPPO if args.lstm else MaskablePPO
        model = model_cls.load(
            args.save_path, env=vec_env, device=args.device,
            custom_objects={"observation_space": vec_env.observation_space},
        )
        model.tensorboard_log = args.log_dir
        return model

    policy_kwargs = {}
    if args.net_arch:
        policy_kwargs["net_arch"] = args.net_arch

    if args.lstm:
        if args.attention:
            from training.attention_extractor import (
                AttentionFeatureExtractor, AttentionV2FeatureExtractor,
                AttentionV3FeatureExtractor, AttentionV4FeatureExtractor,
            )
            extractor_map = {
                "v1": AttentionFeatureExtractor,
                "v2": AttentionV2FeatureExtractor,
                "v3": AttentionV3FeatureExtractor,
                "v4": AttentionV4FeatureExtractor,
            }
            policy_kwargs["features_extractor_class"] = extractor_map[args.attention]
            policy_kwargs["features_extractor_kwargs"] = {"features_dim": 256}

        lr = linear_schedule(args.lr, args.lr_end) if args.lr_end else args.lr
        batch_sz = args.batch_size or args.n_steps
        return MaskableRecurrentPPO(
            "MlpLstmPolicy", vec_env,
            learning_rate=lr, n_steps=args.n_steps, batch_size=batch_sz,
            n_epochs=10, gamma=args.gamma, gae_lambda=0.95, clip_range=0.2,
            ent_coef=args.ent_coef, verbose=1,
            tensorboard_log=args.log_dir, device=args.device,
            policy_kwargs=policy_kwargs or None,
        )
    else:
        mlp_batch = args.batch_size or 256
        return MaskablePPO(
            "MlpPolicy", vec_env,
            learning_rate=args.lr, n_steps=args.n_steps, batch_size=mlp_batch,
            n_epochs=10, gamma=args.gamma, gae_lambda=0.95, clip_range=0.2,
            ent_coef=args.ent_coef, verbose=1,
            tensorboard_log=args.log_dir, device=args.device,
            policy_kwargs=policy_kwargs or None,
        )


# ============================================================
# CURRICULUM MODE
# ============================================================

def _run_curriculum(args):
    """Single training run with graduated opponent curriculum."""
    algo_name = "MaskableRecurrentPPO (LSTM)" if args.lstm else "MaskablePPO (MLP)"
    use_masker = not args.lstm
    eval_freq = args.eval_freq or max(args.total_steps // 20, 1)

    schedule = DEFAULT_CURRICULUM if args.schedule == "default" else GENTLE_CURRICULUM
    has_selfplay = any(w.get("selfplay", 0) > 0 for _, w in schedule)

    print("=" * 60)
    print(f"  Crystal Battle -- {algo_name} (Curriculum)")
    print("=" * 60)
    print(f"  Total steps:                 {args.total_steps:>10,}")
    print(f"  Gamma:                       {args.gamma}")
    print(f"  Entropy coef:                {args.ent_coef}")
    lr_desc = f"{args.lr} -> {args.lr_end}" if args.lr_end else f"{args.lr} (constant)"
    print(f"  Learning rate:               {lr_desc}")
    net_desc = "x".join(str(x) for x in args.net_arch)
    if args.attention:
        net_desc += " + attention"
    print(f"  Network arch:                {net_desc}")
    print(f"  Eval every:                  {eval_freq:>10,} steps")
    print(f"  Reward mode:                 {args.reward_mode}")
    print(f"  Schedule:                    {args.schedule}")
    if has_selfplay:
        print(f"  Self-play:                   enabled (pool size 20)")
    print()
    print("  Curriculum schedule:")
    for frac, w in schedule:
        parts = [f"{k}={v:.0%}" for k, v in w.items() if v > 0]
        print(f"    {frac:>5.0%}: {', '.join(parts)}")
    print("=" * 60)

    # shared mutable weights dict -- curriculum callback writes, envs read
    weights_ref = {"maxdmg": 0.0, "smart": 0.0, "crystal": 0.0, "selfplay": 0.0}

    crystal_layers = {"champion": AI_CHAMPION, "executive": AI_EXECUTIVE,
                      "trainer": AI_TRAINER}[args.crystal_tier]

    # self-play pool (snapshots added by SelfPlayCallback)
    pool = OpponentPool(max_size=20) if has_selfplay else None
    tc = TypeChart.load() if has_selfplay else None

    # build model first so we can pass it to MixedOpponent for self-play
    # (need a temp env for model creation, then rebuild with opponent)
    tmp_env = DummyVecEnv([
        make_env(seed=i, use_action_masker=use_masker, reward_mode=args.reward_mode)
        for i in range(args.n_envs)
    ])
    model = _build_model(args, tmp_env)
    tmp_env.close()

    vec_env = DummyVecEnv([
        make_env(
            seed=i,
            opponent_policy=make_mixed_opponent(
                weights_ref=weights_ref,
                crystal_layers=crystal_layers,
                seed=i,
                model=model, pool=pool, type_chart=tc,
            ),
            use_action_masker=use_masker,
            reward_mode=args.reward_mode,
            opp_team_strategy=args.opp_team_strategy,
        )
        for i in range(args.n_envs)
    ])
    model.set_env(vec_env)

    action_cb = ActionTrackerCallback(log_freq=eval_freq, verbose=1)
    eval_cb = EvalCheckpointCallback(
        eval_freq=eval_freq, save_path=args.save_path,
        run_id=args.run_id, n_games=args.eval_games,
        action_tracker=action_cb, verbose=1,
    )
    live_cb = LiveMetricsCallback(run_id=args.run_id, action_tracker=action_cb)
    curriculum_cb = CurriculumCallback(
        weights_ref=weights_ref, total_steps=args.total_steps,
        schedule=schedule,
    )
    callbacks = [curriculum_cb, eval_cb, action_cb, live_cb]

    # self-play: snapshot policy into pool every 100k steps
    if pool is not None:
        sp_cb = SelfPlayCallback(pool, snapshot_freq=100_000, verbose=1)
        callbacks.insert(0, sp_cb)

    reset_steps = not args.resume
    model.learn(
        total_timesteps=args.total_steps,
        callback=callbacks,
        tb_log_name="crystal_curriculum",
        reset_num_timesteps=reset_steps,
    )
    vec_env.close()

    # ---- Save ----
    model.save(args.save_path)
    print(f"\nModel saved to {args.save_path}")

    # ---- Quick eval ----
    print("\nQuick evaluation...")
    from training.evaluate import evaluate_vs_baseline, print_evaluation
    for baseline in ["random", "max_damage", "smart", "crystal_ai"]:
        results = evaluate_vs_baseline(model, baseline=baseline, n_games=100)
        print_evaluation(results, baseline)

    print("\nDone. View logs with: tensorboard --logdir", args.log_dir)


# ============================================================
# PHASED MODE (LEGACY)
# ============================================================

def _run_phased(args):
    """Original 3-phase training: Random -> Mixed -> Self-play."""
    # default phase splits: 10% random, 40% MaxDamage, 50% self-play+MaxDamage
    p1 = args.phase1_steps if args.phase1_steps is not None else args.total_steps // 10
    p2 = args.phase2_steps if args.phase2_steps is not None else args.total_steps * 2 // 5
    p3 = args.total_steps - p1 - p2

    if args.skip_to_phase:
        if args.skip_to_phase >= 2:
            p1 = 0
        if args.skip_to_phase >= 3:
            p2 = 0
            p3 = args.total_steps

    algo_name = "MaskableRecurrentPPO (LSTM)" if args.lstm else "MaskablePPO (MLP)"
    print("=" * 60)
    print(f"  Crystal Battle -- {algo_name}")
    print("=" * 60)
    print(f"  Phase 1 (vs random):           {p1:>10,} steps")
    print(f"  Phase 2 (vs MaxDamage mix):    {p2:>10,} steps")
    print(f"  Phase 3 (self-play + MaxDmg):  {p3:>10,} steps")
    print(f"  MaxDamage mix weight:        {args.md_weight:.0%}")
    print(f"  SmartAgent mix weight:       {args.smart_weight:.0%}")
    print(f"  Gamma:                       {args.gamma}")
    print(f"  Entropy coef:                {args.ent_coef}")
    lr_desc = f"{args.lr} -> {args.lr_end}" if args.lr_end else f"{args.lr} (constant)"
    print(f"  Learning rate:               {lr_desc}")
    net_desc = "x".join(str(x) for x in args.net_arch)
    if args.attention:
        net_desc += " + attention"
    eval_freq_val = args.eval_freq or max(args.total_steps // 20, 1)
    print(f"  Network arch:                {net_desc}")
    print(f"  Eval every:                  {eval_freq_val:>10,} steps")
    print(f"  Reward mode:                 {args.reward_mode}")
    print("=" * 60)

    # LSTM doesn't use ActionMasker wrapper (masking handled in the algorithm)
    use_masker = not args.lstm

    # ============================================================
    # PHASE 1: VS RANDOM
    # ============================================================
    print(f"\nPhase 1: Training vs Random for {p1:,} steps...")

    vec_env = DummyVecEnv([
        make_env(seed=i, use_action_masker=use_masker, reward_mode=args.reward_mode)
        for i in range(args.n_envs)
    ])

    model = _build_model(args, vec_env)
    if args.resume:
        prior = model.num_timesteps
        print(f"  Checkpoint at {prior:,} steps, training {args.total_steps:,} more")
        p1 += prior
        p2 += prior
        p3 += prior

    eval_freq = args.eval_freq or max(args.total_steps // 20, 1)
    action_cb = ActionTrackerCallback(log_freq=eval_freq, verbose=1)
    eval_cb = EvalCheckpointCallback(
        eval_freq=eval_freq, save_path=args.save_path,
        run_id=args.run_id, n_games=args.eval_games,
        action_tracker=action_cb, verbose=1,
    )
    live_cb = LiveMetricsCallback(run_id=args.run_id, action_tracker=action_cb)
    callbacks = [eval_cb, action_cb, live_cb]

    # only create self-play pool if phase 3 exists
    pool = None
    if p3 > 0:
        pool = OpponentPool(max_size=20)
        self_play_cb = SelfPlayCallback(pool, snapshot_freq=50_000, verbose=1)
        callbacks.insert(0, self_play_cb)

    if p1 > 0:
        model.learn(
            total_timesteps=p1,
            callback=callbacks,
            tb_log_name="crystal_phase1",
        )

    pool_size = len(pool) if pool else 0
    print(f"\nPhase 1 complete. Pool size: {pool_size}")
    vec_env.close()

    # ============================================================
    # PHASE 2: VS MIXED (RANDOM + MAX DAMAGE)
    # ============================================================
    if p2 > 0:
        crystal_layers = {"champion": AI_CHAMPION, "executive": AI_EXECUTIVE,
                          "trainer": AI_TRAINER}[args.crystal_tier]
        opp_parts = [f"{args.md_weight:.0%} MaxDmg"]
        if args.smart_weight > 0:
            opp_parts.append(f"{args.smart_weight:.0%} Smart")
        if args.crystal_weight > 0:
            opp_parts.append(f"{args.crystal_weight:.0%} Crystal-{args.crystal_tier}")
        remainder = 1.0 - args.md_weight - args.smart_weight - args.crystal_weight
        if remainder > 0.01:
            opp_parts.append(f"{remainder:.0%} Random")
        print(f"\nPhase 2: Training vs Mixed ({', '.join(opp_parts)}) "
              f"for {p2:,} steps...")

        vec_env_mixed = DummyVecEnv([
            make_env(seed=100 + i,
                     opponent_policy=make_mixed_opponent(
                         args.md_weight, args.smart_weight,
                         crystal_weight=args.crystal_weight,
                         crystal_layers=crystal_layers, seed=100 + i),
                     use_action_masker=use_masker, reward_mode=args.reward_mode,
                     opp_team_strategy=args.opp_team_strategy)
            for i in range(args.n_envs)
        ])
        model.set_env(vec_env_mixed)

        model.learn(
            total_timesteps=p2,
            callback=callbacks,
            tb_log_name="crystal_phase2",
            reset_num_timesteps=False,
        )

        pool_size = len(pool) if pool else 0
        print(f"\nPhase 2 complete. Pool size: {pool_size}")
        vec_env_mixed.close()

    # ============================================================
    # PHASE 3: SELF-PLAY + MIXED
    # ============================================================
    if p3 > 0 and pool is not None and (len(pool) > 0 or args.resume):
        nw = args.neural_weight
        nw_desc = f"adaptive (0.2 -> 0.6)" if nw < 0 else f"fixed {nw:.0%}"
        print(f"\nPhase 3: Neural self-play + MaxDamage for {p3:,} steps...")
        print(f"  Opponent pool size: {len(pool)}")
        print(f"  Neural weight: {nw_desc}")

        tc = TypeChart.load()
        md = MaxDamageAgent()
        vec_env_sp = DummyVecEnv([
            make_env(seed=200 + i,
                     opponent_policy=make_mixed_neural_opponent(
                         model, pool, tc, md,
                         neural_weight=nw,
                         rng=_random.Random(200 + i),
                     ),
                     use_action_masker=use_masker, reward_mode=args.reward_mode)
            for i in range(args.n_envs)
        ])
        model.set_env(vec_env_sp)

        model.learn(
            total_timesteps=p3,
            callback=callbacks,
            tb_log_name="crystal_phase3",
            reset_num_timesteps=False,
        )

        vec_env_sp.close()

    # ---- Save ----
    model.save(args.save_path)
    print(f"\nModel saved to {args.save_path}")

    # ---- Quick eval ----
    print("\nQuick evaluation...")
    from training.evaluate import evaluate_vs_baseline, print_evaluation
    for baseline in ["random", "max_damage"]:
        results = evaluate_vs_baseline(model, baseline=baseline, n_games=100)
        print_evaluation(results, baseline)

    print("\nDone. View logs with: tensorboard --logdir", args.log_dir)


if __name__ == "__main__":
    main()
