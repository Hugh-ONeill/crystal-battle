# Expert Iteration PPO: subclass of MaskableRecurrentPPO that uses
# Rust 2-ply search to override actions during rollout collection.
# the policy's log_prob for the search action is stored so PPO's
# importance sampling remains correct.
#
# Usage:
#   python training/exit_ppo.py --total-steps 10000000 --search-frac 0.3

from __future__ import annotations

import argparse
import random
import sys
from copy import deepcopy
from pathlib import Path

import numpy as np
import torch as th
from gymnasium import spaces

sys.path.insert(0, str(Path(__file__).parent.parent))

from stable_baselines3.common.utils import obs_as_tensor
from training.maskable_recurrent_ppo import MaskableRecurrentPPO, _get_action_masks

import crystal_engine_rs as ce
from engine.types import TypeChart
from training.opponent_model import OpponentPredictor
from gym_env.obs_builder import build_observation


class ExitPPO(MaskableRecurrentPPO):
    """PPO with Expert Iteration: search overrides some actions during rollouts."""

    # exclude unpicklable objects from SB3 save
    _exclude_from_save = {"opp_model", "rs_data", "tc", "_opp_hidden"}

    def __init__(self, *args, search_frac: float = 0.3, search_depth: int = 2,
                 opp_model_path: str = "opp_model.pt", **kwargs):
        super().__init__(*args, **kwargs)
        self.search_frac = search_frac
        self.search_depth = search_depth
        self._opp_model_path = opp_model_path
        self._init_search()

    def _init_search(self):
        """Initialize search components (called on init and after load)."""
        self.opp_model = OpponentPredictor()
        self.opp_model.load_state_dict(
            th.load(self._opp_model_path, map_location="cpu", weights_only=True))
        self.opp_model.eval()
        self.rs_data = ce.DataStore(str(Path(__file__).parent.parent / "data"))
        self.tc = TypeChart.load()
        n_envs = self.n_envs if hasattr(self, "n_envs") else 8
        self._opp_hidden = [None] * n_envs

    def save(self, path, *args, **kwargs):
        """Save without unpicklable search objects."""
        # temporarily remove search objects
        saved = {}
        for attr in self._exclude_from_save:
            if hasattr(self, attr):
                saved[attr] = getattr(self, attr)
                delattr(self, attr)
        try:
            super().save(path, *args, **kwargs)
        finally:
            for attr, val in saved.items():
                setattr(self, attr, val)

    def collect_rollouts(self, env, callback, rollout_buffer, n_rollout_steps):
        """Override: sometimes replace policy actions with search actions."""
        assert self._last_obs is not None
        self.policy.set_training_mode(False)

        n_steps = 0
        rollout_buffer.reset()
        callback.on_rollout_start()
        lstm_states = deepcopy(self._last_lstm_states)

        while n_steps < n_rollout_steps:
            action_masks = _get_action_masks(env)

            with th.no_grad():
                obs_tensor = obs_as_tensor(self._last_obs, self.device)
                episode_starts = th.tensor(
                    self._last_episode_starts, dtype=th.float32, device=self.device,
                )
                actions, values, log_probs, lstm_states = self.policy.forward(
                    obs_tensor, lstm_states, episode_starts,
                    action_masks=action_masks,
                )

            actions_np = actions.cpu().numpy()

            # ---- ExIt: override some actions with search ----
            for i in range(env.num_envs):
                if random.random() >= self.search_frac:
                    continue

                # get the underlying env's battle state
                try:
                    battle = env.envs[i].unwrapped._battle
                    if battle is None or battle.is_over:
                        continue
                except (AttributeError, IndexError):
                    continue

                mask_i = action_masks[i] if action_masks is not None else None
                if mask_i is None:
                    continue

                valid = [j for j in range(10) if mask_i[j]]
                if len(valid) <= 1:
                    continue

                # run search
                search_action = self._search_action(battle, valid, i)
                if search_action is not None and search_action != actions_np[i]:
                    actions_np[i] = search_action

                    # recompute log_prob for the search action under current policy
                    with th.no_grad():
                        obs_i = obs_tensor[i:i+1]
                        ep_start_i = episode_starts[i:i+1]
                        lstm_i_pi = (
                            lstm_states.pi[0][:, i:i+1, :].contiguous(),
                            lstm_states.pi[1][:, i:i+1, :].contiguous(),
                        )
                        lstm_i_vf = (
                            lstm_states.vf[0][:, i:i+1, :].contiguous(),
                            lstm_states.vf[1][:, i:i+1, :].contiguous(),
                        )
                        from sb3_contrib.common.recurrent.type_aliases import RNNStates
                        lstm_i = RNNStates(lstm_i_pi, lstm_i_vf)
                        search_action_t = th.tensor([[search_action]], device=self.device)
                        mask_t = th.tensor(mask_i, dtype=th.float32,
                                          device=self.device).unsqueeze(0)
                        _, new_log_prob, _ = self.policy.evaluate_actions(
                            obs_i, search_action_t, lstm_i, ep_start_i,
                            action_masks=mask_t,
                        )
                        log_probs[i] = new_log_prob.squeeze()

            # update actions tensor with any search overrides
            actions = th.tensor(actions_np, device=self.device)

            clipped_actions = actions_np
            new_obs, rewards, dones, infos = env.step(clipped_actions)
            self.num_timesteps += env.num_envs

            callback.update_locals(locals())
            if not callback.on_step():
                return False

            self._update_info_buffer(infos, dones)
            n_steps += 1

            if isinstance(self.action_space, spaces.Discrete):
                actions_store = actions.cpu().numpy().reshape(-1, 1)
            else:
                actions_store = actions.cpu().numpy()

            # handle timeout bootstrapping
            for idx, done_ in enumerate(dones):
                if (
                    done_
                    and infos[idx].get("terminal_observation") is not None
                    and infos[idx].get("TimeLimit.truncated", False)
                ):
                    terminal_obs = self.policy.obs_to_tensor(
                        infos[idx]["terminal_observation"])[0]
                    with th.no_grad():
                        terminal_lstm_state = (
                            lstm_states.vf[0][:, idx:idx + 1, :].contiguous(),
                            lstm_states.vf[1][:, idx:idx + 1, :].contiguous(),
                        )
                        ep_starts = th.tensor([False], dtype=th.float32,
                                              device=self.device)
                        terminal_value = self.policy.predict_values(
                            terminal_obs, terminal_lstm_state, ep_starts,
                        )[0]
                    rewards[idx] += self.gamma * terminal_value

                # reset opp hidden on episode end
                if done_:
                    self._opp_hidden[idx] = None

            rollout_buffer.add(
                self._last_obs,
                actions_store,
                rewards,
                self._last_episode_starts,
                values,
                log_probs,
                lstm_states=self._last_lstm_states,
                action_masks=action_masks,
            )

            self._last_obs = new_obs
            self._last_episode_starts = dones
            self._last_lstm_states = lstm_states

        with th.no_grad():
            episode_starts = th.tensor(
                dones.astype(np.float32), device=self.device)
            values = self.policy.predict_values(
                obs_as_tensor(new_obs, self.device),
                lstm_states.vf, episode_starts,
            )

        rollout_buffer.compute_returns_and_advantage(
            last_values=values, dones=dones)
        callback.update_locals(locals())
        callback.on_rollout_end()
        return True

    def _search_action(self, battle, valid, env_idx):
        """Run Rust 2-ply search for a single env."""
        try:
            obs = build_observation(
                battle.p1, battle.p2, self.tc, turn=battle.turn,
                weather=battle.weather, weather_turns=battle.weather_turns,
            )

            # predict opponent actions
            opp_probs, self._opp_hidden[env_idx] = self.opp_model.predict_single(
                obs, self._opp_hidden[env_idx])

            opp_mask = battle.p2.valid_action_mask(battle.p1, type_chart=self.tc)
            opp_probs = opp_probs * np.array(opp_mask, dtype=np.float32)
            s = opp_probs.sum()
            if s > 0:
                opp_probs /= s
            else:
                return None

            top_idx = np.argsort(opp_probs)[::-1][:5]
            opp_actions = [(int(j), float(opp_probs[j])) for j in top_idx
                          if opp_probs[j] > 0.01]
            if not opp_actions:
                return None

            # build Rust battle state
            rs_t1 = [self.rs_data.build_pokemon(
                p.species.id, [s.template.id for s in p.move_slots])
                for p in battle.p1.team]
            rs_t2 = [self.rs_data.build_pokemon(
                p.species.id, [s.template.id for s in p.move_slots])
                for p in battle.p2.team]

            rs_battle = ce.create_battle(rs_t1, rs_t2, seed=battle.turn * 1000)

            # sync HP/status
            for py_mon, rs_mon in zip(battle.p1.team, rs_battle.p1.team):
                rs_mon.set_hp(py_mon.current_hp)
                if py_mon.status:
                    rs_mon.set_status(py_mon.status)
            for py_mon, rs_mon in zip(battle.p2.team, rs_battle.p2.team):
                rs_mon.set_hp(py_mon.current_hp)
                if py_mon.status:
                    rs_mon.set_status(py_mon.status)

            rs_battle.set_active(0, battle.p1.active_index)
            rs_battle.set_active(1, battle.p2.active_index)

            # run search
            if self.search_depth >= 2:
                ranked = ce.search_2ply(
                    rs_battle, valid, opp_actions, opp_actions,
                    base_seed=battle.turn * 1000,
                )
            else:
                ranked = ce.search_1ply(
                    rs_battle, valid, opp_actions,
                    base_seed=battle.turn * 1000,
                )

            return ranked[0][0] if ranked else None

        except Exception:
            return None


# ============================================================
# CLI
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="ExIt PPO training")
    parser.add_argument("--total-steps", type=int, default=10000000)
    parser.add_argument("--n-envs", type=int, default=8)
    parser.add_argument("--n-steps", type=int, default=4096)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--ent-coef", type=float, default=0.02)
    parser.add_argument("--eval-freq", type=int, default=250000)
    parser.add_argument("--eval-games", type=int, default=400)
    parser.add_argument("--save-path", type=str, default="exit_policy")
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--search-frac", type=float, default=0.3,
                        help="Fraction of steps where search overrides the policy")
    parser.add_argument("--search-depth", type=int, default=2)
    parser.add_argument("--run-id", type=str, default="exit")
    parser.add_argument("--md-weight", type=float, default=0.5)
    parser.add_argument("--smart-weight", type=float, default=0.5)
    args = parser.parse_args()

    from stable_baselines3.common.vec_env import DummyVecEnv
    from training.train import make_env, make_mixed_opponent, EvalCheckpointCallback
    from training.attention_extractor import AttentionFeatureExtractor

    print("=" * 60)
    print("  Expert Iteration (ExIt) PPO Training")
    print("=" * 60)
    print(f"  Total steps:   {args.total_steps:,}")
    print(f"  Search frac:   {args.search_frac:.0%}")
    print(f"  Search depth:  {args.search_depth}-ply")
    print()

    vec_env = DummyVecEnv([
        make_env(seed=i,
                 opponent_policy=make_mixed_opponent(args.md_weight, args.smart_weight),
                 reward_mode="shaped")
        for i in range(args.n_envs)
    ])

    policy_kwargs = {
        "features_extractor_class": AttentionFeatureExtractor,
        "features_extractor_kwargs": {"features_dim": 256},
        "net_arch": [256, 256],
    }

    if args.resume:
        print(f"  Resuming from {args.resume}...")
        # create fresh ExitPPO then load weights from checkpoint
        model = ExitPPO(
            "MlpLstmPolicy", vec_env,
            learning_rate=args.lr, n_steps=args.n_steps,
            batch_size=args.n_steps, n_epochs=10,
            gamma=0.99, gae_lambda=0.95, clip_range=0.2,
            ent_coef=args.ent_coef, verbose=1,
            tensorboard_log="./tb_logs", device=args.device,
            policy_kwargs=policy_kwargs,
            search_frac=args.search_frac, search_depth=args.search_depth,
        )
        # load pre-trained weights
        base = MaskableRecurrentPPO.load(args.resume, device=args.device)
        model.policy.load_state_dict(base.policy.state_dict())
        del base
    else:
        model = ExitPPO(
            "MlpLstmPolicy", vec_env,
            learning_rate=args.lr, n_steps=args.n_steps,
            batch_size=args.n_steps, n_epochs=10,
            gamma=0.99, gae_lambda=0.95, clip_range=0.2,
            ent_coef=args.ent_coef, verbose=1,
            tensorboard_log="./tb_logs", device=args.device,
            policy_kwargs=policy_kwargs,
            search_frac=args.search_frac, search_depth=args.search_depth,
        )

    eval_cb = EvalCheckpointCallback(
        eval_freq=args.eval_freq,
        n_games=args.eval_games, run_id=args.run_id,
        save_path=args.save_path,
    )

    print(f"  Training for {args.total_steps:,} steps...")
    model.learn(
        total_timesteps=args.total_steps,
        callback=eval_cb,
        progress_bar=False,
    )

    model.save(args.save_path)
    vec_env.close()
    print("Done!")


if __name__ == "__main__":
    main()
