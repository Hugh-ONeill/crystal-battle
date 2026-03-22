# MaskableRecurrentPPO: RecurrentPPO with invalid action masking
# Subclasses RecurrentPPO + RecurrentActorCriticPolicy to thread action
# masks through rollout collection, training, and inference.

from __future__ import annotations

from copy import deepcopy
from typing import NamedTuple, Optional

import numpy as np
import torch as th
from gymnasium import spaces
from stable_baselines3.common.utils import obs_as_tensor
from stable_baselines3.common.vec_env import VecEnv

from sb3_contrib.common.recurrent.buffers import RecurrentRolloutBuffer
from sb3_contrib.common.recurrent.policies import RecurrentActorCriticPolicy
from sb3_contrib.common.recurrent.type_aliases import RNNStates
from sb3_contrib.ppo_recurrent import RecurrentPPO

HUGE_NEG = -1e8


def _apply_mask_to_logits(logits: th.Tensor, masks: th.Tensor) -> th.Tensor:
    """Set logits of invalid actions to a large negative value."""
    return th.where(masks.bool(), logits, th.tensor(HUGE_NEG, dtype=logits.dtype, device=logits.device))


# ============================================================
# POLICY
# ============================================================

class MaskableRecurrentPolicy(RecurrentActorCriticPolicy):
    """RecurrentActorCriticPolicy with action mask support."""

    def __init__(self, *args, use_sde=False, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(
        self,
        obs: th.Tensor,
        lstm_states: RNNStates,
        episode_starts: th.Tensor,
        deterministic: bool = False,
        action_masks: Optional[np.ndarray] = None,
    ) -> tuple[th.Tensor, th.Tensor, th.Tensor, RNNStates]:
        features = self.extract_features(obs)
        if self.share_features_extractor:
            pi_features = vf_features = features
        else:
            pi_features, vf_features = features

        latent_pi, lstm_states_pi = self._process_sequence(
            pi_features, lstm_states.pi, episode_starts, self.lstm_actor,
        )
        if self.lstm_critic is not None:
            latent_vf, lstm_states_vf = self._process_sequence(
                vf_features, lstm_states.vf, episode_starts, self.lstm_critic,
            )
        elif self.shared_lstm:
            latent_vf = latent_pi.detach()
            lstm_states_vf = (lstm_states_pi[0].detach(), lstm_states_pi[1].detach())
        else:
            latent_vf = self.critic(vf_features)
            lstm_states_vf = lstm_states_pi

        latent_pi = self.mlp_extractor.forward_actor(latent_pi)
        latent_vf = self.mlp_extractor.forward_critic(latent_vf)

        values = self.value_net(latent_vf)

        # get logits and apply mask before creating distribution
        mean_actions = self.action_net(latent_pi)
        if action_masks is not None:
            masks_th = th.as_tensor(action_masks, dtype=th.bool, device=mean_actions.device)
            mean_actions = _apply_mask_to_logits(mean_actions, masks_th)

        distribution = self.action_dist.proba_distribution(action_logits=mean_actions)
        actions = distribution.get_actions(deterministic=deterministic)
        log_prob = distribution.log_prob(actions)

        return actions, values, log_prob, RNNStates(lstm_states_pi, lstm_states_vf)

    def evaluate_actions(
        self,
        obs: th.Tensor,
        actions: th.Tensor,
        lstm_states: RNNStates,
        episode_starts: th.Tensor,
        action_masks: Optional[th.Tensor] = None,
    ) -> tuple[th.Tensor, th.Tensor, th.Tensor]:
        features = self.extract_features(obs)
        if self.share_features_extractor:
            pi_features = vf_features = features
        else:
            pi_features, vf_features = features

        latent_pi, _ = self._process_sequence(
            pi_features, lstm_states.pi, episode_starts, self.lstm_actor,
        )
        if self.lstm_critic is not None:
            latent_vf, _ = self._process_sequence(
                vf_features, lstm_states.vf, episode_starts, self.lstm_critic,
            )
        elif self.shared_lstm:
            latent_vf = latent_pi.detach()
        else:
            latent_vf = self.critic(vf_features)

        latent_pi = self.mlp_extractor.forward_actor(latent_pi)
        latent_vf = self.mlp_extractor.forward_critic(latent_vf)

        mean_actions = self.action_net(latent_pi)
        if action_masks is not None:
            mean_actions = _apply_mask_to_logits(mean_actions, action_masks.bool())

        distribution = self.action_dist.proba_distribution(action_logits=mean_actions)
        log_prob = distribution.log_prob(actions)
        values = self.value_net(latent_vf)

        return values, log_prob, distribution.entropy()

    def predict(
        self,
        observation: np.ndarray | dict[str, np.ndarray],
        state: tuple[np.ndarray, ...] | None = None,
        episode_start: np.ndarray | None = None,
        deterministic: bool = False,
        action_masks: Optional[np.ndarray] = None,
    ) -> tuple[np.ndarray, tuple[np.ndarray, ...] | None]:
        self.set_training_mode(False)
        obs, vectorized_env = self.obs_to_tensor(observation)

        if episode_start is None:
            episode_start = np.array([False])
        episode_start = th.tensor(episode_start, dtype=th.float32, device=self.device)

        if state is None:
            # initial hidden states
            n_envs = obs.shape[0] if vectorized_env else 1
            state = self._recurrent_initial_state(n_envs)
        else:
            state = (
                th.tensor(state[0], device=self.device, dtype=th.float32),
                th.tensor(state[1], device=self.device, dtype=th.float32),
                th.tensor(state[2], device=self.device, dtype=th.float32),
                th.tensor(state[3], device=self.device, dtype=th.float32),
            )

        lstm_states = RNNStates(
            (state[0], state[1]),
            (state[2], state[3]),
        )

        with th.no_grad():
            actions, _, _, lstm_states = self.forward(
                obs, lstm_states, episode_start,
                deterministic=deterministic, action_masks=action_masks,
            )

        new_state = (
            lstm_states.pi[0].cpu().numpy(),
            lstm_states.pi[1].cpu().numpy(),
            lstm_states.vf[0].cpu().numpy(),
            lstm_states.vf[1].cpu().numpy(),
        )

        actions = actions.cpu().numpy()
        if not vectorized_env:
            actions = actions.squeeze(0)

        return actions, new_state

    def _recurrent_initial_state(self, n_envs: int) -> tuple[th.Tensor, ...]:
        """Return zero-initialized LSTM hidden states."""
        single = lambda lstm: (
            th.zeros(lstm.num_layers, n_envs, lstm.hidden_size, device=self.device),
            th.zeros(lstm.num_layers, n_envs, lstm.hidden_size, device=self.device),
        )
        pi = single(self.lstm_actor)
        vf = single(self.lstm_critic) if self.lstm_critic is not None else pi
        return (pi[0], pi[1], vf[0], vf[1])


# ============================================================
# BUFFER SAMPLES WITH ACTION MASKS
# ============================================================

class MaskableRecurrentRolloutBufferSamples(NamedTuple):
    observations: th.Tensor
    actions: th.Tensor
    old_values: th.Tensor
    old_log_prob: th.Tensor
    advantages: th.Tensor
    returns: th.Tensor
    lstm_states: RNNStates
    episode_starts: th.Tensor
    mask: th.Tensor
    action_masks: th.Tensor


# ============================================================
# BUFFER
# ============================================================

class MaskableRecurrentRolloutBuffer(RecurrentRolloutBuffer):
    """RecurrentRolloutBuffer that also stores action masks."""

    def __init__(self, *args, action_dim: int = 10, **kwargs):
        self._action_dim = action_dim
        super().__init__(*args, **kwargs)

    def reset(self):
        super().reset()
        self.action_masks_buf = np.zeros(
            (self.buffer_size, self.n_envs, self._action_dim), dtype=np.float32,
        )

    def add(self, *args, action_masks: np.ndarray | None = None, **kwargs) -> None:
        if action_masks is not None:
            self.action_masks_buf[self.pos] = action_masks
        super().add(*args, **kwargs)

    def get(self, batch_size=None):
        assert self.full
        if not self.generator_ready:
            # flatten action_masks the same way parent flattens other tensors
            # (parent.get will set generator_ready=True and flatten its own tensors)
            self.action_masks_buf = self.swap_and_flatten(self.action_masks_buf)
        yield from super().get(batch_size)

    def _get_samples(self, batch_inds, env_change, env=None):
        samples = super()._get_samples(batch_inds, env_change, env)

        # apply same padding as other tensors
        padded_masks = self.pad(self.action_masks_buf[batch_inds])
        n_seq = len(self.seq_start_indices)
        max_length = padded_masks.shape[1]
        padded_masks = padded_masks.reshape(n_seq * max_length, self._action_dim)
        action_masks_th = th.as_tensor(padded_masks, device=self.device)

        return MaskableRecurrentRolloutBufferSamples(
            observations=samples.observations,
            actions=samples.actions,
            old_values=samples.old_values,
            old_log_prob=samples.old_log_prob,
            advantages=samples.advantages,
            returns=samples.returns,
            lstm_states=samples.lstm_states,
            episode_starts=samples.episode_starts,
            mask=samples.mask,
            action_masks=action_masks_th,
        )


# ============================================================
# ALGORITHM
# ============================================================

def _get_action_masks(env: VecEnv) -> np.ndarray:
    """Get action masks from vectorized env."""
    return np.stack(env.env_method("action_masks"))


class MaskableRecurrentPPO(RecurrentPPO):
    """RecurrentPPO with invalid action masking support."""

    policy_aliases = {
        "MlpLstmPolicy": MaskableRecurrentPolicy,
    }

    def __init__(self, policy, env, **kwargs):
        # resolve alias
        if isinstance(policy, str) and policy in self.policy_aliases:
            policy = self.policy_aliases[policy]
        super().__init__(policy, env, **kwargs)

    @classmethod
    def load(cls, path, env=None, device="auto", **kwargs):
        # ensure our custom policy class is used instead of SB3 default
        custom = kwargs.pop("custom_objects", {})
        custom["policy_class"] = MaskableRecurrentPolicy
        return super().load(path, env=env, device=device, custom_objects=custom, **kwargs)

    def _setup_model(self) -> None:
        super()._setup_model()
        # replace the rollout buffer with our maskable version
        action_dim = self.action_space.n if isinstance(self.action_space, spaces.Discrete) else 0
        lstm = self.policy.lstm_actor
        hidden_state_buffer_shape = (self.n_steps, lstm.num_layers, self.n_envs, lstm.hidden_size)
        self.rollout_buffer = MaskableRecurrentRolloutBuffer(
            buffer_size=self.n_steps,
            observation_space=self.observation_space,
            action_space=self.action_space,
            hidden_state_shape=hidden_state_buffer_shape,
            device=self.device,
            gamma=self.gamma,
            gae_lambda=self.gae_lambda,
            n_envs=self.n_envs,
            action_dim=action_dim,
        )

    def collect_rollouts(self, env, callback, rollout_buffer, n_rollout_steps):
        assert self._last_obs is not None
        self.policy.set_training_mode(False)

        n_steps = 0
        rollout_buffer.reset()

        if self.use_sde:
            self.policy.reset_noise(env.num_envs)

        callback.on_rollout_start()
        lstm_states = deepcopy(self._last_lstm_states)

        while n_steps < n_rollout_steps:
            if self.use_sde and self.sde_sample_freq > 0 and n_steps % self.sde_sample_freq == 0:
                self.policy.reset_noise(env.num_envs)

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

            actions = actions.cpu().numpy()
            clipped_actions = actions
            if isinstance(self.action_space, spaces.Box):
                clipped_actions = np.clip(actions, self.action_space.low, self.action_space.high)

            new_obs, rewards, dones, infos = env.step(clipped_actions)
            self.num_timesteps += env.num_envs

            callback.update_locals(locals())
            if not callback.on_step():
                return False

            self._update_info_buffer(infos, dones)
            n_steps += 1

            if isinstance(self.action_space, spaces.Discrete):
                actions = actions.reshape(-1, 1)

            # handle timeout bootstrapping
            for idx, done_ in enumerate(dones):
                if (
                    done_
                    and infos[idx].get("terminal_observation") is not None
                    and infos[idx].get("TimeLimit.truncated", False)
                ):
                    terminal_obs = self.policy.obs_to_tensor(infos[idx]["terminal_observation"])[0]
                    with th.no_grad():
                        terminal_lstm_state = (
                            lstm_states.vf[0][:, idx:idx + 1, :].contiguous(),
                            lstm_states.vf[1][:, idx:idx + 1, :].contiguous(),
                        )
                        ep_starts = th.tensor([False], dtype=th.float32, device=self.device)
                        terminal_value = self.policy.predict_values(
                            terminal_obs, terminal_lstm_state, ep_starts,
                        )[0]
                    rewards[idx] += self.gamma * terminal_value

            rollout_buffer.add(
                self._last_obs,
                actions,
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
            episode_starts = th.tensor(dones, dtype=th.float32, device=self.device)
            values = self.policy.predict_values(
                obs_as_tensor(new_obs, self.device), lstm_states.vf, episode_starts,
            )

        rollout_buffer.compute_returns_and_advantage(last_values=values, dones=dones)
        callback.on_rollout_end()
        return True

    def train(self) -> None:
        self.policy.set_training_mode(True)
        self._update_learning_rate(self.policy.optimizer)
        clip_range = self.clip_range(self._current_progress_remaining)
        clip_range_vf = None
        if self.clip_range_vf is not None:
            clip_range_vf = self.clip_range_vf(self._current_progress_remaining)

        entropy_losses, pg_losses, value_losses = [], [], []
        clip_fractions = []
        continue_training = True

        for epoch in range(self.n_epochs):
            approx_kl_divs = []
            for rollout_data in self.rollout_buffer.get(self.batch_size):
                actions = rollout_data.actions
                if isinstance(self.action_space, spaces.Discrete):
                    actions = rollout_data.actions.long().flatten()

                mask = rollout_data.mask > 1e-8

                values, log_prob, entropy = self.policy.evaluate_actions(
                    rollout_data.observations,
                    actions,
                    rollout_data.lstm_states,
                    rollout_data.episode_starts,
                    action_masks=rollout_data.action_masks,
                )

                values = values.flatten()
                advantages = rollout_data.advantages
                if self.normalize_advantage:
                    advantages = (advantages - advantages[mask].mean()) / (advantages[mask].std() + 1e-8)

                ratio = th.exp(log_prob - rollout_data.old_log_prob)

                policy_loss_1 = advantages * ratio
                policy_loss_2 = advantages * th.clamp(ratio, 1 - clip_range, 1 + clip_range)
                policy_loss = -th.mean(th.min(policy_loss_1, policy_loss_2)[mask])

                pg_losses.append(policy_loss.item())
                clip_fraction = th.mean((th.abs(ratio - 1) > clip_range).float()[mask]).item()
                clip_fractions.append(clip_fraction)

                if clip_range_vf is None:
                    values_pred = values
                else:
                    values_pred = rollout_data.old_values + th.clamp(
                        values - rollout_data.old_values, -clip_range_vf, clip_range_vf,
                    )

                value_loss = th.mean(((rollout_data.returns - values_pred) ** 2)[mask])
                value_losses.append(value_loss.item())

                if entropy is None:
                    entropy_loss = -th.mean(-log_prob[mask])
                else:
                    entropy_loss = -th.mean(entropy[mask])
                entropy_losses.append(entropy_loss.item())

                loss = policy_loss + self.ent_coef * entropy_loss + self.vf_coef * value_loss

                with th.no_grad():
                    log_ratio = log_prob - rollout_data.old_log_prob
                    approx_kl_div = th.mean(((th.exp(log_ratio) - 1) - log_ratio)[mask]).cpu().numpy()
                    approx_kl_divs.append(approx_kl_div)

                if self.target_kl is not None and approx_kl_div > 1.5 * self.target_kl:
                    continue_training = False
                    break

                self.policy.optimizer.zero_grad()
                loss.backward()
                th.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
                self.policy.optimizer.step()

            self._n_updates += self.n_epochs
            if not continue_training:
                break

        explained_var = self._compute_explained_variance()

        self.logger.record("train/entropy_loss", np.mean(entropy_losses))
        self.logger.record("train/policy_gradient_loss", np.mean(pg_losses))
        self.logger.record("train/value_loss", np.mean(value_losses))
        self.logger.record("train/approx_kl", np.mean(approx_kl_divs))
        self.logger.record("train/clip_fraction", np.mean(clip_fractions))
        self.logger.record("train/loss", loss.item())
        self.logger.record("train/explained_variance", explained_var)
        if hasattr(self.policy, "log_std"):
            self.logger.record("train/std", th.exp(self.policy.log_std).mean().item())
        self.logger.record("train/n_updates", self._n_updates, exclude="tensorboard")
        self.logger.record("train/clip_range", clip_range)
        if self.clip_range_vf is not None:
            self.logger.record("train/clip_range_vf", clip_range_vf)

    def _compute_explained_variance(self) -> float:
        """Compute explained variance from the rollout buffer."""
        values = self.rollout_buffer.values.flatten()
        returns = self.rollout_buffer.returns.flatten()
        var_returns = np.var(returns)
        if var_returns == 0:
            return float("nan")
        return float(1 - np.var(returns - values) / var_returns)

    def predict(
        self,
        observation,
        state=None,
        episode_start=None,
        deterministic=False,
        action_masks=None,
    ):
        return self.policy.predict(
            observation, state, episode_start, deterministic,
            action_masks=action_masks,
        )
