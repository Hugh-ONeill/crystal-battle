# Neural net evaluator for MCTS leaf nodes
# Loads the PPO model or standalone weights checkpoint
# Provides batched (value, policy_prior) inference with zero-init LSTM
#
# Usage:
#   evaluator = MctsEvaluator("imitation_ppo", device="cpu")
#   values, priors = evaluator.evaluate_batch(obs_batch, mask_batch)

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).parent.parent))

from gym_env.obs_builder import OBS_SIZE


class MctsEvaluator:
    """Wraps the policy/value model for batched MCTS leaf evaluation.

    Can load from either:
    - PPO .zip file (uses actor/critic LSTMs from SB3)
    - Standalone _weights.pt checkpoint (uses shared LSTM + custom value head)
    """

    def __init__(self, policy_path: str, device: str = "cpu"):
        self.device = device
        weights_path = f"{policy_path}_weights.pt"

        if Path(weights_path).exists():
            self._load_from_checkpoint(weights_path, device)
        else:
            self._load_from_ppo(policy_path, device)

    def _load_from_checkpoint(self, weights_path: str, device: str):
        """Load from standalone weights checkpoint (has trained value head)."""
        from training.attention_extractor import AttentionFeatureExtractor
        import gymnasium

        checkpoint = torch.load(weights_path, map_location=device, weights_only=True)
        features_dim = checkpoint["features_dim"]
        lstm_hidden = checkpoint["lstm_hidden"]
        net_arch = checkpoint["net_arch"]

        obs_space = gymnasium.spaces.Box(low=-10, high=10, shape=(OBS_SIZE,), dtype=np.float32)
        self.extractor = AttentionFeatureExtractor(obs_space, features_dim=features_dim).eval().to(device)
        self.lstm = nn.LSTM(features_dim, lstm_hidden, num_layers=1, batch_first=False).eval().to(device)

        # policy head
        head_layers = []
        in_dim = lstm_hidden
        for h in net_arch:
            head_layers.extend([nn.Linear(in_dim, h), nn.ReLU()])
            in_dim = h
        head_layers.append(nn.Linear(in_dim, 10))
        self.policy_head = nn.Sequential(*head_layers).eval().to(device)

        # value head
        self.value_head = nn.Sequential(
            nn.Linear(lstm_hidden, 256),
            nn.ReLU(),
            nn.Linear(256, 1),
            nn.Tanh(),
        ).eval().to(device)

        self.extractor.load_state_dict(checkpoint["extractor"])
        self.lstm.load_state_dict(checkpoint["lstm"])
        self.policy_head.load_state_dict(checkpoint["policy_head"])
        if "value_head" in checkpoint:
            self.value_head.load_state_dict(checkpoint["value_head"])

        self.lstm_hidden_size = lstm_hidden
        self._mode = "checkpoint"

    def _load_from_ppo(self, policy_path: str, device: str):
        """Load from PPO .zip file (separate actor/critic LSTMs)."""
        from training.maskable_recurrent_ppo import MaskableRecurrentPPO

        model = MaskableRecurrentPPO.load(policy_path, device=device)
        p = model.policy

        self.extractor = p.features_extractor.eval()
        self.lstm_actor = p.lstm_actor.eval()
        self.lstm_critic = p.lstm_critic.eval()
        self.policy_net = p.mlp_extractor.policy_net.eval()
        self.value_net = p.mlp_extractor.value_net.eval()
        self.action_net = p.action_net.eval()
        self.value_output = p.value_net.eval()

        self.lstm_hidden_size = p.lstm_actor.hidden_size
        self._mode = "ppo"

    @torch.no_grad()
    def evaluate_batch(
        self,
        obs_batch: np.ndarray,
        mask_batch: np.ndarray | None = None,
    ) -> tuple[list[float], list[list[float]]]:
        """Evaluate a batch of observations.

        Returns:
            values: list of N floats (-1 to 1, P1 perspective)
            priors: list of N lists of 10 floats (action probabilities)
        """
        n = len(obs_batch)
        if n == 0:
            return [], []

        obs_t = torch.tensor(np.asarray(obs_batch), dtype=torch.float32, device=self.device)
        if obs_t.dim() == 1:
            obs_t = obs_t.unsqueeze(0)

        features = self.extractor(obs_t)
        n = features.shape[0]

        if self._mode == "checkpoint":
            return self._eval_checkpoint(features, n, mask_batch)
        else:
            return self._eval_ppo(features, n, mask_batch)

    def _eval_checkpoint(self, features, n, mask_batch):
        """Eval using shared LSTM + custom heads."""
        h0 = torch.zeros(1, n, self.lstm_hidden_size, device=self.device)
        c0 = torch.zeros(1, n, self.lstm_hidden_size, device=self.device)

        features_seq = features.unsqueeze(0)  # (1, N, 256)
        lstm_out, _ = self.lstm(features_seq, (h0, c0))
        hidden = lstm_out.squeeze(0)  # (N, 256)

        # policy
        logits = self.policy_head(hidden)
        if mask_batch is not None:
            mask_t = torch.tensor(mask_batch, dtype=torch.float32, device=self.device)
            logits = logits + (1.0 - mask_t) * -1e9
        priors = F.softmax(logits, dim=-1)

        # value
        values = self.value_head(hidden).squeeze(-1)

        return values.cpu().tolist(), priors.cpu().tolist()

    def _eval_ppo(self, features, n, mask_batch):
        """Eval using separate actor/critic LSTMs from PPO."""
        h0 = torch.zeros(1, n, self.lstm_hidden_size, device=self.device)
        c0 = torch.zeros(1, n, self.lstm_hidden_size, device=self.device)

        features_seq = features.unsqueeze(0)
        actor_out, _ = self.lstm_actor(features_seq, (h0.clone(), c0.clone()))
        critic_out, _ = self.lstm_critic(features_seq, (h0.clone(), c0.clone()))
        actor_out = actor_out.squeeze(0)
        critic_out = critic_out.squeeze(0)

        logits = self.action_net(self.policy_net(actor_out))
        if mask_batch is not None:
            mask_t = torch.tensor(mask_batch, dtype=torch.float32, device=self.device)
            logits = logits + (1.0 - mask_t) * -1e9
        priors = F.softmax(logits, dim=-1)

        value_latent = self.value_net(critic_out)
        values = torch.tanh(self.value_output(value_latent)).squeeze(-1)

        return values.cpu().tolist(), priors.cpu().tolist()

    def evaluate_single(
        self,
        obs: np.ndarray,
        mask: np.ndarray | None = None,
    ) -> tuple[float, list[float]]:
        obs_batch = obs.reshape(1, -1)
        mask_batch = mask.reshape(1, -1) if mask is not None else None
        values, priors = self.evaluate_batch(obs_batch, mask_batch)
        return values[0], priors[0]
