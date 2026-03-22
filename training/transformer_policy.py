# transformer-based policy for Pokemon battle RL
# replaces LSTM with causal self-attention over a window of past observations
# the "hidden state" is the observation history buffer

from __future__ import annotations

import math
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import gymnasium
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

from training.attention_extractor import AttentionFeatureExtractor


class CausalTransformerBlock(nn.Module):
    """Single transformer block with causal (backward-only) attention."""

    def __init__(self, d_model: int, n_heads: int, ff_dim: int, dropout: float = 0.0):
        super().__init__()
        self.attn = nn.MultiheadAttention(d_model, n_heads, batch_first=True, dropout=dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, ff_dim),
            nn.GELU(),
            nn.Linear(ff_dim, d_model),
        )
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        # causal self-attention
        attn_out, _ = self.attn(x, x, x, attn_mask=mask)
        x = self.norm1(x + attn_out)
        # feedforward
        x = self.norm2(x + self.ff(x))
        return x


class TransformerCore(nn.Module):
    """Causal transformer over a sequence of feature tokens.

    Takes (batch, seq_len, d_model) and returns (batch, d_model) from the
    last token position.
    """

    def __init__(self, d_model: int = 256, n_heads: int = 4, n_layers: int = 3,
                 ff_dim: int = 512, max_seq_len: int = 64, dropout: float = 0.0):
        super().__init__()
        self.d_model = d_model
        self.max_seq_len = max_seq_len

        # learnable positional embeddings
        self.pos_embedding = nn.Embedding(max_seq_len, d_model)

        # transformer blocks
        self.blocks = nn.ModuleList([
            CausalTransformerBlock(d_model, n_heads, ff_dim, dropout)
            for _ in range(n_layers)
        ])

        self.norm = nn.LayerNorm(d_model)

    def _causal_mask(self, seq_len: int, device: torch.device) -> torch.Tensor:
        """Upper triangular mask for causal attention."""
        return torch.triu(torch.ones(seq_len, seq_len, device=device), diagonal=1).bool()

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        """
        Args:
            tokens: (batch, seq_len, d_model)
        Returns:
            (batch, d_model) -- output from last token position
        """
        batch, seq_len, _ = tokens.shape
        assert seq_len <= self.max_seq_len

        # add positional embeddings
        positions = torch.arange(seq_len, device=tokens.device)
        tokens = tokens + self.pos_embedding(positions)

        # causal mask
        mask = self._causal_mask(seq_len, tokens.device)

        # run through transformer blocks
        for block in self.blocks:
            tokens = block(tokens, mask=mask)

        tokens = self.norm(tokens)

        # return last token
        return tokens[:, -1, :]


class TransformerBattlePolicy(nn.Module):
    """Full policy network: obs extractor + transformer + policy/value heads.

    Maintains a rolling buffer of past observations as its "memory".
    No LSTM -- temporal reasoning comes from the transformer attending
    to all previous turns in the window.
    """

    def __init__(
        self,
        obs_space: gymnasium.spaces.Box,
        features_dim: int = 256,
        d_model: int = 256,
        n_heads: int = 4,
        n_layers: int = 3,
        ff_dim: int = 512,
        max_seq_len: int = 64,
        net_arch: list[int] | None = None,
    ):
        super().__init__()
        if net_arch is None:
            net_arch = [256, 256]

        self.features_dim = features_dim
        self.d_model = d_model
        self.max_seq_len = max_seq_len

        # per-turn feature extractor (same as v1 attention)
        self.extractor = AttentionFeatureExtractor(obs_space, features_dim=features_dim)

        # project features to transformer dimension if different
        if features_dim != d_model:
            self.proj = nn.Linear(features_dim, d_model)
        else:
            self.proj = nn.Identity()

        # transformer core
        self.transformer = TransformerCore(
            d_model=d_model, n_heads=n_heads, n_layers=n_layers,
            ff_dim=ff_dim, max_seq_len=max_seq_len,
        )

        # policy head
        policy_layers = []
        in_dim = d_model
        for h in net_arch:
            policy_layers.extend([nn.Linear(in_dim, h), nn.ReLU()])
            in_dim = h
        self.policy_net = nn.Sequential(*policy_layers)
        self.action_net = nn.Linear(in_dim, 10)

        # value head
        value_layers = []
        in_dim = d_model
        for h in net_arch:
            value_layers.extend([nn.Linear(in_dim, h), nn.ReLU()])
            in_dim = h
        self.value_net = nn.Sequential(*value_layers)
        self.value_out = nn.Linear(in_dim, 1)

    def forward(self, obs_sequence: torch.Tensor, action_masks: torch.Tensor | None = None):
        """
        Args:
            obs_sequence: (batch, seq_len, obs_dim) -- sequence of observations
            action_masks: (batch, 10) -- valid action mask for current step

        Returns:
            action_logits: (batch, 10)
            value: (batch, 1)
        """
        batch, seq_len, obs_dim = obs_sequence.shape

        # extract features for each timestep
        # reshape to (batch*seq_len, obs_dim) for the extractor
        flat_obs = obs_sequence.reshape(batch * seq_len, obs_dim)
        flat_features = self.extractor(flat_obs)  # (batch*seq_len, features_dim)
        features = flat_features.reshape(batch, seq_len, -1)

        # project to transformer dim
        tokens = self.proj(features)

        # run transformer
        output = self.transformer(tokens)  # (batch, d_model)

        # policy head
        policy_features = self.policy_net(output)
        logits = self.action_net(policy_features)

        # mask invalid actions
        if action_masks is not None:
            logits = logits + (1 - action_masks) * -1e9

        # value head
        value_features = self.value_net(output)
        value = self.value_out(value_features)

        return logits, value

    def predict(self, obs_sequence: torch.Tensor, action_masks: torch.Tensor | None = None,
                deterministic: bool = True):
        """Predict action from observation sequence."""
        with torch.no_grad():
            logits, value = self.forward(obs_sequence, action_masks)
            if deterministic:
                action = logits.argmax(dim=-1)
            else:
                probs = F.softmax(logits, dim=-1)
                action = torch.multinomial(probs, 1).squeeze(-1)
            return action, value


class TransformerAgent:
    """Wrapper that manages the observation buffer and provides a simple act() interface."""

    def __init__(self, model: TransformerBattlePolicy, max_seq_len: int = 64,
                 device: str = "cpu"):
        self.model = model.to(device)
        self.model.eval()
        self.max_seq_len = max_seq_len
        self.device = device
        self.obs_buffer = []

    def reset(self):
        """Clear the observation buffer for a new game."""
        self.obs_buffer = []

    def act(self, obs: np.ndarray, mask: np.ndarray, deterministic: bool = True) -> int:
        """Pick an action given current observation and valid action mask."""
        self.obs_buffer.append(obs)

        # keep only last max_seq_len observations
        if len(self.obs_buffer) > self.max_seq_len:
            self.obs_buffer = self.obs_buffer[-self.max_seq_len:]

        # build sequence tensor
        obs_seq = np.array(self.obs_buffer, dtype=np.float32)
        obs_t = torch.tensor(obs_seq, dtype=torch.float32, device=self.device).unsqueeze(0)
        mask_t = torch.tensor(mask, dtype=torch.float32, device=self.device).unsqueeze(0)

        action, _ = self.model.predict(obs_t, action_masks=mask_t, deterministic=deterministic)
        return action.item()
