# Attention-based feature extractor for team-structured observations
# v1: single-head cross-attention (active -> bench)
# v2: multi-head cross-attention + self-attention within team

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

import gymnasium
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

def _safe_mask(mask: torch.Tensor) -> torch.Tensor:
    """Prevent all-True masks which cause NaN in MultiheadAttention.

    When all slots are masked (all fainted), unmask everything so attention
    produces valid (if meaningless) output instead of NaN.
    """
    all_masked = mask.all(dim=-1, keepdim=True)
    return mask & ~all_masked


# obs layout: active(407) + my_team(6x63=378) + opp_team(6x42=252) + global(15) = 1052
ACTIVE_DIM = 407
MY_TEAM_SLOTS = 6
MY_TEAM_PER = 63
OPP_TEAM_SLOTS = 6
OPP_TEAM_PER = 42
GLOBAL_DIM = 15

# ---- active section move layout ----
# active = hp_speed(5) + my_moves(4x39) + opp_moves(4x36) + rest(100)
MY_MOVE_FEATURES = 39
OPP_MOVE_FEATURES = 36
N_MOVE_SLOTS = 4
_PRE_MOVE_DIM = 5               # hp + speed features before moves
_MY_MOVES_START = _PRE_MOVE_DIM
_OPP_MOVES_START = _MY_MOVES_START + N_MOVE_SLOTS * MY_MOVE_FEATURES  # 157
_POST_MOVES_START = _OPP_MOVES_START + N_MOVE_SLOTS * OPP_MOVE_FEATURES  # 301
_NON_MOVE_DIM = ACTIVE_DIM - N_MOVE_SLOTS * (MY_MOVE_FEATURES + OPP_MOVE_FEATURES)  # 109


class AttentionFeatureExtractor(BaseFeaturesExtractor):
    """v1: single-head cross-attention (active matchup queries team slots).

    Uses shared move encoders applied per-slot, with pooled summaries for the
    active encoder (position-invariant matchup understanding) and per-slot
    embeddings routed to the output (position-aware action selection).
    """

    def __init__(
        self,
        observation_space: gymnasium.spaces.Box,
        features_dim: int = 192,
        embed_dim: int = 64,
        move_dim: int = 32,
    ):
        super().__init__(observation_space, features_dim)

        d = embed_dim
        # active encoder sees pooled moves (1 x move_dim each) instead of concat (4 x move_dim)
        active_input_dim = _NON_MOVE_DIM + move_dim * 2

        # ---- Shared move encoders (position-invariant) ----
        self.my_move_encoder = nn.Sequential(
            nn.Linear(MY_MOVE_FEATURES, move_dim),
            nn.ReLU(),
        )
        self.opp_move_encoder = nn.Sequential(
            nn.Linear(OPP_MOVE_FEATURES, move_dim),
            nn.ReLU(),
        )

        # ---- Encoders ----
        self.active_encoder = nn.Sequential(
            nn.Linear(active_input_dim, 256),
            nn.ReLU(),
        )
        self.my_team_encoder = nn.Sequential(
            nn.Linear(MY_TEAM_PER, d),
            nn.ReLU(),
        )
        self.opp_team_encoder = nn.Sequential(
            nn.Linear(OPP_TEAM_PER, d),
            nn.ReLU(),
        )
        self.global_encoder = nn.Sequential(
            nn.Linear(GLOBAL_DIM, 32),
            nn.ReLU(),
        )

        # ---- Single-head attention projections ----
        self.my_query = nn.Linear(256, d)
        self.my_key = nn.Linear(d, d)
        self.my_value = nn.Linear(d, d)

        self.opp_query = nn.Linear(256, d)
        self.opp_key = nn.Linear(d, d)
        self.opp_value = nn.Linear(d, d)

        # ---- Output ----
        # per-slot my_move embeddings (4*move_dim) appended for action routing
        self.output = nn.Sequential(
            nn.Linear(256 + N_MOVE_SLOTS * move_dim + d + d + 32, features_dim),
            nn.ReLU(),
        )

    def _attention(self, query, keys, values, mask):
        d_k = query.size(-1)
        scores = torch.bmm(query.unsqueeze(1), keys.transpose(1, 2)).squeeze(1)
        scores = scores / (d_k ** 0.5)
        scores = scores.masked_fill(mask, float("-inf"))
        weights = F.softmax(scores, dim=-1).nan_to_num(0.0)
        return torch.bmm(weights.unsqueeze(1), values).squeeze(1)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        batch = obs.size(0)
        idx = 0
        active_raw = obs[:, idx:idx + ACTIVE_DIM]; idx += ACTIVE_DIM
        my_team_flat = obs[:, idx:idx + MY_TEAM_SLOTS * MY_TEAM_PER]; idx += MY_TEAM_SLOTS * MY_TEAM_PER
        opp_team_flat = obs[:, idx:idx + OPP_TEAM_SLOTS * OPP_TEAM_PER]; idx += OPP_TEAM_SLOTS * OPP_TEAM_PER
        global_raw = obs[:, idx:idx + GLOBAL_DIM]

        # split active into non-move features + move slots
        pre_moves = active_raw[:, :_PRE_MOVE_DIM]
        my_moves = active_raw[:, _MY_MOVES_START:_OPP_MOVES_START].view(
            batch, N_MOVE_SLOTS, MY_MOVE_FEATURES)
        opp_moves = active_raw[:, _OPP_MOVES_START:_POST_MOVES_START].view(
            batch, N_MOVE_SLOTS, OPP_MOVE_FEATURES)
        post_moves = active_raw[:, _POST_MOVES_START:]

        # shared move encoding (same weights for all 4 slots)
        my_move_emb = self.my_move_encoder(my_moves)       # (batch, 4, move_dim)
        opp_move_emb = self.opp_move_encoder(opp_moves)    # (batch, 4, move_dim)

        # pooled summaries for active encoder (position-invariant understanding)
        my_move_pool = my_move_emb.mean(dim=1)             # (batch, move_dim)
        opp_move_pool = opp_move_emb.mean(dim=1)

        # per-slot embeddings for output (position-aware action selection)
        my_move_flat = my_move_emb.view(batch, -1)         # (batch, 4*move_dim)

        # active encoder: pooled moves + non-move context
        active_combined = torch.cat([pre_moves, my_move_pool, opp_move_pool, post_moves], dim=1)
        active_enc = self.active_encoder(active_combined)

        my_team = my_team_flat.view(batch, MY_TEAM_SLOTS, MY_TEAM_PER)
        opp_team = opp_team_flat.view(batch, OPP_TEAM_SLOTS, OPP_TEAM_PER)

        my_team_enc = self.my_team_encoder(my_team)
        opp_team_enc = self.opp_team_encoder(opp_team)
        global_enc = self.global_encoder(global_raw)

        # mask fainted/empty: alive is feature index 1 (after is_active)
        my_mask = my_team[:, :, 1] < 0.5
        q = self.my_query(active_enc)
        my_attn = self._attention(q, self.my_key(my_team_enc), self.my_value(my_team_enc), my_mask)

        opp_mask = opp_team[:, :, 1] < 0.5
        q = self.opp_query(active_enc)
        opp_attn = self._attention(q, self.opp_key(opp_team_enc), self.opp_value(opp_team_enc), opp_mask)

        # active understanding + per-slot moves + team attention + global
        combined = torch.cat([active_enc, my_move_flat, my_attn, opp_attn, global_enc], dim=1)
        return self.output(combined)


class AttentionV2FeatureExtractor(BaseFeaturesExtractor):
    """v2: multi-head cross-attention + self-attention within team.

    Improvements over v1:
    - Multi-head attention (4 heads) captures different matchup aspects
    - Self-attention within my team captures team synergies/weaknesses
    - LayerNorm + residual connections for training stability
    """

    def __init__(
        self,
        observation_space: gymnasium.spaces.Box,
        features_dim: int = 192,
        embed_dim: int = 64,
        n_heads: int = 4,
    ):
        super().__init__(observation_space, features_dim)

        d = embed_dim

        # ---- Encoders ----
        self.active_encoder = nn.Sequential(
            nn.Linear(ACTIVE_DIM, 128),
            nn.ReLU(),
        )
        self.my_team_encoder = nn.Sequential(
            nn.Linear(MY_TEAM_PER, d),
            nn.ReLU(),
        )
        self.opp_team_encoder = nn.Sequential(
            nn.Linear(OPP_TEAM_PER, d),
            nn.ReLU(),
        )
        self.global_encoder = nn.Sequential(
            nn.Linear(GLOBAL_DIM, 32),
            nn.ReLU(),
        )

        # ---- Self-attention: my team mons attend to each other ----
        self.my_self_attn = nn.MultiheadAttention(d, n_heads, batch_first=True)
        self.my_self_norm = nn.LayerNorm(d)

        # ---- Cross-attention: active queries my team (multi-head) ----
        self.active_to_my_query = nn.Linear(128, d)
        self.my_cross_attn = nn.MultiheadAttention(d, n_heads, batch_first=True)

        # ---- Cross-attention: active queries opp team (multi-head) ----
        self.active_to_opp_query = nn.Linear(128, d)
        self.opp_cross_attn = nn.MultiheadAttention(d, n_heads, batch_first=True)

        # ---- Output ----
        self.output = nn.Sequential(
            nn.Linear(128 + d + d + 32, features_dim),
            nn.ReLU(),
        )

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        batch = obs.size(0)

        # ---- Split observation ----
        idx = 0
        active_raw = obs[:, idx:idx + ACTIVE_DIM]; idx += ACTIVE_DIM
        my_team_flat = obs[:, idx:idx + MY_TEAM_SLOTS * MY_TEAM_PER]; idx += MY_TEAM_SLOTS * MY_TEAM_PER
        opp_team_flat = obs[:, idx:idx + OPP_TEAM_SLOTS * OPP_TEAM_PER]; idx += OPP_TEAM_SLOTS * OPP_TEAM_PER
        global_raw = obs[:, idx:idx + GLOBAL_DIM]

        my_team = my_team_flat.view(batch, MY_TEAM_SLOTS, MY_TEAM_PER)
        opp_team = opp_team_flat.view(batch, OPP_TEAM_SLOTS, OPP_TEAM_PER)

        # ---- Encode ----
        active_enc = self.active_encoder(active_raw)        # (batch, 128)
        my_team_enc = self.my_team_encoder(my_team)          # (batch, 6, d)
        opp_team_enc = self.opp_team_encoder(opp_team)       # (batch, 6, d)
        global_enc = self.global_encoder(global_raw)         # (batch, 32)

        # masks: True = ignore (fainted/empty), safe against all-masked
        # alive is feature index 1 (after is_active)
        my_mask = _safe_mask(my_team[:, :, 1] < 0.5)
        opp_mask = _safe_mask(opp_team[:, :, 1] < 0.5)

        # ---- Self-attention within my team ----
        my_self_out, _ = self.my_self_attn(
            my_team_enc, my_team_enc, my_team_enc,
            key_padding_mask=my_mask,
        )
        my_team_enc = self.my_self_norm(my_team_enc + my_self_out)

        # ---- Cross-attention: active -> my team (multi-head) ----
        my_query = self.active_to_my_query(active_enc).unsqueeze(1)  # (batch, 1, d)
        my_cross_out, _ = self.my_cross_attn(
            my_query, my_team_enc, my_team_enc,
            key_padding_mask=my_mask,
        )
        my_summary = my_cross_out.squeeze(1)  # (batch, d)

        # ---- Cross-attention: active -> opp team (multi-head) ----
        opp_query = self.active_to_opp_query(active_enc).unsqueeze(1)  # (batch, 1, d)
        opp_cross_out, _ = self.opp_cross_attn(
            opp_query, opp_team_enc, opp_team_enc,
            key_padding_mask=opp_mask,
        )
        opp_summary = opp_cross_out.squeeze(1)  # (batch, d)

        # ---- Combine and project ----
        combined = torch.cat([active_enc, my_summary, opp_summary, global_enc], dim=1)
        return self.output(combined)


class AttentionV3FeatureExtractor(BaseFeaturesExtractor):
    """v3: full team reasoning with cross-team attention.

    Adds over v2:
    - Cross-team attention: my team attends to opp team (team-vs-team matchups)
    - Opp self-attention: opp team mons attend to each other (opponent team structure)
    - 2-layer stacked attention for deeper reasoning
    """

    def __init__(
        self,
        observation_space: gymnasium.spaces.Box,
        features_dim: int = 192,
        embed_dim: int = 64,
        n_heads: int = 4,
    ):
        super().__init__(observation_space, features_dim)

        d = embed_dim

        # ---- Encoders ----
        self.active_encoder = nn.Sequential(
            nn.Linear(ACTIVE_DIM, 128),
            nn.ReLU(),
        )
        self.my_team_encoder = nn.Sequential(
            nn.Linear(MY_TEAM_PER, d),
            nn.ReLU(),
        )
        self.opp_team_encoder = nn.Sequential(
            nn.Linear(OPP_TEAM_PER, d),
            nn.ReLU(),
        )
        self.global_encoder = nn.Sequential(
            nn.Linear(GLOBAL_DIM, 32),
            nn.ReLU(),
        )

        # ---- Layer 1: self-attention within each team ----
        self.my_self_attn = nn.MultiheadAttention(d, n_heads, batch_first=True)
        self.my_self_norm = nn.LayerNorm(d)

        self.opp_self_attn = nn.MultiheadAttention(d, n_heads, batch_first=True)
        self.opp_self_norm = nn.LayerNorm(d)

        # ---- Layer 2: cross-team attention (my team -> opp team) ----
        self.cross_team_attn = nn.MultiheadAttention(d, n_heads, batch_first=True)
        self.cross_team_norm = nn.LayerNorm(d)

        # ---- Final aggregation: active queries both teams ----
        self.active_to_my_query = nn.Linear(128, d)
        self.my_cross_attn = nn.MultiheadAttention(d, n_heads, batch_first=True)

        self.active_to_opp_query = nn.Linear(128, d)
        self.opp_cross_attn = nn.MultiheadAttention(d, n_heads, batch_first=True)

        # ---- Output ----
        self.output = nn.Sequential(
            nn.Linear(128 + d + d + 32, features_dim),
            nn.ReLU(),
        )

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        batch = obs.size(0)

        # ---- Split observation ----
        idx = 0
        active_raw = obs[:, idx:idx + ACTIVE_DIM]; idx += ACTIVE_DIM
        my_team_flat = obs[:, idx:idx + MY_TEAM_SLOTS * MY_TEAM_PER]; idx += MY_TEAM_SLOTS * MY_TEAM_PER
        opp_team_flat = obs[:, idx:idx + OPP_TEAM_SLOTS * OPP_TEAM_PER]; idx += OPP_TEAM_SLOTS * OPP_TEAM_PER
        global_raw = obs[:, idx:idx + GLOBAL_DIM]

        my_team = my_team_flat.view(batch, MY_TEAM_SLOTS, MY_TEAM_PER)
        opp_team = opp_team_flat.view(batch, OPP_TEAM_SLOTS, OPP_TEAM_PER)

        # ---- Encode ----
        active_enc = self.active_encoder(active_raw)
        my_team_enc = self.my_team_encoder(my_team)
        opp_team_enc = self.opp_team_encoder(opp_team)
        global_enc = self.global_encoder(global_raw)

        # alive is feature index 1 (after is_active)
        my_mask = _safe_mask(my_team[:, :, 1] < 0.5)
        opp_mask = _safe_mask(opp_team[:, :, 1] < 0.5)

        # ---- Layer 1: self-attention within each team ----
        out, _ = self.my_self_attn(my_team_enc, my_team_enc, my_team_enc,
                                   key_padding_mask=my_mask)
        my_team_enc = self.my_self_norm(my_team_enc + out)

        out, _ = self.opp_self_attn(opp_team_enc, opp_team_enc, opp_team_enc,
                                    key_padding_mask=opp_mask)
        opp_team_enc = self.opp_self_norm(opp_team_enc + out)

        # ---- Layer 2: cross-team attention ----
        out, _ = self.cross_team_attn(my_team_enc, opp_team_enc, opp_team_enc,
                                      key_padding_mask=opp_mask)
        my_team_enc = self.cross_team_norm(my_team_enc + out)

        # ---- Aggregation: active queries both team-aware representations ----
        my_q = self.active_to_my_query(active_enc).unsqueeze(1)
        my_out, _ = self.my_cross_attn(my_q, my_team_enc, my_team_enc,
                                       key_padding_mask=my_mask)
        my_summary = my_out.squeeze(1)

        opp_q = self.active_to_opp_query(active_enc).unsqueeze(1)
        opp_out, _ = self.opp_cross_attn(opp_q, opp_team_enc, opp_team_enc,
                                         key_padding_mask=opp_mask)
        opp_summary = opp_out.squeeze(1)

        combined = torch.cat([active_enc, my_summary, opp_summary, global_enc], dim=1)
        return self.output(combined)


class _FFN(nn.Module):
    """Feedforward block with residual + LayerNorm."""
    def __init__(self, d: int, expansion: int = 2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d, d * expansion),
            nn.ReLU(),
            nn.Linear(d * expansion, d),
        )
        self.norm = nn.LayerNorm(d)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.norm(x + self.net(x))


class AttentionV4FeatureExtractor(BaseFeaturesExtractor):
    """v4: improved architecture over v1-v3.

    Changes:
    - Parses tail features (multi-turn costs, active turns) into active encoding
    - 2-layer active encoder (wider first layer)
    - Global conditioning: weather/screens/stages injected into team slot
      representations before attention, so attention is context-aware
    - Self-attention within both teams (team structure reasoning)
    - FFN blocks after attention (standard transformer pattern)
    - LayerNorm + residual throughout
    """

    def __init__(
        self,
        observation_space: gymnasium.spaces.Box,
        features_dim: int = 192,
        embed_dim: int = 64,
        n_heads: int = 2,
    ):
        super().__init__(observation_space, features_dim)

        d = embed_dim

        # ---- Encoders ----
        self.active_encoder = nn.Sequential(
            nn.Linear(ACTIVE_DIM, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
        )
        self.my_team_encoder = nn.Sequential(
            nn.Linear(MY_TEAM_PER, d),
            nn.ReLU(),
        )
        self.opp_team_encoder = nn.Sequential(
            nn.Linear(OPP_TEAM_PER, d),
            nn.ReLU(),
        )
        self.global_encoder = nn.Sequential(
            nn.Linear(GLOBAL_DIM, d),
            nn.ReLU(),
        )

        # ---- Global conditioning: inject into team slots ----
        self.global_to_team = nn.Linear(d, d)

        # ---- Self-attention within each team ----
        self.my_self_attn = nn.MultiheadAttention(d, n_heads, batch_first=True)
        self.my_self_norm = nn.LayerNorm(d)
        self.my_ffn = _FFN(d)

        self.opp_self_attn = nn.MultiheadAttention(d, n_heads, batch_first=True)
        self.opp_self_norm = nn.LayerNorm(d)
        self.opp_ffn = _FFN(d)

        # ---- Cross-attention: active queries both teams ----
        self.active_to_my_query = nn.Linear(128, d)
        self.my_cross_attn = nn.MultiheadAttention(d, n_heads, batch_first=True)

        self.active_to_opp_query = nn.Linear(128, d)
        self.opp_cross_attn = nn.MultiheadAttention(d, n_heads, batch_first=True)

        # ---- Output ----
        self.output = nn.Sequential(
            nn.Linear(128 + d + d + d, features_dim),
            nn.ReLU(),
        )

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        batch = obs.size(0)

        # ---- Split observation ----
        idx = 0
        active_raw = obs[:, idx:idx + ACTIVE_DIM]; idx += ACTIVE_DIM
        my_team_flat = obs[:, idx:idx + MY_TEAM_SLOTS * MY_TEAM_PER]; idx += MY_TEAM_SLOTS * MY_TEAM_PER
        opp_team_flat = obs[:, idx:idx + OPP_TEAM_SLOTS * OPP_TEAM_PER]; idx += OPP_TEAM_SLOTS * OPP_TEAM_PER
        global_raw = obs[:, idx:idx + GLOBAL_DIM]

        my_team = my_team_flat.view(batch, MY_TEAM_SLOTS, MY_TEAM_PER)
        opp_team = opp_team_flat.view(batch, OPP_TEAM_SLOTS, OPP_TEAM_PER)

        # ---- Encode ----
        active_enc = self.active_encoder(active_raw)  # (batch, 128)
        my_team_enc = self.my_team_encoder(my_team)       # (batch, 6, d)
        opp_team_enc = self.opp_team_encoder(opp_team)    # (batch, 6, d)
        global_enc = self.global_encoder(global_raw)      # (batch, d)

        # ---- Global conditioning on team slots ----
        global_cond = self.global_to_team(global_enc).unsqueeze(1)  # (batch, 1, d)
        my_team_enc = my_team_enc + global_cond
        opp_team_enc = opp_team_enc + global_cond

        # masks: True = ignore (fainted/empty)
        my_mask = _safe_mask(my_team[:, :, 1] < 0.5)
        opp_mask = _safe_mask(opp_team[:, :, 1] < 0.5)

        # ---- Self-attention within each team + FFN ----
        out, _ = self.my_self_attn(my_team_enc, my_team_enc, my_team_enc,
                                   key_padding_mask=my_mask)
        my_team_enc = self.my_ffn(self.my_self_norm(my_team_enc + out))

        out, _ = self.opp_self_attn(opp_team_enc, opp_team_enc, opp_team_enc,
                                    key_padding_mask=opp_mask)
        opp_team_enc = self.opp_ffn(self.opp_self_norm(opp_team_enc + out))

        # ---- Cross-attention: active queries both teams ----
        my_q = self.active_to_my_query(active_enc).unsqueeze(1)
        my_out, _ = self.my_cross_attn(my_q, my_team_enc, my_team_enc,
                                       key_padding_mask=my_mask)
        my_summary = my_out.squeeze(1)  # (batch, d)

        opp_q = self.active_to_opp_query(active_enc).unsqueeze(1)
        opp_out, _ = self.opp_cross_attn(opp_q, opp_team_enc, opp_team_enc,
                                         key_padding_mask=opp_mask)
        opp_summary = opp_out.squeeze(1)  # (batch, d)

        # ---- Combine and project ----
        combined = torch.cat([active_enc, my_summary, opp_summary, global_enc], dim=1)
        return self.output(combined)
