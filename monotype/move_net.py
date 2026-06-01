"""
Per-turn move-prediction net for monotype.

Given the actor's active mon, the opp's active mon, current HP fractions,
both teams' monotypes, and the actor's 4 candidate moves, output a 4-way
softmax over which move the actor will pick.

Architecture (~80K params):
  - move embedding table (vocab_size -> 64)
  - team_type embedding (18 -> 16)
  - state encoder: cat(actor_ftr, opp_ftr, hp_actor, hp_opp,
                       tt_actor_emb, tt_opp_emb) -> MLP -> 128
  - per-move scoring head: cat(state_emb, move_emb) [192] -> MLP -> scalar
  - 4 scores -> softmax = move-pick distribution
"""

from __future__ import annotations

import torch
import torch.nn as nn

from monotype.featurizer_lead_preview import MON_DIM
from monotype.featurizer_move_state import STATE_DIM as TURN_STATE_DIM


class MoveNet(nn.Module):
    """V1 net — actor/opp + HP + team types. ~95K params, 52% val top-1.

    Retained for back-compat with `monotype/move_net.pt`.
    """
    def __init__(self, n_moves: int, n_types: int = 18,
                 move_emb_dim: int = 64, type_emb_dim: int = 16,
                 state_dim: int = 128, hidden_dim: int = 128):
        super().__init__()
        self.move_emb = nn.Embedding(n_moves + 1, move_emb_dim, padding_idx=n_moves)
        self.type_emb = nn.Embedding(n_types, type_emb_dim)

        state_input_dim = MON_DIM * 2 + 2 + 2 * type_emb_dim
        self.state_encoder = nn.Sequential(
            nn.Linear(state_input_dim, state_dim),
            nn.ReLU(),
            nn.Linear(state_dim, state_dim),
            nn.ReLU(),
        )
        self.score_head = nn.Sequential(
            nn.Linear(state_dim + move_emb_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, actor_ftr, opp_ftr, candidate_moves,
                hp_actor, hp_opp, tt_actor, tt_opp):
        tte_a = self.type_emb(tt_actor)
        tte_o = self.type_emb(tt_opp)
        state_in = torch.cat([
            actor_ftr, opp_ftr,
            hp_actor.unsqueeze(-1), hp_opp.unsqueeze(-1),
            tte_a, tte_o,
        ], dim=-1)
        state_emb = self.state_encoder(state_in)
        cand = candidate_moves.clamp_min(0)
        move_embs = self.move_emb(cand)
        state_b = state_emb.unsqueeze(1).expand(-1, 4, -1)
        cat = torch.cat([state_b, move_embs], dim=-1)
        scores = self.score_head(cat).squeeze(-1)
        return scores


class MoveNetV2(nn.Module):
    """V2 net — V1 inputs plus 53-dim dynamic state context (boosts,
    status, weather, terrain, hazards, screens).
    """
    def __init__(self, n_moves: int, n_types: int = 18,
                 move_emb_dim: int = 64, type_emb_dim: int = 16,
                 state_dim: int = 128, hidden_dim: int = 128,
                 turn_state_dim: int = TURN_STATE_DIM):
        super().__init__()
        self.move_emb = nn.Embedding(n_moves + 1, move_emb_dim, padding_idx=n_moves)
        self.type_emb = nn.Embedding(n_types, type_emb_dim)

        state_input_dim = (MON_DIM * 2 + 2 + 2 * type_emb_dim + turn_state_dim)
        self.state_encoder = nn.Sequential(
            nn.Linear(state_input_dim, state_dim),
            nn.ReLU(),
            nn.Linear(state_dim, state_dim),
            nn.ReLU(),
        )
        self.score_head = nn.Sequential(
            nn.Linear(state_dim + move_emb_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, actor_ftr, opp_ftr, turn_state, candidate_moves,
                hp_actor, hp_opp, tt_actor, tt_opp):
        """
        actor_ftr     (B, MON_DIM)
        opp_ftr       (B, MON_DIM)
        turn_state    (B, TURN_STATE_DIM)   — actor-relative dynamic state
        candidate_moves (B, 4)
        hp_actor, hp_opp (B,)
        tt_actor, tt_opp (B,)
        returns logits (B, 4)
        """
        tte_a = self.type_emb(tt_actor)
        tte_o = self.type_emb(tt_opp)
        state_in = torch.cat([
            actor_ftr, opp_ftr, turn_state,
            hp_actor.unsqueeze(-1), hp_opp.unsqueeze(-1),
            tte_a, tte_o,
        ], dim=-1)
        state_emb = self.state_encoder(state_in)
        cand = candidate_moves.clamp_min(0)
        move_embs = self.move_emb(cand)
        state_b = state_emb.unsqueeze(1).expand(-1, 4, -1)
        cat = torch.cat([state_b, move_embs], dim=-1)
        scores = self.score_head(cat).squeeze(-1)
        return scores
