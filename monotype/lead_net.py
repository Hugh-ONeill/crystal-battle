"""
Small policy net for monotype lead picking.

Architecture:
  - Per-mon shared MLP encoder: 51 -> 128
  - Permutation-invariant opp-team context: mean of 6 mon embeddings -> 128
  - Per-own-mon score head: cat(own_mon_emb, opp_ctx) [256] -> MLP -> scalar
  - 6 scores -> softmax = lead-pick distribution over own team

The net is symmetric: each own-team slot is scored independently given the
opp context, so the predicted lead index points directly at a slot in the
input team. ~40K params; trains in seconds on CPU.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from monotype.featurizer_lead_preview import MON_DIM


class LeadPickerNet(nn.Module):
    def __init__(self, mon_dim: int = MON_DIM, emb_dim: int = 128,
                 hidden_dim: int = 128):
        super().__init__()
        self.mon_encoder = nn.Sequential(
            nn.Linear(mon_dim, emb_dim),
            nn.ReLU(),
            nn.Linear(emb_dim, emb_dim),
            nn.ReLU(),
        )
        self.score_head = nn.Sequential(
            nn.Linear(emb_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, own_team: torch.Tensor, opp_team: torch.Tensor) -> torch.Tensor:
        """
        Args:
            own_team: (B, 6, MON_DIM)
            opp_team: (B, 6, MON_DIM)
        Returns:
            logits: (B, 6) — pre-softmax lead-pick scores over own team.
        """
        B = own_team.size(0)
        own_emb = self.mon_encoder(own_team)  # (B, 6, emb)
        opp_emb = self.mon_encoder(opp_team)  # (B, 6, emb)
        opp_ctx = opp_emb.mean(dim=1, keepdim=True)  # (B, 1, emb)
        opp_ctx_b = opp_ctx.expand(-1, 6, -1)         # (B, 6, emb)
        cat = torch.cat([own_emb, opp_ctx_b], dim=-1)  # (B, 6, 2*emb)
        scores = self.score_head(cat).squeeze(-1)      # (B, 6)
        return scores

    def predict_lead(self, own_team: torch.Tensor, opp_team: torch.Tensor) -> torch.Tensor:
        """Convenience: argmax over the 6-way softmax → lead index per batch."""
        with torch.no_grad():
            return self.forward(own_team, opp_team).argmax(dim=-1)
