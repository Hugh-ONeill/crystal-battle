"""BO-locked featurizer assembled from feature blocks.

Composes the five blocks that make up the 184-dim BO-locked state encoding:
  bo_canonical_slots (30) + opp_side_with_roles (126)
  + field_global (10) + hazards_screens_both_sides (8)
  + bo_matchup_signals (10) = 184

`make_bo_featurizer()` returns the configured `Featurizer`. Side orientation
is auto-detected from BO mon-id signatures via `detect_bo_side`; the wrapper
builds the ctx so blocks read BO-first regardless of which engine side holds
the team.
"""

from __future__ import annotations

import numpy as np

from showdown.feature_block import Featurizer, build
from showdown.feature_blocks import (
    BO_ITEM_IDX,
    BO_MON_SET,
    BO_N_ITEM_FLAGS,
    BO_N_OPP_ROLES,
    BO_OPP_ROLE_OF,
)
from showdown.features_core import BattleState, _mon_id


def detect_bo_side(state_str: str) -> int:
    """Return 0 if side_one holds the BO team, 1 if side_two. Decision is
    based on the count of BO mon ids in each side's first 6 slots."""
    state = BattleState.parse(state_str)
    s1 = sum(1 for f in state.sides[0].mons if _mon_id(f) in BO_MON_SET)
    s2 = sum(1 for f in state.sides[1].mons if _mon_id(f) in BO_MON_SET)
    return 0 if s1 >= s2 else 1


def make_bo_featurizer() -> Featurizer:
    """Build the BO-locked featurizer. Stateless; call once and reuse."""
    return Featurizer([
        build("bo_canonical_slots"),
        build("opp_side_with_roles",
              role_table=BO_OPP_ROLE_OF,
              n_roles=BO_N_OPP_ROLES,
              item_table=BO_ITEM_IDX,
              n_item_flags=BO_N_ITEM_FLAGS),
        build("field_global"),
        build("hazards_screens_both_sides"),
        build("bo_matchup_signals"),
    ])


# Pre-build at import time so the public shim avoids per-call construction.
_BO_FEATURIZER: Featurizer | None = None


def _featurizer() -> Featurizer:
    global _BO_FEATURIZER
    if _BO_FEATURIZER is None:
        _BO_FEATURIZER = make_bo_featurizer()
    return _BO_FEATURIZER


def parse_state_bo(state_str: str) -> np.ndarray:
    """Convert a poke_engine state string to a 184-dim BO-locked feature
    vector. Auto-detects which side holds the BO team."""
    f = _featurizer()
    parts = state_str.split("/")
    if len(parts) < 2:
        return np.zeros(f.dim, dtype=np.float32)
    bo_side = detect_bo_side(state_str)
    state = BattleState.parse(state_str)
    return f.from_state(state, ctx={"own_idx": bo_side, "opp_idx": 1 - bo_side})


STATE_BO_FEATURES = make_bo_featurizer().dim
