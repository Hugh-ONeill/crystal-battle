"""Compat shim for the BO-locked featurizer.

The featurizer was migrated to a block-based design (see
`feature_block.py`, `feature_blocks.py`, `featurizer_bo.py`). This module
re-exports the three symbols downstream code imports — `parse_state_bo`,
`STATE_BO_FEATURES`, `_detect_bo_side` — plus the BO team-identity tables.
Add new featurizer logic in `featurizer_bo.py` / `feature_blocks.py`, not
here.
"""

from __future__ import annotations

from showdown.feature_blocks import (
    BO_ITEM_FLAGS,
    BO_ITEM_IDX,
    BO_MON_LIST,
    BO_MON_SET,
    BO_N_ITEM_FLAGS,
    BO_N_OPP_ROLES,
    BO_OPP_ROLES,
    BO_OPP_ROLE_OF,
    BO_SLOT_OF,
    DEFENSIVE_STEEL_GHOST,
    FIRE_STEEL_WALLS,
    PRIORITY_USERS,
    TERRAIN_IDX,
    TERRAINS,
    TRICK_ITEM_USERS,
    TYPE_IDX,
    TYPES_18,
    WEATHER_IDX,
    WEATHERS,
)
from showdown.featurizer_bo import (
    STATE_BO_FEATURES,
    detect_bo_side as _detect_bo_side,
    parse_state_bo,
)

# Aliases kept for parity with the pre-refactor module:
N_TYPES = len(TYPES_18)
N_OPP_ROLES = BO_N_OPP_ROLES
ROLE_IDX = {r: i for i, r in enumerate(BO_OPP_ROLES)}
OPP_ROLES = BO_OPP_ROLES
OPP_ROLE_OF = BO_OPP_ROLE_OF
ITEM_FLAGS = BO_ITEM_FLAGS
N_ITEM_FLAGS = BO_N_ITEM_FLAGS
ITEM_IDX = BO_ITEM_IDX
