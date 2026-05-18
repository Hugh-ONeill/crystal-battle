"""Compat shim for the v2 gen 2 featurizer.

Logic was migrated to the block-based design (see `feature_block.py`,
`feature_blocks.py`, `featurizer_v2.py`). This module re-exports the
symbols downstream code imports.
"""

from __future__ import annotations

from showdown.feature_blocks import (
    N_TYPES_V2,
    N_V2_MOVE_FEATURES,
    N_V2_POKEMON_FEATURES,
    N_V2_SIDE_EXTRAS,
    TYPE_IDX_V2,
    TYPES_V2,
    V2_ITEM_IDX,
    V2_ITEMS,
    V2_MOVE_DB,
    V2_PHYSICAL_TYPES,
    V2_TYPE_CHART,
    V3_STATUS_IDX,
    V3_STATUSES,
    v2_get_move_props as get_move_props,
    v2_type_effectiveness as type_effectiveness,
)
from showdown.featurizer_v2 import (
    MOVE_FEATURES,
    POKEMON_FEATURES_V2,
    SIDE_EXTRAS_V2,
    STATE_FEATURES_V2,
    parse_state_v2,
)

# Legacy aliases:
N_TYPES = N_TYPES_V2
TYPE_CHART = V2_TYPE_CHART
PHYSICAL_TYPES = V2_PHYSICAL_TYPES
MOVE_DB = V2_MOVE_DB
ITEMS_V2 = V2_ITEMS
ITEM_IDX_V2 = V2_ITEM_IDX
STATUSES = V3_STATUSES
STATUS_IDX = V3_STATUS_IDX
N_ACTIONS = 9
