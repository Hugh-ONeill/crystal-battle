"""Compat shim for the v3 general Gen 9 OU featurizer.

Logic was migrated to the block-based design (see `feature_block.py`,
`feature_blocks.py`, `featurizer_v3.py`). This module re-exports the
symbols downstream training/eval scripts import. Add new featurizer logic
in `featurizer_v3.py` / `feature_blocks.py`, not here.
"""

from __future__ import annotations

from showdown.feature_blocks import (
    N_TYPES_V3,
    N_V3_ABILITY_FLAGS,
    N_V3_ITEM_FLAGS,
    N_V3_MOVE_FEATS,
    N_V3_POKEMON_FEATURES,
    N_V3_SIDE_EXTRAS,
    TYPE_IDX_V3,
    TYPES_V3,
    V3_STATUS_IDX,
    V3_STATUSES,
)
from showdown.featurizer_v3 import (
    N_GLOBAL,
    N_MOVE_FEATS,
    POKEMON_V3_FEATURES,
    SIDE_V3_EXTRAS,
    STATE_V3_FEATURES,
    parse_state_v3,
)

# Legacy aliases used by some training scripts:
N_ABILITY_FLAGS = N_V3_ABILITY_FLAGS
N_ITEM_FLAGS = N_V3_ITEM_FLAGS
STATUSES = V3_STATUSES
STATUS_IDX = V3_STATUS_IDX
SIDE_V3_FEATURES = 6 * POKEMON_V3_FEATURES + SIDE_V3_EXTRAS
