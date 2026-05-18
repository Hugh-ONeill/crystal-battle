"""Gen 9 OU v3 featurizer assembled from feature blocks.

Composes five blocks into the 2738-dim general v3 vector:
  v3_pokemon_side(0)  (1338) + v3_side_extras(0)  (26)
  + v3_pokemon_side(1)  (1338) + v3_side_extras(1)  (26)
  + field_global  (10) = 2738

Side orientation is absolute (engine sides in order), not own/opp — this
featurizer is symmetric across teams. `parse_state_v3(state_str)` is the
public entry point and is preserved as the contract for downstream models.
"""

from __future__ import annotations

import numpy as np

from showdown.feature_block import Featurizer, build
from showdown.feature_blocks import (
    N_V3_MOVE_FEATS,
    N_V3_POKEMON_FEATURES,
    N_V3_SIDE_EXTRAS,
)
from showdown.features_core import BattleState


def make_v3_featurizer() -> Featurizer:
    """Build the v3 featurizer. Stateless; call once and reuse."""
    return Featurizer([
        build("v3_pokemon_side", side_idx=0),
        build("v3_side_extras", side_idx=0),
        build("v3_pokemon_side", side_idx=1),
        build("v3_side_extras", side_idx=1),
        build("field_global"),
    ])


_V3_FEATURIZER: Featurizer | None = None


def _featurizer() -> Featurizer:
    global _V3_FEATURIZER
    if _V3_FEATURIZER is None:
        _V3_FEATURIZER = make_v3_featurizer()
    return _V3_FEATURIZER


def parse_state_v3(state_str: str) -> np.ndarray:
    """Convert a poke_engine state string to a 2738-dim feature vector."""
    f = _featurizer()
    parts = state_str.split("/")
    if len(parts) < 2:
        return np.zeros(f.dim, dtype=np.float32)
    return f(state_str)


STATE_V3_FEATURES = make_v3_featurizer().dim

# Names downstream training scripts import — preserved verbatim.
POKEMON_V3_FEATURES = N_V3_POKEMON_FEATURES
SIDE_V3_EXTRAS = N_V3_SIDE_EXTRAS
N_GLOBAL = 10
N_MOVE_FEATS = N_V3_MOVE_FEATS
