"""Gen 2 v2 featurizer assembled from feature blocks.

Composes the legacy 609-dim v2 vector:
  v2_active_moves  (124) + v2_pokemon_side(0)  (228) + v2_pokemon_side(1)  (228)
  + v2_side_extras(0)  (12) + v2_side_extras(1)  (12) + v2_global  (5) = 609

v2 is side_one-perspective by convention; the public `parse_state_v2` shim
hardcodes ctx={'own_idx': 0, 'opp_idx': 1}.
"""

from __future__ import annotations

import numpy as np

from showdown.feature_block import Featurizer, build
from showdown.feature_blocks import (
    N_V2_MOVE_FEATURES,
    N_V2_POKEMON_FEATURES,
    N_V2_SIDE_EXTRAS,
)


def make_v2_featurizer() -> Featurizer:
    return Featurizer([
        build("v2_active_moves"),
        build("v2_pokemon_side", side_idx=0),
        build("v2_pokemon_side", side_idx=1),
        build("v2_side_extras", side_idx=0),
        build("v2_side_extras", side_idx=1),
        build("v2_global"),
    ])


_V2_FEATURIZER: Featurizer | None = None


def _featurizer() -> Featurizer:
    global _V2_FEATURIZER
    if _V2_FEATURIZER is None:
        _V2_FEATURIZER = make_v2_featurizer()
    return _V2_FEATURIZER


def parse_state_v2(state_str: str) -> np.ndarray:
    """Convert a poke_engine state string to a 609-dim v2 feature vector."""
    f = _featurizer()
    parts = state_str.split("/")
    if len(parts) < 2:
        return np.zeros(f.dim, dtype=np.float32)
    return f(state_str, ctx={"own_idx": 0, "opp_idx": 1})


STATE_FEATURES_V2 = make_v2_featurizer().dim
MOVE_FEATURES = N_V2_MOVE_FEATURES
POKEMON_FEATURES_V2 = N_V2_POKEMON_FEATURES
SIDE_EXTRAS_V2 = N_V2_SIDE_EXTRAS
