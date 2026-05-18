"""Block-registry layer for crystal-battle featurizers.

A `FeatureBlock` emits a fixed-dim slice of the final feature vector. The
`Featurizer` concatenates blocks in order and exposes per-block slices so
training code can ablate or inspect a block by name.

Per-team featurizers are just a list of block names + kwargs — see
`featurizers/bo.py` for the canonical example. Block boundaries enforce the
output-dim contract: if a block's `dim` field gets out of sync with what
`extract` returns, `Featurizer.__call__` raises immediately rather than
silently producing a wrong-shape vector.
"""

from __future__ import annotations

from typing import Callable, Protocol, runtime_checkable

import numpy as np

from showdown.features_core import BattleState


# ============================================================
# BLOCK PROTOCOL
# ============================================================

@runtime_checkable
class FeatureBlock(Protocol):
    """A block emits `dim` floats from a parsed `BattleState`.

    `name` must be unique within a Featurizer (used for slicing/ablation).
    `ctx` is a free-form dict the assembler passes through — typical keys are
    `own_idx` / `opp_idx` for side orientation. Blocks should not mutate it.
    """
    name: str
    dim: int

    def extract(self, state: BattleState, ctx: dict) -> np.ndarray: ...


# ============================================================
# REGISTRY
# ============================================================

_REGISTRY: dict[str, Callable[..., FeatureBlock]] = {}


def register(name: str):
    """Decorator: register a block class under `name`. Calling
    `build(name, **kwargs)` later instantiates it. Re-registering the same
    name overwrites — useful for notebooks; in production, don't."""
    def deco(cls):
        _REGISTRY[name] = cls
        return cls
    return deco


def build(name: str, **kwargs) -> FeatureBlock:
    """Instantiate a registered block."""
    if name not in _REGISTRY:
        raise KeyError(f"no feature block registered as {name!r}; "
                       f"registered: {sorted(_REGISTRY)}")
    return _REGISTRY[name](**kwargs)


def registered() -> list[str]:
    """Names of all currently-registered blocks. Order is insertion order."""
    return list(_REGISTRY)


# ============================================================
# ASSEMBLER
# ============================================================

class Featurizer:
    """Ordered list of blocks. Call as `f(state_str, ctx)` to get a 1-D
    float32 array of length `f.dim`. Use `f.slices[name]` to read just the
    chunk emitted by a named block."""

    def __init__(self, blocks: list[FeatureBlock]):
        if len({b.name for b in blocks}) != len(blocks):
            raise ValueError("block names must be unique within a Featurizer")
        self.blocks = blocks
        self.slices: dict[str, slice] = {}
        off = 0
        for b in blocks:
            self.slices[b.name] = slice(off, off + b.dim)
            off += b.dim
        self.dim = off

    def __call__(self, state_str: str, ctx: dict | None = None) -> np.ndarray:
        state = BattleState.parse(state_str)
        return self.from_state(state, ctx)

    def from_state(self, state: BattleState, ctx: dict | None = None) -> np.ndarray:
        """Skip re-parsing if the caller already has a BattleState (e.g. when
        running multiple featurizers on the same state)."""
        ctx = ctx or {}
        out = np.empty(self.dim, dtype=np.float32)
        for b in self.blocks:
            chunk = b.extract(state, ctx)
            if chunk.shape != (b.dim,):
                raise ValueError(
                    f"block {b.name!r} returned shape {chunk.shape}, "
                    f"declared dim {b.dim}")
            out[self.slices[b.name]] = chunk
        return out


__all__ = ["FeatureBlock", "Featurizer", "register", "build", "registered"]
