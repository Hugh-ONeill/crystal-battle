"""Shared parsing layer for crystal-battle featurizers.

`BattleState.parse(state_str)` runs the split-and-extract work once and hands
typed objects to feature blocks. Individual `_*` helpers below are the same
ones that were copy-pasted across features_v2/v3/bo; they're consolidated here
so blocks can import a single source of truth.

This module owns no domain semantics (no team tables, no role lookups) —
those live with the blocks that need them. Keep it that way; this file is the
parsing layer, not a featurizer.
"""

from __future__ import annotations

from dataclasses import dataclass


# ============================================================
# DATA TYPES
# ============================================================

@dataclass(frozen=True)
class Side:
    """One side's pre-parsed view. `parts` is the raw '=' split (mons + side
    extras); `mons` is each of the first 6 slots' field lists; `active_idx`
    is the currently-active mon's slot."""
    parts: list[str]
    mons: list[list[str]]
    active_idx: int


@dataclass(frozen=True)
class BattleState:
    """One parse per call. Blocks read from this; they don't re-split the raw
    string. `sides` is (side_one, side_two); the field strings carry their
    own trailing `;turns_remaining` segments and are passed through as-is for
    callers that want to split further."""
    raw: str
    sides: tuple[Side, Side]
    weather: str
    terrain: str
    trick_room: str

    @classmethod
    def parse(cls, state_str: str) -> "BattleState":
        parts = state_str.split("/")
        s1 = _parse_side(parts[0] if len(parts) > 0 else "")
        s2 = _parse_side(parts[1] if len(parts) > 1 else "")
        return cls(
            raw=state_str,
            sides=(s1, s2),
            weather=parts[2] if len(parts) > 2 else "NONE;0",
            terrain=parts[3] if len(parts) > 3 else "NONE;0",
            trick_room=parts[4] if len(parts) > 4 else "false;0",
        )


def _parse_side(side_str: str) -> Side:
    parts = side_str.split("=")
    mons = []
    for i in range(6):
        s = parts[i] if i < len(parts) else ""
        mons.append(s.split(","))
    return Side(parts=parts, mons=mons, active_idx=_active_idx(parts))


# ============================================================
# MON FIELD HELPERS
# ============================================================
# poke-engine emits each mon as a comma-separated record; these helpers are
# the canonical readers. Indices match the format consumed by features_bo.

def _mon_id(fields: list[str]) -> str:
    return fields[0].upper() if fields and fields[0] else ""


def _types(fields: list[str]) -> tuple[str, str]:
    """Current types (post-tera-aware), uppercase."""
    t1 = fields[2].upper() if len(fields) > 2 else ""
    t2 = fields[3].upper() if len(fields) > 3 else ""
    return t1, t2


def _hp_frac(fields: list[str]) -> float:
    if len(fields) < 8:
        return 0.0
    try:
        hp = int(fields[6]); maxhp = int(fields[7])
    except ValueError:
        return 0.0
    return hp / max(maxhp, 1)


def _alive(fields: list[str]) -> float:
    if len(fields) < 8:
        return 0.0
    try:
        return 1.0 if int(fields[6]) > 0 else 0.0
    except ValueError:
        return 0.0


def _item(fields: list[str]) -> str:
    return fields[10].upper() if len(fields) > 10 else ""


def _speed(fields: list[str]) -> float:
    if len(fields) < 18:
        return 0.0
    try:
        return float(fields[17])
    except ValueError:
        return 0.0


def _status_any(fields: list[str]) -> float:
    if len(fields) < 19:
        return 0.0
    return 0.0 if fields[18].upper() == "NONE" else 1.0


def _move_disabled_flags(fields: list[str]) -> list[float]:
    """4-dim 0/1 vector: 1.0 if the corresponding move slot is disabled (e.g.
    Choice-locked or Disabled)."""
    out = [0.0, 0.0, 0.0, 0.0]
    for i, idx in enumerate((22, 23, 24, 25)):
        if idx < len(fields):
            parts = fields[idx].split(";")
            if len(parts) > 1 and parts[1].lower() == "true":
                out[i] = 1.0
    return out


# ============================================================
# SIDE FIELD HELPERS
# ============================================================
# `parts` is the side string split on '='. Slots 0..5 are mons; slots 6+ are
# side-wide extras (active idx, boosts, side conditions). These readers know
# those indices.

def _active_idx(parts: list[str]) -> int:
    if len(parts) < 7:
        return 0
    try:
        ai = int(parts[6])
        return ai if 0 <= ai < 6 else 0
    except ValueError:
        return 0


def _active_boosts(parts: list[str]):
    """5 dims: atk/def/spa/spd/spe boost stages, scaled to [-1, 1] (÷6)."""
    import numpy as np
    out = np.zeros(5, dtype=np.float32)
    for i in range(5):
        idx = 11 + i
        if idx < len(parts):
            try:
                out[i] = max(min(int(parts[idx]) / 6.0, 1.0), -1.0)
            except ValueError:
                pass
    return out


def _hazards(parts: list[str]) -> tuple[float, float, float]:
    """(stealth_rock, spikes, toxic_spikes) on this side, normalized."""
    if len(parts) < 8:
        return 0.0, 0.0, 0.0
    sc = parts[7].split(";")

    def _get(idx, denom=1.0):
        try:
            return min(int(sc[idx]) / denom, 1.0) if idx < len(sc) else 0.0
        except ValueError:
            return 0.0

    return _get(13), _get(12, 3.0), _get(17, 2.0)


def _screens(parts: list[str]) -> float:
    """1.0 if any of reflect/light_screen/aurora_veil is up on this side."""
    if len(parts) < 8:
        return 0.0
    sc = parts[7].split(";")

    def _gt0(idx):
        try:
            return int(sc[idx]) > 0 if idx < len(sc) else False
        except ValueError:
            return False

    return 1.0 if (_gt0(10) or _gt0(3) or _gt0(0)) else 0.0


__all__ = [
    "BattleState", "Side",
    "_mon_id", "_types", "_hp_frac", "_alive", "_item", "_speed",
    "_status_any", "_move_disabled_flags",
    "_active_idx", "_active_boosts", "_hazards", "_screens",
]
