# name normalization between Showdown and our engine
#
# Showdown uses lowercase, no-space identifiers ("thunderbolt", "selfdestruct")
# Our data uses title-case with spaces/hyphens ("Thunderbolt", "Self-Destruct")

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from engine.data_loader import DataStore


def _normalize(name: str) -> str:
    """Normalize a name for matching: lowercase, strip non-alphanumeric."""
    return re.sub(r"[^a-z0-9]", "", name.lower())


class NameMapper:
    """Bidirectional name mapping between Showdown and our engine."""

    def __init__(self, data: DataStore | None = None):
        if data is None:
            data = DataStore()

        # ---- species: normalized name -> species id ----
        self._species_by_name: dict[str, int] = {}
        self._species_name_by_id: dict[int, str] = {}
        for sid, pkmn in data.pokemon.items():
            norm = _normalize(pkmn["name"])
            self._species_by_name[norm] = sid
            self._species_name_by_id[sid] = pkmn["name"]

        # ---- moves: normalized name -> move id ----
        self._move_by_name: dict[str, int] = {}
        self._move_name_by_id: dict[int, str] = {}
        self._move_data: dict[int, dict[str, Any]] = data.moves
        for mid, move in data.moves.items():
            norm = _normalize(move["name"])
            self._move_by_name[norm] = mid
            self._move_name_by_id[mid] = move["name"]

        self._pokemon_data: dict[int, dict[str, Any]] = data.pokemon

    def species_id(self, showdown_name: str) -> int | None:
        """Resolve a Showdown species name to our species id."""
        return self._species_by_name.get(_normalize(showdown_name))

    def species_name(self, species_id: int) -> str | None:
        """Get our engine's species name from id."""
        return self._species_name_by_id.get(species_id)

    def move_id(self, showdown_name: str) -> int | None:
        """Resolve a Showdown move name to our move id."""
        return self._move_by_name.get(_normalize(showdown_name))

    def move_name(self, move_id: int) -> str | None:
        """Get our engine's move name from id."""
        return self._move_name_by_id.get(move_id)

    def move_data(self, move_id: int) -> dict[str, Any] | None:
        """Get raw move dict from our data."""
        return self._move_data.get(move_id)

    def pokemon_data(self, species_id: int) -> dict[str, Any] | None:
        """Get raw pokemon dict from our data."""
        return self._pokemon_data.get(species_id)


# ---- status mapping ----
# poke-env Status enum name -> our engine status string
STATUS_FROM_SHOWDOWN = {
    "BRN": "brn",
    "PAR": "par",
    "SLP": "slp",
    "FRZ": "frz",
    "PSN": "psn",
    "TOX": "tox",
}

# ---- weather mapping ----
# poke-env Weather enum name -> our engine weather string
WEATHER_FROM_SHOWDOWN = {
    "SUNNYDAY": "sun",
    "RAINDANCE": "rain",
    "SANDSTORM": "sandstorm",
}

# ---- stat name mapping ----
# poke-env boost keys -> our engine stat_stages keys
STAT_FROM_SHOWDOWN = {
    "atk": "attack",
    "def": "defense",
    "spa": "special_attack",
    "spd": "special_defense",
    "spe": "speed",
    "accuracy": "accuracy",
    "evasion": "evasion",
}
