# DataStore: load JSON data into lookup dicts

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


class DataStore:
    """Loads and caches pokemon.json and moves.json."""

    def __init__(self, data_dir: Path | None = None):
        if data_dir is None:
            data_dir = Path(__file__).parent.parent / "data"
        self._data_dir = data_dir

        with open(data_dir / "pokemon.json") as f:
            pkmn_list = json.load(f)
        self.pokemon: dict[int, dict[str, Any]] = {p["id"]: p for p in pkmn_list}

        with open(data_dir / "moves.json") as f:
            move_list = json.load(f)
        self.moves: dict[int, dict[str, Any]] = {m["id"]: m for m in move_list}

    def get_pokemon(self, species_id: int) -> dict[str, Any]:
        return self.pokemon[species_id]

    def get_move(self, move_id: int) -> dict[str, Any]:
        return self.moves[move_id]

    def get_damaging_moves(self, species_id: int) -> list[int]:
        """Return move ids from learnset that have power > 0."""
        pkmn = self.pokemon[species_id]
        return [mid for mid in pkmn["learnset"] if self.moves.get(mid, {}).get("power", 0) > 0]
