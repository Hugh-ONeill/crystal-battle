# TypeChart: effectiveness lookups from type_chart.json

from __future__ import annotations

import json
from pathlib import Path


class TypeChart:
    """17x17 Gen 2 type effectiveness chart."""

    def __init__(self, chart: dict[str, dict[str, float]]):
        self._chart = chart
        self._types = sorted(chart.keys())

    @classmethod
    def load(cls, path: Path | None = None) -> TypeChart:
        if path is None:
            path = Path(__file__).parent.parent / "data" / "type_chart.json"
        with open(path) as f:
            return cls(json.load(f))

    @property
    def types(self) -> list[str]:
        return list(self._types)

    def effectiveness(self, atk_type: str, def_type: str) -> float:
        """Single type vs single type multiplier."""
        return self._chart.get(atk_type, {}).get(def_type, 1.0)

    def combined_effectiveness(self, atk_type: str, def_types: list[str]) -> float:
        """Move type vs defender's type(s)."""
        mult = 1.0
        for dt in def_types:
            mult *= self.effectiveness(atk_type, dt)
        return mult
