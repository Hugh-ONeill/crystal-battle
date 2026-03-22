# MoveTemplate (immutable) + MoveSlot (tracks PP)

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class MoveTemplate:
    """Immutable move definition from data."""
    id: int
    name: str
    type: str
    power: int
    accuracy: int | None  # None = always hits
    pp: int
    priority: int
    damage_class: str  # "physical", "special", "status"
    meta: dict[str, Any] | None = field(default=None, repr=False)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> MoveTemplate:
        return cls(
            id=d["id"],
            name=d["name"],
            type=d["type"],
            power=d["power"],
            accuracy=d["accuracy"],
            pp=d["pp"],
            priority=d.get("priority", 0),
            damage_class=d["damage_class"],
            meta=d.get("meta"),
        )


# special singleton for Struggle
STRUGGLE = MoveTemplate(
    id=165, name="Struggle", type="normal", power=50,
    accuracy=None, pp=999, priority=0, damage_class="physical",
)


@dataclass
class MoveSlot:
    """Mutable move slot on a pokemon -- tracks remaining PP."""
    template: MoveTemplate
    current_pp: int = -1  # -1 means uninitialized

    def __post_init__(self):
        if self.current_pp == -1:
            self.current_pp = self.template.pp

    @property
    def has_pp(self) -> bool:
        return self.current_pp > 0

    def use(self) -> None:
        if self.current_pp > 0:
            self.current_pp -= 1
