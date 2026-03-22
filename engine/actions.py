# Action dataclasses for battle commands

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class UseMove:
    """Use a move by slot index (0-3)."""
    slot_index: int


@dataclass(frozen=True)
class Switch:
    """Switch to a team member by team index (0-5)."""
    team_index: int


@dataclass(frozen=True)
class Struggle:
    """Forced when all PP exhausted."""
    pass


@dataclass(frozen=True)
class Forfeit:
    """Give up the battle."""
    pass


Action = UseMove | Switch | Struggle | Forfeit
