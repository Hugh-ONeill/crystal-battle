# BattleState: top-level container for a battle in progress

from __future__ import annotations

import random
from dataclasses import dataclass, field

from .player_state import PlayerState


# weather constants
SUN = "sun"
RAIN = "rain"
SANDSTORM = "sandstorm"
WEATHER_DURATION = 5


@dataclass
class BattleState:
    p1: PlayerState
    p2: PlayerState
    turn: int = 0
    winner: int | None = None  # 1, 2, or None
    rng: random.Random = field(default_factory=lambda: random.Random())
    weather: str | None = None
    weather_turns: int = 0

    @property
    def is_over(self) -> bool:
        return self.winner is not None

    def check_winner(self) -> int | None:
        """Check and set winner if a side is fully defeated."""
        if self.p1.is_defeated:
            self.winner = 2
        elif self.p2.is_defeated:
            self.winner = 1
        return self.winner

    def get_player(self, player_num: int) -> PlayerState:
        return self.p1 if player_num == 1 else self.p2
