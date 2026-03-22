# Battle events for logging / replay

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class SwitchEvent:
    player: int  # 1 or 2
    from_name: str | None
    to_name: str

@dataclass
class MoveEvent:
    player: int
    pokemon_name: str
    move_name: str
    damage: int
    effectiveness: float
    is_crit: bool
    target_hp_remaining: int

@dataclass
class FaintEvent:
    player: int
    pokemon_name: str

@dataclass
class StruggleEvent:
    player: int
    pokemon_name: str
    damage: int

@dataclass
class MissEvent:
    player: int
    pokemon_name: str
    move_name: str

@dataclass
class StatusMoveEvent:
    """Status move used -- no-op in v1."""
    player: int
    pokemon_name: str
    move_name: str

@dataclass
class StatusAppliedEvent:
    player: int
    pokemon_name: str
    status: str

@dataclass
class StatusCuredEvent:
    player: int
    pokemon_name: str
    status: str

@dataclass
class StatusPreventedEvent:
    player: int
    pokemon_name: str
    status: str
    reason: str  # "fully paralyzed", "fast asleep", "frozen solid"

@dataclass
class ResidualDamageEvent:
    player: int
    pokemon_name: str
    status: str
    damage: int

@dataclass
class ConfusionAppliedEvent:
    player: int
    pokemon_name: str

@dataclass
class ConfusionHitSelfEvent:
    player: int
    pokemon_name: str
    damage: int

@dataclass
class StatChangeEvent:
    player: int
    pokemon_name: str
    stat: str
    stages: int  # positive = boost, negative = drop

@dataclass
class HealEvent:
    player: int
    pokemon_name: str
    amount: int
    source: str  # "drain", "recover", etc.

@dataclass
class FlinchEvent:
    player: int
    pokemon_name: str

@dataclass
class SpikesSetEvent:
    player: int  # side that received spikes

@dataclass
class SpikesDamageEvent:
    player: int
    pokemon_name: str
    damage: int

@dataclass
class ScreenSetEvent:
    player: int
    screen: str  # "reflect" or "light_screen"

@dataclass
class ScreenExpiredEvent:
    player: int
    screen: str

@dataclass
class ProtectEvent:
    player: int
    pokemon_name: str
    success: bool

@dataclass
class LeechSeedAppliedEvent:
    player: int  # the seeded pokemon's side
    pokemon_name: str

@dataclass
class LeechSeedDrainEvent:
    player: int  # side being drained
    pokemon_name: str
    damage: int

@dataclass
class PhazeEvent:
    player: int  # side forced to switch
    pokemon_name: str
    forced_in: str

@dataclass
class HazeEvent:
    player: int

@dataclass
class WeatherSetEvent:
    player: int
    weather: str  # "sun", "rain", "sandstorm"

@dataclass
class WeatherDamageEvent:
    player: int
    pokemon_name: str
    damage: int

@dataclass
class WeatherExpiredEvent:
    weather: str


Event = (
    SwitchEvent | MoveEvent | FaintEvent | StruggleEvent | MissEvent
    | StatusMoveEvent | StatusAppliedEvent | StatusCuredEvent
    | StatusPreventedEvent | ResidualDamageEvent
    | ConfusionAppliedEvent | ConfusionHitSelfEvent
    | StatChangeEvent | HealEvent | FlinchEvent
    | SpikesSetEvent | SpikesDamageEvent
    | ScreenSetEvent | ScreenExpiredEvent
    | ProtectEvent | LeechSeedAppliedEvent | LeechSeedDrainEvent
    | PhazeEvent | HazeEvent
    | WeatherSetEvent | WeatherDamageEvent | WeatherExpiredEvent
)
