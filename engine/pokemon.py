# PokemonSpecies (immutable) + Pokemon (mutable battle instance)

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .move import MoveSlot, MoveTemplate


# ============================================================
# GEN 2 STAT CALCULATION
# ============================================================

# v1: perfect DVs (15) and max stat experience (65535) for all pokemon
# stat_exp bonus = floor(ceil(sqrt(65535)) / 4) = floor(256/4) = 64
# DV = 15 for all stats
PERFECT_DV = 15
MAX_STAT_EXP_BONUS = 64
LEVEL = 100

# Gen 2 Hidden Power type table (index 0-15)
_HP_TYPES = [
    "fighting", "flying", "poison", "ground", "rock", "bug", "ghost", "steel",
    "fire", "water", "grass", "electric", "psychic", "ice", "dragon", "dark",
]

_HIDDEN_POWER_ID = 237


def calc_stat(base: int, is_hp: bool = False, dv: int = PERFECT_DV) -> int:
    """Gen 2 stat formula at level 100 with max stat exp."""
    core = ((base + dv) * 2 + MAX_STAT_EXP_BONUS) * LEVEL // 100
    if is_hp:
        return core + LEVEL + 10
    return core + 5


def calc_hidden_power(dvs: dict[str, int]) -> tuple[str, int]:
    """Gen 2 Hidden Power type and power from DVs.

    Type = ((atk_dv & 3) << 2 | (def_dv & 3)) % 16
    Power = ((5 * (bit3 sum) + (spc_dv & 3)) // 2) + 31
    where bit3 sum = atk(8) + def(4) + spc(2) + speed(1) for each DV with bit 3 set
    """
    atk = dvs.get("attack", PERFECT_DV)
    dfn = dvs.get("defense", PERFECT_DV)
    spc = dvs.get("special", PERFECT_DV)
    spd = dvs.get("speed", PERFECT_DV)

    type_idx = ((atk & 3) << 2 | (dfn & 3)) % 16
    hp_type = _HP_TYPES[type_idx]

    # v=spc(1), w=spd(2), x=def(4), y=atk(8) per Bulbapedia
    bit3_sum = (8 if atk & 8 else 0) + (4 if dfn & 8 else 0) + \
               (1 if spc & 8 else 0) + (2 if spd & 8 else 0)
    hp_power = ((5 * bit3_sum + (spc & 3)) // 2) + 31

    return hp_type, hp_power


def dvs_for_hidden_power(hp_type: str, max_power: bool = True) -> dict[str, int]:
    """Find DVs that produce a given Hidden Power type.

    Keeps DVs as high as possible for minimal stat loss.
    """
    target_idx = _HP_TYPES.index(hp_type)
    best_dvs = None
    best_power = 0
    best_stat_sum = 0

    # brute force the 4 relevant DVs (atk, def, spc, speed)
    # only need to check atk & 3 and def & 3 for type, rest affects power
    for atk in range(16):
        for dfn in range(16):
            if ((atk & 3) << 2 | (dfn & 3)) % 16 != target_idx:
                continue
            for spc in range(16):
                for spd in range(16):
                    dvs = {"attack": atk, "defense": dfn, "special": spc, "speed": spd}
                    _, power = calc_hidden_power(dvs)
                    stat_sum = atk + dfn + spc + spd
                    if max_power:
                        if power > best_power or (power == best_power and stat_sum > best_stat_sum):
                            best_power = power
                            best_stat_sum = stat_sum
                            best_dvs = dvs
                    else:
                        if stat_sum > best_stat_sum:
                            best_stat_sum = stat_sum
                            best_dvs = dvs
                            best_power = power

    return best_dvs or {"attack": 15, "defense": 15, "special": 15, "speed": 15}


@dataclass(frozen=True)
class PokemonSpecies:
    """Immutable species definition."""
    id: int
    name: str
    types: list[str]
    base_stats: dict[str, int]
    learnset: list[int]

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> PokemonSpecies:
        return cls(
            id=d["id"],
            name=d["name"],
            types=d["types"],
            base_stats=d["base_stats"],
            learnset=d["learnset"],
        )

    @property
    def computed_stats(self) -> dict[str, int]:
        return {
            "hp": calc_stat(self.base_stats["hp"], is_hp=True),
            "attack": calc_stat(self.base_stats["attack"]),
            "defense": calc_stat(self.base_stats["defense"]),
            "special_attack": calc_stat(self.base_stats["special_attack"]),
            "special_defense": calc_stat(self.base_stats["special_defense"]),
            "speed": calc_stat(self.base_stats["speed"]),
        }


@dataclass
class Pokemon:
    """Mutable battle instance of a pokemon."""
    species: PokemonSpecies
    dvs: dict[str, int] = field(default_factory=dict)  # Gen 2 DVs (0-15)
    stats: dict[str, int] = field(default_factory=dict)
    current_hp: int = -1
    move_slots: list[MoveSlot] = field(default_factory=list)
    item: str | None = None  # held item ("leftovers", "thickclub", etc.)
    status: str | None = None
    status_turns: int = 0
    confusion_turns: int = 0
    stat_stages: dict[str, int] = field(default_factory=dict)
    flinched: bool = False
    leech_seeded: bool = False
    protected: bool = False
    protect_consecutive: int = 0  # for Protect success rate decay
    recharging: bool = False      # Hyper Beam recharge turn
    charging_move: MoveTemplate | None = field(default=None, repr=False)  # charge turn (Solar Beam, Fly, etc.)
    semi_invulnerable: str | None = None  # "fly" or "dig" during charge turn
    locked_move: MoveTemplate | None = field(default=None, repr=False)  # Thrash/Outrage lock-in
    locked_turns: int = 0         # remaining lock-in turns

    def __post_init__(self):
        if not self.dvs:
            self.dvs = {
                "attack": PERFECT_DV, "defense": PERFECT_DV,
                "special": PERFECT_DV, "speed": PERFECT_DV,
            }
        if not self.stats:
            bs = self.species.base_stats
            self.stats = {
                "hp": calc_stat(bs["hp"], is_hp=True),
                "attack": calc_stat(bs["attack"], dv=self.dvs.get("attack", PERFECT_DV)),
                "defense": calc_stat(bs["defense"], dv=self.dvs.get("defense", PERFECT_DV)),
                "special_attack": calc_stat(bs["special_attack"], dv=self.dvs.get("special", PERFECT_DV)),
                "special_defense": calc_stat(bs["special_defense"], dv=self.dvs.get("special", PERFECT_DV)),
                "speed": calc_stat(bs["speed"], dv=self.dvs.get("speed", PERFECT_DV)),
            }
        if self.current_hp == -1:
            self.current_hp = self.stats["hp"]
        if not self.stat_stages:
            self.stat_stages = {
                "attack": 0, "defense": 0,
                "special_attack": 0, "special_defense": 0,
                "speed": 0, "accuracy": 0, "evasion": 0,
            }
        # resolve Hidden Power type/power for any HP move slots
        for slot in self.move_slots:
            if slot.template.id == _HIDDEN_POWER_ID:
                hp_type, hp_power = calc_hidden_power(self.dvs)
                slot.template = MoveTemplate(
                    id=_HIDDEN_POWER_ID, name="Hidden Power", type=hp_type,
                    power=hp_power, accuracy=slot.template.accuracy,
                    pp=slot.template.pp, priority=0,
                    damage_class=slot.template.damage_class, meta=slot.template.meta,
                )

    @property
    def max_hp(self) -> int:
        return self.stats["hp"]

    @property
    def hp_frac(self) -> float:
        return self.current_hp / self.max_hp if self.max_hp > 0 else 0.0

    @property
    def is_fainted(self) -> bool:
        return self.current_hp <= 0

    @property
    def name(self) -> str:
        return self.species.name

    @property
    def types(self) -> list[str]:
        return self.species.types

    def has_any_pp(self) -> bool:
        return any(slot.has_pp for slot in self.move_slots)

    def clear_status(self) -> None:
        """Remove non-volatile status."""
        self.status = None
        self.status_turns = 0

    def clear_confusion(self) -> None:
        """Remove confusion."""
        self.confusion_turns = 0

    def take_damage(self, amount: int) -> int:
        """Apply damage, clamp to 0. Return actual damage dealt."""
        actual = min(amount, self.current_hp)
        self.current_hp -= actual
        return actual

    def heal(self, amount: int) -> int:
        """Heal HP, clamp to max_hp. Return actual amount healed."""
        actual = min(amount, self.max_hp - self.current_hp)
        self.current_hp += actual
        return actual

    @classmethod
    def from_species(cls, species: PokemonSpecies, move_templates: list[MoveTemplate]) -> Pokemon:
        slots = [MoveSlot(template=mt) for mt in move_templates[:4]]
        return cls(species=species, move_slots=slots)
