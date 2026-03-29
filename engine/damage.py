# calc_damage(): Gen 2 damage formula (integer math)

from __future__ import annotations

import random as _random
from typing import TYPE_CHECKING

from .move import MoveTemplate
from .stat_stages import get_stage_multiplier
from .types import TypeChart

if TYPE_CHECKING:
    from .player_state import SideConditions
    from .pokemon import Pokemon

# Gen 2 type-boosting held items -> type they boost (1.1x damage)
_ITEM_TYPE_BOOST: dict[str | None, str] = {
    "charcoal": "fire",
    "mysticwater": "water",
    "magnet": "electric",
    "miracleseed": "grass",
    "nevermeltice": "ice",
    "blackbelt": "fighting",
    "poisonbarb": "poison",
    "softsand": "ground",
    "sharpbeak": "flying",
    "twistedspoon": "psychic",
    "silverpowder": "bug",
    "hardstone": "rock",
    "spelltag": "ghost",
    "dragonfang": "dragon",
    "blackglasses": "dark",
    "metalcoat": "steel",
    "pinkbow": "normal",
    "polkadotbow": "normal",
}


def calc_damage(
    attacker: Pokemon,
    defender: Pokemon,
    move: MoveTemplate,
    type_chart: TypeChart,
    rng: _random.Random | None = None,
    screens: SideConditions | None = None,
    weather: str | None = None,
) -> tuple[int, float, bool]:
    """
    Gen 2 damage formula. Returns (damage, effectiveness, is_crit).

    All integer division, faithful to the original engine.
    Status moves return (0, 1.0, False).
    """
    if move.damage_class == "status" or move.power == 0:
        return 0, 1.0, False

    if rng is None:
        rng = _random.Random()

    level = 100

    # ---- Attack / Defense stats ----
    if move.damage_class == "physical":
        atk_stat = "attack"
        dfn_stat = "defense"
    else:
        atk_stat = "special_attack"
        dfn_stat = "special_defense"

    atk = attacker.stats[atk_stat]
    dfn = defender.stats[dfn_stat]

    # ---- Item-based attack boosts ----
    item = attacker.item
    if item == "thickclub" and move.damage_class == "physical":
        if attacker.species.name in ("Marowak", "Cubone"):
            atk *= 2
    elif item == "lightball" and move.damage_class == "special":
        if attacker.species.name == "Pikachu":
            atk *= 2

    # sandstorm: Rock types get 1.5x SpDef
    if weather == "sandstorm" and dfn_stat == "special_defense":
        if "rock" in defender.types:
            dfn = dfn * 3 // 2

    # burn halves physical attack
    if attacker.status == "brn" and move.damage_class == "physical":
        atk //= 2

    # ---- Crit ----
    crit_threshold = attacker.stats["speed"] // 2
    # high-crit moves (Slash, Karate Chop, etc.) multiply threshold by 4
    meta = move.meta or {}
    if meta.get("crit_rate", 0) > 0:
        crit_threshold *= 4
    # gen 2: threshold / 256 chance (capped at 255)
    is_crit = rng.randint(0, 255) < min(crit_threshold, 255)

    if is_crit:
        crit_mult = 2
        # crits ignore negative attack stages and positive defense stages
        atk_stage = attacker.stat_stages.get(atk_stat, 0)
        if atk_stage > 0:
            atk_num, atk_den = get_stage_multiplier(atk_stage)
            atk = atk * atk_num // atk_den
        dfn_stage = defender.stat_stages.get(dfn_stat, 0)
        if dfn_stage < 0:
            dfn_num, dfn_den = get_stage_multiplier(dfn_stage)
            dfn = dfn * dfn_num // dfn_den
    else:
        crit_mult = 1
        # apply all stat stages normally
        atk_stage = attacker.stat_stages.get(atk_stat, 0)
        if atk_stage != 0:
            atk_num, atk_den = get_stage_multiplier(atk_stage)
            atk = atk * atk_num // atk_den
        dfn_stage = defender.stat_stages.get(dfn_stat, 0)
        if dfn_stage != 0:
            dfn_num, dfn_den = get_stage_multiplier(dfn_stage)
            dfn = dfn * dfn_num // dfn_den

    # ---- Base damage ----
    # ((2*Level/5+2) * Power * Atk/Def) / 50 + 2
    base = ((2 * level * crit_mult // 5 + 2) * move.power * atk // dfn) // 50 + 2

    # ---- STAB ----
    stab = 1.5 if move.type in attacker.types else 1.0

    # ---- Type-boosting held items (1.1x) ----
    item_boost = _ITEM_TYPE_BOOST.get(attacker.item, None)
    if item_boost is not None and move.type == item_boost:
        stab *= 1.1  # stacks with STAB

    # ---- Weather modifier ----
    # sun: fire 1.5x, water 0.5x | rain: water 1.5x, fire 0.5x
    weather_mult = 1.0
    if weather == "sun":
        if move.type == "fire":
            weather_mult = 1.5
        elif move.type == "water":
            weather_mult = 0.5
    elif weather == "rain":
        if move.type == "water":
            weather_mult = 1.5
        elif move.type == "fire":
            weather_mult = 0.5

    # ---- Type effectiveness ----
    effectiveness = type_chart.combined_effectiveness(move.type, defender.types)

    if effectiveness == 0.0:
        return 0, 0.0, is_crit

    # ---- Random factor (85-100) / 100 ----
    rand_factor = rng.randint(85, 100)

    # apply multipliers with integer math
    damage = base

    # stab: multiply by 3, divide by 2 (for 1.5x)
    if stab > 1.0:
        damage = damage * 3 // 2

    # weather
    if weather_mult == 1.5:
        damage = damage * 3 // 2
    elif weather_mult == 0.5:
        damage //= 2

    # type effectiveness (can be 0.25, 0.5, 1.0, 2.0, 4.0)
    # multiply as fraction to keep integer math
    if effectiveness == 0.25:
        damage = damage // 4
    elif effectiveness == 0.5:
        damage = damage // 2
    elif effectiveness == 2.0:
        damage = damage * 2
    elif effectiveness == 4.0:
        damage = damage * 4

    # random factor
    damage = damage * rand_factor // 100

    # screens: Reflect halves physical, Light Screen halves special
    # crits ignore screens in gen 2
    if not is_crit and screens is not None:
        if move.damage_class == "physical" and screens.reflect_turns > 0:
            damage //= 2
        elif move.damage_class == "special" and screens.light_screen_turns > 0:
            damage //= 2

    # minimum 1 damage
    damage = max(damage, 1)

    return damage, effectiveness, is_crit


def calc_expected_damage(
    attacker: Pokemon,
    defender: Pokemon,
    move: MoveTemplate,
    type_chart: TypeChart,
    weather: str | None = None,
    screens: SideConditions | None = None,
) -> float:
    """Expected damage (no crit, average random factor = 92.5/100)."""
    if move.damage_class == "status" or move.power == 0:
        return 0.0

    level = 100
    if move.damage_class == "physical":
        atk_stat, dfn_stat = "attack", "defense"
    else:
        atk_stat, dfn_stat = "special_attack", "special_defense"

    atk = attacker.stats[atk_stat]
    dfn = defender.stats[dfn_stat]

    # item-based attack boosts
    item = attacker.item
    if item == "thickclub" and move.damage_class == "physical":
        if attacker.species.name in ("Marowak", "Cubone"):
            atk *= 2
    elif item == "lightball" and move.damage_class == "special":
        if attacker.species.name == "Pikachu":
            atk *= 2

    # sandstorm: Rock types get 1.5x SpDef
    if weather == "sandstorm" and dfn_stat == "special_defense":
        if "rock" in defender.types:
            dfn = dfn * 3 // 2

    # apply stat stages
    atk_stage = attacker.stat_stages.get(atk_stat, 0)
    if atk_stage != 0:
        atk_num, atk_den = get_stage_multiplier(atk_stage)
        atk = atk * atk_num // atk_den
    dfn_stage = defender.stat_stages.get(dfn_stat, 0)
    if dfn_stage != 0:
        dfn_num, dfn_den = get_stage_multiplier(dfn_stage)
        dfn = dfn * dfn_num // dfn_den

    # burn halves physical attack
    if attacker.status == "brn" and move.damage_class == "physical":
        atk //= 2

    base = ((2 * level // 5 + 2) * move.power * atk // dfn) // 50 + 2

    stab = 1.5 if move.type in attacker.types else 1.0

    # type-boosting items
    item_boost = _ITEM_TYPE_BOOST.get(attacker.item, None)
    if item_boost is not None and move.type == item_boost:
        stab *= 1.1

    # weather modifier
    weather_mult = 1.0
    if weather == "sun":
        if move.type == "fire":
            weather_mult = 1.5
        elif move.type == "water":
            weather_mult = 0.5
    elif weather == "rain":
        if move.type == "water":
            weather_mult = 1.5
        elif move.type == "fire":
            weather_mult = 0.5

    effectiveness = type_chart.combined_effectiveness(move.type, defender.types)

    if effectiveness == 0.0:
        return 0.0

    damage = float(base) * stab * effectiveness * weather_mult * 0.925

    # screens: Reflect halves physical, Light Screen halves special
    if screens is not None:
        if move.damage_class == "physical" and screens.reflect_turns > 0:
            damage *= 0.5
        elif move.damage_class == "special" and screens.light_screen_turns > 0:
            damage *= 0.5

    # factor in accuracy
    if move.accuracy is not None:
        damage *= move.accuracy / 100.0

    return max(damage, 1.0)
