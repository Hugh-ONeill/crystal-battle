# Status condition logic: application, prevention, end-of-turn effects

from __future__ import annotations

import random as _random
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .pokemon import Pokemon


# ============================================================
# STATUS CONSTANTS
# ============================================================

BRN = "brn"
PAR = "par"
SLP = "slp"
FRZ = "frz"
PSN = "psn"
TOX = "tox"

NON_VOLATILE = {BRN, PAR, SLP, FRZ, PSN, TOX}

# ailment_id mapping from move data
AILMENT_MAP = {
    1: PAR,
    2: SLP,
    3: FRZ,
    4: BRN,
    5: PSN,  # Toxic overrides to TOX by move name
}

# type immunities: type -> set of statuses it's immune to
TYPE_IMMUNITIES: dict[str, set[str]] = {
    "fire": {BRN},
    "electric": {PAR},
    "poison": {PSN, TOX},
    "steel": {PSN, TOX},
    "ice": {FRZ},
}


# ============================================================
# APPLICATION
# ============================================================

def can_apply_status(pokemon: Pokemon, status: str) -> tuple[bool, str]:
    """Check if a status can be applied. Returns (can_apply, reason)."""
    # already has a non-volatile status
    if pokemon.status is not None:
        return False, "already has a status"

    if pokemon.is_fainted:
        return False, "fainted"

    # type immunities
    for ptype in pokemon.types:
        if ptype in TYPE_IMMUNITIES and status in TYPE_IMMUNITIES[ptype]:
            return False, f"{ptype} type is immune"

    return True, ""


def apply_status(
    pokemon: Pokemon,
    status: str,
    rng: _random.Random,
) -> bool:
    """Apply a non-volatile status. Returns True if applied."""
    can, _ = can_apply_status(pokemon, status)
    if not can:
        return False

    pokemon.status = status

    if status == SLP:
        # gen 2: sleep lasts 1-7 turns
        pokemon.status_turns = rng.randint(1, 7)
    elif status == TOX:
        # toxic counter starts at 1 (1/16 first turn)
        pokemon.status_turns = 0
    else:
        pokemon.status_turns = 0

    return True


def apply_confusion(pokemon: Pokemon, rng: _random.Random) -> bool:
    """Apply confusion. Returns True if applied."""
    if pokemon.confusion_turns > 0 or pokemon.is_fainted:
        return False
    # gen 2: confusion lasts 2-5 turns
    pokemon.confusion_turns = rng.randint(2, 5)
    return True


# ============================================================
# MOVE PREVENTION
# ============================================================

def check_move_prevention(
    pokemon: Pokemon,
    rng: _random.Random,
) -> tuple[bool, str | None]:
    """
    Check if a pokemon can act this turn due to status.
    Returns (can_act, reason_string_or_None).
    """
    status = pokemon.status

    if status == SLP:
        if pokemon.status_turns > 0:
            pokemon.status_turns -= 1
            if pokemon.status_turns == 0:
                pokemon.clear_status()
                return True, "woke up"
            return False, "fast asleep"
        # counter hit 0 -- wake up
        pokemon.clear_status()
        return True, "woke up"

    if status == FRZ:
        # 20% chance to thaw each turn
        if rng.random() < 0.20:
            pokemon.clear_status()
            return True, "thawed out"
        return False, "frozen solid"

    if status == PAR:
        # 25% chance to be fully paralyzed
        if rng.random() < 0.25:
            return False, "fully paralyzed"

    return True, None


def check_confusion(
    pokemon: Pokemon,
    rng: _random.Random,
) -> tuple[bool, int]:
    """
    Check confusion before attacking. Returns (hit_self, self_damage).
    Decrements confusion counter. If hit_self is True, the pokemon
    hits itself for self_damage instead of using its move.
    """
    if pokemon.confusion_turns <= 0:
        return False, 0

    pokemon.confusion_turns -= 1

    # 50% chance to hit self
    if rng.random() < 0.5:
        # 40 power typeless physical hit against self
        level = 100
        atk = pokemon.stats["attack"]
        dfn = pokemon.stats["defense"]
        # simplified damage formula: no stab, no type, no crit, no random
        damage = ((2 * level // 5 + 2) * 40 * atk // dfn) // 50 + 2
        return True, damage

    return False, 0


# ============================================================
# END-OF-TURN EFFECTS
# ============================================================

def end_of_turn_damage(pokemon: Pokemon) -> int:
    """Calculate end-of-turn residual damage. Returns damage amount (0 if none)."""
    status = pokemon.status

    if status == BRN:
        return pokemon.max_hp // 8

    if status == PSN:
        return pokemon.max_hp // 8

    if status == TOX:
        pokemon.status_turns += 1
        return (pokemon.max_hp * pokemon.status_turns) // 16

    return 0


# ============================================================
# STATUS FROM MOVE DATA
# ============================================================

def status_from_move(move_name: str, ailment_id: int) -> str | None:
    """Determine which status a move applies. Returns None if out of scope."""
    if ailment_id not in AILMENT_MAP:
        return None

    # Toxic (move id 92) applies toxic poison, not regular poison
    if ailment_id == 5 and move_name == "Toxic":
        return TOX

    return AILMENT_MAP[ailment_id]


def confusion_from_move(ailment_id: int) -> bool:
    """Check if a move applies confusion."""
    return ailment_id == 6


def effective_speed(pokemon: Pokemon) -> int:
    """Get effective speed accounting for stat stages and paralysis."""
    from .stat_stages import get_stage_multiplier
    speed = pokemon.stats["speed"]
    stage = pokemon.stat_stages.get("speed", 0)
    if stage != 0:
        num, den = get_stage_multiplier(stage)
        speed = speed * num // den
    if pokemon.status == PAR:
        speed //= 4
    return speed
