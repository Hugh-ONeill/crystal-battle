# Stat stage mechanics: application, multipliers, and move-to-effect mapping

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .pokemon import Pokemon


# ============================================================
# STAGE MECHANICS
# ============================================================

MIN_STAGE = -6
MAX_STAGE = 6


def get_stage_multiplier(stage: int) -> tuple[int, int]:
    """Return (numerator, denominator) for a stat stage.

    Gen 2: (2 + max(0, stage)) / (2 + max(0, -stage))
    """
    num = 2 + max(0, stage)
    den = 2 + max(0, -stage)
    return num, den


def apply_stat_change(pokemon: Pokemon, stat: str, stages: int) -> int:
    """Apply stat stage change, clamped to [-6, +6]. Returns actual change."""
    current = pokemon.stat_stages[stat]
    new = max(MIN_STAGE, min(MAX_STAGE, current + stages))
    actual = new - current
    pokemon.stat_stages[stat] = new
    return actual


# ============================================================
# MOVE STAT EFFECTS
# ============================================================

# (stat, stages, target) where target is "self" or "opponent"
StatEffect = tuple[str, int, str]

MOVE_STAT_EFFECTS: dict[str, list[StatEffect]] = {
    # ---- Self-boosting status moves ----
    "Swords Dance": [("attack", +2, "self")],
    "Growth": [("special_attack", +1, "self")],
    "Meditate": [("attack", +1, "self")],
    "Sharpen": [("attack", +1, "self")],
    "Agility": [("speed", +2, "self")],
    "Amnesia": [("special_defense", +2, "self")],
    "Barrier": [("defense", +2, "self")],
    "Acid Armor": [("defense", +2, "self")],
    "Double Team": [("evasion", +1, "self")],
    "Minimize": [("evasion", +1, "self")],
    "Harden": [("defense", +1, "self")],
    "Withdraw": [("defense", +1, "self")],
    "Defense Curl": [("defense", +1, "self")],
    # Curse: non-Ghost gets atk+1/def+1/speed-1 (Ghost curse is unimplemented)
    "Curse": [("attack", +1, "self"), ("defense", +1, "self"), ("speed", -1, "self")],
    "Belly Drum": [("attack", +6, "self")],

    # ---- Opponent-lowering status moves ----
    "Growl": [("attack", -1, "opponent")],
    "Leer": [("defense", -1, "opponent")],
    "Tail Whip": [("defense", -1, "opponent")],
    "String Shot": [("speed", -1, "opponent")],
    "Screech": [("defense", -2, "opponent")],
    "Charm": [("attack", -2, "opponent")],
    "Scary Face": [("speed", -2, "opponent")],
    "Sweet Scent": [("evasion", -1, "opponent")],
    "Sand Attack": [("accuracy", -1, "opponent")],
    "Smokescreen": [("accuracy", -1, "opponent")],
    "Flash": [("accuracy", -1, "opponent")],
    "Cotton Spore": [("speed", -2, "opponent")],
    "Kinesis": [("accuracy", -1, "opponent")],

    # ---- Secondary effects on damaging moves (rolled via stat_chance) ----
    "Acid": [("defense", -1, "opponent")],
    "Bubble": [("speed", -1, "opponent")],
    "Bubble Beam": [("speed", -1, "opponent")],
    "Aurora Beam": [("attack", -1, "opponent")],
    "Psychic": [("special_defense", -1, "opponent")],
    "Constrict": [("speed", -1, "opponent")],
    "Mud-Slap": [("accuracy", -1, "opponent")],
    "Octazooka": [("accuracy", -1, "opponent")],
    "Icy Wind": [("speed", -1, "opponent")],
    "Crunch": [("special_defense", -1, "opponent")],
    "Shadow Ball": [("special_defense", -1, "opponent")],
    "Rock Smash": [("defense", -1, "opponent")],
    "Iron Tail": [("defense", -1, "opponent")],
    # self-boosting secondaries
    "Steel Wing": [("defense", +1, "self")],
    "Metal Claw": [("attack", +1, "self")],
    "Ancient Power": [
        ("attack", +1, "self"),
        ("defense", +1, "self"),
        ("special_attack", +1, "self"),
        ("special_defense", +1, "self"),
        ("speed", +1, "self"),
    ],
}
