"""
Team-preview featurizer for the lead-pick learning net.

Takes a team paste body (6-mon Showdown paste) and builds a tensor that
encodes each mon's defining lead-pick properties:
  - dual typing (one-hot)
  - role flags: hazard setter, weather setter (by move or ability), screen
    setter, terrain setter, setup user, recovery user, priority user, pivot
    user, status spreader
  - derived stat block (hp/atk/def/spa/spd/spe) normalized

The featurizer goes through poke-engine's `build_pe_state_gen9` so the
stat normalization respects EVs/nature, not just base. This means the same
species with different sets (Bulky Sub Heatran vs Specs Heatran) gets
distinct features — which matters for lead-pick decisions.

Output shape: (6, MON_DIM) per team, where MON_DIM = 51 (18+18+9+6).
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from showdown.local_battle import build_pe_state_gen9


# Canonical Gen 9 type ordering (18 types; "typeless" handled as all-zero).
TYPE_ORDER = (
    "normal", "fire", "water", "electric", "grass", "ice", "fighting",
    "poison", "ground", "flying", "psychic", "bug", "rock", "ghost",
    "dragon", "dark", "steel", "fairy",
)
TYPE_INDEX = {t: i for i, t in enumerate(TYPE_ORDER)}
N_TYPES = len(TYPE_ORDER)

# Per-mon role-flag move/ability sets. All keys are normalized: lower-case,
# no spaces, hyphens, or apostrophes.
HAZARD_MOVES = frozenset({"stealthrock", "spikes", "toxicspikes", "stickyweb"})
SCREEN_MOVES = frozenset({"reflect", "lightscreen", "auroraveil"})
TERRAIN_MOVES = frozenset({
    "electricterrain", "grassyterrain", "psychicterrain", "mistyterrain",
})
WEATHER_MOVES = frozenset({
    "sunnyday", "raindance", "snowscape", "snowyday", "sandstorm",
    "chillyreception",
})
WEATHER_ABILITIES = frozenset({
    "drought", "drizzle", "snowwarning", "sandstream",
    "orichalcumpulse", "hadronengine", "primordialsea", "desolateland",
})
SETUP_MOVES = frozenset({
    "swordsdance", "nastyplot", "calmmind", "bulkup", "irondefense",
    "amnesia", "agility", "dragondance", "quiverdance", "shellsmash",
    "noretreat", "geomancy", "rockpolish", "shiftgear", "tailglow",
    "victorydance", "cosmicpower", "stockpile", "coil", "bellydrum",
    "curse", "workup", "howl", "meditate", "growth",
})
RECOVERY_MOVES = frozenset({
    "recover", "roost", "softboiled", "slackoff", "milkdrink", "shoreup",
    "moonlight", "morningsun", "synthesis", "lifedew", "wish", "painsplit",
    "strengthsap", "rest",
})
PRIORITY_MOVES = frozenset({
    "aquajet", "bulletpunch", "shadowsneak", "machpunch", "iceshard",
    "quickattack", "extremespeed", "suckerpunch", "fakeout", "watershuriken",
    "feint", "vacuumwave", "thunderclap", "jetpunch", "firstimpression",
    "accelerock", "grassywhistle", "tailwind",  # tailwind is team priority
})
PIVOT_MOVES = frozenset({
    "uturn", "voltswitch", "flipturn", "partingshot", "chillyreception",
    "teleport", "batonpass",
})
STATUS_MOVES = frozenset({
    "willowisp", "thunderwave", "toxic", "spore", "sleeppowder", "yawn",
    "stunspore", "glare", "hypnosis", "darkvoid", "nuzzle", "scaleshot",
    "bodyslam",  # potential para chance but mostly attack — drop?
})

# Stat normalization caps. Typical max derived stat at level 100 with EVs is
# ~400-500 for offensive mons, ~700 for HP. These caps put most mons in [0,1].
STAT_NORM = {
    "hp": 700.0,
    "attack": 500.0,
    "defense": 500.0,
    "special_attack": 500.0,
    "special_defense": 500.0,
    "speed": 500.0,
}

# Layout (each mon block):
#   [0:18]   type1 one-hot
#   [18:36]  type2 one-hot (zeros if mono-typed)
#   [36:45]  role flags (9)
#   [45:51]  normalized stat block (6)
MON_DIM = 18 + 18 + 9 + 6


def featurize_mon(pkmn) -> np.ndarray:
    """Encode one poke-engine Pokemon object into a (MON_DIM,) feature vector."""
    out = np.zeros(MON_DIM, dtype=np.float32)

    # Typing — pkmn.types is a tuple of lowercase strings, possibly ("psychic","fairy")
    # or ("steel","typeless") for mono-typed.
    t1, t2 = (pkmn.types + ("typeless", "typeless"))[:2]
    if t1 in TYPE_INDEX:
        out[TYPE_INDEX[t1]] = 1.0
    if t2 in TYPE_INDEX and t2 != t1:
        out[N_TYPES + TYPE_INDEX[t2]] = 1.0

    # Role flags from moveset + ability.
    move_ids = {(m.id or "").lower().replace(" ", "").replace("-", "") for m in pkmn.moves}
    ability = (pkmn.ability or "").lower().replace(" ", "").replace("-", "")

    has_hazard = bool(move_ids & HAZARD_MOVES)
    has_screen = bool(move_ids & SCREEN_MOVES)
    has_terrain = bool(move_ids & TERRAIN_MOVES)
    has_weather = (ability in WEATHER_ABILITIES) or bool(move_ids & WEATHER_MOVES)
    has_setup = bool(move_ids & SETUP_MOVES)
    has_recovery = bool(move_ids & RECOVERY_MOVES)
    has_priority = bool(move_ids & PRIORITY_MOVES)
    has_pivot = bool(move_ids & PIVOT_MOVES)
    has_status = bool(move_ids & STATUS_MOVES)

    out[36] = float(has_hazard)
    out[37] = float(has_weather)
    out[38] = float(has_screen)
    out[39] = float(has_terrain)
    out[40] = float(has_setup)
    out[41] = float(has_recovery)
    out[42] = float(has_priority)
    out[43] = float(has_pivot)
    out[44] = float(has_status)

    # Stat block (derived stats — reflect EVs/nature).
    out[45] = min(pkmn.maxhp / STAT_NORM["hp"], 1.0)
    out[46] = min(pkmn.attack / STAT_NORM["attack"], 1.0)
    out[47] = min(pkmn.defense / STAT_NORM["defense"], 1.0)
    out[48] = min(pkmn.special_attack / STAT_NORM["special_attack"], 1.0)
    out[49] = min(pkmn.special_defense / STAT_NORM["special_defense"], 1.0)
    out[50] = min(pkmn.speed / STAT_NORM["speed"], 1.0)

    return out


def featurize_team_from_state_side(side) -> np.ndarray:
    """Build a (6, MON_DIM) tensor for a side of a built engine state."""
    feats = np.zeros((6, MON_DIM), dtype=np.float32)
    for i, p in enumerate(side.pokemon[:6]):
        feats[i] = featurize_mon(p)
    return feats


def featurize_preview(team1_body: str, team2_body: str) -> tuple[np.ndarray, np.ndarray]:
    """Build (6, MON_DIM) tensors for both teams from paste bodies.

    The team order in the output matches paste-block order, so the lead-pick
    head can output a 6-way softmax indexed by paste slot.
    """
    state = build_pe_state_gen9(team1_body, team2_body)
    return (
        featurize_team_from_state_side(state.side_one),
        featurize_team_from_state_side(state.side_two),
    )


def role_summary(team_feats: np.ndarray) -> dict[str, int]:
    """Diagnostic: count which role flags fire across the 6 mons."""
    role_names = ["hazard", "weather", "screen", "terrain", "setup",
                  "recovery", "priority", "pivot", "status"]
    return {role_names[i]: int(team_feats[:, 36 + i].sum()) for i in range(9)}
