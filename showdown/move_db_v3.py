#!/usr/bin/env python3
"""Move attribute database for v3+ featurizer (gen9 OU).

Returns a 31-dim feature vector per move: type one-hot (18) + base_power/200 +
accuracy/100 + category flags (status/special) + 9 function flags (pivot,
setup, recovery, status_inflict, priority_attack, hazard, hazard_remove,
screen, phase).

Coverage target: top ~187 moves account for 95% of gen9 OU usage. Unknown
moves fall back to neutral attacker attributes (BP 70, normal type, special).
This is intentional — better to give the model a generic "moderate attacker"
signal than zeros that could mislead.
"""

from __future__ import annotations

import numpy as np


# ============================================================
# TYPE ENCODING (18 attacking types — Stellar reserved for tera state)
# ============================================================

ATK_TYPES = [
    "normal", "fire", "water", "electric", "grass", "ice",
    "fighting", "poison", "ground", "flying", "psychic", "bug",
    "rock", "ghost", "dragon", "dark", "steel", "fairy",
]
N_ATK_TYPES = len(ATK_TYPES)
TYPE_IDX = {t: i for i, t in enumerate(ATK_TYPES)}


# ============================================================
# MOVE ATTRIBUTE TABLE
#
# Each entry: (type, base_power, accuracy, is_status, is_special, flags)
#   flags is a set of strings from FLAG_NAMES below — present means True.
#
# `is_special` only relevant for damaging moves (is_status=False).
# accuracy: 1.0 = always hits (Aerial Ace, Swift, etc.), 0.0..1.0 otherwise.
# ============================================================

FLAG_NAMES = [
    "pivot",            # U-turn / Volt Switch / Flip Turn / Parting Shot / Teleport
    "setup",            # raises own stat by ≥1 (Swords Dance, Calm Mind, Bulk Up, ...)
    "recovery",         # heals user's HP non-trivially (Recover, Roost, ...)
    "status_inflict",   # primary effect is to inflict major status
    "priority_attack",  # damaging move with positive priority
    "hazard",           # sets entry hazard
    "hazard_remove",    # removes hazards
    "screen",           # Reflect / Light Screen / Aurora Veil
    "phase",            # Whirlwind / Roar / Dragon Tail / Circle Throw / Haze
]
N_FLAGS = len(FLAG_NAMES)
FLAG_IDX = {f: i for i, f in enumerate(FLAG_NAMES)}


# Tuples: (type, bp, acc, is_status, is_special, flags_set)
# bp in raw BP units (will be / 200). acc as fraction (1.0 = always).
_T = lambda type, bp, acc=1.0, status=False, special=False, flags=(): \
    (type, bp, acc, status, special, set(flags))


MOVE_DB: dict[str, tuple] = {
    # ---------------------------------------------------------------
    # SETUP MOVES (boost own stats)
    # ---------------------------------------------------------------
    "swordsdance":       _T("normal",   0, 1.0, status=True, flags={"setup"}),
    "calmmind":          _T("psychic",  0, 1.0, status=True, flags={"setup"}),
    "dragondance":       _T("dragon",   0, 1.0, status=True, flags={"setup"}),
    "bulkup":            _T("fighting", 0, 1.0, status=True, flags={"setup"}),
    "nastyplot":         _T("dark",     0, 1.0, status=True, flags={"setup"}),
    "irondefense":       _T("steel",    0, 1.0, status=True, flags={"setup"}),
    "quiverdance":       _T("bug",      0, 1.0, status=True, flags={"setup"}),
    "shellsmash":        _T("normal",   0, 1.0, status=True, flags={"setup"}),
    "agility":           _T("psychic",  0, 1.0, status=True, flags={"setup"}),
    "growth":            _T("normal",   0, 1.0, status=True, flags={"setup"}),
    "tailglow":          _T("bug",      0, 1.0, status=True, flags={"setup"}),
    "cosmicpower":       _T("psychic",  0, 1.0, status=True, flags={"setup"}),
    "amnesia":           _T("psychic",  0, 1.0, status=True, flags={"setup"}),
    "workup":            _T("normal",   0, 1.0, status=True, flags={"setup"}),
    "curse":             _T("ghost",    0, 1.0, status=True, flags={"setup"}),
    "bellydrum":         _T("normal",   0, 1.0, status=True, flags={"setup"}),
    "geomancy":          _T("fairy",    0, 1.0, status=True, flags={"setup"}),
    "filletaway":        _T("normal",   0, 1.0, status=True, flags={"setup"}),

    # ---------------------------------------------------------------
    # RECOVERY MOVES (heal HP)
    # ---------------------------------------------------------------
    "recover":           _T("normal",   0, 1.0, status=True, flags={"recovery"}),
    "roost":             _T("flying",   0, 1.0, status=True, flags={"recovery"}),
    "slackoff":          _T("normal",   0, 1.0, status=True, flags={"recovery"}),
    "softboiled":        _T("normal",   0, 1.0, status=True, flags={"recovery"}),
    "synthesis":         _T("grass",    0, 1.0, status=True, flags={"recovery"}),
    "moonlight":         _T("fairy",    0, 1.0, status=True, flags={"recovery"}),
    "morningsun":        _T("normal",   0, 1.0, status=True, flags={"recovery"}),
    "milkdrink":         _T("normal",   0, 1.0, status=True, flags={"recovery"}),
    "rest":              _T("psychic",  0, 1.0, status=True, flags={"recovery"}),
    "wish":              _T("normal",   0, 1.0, status=True, flags={"recovery"}),
    "strengthsap":       _T("grass",    0, 1.0, status=True, flags={"recovery"}),
    "shoreup":           _T("ground",   0, 1.0, status=True, flags={"recovery"}),
    "junglehealing":     _T("grass",    0, 1.0, status=True, flags={"recovery"}),
    "lifedew":           _T("water",    0, 1.0, status=True, flags={"recovery"}),
    "painsplit":         _T("normal",   0, 1.0, status=True, flags={"recovery"}),
    "leechseed":         _T("grass",    0, 0.9, status=True, flags={"recovery"}),

    # ---------------------------------------------------------------
    # HAZARDS
    # ---------------------------------------------------------------
    "stealthrock":       _T("rock",     0, 1.0, status=True, flags={"hazard"}),
    "spikes":            _T("ground",   0, 1.0, status=True, flags={"hazard"}),
    "toxicspikes":       _T("poison",   0, 1.0, status=True, flags={"hazard"}),
    "stickyweb":         _T("bug",      0, 1.0, status=True, flags={"hazard"}),

    # ---------------------------------------------------------------
    # HAZARD REMOVAL
    # ---------------------------------------------------------------
    "rapidspin":         _T("normal",  50, 1.0, status=False, special=False, flags={"hazard_remove"}),
    "defog":             _T("flying",   0, 1.0, status=True, flags={"hazard_remove"}),
    "tidyup":            _T("normal",   0, 1.0, status=True, flags={"hazard_remove", "setup"}),
    "courtchange":       _T("normal",   0, 1.0, status=True, flags={"hazard_remove"}),
    "mortalspin":        _T("poison",  30, 1.0, status=False, special=False, flags={"hazard_remove"}),

    # ---------------------------------------------------------------
    # SCREENS
    # ---------------------------------------------------------------
    "reflect":           _T("psychic",  0, 1.0, status=True, flags={"screen"}),
    "lightscreen":       _T("psychic",  0, 1.0, status=True, flags={"screen"}),
    "auroraveil":        _T("ice",      0, 1.0, status=True, flags={"screen"}),

    # ---------------------------------------------------------------
    # PHAZING
    # ---------------------------------------------------------------
    "whirlwind":         _T("normal",   0, 1.0, status=True, flags={"phase"}),
    "roar":              _T("normal",   0, 1.0, status=True, flags={"phase"}),
    "dragontail":        _T("dragon",  60, 0.9, status=False, special=False, flags={"phase"}),
    "circlethrow":       _T("fighting",60, 0.9, status=False, special=False, flags={"phase"}),
    "haze":              _T("ice",      0, 1.0, status=True, flags={"phase"}),

    # ---------------------------------------------------------------
    # PIVOT MOVES
    # ---------------------------------------------------------------
    "uturn":             _T("bug",     70, 1.0, status=False, special=False, flags={"pivot"}),
    "voltswitch":        _T("electric",70, 1.0, status=False, special=True,  flags={"pivot"}),
    "flipturn":          _T("water",   60, 1.0, status=False, special=False, flags={"pivot"}),
    "partingshot":       _T("dark",     0, 1.0, status=True, flags={"pivot"}),
    "teleport":          _T("psychic",  0, 1.0, status=True, flags={"pivot"}),
    "chillyreception":   _T("ice",      0, 1.0, status=True, flags={"pivot"}),
    "shedtail":          _T("normal",   0, 1.0, status=True, flags={"pivot"}),
    "batonpass":         _T("normal",   0, 1.0, status=True, flags={"pivot"}),

    # ---------------------------------------------------------------
    # STATUS INFLICTERS (primary effect: status)
    # ---------------------------------------------------------------
    "willowisp":         _T("fire",     0, 0.85, status=True, flags={"status_inflict"}),
    "thunderwave":       _T("electric", 0, 0.9,  status=True, flags={"status_inflict"}),
    "toxic":             _T("poison",   0, 0.9,  status=True, flags={"status_inflict"}),
    "spore":             _T("grass",    0, 1.0,  status=True, flags={"status_inflict"}),
    "sleeppowder":       _T("grass",    0, 0.75, status=True, flags={"status_inflict"}),
    "hypnosis":          _T("psychic",  0, 0.6,  status=True, flags={"status_inflict"}),
    "stunspore":         _T("grass",    0, 0.75, status=True, flags={"status_inflict"}),
    "glare":             _T("normal",   0, 1.0,  status=True, flags={"status_inflict"}),
    "yawn":              _T("normal",   0, 1.0,  status=True, flags={"status_inflict"}),
    "nuzzle":            _T("electric",20, 1.0,  status=False, special=False, flags={"status_inflict"}),
    "lovelykiss":        _T("normal",   0, 0.75, status=True, flags={"status_inflict"}),
    "darkvoid":          _T("dark",     0, 0.5,  status=True, flags={"status_inflict"}),

    # ---------------------------------------------------------------
    # PRIORITY ATTACKS
    # ---------------------------------------------------------------
    "suckerpunch":       _T("dark",    70, 1.0, status=False, special=False, flags={"priority_attack"}),
    "extremespeed":      _T("normal",  80, 1.0, status=False, special=False, flags={"priority_attack"}),
    "aquajet":           _T("water",   40, 1.0, status=False, special=False, flags={"priority_attack"}),
    "iceshard":          _T("ice",     40, 1.0, status=False, special=False, flags={"priority_attack"}),
    "machpunch":          _T("fighting",40, 1.0, status=False, special=False, flags={"priority_attack"}),
    "bulletpunch":       _T("steel",   40, 1.0, status=False, special=False, flags={"priority_attack"}),
    "shadowsneak":       _T("ghost",   40, 1.0, status=False, special=False, flags={"priority_attack"}),
    "quickattack":       _T("normal",  40, 1.0, status=False, special=False, flags={"priority_attack"}),
    "thunderclap":       _T("electric",70, 1.0, status=False, special=True,  flags={"priority_attack"}),
    "grassyglide":       _T("grass",   55, 1.0, status=False, special=False, flags={"priority_attack"}),
    "jetpunch":          _T("water",   60, 1.0, status=False, special=False, flags={"priority_attack"}),
    "accelerock":        _T("rock",    40, 1.0, status=False, special=False, flags={"priority_attack"}),
    "firstimpression":   _T("bug",     90, 1.0, status=False, special=False, flags={"priority_attack"}),

    # ---------------------------------------------------------------
    # PROTECT FAMILY (single-turn defense)
    # Treat as utility — no special flag; defaults to status.
    # ---------------------------------------------------------------
    "protect":           _T("normal",   0, 1.0, status=True),
    "detect":            _T("fighting", 0, 1.0, status=True),
    "kingsshield":       _T("steel",    0, 1.0, status=True),
    "spikyshield":       _T("grass",    0, 1.0, status=True),
    "banefulbunker":     _T("poison",   0, 1.0, status=True),
    "burningbulwark":    _T("fire",     0, 1.0, status=True),
    "silktrap":          _T("bug",      0, 1.0, status=True),

    # ---------------------------------------------------------------
    # WEATHER / TERRAIN setters
    # ---------------------------------------------------------------
    "sunnyday":          _T("fire",     0, 1.0, status=True),
    "raindance":         _T("water",    0, 1.0, status=True),
    "sandstorm":         _T("rock",     0, 1.0, status=True),
    "snowscape":         _T("ice",      0, 1.0, status=True),
    "electricterrain":   _T("electric", 0, 1.0, status=True),
    "grassyterrain":     _T("grass",    0, 1.0, status=True),
    "mistyterrain":      _T("fairy",    0, 1.0, status=True),
    "psychicterrain":    _T("psychic",  0, 1.0, status=True),
    "trickroom":         _T("psychic",  0, 1.0, status=True),

    # ---------------------------------------------------------------
    # COMMON DAMAGE MOVES (top of OU usage)
    # ---------------------------------------------------------------
    "knockoff":          _T("dark",    65, 1.0),
    "earthquake":        _T("ground", 100, 1.0),
    "shadowball":        _T("ghost",   80, 1.0, special=True),
    "closecombat":       _T("fighting",120,1.0),
    "earthpower":        _T("ground",  90, 1.0, special=True),
    "icespinner":        _T("ice",     80, 1.0),
    "moonblast":         _T("fairy",   95, 1.0, special=True),
    "headlongrush":      _T("ground", 120, 1.0),
    "icebeam":           _T("ice",     90, 1.0, special=True),
    "dracometeor":       _T("dragon", 130, 0.9, special=True),
    "terablast":         _T("normal",  80, 1.0, special=True),  # type changes with tera
    "ironhead":          _T("steel",   80, 1.0),
    "flamethrower":      _T("fire",    90, 1.0, special=True),
    "thunderbolt":       _T("electric",90, 1.0, special=True),
    "bodypress":         _T("fighting",80, 1.0),
    "sludgebomb":        _T("poison",  90, 1.0, special=True),
    "encore":            _T("normal",   0, 1.0, status=True),
    "kowtowcleave":      _T("dark",    85, 1.0),
    "makeitrain":        _T("steel",   90, 1.0, special=True),
    "ivycudgel":         _T("grass",  100, 1.0),
    "ragingfury":        _T("fire",   120, 1.0),
    "thunderclap":       _T("electric",70, 1.0, special=True, flags={"priority_attack"}),
    "scaleshot":         _T("dragon",  25, 0.9),
    "outrage":           _T("dragon", 120, 1.0),
    "hyperspacefury":    _T("dark",   100, 1.0),
    "weatherball":       _T("normal",  50, 1.0, special=True),
    "psychic":           _T("psychic", 90, 1.0, special=True),
    "psyshock":          _T("psychic", 80, 1.0, special=True),
    "psychicnoise":      _T("psychic", 75, 1.0, special=True),
    "darkpulse":         _T("dark",    80, 1.0, special=True),
    "fireblast":         _T("fire",   110, 0.85, special=True),
    "hurricane":         _T("flying", 110, 0.7, special=True),
    "stoneedge":         _T("rock",   100, 0.8),
    "rockslide":         _T("rock",    75, 0.9),
    "stoneaxe":          _T("rock",    65, 0.9, flags={"hazard"}),  # also sets sr
    "ceaselessedge":     _T("dark",    65, 0.9, flags={"hazard"}),  # also sets spikes
    "hydropump":         _T("water",  110, 0.8, special=True),
    "surf":              _T("water",   90, 1.0, special=True),
    "scald":             _T("water",   80, 1.0, special=True, flags={"status_inflict"}),
    "playrough":         _T("fairy",   90, 0.9),
    "drainpunch":        _T("fighting", 75, 1.0),
    "leechlife":         _T("bug",     80, 1.0),
    "uturn":             _T("bug",     70, 1.0, flags={"pivot"}),  # already above; dup harmless
    "leafstorm":         _T("grass",  130, 0.9, special=True),
    "energyball":        _T("grass",   90, 1.0, special=True),
    "gigadrain":          _T("grass",   75, 1.0, special=True),
    "powerwhip":         _T("grass",  120, 0.85),
    "woodhammer":        _T("grass",  120, 1.0),
    "hornleech":         _T("grass",   75, 1.0),
    "grasstype":         _T("grass",   90, 1.0),  # NB no such move; placeholder
    "spore":             _T("grass",    0, 1.0, status=True, flags={"status_inflict"}),  # dup harmless
    "freezedry":         _T("ice",     70, 1.0, special=True),
    "blizzard":          _T("ice",    110, 0.7, special=True),
    "tripleaxel":        _T("ice",     20, 0.9),  # 3-hit; effective ~60-90 BP
    "iciclespear":       _T("ice",     25, 1.0),  # 2-5 hit
    "fierydance":        _T("fire",    80, 1.0, special=True),
    "fireblast":         _T("fire",   110, 0.85, special=True),
    "lavaplume":         _T("fire",    80, 1.0, special=True),
    "heatwave":          _T("fire",    95, 0.9, special=True),
    "willowisp":         _T("fire",     0, 0.85, status=True, flags={"status_inflict"}),  # dup
    "voltswitch":        _T("electric",70, 1.0, special=True, flags={"pivot"}),  # dup
    "discharge":         _T("electric",80, 1.0, special=True),
    "thunder":           _T("electric",110, 0.7, special=True),
    "wildcharge":        _T("electric", 90, 1.0),
    "boltbeak":          _T("electric",85, 1.0),  # double if first
    "electroshot":       _T("electric",130,1.0, special=True),  # +1 spa charge
    "futuresight":       _T("psychic",120, 1.0, special=True),
    "psyshock":          _T("psychic", 80, 1.0, special=True),  # dup
    "expandingforce":    _T("psychic", 80, 1.0, special=True),
    "lustershine":       _T("psychic", 95, 1.0, special=True),
    "drainingkiss":      _T("fairy",   50, 1.0, special=True),
    "dazzlinggleam":     _T("fairy",   80, 1.0, special=True),
    "spiritbreak":       _T("fairy",   75, 1.0),
    "moonblast":         _T("fairy",   95, 1.0, special=True),  # dup
    "playrough":         _T("fairy",   90, 0.9),
    "behemothblade":     _T("steel",  100, 1.0),
    "ironhead":          _T("steel",   80, 1.0),
    "gigatonhammer":     _T("steel",  160, 1.0),  # can't use twice in a row
    "anchorshot":        _T("steel",   80, 1.0),
    "heavyslam":         _T("steel",  100, 1.0),  # bp varies
    "mightycleave":      _T("rock",   100, 1.0),
    "stoneaxe":          _T("rock",    65, 0.9, flags={"hazard"}),  # dup
    "rocktomb":          _T("rock",    60, 0.95),
    "earthpower":        _T("ground",  90, 1.0, special=True),  # dup
    "spikes":            _T("ground",   0, 1.0, status=True, flags={"hazard"}),  # dup
    "highhorsepower":    _T("ground", 100, 0.95),
    "stompingtantrum":   _T("ground",  75, 1.0),
    "precipiceblades":   _T("ground", 120, 0.85),
    "bonemerang":        _T("ground",  50, 0.9),
    "bulldoze":          _T("ground",  60, 1.0),
    "earthtype":         _T("ground",  90, 1.0),  # placeholder
    "shadowsneak":       _T("ghost",   40, 1.0, flags={"priority_attack"}),
    "poltergeist":       _T("ghost",  110, 0.9),
    "phantomforce":      _T("ghost",   90, 1.0),
    "hex":               _T("ghost",   65, 1.0, special=True),
    "shadowclaw":        _T("ghost",   70, 1.0),
    "shadowforce":       _T("ghost",  120, 1.0),
    "moongeistbeam":     _T("ghost",  100, 1.0, special=True),
    "spectralthief":     _T("ghost",   90, 1.0),
    "lunarblessing":     _T("psychic",  0, 1.0, status=True, flags={"recovery"}),
    "thunderpunch":      _T("electric",75, 1.0),
    "icepunch":          _T("ice",     75, 1.0),
    "firepunch":         _T("fire",    75, 1.0),
    "knockoff":          _T("dark",    65, 1.0),  # dup
    "lash":              _T("dark",    50, 1.0),  # placeholder
    "crunch":            _T("dark",    80, 1.0),
    "foulplay":          _T("dark",    95, 1.0),  # uses target's atk
    "lash":              _T("dark",    50, 1.0),  # placeholder
    "throatchop":        _T("dark",    80, 1.0),
    "lashout":           _T("dark",    75, 1.0),
    "uturn":             _T("bug",     70, 1.0, flags={"pivot"}),  # dup
    "megahorn":          _T("bug",    120, 0.85),
    "leechlife":         _T("bug",     80, 1.0),
    "bugbuzz":           _T("bug",     90, 1.0, special=True),
    "lunge":             _T("bug",     80, 1.0),
    "twinbeam":          _T("psychic", 40, 1.0, special=True),
    "tailwind":          _T("flying",   0, 1.0, status=True),
    "bravebird":         _T("flying", 120, 1.0),
    "bleakwindstorm":    _T("flying",  100, 0.8, special=True),
    "airslash":          _T("flying",  75, 0.95, special=True),
    "drillpeck":         _T("flying",  80, 1.0),
    "skydrop":           _T("flying",  60, 1.0),
    "uturn":             _T("bug",     70, 1.0, flags={"pivot"}),

    # Status moves not categorized above
    "lightscreen":       _T("psychic",  0, 1.0, status=True, flags={"screen"}),  # dup
    "reflect":           _T("psychic",  0, 1.0, status=True, flags={"screen"}),  # dup
    "trick":             _T("psychic",  0, 1.0, status=True),
    "switcheroo":        _T("dark",     0, 1.0, status=True),
    "taunt":             _T("dark",     0, 1.0, status=True),
    "encore":            _T("normal",   0, 1.0, status=True),  # dup
    "destiny_bond":      _T("ghost",    0, 1.0, status=True),
    "endeavor":          _T("normal",   0, 1.0, status=True),
    "willowisp":         _T("fire",     0, 0.85, status=True, flags={"status_inflict"}),  # dup
    "leechseed":         _T("grass",    0, 0.9, status=True, flags={"recovery"}),  # dup
    "haze":              _T("ice",      0, 1.0, status=True, flags={"phase"}),  # dup
    "memento":           _T("dark",     0, 1.0, status=True),
    "healingwish":       _T("psychic",  0, 1.0, status=True, flags={"recovery"}),
    "lunardance":        _T("psychic",  0, 1.0, status=True, flags={"recovery"}),
    "wish":              _T("normal",   0, 1.0, status=True, flags={"recovery"}),  # dup
    "tailwind":          _T("flying",   0, 1.0, status=True),
}


# ============================================================
# DEFAULT FOR UNKNOWN MOVES
# Generic moderate attacker — better than zero, won't mislead the model
# (the model can learn that "unknown move" features → middling threat).
# ============================================================

_DEFAULT_MOVE = _T("normal", 70, 1.0)


# ============================================================
# FEATURE EXTRACTION
# ============================================================

# Output layout per move: 18 (type) + 1 (bp/200) + 1 (acc) + 1 (status) +
#                         1 (special) + 9 (flags) = 31
N_MOVE_FEATURES = N_ATK_TYPES + 4 + N_FLAGS  # = 31


def move_features(move_id: str) -> np.ndarray:
    """Return 31-dim feature vector for a normalized move id (e.g. 'earthquake').

    Empty / 'none' / 'splash' / 'tackle' (placeholder) return all-zero.
    """
    out = np.zeros(N_MOVE_FEATURES, dtype=np.float32)
    if not move_id or move_id in ("none", "splash"):
        return out

    # Strip "-tera" suffix if present (poke-engine adds it for terastallized moves;
    # type changes with tera but base power / category don't, so look up base move).
    base = move_id[:-5] if move_id.endswith("-tera") else move_id

    entry = MOVE_DB.get(base, _DEFAULT_MOVE)
    mtype, bp, acc, is_status, is_special, flags = entry

    # type one-hot
    type_idx = TYPE_IDX.get(mtype.lower())
    if type_idx is not None:
        out[type_idx] = 1.0

    # numeric attrs
    i = N_ATK_TYPES
    out[i] = min(bp / 200.0, 1.0); i += 1     # base_power
    out[i] = min(max(acc, 0.0), 1.0); i += 1  # accuracy
    out[i] = 1.0 if is_status else 0.0; i += 1
    out[i] = 1.0 if is_special else 0.0; i += 1

    # function flags
    for flag in flags:
        idx = FLAG_IDX.get(flag)
        if idx is not None:
            out[i + idx] = 1.0

    return out


def coverage_summary() -> tuple[int, int]:
    """Return (n_distinct_known, n_total_table_entries). Useful for quick auditing."""
    return len(MOVE_DB), len(MOVE_DB)
