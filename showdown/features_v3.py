#!/usr/bin/env python3
"""Gen9 OU state featurizer (v3).

Replaces parse_state_v2's gen2-flavored featurization with one that captures
gen9-specific state: 18 types + Stellar tera, terastallization status, gen9-
relevant items (HDB, Booster Energy, Choice, Life Orb, etc.) and abilities
(Protosynthesis/Quark Drive, Regenerator, Multiscale, Magic Guard, etc.).

State string format (from poke_engine State.to_string()):
    side_one/side_two/weather/terrain/trick_room/team_preview

Each side: pokemon0=pokemon1=...=pokemon5=active_idx=side_conditions=...
Each pokemon: 28 comma-separated fields. Order:
    [0] id           [1] level
    [2] type1        [3] type2          (current — post-tera-aware)
    [4] base_type1   [5] base_type2     (pre-tera)
    [6] hp           [7] maxhp
    [8] ability      [9] base_ability
    [10] item        [11] nature
    [12] evs (h;a;d;sa;sd;sp)
    [13-17] stats (atk/def/spa/spd/spe)
    [18] status      [19] sleep_turns   [20] rest_turns   [21] (unused)
    [22-25] moves (id;disabled;pp)
    [26] terastallized (true/false)
    [27] tera_type
"""

from __future__ import annotations

import numpy as np

import poke_engine as pe

# Number of features per move from poke-engine's move feature extractor.
# Must match Rust policy::N_MOVE_FEATURES.
N_MOVE_FEATS = 31


# ============================================================
# TYPE SYSTEM (gen9, 18 types + Stellar + Typeless sentinel)
# ============================================================

TYPES_V3 = [
    "NORMAL", "FIRE", "WATER", "ELECTRIC", "GRASS", "ICE",
    "FIGHTING", "POISON", "GROUND", "FLYING", "PSYCHIC", "BUG",
    "ROCK", "GHOST", "DRAGON", "DARK", "STEEL", "FAIRY",
    "STELLAR", "TYPELESS",
]
TYPE_IDX_V3 = {t: i for i, t in enumerate(TYPES_V3)}
N_TYPES_V3 = len(TYPES_V3)  # 20


def _type_onehot(t: str) -> np.ndarray:
    v = np.zeros(N_TYPES_V3, dtype=np.float32)
    idx = TYPE_IDX_V3.get(t.upper())
    if idx is not None:
        v[idx] = 1.0
    return v


# ============================================================
# STATUS / VOLATILE
# ============================================================

STATUSES = ["NONE", "BURN", "SLEEP", "FREEZE", "PARALYZE", "POISON", "TOXIC"]
STATUS_IDX = {s: i for i, s in enumerate(STATUSES)}


# ============================================================
# ABILITY CATEGORIES (function-based encoding — value-relevant only)
# ============================================================

# damage reducers when at full HP / multipliers
_ABILITY_MULTISCALE = {"multiscale", "shadowshield"}
# active stat-boost from sun / electric terrain / Booster Energy
_ABILITY_BOOSTABLE = {"protosynthesis", "quarkdrive"}
# heals % HP on switch
_ABILITY_REGENERATOR = {"regenerator"}
# ignores hazards / indirect damage
_ABILITY_HAZARD_IMMUNE = {"magicguard", "levitate"}
# lowers attacker's atk on switch-in
_ABILITY_INTIMIDATE = {"intimidate"}
# ignores stat boosts on opponent
_ABILITY_UNAWARE = {"unaware"}
# reflects status moves / ignores them
_ABILITY_STATUS_IMMUNE = {"magicbounce", "goodasgold", "purifyingsalt"}
# kingambit-style snowball
_ABILITY_SUPREME = {"supremeoverlord"}


def _ability_flags(ability: str) -> np.ndarray:
    a = ability.lower().replace(" ", "").replace("-", "").replace("_", "")
    return np.array([
        1.0 if a in _ABILITY_MULTISCALE else 0.0,
        1.0 if a in _ABILITY_BOOSTABLE else 0.0,
        1.0 if a in _ABILITY_REGENERATOR else 0.0,
        1.0 if a in _ABILITY_HAZARD_IMMUNE else 0.0,
        1.0 if a in _ABILITY_INTIMIDATE else 0.0,
        1.0 if a in _ABILITY_UNAWARE else 0.0,
        1.0 if a in _ABILITY_STATUS_IMMUNE else 0.0,
        1.0 if a in _ABILITY_SUPREME else 0.0,
    ], dtype=np.float32)


N_ABILITY_FLAGS = 8


# ============================================================
# ITEM CATEGORIES (function-based encoding)
# ============================================================

_ITEM_HDB = {"heavydutyboots"}
_ITEM_CHOICE = {"choiceband", "choicespecs", "choicescarf"}
_ITEM_LIFEORB = {"lifeorb"}
_ITEM_FOCUSSASH = {"focussash"}
_ITEM_AIRBALLOON = {"airballoon"}
_ITEM_RESTORE = {"leftovers", "blacksludge", "sitrusberry"}
_ITEM_ROCKYHELMET = {"rockyhelmet"}
_ITEM_BOOSTERENERGY = {"boosterenergy"}
# weather rocks (extends weather duration; modest value impact)
_ITEM_WEATHERROCK = {"damprock", "smoothrock", "heatrock", "icyrock"}
# orbs that self-status
_ITEM_BADORB = {"toxicorb", "flameorb"}


def _item_flags(item: str) -> np.ndarray:
    i = item.lower().replace(" ", "").replace("-", "").replace("_", "")
    return np.array([
        1.0 if i in _ITEM_HDB else 0.0,
        1.0 if i in _ITEM_CHOICE else 0.0,
        1.0 if i == "choicescarf" else 0.0,  # extra bit: scarf gives speed
        1.0 if i in _ITEM_LIFEORB else 0.0,
        1.0 if i in _ITEM_FOCUSSASH else 0.0,
        1.0 if i in _ITEM_AIRBALLOON else 0.0,
        1.0 if i in _ITEM_RESTORE else 0.0,
        1.0 if i in _ITEM_ROCKYHELMET else 0.0,
        1.0 if i in _ITEM_BOOSTERENERGY else 0.0,
        1.0 if i in _ITEM_WEATHERROCK else 0.0,
        1.0 if i in _ITEM_BADORB else 0.0,
    ], dtype=np.float32)


N_ITEM_FLAGS = 11


# ============================================================
# POKEMON FEATURES
# ============================================================

# layout per pokemon (indices accumulated below):
#   hp_frac: 1
#   alive: 1
#   types (current, multi-hot 20): 20
#   base types (pre-tera, multi-hot 20): 20
#   tera_type (one-hot 20): 20
#   terastallized: 1
#   stats (atk/def/spa/spd/spe normalized): 5
#   status (one-hot 7): 7
#   sleep_turns / 4: 1
#   ability flags: 8
#   item flags: 11
#   move PP fractions (4): 4
#   move features (4 × N_MOVE_FEATS): 4*31
POKEMON_V3_FEATURES = (
    1 + 1 + N_TYPES_V3 + N_TYPES_V3 + N_TYPES_V3 + 1 + 5 + len(STATUSES) + 1
    + N_ABILITY_FLAGS + N_ITEM_FLAGS + 4 + 4 * N_MOVE_FEATS
)


_ZERO_MOVE_FEATS = np.zeros(N_MOVE_FEATS, dtype=np.float32)
_MOVE_FEAT_CACHE: dict[str, np.ndarray] = {"": _ZERO_MOVE_FEATS}


def _move_feats(move_id: str) -> np.ndarray:
    cached = _MOVE_FEAT_CACHE.get(move_id)
    if cached is not None:
        return cached
    try:
        v = np.asarray(pe.move_features_v3(move_id), dtype=np.float32)
        if v.shape[0] != N_MOVE_FEATS:
            v = _ZERO_MOVE_FEATS
    except Exception:
        v = _ZERO_MOVE_FEATS
    _MOVE_FEAT_CACHE[move_id] = v
    return v


def _parse_pokemon(fields: list[str]) -> np.ndarray:
    feats = []

    # HP
    try:
        hp = int(fields[6]); maxhp = int(fields[7])
    except (ValueError, IndexError):
        hp, maxhp = 0, 1
    feats.append(hp / max(maxhp, 1))
    feats.append(1.0 if hp > 0 else 0.0)

    # current types (multi-hot)
    cur = np.zeros(N_TYPES_V3, dtype=np.float32)
    for j in (2, 3):
        idx = TYPE_IDX_V3.get(fields[j].upper()) if j < len(fields) else None
        if idx is not None:
            cur[idx] = 1.0
    feats.append(cur)

    # base types (pre-tera, multi-hot)
    base = np.zeros(N_TYPES_V3, dtype=np.float32)
    for j in (4, 5):
        idx = TYPE_IDX_V3.get(fields[j].upper()) if j < len(fields) else None
        if idx is not None:
            base[idx] = 1.0
    feats.append(base)

    # terastallized + tera_type
    terad = (len(fields) > 26 and fields[26].lower() == "true")
    tera = np.zeros(N_TYPES_V3, dtype=np.float32)
    if len(fields) > 27:
        idx = TYPE_IDX_V3.get(fields[27].upper())
        if idx is not None:
            tera[idx] = 1.0
    feats.append(tera)
    feats.append(1.0 if terad else 0.0)

    # stats (atk/def/spa/spd/spe), normalized by 500
    for s_idx in range(13, 18):
        try:
            feats.append(int(fields[s_idx]) / 500.0)
        except (ValueError, IndexError):
            feats.append(0.0)

    # status one-hot
    status = np.zeros(len(STATUSES), dtype=np.float32)
    s_field = fields[18].upper() if len(fields) > 18 else "NONE"
    if s_field in STATUS_IDX:
        status[STATUS_IDX[s_field]] = 1.0
    else:
        status[0] = 1.0
    feats.append(status)

    # sleep_turns / 4 (capped)
    try:
        st = int(fields[19]) if len(fields) > 19 else 0
    except ValueError:
        st = 0
    feats.append(min(st / 4.0, 1.0))

    # ability flags
    ab = fields[8] if len(fields) > 8 else ""
    feats.append(_ability_flags(ab))

    # item flags
    it = fields[10] if len(fields) > 10 else ""
    feats.append(_item_flags(it))

    # move PP (4 moves) + move identity features (per slot)
    move_ids: list[str] = []
    for k in range(4):
        m_idx = 22 + k
        mid = ""
        if m_idx < len(fields):
            parts = fields[m_idx].split(";")
            mid = parts[0] if parts else ""
            try:
                pp = int(parts[2]) if len(parts) >= 3 else 0
                feats.append(min(pp / 32.0, 1.0))
            except ValueError:
                feats.append(0.0)
        else:
            feats.append(0.0)
        move_ids.append(mid)
    for mid in move_ids:
        feats.append(_move_feats(mid))

    # flatten into 1-D float32
    out = []
    for x in feats:
        if isinstance(x, np.ndarray):
            out.extend(x.tolist())
        else:
            out.append(float(x))
    return np.array(out, dtype=np.float32)


# ============================================================
# SIDE FEATURES
# ============================================================

# active_idx (one-hot 6) + boosts (atk/def/spa/spd/spe/acc/eva, 7)
# + hazards (spikes/sr/web/tspikes/reflect/lscreen/auroraveil/tailwind/safeguard, 9)
# + force flags (force_switch, force_trapped, slow_uturn_move, baton_passing, 4)
SIDE_V3_EXTRAS = 6 + 7 + 9 + 4


def _parse_side(side_str: str) -> np.ndarray:
    parts = side_str.split("=")
    pokemon_strs = parts[:6]
    while len(pokemon_strs) < 6:
        pokemon_strs.append("")

    feats = []
    for pstr in pokemon_strs:
        flds = pstr.split(",")
        if len(flds) >= 28:
            feats.append(_parse_pokemon(flds))
        else:
            feats.append(np.zeros(POKEMON_V3_FEATURES, dtype=np.float32))

    extras = np.zeros(SIDE_V3_EXTRAS, dtype=np.float32)

    # active_idx one-hot
    try:
        ai = int(parts[6]) if len(parts) > 6 else 0
        if 0 <= ai < 6:
            extras[ai] = 1.0
    except ValueError:
        pass

    # boosts: parts[11..18] = atk, def, spa, spd, spe, acc, eva (7 fields)
    # parts[10] is substitute_health; boosts start at parts[11].
    for i in range(7):
        idx = 11 + i
        try:
            extras[6 + i] = max(min(int(parts[idx]) / 6.0, 1.0), -1.0) if idx < len(parts) else 0.0
        except ValueError:
            pass

    # side conditions: parts[7] = ";"-separated 19 fields (alphabetical)
    # We pull: spikes(12), stealth_rock(13), sticky_web(14), toxic_spikes(17),
    # reflect(10), light_screen(3), aurora_veil(0), tailwind(15), safeguard(11)
    sc = parts[7].split(";") if len(parts) > 7 else []
    def _sc(idx, scale=1.0):
        try:
            return min(int(sc[idx]) / scale, 1.0) if idx < len(sc) else 0.0
        except ValueError:
            return 0.0
    extras[13] = _sc(12, 3.0)            # spikes (0-3)
    extras[14] = _sc(13, 1.0)            # stealth_rock (0-1)
    extras[15] = _sc(14, 1.0)            # sticky_web
    extras[16] = _sc(17, 2.0)            # toxic_spikes (0-2)
    extras[17] = float(_sc(10) > 0)      # reflect active
    extras[18] = float(_sc(3) > 0)       # light_screen active
    extras[19] = float(_sc(0) > 0)       # aurora_veil active
    extras[20] = float(_sc(15) > 0)      # tailwind
    extras[21] = float(_sc(11) > 0)      # safeguard

    # force flags. State string layout (see Side::serialize):
    #   parts[22] force_switch, parts[24] baton_passing,
    #   parts[26] force_trapped, parts[28] slow_uturn_move
    def _b(idx):
        return 1.0 if (len(parts) > idx and parts[idx].lower() == "true") else 0.0
    extras[22] = _b(22)  # force_switch
    extras[23] = _b(26)  # force_trapped
    extras[24] = _b(28)  # slow_uturn_move
    extras[25] = _b(24)  # baton_passing

    return np.concatenate([f for f in feats] + [extras]).astype(np.float32)


SIDE_V3_FEATURES = 6 * POKEMON_V3_FEATURES + SIDE_V3_EXTRAS


# ============================================================
# TOP-LEVEL FEATURES
# ============================================================

WEATHERS = ["NONE", "SUN", "RAIN", "SAND", "SNOW", "HAIL"]
WEATHER_IDX = {w: i for i, w in enumerate(WEATHERS)}
TERRAINS = ["NONE", "ELECTRIC", "GRASSY", "MISTY", "PSYCHIC"]
TERRAIN_IDX = {t: i for i, t in enumerate(TERRAINS)}

# weather (5 active types, drop NONE) + terrain (4 active, drop NONE) + trick_room
N_GLOBAL = 5 + 4 + 1


def _parse_global(weather_str: str, terrain_str: str, trick_room_str: str) -> np.ndarray:
    out = np.zeros(N_GLOBAL, dtype=np.float32)
    w = weather_str.split(";")[0].upper()
    if w in WEATHER_IDX and w != "NONE":
        out[WEATHER_IDX[w] - 1] = 1.0  # -1 because NONE is dropped
    t = terrain_str.split(";")[0].upper()
    if t in TERRAIN_IDX and t != "NONE":
        out[5 + TERRAIN_IDX[t] - 1] = 1.0
    tr = trick_room_str.split(";")[0].lower()
    if tr == "true":
        out[-1] = 1.0
    return out


# ============================================================
# TOP-LEVEL ENTRY POINT
# ============================================================

STATE_V3_FEATURES = 2 * SIDE_V3_FEATURES + N_GLOBAL


def parse_state_v3(state_str: str) -> np.ndarray:
    """Convert a poke_engine state string to a fixed-length feature vector.

    Returns a 1-D float32 array of length STATE_V3_FEATURES.
    """
    parts = state_str.split("/")
    if len(parts) < 2:
        return np.zeros(STATE_V3_FEATURES, dtype=np.float32)
    s1 = _parse_side(parts[0])
    s2 = _parse_side(parts[1])
    weather = parts[2] if len(parts) > 2 else "NONE;0"
    terrain = parts[3] if len(parts) > 3 else "NONE;0"
    trick_room = parts[4] if len(parts) > 4 else "false;0"
    g = _parse_global(weather, terrain, trick_room)
    return np.concatenate([s1, s2, g]).astype(np.float32)
