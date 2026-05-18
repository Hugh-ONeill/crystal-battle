#!/usr/bin/env python3
"""BO-locked state featurizer for crystal-battle's gen9 OU value net.

Replaces features_v3.parse_state_v3 for value/policy nets that are trained
exclusively against the Bulky Offense team (idx 1 in SAMPLE_TEAMS_GEN9):
  Garganacl, Darkrai, Great Tusk, Hatterene, Tornadus-Therian, Dragonite.

Design principle: every dim must justify its existence. BO mon identity,
items, abilities, and types are constants on our side — we don't waste
capacity learning to ignore those one-hots. The general v3 featurizer has
2738 dims; this one has STATE_BO_FEATURES (~184), most of which encode
opponent state (the only thing that actually varies).

Output dim is constant regardless of which side (p1/p2) holds the BO team;
the featurizer auto-detects via mon-id signature and orients the encoding.
"""

from __future__ import annotations

import numpy as np


# ============================================================
# BO TEAM CONSTANTS
# ============================================================

BO_MON_LIST = [
    "GARGANACL",
    "DARKRAI",
    "GREATTUSK",
    "HATTERENE",
    "TORNADUSTHERIAN",
    "DRAGONITE",
]
BO_MON_SET = set(BO_MON_LIST)
BO_SLOT_OF = {mid: i for i, mid in enumerate(BO_MON_LIST)}

GARGANACL_SLOT = 0
TUSK_SLOT = 2
HATT_SLOT = 3
DNITE_SLOT = 5


# ============================================================
# OPPONENT ROLE TABLE
# ============================================================
# 10-category role lookup for opp bench encoding. Roles chosen so each
# matters for at least one BO mon's matchup. Unknown mons fall to "OTHER".

OPP_ROLES = [
    "FIRE_STEEL_WALL",     # Heatran, Skarmory, Corviknight, Gholdengo
    "PRIORITY_BREAKER",    # Kingambit, Mamoswine, Rillaboom, Dragonite, Bisharp
    "SCARFER",             # generic scarf revenge — inferred from item, not table
    "WEATHER_SETTER",      # Pelipper, Torkoal, Tyranitar, Ninetales, Excadrill (sand mode)
    "HAZARD_SETTER",       # Ting-Lu, Glimmora, Landorus-T, Garganacl, Skarmory, Iron Treads
    "REMOVER",             # Great Tusk, Iron Treads, Defog users — spinners/defoggers
    "PHYS_SWEEPER",        # Roaring Moon, Iron Moth (mixed), Volcarona, Dragonite
    "SPEC_SWEEPER",        # Walking Wake, Iron Valiant, Gholdengo, Volcarona
    "WALLBREAKER",         # Specs Darkrai, Specs Pecharunt, Hoopa-U, LO Iron Valiant
    "OTHER",
]
N_OPP_ROLES = len(OPP_ROLES)  # 10
ROLE_IDX = {r: i for i, r in enumerate(OPP_ROLES)}

OPP_ROLE_OF = {
    # walls
    "HEATRAN": ROLE_IDX["FIRE_STEEL_WALL"],
    "SKARMORY": ROLE_IDX["FIRE_STEEL_WALL"],
    "CORVIKNIGHT": ROLE_IDX["FIRE_STEEL_WALL"],
    "GHOLDENGO": ROLE_IDX["FIRE_STEEL_WALL"],
    "CLODSIRE": ROLE_IDX["FIRE_STEEL_WALL"],
    "TOXAPEX": ROLE_IDX["FIRE_STEEL_WALL"],
    "BLISSEY": ROLE_IDX["FIRE_STEEL_WALL"],
    "CLEFABLE": ROLE_IDX["FIRE_STEEL_WALL"],
    "DONDOZO": ROLE_IDX["FIRE_STEEL_WALL"],
    "TINGLU": ROLE_IDX["HAZARD_SETTER"],
    "GLIMMORA": ROLE_IDX["HAZARD_SETTER"],
    "LANDORUSTHERIAN": ROLE_IDX["HAZARD_SETTER"],
    "IRONTREADS": ROLE_IDX["REMOVER"],
    # priority / phys threats
    "KINGAMBIT": ROLE_IDX["PRIORITY_BREAKER"],
    "MAMOSWINE": ROLE_IDX["PRIORITY_BREAKER"],
    "RILLABOOM": ROLE_IDX["PRIORITY_BREAKER"],
    "BISHARP": ROLE_IDX["PRIORITY_BREAKER"],
    # weather setters
    "PELIPPER": ROLE_IDX["WEATHER_SETTER"],
    "TORKOAL": ROLE_IDX["WEATHER_SETTER"],
    "TYRANITAR": ROLE_IDX["WEATHER_SETTER"],
    "NINETALES": ROLE_IDX["WEATHER_SETTER"],
    # phys sweepers
    "ROARINGMOON": ROLE_IDX["PHYS_SWEEPER"],
    "DRAGONITE": ROLE_IDX["PHYS_SWEEPER"],
    "VOLCARONA": ROLE_IDX["PHYS_SWEEPER"],
    "BAXCALIBUR": ROLE_IDX["PHYS_SWEEPER"],
    "BARRASKEWDA": ROLE_IDX["PHYS_SWEEPER"],
    "EXCADRILL": ROLE_IDX["PHYS_SWEEPER"],
    "GREATTUSK": ROLE_IDX["PHYS_SWEEPER"],
    # spec sweepers
    "WALKINGWAKE": ROLE_IDX["SPEC_SWEEPER"],
    "IRONVALIANT": ROLE_IDX["SPEC_SWEEPER"],
    "IRONMOTH": ROLE_IDX["SPEC_SWEEPER"],
    "RAGINGBOLT": ROLE_IDX["SPEC_SWEEPER"],
    "MANAPHY": ROLE_IDX["SPEC_SWEEPER"],
    "ARCHALUDON": ROLE_IDX["SPEC_SWEEPER"],
    # wallbreakers
    "DARKRAI": ROLE_IDX["WALLBREAKER"],
    "PECHARUNT": ROLE_IDX["WALLBREAKER"],
    "HOOPAUNBOUND": ROLE_IDX["WALLBREAKER"],
    "KYUREM": ROLE_IDX["WALLBREAKER"],
    "URSALUNA": ROLE_IDX["WALLBREAKER"],
}


# Hand lists for matchup-signal dims. Names match poke-engine id format.

FIRE_STEEL_WALLS = {"HEATRAN", "SKARMORY", "CORVIKNIGHT", "GHOLDENGO"}

# Mons commonly running strong priority that BO struggles with at low HP.
PRIORITY_USERS = {
    "KINGAMBIT", "MAMOSWINE", "RILLABOOM", "DRAGONITE",
    "BISHARP", "RAGINGBOLT",
}

# Common Trick item users: opp may swap their Choice item onto Garg's Leftovers.
TRICK_ITEM_USERS = {
    "GHOLDENGO", "IRONBOULDER", "HOOPAUNBOUND",
    "ALAKAZAM", "GENGAR", "LATIOS", "LATIAS",
}


# ============================================================
# TYPE SYSTEM (reused for opp active type encoding)
# ============================================================

TYPES_18 = [
    "NORMAL", "FIRE", "WATER", "ELECTRIC", "GRASS", "ICE",
    "FIGHTING", "POISON", "GROUND", "FLYING", "PSYCHIC", "BUG",
    "ROCK", "GHOST", "DRAGON", "DARK", "STEEL", "FAIRY",
]
TYPE_IDX = {t: i for i, t in enumerate(TYPES_18)}
N_TYPES = len(TYPES_18)  # 18

DEFENSIVE_STEEL_GHOST = {"STEEL", "GHOST"}


# ============================================================
# ITEM TABLE FOR OPP ACTIVE
# ============================================================

ITEM_FLAGS = [
    "HEAVYDUTYBOOTS",
    "CHOICESCARF",
    "CHOICESPECS",
    "CHOICEBAND",
    "LIFEORB",
    "BOOSTERENERGY",
    "OTHER",
]
N_ITEM_FLAGS = len(ITEM_FLAGS)  # 7
ITEM_IDX = {name: i for i, name in enumerate(ITEM_FLAGS)}


# ============================================================
# GLOBAL FEATURES (mirrored from features_v3 for parity)
# ============================================================

WEATHERS = ["NONE", "SUN", "RAIN", "SAND", "SNOW", "HAIL"]
WEATHER_IDX = {w: i for i, w in enumerate(WEATHERS)}
TERRAINS = ["NONE", "ELECTRIC", "GRASSY", "MISTY", "PSYCHIC"]
TERRAIN_IDX = {t: i for i, t in enumerate(TERRAINS)}

N_GLOBAL = 5 + 4 + 1  # weather (drop NONE) + terrain (drop NONE) + TR


# ============================================================
# PARSING HELPERS
# ============================================================

def _split_mons(side_str: str) -> list[list[str]]:
    """Return up to 6 mon field-lists from a side string. Missing slots
    return empty lists so callers can null-check on len < 28."""
    parts = side_str.split("=")
    mons = []
    for i in range(6):
        s = parts[i] if i < len(parts) else ""
        mons.append(s.split(","))
    return mons


def _hp_frac(fields: list[str]) -> float:
    if len(fields) < 8:
        return 0.0
    try:
        hp = int(fields[6]); maxhp = int(fields[7])
    except ValueError:
        return 0.0
    return hp / max(maxhp, 1)


def _alive(fields: list[str]) -> float:
    if len(fields) < 8:
        return 0.0
    try:
        return 1.0 if int(fields[6]) > 0 else 0.0
    except ValueError:
        return 0.0


def _status_any(fields: list[str]) -> float:
    if len(fields) < 19:
        return 0.0
    return 0.0 if fields[18].upper() == "NONE" else 1.0


def _mon_id(fields: list[str]) -> str:
    return fields[0].upper() if fields and fields[0] else ""


def _item(fields: list[str]) -> str:
    return fields[10].upper() if len(fields) > 10 else ""


def _types(fields: list[str]) -> tuple[str, str]:
    """Current types (post-tera-aware), uppercase."""
    t1 = fields[2].upper() if len(fields) > 2 else ""
    t2 = fields[3].upper() if len(fields) > 3 else ""
    return t1, t2


def _speed(fields: list[str]) -> float:
    if len(fields) < 18:
        return 0.0
    try:
        return float(fields[17])
    except ValueError:
        return 0.0


def _move_disabled_flags(fields: list[str]) -> list[float]:
    """4-dim 0/1 vector for moves 0-3 disabled flag."""
    out = [0.0, 0.0, 0.0, 0.0]
    for i, idx in enumerate((22, 23, 24, 25)):
        if idx < len(fields):
            parts = fields[idx].split(";")
            if len(parts) > 1 and parts[1].lower() == "true":
                out[i] = 1.0
    return out


def _side_parts(side_str: str) -> list[str]:
    return side_str.split("=")


def _active_idx(side_parts: list[str]) -> int:
    if len(side_parts) < 7:
        return 0
    try:
        ai = int(side_parts[6])
        return ai if 0 <= ai < 6 else 0
    except ValueError:
        return 0


def _active_boosts(side_parts: list[str]) -> np.ndarray:
    """5 dims: atk/def/spa/spd/spe boost stages, scaled to [-1, 1] (÷6)."""
    out = np.zeros(5, dtype=np.float32)
    for i in range(5):
        idx = 11 + i
        if idx < len(side_parts):
            try:
                out[i] = max(min(int(side_parts[idx]) / 6.0, 1.0), -1.0)
            except ValueError:
                pass
    return out


def _hazards(side_parts: list[str]) -> tuple[float, float, float]:
    """(stealth_rock, spikes, toxic_spikes) on this side, normalized."""
    if len(side_parts) < 8:
        return 0.0, 0.0, 0.0
    sc = side_parts[7].split(";")
    def _get(idx, denom=1.0):
        try:
            return min(int(sc[idx]) / denom, 1.0) if idx < len(sc) else 0.0
        except ValueError:
            return 0.0
    return _get(13), _get(12, 3.0), _get(17, 2.0)


def _screens(side_parts: list[str]) -> float:
    """1.0 if any of reflect/light_screen/aurora_veil is up."""
    if len(side_parts) < 8:
        return 0.0
    sc = side_parts[7].split(";")
    def _gt0(idx):
        try:
            return int(sc[idx]) > 0 if idx < len(sc) else False
        except ValueError:
            return False
    return 1.0 if (_gt0(10) or _gt0(3) or _gt0(0)) else 0.0


# ============================================================
# SIDE DETECTION
# ============================================================

def _detect_bo_side(state_str: str) -> int:
    """Return 0 if side1 holds the BO team, 1 if side2 does. Decision is
    based on the count of BO mon ids in each side's first 6 slots."""
    parts = state_str.split("/")
    if len(parts) < 2:
        return 0
    def _bo_count(s: str) -> int:
        n = 0
        for fields in _split_mons(s):
            if _mon_id(fields) in BO_MON_SET:
                n += 1
        return n
    s1 = _bo_count(parts[0])
    s2 = _bo_count(parts[1])
    return 0 if s1 >= s2 else 1


# ============================================================
# BO SIDE ENCODING (30 dims)
# ============================================================

N_BO_SIDE = (
    6 * 3       # per BO mon: hp%, status_any, alive
    + 6         # active idx one-hot (in canonical BO order)
    + 5         # active boosts
    + 1         # Tusk Booster-consumed (= Proto online)
)  # = 30


def _encode_bo_side(side_str: str) -> np.ndarray:
    out = np.zeros(N_BO_SIDE, dtype=np.float32)
    parts = _side_parts(side_str)
    mons = _split_mons(side_str)

    # Per-mon block in *canonical BO order* (Garg, Darkrai, Tusk, Hatt, Torn-T, Dnite).
    # Walk slots, map mon_id back to its canonical slot. Slots not matched stay zero
    # (covers teampreview before reveal or unexpected substitutes).
    slot_to_canon = {}
    tusk_canon_alive = False
    tusk_item = ""
    for slot, fields in enumerate(mons):
        mid = _mon_id(fields)
        if mid in BO_SLOT_OF:
            canon = BO_SLOT_OF[mid]
            slot_to_canon[slot] = canon
            base = canon * 3
            out[base + 0] = _hp_frac(fields)
            out[base + 1] = _status_any(fields)
            out[base + 2] = _alive(fields)
            if mid == "GREATTUSK":
                tusk_canon_alive = _alive(fields) > 0.5
                tusk_item = _item(fields)

    # Active idx one-hot (canonical).
    active_slot = _active_idx(parts)
    active_canon = slot_to_canon.get(active_slot)
    if active_canon is not None:
        out[18 + active_canon] = 1.0

    # Active boosts (5).
    out[24:29] = _active_boosts(parts)

    # Tusk Booster-consumed → Protosynthesis online. Tusk must be alive AND
    # have lost its Booster Energy item.
    out[29] = 1.0 if (tusk_canon_alive and tusk_item != "BOOSTERENERGY") else 0.0
    return out


# ============================================================
# OPP SIDE ENCODING (126 dims)
# ============================================================

N_OPP_SIDE = (
    6 * 3                  # per opp mon: hp%, status_any, alive
    + 2 * N_TYPES          # opp active type1+type2 one-hots
    + 5 * N_OPP_ROLES      # opp bench × role one-hot (active gets all-zero in this block)
    + 6                    # opp active idx one-hot
    + 5                    # opp active boosts
    + N_ITEM_FLAGS         # opp active item flags
    + 4                    # opp active move-disabled flags
)  # = 18 + 36 + 50 + 6 + 5 + 7 + 4 = 126


def _encode_opp_side(side_str: str) -> np.ndarray:
    out = np.zeros(N_OPP_SIDE, dtype=np.float32)
    parts = _side_parts(side_str)
    mons = _split_mons(side_str)
    active = _active_idx(parts)

    # 0..17: per-mon hp/status/alive in slot order.
    for slot, fields in enumerate(mons[:6]):
        base = slot * 3
        out[base + 0] = _hp_frac(fields)
        out[base + 1] = _status_any(fields)
        out[base + 2] = _alive(fields)

    # 18..53: opp active type1+type2 (one-hot each).
    if active < len(mons):
        t1, t2 = _types(mons[active])
        if t1 in TYPE_IDX:
            out[18 + TYPE_IDX[t1]] = 1.0
        if t2 in TYPE_IDX:
            out[18 + N_TYPES + TYPE_IDX[t2]] = 1.0

    # 54..103: opp bench role one-hots. 5 slots × 10 roles; active excluded.
    bench_idx = 0
    for slot, fields in enumerate(mons[:6]):
        if slot == active:
            continue
        if bench_idx >= 5:
            break
        mid = _mon_id(fields)
        if not mid:
            bench_idx += 1
            continue
        role = OPP_ROLE_OF.get(mid, ROLE_IDX["OTHER"])
        out[54 + bench_idx * N_OPP_ROLES + role] = 1.0
        bench_idx += 1

    # 104..109: opp active idx one-hot.
    if 0 <= active < 6:
        out[104 + active] = 1.0

    # 110..114: opp active boosts.
    out[110:115] = _active_boosts(parts)

    # 115..121: opp active item flags.
    if active < len(mons):
        item = _item(mons[active])
        out[115 + ITEM_IDX.get(item, ITEM_IDX["OTHER"])] = 1.0

    # 122..125: opp active move-disabled (Choice-lock proxy).
    if active < len(mons):
        for i, v in enumerate(_move_disabled_flags(mons[active])):
            out[122 + i] = v

    return out


# ============================================================
# FIELD + HAZARDS (18 dims)
# ============================================================

N_FIELD = 5 + 4 + 1 + 6 + 2  # weather + terrain + TR + 3 hazards × 2 sides + screens × 2 sides


def _encode_field(weather_str: str, terrain_str: str, trick_room_str: str,
                  bo_side_str: str, opp_side_str: str) -> np.ndarray:
    out = np.zeros(N_FIELD, dtype=np.float32)
    w = weather_str.split(";")[0].upper()
    if w in WEATHER_IDX and w != "NONE":
        out[WEATHER_IDX[w] - 1] = 1.0
    t = terrain_str.split(";")[0].upper()
    if t in TERRAIN_IDX and t != "NONE":
        out[5 + TERRAIN_IDX[t] - 1] = 1.0
    tr = trick_room_str.split(";")[0].lower()
    if tr == "true":
        out[9] = 1.0

    bo_sr, bo_sp, bo_ts = _hazards(_side_parts(bo_side_str))
    opp_sr, opp_sp, opp_ts = _hazards(_side_parts(opp_side_str))
    out[10] = bo_sr; out[11] = bo_sp; out[12] = bo_ts
    out[13] = opp_sr; out[14] = opp_sp; out[15] = opp_ts

    out[16] = _screens(_side_parts(bo_side_str))
    out[17] = _screens(_side_parts(opp_side_str))
    return out


# ============================================================
# BO MATCHUP SIGNALS (10 dims)
# ============================================================

N_MATCHUP = 10


def _encode_matchup(bo_side_str: str, opp_side_str: str) -> np.ndarray:
    out = np.zeros(N_MATCHUP, dtype=np.float32)
    bo_parts = _side_parts(bo_side_str)
    opp_parts = _side_parts(opp_side_str)
    bo_mons = _split_mons(bo_side_str)
    opp_mons = _split_mons(opp_side_str)

    # Find canonical BO slots.
    canon_slot: dict[str, int] = {}
    for slot, fields in enumerate(bo_mons):
        mid = _mon_id(fields)
        if mid in BO_MON_SET:
            canon_slot[mid] = slot

    bo_active = _active_idx(bo_parts)
    opp_active = _active_idx(opp_parts)
    opp_active_fields = opp_mons[opp_active] if opp_active < len(opp_mons) else []

    # 0: Dnite DD level. Only meaningful when Dnite is active. Use atk-stage
    # as the DD proxy (DD adds +1 atk +1 spe per use; atk caps at +6).
    dnite_slot = canon_slot.get("DRAGONITE")
    if dnite_slot is not None and bo_active == dnite_slot:
        try:
            atk_stage = int(bo_parts[11]) if len(bo_parts) > 11 else 0
        except ValueError:
            atk_stage = 0
        out[0] = max(min(atk_stage / 6.0, 1.0), 0.0)

    # 1: Hatt active (Magic Bounce ceiling on incoming hazards/status).
    hatt_slot = canon_slot.get("HATTERENE")
    if hatt_slot is not None and bo_active == hatt_slot:
        out[1] = 1.0

    # 2: Tusk active + Booster live (Proto +Atk online).
    tusk_slot = canon_slot.get("GREATTUSK")
    if tusk_slot is not None and bo_active == tusk_slot:
        item = _item(bo_mons[tusk_slot])
        if item != "BOOSTERENERGY":
            out[2] = 1.0  # consumed → Proto active

    # 3: Garg active + Salt Cure applicable (opp not Steel/Ghost).
    garg_slot = canon_slot.get("GARGANACL")
    if garg_slot is not None and bo_active == garg_slot and opp_active_fields:
        t1, t2 = _types(opp_active_fields)
        if t1 not in DEFENSIVE_STEEL_GHOST and t2 not in DEFENSIVE_STEEL_GHOST:
            out[3] = 1.0

    # 4: Opp active outspeeds Dnite + 1 (rough proxy). Compare raw speed
    # stats; +1 multiplies by 1.5, so threshold is opp_spe > dnite_spe * 1.5.
    if dnite_slot is not None and opp_active_fields:
        dnite_spe = _speed(bo_mons[dnite_slot])
        opp_spe = _speed(opp_active_fields)
        if dnite_spe > 0 and opp_spe > dnite_spe * 1.5:
            out[4] = 1.0

    # 5: Opp has fire-steel wall alive (hand list).
    for fields in opp_mons:
        if _alive(fields) > 0.5 and _mon_id(fields) in FIRE_STEEL_WALLS:
            out[5] = 1.0
            break

    # 6: Opp has strong priority user alive.
    for fields in opp_mons:
        if _alive(fields) > 0.5 and _mon_id(fields) in PRIORITY_USERS:
            out[6] = 1.0
            break

    # 7: Opp has Rocky Helmet user alive (Tusk recoil concern on contact moves).
    for fields in opp_mons:
        if _alive(fields) > 0.5 and _item(fields) == "ROCKYHELMET":
            out[7] = 1.0
            break

    # 8: Opp has Trick-item user alive (Choice-item swap threat to Garg Lefties).
    for fields in opp_mons:
        if _alive(fields) > 0.5 and _mon_id(fields) in TRICK_ITEM_USERS:
            it = _item(fields)
            if it in {"CHOICESCARF", "CHOICESPECS", "CHOICEBAND"}:
                out[8] = 1.0
                break

    # 9: Sum of opp active positive boosts (atk/def/spa/spd/spe), capped at 1.
    pos_sum = 0
    for i in range(5):
        idx = 11 + i
        if idx < len(opp_parts):
            try:
                v = int(opp_parts[idx])
            except ValueError:
                v = 0
            if v > 0:
                pos_sum += v
    out[9] = min(pos_sum / 6.0, 1.0)

    return out


# ============================================================
# TOP-LEVEL ENTRY POINT
# ============================================================

STATE_BO_FEATURES = N_BO_SIDE + N_OPP_SIDE + N_FIELD + N_MATCHUP  # = 184


def parse_state_bo(state_str: str) -> np.ndarray:
    """Convert a poke_engine state string to a fixed-length BO-locked feature vector.

    Returns a 1-D float32 array of length STATE_BO_FEATURES. Side that holds the
    BO team is auto-detected from mon-id signatures; the encoding is always
    oriented BO-first regardless of p1/p2 assignment.
    """
    parts = state_str.split("/")
    if len(parts) < 2:
        return np.zeros(STATE_BO_FEATURES, dtype=np.float32)

    bo_side_idx = _detect_bo_side(state_str)
    bo_side = parts[bo_side_idx]
    opp_side = parts[1 - bo_side_idx]

    weather = parts[2] if len(parts) > 2 else "NONE;0"
    terrain = parts[3] if len(parts) > 3 else "NONE;0"
    trick_room = parts[4] if len(parts) > 4 else "false;0"

    bo = _encode_bo_side(bo_side)
    opp = _encode_opp_side(opp_side)
    field = _encode_field(weather, terrain, trick_room, bo_side, opp_side)
    match = _encode_matchup(bo_side, opp_side)

    return np.concatenate([bo, opp, field, match]).astype(np.float32)
