"""Feature blocks for crystal-battle featurizers.

Blocks are grouped:
  - COMMON blocks (reusable across teams): own/opp side mons, field globals,
    hazards+screens, opponent encoding with a pluggable role table.
  - BO TEAM blocks (specific to the Bulky-Offense narrow-eval featurizer):
    canonical-slot reorder + matchup signals.

Add a new block by writing a class with `name`, `dim`, and an `extract`
method, and decorate it with `@register("...")`. The class is then available
via `feature_block.build(...)`. Per-team featurizers compose these into a
`Featurizer` (see `featurizer_bo.py` for the canonical example).
"""

from __future__ import annotations

import numpy as np

import poke_engine as pe

from showdown.feature_block import register
from showdown.features_core import (
    BattleState,
    _active_boosts,
    _alive,
    _hazards,
    _hp_frac,
    _item,
    _mon_id,
    _move_disabled_flags,
    _screens,
    _speed,
    _status_any,
    _types,
)


# ============================================================
# SHARED VOCABULARIES
# ============================================================

TYPES_18 = [
    "NORMAL", "FIRE", "WATER", "ELECTRIC", "GRASS", "ICE",
    "FIGHTING", "POISON", "GROUND", "FLYING", "PSYCHIC", "BUG",
    "ROCK", "GHOST", "DRAGON", "DARK", "STEEL", "FAIRY",
]
TYPE_IDX = {t: i for i, t in enumerate(TYPES_18)}
N_TYPES = len(TYPES_18)

WEATHERS = ["NONE", "SUN", "RAIN", "SAND", "SNOW", "HAIL"]
WEATHER_IDX = {w: i for i, w in enumerate(WEATHERS)}

TERRAINS = ["NONE", "ELECTRIC", "GRASSY", "MISTY", "PSYCHIC"]
TERRAIN_IDX = {t: i for i, t in enumerate(TERRAINS)}


# ============================================================
# COMMON BLOCKS
# ============================================================

@register("own_side_compact")
class OwnSideCompact:
    """Per-slot hp%/status_any/alive for the own side, in slot order (no
    canonical reorder). 18 dims = 6 slots × 3. Use this for teams where the
    six-slot ordering carries no semantic meaning beyond what the engine
    emits; for fixed-roster teams that want a canonical slot order, see the
    team-specific equivalents (e.g. `bo_canonical_slots`)."""
    name = "own_side_compact"
    dim = 18

    def extract(self, state: BattleState, ctx: dict) -> np.ndarray:
        out = np.zeros(self.dim, dtype=np.float32)
        side = state.sides[ctx["own_idx"]]
        for slot, fields in enumerate(side.mons[:6]):
            base = slot * 3
            out[base + 0] = _hp_frac(fields)
            out[base + 1] = _status_any(fields)
            out[base + 2] = _alive(fields)
        return out


@register("opp_side_with_roles")
class OppSideWithRoles:
    """Opponent encoding: hp/status/alive per slot, active type one-hots,
    bench-role one-hots (active excluded), active idx, boosts, item flags,
    move-disabled flags.

    The role table and item table are pluggable so different teams can use
    a matchup-relevant role taxonomy. `role_table` maps MON_ID -> role
    index; `n_roles` is the total role count; `item_table` maps ITEM_ID ->
    item-flag index; `n_item_flags` is the total. The OTHER fallback role
    is implicit at `n_roles - 1` (last entry).
    """
    name = "opp_side_with_roles"

    def __init__(self, role_table: dict[str, int], n_roles: int,
                 item_table: dict[str, int], n_item_flags: int):
        self.role_table = role_table
        self.n_roles = n_roles
        self.item_table = item_table
        self.n_item_flags = n_item_flags
        self.dim = (6 * 3
                    + 2 * N_TYPES
                    + 5 * n_roles
                    + 6
                    + 5
                    + n_item_flags
                    + 4)

    def extract(self, state: BattleState, ctx: dict) -> np.ndarray:
        out = np.zeros(self.dim, dtype=np.float32)
        side = state.sides[ctx["opp_idx"]]
        mons = side.mons
        active = side.active_idx
        n_roles = self.n_roles

        for slot, f in enumerate(mons[:6]):
            base = slot * 3
            out[base + 0] = _hp_frac(f)
            out[base + 1] = _status_any(f)
            out[base + 2] = _alive(f)

        type_base = 18
        if active < len(mons):
            t1, t2 = _types(mons[active])
            if t1 in TYPE_IDX:
                out[type_base + TYPE_IDX[t1]] = 1.0
            if t2 in TYPE_IDX:
                out[type_base + N_TYPES + TYPE_IDX[t2]] = 1.0

        role_base = type_base + 2 * N_TYPES  # 54 for n_types=18
        bench_idx = 0
        for slot, f in enumerate(mons[:6]):
            if slot == active:
                continue
            if bench_idx >= 5:
                break
            mid = _mon_id(f)
            if not mid:
                bench_idx += 1
                continue
            role = self.role_table.get(mid, n_roles - 1)  # OTHER = last
            out[role_base + bench_idx * n_roles + role] = 1.0
            bench_idx += 1

        active_idx_base = role_base + 5 * n_roles
        if 0 <= active < 6:
            out[active_idx_base + active] = 1.0

        boost_base = active_idx_base + 6
        out[boost_base:boost_base + 5] = _active_boosts(side.parts)

        item_base = boost_base + 5
        if active < len(mons):
            item = _item(mons[active])
            out[item_base + self.item_table.get(item, self.n_item_flags - 1)] = 1.0

        disabled_base = item_base + self.n_item_flags
        if active < len(mons):
            for i, v in enumerate(_move_disabled_flags(mons[active])):
                out[disabled_base + i] = v

        return out


@register("field_global")
class FieldGlobal:
    """Weather (5 dims, NONE dropped), terrain (4 dims, NONE dropped),
    trick room (1 dim). 10 dims total, side-agnostic."""
    name = "field_global"
    dim = 10

    def extract(self, state: BattleState, ctx: dict) -> np.ndarray:
        out = np.zeros(self.dim, dtype=np.float32)
        w = state.weather.split(";")[0].upper()
        if w in WEATHER_IDX and w != "NONE":
            out[WEATHER_IDX[w] - 1] = 1.0
        t = state.terrain.split(";")[0].upper()
        if t in TERRAIN_IDX and t != "NONE":
            out[5 + TERRAIN_IDX[t] - 1] = 1.0
        if state.trick_room.split(";")[0].lower() == "true":
            out[9] = 1.0
        return out


@register("hazards_screens_both_sides")
class HazardsScreensBothSides:
    """Hazards (SR, Spikes-stacked/3, TSpikes-stacked/2) on own then opp,
    then screens-any flag on own then opp. 8 dims. Side orientation comes
    from ctx['own_idx']."""
    name = "hazards_screens_both_sides"
    dim = 8

    def extract(self, state: BattleState, ctx: dict) -> np.ndarray:
        out = np.zeros(self.dim, dtype=np.float32)
        own_parts = state.sides[ctx["own_idx"]].parts
        opp_parts = state.sides[ctx["opp_idx"]].parts
        out[0], out[1], out[2] = _hazards(own_parts)
        out[3], out[4], out[5] = _hazards(opp_parts)
        out[6] = _screens(own_parts)
        out[7] = _screens(opp_parts)
        return out


# ============================================================
# BO-TEAM ROLE / ITEM TABLES
# ============================================================
# Smogon Gen 9 OU role taxonomy chosen so each role matters for at least one
# BO mon's matchup. Tables live here so the BO blocks below can reference
# them, and `featurizer_bo.py` can hand them to `opp_side_with_roles`.

BO_OPP_ROLES = [
    "FIRE_STEEL_WALL",
    "PRIORITY_BREAKER",
    "SCARFER",
    "WEATHER_SETTER",
    "HAZARD_SETTER",
    "REMOVER",
    "PHYS_SWEEPER",
    "SPEC_SWEEPER",
    "WALLBREAKER",
    "OTHER",
]
BO_N_OPP_ROLES = len(BO_OPP_ROLES)
_BO_ROLE_IDX = {r: i for i, r in enumerate(BO_OPP_ROLES)}

BO_OPP_ROLE_OF: dict[str, int] = {
    "HEATRAN":          _BO_ROLE_IDX["FIRE_STEEL_WALL"],
    "SKARMORY":         _BO_ROLE_IDX["FIRE_STEEL_WALL"],
    "CORVIKNIGHT":      _BO_ROLE_IDX["FIRE_STEEL_WALL"],
    "GHOLDENGO":        _BO_ROLE_IDX["FIRE_STEEL_WALL"],
    "CLODSIRE":         _BO_ROLE_IDX["FIRE_STEEL_WALL"],
    "TOXAPEX":          _BO_ROLE_IDX["FIRE_STEEL_WALL"],
    "BLISSEY":          _BO_ROLE_IDX["FIRE_STEEL_WALL"],
    "CLEFABLE":         _BO_ROLE_IDX["FIRE_STEEL_WALL"],
    "DONDOZO":          _BO_ROLE_IDX["FIRE_STEEL_WALL"],
    "TINGLU":           _BO_ROLE_IDX["HAZARD_SETTER"],
    "GLIMMORA":         _BO_ROLE_IDX["HAZARD_SETTER"],
    "LANDORUSTHERIAN":  _BO_ROLE_IDX["HAZARD_SETTER"],
    "IRONTREADS":       _BO_ROLE_IDX["REMOVER"],
    "KINGAMBIT":        _BO_ROLE_IDX["PRIORITY_BREAKER"],
    "MAMOSWINE":        _BO_ROLE_IDX["PRIORITY_BREAKER"],
    "RILLABOOM":        _BO_ROLE_IDX["PRIORITY_BREAKER"],
    "BISHARP":          _BO_ROLE_IDX["PRIORITY_BREAKER"],
    "PELIPPER":         _BO_ROLE_IDX["WEATHER_SETTER"],
    "TORKOAL":          _BO_ROLE_IDX["WEATHER_SETTER"],
    "TYRANITAR":        _BO_ROLE_IDX["WEATHER_SETTER"],
    "NINETALES":        _BO_ROLE_IDX["WEATHER_SETTER"],
    "ROARINGMOON":      _BO_ROLE_IDX["PHYS_SWEEPER"],
    "DRAGONITE":        _BO_ROLE_IDX["PHYS_SWEEPER"],
    "VOLCARONA":        _BO_ROLE_IDX["PHYS_SWEEPER"],
    "BAXCALIBUR":       _BO_ROLE_IDX["PHYS_SWEEPER"],
    "BARRASKEWDA":      _BO_ROLE_IDX["PHYS_SWEEPER"],
    "EXCADRILL":        _BO_ROLE_IDX["PHYS_SWEEPER"],
    "GREATTUSK":        _BO_ROLE_IDX["PHYS_SWEEPER"],
    "WALKINGWAKE":      _BO_ROLE_IDX["SPEC_SWEEPER"],
    "IRONVALIANT":      _BO_ROLE_IDX["SPEC_SWEEPER"],
    "IRONMOTH":         _BO_ROLE_IDX["SPEC_SWEEPER"],
    "RAGINGBOLT":       _BO_ROLE_IDX["SPEC_SWEEPER"],
    "MANAPHY":          _BO_ROLE_IDX["SPEC_SWEEPER"],
    "ARCHALUDON":       _BO_ROLE_IDX["SPEC_SWEEPER"],
    "DARKRAI":          _BO_ROLE_IDX["WALLBREAKER"],
    "PECHARUNT":        _BO_ROLE_IDX["WALLBREAKER"],
    "HOOPAUNBOUND":     _BO_ROLE_IDX["WALLBREAKER"],
    "KYUREM":           _BO_ROLE_IDX["WALLBREAKER"],
    "URSALUNA":         _BO_ROLE_IDX["WALLBREAKER"],
}

BO_ITEM_FLAGS = [
    "HEAVYDUTYBOOTS",
    "CHOICESCARF",
    "CHOICESPECS",
    "CHOICEBAND",
    "LIFEORB",
    "BOOSTERENERGY",
    "OTHER",
]
BO_N_ITEM_FLAGS = len(BO_ITEM_FLAGS)
BO_ITEM_IDX = {name: i for i, name in enumerate(BO_ITEM_FLAGS)}


# ============================================================
# BO TEAM CONSTANTS (used by team-specific blocks below)
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

FIRE_STEEL_WALLS = {"HEATRAN", "SKARMORY", "CORVIKNIGHT", "GHOLDENGO"}

PRIORITY_USERS = {
    "KINGAMBIT", "MAMOSWINE", "RILLABOOM", "DRAGONITE",
    "BISHARP", "RAGINGBOLT",
}

TRICK_ITEM_USERS = {
    "GHOLDENGO", "IRONBOULDER", "HOOPAUNBOUND",
    "ALAKAZAM", "GENGAR", "LATIOS", "LATIAS",
}

DEFENSIVE_STEEL_GHOST = {"STEEL", "GHOST"}


# ============================================================
# BO TEAM BLOCKS
# ============================================================

@register("bo_canonical_slots")
class BOCanonicalSlots:
    """Reorder BO mons into canonical team-list order (Garg, Darkrai, Tusk,
    Hatt, Torn-T, Dnite) regardless of how the engine slotted them, then
    encode hp/status/alive per canonical slot, active idx one-hot (also
    canonical), boosts, and the Tusk-Booster-consumed bit.

    30 dims. Slots not matched to a BO mon stay zero — covers teampreview
    pre-reveal and unexpected substitutes."""
    name = "bo_canonical_slots"
    dim = 30

    def extract(self, state: BattleState, ctx: dict) -> np.ndarray:
        out = np.zeros(self.dim, dtype=np.float32)
        side = state.sides[ctx["own_idx"]]
        mons = side.mons

        slot_to_canon: dict[int, int] = {}
        tusk_canon_alive = False
        tusk_item = ""
        for slot, f in enumerate(mons):
            mid = _mon_id(f)
            if mid in BO_SLOT_OF:
                canon = BO_SLOT_OF[mid]
                slot_to_canon[slot] = canon
                base = canon * 3
                out[base + 0] = _hp_frac(f)
                out[base + 1] = _status_any(f)
                out[base + 2] = _alive(f)
                if mid == "GREATTUSK":
                    tusk_canon_alive = _alive(f) > 0.5
                    tusk_item = _item(f)

        active_canon = slot_to_canon.get(side.active_idx)
        if active_canon is not None:
            out[18 + active_canon] = 1.0

        out[24:29] = _active_boosts(side.parts)

        out[29] = 1.0 if (tusk_canon_alive and tusk_item != "BOOSTERENERGY") else 0.0
        return out


@register("bo_matchup_signals")
class BOMatchupSignals:
    """BO-specific matchup signals: Dnite DD level, Hatt active (Magic
    Bounce), Tusk Proto online, Garg+Salt-Cure-applicable, opp-outspeeds-
    Dnite-+1, opp has fire/steel wall, opp has priority user, opp has
    Rocky Helmet user, opp has Trick item user, opp positive-boost sum.
    10 dims."""
    name = "bo_matchup_signals"
    dim = 10

    def extract(self, state: BattleState, ctx: dict) -> np.ndarray:
        out = np.zeros(self.dim, dtype=np.float32)
        bo_side = state.sides[ctx["own_idx"]]
        opp_side = state.sides[ctx["opp_idx"]]
        bo_mons = bo_side.mons
        opp_mons = opp_side.mons
        bo_parts = bo_side.parts
        opp_parts = opp_side.parts

        canon_slot: dict[str, int] = {}
        for slot, f in enumerate(bo_mons):
            mid = _mon_id(f)
            if mid in BO_MON_SET:
                canon_slot[mid] = slot

        bo_active = bo_side.active_idx
        opp_active = opp_side.active_idx
        opp_active_fields = opp_mons[opp_active] if opp_active < len(opp_mons) else []

        # 0: Dnite DD level (atk boost stage, only when Dnite is active).
        dnite_slot = canon_slot.get("DRAGONITE")
        if dnite_slot is not None and bo_active == dnite_slot:
            try:
                atk_stage = int(bo_parts[11]) if len(bo_parts) > 11 else 0
            except ValueError:
                atk_stage = 0
            out[0] = max(min(atk_stage / 6.0, 1.0), 0.0)

        # 1: Hatt active.
        hatt_slot = canon_slot.get("HATTERENE")
        if hatt_slot is not None and bo_active == hatt_slot:
            out[1] = 1.0

        # 2: Tusk active + Booster consumed (Proto online).
        tusk_slot = canon_slot.get("GREATTUSK")
        if tusk_slot is not None and bo_active == tusk_slot:
            if _item(bo_mons[tusk_slot]) != "BOOSTERENERGY":
                out[2] = 1.0

        # 3: Garg active + Salt Cure applicable (opp not Steel/Ghost).
        garg_slot = canon_slot.get("GARGANACL")
        if garg_slot is not None and bo_active == garg_slot and opp_active_fields:
            t1, t2 = _types(opp_active_fields)
            if t1 not in DEFENSIVE_STEEL_GHOST and t2 not in DEFENSIVE_STEEL_GHOST:
                out[3] = 1.0

        # 4: Opp outspeeds Dnite +1 (proxy: opp raw spe > dnite spe × 1.5).
        if dnite_slot is not None and opp_active_fields:
            dnite_spe = _speed(bo_mons[dnite_slot])
            opp_spe = _speed(opp_active_fields)
            if dnite_spe > 0 and opp_spe > dnite_spe * 1.5:
                out[4] = 1.0

        # 5: Opp has fire/steel wall alive.
        for f in opp_mons:
            if _alive(f) > 0.5 and _mon_id(f) in FIRE_STEEL_WALLS:
                out[5] = 1.0
                break

        # 6: Opp has strong priority user alive.
        for f in opp_mons:
            if _alive(f) > 0.5 and _mon_id(f) in PRIORITY_USERS:
                out[6] = 1.0
                break

        # 7: Opp has Rocky Helmet user alive.
        for f in opp_mons:
            if _alive(f) > 0.5 and _item(f) == "ROCKYHELMET":
                out[7] = 1.0
                break

        # 8: Opp has Trick-item user alive + currently holding a Choice item.
        for f in opp_mons:
            if _alive(f) > 0.5 and _mon_id(f) in TRICK_ITEM_USERS:
                it = _item(f)
                if it in {"CHOICESCARF", "CHOICESPECS", "CHOICEBAND"}:
                    out[8] = 1.0
                    break

        # 9: Sum of opp active positive boosts (capped at 1).
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
# V3 VOCABULARIES (Gen9 OU detailed featurization)
# ============================================================

# 18 types + Stellar (Terapagos tera) + Typeless sentinel
TYPES_V3 = [
    "NORMAL", "FIRE", "WATER", "ELECTRIC", "GRASS", "ICE",
    "FIGHTING", "POISON", "GROUND", "FLYING", "PSYCHIC", "BUG",
    "ROCK", "GHOST", "DRAGON", "DARK", "STEEL", "FAIRY",
    "STELLAR", "TYPELESS",
]
TYPE_IDX_V3 = {t: i for i, t in enumerate(TYPES_V3)}
N_TYPES_V3 = len(TYPES_V3)  # 20

V3_STATUSES = ["NONE", "BURN", "SLEEP", "FREEZE", "PARALYZE", "POISON", "TOXIC"]
V3_STATUS_IDX = {s: i for i, s in enumerate(V3_STATUSES)}

# Ability categories: function-based encoding (only value-relevant flags)
_V3_AB_MULTISCALE = {"multiscale", "shadowshield"}
_V3_AB_BOOSTABLE = {"protosynthesis", "quarkdrive"}
_V3_AB_REGENERATOR = {"regenerator"}
_V3_AB_HAZARD_IMMUNE = {"magicguard", "levitate"}
_V3_AB_INTIMIDATE = {"intimidate"}
_V3_AB_UNAWARE = {"unaware"}
_V3_AB_STATUS_IMMUNE = {"magicbounce", "goodasgold", "purifyingsalt"}
_V3_AB_SUPREME = {"supremeoverlord"}
N_V3_ABILITY_FLAGS = 8

# Item categories: function-based encoding
_V3_IT_HDB = {"heavydutyboots"}
_V3_IT_CHOICE = {"choiceband", "choicespecs", "choicescarf"}
_V3_IT_LIFEORB = {"lifeorb"}
_V3_IT_FOCUSSASH = {"focussash"}
_V3_IT_AIRBALLOON = {"airballoon"}
_V3_IT_RESTORE = {"leftovers", "blacksludge", "sitrusberry"}
_V3_IT_ROCKYHELMET = {"rockyhelmet"}
_V3_IT_BOOSTERENERGY = {"boosterenergy"}
_V3_IT_WEATHERROCK = {"damprock", "smoothrock", "heatrock", "icyrock"}
_V3_IT_BADORB = {"toxicorb", "flameorb"}
N_V3_ITEM_FLAGS = 11

# Move features come from poke-engine's Rust extractor; cache by move id since
# the features don't change per-state. Empty id → zero vector.
N_V3_MOVE_FEATS = 31
_V3_ZERO_MOVE = np.zeros(N_V3_MOVE_FEATS, dtype=np.float32)
_V3_MOVE_CACHE: dict[str, np.ndarray] = {"": _V3_ZERO_MOVE}


def _v3_normalize_label(s: str) -> str:
    return s.lower().replace(" ", "").replace("-", "").replace("_", "")


def _v3_ability_flags(ability: str) -> np.ndarray:
    a = _v3_normalize_label(ability)
    return np.array([
        1.0 if a in _V3_AB_MULTISCALE else 0.0,
        1.0 if a in _V3_AB_BOOSTABLE else 0.0,
        1.0 if a in _V3_AB_REGENERATOR else 0.0,
        1.0 if a in _V3_AB_HAZARD_IMMUNE else 0.0,
        1.0 if a in _V3_AB_INTIMIDATE else 0.0,
        1.0 if a in _V3_AB_UNAWARE else 0.0,
        1.0 if a in _V3_AB_STATUS_IMMUNE else 0.0,
        1.0 if a in _V3_AB_SUPREME else 0.0,
    ], dtype=np.float32)


def _v3_item_flags(item: str) -> np.ndarray:
    i = _v3_normalize_label(item)
    return np.array([
        1.0 if i in _V3_IT_HDB else 0.0,
        1.0 if i in _V3_IT_CHOICE else 0.0,
        1.0 if i == "choicescarf" else 0.0,
        1.0 if i in _V3_IT_LIFEORB else 0.0,
        1.0 if i in _V3_IT_FOCUSSASH else 0.0,
        1.0 if i in _V3_IT_AIRBALLOON else 0.0,
        1.0 if i in _V3_IT_RESTORE else 0.0,
        1.0 if i in _V3_IT_ROCKYHELMET else 0.0,
        1.0 if i in _V3_IT_BOOSTERENERGY else 0.0,
        1.0 if i in _V3_IT_WEATHERROCK else 0.0,
        1.0 if i in _V3_IT_BADORB else 0.0,
    ], dtype=np.float32)


def _v3_move_feats(move_id: str) -> np.ndarray:
    cached = _V3_MOVE_CACHE.get(move_id)
    if cached is not None:
        return cached
    try:
        v = np.asarray(pe.move_features_v3(move_id), dtype=np.float32)
        if v.shape[0] != N_V3_MOVE_FEATS:
            v = _V3_ZERO_MOVE
    except Exception:
        v = _V3_ZERO_MOVE
    _V3_MOVE_CACHE[move_id] = v
    return v


# Per-pokemon dim, identical to the pre-refactor POKEMON_V3_FEATURES.
N_V3_POKEMON_FEATURES = (
    1 + 1 + N_TYPES_V3 + N_TYPES_V3 + N_TYPES_V3 + 1 + 5 + len(V3_STATUSES) + 1
    + N_V3_ABILITY_FLAGS + N_V3_ITEM_FLAGS + 4 + 4 * N_V3_MOVE_FEATS
)  # 223


def _v3_encode_pokemon(fields: list[str]) -> np.ndarray:
    """Encode one mon's 28 fields into a 223-dim vector. Empty/short field
    lists fall back to zeros so empty slots stay zero."""
    if len(fields) < 28:
        return np.zeros(N_V3_POKEMON_FEATURES, dtype=np.float32)

    feats: list[object] = []

    try:
        hp = int(fields[6]); maxhp = int(fields[7])
    except (ValueError, IndexError):
        hp, maxhp = 0, 1
    feats.append(hp / max(maxhp, 1))
    feats.append(1.0 if hp > 0 else 0.0)

    cur = np.zeros(N_TYPES_V3, dtype=np.float32)
    for j in (2, 3):
        idx = TYPE_IDX_V3.get(fields[j].upper()) if j < len(fields) else None
        if idx is not None:
            cur[idx] = 1.0
    feats.append(cur)

    base = np.zeros(N_TYPES_V3, dtype=np.float32)
    for j in (4, 5):
        idx = TYPE_IDX_V3.get(fields[j].upper()) if j < len(fields) else None
        if idx is not None:
            base[idx] = 1.0
    feats.append(base)

    terad = (len(fields) > 26 and fields[26].lower() == "true")
    tera = np.zeros(N_TYPES_V3, dtype=np.float32)
    if len(fields) > 27:
        idx = TYPE_IDX_V3.get(fields[27].upper())
        if idx is not None:
            tera[idx] = 1.0
    feats.append(tera)
    feats.append(1.0 if terad else 0.0)

    for s_idx in range(13, 18):
        try:
            feats.append(int(fields[s_idx]) / 500.0)
        except (ValueError, IndexError):
            feats.append(0.0)

    status = np.zeros(len(V3_STATUSES), dtype=np.float32)
    s_field = fields[18].upper() if len(fields) > 18 else "NONE"
    if s_field in V3_STATUS_IDX:
        status[V3_STATUS_IDX[s_field]] = 1.0
    else:
        status[0] = 1.0
    feats.append(status)

    try:
        st = int(fields[19]) if len(fields) > 19 else 0
    except ValueError:
        st = 0
    feats.append(min(st / 4.0, 1.0))

    feats.append(_v3_ability_flags(fields[8] if len(fields) > 8 else ""))
    feats.append(_v3_item_flags(fields[10] if len(fields) > 10 else ""))

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
        feats.append(_v3_move_feats(mid))

    flat: list[float] = []
    for x in feats:
        if isinstance(x, np.ndarray):
            flat.extend(x.tolist())
        else:
            flat.append(float(x))
    return np.array(flat, dtype=np.float32)


# ============================================================
# V3 SIDE BLOCKS
# ============================================================

N_V3_SIDE_EXTRAS = 6 + 7 + 9 + 4  # active idx (6) + boosts (7) + hazards/screens (9) + force flags (4)


@register("v3_pokemon_side")
class V3PokemonSide:
    """6 mons × 223 dims = 1338 dims for one side. side_idx selects which
    engine side this block reads. Slot-order (no canonical reorder)."""
    name_template = "v3_pokemon_side_{}"

    def __init__(self, side_idx: int):
        self.side_idx = side_idx
        self.name = self.name_template.format(side_idx)
        self.dim = 6 * N_V3_POKEMON_FEATURES

    def extract(self, state: BattleState, ctx: dict) -> np.ndarray:
        side = state.sides[self.side_idx]
        chunks = [_v3_encode_pokemon(side.mons[i]) for i in range(6)]
        return np.concatenate(chunks).astype(np.float32)


@register("v3_side_extras")
class V3SideExtras:
    """Per-side extras: active idx one-hot (6), boosts including acc/eva (7),
    hazards + screens + tailwind + safeguard (9), force-action flags (4).
    26 dims."""
    name_template = "v3_side_extras_{}"

    def __init__(self, side_idx: int):
        self.side_idx = side_idx
        self.name = self.name_template.format(side_idx)
        self.dim = N_V3_SIDE_EXTRAS

    def extract(self, state: BattleState, ctx: dict) -> np.ndarray:
        side = state.sides[self.side_idx]
        parts = side.parts
        out = np.zeros(self.dim, dtype=np.float32)

        # active idx one-hot (parts[6])
        try:
            ai = int(parts[6]) if len(parts) > 6 else 0
            if 0 <= ai < 6:
                out[ai] = 1.0
        except ValueError:
            pass

        # boosts atk/def/spa/spd/spe/acc/eva at parts[11..17] (7 fields)
        for i in range(7):
            idx = 11 + i
            if idx < len(parts):
                try:
                    out[6 + i] = max(min(int(parts[idx]) / 6.0, 1.0), -1.0)
                except ValueError:
                    pass

        # side conditions in parts[7] (";"-separated 19 fields)
        sc = parts[7].split(";") if len(parts) > 7 else []

        def _sc(idx, scale=1.0):
            try:
                return min(int(sc[idx]) / scale, 1.0) if idx < len(sc) else 0.0
            except ValueError:
                return 0.0

        out[13] = _sc(12, 3.0)         # spikes (0-3)
        out[14] = _sc(13, 1.0)         # stealth_rock
        out[15] = _sc(14, 1.0)         # sticky_web
        out[16] = _sc(17, 2.0)         # toxic_spikes (0-2)
        out[17] = float(_sc(10) > 0)   # reflect
        out[18] = float(_sc(3) > 0)    # light_screen
        out[19] = float(_sc(0) > 0)    # aurora_veil
        out[20] = float(_sc(15) > 0)   # tailwind
        out[21] = float(_sc(11) > 0)   # safeguard

        # force flags: parts[22] force_switch, [24] baton_passing,
        # [26] force_trapped, [28] slow_uturn_move
        def _b(idx):
            return 1.0 if (len(parts) > idx and parts[idx].lower() == "true") else 0.0

        out[22] = _b(22)  # force_switch
        out[23] = _b(26)  # force_trapped
        out[24] = _b(28)  # slow_uturn_move
        out[25] = _b(24)  # baton_passing

        return out


# ============================================================
# V2 VOCABULARIES (Gen 2, 17 types + curated move DB)
# ============================================================

# 17 gen2 types (Steel was added in gen2; no Fairy/Stellar/Typeless)
TYPES_V2 = [
    "NORMAL", "FIRE", "WATER", "ELECTRIC", "GRASS", "ICE",
    "FIGHTING", "POISON", "GROUND", "FLYING", "PSYCHIC", "BUG",
    "ROCK", "GHOST", "DRAGON", "DARK", "STEEL",
]
TYPE_IDX_V2 = {t: i for i, t in enumerate(TYPES_V2)}
N_TYPES_V2 = len(TYPES_V2)

# physical/special split is by type in gen2
V2_PHYSICAL_TYPES = {"NORMAL", "FIGHTING", "FLYING", "POISON", "GROUND",
                     "ROCK", "BUG", "GHOST", "STEEL"}

# gen2 type chart [attacker][defender] -> multiplier
V2_TYPE_CHART = [
    [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.5, 0.0, 1.0, 1.0, 0.5],
    [1.0, 0.5, 0.5, 1.0, 2.0, 2.0, 1.0, 1.0, 1.0, 1.0, 1.0, 2.0, 0.5, 1.0, 0.5, 1.0, 2.0],
    [1.0, 2.0, 0.5, 1.0, 0.5, 1.0, 1.0, 1.0, 2.0, 1.0, 1.0, 1.0, 2.0, 1.0, 0.5, 1.0, 1.0],
    [1.0, 1.0, 2.0, 0.5, 0.5, 1.0, 1.0, 1.0, 0.0, 2.0, 1.0, 1.0, 1.0, 1.0, 0.5, 1.0, 1.0],
    [1.0, 0.5, 2.0, 1.0, 0.5, 1.0, 1.0, 0.5, 2.0, 0.5, 1.0, 0.5, 2.0, 1.0, 0.5, 1.0, 0.5],
    [1.0, 0.5, 0.5, 1.0, 2.0, 0.5, 1.0, 1.0, 2.0, 2.0, 1.0, 1.0, 1.0, 1.0, 2.0, 1.0, 0.5],
    [2.0, 1.0, 1.0, 1.0, 1.0, 2.0, 1.0, 0.5, 1.0, 0.5, 0.5, 0.5, 2.0, 0.0, 1.0, 2.0, 2.0],
    [1.0, 1.0, 1.0, 1.0, 2.0, 1.0, 1.0, 0.5, 0.5, 1.0, 1.0, 1.0, 0.5, 0.5, 1.0, 1.0, 0.0],
    [1.0, 2.0, 1.0, 2.0, 0.5, 1.0, 1.0, 2.0, 1.0, 0.0, 1.0, 0.5, 2.0, 1.0, 1.0, 1.0, 2.0],
    [1.0, 1.0, 1.0, 0.5, 2.0, 1.0, 2.0, 1.0, 1.0, 1.0, 1.0, 2.0, 0.5, 1.0, 1.0, 1.0, 0.5],
    [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 2.0, 2.0, 1.0, 1.0, 0.5, 1.0, 1.0, 1.0, 1.0, 0.0, 0.5],
    [1.0, 0.5, 1.0, 1.0, 2.0, 1.0, 0.5, 0.5, 1.0, 0.5, 2.0, 1.0, 1.0, 0.5, 1.0, 2.0, 0.5],
    [1.0, 2.0, 1.0, 1.0, 1.0, 2.0, 0.5, 1.0, 0.5, 2.0, 1.0, 2.0, 1.0, 1.0, 1.0, 1.0, 0.5],
    [0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 2.0, 1.0, 1.0, 2.0, 1.0, 0.5, 1.0],
    [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 2.0, 1.0, 0.5],
    [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.5, 1.0, 1.0, 1.0, 2.0, 1.0, 1.0, 2.0, 1.0, 0.5, 1.0],
    [1.0, 0.5, 0.5, 0.5, 1.0, 2.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 2.0, 1.0, 1.0, 1.0, 0.5],
]


def v2_type_effectiveness(atk_type: str, def_type1: str, def_type2: str) -> float:
    """Gen 2 type effectiveness multiplier."""
    atk_idx = TYPE_IDX_V2.get(atk_type)
    if atk_idx is None:
        return 1.0
    mult = 1.0
    d1 = TYPE_IDX_V2.get(def_type1)
    d2 = TYPE_IDX_V2.get(def_type2)
    if d1 is not None:
        mult *= V2_TYPE_CHART[atk_idx][d1]
    if d2 is not None and d2 != d1:
        mult *= V2_TYPE_CHART[atk_idx][d2]
    return mult


# Curated gen 2 move DB: move_name -> (type, power, accuracy, is_status, effects)
# effects = [sleep, paralyze, poison, burn, boosts_user, heals_user, phazes]
_S = True

V2_MOVE_DB: dict[str, tuple] = {
    "DOUBLEEDGE":    ("NORMAL",   120, 100, False, [0,0,0,0,0,0,0]),
    "BODYSLAM":      ("NORMAL",    85, 100, False, [0,1,0,0,0,0,0]),
    "RETURN":        ("NORMAL",   102, 100, False, [0,0,0,0,0,0,0]),
    "SELFDESTRUCT":  ("NORMAL",   200, 100, False, [0,0,0,0,0,0,0]),
    "EXPLOSION":     ("NORMAL",   250, 100, False, [0,0,0,0,0,0,0]),
    "MEGAHORN":      ("BUG",      120, 85,  False, [0,0,0,0,0,0,0]),
    "CROSSCHOP":     ("FIGHTING", 100, 80,  False, [0,0,0,0,0,0,0]),
    "DYNAMICPUNCH":  ("FIGHTING", 100, 50,  False, [0,0,0,0,0,0,0]),
    "EARTHQUAKE":    ("GROUND",   100, 100, False, [0,0,0,0,0,0,0]),
    "ROCKSLIDE":     ("ROCK",      75, 90,  False, [0,0,0,0,0,0,0]),
    "SURF":          ("WATER",     95, 100, False, [0,0,0,0,0,0,0]),
    "HYDROPUMP":     ("WATER",    120, 80,  False, [0,0,0,0,0,0,0]),
    "ICEBEAM":       ("ICE",       95, 100, False, [0,0,0,0,0,0,0]),
    "BLIZZARD":      ("ICE",      120, 70,  False, [0,0,0,0,0,0,0]),
    "ICEPUNCH":      ("ICE",       75, 100, False, [0,0,0,0,0,0,0]),
    "THUNDERBOLT":   ("ELECTRIC",  95, 100, False, [0,1,0,0,0,0,0]),
    "THUNDER":       ("ELECTRIC", 120, 70,  False, [0,1,0,0,0,0,0]),
    "FIREBLAST":     ("FIRE",     120, 85,  False, [0,0,0,1,0,0,0]),
    "FLAMETHROWER":  ("FIRE",      95, 100, False, [0,0,0,1,0,0,0]),
    "FIREPUNCH":     ("FIRE",      75, 100, False, [0,0,0,1,0,0,0]),
    "PSYCHIC":       ("PSYCHIC",   90, 100, False, [0,0,0,0,0,0,0]),
    "GIGADRAIN":     ("GRASS",     60, 100, False, [0,0,0,0,0,1,0]),
    "HIDDENPOWERICE":("ICE",       70, 100, False, [0,0,0,0,0,0,0]),
    "DRILLPECK":     ("FLYING",    80, 100, False, [0,0,0,0,0,0,0]),
    "PURSUIT":       ("DARK",      40, 100, False, [0,0,0,0,0,0,0]),
    "CRUNCH":        ("DARK",      80, 100, False, [0,0,0,0,0,0,0]),
    "THIEF":         ("DARK",      40, 100, False, [0,0,0,0,0,0,0]),
    "SLUDGEBOMB":    ("POISON",    90, 100, False, [0,0,1,0,0,0,0]),
    "IRONTAIL":      ("STEEL",    100, 75,  False, [0,0,0,0,0,0,0]),
    "HYPNOSIS":      ("PSYCHIC",    0, 60,  _S, [1,0,0,0,0,0,0]),
    "LOVELYKISS":    ("NORMAL",     0, 75,  _S, [1,0,0,0,0,0,0]),
    "SLEEPPOWDER":   ("GRASS",      0, 75,  _S, [1,0,0,0,0,0,0]),
    "SPORE":         ("GRASS",      0, 100, _S, [1,0,0,0,0,0,0]),
    "SING":          ("NORMAL",     0, 55,  _S, [1,0,0,0,0,0,0]),
    "THUNDERWAVE":   ("ELECTRIC",   0, 100, _S, [0,1,0,0,0,0,0]),
    "STUNSPORE":     ("GRASS",      0, 75,  _S, [0,1,0,0,0,0,0]),
    "TOXIC":         ("POISON",     0, 85,  _S, [0,0,1,0,0,0,0]),
    "WILLOWISP":     ("FIRE",       0, 85,  _S, [0,0,0,1,0,0,0]),
    "CURSE":         ("GHOST",      0, 100, _S, [0,0,0,0,1,0,0]),
    "SWORDSDANCE":   ("NORMAL",     0, 100, _S, [0,0,0,0,1,0,0]),
    "GROWTH":        ("NORMAL",     0, 100, _S, [0,0,0,0,1,0,0]),
    "BELLYDRUM":     ("NORMAL",     0, 100, _S, [0,0,0,0,1,0,0]),
    "REST":          ("PSYCHIC",    0, 100, _S, [0,0,0,0,0,1,0]),
    "RECOVER":       ("NORMAL",     0, 100, _S, [0,0,0,0,0,1,0]),
    "MORNINGSUN":    ("NORMAL",     0, 100, _S, [0,0,0,0,0,1,0]),
    "SOFTBOILED":    ("NORMAL",     0, 100, _S, [0,0,0,0,0,1,0]),
    "MOONLIGHT":     ("NORMAL",     0, 100, _S, [0,0,0,0,0,1,0]),
    "SLEEPTALK":     ("NORMAL",     0, 100, _S, [0,0,0,0,0,0,0]),
    "SUBSTITUTE":    ("NORMAL",     0, 100, _S, [0,0,0,0,0,0,0]),
    "PROTECT":       ("NORMAL",     0, 100, _S, [0,0,0,0,0,0,0]),
    "ENCORE":        ("NORMAL",     0, 100, _S, [0,0,0,0,0,0,0]),
    "ROAR":          ("NORMAL",     0, 100, _S, [0,0,0,0,0,0,1]),
    "WHIRLWIND":     ("NORMAL",     0, 100, _S, [0,0,0,0,0,0,1]),
    "SPIKES":        ("GROUND",     0, 100, _S, [0,0,0,0,0,0,0]),
    "RAPIDSPIN":     ("NORMAL",    20, 100, False, [0,0,0,0,0,0,0]),
    "SUNNYDAY":      ("FIRE",       0, 100, _S, [0,0,0,0,0,0,0]),
    "RAINDANCE":     ("WATER",      0, 100, _S, [0,0,0,0,0,0,0]),
    "SCREECH":       ("NORMAL",     0, 85,  _S, [0,0,0,0,0,0,0]),
    "DESTINYBOND":   ("GHOST",      0, 100, _S, [0,0,0,0,0,0,0]),
    "NONE":          ("NORMAL",     0, 100, _S, [0,0,0,0,0,0,0]),
}
# Hidden Power variants share a uniform 70 BP entry per type
for _hp_type in TYPES_V2:
    _key = f"HIDDENPOWER{_hp_type}"
    if _key not in V2_MOVE_DB:
        V2_MOVE_DB[_key] = (_hp_type, 70, 100, False, [0,0,0,0,0,0,0])


def v2_get_move_props(move_name: str) -> tuple:
    """Look up a gen2 move's (type, power, accuracy, is_status, effects)."""
    name = move_name.upper().replace(" ", "").replace("-", "")
    return V2_MOVE_DB.get(name, ("NORMAL", 0, 100, True, [0,0,0,0,0,0,0]))


def _v2_apply_boost(stat: int, stage: int) -> float:
    """Gen 2 stat-boost formula."""
    if stage >= 0:
        return stat * (2 + stage) / 2
    return stat * 2 / (2 - stage)


V2_ITEMS = ["LEFTOVERS", "THICKCLUB", "LIGHTBALL", "MIRACLEBERRY",
            "MINTBERRY", "OTHER", "NONE"]
V2_ITEM_IDX = {i: idx for idx, i in enumerate(V2_ITEMS)}

N_V2_MOVE_FEATURES = 31     # type(17) + power + acc + phys + status + stab + pp + eff + 7 effect bits
N_V2_POKEMON_FEATURES = 38  # hp + alive + types(17) + stats(5) + status(7) + item(7)
N_V2_SIDE_EXTRAS = 12       # hazards/screens + sleeping target + alive + 5 vol statuses + sleep/rest


def _v2_encode_move(move_str: str, user_types: tuple[str, str],
                    opp_types: tuple[str, str]) -> np.ndarray:
    """31 dims for one move slot. Empty/None move encodes as zeros except
    the type one-hot for NORMAL and is_status=1 (matches legacy behaviour)."""
    out = np.zeros(N_V2_MOVE_FEATURES, dtype=np.float32)
    parts = move_str.split(";")
    move_name = parts[0] if parts else "NONE"
    try:
        pp = int(parts[2]) if len(parts) >= 3 else 0
    except ValueError:
        pp = 0

    mtype, power, accuracy, is_status, effects = v2_get_move_props(move_name)
    i = 0

    tidx = TYPE_IDX_V2.get(mtype)
    if tidx is not None:
        out[i + tidx] = 1.0
    i += N_TYPES_V2

    out[i] = power / 250.0; i += 1
    out[i] = accuracy / 100.0; i += 1
    out[i] = 1.0 if mtype in V2_PHYSICAL_TYPES and not is_status else 0.0; i += 1
    out[i] = 1.0 if is_status else 0.0; i += 1
    out[i] = 1.0 if mtype in (user_types[0], user_types[1]) else 0.0; i += 1
    out[i] = min(pp / 32.0, 1.0) if move_name != "NONE" else 0.0; i += 1
    out[i] = v2_type_effectiveness(mtype, opp_types[0], opp_types[1]) / 4.0; i += 1
    for j in range(7):
        out[i] = float(effects[j])
        i += 1
    return out


def _v2_encode_pokemon(fields: list[str], boosts: list[int] | None,
                       is_active: bool) -> np.ndarray:
    """38 dims per mon. Boosts only apply to the active mon."""
    out = np.zeros(N_V2_POKEMON_FEATURES, dtype=np.float32)
    if len(fields) < 28:
        return out

    i = 0
    hp = int(fields[6]); maxhp = int(fields[7])
    out[i] = hp / max(maxhp, 1); i += 1
    out[i] = 1.0 if hp > 0 else 0.0; i += 1

    t1, t2 = fields[2].upper(), fields[3].upper()
    for t in TYPES_V2:
        out[i] = 1.0 if t1 == t or t2 == t else 0.0
        i += 1

    raw_stats = [int(fields[13]), int(fields[14]), int(fields[15]),
                 int(fields[16]), int(fields[17])]
    if is_active and boosts:
        for j in range(5):
            out[i] = _v2_apply_boost(raw_stats[j], boosts[j]) / 500.0
            i += 1
    else:
        for j in range(5):
            out[i] = raw_stats[j] / 500.0
            i += 1

    status = fields[18].upper()
    for s in V3_STATUSES:  # same 7 statuses as v3
        out[i] = 1.0 if status == s else 0.0
        i += 1

    item = fields[10].upper()
    item_idx = V2_ITEM_IDX.get(item)
    if item_idx is not None:
        out[i + item_idx] = 1.0
    elif item != "" and item != "NONE":
        out[i + 5] = 1.0  # OTHER
    else:
        out[i + 6] = 1.0  # NONE
    return out


def _v2_volatile_statuses(side_part_8: str) -> set:
    if not side_part_8:
        return set()
    return {x.strip().upper() for x in side_part_8.split(":") if x.strip()}


# ============================================================
# V2 BLOCKS
# ============================================================

@register("v2_active_moves")
class V2ActiveMoves:
    """The own-side active mon's 4 moves, each encoded with STAB and type
    effectiveness against the opp active mon. 4 × 31 = 124 dims. Side
    orientation comes from ctx['own_idx']/ctx['opp_idx']."""
    name = "v2_active_moves"
    dim = 4 * N_V2_MOVE_FEATURES

    def extract(self, state: BattleState, ctx: dict) -> np.ndarray:
        out = np.zeros(self.dim, dtype=np.float32)
        own = state.sides[ctx["own_idx"]]
        opp = state.sides[ctx["opp_idx"]]
        own_act = own.mons[own.active_idx] if own.active_idx < len(own.mons) else []
        opp_act = opp.mons[opp.active_idx] if opp.active_idx < len(opp.mons) else []
        own_types = ((own_act[2].upper() if len(own_act) > 2 else ""),
                     (own_act[3].upper() if len(own_act) > 3 else ""))
        opp_types = ((opp_act[2].upper() if len(opp_act) > 2 else ""),
                     (opp_act[3].upper() if len(opp_act) > 3 else ""))
        for m in range(4):
            move_field = (own_act[22 + m]
                          if 22 + m < len(own_act) else "NONE;false;0")
            base = m * N_V2_MOVE_FEATURES
            out[base:base + N_V2_MOVE_FEATURES] = _v2_encode_move(
                move_field, own_types, opp_types)
        return out


@register("v2_pokemon_side")
class V2PokemonSide:
    """6 mons × 38 dims = 228 dims for one side. Boosts apply to that side's
    active mon. side_idx is fixed at construction."""
    name_template = "v2_pokemon_side_{}"

    def __init__(self, side_idx: int):
        self.side_idx = side_idx
        self.name = self.name_template.format(side_idx)
        self.dim = 6 * N_V2_POKEMON_FEATURES

    def extract(self, state: BattleState, ctx: dict) -> np.ndarray:
        out = np.zeros(self.dim, dtype=np.float32)
        side = state.sides[self.side_idx]
        # boosts atk/def/spa/spd/spe at side.parts[11..15]
        boosts: list[int] | None = None
        if len(side.parts) >= 18:
            try:
                boosts = [int(side.parts[11 + j]) for j in range(5)]
            except (ValueError, IndexError):
                boosts = [0] * 5
        for p in range(6):
            fields = side.mons[p] if p < len(side.mons) else []
            is_active = (p == side.active_idx)
            chunk = _v2_encode_pokemon(
                fields, boosts if is_active else None, is_active)
            base = p * N_V2_POKEMON_FEATURES
            out[base:base + N_V2_POKEMON_FEATURES] = chunk
        return out


@register("v2_side_extras")
class V2SideExtras:
    """12 side extras: spikes/reflect/lightscreen + has_sleeping_target on
    the OTHER side + num_alive + 5 active volatile statuses + active sleep
    and rest timers. side_idx fixes which side this encodes; the
    has_sleeping_target lookup uses the opposite side."""
    name_template = "v2_side_extras_{}"

    def __init__(self, side_idx: int):
        self.side_idx = side_idx
        self.name = self.name_template.format(side_idx)
        self.dim = N_V2_SIDE_EXTRAS

    def extract(self, state: BattleState, ctx: dict) -> np.ndarray:
        out = np.zeros(self.dim, dtype=np.float32)
        side = state.sides[self.side_idx]
        opp = state.sides[1 - self.side_idx]
        my_parts = side.parts
        opp_parts = opp.parts
        active_fields = (side.mons[side.active_idx]
                         if side.active_idx < len(side.mons) else [])

        if len(my_parts) >= 18:
            try:
                sc = my_parts[7].split(";")
                out[0] = int(sc[12]) / 3.0 if len(sc) > 12 else 0.0
                out[1] = float(int(sc[10]) > 0) if len(sc) > 10 else 0.0
                out[2] = float(int(sc[3]) > 0) if len(sc) > 3 else 0.0
            except (ValueError, IndexError):
                pass

        for p in range(6):
            pf = opp.mons[p] if p < len(opp.mons) else []
            if len(pf) > 18 and pf[18].upper() == "SLEEP":
                out[3] = 1.0
                break

        alive = sum(1 for p in range(6)
                    if p < len(side.mons) and len(side.mons[p]) > 6
                    and (lambda f: int(f) if f.isdigit() or
                         (f.startswith('-') and f[1:].isdigit()) else 0)(side.mons[p][6]) > 0)
        out[4] = alive / 6.0

        vs = _v2_volatile_statuses(my_parts[8] if len(my_parts) > 8 else "")
        out[5] = 1.0 if "SUBSTITUTE" in vs else 0.0
        out[6] = 1.0 if "ENCORE" in vs else 0.0
        out[7] = 1.0 if "DISABLE" in vs else 0.0
        out[8] = 1.0 if "MUSTRECHARGE" in vs else 0.0
        out[9] = 1.0 if "PARTIALLYTRAPPED" in vs else 0.0

        if len(active_fields) > 20:
            try:
                out[10] = max(0.0, min(1.0, int(active_fields[20]) / 7.0))
                out[11] = max(0.0, min(1.0, int(active_fields[19]) / 2.0))
            except ValueError:
                pass
        return out


@register("v2_global")
class V2Global:
    """5 dims: weather (3: sun/rain/sand one-hot), speed comparison
    (1: own boosted-spe > opp boosted-spe), priority placeholder (1: always
    0 in legacy v2). Side orientation from ctx['own_idx']/ctx['opp_idx']."""
    name = "v2_global"
    dim = 5

    def extract(self, state: BattleState, ctx: dict) -> np.ndarray:
        out = np.zeros(self.dim, dtype=np.float32)
        w = state.weather.upper()
        if "SUN" in w:
            out[0] = 1.0
        elif "RAIN" in w:
            out[1] = 1.0
        elif "SAND" in w:
            out[2] = 1.0

        own = state.sides[ctx["own_idx"]]
        opp = state.sides[ctx["opp_idx"]]
        own_act = own.mons[own.active_idx] if own.active_idx < len(own.mons) else []
        opp_act = opp.mons[opp.active_idx] if opp.active_idx < len(opp.mons) else []

        own_boosts: list[int] | None = None
        opp_boosts: list[int] | None = None
        if len(own.parts) >= 18:
            try:
                own_boosts = [int(own.parts[11 + j]) for j in range(5)]
            except (ValueError, IndexError):
                own_boosts = [0] * 5
        if len(opp.parts) >= 18:
            try:
                opp_boosts = [int(opp.parts[11 + j]) for j in range(5)]
            except (ValueError, IndexError):
                opp_boosts = [0] * 5

        my_speed = int(own_act[17]) if len(own_act) > 17 else 100
        opp_speed = int(opp_act[17]) if len(opp_act) > 17 else 100
        if own_boosts:
            my_speed = _v2_apply_boost(my_speed, own_boosts[4])
        if opp_boosts:
            opp_speed = _v2_apply_boost(opp_speed, opp_boosts[4])
        out[3] = 1.0 if my_speed > opp_speed else 0.0
        # out[4] priority placeholder stays 0.0 (matches legacy)
        return out
