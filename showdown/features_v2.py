#!/usr/bin/env python3
# v2 feature extraction: move-aware, boosted stats, type effectiveness
#
# 579 dims: active_moves(120) + pokemon(444) + sides(10) + global(5)

import numpy as np

# ============================================================
# GEN2 TYPE SYSTEM
# ============================================================

# 17 gen2 types (Steel was added in gen2; no fairy/stellar)
TYPES_V2 = [
    "NORMAL", "FIRE", "WATER", "ELECTRIC", "GRASS", "ICE",
    "FIGHTING", "POISON", "GROUND", "FLYING", "PSYCHIC", "BUG",
    "ROCK", "GHOST", "DRAGON", "DARK", "STEEL",
]
TYPE_IDX_V2 = {t: i for i, t in enumerate(TYPES_V2)}
N_TYPES = len(TYPES_V2)

# physical types in gen2 (category determined by type, not move)
PHYSICAL_TYPES = {"NORMAL", "FIGHTING", "FLYING", "POISON", "GROUND", "ROCK", "BUG", "GHOST", "STEEL"}

# gen2 type chart [attacker][defender] -> multiplier
# order matches TYPES_V2 (17x17 with Steel)
TYPE_CHART = [
    #     NOR  FIR  WAT  ELE  GRA  ICE  FIG  POI  GRO  FLY  PSY  BUG  ROC  GHO  DRA  DAR  STE
    [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.5, 0.0, 1.0, 1.0, 0.5],  # NORMAL
    [1.0, 0.5, 0.5, 1.0, 2.0, 2.0, 1.0, 1.0, 1.0, 1.0, 1.0, 2.0, 0.5, 1.0, 0.5, 1.0, 2.0],  # FIRE
    [1.0, 2.0, 0.5, 1.0, 0.5, 1.0, 1.0, 1.0, 2.0, 1.0, 1.0, 1.0, 2.0, 1.0, 0.5, 1.0, 1.0],  # WATER
    [1.0, 1.0, 2.0, 0.5, 0.5, 1.0, 1.0, 1.0, 0.0, 2.0, 1.0, 1.0, 1.0, 1.0, 0.5, 1.0, 1.0],  # ELECTRIC
    [1.0, 0.5, 2.0, 1.0, 0.5, 1.0, 1.0, 0.5, 2.0, 0.5, 1.0, 0.5, 2.0, 1.0, 0.5, 1.0, 0.5],  # GRASS
    [1.0, 0.5, 0.5, 1.0, 2.0, 0.5, 1.0, 1.0, 2.0, 2.0, 1.0, 1.0, 1.0, 1.0, 2.0, 1.0, 0.5],  # ICE
    [2.0, 1.0, 1.0, 1.0, 1.0, 2.0, 1.0, 0.5, 1.0, 0.5, 0.5, 0.5, 2.0, 0.0, 1.0, 2.0, 2.0],  # FIGHTING
    [1.0, 1.0, 1.0, 1.0, 2.0, 1.0, 1.0, 0.5, 0.5, 1.0, 1.0, 1.0, 0.5, 0.5, 1.0, 1.0, 0.0],  # POISON
    [1.0, 2.0, 1.0, 2.0, 0.5, 1.0, 1.0, 2.0, 1.0, 0.0, 1.0, 0.5, 2.0, 1.0, 1.0, 1.0, 2.0],  # GROUND
    [1.0, 1.0, 1.0, 0.5, 2.0, 1.0, 2.0, 1.0, 1.0, 1.0, 1.0, 2.0, 0.5, 1.0, 1.0, 1.0, 0.5],  # FLYING
    [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 2.0, 2.0, 1.0, 1.0, 0.5, 1.0, 1.0, 1.0, 1.0, 0.0, 0.5],  # PSYCHIC
    [1.0, 0.5, 1.0, 1.0, 2.0, 1.0, 0.5, 0.5, 1.0, 0.5, 2.0, 1.0, 1.0, 0.5, 1.0, 2.0, 0.5],  # BUG
    [1.0, 2.0, 1.0, 1.0, 1.0, 2.0, 0.5, 1.0, 0.5, 2.0, 1.0, 2.0, 1.0, 1.0, 1.0, 1.0, 0.5],  # ROCK
    [0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 2.0, 1.0, 1.0, 2.0, 1.0, 0.5, 1.0],  # GHOST
    [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 2.0, 1.0, 0.5],  # DRAGON
    [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.5, 1.0, 1.0, 1.0, 2.0, 1.0, 1.0, 2.0, 1.0, 0.5, 1.0],  # DARK
    [1.0, 0.5, 0.5, 0.5, 1.0, 2.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 2.0, 1.0, 1.0, 1.0, 0.5],  # STEEL
]


def type_effectiveness(atk_type: str, def_type1: str, def_type2: str) -> float:
    """Compute type effectiveness multiplier."""
    atk_idx = TYPE_IDX_V2.get(atk_type)
    if atk_idx is None:
        return 1.0
    mult = 1.0
    d1 = TYPE_IDX_V2.get(def_type1)
    d2 = TYPE_IDX_V2.get(def_type2)
    if d1 is not None:
        mult *= TYPE_CHART[atk_idx][d1]
    if d2 is not None and d2 != d1:
        mult *= TYPE_CHART[atk_idx][d2]
    return mult


# ============================================================
# MOVE PROPERTIES LOOKUP
# ============================================================

# move_name -> (type, power, accuracy, is_status, effects_flags)
# effects: [sleeps, paralyzes, poisons, burns, boosts_user, heals_user, phazes]
_S = True  # status move flag

MOVE_DB = {
    # ---- common attacking moves ----
    "DOUBLEEDGE":    ("NORMAL",   120, 100, False, [0,0,0,0,0,0,0]),
    "BODYSLAM":      ("NORMAL",    85, 100, False, [0,1,0,0,0,0,0]),
    "RETURN":        ("NORMAL",   102, 100, False, [0,0,0,0,0,0,0]),
    "SELFDESTRUCT":  ("NORMAL",   200, 100, False, [0,0,0,0,0,0,0]),
    "EXPLOSION":     ("NORMAL",   250, 100, False, [0,0,0,0,0,0,0]),
    "MEGAHORN":      ("BUG",     120, 85,  False, [0,0,0,0,0,0,0]),
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
    # ---- status moves ----
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

# handle Hidden Power variants
for hp_type in TYPES_V2:
    key = f"HIDDENPOWER{hp_type}"
    if key not in MOVE_DB:
        MOVE_DB[key] = (hp_type, 70, 100, False, [0,0,0,0,0,0,0])


def get_move_props(move_name: str) -> tuple:
    """Look up move properties. Returns (type, power, accuracy, is_status, effects)."""
    name = move_name.upper().replace(" ", "").replace("-", "")
    # handle "HIDDENPOWER" with bracket notation from state strings
    if name.startswith("HIDDENPOWER"):
        # already handled in MOVE_DB
        pass
    return MOVE_DB.get(name, ("NORMAL", 0, 100, True, [0,0,0,0,0,0,0]))


# ============================================================
# BOOST CALCULATION
# ============================================================

def apply_boost(stat: int, stage: int) -> float:
    """Apply gen2 stat boost stage to a stat value."""
    if stage >= 0:
        return stat * (2 + stage) / 2
    else:
        return stat * 2 / (2 - stage)


# ============================================================
# V2 FEATURE EXTRACTION
# ============================================================

# per move: type(17) + power(1) + accuracy(1) + physical(1) + status(1) + stab(1) + pp(1) + effectiveness(1) + effects(7) = 31
MOVE_FEATURES = 31
# per pokemon: hp(1) + alive(1) + types(17) + stats(5) + status(7) + item(7) = 38
POKEMON_FEATURES_V2 = 38
# side extras: spikes(1) + reflect(1) + light_screen(1) + has_sleeping_target(1) + num_alive(1)
#   + active vol statuses (sub/encore/disable/recharge/partial_trap = 5)
#   + active sleep_turns/7 + rest_turns/2 = 12
SIDE_EXTRAS_V2 = 12

# active moves: 4 * 31 = 124
# pokemon: 6 * 38 * 2 = 456
# side extras: 12 * 2 = 24
# global: weather(3) + speed_cmp(1) + priority_cmp(1) = 5
# total: 124 + 456 + 24 + 5 = 609
STATE_FEATURES_V2 = 4 * MOVE_FEATURES + 6 * POKEMON_FEATURES_V2 * 2 + SIDE_EXTRAS_V2 * 2 + 5
N_ACTIONS = 9

# items (trimmed for gen2)
ITEMS_V2 = ["LEFTOVERS", "THICKCLUB", "LIGHTBALL", "MIRACLEBERRY", "MINTBERRY", "OTHER", "NONE"]
ITEM_IDX_V2 = {i: idx for idx, i in enumerate(ITEMS_V2)}

# status
STATUSES = ["NONE", "BURN", "SLEEP", "FREEZE", "PARALYZE", "POISON", "TOXIC"]
STATUS_IDX = {s: i for i, s in enumerate(STATUSES)}


def parse_move_features(move_str: str, user_types: tuple, opp_types: tuple) -> np.ndarray:
    """Extract 30-dim features for one move slot."""
    features = np.zeros(MOVE_FEATURES, dtype=np.float32)
    parts = move_str.split(";")
    move_name = parts[0] if parts else "NONE"
    pp = int(parts[2]) if len(parts) >= 3 else 0

    move_type, power, accuracy, is_status, effects = get_move_props(move_name)

    i = 0
    # type one-hot (16)
    tidx = TYPE_IDX_V2.get(move_type)
    if tidx is not None:
        features[i + tidx] = 1.0
    i += N_TYPES

    # power / 250
    features[i] = power / 250.0
    i += 1

    # accuracy / 100
    features[i] = accuracy / 100.0
    i += 1

    # physical flag (gen2: determined by type)
    features[i] = 1.0 if move_type in PHYSICAL_TYPES and not is_status else 0.0
    i += 1

    # status flag
    features[i] = 1.0 if is_status else 0.0
    i += 1

    # STAB
    features[i] = 1.0 if move_type in (user_types[0], user_types[1]) else 0.0
    i += 1

    # PP fraction
    features[i] = min(pp / 32.0, 1.0) if move_name != "NONE" else 0.0
    i += 1

    # effectiveness vs opponent
    features[i] = type_effectiveness(move_type, opp_types[0], opp_types[1]) / 4.0
    i += 1

    # effect flags (7)
    for j in range(7):
        features[i] = float(effects[j])
        i += 1

    return features


def parse_pokemon_v2(fields: list[str], boosts: list[int] = None,
                     is_active: bool = False) -> np.ndarray:
    """Extract 37-dim features for one pokemon."""
    features = np.zeros(POKEMON_FEATURES_V2, dtype=np.float32)
    i = 0

    # hp fraction
    hp = int(fields[6])
    maxhp = int(fields[7])
    features[i] = hp / max(maxhp, 1)
    i += 1

    # alive
    features[i] = 1.0 if hp > 0 else 0.0
    i += 1

    # types (16)
    t1 = fields[2].upper()
    t2 = fields[3].upper()
    for t in TYPES_V2:
        features[i] = 1.0 if t1 == t or t2 == t else 0.0
        i += 1

    # stats with boosts applied for active (5)
    raw_stats = [int(fields[13]), int(fields[14]), int(fields[15]),
                 int(fields[16]), int(fields[17])]  # atk/def/spa/spd/spe
    if is_active and boosts:
        # boosts order: atk/def/spa/spd/spe
        for j in range(5):
            features[i] = apply_boost(raw_stats[j], boosts[j]) / 500.0
            i += 1
    else:
        for j in range(5):
            features[i] = raw_stats[j] / 500.0
            i += 1

    # status (7)
    status = fields[18].upper()
    for s in STATUSES:
        features[i] = 1.0 if status == s else 0.0
        i += 1

    # item (7)
    item = fields[10].upper()
    item_idx = ITEM_IDX_V2.get(item)
    if item_idx is not None:
        features[i + item_idx] = 1.0
    elif item != "" and item != "NONE":
        features[i + 5] = 1.0  # OTHER
    else:
        features[i + 6] = 1.0  # NONE
    i += 7

    return features


def _parse_volatile_statuses(side_part_8: str) -> set:
    """Parse colon-separated volatile statuses from a side's serialized form."""
    if not side_part_8:
        return set()
    return {x.strip().upper() for x in side_part_8.split(":") if x.strip()}


def _write_side_extras(out, idx, my_parts, opp_parts, my_active_fields):
    """Write 12 side extras starting at out[idx]: spikes/reflect/lightscreen/
    has_sleeping_target/num_alive + 5 active vol statuses + sleep_turns + rest_turns.
    """
    if len(my_parts) >= 18:
        try:
            sc = my_parts[7].split(";")
            out[idx] = int(sc[12]) / 3.0 if len(sc) > 12 else 0.0       # spikes
            out[idx + 1] = float(int(sc[10]) > 0) if len(sc) > 10 else 0.0  # reflect
            out[idx + 2] = float(int(sc[3]) > 0) if len(sc) > 3 else 0.0    # light screen
        except (ValueError, IndexError):
            pass
    # has sleeping target on opp
    for p in range(6):
        pf = opp_parts[p].split(",") if p < len(opp_parts) else []
        if len(pf) > 18 and pf[18].upper() == "SLEEP":
            out[idx + 3] = 1.0
            break
    # num alive / 6
    alive = sum(1 for p in range(6)
                for pf in [my_parts[p].split(",")]
                if len(pf) > 6 and int(pf[6]) > 0)
    out[idx + 4] = alive / 6.0
    # active volatile statuses (5 binary bits)
    vs = _parse_volatile_statuses(my_parts[8] if len(my_parts) > 8 else "")
    out[idx + 5] = 1.0 if "SUBSTITUTE" in vs else 0.0
    out[idx + 6] = 1.0 if "ENCORE" in vs else 0.0
    out[idx + 7] = 1.0 if "DISABLE" in vs else 0.0
    out[idx + 8] = 1.0 if "MUSTRECHARGE" in vs else 0.0
    out[idx + 9] = 1.0 if "PARTIALLYTRAPPED" in vs else 0.0
    # active sleep_turns/7 + rest_turns/2
    if len(my_active_fields) > 20:
        try:
            out[idx + 10] = max(0.0, min(1.0, int(my_active_fields[20]) / 7.0))  # sleep_turns
            out[idx + 11] = max(0.0, min(1.0, int(my_active_fields[19]) / 2.0))  # rest_turns
        except ValueError:
            pass


def parse_state_v2(state_str: str) -> np.ndarray:
    """Extract 609-dim v2 feature vector from a state string."""
    features = np.zeros(STATE_FEATURES_V2, dtype=np.float32)
    idx = 0

    major_parts = state_str.split("/")
    if len(major_parts) < 2:
        return features

    s1_parts = major_parts[0].split("=")
    s2_parts = major_parts[1].split("=")

    # ---- active pokemon types (needed for STAB/effectiveness) ----
    s1_active_idx = int(s1_parts[6]) if len(s1_parts) > 6 else 0
    s2_active_idx = int(s2_parts[6]) if len(s2_parts) > 6 else 0

    s1_active_fields = s1_parts[s1_active_idx].split(",")
    s2_active_fields = s2_parts[s2_active_idx].split(",")

    s1_types = (s1_active_fields[2].upper(), s1_active_fields[3].upper())
    s2_types = (s2_active_fields[2].upper(), s2_active_fields[3].upper())

    # ---- my active moves (4 x 30 = 120) ----
    for m in range(4):
        move_field = s1_active_fields[22 + m] if 22 + m < len(s1_active_fields) else "NONE;false;0"
        features[idx:idx + MOVE_FEATURES] = parse_move_features(
            move_field, s1_types, s2_types)
        idx += MOVE_FEATURES

    # ---- my team (6 x 37 = 222) ----
    s1_boosts = None
    if len(s1_parts) >= 18:
        try:
            s1_boosts = [int(s1_parts[11 + j]) for j in range(5)]  # atk/def/spa/spd/spe
        except (ValueError, IndexError):
            s1_boosts = [0] * 5

    for p in range(6):
        pfields = s1_parts[p].split(",") if p < len(s1_parts) else []
        is_active = (p == s1_active_idx)
        if len(pfields) >= 28:
            features[idx:idx + POKEMON_FEATURES_V2] = parse_pokemon_v2(
                pfields, s1_boosts if is_active else None, is_active)
        idx += POKEMON_FEATURES_V2

    # ---- opponent team (6 x 37 = 222) ----
    s2_boosts = None
    if len(s2_parts) >= 18:
        try:
            s2_boosts = [int(s2_parts[11 + j]) for j in range(5)]
        except (ValueError, IndexError):
            s2_boosts = [0] * 5

    for p in range(6):
        pfields = s2_parts[p].split(",") if p < len(s2_parts) else []
        is_active = (p == s2_active_idx)
        if len(pfields) >= 28:
            features[idx:idx + POKEMON_FEATURES_V2] = parse_pokemon_v2(
                pfields, s2_boosts if is_active else None, is_active)
        idx += POKEMON_FEATURES_V2

    # ---- side 1 extras (12) ----
    _write_side_extras(features, idx, s1_parts, s2_parts, s1_active_fields)
    idx += SIDE_EXTRAS_V2

    # ---- side 2 extras (12) ----
    _write_side_extras(features, idx, s2_parts, s1_parts, s2_active_fields)
    idx += SIDE_EXTRAS_V2

    # ---- global (5) ----
    # weather (3)
    if len(major_parts) > 2:
        w = major_parts[2].upper()
        if "SUN" in w:
            features[idx] = 1.0
        elif "RAIN" in w:
            features[idx + 1] = 1.0
        elif "SAND" in w:
            features[idx + 2] = 1.0
    idx += 3

    # speed comparison (1): my boosted speed > opp boosted speed
    my_speed = int(s1_active_fields[17]) if len(s1_active_fields) > 17 else 100
    opp_speed = int(s2_active_fields[17]) if len(s2_active_fields) > 17 else 100
    if s1_boosts:
        my_speed = apply_boost(my_speed, s1_boosts[4])
    if s2_boosts:
        opp_speed = apply_boost(opp_speed, s2_boosts[4])
    features[idx] = 1.0 if my_speed > opp_speed else 0.0
    idx += 1

    # priority move available (1): any move with priority > 0
    # gen2 doesn't really have priority moves (only Mach Punch, Quick Attack)
    # but we can flag it from the move data
    features[idx] = 0.0  # placeholder, could check move DB
    idx += 1

    return features


if __name__ == "__main__":
    # quick test
    print(f"STATE_FEATURES_V2 = {STATE_FEATURES_V2}")
    assert STATE_FEATURES_V2 == 579

    # test with a real state string
    import sys
    sys.path.insert(0, ".")
    from showdown.local_battle import build_pe_state
    from showdown.sample_teams import SAMPLE_TEAMS

    state = build_pe_state(SAMPLE_TEAMS[13], SAMPLE_TEAMS[3])
    state_str = state.to_string()
    features = parse_state_v2(state_str)
    print(f"Feature vector shape: {features.shape}")
    print(f"Non-zero features: {np.count_nonzero(features)}/{len(features)}")
    print(f"Move slot 0 (first 30 dims): {features[:30]}")
