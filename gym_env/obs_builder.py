# state -> float32 observation vector (1052 used, no padding)
# layout: active(407) + my_team(6x63=378) + opp_team(6x42=252) + global(15)

from __future__ import annotations

import numpy as np

from engine.damage import calc_expected_damage
from engine.player_state import PlayerState
from engine.stat_stages import MOVE_STAT_EFFECTS
from engine.status import TYPE_IMMUNITIES, AILMENT_MAP, effective_speed
from engine.types import TypeChart

# all 17 gen2 types, sorted for consistent encoding
ALL_TYPES = sorted([
    "bug", "dark", "dragon", "electric", "fighting", "fire", "flying",
    "ghost", "grass", "ground", "ice", "normal", "poison", "psychic",
    "rock", "steel", "water",
])
TYPE_TO_IDX = {t: i for i, t in enumerate(ALL_TYPES)}
NUM_TYPES = len(ALL_TYPES)

OBS_SIZE = 1052

# setup move classification
_SELF_BOOST_MOVES = {
    name: effects for name, effects in MOVE_STAT_EFFECTS.items()
    if any(target == "self" and stages > 0 for _, stages, target in effects)
}
_OPP_DEBUFF_MOVES = {
    name: effects for name, effects in MOVE_STAT_EFFECTS.items()
    if any(target == "opponent" and stages < 0 for _, stages, target in effects)
}
_HAZARD_MOVES = frozenset(["Spikes"])
_SCREEN_MOVES = frozenset(["Reflect", "Light Screen", "Safeguard"])
_WEATHER_MOVES = frozenset(["Rain Dance", "Sunny Day", "Sandstorm"])

# moves with effects not captured in PokeAPI meta
_MOVE_OVERRIDES: dict[str, dict[str, float]] = {
    "Rest": {"healing": 100, "ailment_chance": -100},     # full heal, self-sleeps
    "Belly Drum": {"healing": -50},                        # costs 50% HP
}

# normalization constants
MAX_SPEED = 500
MAX_STAT = 500
MAX_POWER = 250
MAX_TURNS = 200
MAX_PP = 40
MAX_STATUS_TURNS = 16  # toxic counter max ~16, sleep max 7
MAX_SCREEN_TURNS = 5
MAX_WEATHER_TURNS = 5
MAX_PROTECT = 4

# status -> one-hot index (0=none, 1-6=brn/par/slp/frz/psn/tox)
STATUS_INDEX = {
    None: 0,
    "brn": 1,
    "par": 2,
    "slp": 3,
    "frz": 4,
    "psn": 5,
    "tox": 6,
}
N_STATUS = 7

# ordinal encoding (legacy, kept for reference)
STATUS_ORDINAL = {
    None: 0.0,
    "brn": 0.14,
    "par": 0.28,
    "slp": 0.42,
    "frz": 0.57,
    "psn": 0.71,
    "tox": 1.0,
}

# damage class encoding
DAMAGE_CLASS_ENC = {
    "physical": 1.0,
    "special": 0.5,
    "status": 0.0,
}

# ailment_id -> ordinal encoding
AILMENT_ORDINAL = {
    1: 0.28,  # paralysis
    2: 0.42,  # sleep
    3: 0.57,  # freeze
    4: 0.14,  # burn
    5: 0.71,  # poison
    6: 0.85,  # confusion
}

# weather encoding indices (one-hot, 3 flags)
WEATHER_INDEX = {
    "sun": 0,
    "rain": 1,
    "sandstorm": 2,
}


def encode_defensive_profile(types: list[str], type_chart: TypeChart) -> list[float]:
    """17-float defensive type profile: how effective each attack type is against this mon.

    Each element = combined_effectiveness(atk_type, types) / 4.0.
    Encodes vulnerability structure directly (e.g. ground=4x vs electric/steel).
    """
    return [
        type_chart.combined_effectiveness(atk_type, types) / 4.0
        for atk_type in ALL_TYPES
    ]


def encode_move_type_onehot(move_type: str) -> list[float]:
    """17-float one-hot for a single move type."""
    vec = [0.0] * NUM_TYPES
    idx = TYPE_TO_IDX.get(move_type)
    if idx is not None:
        vec[idx] = 1.0
    return vec


def encode_status_onehot(status: str | None) -> list[float]:
    """7-float one-hot encoding for status."""
    idx = STATUS_INDEX.get(status, 0)
    vec = [0.0] * N_STATUS
    vec[idx] = 1.0
    return vec


def _move_meta(template, key: str, default: int = 0) -> float:
    """Safely get a meta field, with overrides for moves PokeAPI doesn't tag."""
    override = _MOVE_OVERRIDES.get(template.name)
    if override and key in override:
        return override[key]
    if template.meta is None:
        return default
    return template.meta.get(key, default)


def _multi_hit_avg(template) -> float:
    """Average hit count for multi-hit moves, 0.0 for single-hit."""
    if template.meta is None:
        return 0.0
    min_h = template.meta.get("min_hits")
    max_h = template.meta.get("max_hits")
    if min_h and max_h and max_h > 1:
        return (min_h + max_h) / 2.0 / 5.0  # normalize by max possible (5)
    return 0.0


def _ailment_can_land(template, target_types: list[str], target_status) -> float:
    """Whether this move's status effect can land on the target.

    Returns 1.0 if the ailment can land, 0.0 if the target is immune or
    already statused. Returns 0.0 for moves with no ailment.
    """
    ailment_id = _move_meta(template, "ailment_id")
    if ailment_id == 0:
        return 0.0
    if ailment_id == 6:  # confusion -- no type immunity
        return 1.0
    status = AILMENT_MAP.get(ailment_id)
    if status is None:
        return 0.0
    # special case: Toxic applies TOX, not PSN
    if ailment_id == 5 and template.name == "Toxic":
        status = "tox"
    # already has a non-volatile status
    if target_status is not None:
        return 0.0
    # type immunity check
    for ptype in target_types:
        if ptype in TYPE_IMMUNITIES and status in TYPE_IMMUNITIES[ptype]:
            return 0.0
    return 1.0


# multi-turn move penalties (active moves only)
_CHARGE_DODGE = {19, 91}                       # Fly, Dig (semi-invulnerable charge)
_CHARGE_EXPOSED = {13, 130, 143}               # Razor Wind, Skull Bash, Sky Attack
_SOLAR_BEAM = 76                               # instant in sun
_RECHARGE_MOVES = {63}                         # Hyper Beam
_LOCKIN_MOVES = {37, 80, 200}                  # Thrash, Petal Dance, Outrage

def _multi_turn_cost(template, weather=None) -> float:
    mid = template.id
    if mid == _SOLAR_BEAM:
        return 0.0 if weather == "sun" else 0.5
    if mid in _CHARGE_DODGE:
        return 0.25
    if mid in _CHARGE_EXPOSED:
        return 0.5
    if mid in _RECHARGE_MOVES:
        return 0.75
    if mid in _LOCKIN_MOVES:
        return 1.0
    return 0.0

# ---- per-move feature counts ----
# 17 (type one-hot) + 19 scalars + high_crit + multi_hit + ailment_can_land = 39
MY_MOVE_FEATURES = 39

# 17 (type one-hot) + 18 scalars + ailment_can_land = 36
OPP_MOVE_FEATURES = 36

# ---- per-team-slot feature counts ----
MY_TEAM_PER = 63   # +best_move_eff, +spikes_entry_dmg, +switch_in_cost, +matchup_improvement
OPP_TEAM_PER = 42  # +my_best_eff_vs_them, +move_type_count


def build_observation(
    my_state: PlayerState,
    opp_state: PlayerState,
    type_chart: TypeChart,
    turn: int,
    weather: str | None = None,
    weather_turns: int = 0,
) -> np.ndarray:
    """Build the observation vector (1038 features, padded to 1040)."""
    obs: list[float] = []
    my_active = my_state.active
    opp_active = opp_state.active

    # ============================================================
    # ACTIVE MATCHUP (397 features)
    # ============================================================

    # ---- hp and speed (5) ----
    my_speed = effective_speed(my_active)
    opp_speed = effective_speed(opp_active)
    obs.append(my_active.hp_frac)
    obs.append(my_speed / MAX_SPEED)
    obs.append(opp_active.hp_frac)
    obs.append(opp_speed / MAX_SPEED)
    obs.append(1.0 if my_speed > opp_speed else 0.0)

    # ---- my 4 move slots: 38 features each (152 total) ----
    # pre-pass: find best damage for dmg_rank
    best_my_dmg_frac = 0.0
    my_dmg_fracs = []
    for i in range(4):
        if i < len(my_active.move_slots):
            mt = my_active.move_slots[i].template
            dmg = (calc_expected_damage(my_active, opp_active, mt, type_chart,
                   weather=weather, screens=opp_state.side)
                   / opp_active.max_hp) if opp_active.max_hp > 0 else 0.0
            my_dmg_fracs.append(dmg)
            if mt.power > 0:
                best_my_dmg_frac = max(best_my_dmg_frac, dmg)
        else:
            my_dmg_fracs.append(0.0)

    my_se_count = 0
    my_status_options = 0
    for i in range(4):
        if i < len(my_active.move_slots):
            slot = my_active.move_slots[i]
            mt = slot.template
            eff = type_chart.combined_effectiveness(mt.type, opp_active.types)
            if eff > 1.0 and mt.power > 0:
                my_se_count += 1
            pp_frac = slot.current_pp / mt.pp if mt.pp > 0 else 0.0
            dmg_frac = my_dmg_fracs[i]

            # setup move features
            boost_atk = 0.0
            boost_def = 0.0
            boost_spd = 0.0
            is_boost = 0.0
            if mt.name in _SELF_BOOST_MOVES:
                is_boost = 1.0
                for stat, stages, target in _SELF_BOOST_MOVES[mt.name]:
                    if target == "self":
                        if stat in ("attack", "special_attack"):
                            boost_atk = max(boost_atk, stages / 2.0)
                        elif stat in ("defense", "special_defense"):
                            boost_def = max(boost_def, stages / 2.0)
                        elif stat == "speed":
                            boost_spd = stages / 2.0
            elif mt.name in _HAZARD_MOVES or mt.name in _SCREEN_MOVES:
                is_boost = 0.5

            # debuff features
            is_debuff = 0.0
            debuff_magnitude = 0.0
            if mt.name in _OPP_DEBUFF_MOVES:
                is_debuff = 1.0
                for stat, stages, target in _OPP_DEBUFF_MOVES[mt.name]:
                    if target == "opponent" and stages < 0:
                        debuff_magnitude += abs(stages)
                debuff_magnitude = min(debuff_magnitude / 2.0, 1.0)

            ailment_id = _move_meta(mt, "ailment_id")
            ailment_type = AILMENT_ORDINAL.get(ailment_id, 0.0)
            high_crit = 1.0 if _move_meta(mt, "crit_rate") > 0 else 0.0
            multi_hit = _multi_hit_avg(mt)

            can_land = _ailment_can_land(mt, opp_active.types, opp_active.status)
            if can_land > 0.0:
                my_status_options += 1

            # type one-hot (17) + 19 scalars + high_crit + multi_hit + ailment_can_land = 39
            dmg_rank = (dmg_frac / best_my_dmg_frac) if (best_my_dmg_frac > 0 and dmg_frac > 0) else 0.0
            obs.extend(encode_move_type_onehot(mt.type))
            obs.extend([
                (mt.accuracy / 100.0) if mt.accuracy is not None else 1.0,
                ailment_type,
                eff / 4.0,
                pp_frac,
                1.0 if mt.priority > 0 else 0.0,
                DAMAGE_CLASS_ENC.get(mt.damage_class, 0.0),
                _move_meta(mt, "ailment_chance") / 100.0,
                _move_meta(mt, "flinch_chance") / 100.0,
                _move_meta(mt, "drain") / 100.0,
                _move_meta(mt, "healing") / 100.0,
                min(dmg_frac, 2.0),
                dmg_rank,
                is_boost,
                boost_atk,
                boost_def,
                boost_spd,
                is_debuff,
                debuff_magnitude,
                _multi_turn_cost(mt, weather),
                high_crit,
                multi_hit,
                can_land,
            ])
        else:
            obs.extend([0.0] * MY_MOVE_FEATURES)

    # ---- opponent's 4 moves: 36 features each (144 total) ----
    # pre-pass: find best damage for dmg_rank
    best_opp_dmg_frac = 0.0
    opp_dmg_fracs = []
    opp_predicted_move = None
    opp_predicted_dmg = 0.0
    for i in range(4):
        if i < len(opp_active.move_slots):
            mt = opp_active.move_slots[i].template
            dmg = (calc_expected_damage(opp_active, my_active, mt, type_chart,
                   weather=weather, screens=my_state.side)
                   / my_active.max_hp) if my_active.max_hp > 0 else 0.0
            opp_dmg_fracs.append(dmg)
            if mt.power > 0:
                best_opp_dmg_frac = max(best_opp_dmg_frac, dmg)
                raw_dmg = calc_expected_damage(opp_active, my_active, mt, type_chart,
                                              weather=weather, screens=my_state.side)
                if raw_dmg > opp_predicted_dmg:
                    opp_predicted_dmg = raw_dmg
                    opp_predicted_move = mt
        else:
            opp_dmg_fracs.append(0.0)

    opp_se_count = 0
    for i in range(4):
        if i < len(opp_active.move_slots):
            slot = opp_active.move_slots[i]
            mt = slot.template
            eff = type_chart.combined_effectiveness(mt.type, my_active.types)
            if eff > 1.0 and mt.power > 0:
                opp_se_count += 1
            dmg_frac = opp_dmg_fracs[i]
            pp_frac = slot.current_pp / mt.pp if mt.pp > 0 else 0.0
            ailment_id = _move_meta(mt, "ailment_id")

            high_crit = 1.0 if _move_meta(mt, "crit_rate") > 0 else 0.0

            # boost/debuff for opp moves
            boost_atk = 0.0
            boost_def = 0.0
            boost_spd = 0.0
            is_boost = 0.0
            if mt.name in _SELF_BOOST_MOVES:
                is_boost = 1.0
                for stat, stages, target in _SELF_BOOST_MOVES[mt.name]:
                    if target == "self":
                        if stat in ("attack", "special_attack"):
                            boost_atk = max(boost_atk, stages / 2.0)
                        elif stat in ("defense", "special_defense"):
                            boost_def = max(boost_def, stages / 2.0)
                        elif stat == "speed":
                            boost_spd = stages / 2.0
            elif mt.name in _HAZARD_MOVES or mt.name in _SCREEN_MOVES:
                is_boost = 0.5

            is_debuff = 0.0
            debuff_magnitude = 0.0
            if mt.name in _OPP_DEBUFF_MOVES:
                is_debuff = 1.0
                for stat, stages, target in _OPP_DEBUFF_MOVES[mt.name]:
                    if target == "opponent" and stages < 0:
                        debuff_magnitude += abs(stages)
                debuff_magnitude = min(debuff_magnitude / 2.0, 1.0)

            # type one-hot (17) + 18 scalars + ailment_can_land = 36
            dmg_rank = (dmg_frac / best_opp_dmg_frac) if (best_opp_dmg_frac > 0 and dmg_frac > 0) else 0.0
            obs.extend(encode_move_type_onehot(mt.type))
            obs.extend([
                eff / 4.0,
                min(dmg_frac, 2.0),
                dmg_rank,
                DAMAGE_CLASS_ENC.get(mt.damage_class, 0.0),
                1.0 if mt.priority > 0 else 0.0,
                AILMENT_ORDINAL.get(ailment_id, 0.0),
                pp_frac,
                high_crit,
                _move_meta(mt, "ailment_chance") / 100.0,
                _move_meta(mt, "flinch_chance") / 100.0,
                _move_meta(mt, "drain") / 100.0,
                _move_meta(mt, "healing") / 100.0,
                is_boost,
                boost_atk,
                boost_def,
                boost_spd,
                is_debuff,
                debuff_magnitude,
                _ailment_can_land(mt, my_active.types, my_active.status),
            ])
        else:
            obs.extend([0.0] * OPP_MOVE_FEATURES)

    # ---- damage summaries + KO/2HKO flags + survival (8) ----
    obs.append(min(best_my_dmg_frac, 2.0))
    obs.append(min(best_opp_dmg_frac, 2.0))
    obs.append(1.0 if best_my_dmg_frac >= opp_active.hp_frac else 0.0)
    obs.append(1.0 if best_opp_dmg_frac >= my_active.hp_frac else 0.0)
    obs.append(1.0 if best_my_dmg_frac * 2 >= opp_active.hp_frac else 0.0)
    obs.append(1.0 if best_opp_dmg_frac * 2 >= my_active.hp_frac else 0.0)
    my_survival = min(my_active.hp_frac / max(best_opp_dmg_frac, 0.001), 4.0) / 4.0
    opp_survival = min(opp_active.hp_frac / max(best_my_dmg_frac, 0.001), 4.0) / 4.0
    obs.append(my_survival)
    obs.append(opp_survival)

    # ---- active types: defensive profile (17 per mon, 34 total) ----
    # encodes "how effective is each attack type against me" instead of raw type identity
    obs.extend(encode_defensive_profile(my_active.types, type_chart))
    obs.extend(encode_defensive_profile(opp_active.types, type_chart))

    # ---- active status: one-hot (7 per mon, 14 total) ----
    obs.extend(encode_status_onehot(my_active.status))
    obs.extend(encode_status_onehot(opp_active.status))

    # ---- confusion turns + status turns + leech seed (6) ----
    obs.append(my_active.confusion_turns / 5.0)
    obs.append(opp_active.confusion_turns / 5.0)
    obs.append(my_active.status_turns / MAX_STATUS_TURNS)
    obs.append(opp_active.status_turns / MAX_STATUS_TURNS)
    obs.append(1.0 if my_active.leech_seeded else 0.0)
    obs.append(1.0 if opp_active.leech_seeded else 0.0)

    # ---- protect consecutive counter (2) ----
    obs.append(my_active.protect_consecutive / MAX_PROTECT)
    obs.append(opp_active.protect_consecutive / MAX_PROTECT)

    # ---- active base stats: atk/def/spa/spd normalized (8) ----
    obs.append(my_active.stats["attack"] / MAX_STAT)
    obs.append(my_active.stats["defense"] / MAX_STAT)
    obs.append(my_active.stats["special_attack"] / MAX_STAT)
    obs.append(my_active.stats["special_defense"] / MAX_STAT)
    obs.append(opp_active.stats["attack"] / MAX_STAT)
    obs.append(opp_active.stats["defense"] / MAX_STAT)
    obs.append(opp_active.stats["special_attack"] / MAX_STAT)
    obs.append(opp_active.stats["special_defense"] / MAX_STAT)

    # ---- status count summary (2) ----
    my_statused = sum(1 for p in my_state.team if p.status and not p.is_fainted)
    opp_statused = sum(1 for p in opp_state.team if p.status and not p.is_fainted)
    obs.append(my_statused / 6.0)
    obs.append(opp_statused / 6.0)

    # ---- active type matchup summary (4) ----
    # how well opp's STAB types hit my active, and vice versa
    my_types = my_active.types
    opp_types = opp_active.types
    # opp type 1 vs my types, opp type 2 vs my types
    obs.append(type_chart.combined_effectiveness(opp_types[0], my_types) / 4.0)
    obs.append(type_chart.combined_effectiveness(
        opp_types[1] if len(opp_types) > 1 else opp_types[0], my_types) / 4.0)
    # my type 1 vs opp types, my type 2 vs opp types
    obs.append(type_chart.combined_effectiveness(my_types[0], opp_types) / 4.0)
    obs.append(type_chart.combined_effectiveness(
        my_types[1] if len(my_types) > 1 else my_types[0], opp_types) / 4.0)

    # ---- consecutive turns active (2) ----
    obs.append(min(my_state.active_turns / 10.0, 1.0))
    obs.append(min(opp_state.active_turns / 10.0, 1.0))

    # ---- stat stages (14) ----
    stage_stats = ["attack", "defense", "special_attack", "special_defense",
                   "speed", "accuracy", "evasion"]
    for s in stage_stats:
        obs.append(my_active.stat_stages.get(s, 0) / 6.0)
    for s in stage_stats:
        obs.append(opp_active.stat_stages.get(s, 0) / 6.0)

    # ---- matchup summary features (6) ----
    obs.append(my_se_count / 4.0)          # how many of my moves are SE vs opponent
    obs.append(opp_se_count / 4.0)         # how many of opp's moves are SE vs me
    obs.append(1.0 if best_my_dmg_frac < 0.10 else 0.0)   # i am walled
    obs.append(1.0 if best_opp_dmg_frac < 0.10 else 0.0)  # opponent is walled
    obs.append(my_status_options / 4.0)    # count of status moves that can land
    # best_bench_advantage computed below after team loop, placeholder here
    bench_advantage_idx = len(obs)
    obs.append(0.0)

    # ---- boost value: can KO after boosting + boosted damage frac (2) ----
    from engine.stat_stages import get_stage_multiplier
    boosted_best_dmg = best_my_dmg_frac
    for slot in my_active.move_slots:
        mt = slot.template
        if not slot.has_pp or mt.name not in _SELF_BOOST_MOVES:
            continue
        best_mult = 1.0
        for stat, stages, target in _SELF_BOOST_MOVES[mt.name]:
            if target != "self":
                continue
            if stat in ("attack", "special_attack"):
                current_stage = my_active.stat_stages.get(stat, 0)
                new_stage = min(6, current_stage + stages)
                if new_stage > current_stage:
                    cur_num, cur_den = get_stage_multiplier(current_stage)
                    new_num, new_den = get_stage_multiplier(new_stage)
                    mult = (new_num * cur_den) / (new_den * cur_num)
                    best_mult = max(best_mult, mult)
        boosted_best_dmg = max(boosted_best_dmg, best_my_dmg_frac * best_mult)
    can_ko_after_boost = 1.0 if boosted_best_dmg >= opp_active.hp_frac and boosted_best_dmg > best_my_dmg_frac else 0.0
    obs.append(can_ko_after_boost)
    obs.append(min(boosted_best_dmg, 2.0))  # boosted damage frac for context

    # ============================================================
    # MY TEAM: 6 fixed slots x 61 = 366
    # ============================================================
    # per slot: is_active, alive, hp_frac, defensive_profile(17), dmg_vs_opp,
    #   opp_dmg_vs_me, speed, status_onehot(7), pp_frac, predicted_dmg_vs_me,
    #   can_ko_opp, survives_predicted, outspeeds_opp, can_2hko_opp,
    #   hits_to_survive, defensive_eff, best_move_eff, spikes_entry_dmg,
    #   switch_in_cost, matchup_improvement, base_stats(4), move_type_coverage(17)
    best_bench_eff = 0.0  # best effectiveness among bench mons (for switch advantage)
    active_best_eff = 0.0  # active mon's best effectiveness (computed below)

    # pre-compute active matchup quality for per-slot switch comparison
    active_off_eff = 0.0
    active_def_eff = 0.0
    for slot in my_active.move_slots:
        if slot.template.power > 0:
            eff = type_chart.combined_effectiveness(slot.template.type, opp_active.types)
            active_off_eff = max(active_off_eff, eff)
    for slot in opp_active.move_slots:
        if slot.template.power > 0:
            eff = type_chart.combined_effectiveness(slot.template.type, my_active.types)
            active_def_eff = max(active_def_eff, eff)
    for idx in range(6):
        if idx >= len(my_state.team):
            obs.extend([0.0] * MY_TEAM_PER)
            continue
        p = my_state.team[idx]
        is_active = 1.0 if idx == my_state.active_index else 0.0
        alive = 0.0 if p.is_fainted else 1.0
        hp_frac = p.hp_frac

        best_dmg_frac_vs_opp = 0.0
        best_move_eff_vs_opp = 0.0
        opp_best_dmg_frac_vs_me = 0.0
        predicted_dmg_frac_vs_me = 0.0
        best_opp_eff_vs_me = 0.0
        total_pp = 0
        total_max_pp = 0
        move_coverage = [0.0] * NUM_TYPES
        if not p.is_fainted:
            for slot in p.move_slots:
                total_pp += slot.current_pp
                total_max_pp += slot.template.pp
                # move type coverage
                tidx = TYPE_TO_IDX.get(slot.template.type)
                if tidx is not None and slot.template.power > 0:
                    move_coverage[tidx] = 1.0
                if slot.template.power > 0:
                    d = calc_expected_damage(p, opp_active, slot.template, type_chart,
                                            weather=weather, screens=opp_state.side)
                    if opp_active.max_hp > 0:
                        best_dmg_frac_vs_opp = max(best_dmg_frac_vs_opp, d / opp_active.max_hp)
                    eff = type_chart.combined_effectiveness(slot.template.type, opp_active.types)
                    best_move_eff_vs_opp = max(best_move_eff_vs_opp, eff)
            for slot in opp_active.move_slots:
                if slot.template.power > 0:
                    d = calc_expected_damage(opp_active, p, slot.template, type_chart,
                                            weather=weather, screens=my_state.side)
                    if p.max_hp > 0:
                        opp_best_dmg_frac_vs_me = max(opp_best_dmg_frac_vs_me, d / p.max_hp)
                    eff = type_chart.combined_effectiveness(slot.template.type, p.types)
                    best_opp_eff_vs_me = max(best_opp_eff_vs_me, eff)

            if opp_predicted_move is not None and p.max_hp > 0:
                d = calc_expected_damage(opp_active, p, opp_predicted_move, type_chart,
                                         weather=weather, screens=my_state.side)
                predicted_dmg_frac_vs_me = d / p.max_hp

        pp_frac = total_pp / total_max_pp if total_max_pp > 0 else 0.0

        # track best bench effectiveness for switch advantage feature
        if not p.is_fainted:
            if idx == my_state.active_index:
                active_best_eff = best_move_eff_vs_opp
            else:
                best_bench_eff = max(best_bench_eff, best_move_eff_vs_opp)

        can_ko_opp = 1.0 if best_dmg_frac_vs_opp >= opp_active.hp_frac else 0.0
        survives_predicted = 1.0 if predicted_dmg_frac_vs_me < hp_frac else 0.0
        outspeeds_opp = 1.0 if effective_speed(p) > opp_speed else 0.0
        can_2hko_opp = 1.0 if best_dmg_frac_vs_opp * 2 >= opp_active.hp_frac else 0.0
        hits_to_survive = min(hp_frac / max(opp_best_dmg_frac_vs_me, 0.001), 4.0) / 4.0

        # spikes entry damage: 1/8 HP for grounded mons, 0 for flying
        spikes_entry = 0.0
        if opp_state.side.spikes and not p.is_fainted and "flying" not in p.types:
            spikes_entry = 0.125  # max_hp // 8

        # switch-in cost: total HP fraction lost on switch turn (opp hit + spikes)
        switch_in_cost = min(opp_best_dmg_frac_vs_me + spikes_entry, 1.5)
        # matchup improvement: how much switching here improves type matchup vs staying
        if is_active or p.is_fainted:
            matchup_improvement = 0.0
        else:
            matchup_improvement = max((
                (best_move_eff_vs_opp - best_opp_eff_vs_me)
                - (active_off_eff - active_def_eff)
            ) / 4.0, -1.0)

        obs.extend([
            is_active,
            alive,
            hp_frac,
            *encode_defensive_profile(p.types, type_chart),
            min(best_dmg_frac_vs_opp, 2.0),
            min(opp_best_dmg_frac_vs_me, 2.0),
            effective_speed(p) / MAX_SPEED,
            *encode_status_onehot(p.status),
            pp_frac,
            min(predicted_dmg_frac_vs_me, 2.0),
            can_ko_opp,
            survives_predicted,
            outspeeds_opp,
            can_2hko_opp,
            hits_to_survive,
            best_opp_eff_vs_me / 4.0,
            best_move_eff_vs_opp / 4.0,
            spikes_entry,
            switch_in_cost,
            matchup_improvement,
            p.stats["attack"] / MAX_STAT,
            p.stats["defense"] / MAX_STAT,
            p.stats["special_attack"] / MAX_STAT,
            p.stats["special_defense"] / MAX_STAT,
            *move_coverage,
        ])

    # ============================================================
    # OPPONENT TEAM: 6 fixed slots x 42 = 252
    # ============================================================
    # per slot: is_active, alive, hp_frac, defensive_profile(17), dmg_vs_my_active,
    #   my_dmg_vs_them, speed, status_onehot(7), pp_frac,
    #   can_ko_my_active, survives_my_best, outspeeds_my_active,
    #   can_2hko_my_active, hits_to_survive_my_best,
    #   base_stats(4), my_best_eff_vs_them, move_type_count
    for idx in range(6):
        if idx >= len(opp_state.team):
            obs.extend([0.0] * OPP_TEAM_PER)
            continue
        p = opp_state.team[idx]
        is_active = 1.0 if idx == opp_state.active_index else 0.0
        alive = 0.0 if p.is_fainted else 1.0
        hp_frac = p.hp_frac

        best_dmg_frac_vs_my_active = 0.0
        my_best_dmg_frac_vs_them = 0.0
        my_best_eff_vs_them = 0.0
        total_pp = 0
        total_max_pp = 0
        move_types_seen = set()
        if not p.is_fainted:
            for slot in p.move_slots:
                total_pp += slot.current_pp
                total_max_pp += slot.template.pp
                if slot.template.power > 0:
                    d = calc_expected_damage(p, my_active, slot.template, type_chart,
                                            weather=weather, screens=my_state.side)
                    if my_active.max_hp > 0:
                        best_dmg_frac_vs_my_active = max(best_dmg_frac_vs_my_active, d / my_active.max_hp)
                    move_types_seen.add(slot.template.type)
            for slot in my_active.move_slots:
                if slot.template.power > 0:
                    d = calc_expected_damage(my_active, p, slot.template, type_chart,
                                            weather=weather, screens=opp_state.side)
                    if p.max_hp > 0:
                        my_best_dmg_frac_vs_them = max(my_best_dmg_frac_vs_them, d / p.max_hp)
                    eff = type_chart.combined_effectiveness(slot.template.type, p.types)
                    my_best_eff_vs_them = max(my_best_eff_vs_them, eff)

        pp_frac = total_pp / total_max_pp if total_max_pp > 0 else 0.0

        can_ko_my_active = 1.0 if best_dmg_frac_vs_my_active >= my_active.hp_frac else 0.0
        survives_my_best = 1.0 if my_best_dmg_frac_vs_them < hp_frac else 0.0
        outspeeds_my_active = 1.0 if effective_speed(p) > my_speed else 0.0
        can_2hko_my_active = 1.0 if best_dmg_frac_vs_my_active * 2 >= my_active.hp_frac else 0.0
        hits_to_survive = min(hp_frac / max(my_best_dmg_frac_vs_them, 0.001), 4.0) / 4.0

        obs.extend([
            is_active,
            alive,
            hp_frac,
            *encode_defensive_profile(p.types, type_chart),
            min(best_dmg_frac_vs_my_active, 2.0),
            min(my_best_dmg_frac_vs_them, 2.0),
            effective_speed(p) / MAX_SPEED,
            *encode_status_onehot(p.status),
            pp_frac,
            can_ko_my_active,
            survives_my_best,
            outspeeds_my_active,
            can_2hko_my_active,
            hits_to_survive,
            p.stats["attack"] / MAX_STAT,
            p.stats["defense"] / MAX_STAT,
            p.stats["special_attack"] / MAX_STAT,
            p.stats["special_defense"] / MAX_STAT,
            my_best_eff_vs_them / 4.0,
            len(move_types_seen) / 4.0,
        ])

    # backfill bench switch advantage (needed team loop to compute)
    obs[bench_advantage_idx] = max((best_bench_eff - active_best_eff) / 4.0, -0.5)

    # ============================================================
    # GLOBAL (15)
    # ============================================================
    obs.append(turn / MAX_TURNS)
    obs.append(my_state.alive_count / len(my_state.team))
    obs.append(opp_state.alive_count / len(opp_state.team))
    obs.append(my_state.total_hp_frac)
    obs.append(opp_state.total_hp_frac)

    # side conditions
    obs.append(1.0 if my_state.side.spikes else 0.0)
    obs.append(1.0 if opp_state.side.spikes else 0.0)
    obs.append(my_state.side.reflect_turns / MAX_SCREEN_TURNS)
    obs.append(opp_state.side.reflect_turns / MAX_SCREEN_TURNS)
    obs.append(my_state.side.light_screen_turns / MAX_SCREEN_TURNS)
    obs.append(opp_state.side.light_screen_turns / MAX_SCREEN_TURNS)

    # weather (3 binary flags + turns)
    weather_vec = [0.0, 0.0, 0.0]
    if weather in WEATHER_INDEX:
        weather_vec[WEATHER_INDEX[weather]] = 1.0
    obs.extend(weather_vec)
    obs.append(weather_turns / MAX_WEATHER_TURNS)

    # pad to OBS_SIZE
    padding = OBS_SIZE - len(obs)
    assert padding >= 0, f"obs too large: {len(obs)} > {OBS_SIZE}"
    if padding > 0:
        obs.extend([0.0] * padding)
    arr = np.array(obs, dtype=np.float32)
    return arr
