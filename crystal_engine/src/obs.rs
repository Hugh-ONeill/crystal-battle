// state -> f32 observation vector (1052 features)
// layout: active(~407) + my_team(6x63=378) + opp_team(6x42=252) + global(15)

use crate::battle::BattleState;
use crate::damage::{calc_expected_damage, Weather};
use crate::moves::DamageClass;
use crate::stat_stages::{
    get_stage_multiplier, move_stat_effects, STAT_ATK, STAT_DEF, STAT_SPA, STAT_SPD, STAT_SPE,
};
use crate::status::{effective_speed, Status};
use crate::types::{combined_effectiveness, Type, NUM_TYPES};

// ============================================================
// CONSTANTS
// ============================================================

pub const OBS_SIZE: usize = 1052;

const MAX_SPEED: f32 = 500.0;
const MAX_STAT: f32 = 500.0;
const MAX_TURNS: f32 = 200.0;
const MAX_STATUS_TURNS: f32 = 16.0;
const MAX_SCREEN_TURNS: f32 = 5.0;
const MAX_WEATHER_TURNS: f32 = 5.0;
const MAX_PROTECT: f32 = 4.0;

const MY_MOVE_FEATURES: usize = 39;
const OPP_MOVE_FEATURES: usize = 36;
const MY_TEAM_PER: usize = 63;
const OPP_TEAM_PER: usize = 42;
const N_STATUS: usize = 7;

// all 17 types in enum order (Bug=0 .. Water=16)
const ALL_TYPES: [Type; NUM_TYPES] = [
    Type::Bug,
    Type::Dark,
    Type::Dragon,
    Type::Electric,
    Type::Fighting,
    Type::Fire,
    Type::Flying,
    Type::Ghost,
    Type::Grass,
    Type::Ground,
    Type::Ice,
    Type::Normal,
    Type::Poison,
    Type::Psychic,
    Type::Rock,
    Type::Steel,
    Type::Water,
];

// ailment_id -> ordinal encoding
fn ailment_ordinal(id: u8) -> f32 {
    match id {
        1 => 0.28, // paralysis
        2 => 0.42, // sleep
        3 => 0.57, // freeze
        4 => 0.14, // burn
        5 => 0.71, // poison
        6 => 0.85, // confusion
        _ => 0.0,
    }
}

// ailment_id -> Status variant for type immunity checks
fn ailment_to_status(id: u8) -> Option<Status> {
    match id {
        1 => Some(Status::Paralysis),
        2 => Some(Status::Sleep),
        3 => Some(Status::Freeze),
        4 => Some(Status::Burn),
        5 => Some(Status::Poison),
        _ => None,
    }
}

// multi-turn move IDs
const SOLAR_BEAM: u16 = 76;
const CHARGE_DODGE: [u16; 2] = [19, 91]; // Fly, Dig
const CHARGE_EXPOSED: [u16; 3] = [13, 130, 143]; // Razor Wind, Skull Bash, Sky Attack
const RECHARGE_MOVES: [u16; 1] = [63]; // Hyper Beam
const LOCKIN_MOVES: [u16; 3] = [37, 80, 200]; // Thrash, Petal Dance, Outrage

// ============================================================
// HELPERS
// ============================================================

fn status_index(s: Status) -> usize {
    match s {
        Status::None => 0,
        Status::Burn => 1,
        Status::Paralysis => 2,
        Status::Sleep => 3,
        Status::Freeze => 4,
        Status::Poison => 5,
        Status::Toxic => 6,
    }
}

fn damage_class_enc(dc: DamageClass) -> f32 {
    match dc {
        DamageClass::Physical => 1.0,
        DamageClass::Special => 0.5,
        DamageClass::Status => 0.0,
    }
}

fn encode_status_onehot(obs: &mut Vec<f32>, s: Status) {
    let idx = status_index(s);
    for i in 0..N_STATUS {
        obs.push(if i == idx { 1.0 } else { 0.0 });
    }
}

fn encode_defensive_profile(obs: &mut Vec<f32>, types: &[Type]) {
    for &atk_type in &ALL_TYPES {
        obs.push(combined_effectiveness(atk_type, types) / 4.0);
    }
}

fn encode_move_type_onehot(obs: &mut Vec<f32>, move_type: Type) {
    let idx = move_type as usize;
    for i in 0..NUM_TYPES {
        obs.push(if i == idx { 1.0 } else { 0.0 });
    }
}

/// Get move meta field with overrides for Rest and Belly Drum.
/// Returns (ailment_id, ailment_chance, drain_frac, healing_frac, crit_rate, flinch_chance)
/// as raw values (not yet /100).
struct MoveMetas {
    ailment_id: u8,
    ailment_chance: f32,
    drain: f32,
    healing: f32,
    crit_rate: u8,
    flinch_chance: f32,
}

fn get_move_metas(name: &str, meta: &crate::moves::MoveMeta) -> MoveMetas {
    match name {
        "Rest" => MoveMetas {
            ailment_id: meta.ailment_id,
            ailment_chance: -100.0, // override: signals self-sleep
            drain: meta.drain as f32,
            healing: 100.0,
            crit_rate: meta.crit_rate,
            flinch_chance: meta.flinch_chance as f32,
        },
        "Belly Drum" => MoveMetas {
            ailment_id: meta.ailment_id,
            ailment_chance: meta.ailment_chance as f32,
            drain: meta.drain as f32,
            healing: -50.0, // costs 50% HP
            crit_rate: meta.crit_rate,
            flinch_chance: meta.flinch_chance as f32,
        },
        _ => MoveMetas {
            ailment_id: meta.ailment_id,
            ailment_chance: meta.ailment_chance as f32,
            drain: meta.drain as f32,
            healing: meta.healing as f32,
            crit_rate: meta.crit_rate,
            flinch_chance: meta.flinch_chance as f32,
        },
    }
}

fn multi_hit_avg(meta: &crate::moves::MoveMeta) -> f32 {
    match (meta.min_hits, meta.max_hits) {
        (Some(min_h), Some(max_h)) if max_h > 1 => {
            (min_h as f32 + max_h as f32) / 2.0 / 5.0
        }
        _ => 0.0,
    }
}

fn ailment_can_land(
    name: &str,
    ailment_id: u8,
    target_types: &[Type],
    target_status: Status,
) -> f32 {
    if ailment_id == 0 {
        return 0.0;
    }
    // confusion -- no type immunity
    if ailment_id == 6 {
        return 1.0;
    }
    let mut status = match ailment_to_status(ailment_id) {
        Some(s) => s,
        None => return 0.0,
    };
    // Toxic applies Toxic, not Poison
    if ailment_id == 5 && name == "Toxic" {
        status = Status::Toxic;
    }
    // already has a non-volatile status
    if target_status != Status::None {
        return 0.0;
    }
    // type immunity checks
    for &ptype in target_types {
        let immune = match status {
            Status::Burn => ptype == Type::Fire,
            Status::Paralysis => ptype == Type::Electric,
            Status::Poison | Status::Toxic => ptype == Type::Poison || ptype == Type::Steel,
            Status::Freeze => ptype == Type::Ice,
            _ => false,
        };
        if immune {
            return 0.0;
        }
    }
    1.0
}

fn multi_turn_cost(id: u16, weather: Option<Weather>) -> f32 {
    if id == SOLAR_BEAM {
        return if weather == Some(Weather::Sun) { 0.0 } else { 0.5 };
    }
    if CHARGE_DODGE.contains(&id) {
        return 0.25;
    }
    if CHARGE_EXPOSED.contains(&id) {
        return 0.5;
    }
    if RECHARGE_MOVES.contains(&id) {
        return 0.75;
    }
    if LOCKIN_MOVES.contains(&id) {
        return 1.0;
    }
    0.0
}

/// Check if a move is a self-boost (any effect with targets_self=true and stages>0).
fn is_self_boost(name: &str) -> bool {
    if let Some(effects) = move_stat_effects(name) {
        effects.iter().any(|&(_, stages, targets_self)| targets_self && stages > 0)
    } else {
        false
    }
}

/// Check if a move is an opponent debuff (any effect with targets_self=false and stages<0).
fn is_opp_debuff(name: &str) -> bool {
    if let Some(effects) = move_stat_effects(name) {
        effects.iter().any(|&(_, stages, targets_self)| !targets_self && stages < 0)
    } else {
        false
    }
}

fn is_hazard_move(name: &str) -> bool {
    name == "Spikes"
}

fn is_screen_move(name: &str) -> bool {
    matches!(name, "Reflect" | "Light Screen" | "Safeguard")
}

// ============================================================
// BUILD OBSERVATION
// ============================================================

pub fn build_observation(state: &BattleState) -> [f32; OBS_SIZE] {
    let mut obs: Vec<f32> = Vec::with_capacity(OBS_SIZE);

    let my_state = &state.p1;
    let opp_state = &state.p2;
    let my_active = my_state.active();
    let opp_active = opp_state.active();
    let weather = state.weather;

    // ============================================================
    // ACTIVE MATCHUP
    // ============================================================

    // ---- hp and speed (5) ----
    let my_speed = effective_speed(my_active) as f32;
    let opp_speed = effective_speed(opp_active) as f32;
    obs.push(my_active.hp_frac());
    obs.push(my_speed / MAX_SPEED);
    obs.push(opp_active.hp_frac());
    obs.push(opp_speed / MAX_SPEED);
    obs.push(if my_speed > opp_speed { 1.0 } else { 0.0 });

    // ---- my 4 move slots: 39 features each (156 total) ----
    // pre-pass: find best damage for dmg_rank
    let mut best_my_dmg_frac: f32 = 0.0;
    let mut my_dmg_fracs = [0.0f32; 4];
    for i in 0..4 {
        if i < my_active.move_slots.len() {
            let mt = &my_active.move_slots[i].template;
            let dmg = if opp_active.max_hp() > 0 {
                calc_expected_damage(
                    my_active, opp_active, mt,
                    Some(&opp_state.side), weather,
                ) / opp_active.max_hp() as f32
            } else {
                0.0
            };
            my_dmg_fracs[i] = dmg;
            if mt.power > 0 {
                best_my_dmg_frac = best_my_dmg_frac.max(dmg);
            }
        }
    }

    let mut my_se_count: u32 = 0;
    let mut my_status_options: u32 = 0;
    for i in 0..4 {
        if i < my_active.move_slots.len() {
            let slot = &my_active.move_slots[i];
            let mt = &slot.template;
            let eff = combined_effectiveness(mt.move_type, opp_active.types());
            if eff > 1.0 && mt.power > 0 {
                my_se_count += 1;
            }
            let pp_frac = if mt.pp > 0 {
                slot.current_pp as f32 / mt.pp as f32
            } else {
                0.0
            };
            let dmg_frac = my_dmg_fracs[i];

            // setup move features (boost)
            let mut boost_atk: f32 = 0.0;
            let mut boost_def: f32 = 0.0;
            let mut boost_spd: f32 = 0.0;
            let mut is_boost: f32 = 0.0;

            if is_self_boost(&mt.name) {
                is_boost = 1.0;
                if let Some(effects) = move_stat_effects(&mt.name) {
                    for &(stat_idx, stages, targets_self) in effects {
                        if targets_self {
                            let s = stages as f32 / 2.0;
                            match stat_idx {
                                STAT_ATK | STAT_SPA => boost_atk = boost_atk.max(s),
                                STAT_DEF | STAT_SPD => boost_def = boost_def.max(s),
                                STAT_SPE => boost_spd = s,
                                _ => {}
                            }
                        }
                    }
                }
            } else if is_hazard_move(&mt.name) || is_screen_move(&mt.name) {
                is_boost = 0.5;
            }

            // debuff features
            let mut is_debuff: f32 = 0.0;
            let mut debuff_magnitude: f32 = 0.0;
            if is_opp_debuff(&mt.name) {
                is_debuff = 1.0;
                if let Some(effects) = move_stat_effects(&mt.name) {
                    for &(_, stages, targets_self) in effects {
                        if !targets_self && stages < 0 {
                            debuff_magnitude += (-stages) as f32;
                        }
                    }
                }
                debuff_magnitude = (debuff_magnitude / 2.0).min(1.0);
            }

            let metas = get_move_metas(&mt.name, &mt.meta);
            let ailment_type = ailment_ordinal(metas.ailment_id);
            let high_crit = if metas.crit_rate > 0 { 1.0 } else { 0.0 };
            let multi_hit = multi_hit_avg(&mt.meta);

            let can_land = ailment_can_land(
                &mt.name,
                metas.ailment_id,
                opp_active.types(),
                opp_active.status,
            );
            if can_land > 0.0 {
                my_status_options += 1;
            }

            let dmg_rank = if best_my_dmg_frac > 0.0 && dmg_frac > 0.0 {
                dmg_frac / best_my_dmg_frac
            } else {
                0.0
            };

            // type one-hot (17) + 22 scalars = 39
            encode_move_type_onehot(&mut obs, mt.move_type);
            obs.push(if let Some(acc) = mt.accuracy { acc as f32 / 100.0 } else { 1.0 });
            obs.push(ailment_type);
            obs.push(eff / 4.0);
            obs.push(pp_frac);
            obs.push(if mt.priority > 0 { 1.0 } else { 0.0 });
            obs.push(damage_class_enc(mt.damage_class));
            obs.push(metas.ailment_chance / 100.0);
            obs.push(metas.flinch_chance / 100.0);
            obs.push(metas.drain / 100.0);
            obs.push(metas.healing / 100.0);
            obs.push(dmg_frac.min(2.0));
            obs.push(dmg_rank);
            obs.push(is_boost);
            obs.push(boost_atk);
            obs.push(boost_def);
            obs.push(boost_spd);
            obs.push(is_debuff);
            obs.push(debuff_magnitude);
            obs.push(multi_turn_cost(mt.id, weather));
            obs.push(high_crit);
            obs.push(multi_hit);
            obs.push(can_land);
        } else {
            obs.extend_from_slice(&[0.0; MY_MOVE_FEATURES]);
        }
    }

    // ---- opponent's 4 moves: 36 features each (144 total) ----
    let mut best_opp_dmg_frac: f32 = 0.0;
    let mut opp_dmg_fracs = [0.0f32; 4];
    let mut opp_predicted_move: Option<&crate::moves::MoveTemplate> = None;
    let mut opp_predicted_dmg: f32 = 0.0;
    for i in 0..4 {
        if i < opp_active.move_slots.len() {
            let mt = &opp_active.move_slots[i].template;
            let dmg = if my_active.max_hp() > 0 {
                calc_expected_damage(
                    opp_active, my_active, mt,
                    Some(&my_state.side), weather,
                ) / my_active.max_hp() as f32
            } else {
                0.0
            };
            opp_dmg_fracs[i] = dmg;
            if mt.power > 0 {
                best_opp_dmg_frac = best_opp_dmg_frac.max(dmg);
                let raw_dmg = calc_expected_damage(
                    opp_active, my_active, mt,
                    Some(&my_state.side), weather,
                );
                if raw_dmg > opp_predicted_dmg {
                    opp_predicted_dmg = raw_dmg;
                    opp_predicted_move = Some(mt);
                }
            }
        }
    }

    let mut opp_se_count: u32 = 0;
    for i in 0..4 {
        if i < opp_active.move_slots.len() {
            let slot = &opp_active.move_slots[i];
            let mt = &slot.template;
            let eff = combined_effectiveness(mt.move_type, my_active.types());
            if eff > 1.0 && mt.power > 0 {
                opp_se_count += 1;
            }
            let dmg_frac = opp_dmg_fracs[i];
            let pp_frac = if mt.pp > 0 {
                slot.current_pp as f32 / mt.pp as f32
            } else {
                0.0
            };
            let metas = get_move_metas(&mt.name, &mt.meta);
            let high_crit = if metas.crit_rate > 0 { 1.0 } else { 0.0 };

            // boost/debuff for opp moves
            let mut boost_atk: f32 = 0.0;
            let mut boost_def: f32 = 0.0;
            let mut boost_spd: f32 = 0.0;
            let mut is_boost: f32 = 0.0;

            if is_self_boost(&mt.name) {
                is_boost = 1.0;
                if let Some(effects) = move_stat_effects(&mt.name) {
                    for &(stat_idx, stages, targets_self) in effects {
                        if targets_self {
                            let s = stages as f32 / 2.0;
                            match stat_idx {
                                STAT_ATK | STAT_SPA => boost_atk = boost_atk.max(s),
                                STAT_DEF | STAT_SPD => boost_def = boost_def.max(s),
                                STAT_SPE => boost_spd = s,
                                _ => {}
                            }
                        }
                    }
                }
            } else if is_hazard_move(&mt.name) || is_screen_move(&mt.name) {
                is_boost = 0.5;
            }

            let mut is_debuff: f32 = 0.0;
            let mut debuff_magnitude: f32 = 0.0;
            if is_opp_debuff(&mt.name) {
                is_debuff = 1.0;
                if let Some(effects) = move_stat_effects(&mt.name) {
                    for &(_, stages, targets_self) in effects {
                        if !targets_self && stages < 0 {
                            debuff_magnitude += (-stages) as f32;
                        }
                    }
                }
                debuff_magnitude = (debuff_magnitude / 2.0).min(1.0);
            }

            let dmg_rank = if best_opp_dmg_frac > 0.0 && dmg_frac > 0.0 {
                dmg_frac / best_opp_dmg_frac
            } else {
                0.0
            };

            // type one-hot (17) + 19 scalars = 36
            encode_move_type_onehot(&mut obs, mt.move_type);
            obs.push(eff / 4.0);
            obs.push(dmg_frac.min(2.0));
            obs.push(dmg_rank);
            obs.push(damage_class_enc(mt.damage_class));
            obs.push(if mt.priority > 0 { 1.0 } else { 0.0 });
            obs.push(ailment_ordinal(metas.ailment_id));
            obs.push(pp_frac);
            obs.push(high_crit);
            obs.push(metas.ailment_chance / 100.0);
            obs.push(metas.flinch_chance / 100.0);
            obs.push(metas.drain / 100.0);
            obs.push(metas.healing / 100.0);
            obs.push(is_boost);
            obs.push(boost_atk);
            obs.push(boost_def);
            obs.push(boost_spd);
            obs.push(is_debuff);
            obs.push(debuff_magnitude);
            obs.push(ailment_can_land(
                &mt.name,
                metas.ailment_id,
                my_active.types(),
                my_active.status,
            ));
        } else {
            obs.extend_from_slice(&[0.0; OPP_MOVE_FEATURES]);
        }
    }

    // ---- damage summaries + KO/2HKO flags + survival (8) ----
    obs.push(best_my_dmg_frac.min(2.0));
    obs.push(best_opp_dmg_frac.min(2.0));
    obs.push(if best_my_dmg_frac >= opp_active.hp_frac() { 1.0 } else { 0.0 });
    obs.push(if best_opp_dmg_frac >= my_active.hp_frac() { 1.0 } else { 0.0 });
    obs.push(if best_my_dmg_frac * 2.0 >= opp_active.hp_frac() { 1.0 } else { 0.0 });
    obs.push(if best_opp_dmg_frac * 2.0 >= my_active.hp_frac() { 1.0 } else { 0.0 });
    let my_survival = (my_active.hp_frac() / best_opp_dmg_frac.max(0.001)).min(4.0) / 4.0;
    let opp_survival = (opp_active.hp_frac() / best_my_dmg_frac.max(0.001)).min(4.0) / 4.0;
    obs.push(my_survival);
    obs.push(opp_survival);

    // ---- active types: defensive profile (17 per mon, 34 total) ----
    encode_defensive_profile(&mut obs, my_active.types());
    encode_defensive_profile(&mut obs, opp_active.types());

    // ---- active status: one-hot (7 per mon, 14 total) ----
    encode_status_onehot(&mut obs, my_active.status);
    encode_status_onehot(&mut obs, opp_active.status);

    // ---- confusion turns + status turns + leech seed (6) ----
    obs.push(my_active.confusion_turns as f32 / 5.0);
    obs.push(opp_active.confusion_turns as f32 / 5.0);
    obs.push(my_active.status_turns as f32 / MAX_STATUS_TURNS);
    obs.push(opp_active.status_turns as f32 / MAX_STATUS_TURNS);
    obs.push(if my_active.leech_seeded { 1.0 } else { 0.0 });
    obs.push(if opp_active.leech_seeded { 1.0 } else { 0.0 });

    // ---- protect consecutive counter (2) ----
    obs.push(my_active.protect_consecutive as f32 / MAX_PROTECT);
    obs.push(opp_active.protect_consecutive as f32 / MAX_PROTECT);

    // ---- active base stats: atk/def/spa/spd normalized (8) ----
    obs.push(my_active.stats[1] as f32 / MAX_STAT); // atk
    obs.push(my_active.stats[2] as f32 / MAX_STAT); // def
    obs.push(my_active.stats[3] as f32 / MAX_STAT); // spa
    obs.push(my_active.stats[4] as f32 / MAX_STAT); // spd
    obs.push(opp_active.stats[1] as f32 / MAX_STAT);
    obs.push(opp_active.stats[2] as f32 / MAX_STAT);
    obs.push(opp_active.stats[3] as f32 / MAX_STAT);
    obs.push(opp_active.stats[4] as f32 / MAX_STAT);

    // ---- status count summary (2) ----
    let my_statused = my_state.team.iter()
        .filter(|p| p.status != Status::None && !p.is_fainted())
        .count() as f32;
    let opp_statused = opp_state.team.iter()
        .filter(|p| p.status != Status::None && !p.is_fainted())
        .count() as f32;
    obs.push(my_statused / 6.0);
    obs.push(opp_statused / 6.0);

    // ---- active type matchup summary (4) ----
    let my_types = my_active.types();
    let opp_types = opp_active.types();
    obs.push(combined_effectiveness(opp_types[0], my_types) / 4.0);
    obs.push(combined_effectiveness(
        if opp_types.len() > 1 { opp_types[1] } else { opp_types[0] },
        my_types,
    ) / 4.0);
    obs.push(combined_effectiveness(my_types[0], opp_types) / 4.0);
    obs.push(combined_effectiveness(
        if my_types.len() > 1 { my_types[1] } else { my_types[0] },
        opp_types,
    ) / 4.0);

    // ---- consecutive turns active (2) ----
    obs.push((my_state.active_turns as f32 / 10.0).min(1.0));
    obs.push((opp_state.active_turns as f32 / 10.0).min(1.0));

    // ---- stat stages (14) ----
    for i in 0..7 {
        obs.push(my_active.stat_stages[i] as f32 / 6.0);
    }
    for i in 0..7 {
        obs.push(opp_active.stat_stages[i] as f32 / 6.0);
    }

    // ---- matchup summary features (6) ----
    obs.push(my_se_count as f32 / 4.0);
    obs.push(opp_se_count as f32 / 4.0);
    obs.push(if best_my_dmg_frac < 0.10 { 1.0 } else { 0.0 });
    obs.push(if best_opp_dmg_frac < 0.10 { 1.0 } else { 0.0 });
    obs.push(my_status_options as f32 / 4.0);
    // best_bench_advantage -- placeholder, filled after team loop
    let bench_advantage_idx = obs.len();
    obs.push(0.0);

    // ---- boost value: can KO after boosting + boosted damage frac (2) ----
    let mut boosted_best_dmg = best_my_dmg_frac;
    for slot in &my_active.move_slots {
        let mt = &slot.template;
        if !slot.has_pp() || !is_self_boost(&mt.name) {
            continue;
        }
        let mut best_mult: f32 = 1.0;
        if let Some(effects) = move_stat_effects(&mt.name) {
            for &(stat_idx, stages, targets_self) in effects {
                if !targets_self {
                    continue;
                }
                if stat_idx == STAT_ATK || stat_idx == STAT_SPA {
                    let current_stage = my_active.stat_stages[stat_idx];
                    let new_stage = (current_stage + stages).min(6);
                    if new_stage > current_stage {
                        let (cur_num, cur_den) = get_stage_multiplier(current_stage);
                        let (new_num, new_den) = get_stage_multiplier(new_stage);
                        let mult = (new_num as f32 * cur_den as f32)
                            / (new_den as f32 * cur_num as f32);
                        best_mult = best_mult.max(mult);
                    }
                }
            }
        }
        boosted_best_dmg = boosted_best_dmg.max(best_my_dmg_frac * best_mult);
    }
    let can_ko_after_boost = if boosted_best_dmg >= opp_active.hp_frac()
        && boosted_best_dmg > best_my_dmg_frac
    {
        1.0
    } else {
        0.0
    };
    obs.push(can_ko_after_boost);
    obs.push(boosted_best_dmg.min(2.0));

    // ============================================================
    // MY TEAM: 6 fixed slots x 63 = 378
    // ============================================================
    let mut best_bench_eff: f32 = 0.0;
    let mut active_best_eff: f32 = 0.0;

    // pre-compute active matchup quality for per-slot switch comparison
    let mut active_off_eff: f32 = 0.0;
    let mut active_def_eff: f32 = 0.0;
    for slot in &my_active.move_slots {
        if slot.template.power > 0 {
            let eff = combined_effectiveness(slot.template.move_type, opp_active.types());
            active_off_eff = active_off_eff.max(eff);
        }
    }
    for slot in &opp_active.move_slots {
        if slot.template.power > 0 {
            let eff = combined_effectiveness(slot.template.move_type, my_active.types());
            active_def_eff = active_def_eff.max(eff);
        }
    }

    for idx in 0..6 {
        if idx >= my_state.team.len() {
            obs.extend_from_slice(&[0.0; MY_TEAM_PER]);
            continue;
        }
        let p = &my_state.team[idx];
        let is_active = if idx == my_state.active_index as usize { 1.0 } else { 0.0 };
        let alive = if p.is_fainted() { 0.0 } else { 1.0 };
        let hp_frac = p.hp_frac();

        let mut best_dmg_frac_vs_opp: f32 = 0.0;
        let mut best_move_eff_vs_opp: f32 = 0.0;
        let mut opp_best_dmg_frac_vs_me: f32 = 0.0;
        let mut predicted_dmg_frac_vs_me: f32 = 0.0;
        let mut best_opp_eff_vs_me: f32 = 0.0;
        let mut total_pp: u32 = 0;
        let mut total_max_pp: u32 = 0;
        let mut move_coverage = [0.0f32; NUM_TYPES];

        if !p.is_fainted() {
            for slot in &p.move_slots {
                total_pp += slot.current_pp as u32;
                total_max_pp += slot.template.pp as u32;
                // move type coverage
                if slot.template.power > 0 {
                    let tidx = slot.template.move_type as usize;
                    move_coverage[tidx] = 1.0;
                }
                if slot.template.power > 0 {
                    let d = calc_expected_damage(
                        p, opp_active, &slot.template,
                        Some(&opp_state.side), weather,
                    );
                    if opp_active.max_hp() > 0 {
                        best_dmg_frac_vs_opp = best_dmg_frac_vs_opp
                            .max(d / opp_active.max_hp() as f32);
                    }
                    let eff = combined_effectiveness(slot.template.move_type, opp_active.types());
                    best_move_eff_vs_opp = best_move_eff_vs_opp.max(eff);
                }
            }
            for slot in &opp_active.move_slots {
                if slot.template.power > 0 {
                    let d = calc_expected_damage(
                        opp_active, p, &slot.template,
                        Some(&my_state.side), weather,
                    );
                    if p.max_hp() > 0 {
                        opp_best_dmg_frac_vs_me = opp_best_dmg_frac_vs_me
                            .max(d / p.max_hp() as f32);
                    }
                    let eff = combined_effectiveness(slot.template.move_type, p.types());
                    best_opp_eff_vs_me = best_opp_eff_vs_me.max(eff);
                }
            }

            if let Some(pred_mt) = opp_predicted_move {
                if p.max_hp() > 0 {
                    let d = calc_expected_damage(
                        opp_active, p, pred_mt,
                        Some(&my_state.side), weather,
                    );
                    predicted_dmg_frac_vs_me = d / p.max_hp() as f32;
                }
            }
        }

        let pp_frac = if total_max_pp > 0 {
            total_pp as f32 / total_max_pp as f32
        } else {
            0.0
        };

        // track best bench effectiveness for switch advantage feature
        if !p.is_fainted() {
            if idx == my_state.active_index as usize {
                active_best_eff = best_move_eff_vs_opp;
            } else {
                best_bench_eff = best_bench_eff.max(best_move_eff_vs_opp);
            }
        }

        let can_ko_opp = if best_dmg_frac_vs_opp >= opp_active.hp_frac() { 1.0 } else { 0.0 };
        let survives_predicted = if predicted_dmg_frac_vs_me < hp_frac { 1.0 } else { 0.0 };
        let outspeeds_opp = if (effective_speed(p) as f32) > opp_speed { 1.0 } else { 0.0 };
        let can_2hko_opp = if best_dmg_frac_vs_opp * 2.0 >= opp_active.hp_frac() { 1.0 } else { 0.0 };
        let hits_to_survive = (hp_frac / opp_best_dmg_frac_vs_me.max(0.001)).min(4.0) / 4.0;

        // spikes entry damage
        let spikes_entry = if opp_state.side.spikes
            && !p.is_fainted()
            && !p.has_type(Type::Flying)
        {
            0.125
        } else {
            0.0
        };

        // switch-in cost
        let switch_in_cost = (opp_best_dmg_frac_vs_me + spikes_entry).min(1.5);

        // matchup improvement
        let matchup_improvement = if is_active == 1.0 || p.is_fainted() {
            0.0
        } else {
            (((best_move_eff_vs_opp - best_opp_eff_vs_me)
                - (active_off_eff - active_def_eff))
                / 4.0)
                .max(-1.0)
        };

        obs.push(is_active);
        obs.push(alive);
        obs.push(hp_frac);
        encode_defensive_profile(&mut obs, p.types());
        obs.push(best_dmg_frac_vs_opp.min(2.0));
        obs.push(opp_best_dmg_frac_vs_me.min(2.0));
        obs.push(effective_speed(p) as f32 / MAX_SPEED);
        encode_status_onehot(&mut obs, p.status);
        obs.push(pp_frac);
        obs.push(predicted_dmg_frac_vs_me.min(2.0));
        obs.push(can_ko_opp);
        obs.push(survives_predicted);
        obs.push(outspeeds_opp);
        obs.push(can_2hko_opp);
        obs.push(hits_to_survive);
        obs.push(best_opp_eff_vs_me / 4.0);
        obs.push(best_move_eff_vs_opp / 4.0);
        obs.push(spikes_entry);
        obs.push(switch_in_cost);
        obs.push(matchup_improvement);
        obs.push(p.stats[1] as f32 / MAX_STAT); // atk
        obs.push(p.stats[2] as f32 / MAX_STAT); // def
        obs.push(p.stats[3] as f32 / MAX_STAT); // spa
        obs.push(p.stats[4] as f32 / MAX_STAT); // spd
        for &c in &move_coverage {
            obs.push(c);
        }
    }

    // ============================================================
    // OPPONENT TEAM: 6 fixed slots x 42 = 252
    // ============================================================
    for idx in 0..6 {
        if idx >= opp_state.team.len() {
            obs.extend_from_slice(&[0.0; OPP_TEAM_PER]);
            continue;
        }
        let p = &opp_state.team[idx];
        let is_active = if idx == opp_state.active_index as usize { 1.0 } else { 0.0 };
        let alive = if p.is_fainted() { 0.0 } else { 1.0 };
        let hp_frac = p.hp_frac();

        let mut best_dmg_frac_vs_my_active: f32 = 0.0;
        let mut my_best_dmg_frac_vs_them: f32 = 0.0;
        let mut my_best_eff_vs_them: f32 = 0.0;
        let mut total_pp: u32 = 0;
        let mut total_max_pp: u32 = 0;
        let mut move_types_seen = [false; NUM_TYPES];

        if !p.is_fainted() {
            for slot in &p.move_slots {
                total_pp += slot.current_pp as u32;
                total_max_pp += slot.template.pp as u32;
                if slot.template.power > 0 {
                    let d = calc_expected_damage(
                        p, my_active, &slot.template,
                        Some(&my_state.side), weather,
                    );
                    if my_active.max_hp() > 0 {
                        best_dmg_frac_vs_my_active = best_dmg_frac_vs_my_active
                            .max(d / my_active.max_hp() as f32);
                    }
                    move_types_seen[slot.template.move_type as usize] = true;
                }
            }
            for slot in &my_active.move_slots {
                if slot.template.power > 0 {
                    let d = calc_expected_damage(
                        my_active, p, &slot.template,
                        Some(&opp_state.side), weather,
                    );
                    if p.max_hp() > 0 {
                        my_best_dmg_frac_vs_them = my_best_dmg_frac_vs_them
                            .max(d / p.max_hp() as f32);
                    }
                    let eff = combined_effectiveness(slot.template.move_type, p.types());
                    my_best_eff_vs_them = my_best_eff_vs_them.max(eff);
                }
            }
        }

        let pp_frac = if total_max_pp > 0 {
            total_pp as f32 / total_max_pp as f32
        } else {
            0.0
        };

        let can_ko_my_active = if best_dmg_frac_vs_my_active >= my_active.hp_frac() { 1.0 } else { 0.0 };
        let survives_my_best = if my_best_dmg_frac_vs_them < hp_frac { 1.0 } else { 0.0 };
        let outspeeds_my_active = if (effective_speed(p) as f32) > my_speed { 1.0 } else { 0.0 };
        let can_2hko_my_active = if best_dmg_frac_vs_my_active * 2.0 >= my_active.hp_frac() { 1.0 } else { 0.0 };
        let hits_to_survive = (hp_frac / my_best_dmg_frac_vs_them.max(0.001)).min(4.0) / 4.0;
        let move_type_count = move_types_seen.iter().filter(|&&b| b).count() as f32;

        obs.push(is_active);
        obs.push(alive);
        obs.push(hp_frac);
        encode_defensive_profile(&mut obs, p.types());
        obs.push(best_dmg_frac_vs_my_active.min(2.0));
        obs.push(my_best_dmg_frac_vs_them.min(2.0));
        obs.push(effective_speed(p) as f32 / MAX_SPEED);
        encode_status_onehot(&mut obs, p.status);
        obs.push(pp_frac);
        obs.push(can_ko_my_active);
        obs.push(survives_my_best);
        obs.push(outspeeds_my_active);
        obs.push(can_2hko_my_active);
        obs.push(hits_to_survive);
        obs.push(p.stats[1] as f32 / MAX_STAT); // atk
        obs.push(p.stats[2] as f32 / MAX_STAT); // def
        obs.push(p.stats[3] as f32 / MAX_STAT); // spa
        obs.push(p.stats[4] as f32 / MAX_STAT); // spd
        obs.push(my_best_eff_vs_them / 4.0);
        obs.push(move_type_count / 4.0);
    }

    // backfill bench switch advantage
    obs[bench_advantage_idx] = ((best_bench_eff - active_best_eff) / 4.0).max(-0.5);

    // ============================================================
    // GLOBAL (15)
    // ============================================================
    obs.push(state.turn as f32 / MAX_TURNS);
    obs.push(my_state.alive_count() as f32 / my_state.team.len() as f32);
    obs.push(opp_state.alive_count() as f32 / opp_state.team.len() as f32);
    obs.push(my_state.total_hp_frac());
    obs.push(opp_state.total_hp_frac());

    // side conditions
    obs.push(if my_state.side.spikes { 1.0 } else { 0.0 });
    obs.push(if opp_state.side.spikes { 1.0 } else { 0.0 });
    obs.push(my_state.side.reflect_turns as f32 / MAX_SCREEN_TURNS);
    obs.push(opp_state.side.reflect_turns as f32 / MAX_SCREEN_TURNS);
    obs.push(my_state.side.light_screen_turns as f32 / MAX_SCREEN_TURNS);
    obs.push(opp_state.side.light_screen_turns as f32 / MAX_SCREEN_TURNS);

    // weather (3 binary flags + turns)
    let (w_sun, w_rain, w_sand) = match weather {
        Some(Weather::Sun) => (1.0, 0.0, 0.0),
        Some(Weather::Rain) => (0.0, 1.0, 0.0),
        Some(Weather::Sandstorm) => (0.0, 0.0, 1.0),
        None => (0.0, 0.0, 0.0),
    };
    obs.push(w_sun);
    obs.push(w_rain);
    obs.push(w_sand);
    obs.push(state.weather_turns as f32 / MAX_WEATHER_TURNS);

    // pad to OBS_SIZE
    debug_assert!(
        obs.len() <= OBS_SIZE,
        "obs too large: {} > {}",
        obs.len(),
        OBS_SIZE
    );
    obs.resize(OBS_SIZE, 0.0);

    let mut out = [0.0f32; OBS_SIZE];
    out.copy_from_slice(&obs);
    out
}
