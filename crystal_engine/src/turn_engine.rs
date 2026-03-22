// resolve_turn(): order actions, execute, handle faints + status effects
//
// Borrow strategy: BattleState has fields p1, p2, rng, weather, etc.
// Rust allows simultaneous mutable borrows of DISTINCT fields.
// So we access state.p1/state.p2/state.rng directly to avoid conflicts.
// Helper fn `ps_mut` returns &mut PlayerState from the state by number,
// but for functions needing both a player AND rng, we access fields directly.

use rand::Rng;

use crate::actions::Action;
use crate::battle::{BattleState, WEATHER_DURATION};
use crate::damage::{calc_damage, Weather};
use crate::events::Event;
use crate::moves::{self, DamageClass, MoveTemplate};
use crate::player::PlayerState;
use crate::pokemon::{Pokemon, SemiInvuln};
use crate::stat_stages::{self, apply_stat_change, get_stage_multiplier, move_stat_effects};
use crate::status::{
    apply_confusion, apply_status, can_apply_status, check_confusion,
    check_move_prevention, confusion_from_move, effective_speed, end_of_turn_damage,
    status_from_move, MovePreventionResult, Status,
};
use crate::types::Type;

// ============================================================
// SPECIAL MOVE SETS
// ============================================================

const HYPER_BEAM: u16 = 63;
const FLY: u16 = 19;
const DIG: u16 = 91;
const RAZOR_WIND: u16 = 13;
const SKULL_BASH: u16 = 130;
const SKY_ATTACK: u16 = 143;
const SOLAR_BEAM: u16 = 76;
const THRASH: u16 = 37;
const PETAL_DANCE: u16 = 80;
const OUTRAGE: u16 = 200;

const SCREEN_DURATION: u8 = 5;

fn is_special_move(name: &str) -> bool {
    matches!(
        name,
        "Spikes" | "Reflect" | "Light Screen" | "Protect" | "Detect"
            | "Roar" | "Whirlwind" | "Leech Seed" | "Haze" | "Rest"
            | "Rapid Spin" | "Sunny Day" | "Rain Dance" | "Sandstorm"
    )
}

fn is_recharge_move(id: u16) -> bool { id == HYPER_BEAM }

fn is_charge_invuln(id: u16) -> Option<SemiInvuln> {
    match id {
        FLY => Some(SemiInvuln::Fly),
        DIG => Some(SemiInvuln::Dig),
        _ => None,
    }
}

fn is_charge_exposed(id: u16) -> bool { matches!(id, RAZOR_WIND | SKULL_BASH | SKY_ATTACK) }
fn is_lockin_move(id: u16) -> bool { matches!(id, THRASH | PETAL_DANCE | OUTRAGE) }

fn weather_from_move(name: &str) -> Option<Weather> {
    match name {
        "Sunny Day" => Some(Weather::Sun),
        "Rain Dance" => Some(Weather::Rain),
        "Sandstorm" => Some(Weather::Sandstorm),
        _ => None,
    }
}

fn is_sandstorm_immune(t: Type) -> bool { matches!(t, Type::Rock | Type::Ground | Type::Steel) }

/// Get mutable player state ref. Only use when you DON'T also need &mut rng.
fn ps_mut(state: &mut BattleState, num: u8) -> &mut PlayerState {
    if num == 1 { &mut state.p1 } else { &mut state.p2 }
}

fn ps_ref(state: &BattleState, num: u8) -> &PlayerState {
    if num == 1 { &state.p1 } else { &state.p2 }
}

// ============================================================
// RESOLVE TURN
// ============================================================

pub fn resolve_turn(
    state: &mut BattleState,
    action1: Action,
    action2: Action,
) -> Vec<Event> {
    state.turn += 1;
    let mut events = Vec::new();

    // clear per-turn volatile state
    state.p1.active_mut().flinched = false;
    state.p2.active_mut().flinched = false;
    state.p1.active_mut().protected = false;
    state.p2.active_mut().protected = false;
    state.p1.active_mut().semi_invulnerable = None;
    state.p2.active_mut().semi_invulnerable = None;

    if action1 == Action::Forfeit { state.winner = Some(2); return events; }
    if action2 == Action::Forfeit { state.winner = Some(1); return events; }

    let order = determine_order(state, action1, action2);

    for &(pnum, action) in &order {
        if state.is_over() { break; }
        if ps_ref(state, pnum).active().is_fainted() { continue; }

        let dnum = if pnum == 1 { 2u8 } else { 1u8 };
        let new_events = execute_action(state, pnum, dnum, action);
        events.extend(new_events);

        if !state.is_over() && !ps_ref(state, pnum).active().is_fainted() {
            let res = apply_residual_damage(state, pnum, dnum);
            events.extend(res);
        }
    }

    if !state.is_over() {
        events.extend(end_of_turn(state));
    }

    state.check_winner();
    state.p1.active_turns += 1;
    state.p2.active_turns += 1;
    events
}

// ============================================================
// ORDER
// ============================================================

fn determine_order(state: &mut BattleState, a1: Action, a2: Action) -> Vec<(u8, Action)> {
    let s1 = a1.is_switch();
    let s2 = a2.is_switch();
    if s1 && !s2 { return vec![(1, a1), (2, a2)]; }
    if s2 && !s1 { return vec![(2, a2), (1, a1)]; }
    if s1 && s2 { return speed_order(state, a1, a2); }

    let p1_pri = get_priority(&state.p1, a1);
    let p2_pri = get_priority(&state.p2, a2);
    if p1_pri > p2_pri { return vec![(1, a1), (2, a2)]; }
    if p2_pri > p1_pri { return vec![(2, a2), (1, a1)]; }
    speed_order(state, a1, a2)
}

fn get_priority(player: &PlayerState, action: Action) -> i8 {
    match action {
        Action::UseMove(slot) => player.active().move_slots[slot as usize].template.priority,
        _ => 0,
    }
}

fn speed_order(state: &mut BattleState, a1: Action, a2: Action) -> Vec<(u8, Action)> {
    let sp1 = effective_speed(state.p1.active());
    let sp2 = effective_speed(state.p2.active());
    if sp1 > sp2 { vec![(1, a1), (2, a2)] }
    else if sp2 > sp1 { vec![(2, a2), (1, a1)] }
    else if state.rng.random_bool(0.5) { vec![(1, a1), (2, a2)] }
    else { vec![(2, a2), (1, a1)] }
}

// ============================================================
// ACTION EXECUTION
// ============================================================

fn execute_action(state: &mut BattleState, pnum: u8, dnum: u8, action: Action) -> Vec<Event> {
    let mut events = Vec::new();

    // ---- Switch ----
    if let Action::Switch(team_idx) = action {
        let ps = ps_mut(state, pnum);
        let old_name = ps.switch_to(team_idx);
        let to_name = ps.active().name.clone();
        events.push(Event::Switch { player: pnum, from_name: old_name, to_name });
        apply_spikes_damage(ps_mut(state, pnum), pnum, &mut events);
        return events;
    }

    // get move template
    let is_struggle = action == Action::Struggle;
    let mv_template = if is_struggle {
        moves::struggle()
    } else if let Action::UseMove(slot_idx) = action {
        let ps = ps_mut(state, pnum);
        ps.active_mut().move_slots[slot_idx as usize].use_pp();
        ps.active().move_slots[slot_idx as usize].template.clone()
    } else {
        return events;
    };
    let mut mv = mv_template;

    // ---- Recharging ----
    if ps_ref(state, pnum).active().recharging {
        ps_mut(state, pnum).active_mut().recharging = false;
        reset_protect_consecutive(ps_mut(state, pnum).active_mut(), &mv);
        return events;
    }

    // ---- Locked-in move override ----
    if let Some(locked_id) = ps_ref(state, pnum).active().locked_move_id {
        if let Some(si) = ps_ref(state, pnum).active().find_move_slot(locked_id) {
            mv = ps_ref(state, pnum).active().move_slots[si].template.clone();
        }
        ps_mut(state, pnum).active_mut().locked_turns -= 1;
    }

    // ---- Charging move -- execute on turn 2 ----
    if let Some(charge_id) = ps_ref(state, pnum).active().charging_move_id {
        if let Some(si) = ps_ref(state, pnum).active().find_move_slot(charge_id) {
            mv = ps_ref(state, pnum).active().move_slots[si].template.clone();
        }
        ps_mut(state, pnum).active_mut().charging_move_id = None;
        return exec_damaging(state, pnum, dnum, &mv, is_struggle, &mut events);
    }

    // ---- Flinch ----
    if ps_ref(state, pnum).active().flinched {
        let name = ps_ref(state, pnum).active().name.clone();
        events.push(Event::Flinch { player: pnum, pokemon_name: name });
        reset_protect_consecutive(ps_mut(state, pnum).active_mut(), &mv);
        break_lock_in(ps_mut(state, pnum).active_mut());
        return events;
    }

    // ---- Status prevention ----
    // Access p1/p2 directly to allow simultaneous rng borrow
    let prevention = {
        let (active, rng) = get_active_and_rng(state, pnum);
        check_move_prevention(active, rng)
    };
    match prevention {
        MovePreventionResult::WokeUp => {
            let name = ps_ref(state, pnum).active().name.clone();
            events.push(Event::StatusCured { player: pnum, pokemon_name: name, status: "slp".into() });
        }
        MovePreventionResult::ThawedOut => {
            let name = ps_ref(state, pnum).active().name.clone();
            events.push(Event::StatusCured { player: pnum, pokemon_name: name, status: "frz".into() });
        }
        MovePreventionResult::Prevented(reason) => {
            let a = ps_ref(state, pnum).active();
            let name = a.name.clone();
            let st = a.status.as_str().to_string();
            events.push(Event::StatusPrevented { player: pnum, pokemon_name: name, status: st, reason: reason.into() });
            reset_protect_consecutive(ps_mut(state, pnum).active_mut(), &mv);
            break_lock_in(ps_mut(state, pnum).active_mut());
            return events;
        }
        MovePreventionResult::CanAct => {}
    }

    // ---- Confusion ----
    let (hit_self, self_dmg) = {
        let (active, rng) = get_active_and_rng(state, pnum);
        check_confusion(active, rng)
    };
    if hit_self {
        let actual = ps_mut(state, pnum).active_mut().take_damage(self_dmg);
        let name = ps_ref(state, pnum).active().name.clone();
        events.push(Event::ConfusionHitSelf { player: pnum, pokemon_name: name.clone(), damage: actual });
        if ps_ref(state, pnum).active().is_fainted() {
            events.push(Event::Faint { player: pnum, pokemon_name: name });
        }
        reset_protect_consecutive(ps_mut(state, pnum).active_mut(), &mv);
        break_lock_in(ps_mut(state, pnum).active_mut());
        return events;
    }

    // ---- Charge turn initiation ----
    let mid = mv.id;
    let not_locked = ps_ref(state, pnum).active().locked_move_id.is_none();
    if not_locked {
        if mid == SOLAR_BEAM && state.weather != Some(Weather::Sun) {
            ps_mut(state, pnum).active_mut().charging_move_id = Some(mid);
            return events;
        }
        if let Some(invuln) = is_charge_invuln(mid) {
            let a = ps_mut(state, pnum).active_mut();
            a.charging_move_id = Some(mid);
            a.semi_invulnerable = Some(invuln);
            return events;
        }
        if is_charge_exposed(mid) {
            ps_mut(state, pnum).active_mut().charging_move_id = Some(mid);
            return events;
        }
    }

    // ---- Lock-in initiation ----
    if is_lockin_move(mid) && ps_ref(state, pnum).active().locked_move_id.is_none() {
        let turns = state.rng.random_range(1..=2u8);
        let a = ps_mut(state, pnum).active_mut();
        a.locked_move_id = Some(mid);
        a.locked_turns = turns;
    }

    // ---- Status moves ----
    if mv.damage_class == DamageClass::Status {
        let applied = handle_status_move(state, pnum, dnum, &mv, &mut events);
        if !applied {
            let name = ps_ref(state, pnum).active().name.clone();
            events.push(Event::StatusMove { player: pnum, pokemon_name: name, move_name: mv.name.clone() });
        }
        return events;
    }

    // ---- Protect blocks ----
    if ps_ref(state, dnum).active().protected {
        let name = ps_ref(state, pnum).active().name.clone();
        events.push(Event::Miss { player: pnum, pokemon_name: name, move_name: mv.name.clone() });
        return events;
    }

    // ---- Accuracy check ----
    if let Some(accuracy) = mv.accuracy {
        let eff_acc = calc_effective_accuracy(accuracy, ps_ref(state, pnum).active(), ps_ref(state, dnum).active());
        let roll = state.rng.random_range(1..=100u32);
        if roll > eff_acc as u32 {
            let name = ps_ref(state, pnum).active().name.clone();
            events.push(Event::Miss { player: pnum, pokemon_name: name, move_name: mv.name.clone() });
            return events;
        }
    }

    exec_damaging(state, pnum, dnum, &mv, is_struggle, &mut events)
}

/// Get mutable references to active pokemon AND rng simultaneously.
/// This works because we borrow distinct struct fields (p1/p2 vs rng).
fn get_active_and_rng(state: &mut BattleState, pnum: u8) -> (&mut Pokemon, &mut rand::rngs::SmallRng) {
    if pnum == 1 {
        (state.p1.active_mut(), &mut state.rng)
    } else {
        (state.p2.active_mut(), &mut state.rng)
    }
}


// ============================================================
// DAMAGING MOVE
// ============================================================

fn exec_damaging(
    state: &mut BattleState,
    pnum: u8, dnum: u8,
    mv: &MoveTemplate,
    is_struggle: bool,
    events: &mut Vec<Event>,
) -> Vec<Event> {
    // semi-invulnerable dodge
    if let Some(invuln) = ps_ref(state, dnum).active().semi_invulnerable {
        let can_hit = match invuln {
            SemiInvuln::Dig => matches!(mv.name.as_str(), "Earthquake" | "Magnitude"),
            SemiInvuln::Fly => matches!(mv.name.as_str(), "Thunder" | "Gust" | "Twister"),
        };
        if !can_hit {
            let name = ps_ref(state, pnum).active().name.clone();
            events.push(Event::Miss { player: pnum, pokemon_name: name, move_name: mv.name.clone() });
            return std::mem::take(events);
        }
    }

    let num_hits = roll_hit_count(&mv.meta, &mut state.rng);

    let mut total_damage: u32 = 0;
    let mut any_crit = false;
    let mut last_eff = 1.0f32;

    for _ in 0..num_hits {
        if ps_ref(state, dnum).active().is_fainted() { break; }

        let screens = ps_ref(state, dnum).side.clone();
        let weather = state.weather;

        // Borrow distinct fields for calc_damage: p1, p2, rng
        let result = if pnum == 1 {
            calc_damage(state.p1.active(), state.p2.active(), mv, &mut state.rng, Some(&screens), weather)
        } else {
            calc_damage(state.p2.active(), state.p1.active(), mv, &mut state.rng, Some(&screens), weather)
        };

        let actual = ps_mut(state, dnum).active_mut().take_damage(result.damage);
        total_damage += actual as u32;
        if result.is_crit { any_crit = true; }
        last_eff = result.effectiveness;
    }

    // emit event
    if is_struggle {
        let name = ps_ref(state, pnum).active().name.clone();
        events.push(Event::Struggle { player: pnum, pokemon_name: name, damage: total_damage as u16 });
    } else {
        let aname = ps_ref(state, pnum).active().name.clone();
        let dhp = ps_ref(state, dnum).active().current_hp;
        events.push(Event::Move {
            player: pnum, pokemon_name: aname, move_name: mv.name.clone(),
            damage: total_damage as u16, effectiveness: last_eff, is_crit: any_crit,
            target_hp_remaining: dhp,
        });
    }

    // drain / recoil
    let drain_pct = mv.meta.drain;
    if drain_pct != 0 && total_damage > 0 && !is_struggle {
        let drain_amt = ((total_damage * drain_pct.unsigned_abs() as u32) / 100).max(1) as u16;
        if drain_pct > 0 {
            let healed = ps_mut(state, pnum).active_mut().heal(drain_amt);
            if healed > 0 {
                let name = ps_ref(state, pnum).active().name.clone();
                events.push(Event::Heal { player: pnum, pokemon_name: name, amount: healed, source: "drain".into() });
            }
        } else {
            ps_mut(state, pnum).active_mut().take_damage(drain_amt);
            if ps_ref(state, pnum).active().is_fainted() {
                let name = ps_ref(state, pnum).active().name.clone();
                events.push(Event::Faint { player: pnum, pokemon_name: name });
            }
        }
    }

    // defender faint
    if ps_ref(state, dnum).active().is_fainted() {
        let name = ps_ref(state, dnum).active().name.clone();
        events.push(Event::Faint { player: dnum, pokemon_name: name });
    }

    // struggle recoil
    if is_struggle && total_damage > 0 {
        let recoil = (total_damage / 4).max(1) as u16;
        ps_mut(state, pnum).active_mut().take_damage(recoil);
        if ps_ref(state, pnum).active().is_fainted() {
            let name = ps_ref(state, pnum).active().name.clone();
            events.push(Event::Faint { player: pnum, pokemon_name: name });
        }
    }

    // Rapid Spin
    if mv.name == "Rapid Spin" && total_damage > 0 {
        let ps = ps_mut(state, pnum);
        ps.side.spikes = false;
        ps.active_mut().leech_seeded = false;
    }

    // flinch
    if !ps_ref(state, dnum).active().is_fainted() && total_damage > 0 && mv.meta.flinch_chance > 0 {
        let roll = state.rng.random_range(1..=100u32);
        if roll <= mv.meta.flinch_chance as u32 {
            ps_mut(state, dnum).active_mut().flinched = true;
        }
    }

    // secondary effects
    if !ps_ref(state, dnum).active().is_fainted() && total_damage > 0 {
        try_secondary_effect(state, pnum, dnum, mv, events);
    }

    // recharge
    if is_recharge_move(mv.id) && total_damage > 0 {
        ps_mut(state, pnum).active_mut().recharging = true;
    }

    // lock-in end -> confusion
    let active = ps_ref(state, pnum).active();
    if active.locked_move_id.is_some() && active.locked_turns == 0 {
        {
            let a = ps_mut(state, pnum).active_mut();
            a.locked_move_id = None;
            a.locked_turns = 0;
        }
        let (active, rng) = get_active_and_rng(state, pnum);
        let confused = apply_confusion(active, rng);
        if confused {
            let name = ps_ref(state, pnum).active().name.clone();
            events.push(Event::ConfusionApplied { player: pnum, pokemon_name: name });
        }
    }

    std::mem::take(events)
}

// ============================================================
// HELPERS
// ============================================================

fn reset_protect_consecutive(pokemon: &mut Pokemon, mv: &MoveTemplate) {
    if mv.name != "Protect" && mv.name != "Detect" { pokemon.protect_consecutive = 0; }
}

fn break_lock_in(pokemon: &mut Pokemon) {
    if pokemon.locked_move_id.is_some() { pokemon.locked_move_id = None; pokemon.locked_turns = 0; }
}

fn calc_effective_accuracy(base: u8, attacker: &Pokemon, defender: &Pokemon) -> u8 {
    let net = (attacker.stat_stages[5] - defender.stat_stages[6]).clamp(-6, 6);
    if net == 0 { return base; }
    let (num, den) = get_stage_multiplier(net);
    (base as i32 * num / den).clamp(1, 100) as u8
}

fn roll_hit_count(meta: &crate::moves::MoveMeta, rng: &mut impl Rng) -> u8 {
    match (meta.min_hits, meta.max_hits) {
        (Some(min), Some(max)) if min == max => min,
        (Some(_), Some(_)) => match rng.random_range(1..=8u32) {
            1..=3 => 2, 4..=6 => 3, 7 => 4, _ => 5,
        },
        _ => 1,
    }
}

fn apply_spikes_damage(ps: &mut PlayerState, pnum: u8, events: &mut Vec<Event>) {
    if !ps.side.spikes { return; }
    if ps.active().has_type(Type::Flying) { return; }
    let max_hp = ps.active().max_hp();
    let dmg = (max_hp / 8).max(1);
    let actual = ps.active_mut().take_damage(dmg);
    let name = ps.active().name.clone();
    events.push(Event::SpikesDamage { player: pnum, pokemon_name: name.clone(), damage: actual });
    if ps.active().is_fainted() {
        events.push(Event::Faint { player: pnum, pokemon_name: name });
    }
}

// ============================================================
// STATUS MOVE HANDLING
// ============================================================

fn handle_status_move(
    state: &mut BattleState, pnum: u8, dnum: u8, mv: &MoveTemplate, events: &mut Vec<Event>,
) -> bool {
    let meta = &mv.meta;
    let ailment_id = meta.ailment_id;
    let stat_effects = move_stat_effects(&mv.name);
    let is_self = is_self_target_move(&mv.name, stat_effects, meta);

    // protect blocks
    if ps_ref(state, dnum).active().protected && !is_self {
        let name = ps_ref(state, pnum).active().name.clone();
        events.push(Event::Miss { player: pnum, pokemon_name: name, move_name: mv.name.clone() });
        return true;
    }

    if is_special_move(&mv.name) {
        return handle_special_move(state, pnum, dnum, mv, events);
    }

    // stat-change moves
    if let Some(effects) = stat_effects {
        if ailment_id == 0 {
            if let Some(acc) = mv.accuracy {
                let roll = state.rng.random_range(1..=100u32);
                if roll > acc as u32 {
                    let name = ps_ref(state, pnum).active().name.clone();
                    events.push(Event::Miss { player: pnum, pokemon_name: name, move_name: mv.name.clone() });
                    return true;
                }
            }
            if mv.name == "Belly Drum" {
                let a = ps_ref(state, pnum).active();
                let cost = a.max_hp() / 2;
                if a.current_hp <= cost {
                    let name = a.name.clone();
                    events.push(Event::StatusMove { player: pnum, pokemon_name: name, move_name: mv.name.clone() });
                    return true;
                }
                ps_mut(state, pnum).active_mut().take_damage(cost);
            }
            do_stat_effects(state, effects, pnum, dnum, events);
            return true;
        }
    }

    // healing
    if meta.healing > 0 && ailment_id == 0 {
        let a = ps_ref(state, pnum).active();
        let heal = a.max_hp() as u32 * meta.healing as u32 / 100;
        let actual = ps_mut(state, pnum).active_mut().heal(heal as u16);
        if actual > 0 {
            let name = ps_ref(state, pnum).active().name.clone();
            events.push(Event::Heal { player: pnum, pokemon_name: name, amount: actual, source: "recover".into() });
        } else {
            let name = ps_ref(state, pnum).active().name.clone();
            events.push(Event::StatusMove { player: pnum, pokemon_name: name, move_name: mv.name.clone() });
        }
        return true;
    }

    // accuracy for other status moves
    if let Some(acc) = mv.accuracy {
        let roll = state.rng.random_range(1..=100u32);
        if roll > acc as u32 {
            let name = ps_ref(state, pnum).active().name.clone();
            events.push(Event::Miss { player: pnum, pokemon_name: name, move_name: mv.name.clone() });
            return true;
        }
    }

    // confusion
    if confusion_from_move(ailment_id) {
        let (def_active, rng) = get_active_and_rng(state, dnum);
        let applied = apply_confusion(def_active, rng);
        if applied {
            let name = ps_ref(state, dnum).active().name.clone();
            events.push(Event::ConfusionApplied { player: dnum, pokemon_name: name });
        } else {
            let name = ps_ref(state, pnum).active().name.clone();
            events.push(Event::StatusMove { player: pnum, pokemon_name: name, move_name: mv.name.clone() });
        }
        return true;
    }

    // non-volatile status
    if let Some(status) = status_from_move(&mv.name, ailment_id) {
        let (can, _) = can_apply_status(ps_ref(state, dnum).active(), status);
        if can {
            let (def_active, rng) = get_active_and_rng(state, dnum);
            apply_status(def_active, status, rng);
            let name = ps_ref(state, dnum).active().name.clone();
            events.push(Event::StatusApplied { player: dnum, pokemon_name: name, status: status.as_str().into() });
        } else {
            let name = ps_ref(state, pnum).active().name.clone();
            events.push(Event::StatusMove { player: pnum, pokemon_name: name, move_name: mv.name.clone() });
        }
        return true;
    }

    false
}

fn is_self_target_move(name: &str, effects: Option<&[stat_stages::StatEffect]>, meta: &crate::moves::MoveMeta) -> bool {
    matches!(name, "Protect"|"Detect"|"Reflect"|"Light Screen"|"Spikes"|"Rest"|"Haze"|"Sunny Day"|"Rain Dance"|"Sandstorm"|"Rapid Spin")
        || effects.is_some_and(|e| e.iter().all(|&(_, _, ts)| ts))
        || (meta.healing > 0 && meta.ailment_id == 0)
}

fn handle_special_move(
    state: &mut BattleState, pnum: u8, dnum: u8, mv: &MoveTemplate, events: &mut Vec<Event>,
) -> bool {
    match mv.name.as_str() {
        "Protect" | "Detect" => {
            let consec = ps_ref(state, pnum).active().protect_consecutive;
            let threshold = 256u32 >> consec;
            let roll = state.rng.random_range(1..=256u32);
            if roll <= threshold {
                let a = ps_mut(state, pnum).active_mut();
                a.protected = true;
                a.protect_consecutive += 1;
                let name = ps_ref(state, pnum).active().name.clone();
                events.push(Event::Protect { player: pnum, pokemon_name: name, success: true });
            } else {
                ps_mut(state, pnum).active_mut().protect_consecutive = 0;
                let name = ps_ref(state, pnum).active().name.clone();
                events.push(Event::Protect { player: pnum, pokemon_name: name, success: false });
            }
            true
        }

        "Spikes" => {
            if !ps_ref(state, dnum).side.spikes {
                ps_mut(state, dnum).side.spikes = true;
                events.push(Event::SpikesSet { player: dnum });
            } else {
                let name = ps_ref(state, pnum).active().name.clone();
                events.push(Event::StatusMove { player: pnum, pokemon_name: name, move_name: mv.name.clone() });
            }
            true
        }

        "Reflect" => {
            if ps_ref(state, pnum).side.reflect_turns == 0 {
                ps_mut(state, pnum).side.reflect_turns = SCREEN_DURATION;
                events.push(Event::ScreenSet { player: pnum, screen: "reflect".into() });
            } else {
                let name = ps_ref(state, pnum).active().name.clone();
                events.push(Event::StatusMove { player: pnum, pokemon_name: name, move_name: mv.name.clone() });
            }
            true
        }

        "Light Screen" => {
            if ps_ref(state, pnum).side.light_screen_turns == 0 {
                ps_mut(state, pnum).side.light_screen_turns = SCREEN_DURATION;
                events.push(Event::ScreenSet { player: pnum, screen: "light_screen".into() });
            } else {
                let name = ps_ref(state, pnum).active().name.clone();
                events.push(Event::StatusMove { player: pnum, pokemon_name: name, move_name: mv.name.clone() });
            }
            true
        }

        "Leech Seed" => {
            if let Some(acc) = mv.accuracy {
                let roll = state.rng.random_range(1..=100u32);
                if roll > acc as u32 {
                    let name = ps_ref(state, pnum).active().name.clone();
                    events.push(Event::Miss { player: pnum, pokemon_name: name, move_name: mv.name.clone() });
                    return true;
                }
            }
            if ps_ref(state, dnum).active().has_type(Type::Grass) {
                let name = ps_ref(state, pnum).active().name.clone();
                events.push(Event::StatusMove { player: pnum, pokemon_name: name, move_name: mv.name.clone() });
                return true;
            }
            if !ps_ref(state, dnum).active().leech_seeded {
                ps_mut(state, dnum).active_mut().leech_seeded = true;
                let name = ps_ref(state, dnum).active().name.clone();
                events.push(Event::LeechSeedApplied { player: dnum, pokemon_name: name });
            } else {
                let name = ps_ref(state, pnum).active().name.clone();
                events.push(Event::StatusMove { player: pnum, pokemon_name: name, move_name: mv.name.clone() });
            }
            true
        }

        "Roar" | "Whirlwind" => {
            let ds = ps_ref(state, dnum);
            let bench: Vec<u8> = ds.team.iter().enumerate()
                .filter(|&(i, p)| i != ds.active_index as usize && !p.is_fainted())
                .map(|(i, _)| i as u8).collect();
            if bench.is_empty() {
                let name = ps_ref(state, pnum).active().name.clone();
                events.push(Event::StatusMove { player: pnum, pokemon_name: name, move_name: mv.name.clone() });
                return true;
            }
            let idx = state.rng.random_range(0..bench.len());
            let target = bench[idx];
            let old = ps_mut(state, dnum).switch_to(target);
            let new_name = ps_ref(state, dnum).active().name.clone();
            events.push(Event::Phaze { player: dnum, pokemon_name: old.unwrap_or_default(), forced_in: new_name });
            apply_spikes_damage(ps_mut(state, dnum), dnum, events);
            true
        }

        "Haze" => {
            for s in state.p1.active_mut().stat_stages.iter_mut() { *s = 0; }
            for s in state.p2.active_mut().stat_stages.iter_mut() { *s = 0; }
            events.push(Event::Haze { player: pnum });
            true
        }

        "Rest" => {
            let a = ps_ref(state, pnum).active();
            if a.current_hp == a.max_hp() {
                let name = a.name.clone();
                events.push(Event::StatusMove { player: pnum, pokemon_name: name, move_name: mv.name.clone() });
                return true;
            }
            let a = ps_mut(state, pnum).active_mut();
            let heal = a.max_hp() - a.current_hp;
            a.heal(heal);
            a.clear_status();
            a.status = Status::Sleep;
            a.status_turns = 2;
            let name = ps_ref(state, pnum).active().name.clone();
            events.push(Event::Heal { player: pnum, pokemon_name: name.clone(), amount: heal, source: "rest".into() });
            events.push(Event::StatusApplied { player: pnum, pokemon_name: name, status: "slp".into() });
            true
        }

        "Sunny Day" | "Rain Dance" | "Sandstorm" => {
            if let Some(w) = weather_from_move(&mv.name) {
                state.weather = Some(w);
                state.weather_turns = WEATHER_DURATION;
                events.push(Event::WeatherSet { player: pnum, weather: w.as_str().into() });
            }
            true
        }

        "Rapid Spin" => false,
        _ => false,
    }
}

// ============================================================
// STAT EFFECTS
// ============================================================

fn do_stat_effects(
    state: &mut BattleState, effects: &[stat_stages::StatEffect],
    pnum: u8, dnum: u8, events: &mut Vec<Event>,
) {
    for &(stat_idx, stages, targets_self) in effects {
        let target_num = if targets_self { pnum } else { dnum };
        let actual = apply_stat_change(&mut ps_mut(state, target_num).active_mut().stat_stages, stat_idx, stages);
        if actual != 0 {
            let name = ps_ref(state, target_num).active().name.clone();
            events.push(Event::StatChange {
                player: target_num, pokemon_name: name,
                stat: stat_stages::stat_name_from_idx(stat_idx).into(), stages: actual,
            });
        }
    }
}

// ============================================================
// SECONDARY EFFECTS
// ============================================================

fn try_secondary_effect(
    state: &mut BattleState, pnum: u8, dnum: u8, mv: &MoveTemplate, events: &mut Vec<Event>,
) {
    let ailment_id = mv.meta.ailment_id;
    let ailment_chance = mv.meta.ailment_chance;

    if ailment_id != 0 && ailment_chance > 0 {
        let roll = state.rng.random_range(1..=100u32);
        if roll <= ailment_chance as u32 {
            if confusion_from_move(ailment_id) {
                let (def, rng) = get_active_and_rng(state, dnum);
                if apply_confusion(def, rng) {
                    let name = ps_ref(state, dnum).active().name.clone();
                    events.push(Event::ConfusionApplied { player: dnum, pokemon_name: name });
                }
            } else if let Some(status) = status_from_move(&mv.name, ailment_id) {
                let (can, _) = can_apply_status(ps_ref(state, dnum).active(), status);
                if can {
                    let (def, rng) = get_active_and_rng(state, dnum);
                    apply_status(def, status, rng);
                    let name = ps_ref(state, dnum).active().name.clone();
                    events.push(Event::StatusApplied { player: dnum, pokemon_name: name, status: status.as_str().into() });
                }
            }
        }
    }

    let stat_chance = mv.meta.stat_chance;
    if let Some(effects) = move_stat_effects(&mv.name) {
        if stat_chance > 0 {
            let roll = state.rng.random_range(1..=100u32);
            if roll <= stat_chance as u32 {
                do_stat_effects(state, effects, pnum, dnum, events);
            }
        }
    }
}

// ============================================================
// RESIDUAL DAMAGE
// ============================================================

fn apply_residual_damage(state: &mut BattleState, pnum: u8, onum: u8) -> Vec<Event> {
    let mut events = Vec::new();
    if ps_ref(state, pnum).active().is_fainted() { return events; }

    let damage = end_of_turn_damage(ps_mut(state, pnum).active_mut());
    if damage > 0 {
        let actual = ps_mut(state, pnum).active_mut().take_damage(damage);
        let a = ps_ref(state, pnum).active();
        let name = a.name.clone();
        let st = a.status.as_str().to_string();
        events.push(Event::ResidualDamage { player: pnum, pokemon_name: name.clone(), status: st, damage: actual });
        if ps_ref(state, pnum).active().is_fainted() {
            events.push(Event::Faint { player: pnum, pokemon_name: name });
            return events;
        }
    }

    if ps_ref(state, pnum).active().leech_seeded && !ps_ref(state, pnum).active().is_fainted() {
        let max_hp = ps_ref(state, pnum).active().max_hp();
        let seed_dmg = (max_hp / 8).max(1);
        let actual = ps_mut(state, pnum).active_mut().take_damage(seed_dmg);
        let name = ps_ref(state, pnum).active().name.clone();
        events.push(Event::LeechSeedDrain { player: pnum, pokemon_name: name.clone(), damage: actual });
        if !ps_ref(state, onum).active().is_fainted() {
            ps_mut(state, onum).active_mut().heal(actual);
        }
        if ps_ref(state, pnum).active().is_fainted() {
            events.push(Event::Faint { player: pnum, pokemon_name: name });
        }
    }

    events
}

// ============================================================
// END OF TURN
// ============================================================

fn end_of_turn(state: &mut BattleState) -> Vec<Event> {
    let mut events = Vec::new();

    for pnum in [1u8, 2] {
        let side = &mut ps_mut(state, pnum).side;
        if side.reflect_turns > 0 {
            side.reflect_turns -= 1;
            if side.reflect_turns == 0 {
                events.push(Event::ScreenExpired { player: pnum, screen: "reflect".into() });
            }
        }
        if side.light_screen_turns > 0 {
            side.light_screen_turns -= 1;
            if side.light_screen_turns == 0 {
                events.push(Event::ScreenExpired { player: pnum, screen: "light_screen".into() });
            }
        }
    }

    if state.weather == Some(Weather::Sandstorm) {
        for pnum in [1u8, 2] {
            if ps_ref(state, pnum).active().is_fainted() { continue; }
            let immune = ps_ref(state, pnum).active().types().iter().any(|t| is_sandstorm_immune(*t));
            if immune { continue; }
            let max_hp = ps_ref(state, pnum).active().max_hp();
            let dmg = (max_hp / 8).max(1);
            let actual = ps_mut(state, pnum).active_mut().take_damage(dmg);
            let name = ps_ref(state, pnum).active().name.clone();
            events.push(Event::WeatherDamage { player: pnum, pokemon_name: name.clone(), damage: actual });
            if ps_ref(state, pnum).active().is_fainted() {
                events.push(Event::Faint { player: pnum, pokemon_name: name });
            }
        }
    }

    if state.weather.is_some() {
        state.weather_turns -= 1;
        if state.weather_turns == 0 {
            let w = state.weather.unwrap().as_str().to_string();
            events.push(Event::WeatherExpired { weather: w });
            state.weather = None;
        }
    }

    events
}

// ============================================================
// FORCED SWITCHES
// ============================================================

pub fn resolve_forced_switches(state: &mut BattleState, sw1: Option<u8>, sw2: Option<u8>) -> Vec<Event> {
    let mut events = Vec::new();
    for (pnum, sw) in [(1u8, sw1), (2u8, sw2)] {
        if let Some(idx) = sw {
            if ps_ref(state, pnum).must_switch() {
                let old = ps_mut(state, pnum).switch_to(idx);
                let new_name = ps_ref(state, pnum).active().name.clone();
                events.push(Event::Switch { player: pnum, from_name: old, to_name: new_name });
                apply_spikes_damage(ps_mut(state, pnum), pnum, &mut events);
            }
        }
    }
    state.check_winner();
    events
}
