// Batch simulation: run N battle state transitions in parallel
//
// Core API:
//   batch_resolve_turn: clone state N times, apply different action pairs, return results
//   batch_evaluate: extract position evaluation from N states
//   batch_sim_search: full 1-ply search in a single call (clone x actions x opp_samples)

use rayon::prelude::*;
use rand::SeedableRng;
use rand::rngs::SmallRng;

use crate::actions::Action;
use crate::battle::BattleState;
use crate::player::PlayerState;
use crate::turn_engine;

// ============================================================
// POSITION EVALUATION
// ============================================================

/// Lightweight position evaluation (same as Python's _evaluate_position)
pub fn evaluate_position(state: &BattleState) -> f32 {
    if state.is_over() {
        return match state.winner {
            Some(1) => 1.0,
            Some(2) => -1.0,
            _ => 0.0,
        };
    }
    let my_hp = state.p1.total_hp_frac();
    let opp_hp = state.p2.total_hp_frac();
    let hp_diff = my_hp - opp_hp;

    let my_alive = state.p1.alive_count() as f32 / state.p1.team.len() as f32;
    let opp_alive = state.p2.alive_count() as f32 / state.p2.team.len() as f32;
    let alive_diff = my_alive - opp_alive;

    hp_diff * 0.6 + alive_diff * 0.4
}

// ============================================================
// FORCED SWITCH HELPER
// ============================================================

/// Handle forced switches after a simulated turn (pick first alive bench mon)
fn handle_forced_switches(state: &mut BattleState) {
    let sw1 = if state.p1.must_switch() {
        first_alive_bench(&state.p1)
    } else {
        None
    };
    let sw2 = if state.p2.must_switch() {
        first_alive_bench(&state.p2)
    } else {
        None
    };
    if sw1.is_some() || sw2.is_some() {
        turn_engine::resolve_forced_switches(state, sw1, sw2);
    }
}

fn first_alive_bench(ps: &PlayerState) -> Option<u8> {
    ps.team.iter().enumerate()
        .find(|&(i, p)| i != ps.active_index as usize && !p.is_fainted())
        .map(|(i, _)| i as u8)
}

// ============================================================
// ACTION DECODE
// ============================================================

fn decode_action(a: u8) -> Action {
    match a {
        0..=3 => Action::UseMove(a),
        4..=9 => Action::Switch(a - 4),
        10 => Action::Struggle,
        _ => Action::Struggle,
    }
}

// ============================================================
// BATCH RESOLVE
// ============================================================

/// Simulate N (action1, action2) pairs from a single root state in parallel.
/// Returns N evaluated post-states (value, winner, is_over) for each pair.
///
/// Each sim gets its own cloned state with a unique RNG seed derived from
/// the base seed + index.
pub fn batch_resolve(
    root: &BattleState,
    action_pairs: &[(u8, u8)],
    base_seed: u64,
) -> Vec<SimResult> {
    action_pairs
        .par_iter()
        .enumerate()
        .map(|(i, &(a1, a2))| {
            let mut state = root.clone();
            // keep cloned RNG state (same starting point for all branches,
            // like Python's copy.deepcopy -- ensures fair comparison between actions)

            turn_engine::resolve_turn(&mut state, decode_action(a1), decode_action(a2));
            handle_forced_switches(&mut state);

            let value = evaluate_position(&state);
            SimResult {
                value,
                winner: state.winner,
                is_over: state.is_over(),
                p1_hp_frac: state.p1.total_hp_frac(),
                p2_hp_frac: state.p2.total_hp_frac(),
                p1_alive: state.p1.alive_count(),
                p2_alive: state.p2.alive_count(),
            }
        })
        .collect()
}

/// Result of a single simulation
#[derive(Debug, Clone)]
pub struct SimResult {
    pub value: f32,
    pub winner: Option<u8>,
    pub is_over: bool,
    pub p1_hp_frac: f32,
    pub p2_hp_frac: f32,
    pub p1_alive: u8,
    pub p2_alive: u8,
}

// ============================================================
// 1-PLY SEARCH
// ============================================================

/// Full 1-ply lookahead: for each valid P1 action, simulate against each
/// weighted opponent action, returning the expected value per P1 action.
///
/// `opp_actions`: list of (action_int, weight) pairs (from opponent model)
///
/// Returns: Vec of (action_int, expected_value) sorted by value descending
pub fn search_1ply(
    root: &BattleState,
    p1_actions: &[u8],
    opp_actions: &[(u8, f32)],
    base_seed: u64,
) -> Vec<(u8, f32)> {
    // build all (my_action, opp_action, opp_weight) combos
    let combos: Vec<(u8, u8, f32)> = p1_actions
        .iter()
        .flat_map(|&a1| {
            opp_actions.iter().map(move |&(a2, w)| (a1, a2, w))
        })
        .collect();

    // run all combos in parallel
    let results: Vec<(u8, f32, f32)> = combos
        .par_iter()
        .enumerate()
        .map(|(i, &(a1, a2, weight))| {
            let mut state = root.clone();

            turn_engine::resolve_turn(&mut state, decode_action(a1), decode_action(a2));
            handle_forced_switches(&mut state);

            let value = evaluate_position(&state);
            (a1, value * weight, weight)
        })
        .collect();

    // aggregate: weighted average per p1 action
    let mut action_totals: Vec<(f32, f32)> = vec![(0.0, 0.0); 10];
    for &(a1, weighted_val, weight) in &results {
        let idx = a1 as usize;
        action_totals[idx].0 += weighted_val;
        action_totals[idx].1 += weight;
    }

    let mut ranked: Vec<(u8, f32)> = p1_actions
        .iter()
        .map(|&a| {
            let (total_val, total_weight) = action_totals[a as usize];
            let avg = if total_weight > 0.0 {
                total_val / total_weight
            } else {
                -1.0
            };
            (a, avg)
        })
        .collect();

    ranked.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    ranked
}

// ============================================================
// 2-PLY SEARCH
// ============================================================

/// Full 2-ply lookahead: for each valid P1 action at depth 1, simulate
/// against opponent samples, then for each resulting state do another 1-ply
/// search to get the best continuation value.
///
/// `opp_actions_d1`: opponent action weights for depth 1
/// `opp_actions_d2`: opponent action weights for depth 2 (can differ if
///   opponent model is re-evaluated, but often the same for simplicity)
///
/// Returns: Vec of (action_int, expected_value) sorted by value descending
pub fn search_2ply(
    root: &BattleState,
    p1_actions: &[u8],
    opp_actions_d1: &[(u8, f32)],
    opp_actions_d2: &[(u8, f32)],
    base_seed: u64,
) -> Vec<(u8, f32)> {
    // for each (my_action, opp_action) at depth 1, simulate and then do 1-ply from there
    let combos: Vec<(u8, u8, f32)> = p1_actions
        .iter()
        .flat_map(|&a1| {
            opp_actions_d1.iter().map(move |&(a2, w)| (a1, a2, w))
        })
        .collect();

    // each combo expands into a full 1-ply search at depth 2
    let results: Vec<(u8, f32, f32)> = combos
        .par_iter()
        .enumerate()
        .map(|(i, &(a1, a2, weight))| {
            let mut state = root.clone();

            turn_engine::resolve_turn(&mut state, decode_action(a1), decode_action(a2));
            handle_forced_switches(&mut state);

            if state.is_over() {
                let val = evaluate_position(&state);
                return (a1, val * weight, weight);
            }

            // get valid actions for depth 2
            let p1_mask = state.p1.valid_action_mask_filtered(Some(&state.p2), true);
            let d2_actions: Vec<u8> = (0..10).filter(|&i| p1_mask[i as usize]).collect();

            if d2_actions.is_empty() {
                let val = evaluate_position(&state);
                return (a1, val * weight, weight);
            }

            // 1-ply search from this state (runs sequentially within this thread)
            let d2_seed = base_seed.wrapping_add((i as u64 + 1) * 1000003);
            let d2_ranked = search_1ply_seq(&state, &d2_actions, opp_actions_d2, d2_seed);

            let best_val = d2_ranked.first().map(|&(_, v)| v).unwrap_or(0.0);
            (a1, best_val * weight, weight)
        })
        .collect();

    // aggregate
    let mut action_totals: Vec<(f32, f32)> = vec![(0.0, 0.0); 10];
    for &(a1, weighted_val, weight) in &results {
        action_totals[a1 as usize].0 += weighted_val;
        action_totals[a1 as usize].1 += weight;
    }

    let mut ranked: Vec<(u8, f32)> = p1_actions
        .iter()
        .map(|&a| {
            let (total_val, total_weight) = action_totals[a as usize];
            let avg = if total_weight > 0.0 {
                total_val / total_weight
            } else {
                -1.0
            };
            (a, avg)
        })
        .collect();

    ranked.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    ranked
}

/// Sequential 1-ply search (used within a rayon task to avoid nested parallelism)
fn search_1ply_seq(
    root: &BattleState,
    p1_actions: &[u8],
    opp_actions: &[(u8, f32)],
    base_seed: u64,
) -> Vec<(u8, f32)> {
    let mut action_totals: Vec<(f32, f32)> = vec![(0.0, 0.0); 10];
    let mut idx = 0u64;

    for &a1 in p1_actions {
        for &(a2, weight) in opp_actions {
            let mut state = root.clone();
            idx += 1;

            turn_engine::resolve_turn(&mut state, decode_action(a1), decode_action(a2));
            handle_forced_switches(&mut state);

            let value = evaluate_position(&state);
            action_totals[a1 as usize].0 += value * weight;
            action_totals[a1 as usize].1 += weight;
        }
    }

    let mut ranked: Vec<(u8, f32)> = p1_actions
        .iter()
        .map(|&a| {
            let (tv, tw) = action_totals[a as usize];
            (a, if tw > 0.0 { tv / tw } else { -1.0 })
        })
        .collect();

    ranked.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    ranked
}
