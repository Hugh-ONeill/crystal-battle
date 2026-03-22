// Batch 2-ply search with external (NN) leaf evaluation
//
// Unlike batch.rs which uses the heuristic evaluator, this module
// collects all leaf states + observations and returns them for
// external evaluation. Designed for a single FFI round trip:
//   1. Rust: simulate all branches, build obs for leaves
//   2. Python: batched NN inference on all leaves at once
//   3. Rust: aggregate values, return best action

use rayon::prelude::*;
use rand::SeedableRng;
use rand::rngs::SmallRng;

use crate::actions::Action;
use crate::battle::BattleState;
use crate::obs;
use crate::turn_engine;

// ============================================================
// HELPERS
// ============================================================

pub fn decode_action(a: u8) -> Action {
    match a {
        0..=3 => Action::UseMove(a),
        4..=9 => Action::Switch(a - 4),
        _ => Action::Struggle,
    }
}

pub fn handle_forced_switches(state: &mut BattleState) {
    let sw1 = if state.p1.must_switch() {
        state.p1.team.iter().enumerate()
            .find(|&(i, p)| i != state.p1.active_index as usize && !p.is_fainted())
            .map(|(i, _)| i as u8)
    } else { None };
    let sw2 = if state.p2.must_switch() {
        state.p2.team.iter().enumerate()
            .find(|&(i, p)| i != state.p2.active_index as usize && !p.is_fainted())
            .map(|(i, _)| i as u8)
    } else { None };
    if sw1.is_some() || sw2.is_some() {
        turn_engine::resolve_forced_switches(state, sw1, sw2);
    }
}

fn terminal_value(state: &BattleState) -> f32 {
    match state.winner {
        Some(1) => 1.0,
        Some(2) => -1.0,
        _ => 0.0,
    }
}

// ============================================================
// 1-PLY WITH EXTERNAL EVAL
// ============================================================

/// Leaf from a 1-ply search: the post-simulation state + prebuilt observation
pub struct SearchLeaf {
    pub p1_action: u8,
    pub opp_weight: f32,
    pub obs: [f32; obs::OBS_SIZE],
    pub mask: [bool; 10],
    pub is_terminal: bool,
    pub terminal_value: f32,
}

/// Simulate all (p1_action, opp_action) combos and collect leaves.
/// Terminal states get evaluated immediately; non-terminal states
/// get their obs built for external NN evaluation.
pub fn search_1ply_collect(
    root: &BattleState,
    p1_actions: &[u8],
    opp_actions: &[(u8, f32)],
    base_seed: u64,
) -> Vec<SearchLeaf> {
    let combos: Vec<(u8, u8, f32)> = p1_actions
        .iter()
        .flat_map(|&a1| opp_actions.iter().map(move |&(a2, w)| (a1, a2, w)))
        .collect();

    combos
        .par_iter()
        .enumerate()
        .map(|(i, &(a1, a2, weight))| {
            let mut state = root.clone();
            turn_engine::resolve_turn(&mut state, decode_action(a1), decode_action(a2));
            handle_forced_switches(&mut state);

            if state.is_over() {
                return SearchLeaf {
                    p1_action: a1,
                    opp_weight: weight,
                    obs: [0.0; obs::OBS_SIZE],
                    mask: [false; 10],
                    is_terminal: true,
                    terminal_value: terminal_value(&state),
                };
            }

            let observation = obs::build_observation(&state);
            let mask = state.p1.valid_action_mask_filtered(Some(&state.p2), true);

            SearchLeaf {
                p1_action: a1,
                opp_weight: weight,
                obs: observation,
                mask,
                is_terminal: false,
                terminal_value: 0.0,
            }
        })
        .collect()
}

/// Given NN values for each leaf, aggregate into per-action expected values.
pub fn aggregate_1ply(
    p1_actions: &[u8],
    leaves: &[SearchLeaf],
    values: &[f32],
) -> Vec<(u8, f32)> {
    let mut totals = [(0.0f32, 0.0f32); 10]; // (weighted_value_sum, weight_sum)

    for (leaf, &value) in leaves.iter().zip(values.iter()) {
        let v = if leaf.is_terminal { leaf.terminal_value } else { value };
        let idx = leaf.p1_action as usize;
        totals[idx].0 += v * leaf.opp_weight;
        totals[idx].1 += leaf.opp_weight;
    }

    let mut ranked: Vec<(u8, f32)> = p1_actions
        .iter()
        .map(|&a| {
            let (tv, tw) = totals[a as usize];
            (a, if tw > 0.0 { tv / tw } else { -1.0 })
        })
        .collect();

    ranked.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    ranked
}

// ============================================================
// 2-PLY WITH EXTERNAL EVAL
// ============================================================

/// Leaf from a 2-ply search: includes the depth-1 action that led here
pub struct Search2PlyLeaf {
    pub d1_action: u8,       // depth-1 P1 action
    pub d1_opp_weight: f32,  // depth-1 opponent weight
    pub d2_action: u8,       // depth-2 P1 action
    pub d2_opp_weight: f32,  // depth-2 opponent weight
    pub obs: [f32; obs::OBS_SIZE],
    pub mask: [bool; 10],
    pub is_terminal: bool,
    pub terminal_value: f32,
}

/// Simulate all 2-ply branches and collect leaves for external evaluation.
/// Each depth-1 branch expands into all depth-2 (p1_action, opp_action) combos.
pub fn search_2ply_collect(
    root: &BattleState,
    p1_actions: &[u8],
    opp_actions_d1: &[(u8, f32)],
    opp_actions_d2: &[(u8, f32)],
    base_seed: u64,
) -> Vec<Search2PlyLeaf> {
    // depth-1 combos
    let d1_combos: Vec<(u8, u8, f32)> = p1_actions
        .iter()
        .flat_map(|&a1| opp_actions_d1.iter().map(move |&(a2, w)| (a1, a2, w)))
        .collect();

    // for each depth-1 combo, simulate and expand depth-2
    d1_combos
        .par_iter()
        .enumerate()
        .flat_map(|(i, &(d1_a1, d1_a2, d1_weight))| {
            let mut d1_state = root.clone();
            turn_engine::resolve_turn(&mut d1_state, decode_action(d1_a1), decode_action(d1_a2));
            handle_forced_switches(&mut d1_state);

            // terminal at depth 1
            if d1_state.is_over() {
                let tv = terminal_value(&d1_state);
                return vec![Search2PlyLeaf {
                    d1_action: d1_a1,
                    d1_opp_weight: d1_weight,
                    d2_action: 255, // sentinel: no depth-2 action
                    d2_opp_weight: 1.0,
                    obs: [0.0; obs::OBS_SIZE],
                    mask: [false; 10],
                    is_terminal: true,
                    terminal_value: tv,
                }];
            }

            // depth-2 valid actions
            let d2_p1_mask = d1_state.p1.valid_action_mask_filtered(Some(&d1_state.p2), true);
            let d2_p1_valid: Vec<u8> = (0..10).filter(|&j| d2_p1_mask[j as usize]).collect();

            if d2_p1_valid.is_empty() {
                let tv = crate::batch::evaluate_position(&d1_state);
                return vec![Search2PlyLeaf {
                    d1_action: d1_a1,
                    d1_opp_weight: d1_weight,
                    d2_action: 255,
                    d2_opp_weight: 1.0,
                    obs: [0.0; obs::OBS_SIZE],
                    mask: [false; 10],
                    is_terminal: true,
                    terminal_value: tv,
                }];
            }

            // expand all depth-2 combos
            let d2_seed = base_seed.wrapping_add((i as u64 + 1) * 1000003);
            let mut leaves = Vec::new();

            for (j, &d2_a1) in d2_p1_valid.iter().enumerate() {
                for (k, &(d2_a2, d2_w)) in opp_actions_d2.iter().enumerate() {
                    let mut d2_state = d1_state.clone();
                    turn_engine::resolve_turn(
                        &mut d2_state,
                        decode_action(d2_a1),
                        decode_action(d2_a2),
                    );
                    handle_forced_switches(&mut d2_state);

                    if d2_state.is_over() {
                        leaves.push(Search2PlyLeaf {
                            d1_action: d1_a1,
                            d1_opp_weight: d1_weight,
                            d2_action: d2_a1,
                            d2_opp_weight: d2_w,
                            obs: [0.0; obs::OBS_SIZE],
                            mask: [false; 10],
                            is_terminal: true,
                            terminal_value: terminal_value(&d2_state),
                        });
                    } else {
                        let observation = obs::build_observation(&d2_state);
                        let mask = d2_state.p1.valid_action_mask_filtered(
                            Some(&d2_state.p2), true,
                        );
                        leaves.push(Search2PlyLeaf {
                            d1_action: d1_a1,
                            d1_opp_weight: d1_weight,
                            d2_action: d2_a1,
                            d2_opp_weight: d2_w,
                            obs: observation,
                            mask,
                            is_terminal: false,
                            terminal_value: 0.0,
                        });
                    }
                }
            }
            leaves
        })
        .collect()
}

/// Aggregate 2-ply results: for each depth-1 action, find the best depth-2
/// continuation (max over d2 actions), weighted by opponent probabilities.
pub fn aggregate_2ply(
    p1_actions: &[u8],
    leaves: &[Search2PlyLeaf],
    values: &[f32],
) -> Vec<(u8, f32)> {
    // group by (d1_action, d1_opp_instance) -> best d2 value
    // then average over d1 opponent weights

    // step 1: for each leaf, compute its effective value
    // step 2: for each (d1_action, d1_opp_idx), pick the max d2 value
    // step 3: weighted average over d1_opp for each d1_action

    // we don't have d1_opp_idx explicitly, but we can group by
    // (d1_action, d1_opp_weight) -- leaves with same d1_action and
    // d1_opp_weight came from the same depth-1 branch

    use std::collections::HashMap;

    // key: (d1_action, branch_id) where branch_id distinguishes opp instances
    // this is tricky without explicit IDs. Instead, process sequentially:
    // leaves are produced in order: for each d1 combo, all d2 leaves are contiguous

    // simpler approach: group leaves by d1_action, then within each group
    // find all d2 sub-branches and take the max d2 value per sub-branch

    // actually simplest correct approach:
    // for each d1_action, compute: avg over opp_d1 of (max over d2 of (avg over opp_d2 of value))

    // group by d1_action -> list of leaves
    let mut by_d1: HashMap<u8, Vec<(f32, u8, f32, f32)>> = HashMap::new(); // d1_action -> [(d1_opp_w, d2_action, d2_opp_w, value)]
    for (leaf, &val) in leaves.iter().zip(values.iter()) {
        let v = if leaf.is_terminal { leaf.terminal_value } else { val };
        by_d1.entry(leaf.d1_action).or_default().push((
            leaf.d1_opp_weight,
            leaf.d2_action,
            leaf.d2_opp_weight,
            v,
        ));
    }

    let mut ranked: Vec<(u8, f32)> = p1_actions
        .iter()
        .map(|&d1_a| {
            let entries = match by_d1.get(&d1_a) {
                Some(e) => e,
                None => return (d1_a, -1.0),
            };

            // group by d1_opp_weight as branch identifier
            // (all entries with same d1_opp_weight came from same d1 branch)
            let mut branches: HashMap<u64, Vec<(u8, f32, f32)>> = HashMap::new();
            for &(d1_w, d2_a, d2_w, v) in entries {
                let key = d1_w.to_bits() as u64;
                branches.entry(key).or_default().push((d2_a, d2_w, v));
            }

            // for each branch: find best d2 action (max of weighted avg over d2_opp)
            let mut d1_total_val = 0.0f32;
            let mut d1_total_weight = 0.0f32;

            for (key, d2_entries) in &branches {
                let d1_w = f32::from_bits(*key as u32);

                if d2_entries.len() == 1 && d2_entries[0].0 == 255 {
                    // terminal at depth 1, no depth-2 expansion
                    d1_total_val += d2_entries[0].2 * d1_w;
                    d1_total_weight += d1_w;
                    continue;
                }

                // group d2 entries by d2_action
                let mut by_d2: HashMap<u8, (f32, f32)> = HashMap::new();
                for &(d2_a, d2_w, v) in d2_entries {
                    let e = by_d2.entry(d2_a).or_insert((0.0, 0.0));
                    e.0 += v * d2_w;
                    e.1 += d2_w;
                }

                // best d2 action = max expected value
                let best_d2_val = by_d2
                    .values()
                    .map(|&(tv, tw)| if tw > 0.0 { tv / tw } else { -1.0 })
                    .fold(f32::NEG_INFINITY, f32::max);

                d1_total_val += best_d2_val * d1_w;
                d1_total_weight += d1_w;
            }

            let avg = if d1_total_weight > 0.0 {
                d1_total_val / d1_total_weight
            } else {
                -1.0
            };
            (d1_a, avg)
        })
        .collect();

    ranked.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    ranked
}
