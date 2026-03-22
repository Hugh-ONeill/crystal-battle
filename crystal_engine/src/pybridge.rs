// PyO3 bindings: thin Python wrappers around Rust types

use std::path::PathBuf;

use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyDict;
use rand::SeedableRng;
use rand::rngs::SmallRng;

use crate::actions::Action;
use crate::battle::BattleState;
use crate::data::DataStore;
use crate::events::Event;
use crate::moves::{MoveSlot, MoveTemplate};
use crate::player::PlayerState;
use crate::pokemon::Pokemon;
use crate::status::Status;
use crate::turn_engine;
use crate::types::Type;

// ============================================================
// PyBattleState
// ============================================================

#[pyclass(name = "BattleState", skip_from_py_object)]
pub struct PyBattleState {
    pub inner: BattleState,
}

#[pymethods]
impl PyBattleState {
    #[getter]
    fn turn(&self) -> u16 {
        self.inner.turn
    }

    #[getter]
    fn winner(&self) -> Option<u8> {
        self.inner.winner
    }

    #[getter]
    fn is_over(&self) -> bool {
        self.inner.is_over()
    }

    #[getter]
    fn weather(&self) -> Option<String> {
        self.inner.weather.map(|w| w.as_str().to_string())
    }

    #[getter]
    fn weather_turns(&self) -> u8 {
        self.inner.weather_turns
    }

    #[getter]
    fn p1(&self) -> PyPlayerState {
        PyPlayerState {
            inner: self.inner.p1.clone(),
        }
    }

    #[getter]
    fn p2(&self) -> PyPlayerState {
        PyPlayerState {
            inner: self.inner.p2.clone(),
        }
    }

    fn resolve_turn(&mut self, action1: u8, action2: u8) -> Vec<PyEvent> {
        let a1 = decode_action(action1);
        let a2 = decode_action(action2);
        let events = turn_engine::resolve_turn(&mut self.inner, a1, a2);
        events.into_iter().map(PyEvent::from).collect()
    }

    fn resolve_forced_switches(&mut self, switch1: Option<u8>, switch2: Option<u8>) -> Vec<PyEvent> {
        let events = turn_engine::resolve_forced_switches(&mut self.inner, switch1, switch2);
        events.into_iter().map(PyEvent::from).collect()
    }

    fn clone_state(&self) -> PyBattleState {
        PyBattleState {
            inner: self.inner.clone(),
        }
    }

    fn __deepcopy__(&self, _memo: &Bound<'_, PyDict>) -> PyBattleState {
        self.clone_state()
    }

    fn get_player(&self, num: u8) -> PyPlayerState {
        PyPlayerState {
            inner: self.inner.get_player(num).clone(),
        }
    }
}

// ============================================================
// PyPlayerState
// ============================================================

#[pyclass(name = "PlayerState", skip_from_py_object)]
#[derive(Clone)]
pub struct PyPlayerState {
    pub inner: PlayerState,
}

#[pymethods]
impl PyPlayerState {
    #[getter]
    fn active_index(&self) -> u8 {
        self.inner.active_index
    }

    #[getter]
    fn active_turns(&self) -> u16 {
        self.inner.active_turns
    }

    #[getter]
    fn active(&self) -> PyPokemon {
        PyPokemon {
            inner: self.inner.active().clone(),
        }
    }

    #[getter]
    fn is_defeated(&self) -> bool {
        self.inner.is_defeated()
    }

    #[getter]
    fn must_switch(&self) -> bool {
        self.inner.must_switch()
    }

    #[getter]
    fn alive_count(&self) -> u8 {
        self.inner.alive_count()
    }

    fn total_hp_frac(&self) -> f32 {
        self.inner.total_hp_frac()
    }

    #[getter]
    fn team(&self) -> Vec<PyPokemon> {
        self.inner
            .team
            .iter()
            .map(|p| PyPokemon { inner: p.clone() })
            .collect()
    }

    #[getter]
    fn side(&self) -> PySideConditions {
        PySideConditions {
            spikes: self.inner.side.spikes,
            reflect_turns: self.inner.side.reflect_turns,
            light_screen_turns: self.inner.side.light_screen_turns,
        }
    }

    #[pyo3(signature = (opponent=None, filter_immune=false))]
    fn valid_actions(&self, opponent: Option<&PyPlayerState>, filter_immune: bool) -> Vec<u8> {
        let opp_ref = opponent.map(|o| &o.inner);
        self.inner
            .valid_actions_filtered(opp_ref, filter_immune)
            .into_iter()
            .map(encode_action)
            .collect()
    }

    #[pyo3(signature = (opponent=None, filter_immune=false))]
    fn valid_action_mask(&self, opponent: Option<&PyPlayerState>, filter_immune: bool) -> [bool; 10] {
        let opp_ref = opponent.map(|o| &o.inner);
        self.inner.valid_action_mask_filtered(opp_ref, filter_immune)
    }
}

// ============================================================
// PySideConditions
// ============================================================

#[pyclass(name = "SideConditions", skip_from_py_object)]
#[derive(Clone)]
pub struct PySideConditions {
    #[pyo3(get)]
    pub spikes: bool,
    #[pyo3(get)]
    pub reflect_turns: u8,
    #[pyo3(get)]
    pub light_screen_turns: u8,
}

// ============================================================
// PyPokemon
// ============================================================

#[pyclass(name = "Pokemon", from_py_object)]
#[derive(Clone)]
pub struct PyPokemon {
    pub inner: Pokemon,
}

#[pymethods]
impl PyPokemon {
    #[getter]
    fn name(&self) -> &str {
        &self.inner.name
    }

    #[getter]
    fn species_id(&self) -> u16 {
        self.inner.species_id
    }

    #[getter]
    fn current_hp(&self) -> u16 {
        self.inner.current_hp
    }

    #[getter]
    fn max_hp(&self) -> u16 {
        self.inner.max_hp()
    }

    #[getter]
    fn hp_frac(&self) -> f32 {
        self.inner.hp_frac()
    }

    #[getter]
    fn is_fainted(&self) -> bool {
        self.inner.is_fainted()
    }

    #[getter]
    fn types(&self) -> Vec<String> {
        self.inner.types().iter().map(|t| t.as_str().to_string()).collect()
    }

    #[getter]
    fn stats(&self) -> [u16; 6] {
        self.inner.stats
    }

    #[getter]
    fn status(&self) -> Option<String> {
        match self.inner.status {
            Status::None => None,
            s => Some(s.as_str().to_string()),
        }
    }

    #[getter]
    fn status_turns(&self) -> u8 {
        self.inner.status_turns
    }

    #[getter]
    fn confusion_turns(&self) -> u8 {
        self.inner.confusion_turns
    }

    #[getter]
    fn stat_stages(&self) -> [i8; 7] {
        self.inner.stat_stages
    }

    #[getter]
    fn flinched(&self) -> bool {
        self.inner.flinched
    }

    #[getter]
    fn leech_seeded(&self) -> bool {
        self.inner.leech_seeded
    }

    #[getter]
    fn protected(&self) -> bool {
        self.inner.protected
    }

    #[getter]
    fn protect_consecutive(&self) -> u8 {
        self.inner.protect_consecutive
    }

    #[getter]
    fn recharging(&self) -> bool {
        self.inner.recharging
    }

    #[getter]
    fn move_slots(&self) -> Vec<PyMoveSlot> {
        self.inner
            .move_slots
            .iter()
            .map(|s| PyMoveSlot { inner: s.clone() })
            .collect()
    }

    fn has_any_pp(&self) -> bool {
        self.inner.has_any_pp()
    }
}

// ============================================================
// PyMoveSlot
// ============================================================

#[pyclass(name = "MoveSlot", skip_from_py_object)]
#[derive(Clone)]
pub struct PyMoveSlot {
    pub inner: MoveSlot,
}

#[pymethods]
impl PyMoveSlot {
    #[getter]
    fn current_pp(&self) -> u8 {
        self.inner.current_pp
    }

    #[getter]
    fn has_pp(&self) -> bool {
        self.inner.has_pp()
    }

    #[getter]
    fn template(&self) -> PyMoveTemplate {
        PyMoveTemplate {
            inner: self.inner.template.clone(),
        }
    }
}

// ============================================================
// PyMoveTemplate
// ============================================================

#[pyclass(name = "MoveTemplate", skip_from_py_object)]
#[derive(Clone)]
pub struct PyMoveTemplate {
    pub inner: MoveTemplate,
}

#[pymethods]
impl PyMoveTemplate {
    #[getter]
    fn id(&self) -> u16 {
        self.inner.id
    }

    #[getter]
    fn name(&self) -> &str {
        &self.inner.name
    }

    #[getter]
    fn power(&self) -> u8 {
        self.inner.power
    }

    #[getter]
    fn accuracy(&self) -> Option<u8> {
        self.inner.accuracy
    }

    #[getter]
    fn pp(&self) -> u8 {
        self.inner.pp
    }

    #[getter]
    fn priority(&self) -> i8 {
        self.inner.priority
    }

    #[getter]
    fn damage_class(&self) -> &str {
        match self.inner.damage_class {
            crate::moves::DamageClass::Physical => "physical",
            crate::moves::DamageClass::Special => "special",
            crate::moves::DamageClass::Status => "status",
        }
    }

    #[getter]
    fn move_type(&self) -> String {
        self.inner.move_type.as_str().to_string()
    }

    #[getter]
    fn meta(&self) -> PyMoveMeta {
        PyMoveMeta {
            inner: self.inner.meta,
        }
    }
}

// ============================================================
// PyMoveMeta
// ============================================================

#[pyclass(name = "MoveMeta", skip_from_py_object)]
#[derive(Clone)]
pub struct PyMoveMeta {
    inner: crate::moves::MoveMeta,
}

#[pymethods]
impl PyMoveMeta {
    #[getter]
    fn ailment_id(&self) -> u8 { self.inner.ailment_id }
    #[getter]
    fn min_hits(&self) -> Option<u8> { self.inner.min_hits }
    #[getter]
    fn max_hits(&self) -> Option<u8> { self.inner.max_hits }
    #[getter]
    fn drain(&self) -> i8 { self.inner.drain }
    #[getter]
    fn healing(&self) -> u8 { self.inner.healing }
    #[getter]
    fn crit_rate(&self) -> u8 { self.inner.crit_rate }
    #[getter]
    fn ailment_chance(&self) -> u8 { self.inner.ailment_chance }
    #[getter]
    fn flinch_chance(&self) -> u8 { self.inner.flinch_chance }
    #[getter]
    fn stat_chance(&self) -> u8 { self.inner.stat_chance }
}

// ============================================================
// PyEvent
// ============================================================

#[pyclass(name = "Event", skip_from_py_object)]
#[derive(Clone)]
pub struct PyEvent {
    pub inner: Event,
}

impl From<Event> for PyEvent {
    fn from(e: Event) -> Self {
        PyEvent { inner: e }
    }
}

#[pymethods]
impl PyEvent {
    #[getter]
    fn kind(&self) -> &str {
        match &self.inner {
            Event::Switch { .. } => "switch",
            Event::Move { .. } => "move",
            Event::Faint { .. } => "faint",
            Event::Struggle { .. } => "struggle",
            Event::Miss { .. } => "miss",
            Event::StatusMove { .. } => "status_move",
            Event::StatusApplied { .. } => "status_applied",
            Event::StatusCured { .. } => "status_cured",
            Event::StatusPrevented { .. } => "status_prevented",
            Event::ResidualDamage { .. } => "residual_damage",
            Event::ConfusionApplied { .. } => "confusion_applied",
            Event::ConfusionHitSelf { .. } => "confusion_hit_self",
            Event::StatChange { .. } => "stat_change",
            Event::Heal { .. } => "heal",
            Event::Flinch { .. } => "flinch",
            Event::SpikesSet { .. } => "spikes_set",
            Event::SpikesDamage { .. } => "spikes_damage",
            Event::ScreenSet { .. } => "screen_set",
            Event::ScreenExpired { .. } => "screen_expired",
            Event::Protect { .. } => "protect",
            Event::LeechSeedApplied { .. } => "leech_seed_applied",
            Event::LeechSeedDrain { .. } => "leech_seed_drain",
            Event::Phaze { .. } => "phaze",
            Event::Haze { .. } => "haze",
            Event::WeatherSet { .. } => "weather_set",
            Event::WeatherDamage { .. } => "weather_damage",
            Event::WeatherExpired { .. } => "weather_expired",
        }
    }

    #[getter]
    fn player(&self) -> Option<u8> {
        match &self.inner {
            Event::Switch { player, .. }
            | Event::Move { player, .. }
            | Event::Faint { player, .. }
            | Event::Struggle { player, .. }
            | Event::Miss { player, .. }
            | Event::StatusMove { player, .. }
            | Event::StatusApplied { player, .. }
            | Event::StatusCured { player, .. }
            | Event::StatusPrevented { player, .. }
            | Event::ResidualDamage { player, .. }
            | Event::ConfusionApplied { player, .. }
            | Event::ConfusionHitSelf { player, .. }
            | Event::StatChange { player, .. }
            | Event::Heal { player, .. }
            | Event::Flinch { player, .. }
            | Event::SpikesSet { player, .. }
            | Event::SpikesDamage { player, .. }
            | Event::ScreenSet { player, .. }
            | Event::ScreenExpired { player, .. }
            | Event::Protect { player, .. }
            | Event::LeechSeedApplied { player, .. }
            | Event::LeechSeedDrain { player, .. }
            | Event::Phaze { player, .. }
            | Event::Haze { player, .. }
            | Event::WeatherSet { player, .. }
            | Event::WeatherDamage { player, .. } => Some(*player),
            Event::WeatherExpired { .. } => None,
        }
    }

    #[getter]
    fn damage(&self) -> Option<u16> {
        match &self.inner {
            Event::Move { damage, .. }
            | Event::Struggle { damage, .. }
            | Event::SpikesDamage { damage, .. }
            | Event::ResidualDamage { damage, .. }
            | Event::ConfusionHitSelf { damage, .. }
            | Event::LeechSeedDrain { damage, .. }
            | Event::WeatherDamage { damage, .. } => Some(*damage),
            _ => None,
        }
    }

    fn __repr__(&self) -> String {
        format!("Event(kind={:?}, player={:?})", self.kind(), self.player())
    }
}

// ============================================================
// PyDataStore
// ============================================================

#[pyclass(name = "DataStore", skip_from_py_object)]
pub struct PyDataStore {
    pub inner: DataStore,
}

#[pymethods]
impl PyDataStore {
    #[new]
    #[pyo3(signature = (data_dir=None))]
    fn new(data_dir: Option<String>) -> PyResult<Self> {
        let path = data_dir
            .map(PathBuf::from)
            .unwrap_or_else(|| PathBuf::from("data"));
        let store = DataStore::load(&path)
            .map_err(|e| PyValueError::new_err(format!("Failed to load data: {e}")))?;
        Ok(PyDataStore { inner: store })
    }

    #[pyo3(signature = (species_id, move_ids, dvs=None))]
    fn build_pokemon(&self, species_id: u16, move_ids: Vec<u16>, dvs: Option<[u16; 4]>) -> PyResult<PyPokemon> {
        let pokemon = match dvs {
            Some(dv) => self.inner.build_pokemon_with_dvs(species_id, &move_ids, dv),
            None => self.inner.build_pokemon(species_id, &move_ids),
        };
        pokemon
            .map(|p| PyPokemon { inner: p })
            .ok_or_else(|| PyValueError::new_err(format!("Species {species_id} not found")))
    }
}

// ============================================================
// MODULE-LEVEL FUNCTIONS
// ============================================================

/// Create a BattleState from two teams of PyPokemon
#[pyfunction]
#[pyo3(signature = (team1, team2, seed=None))]
fn create_battle(team1: Vec<PyPokemon>, team2: Vec<PyPokemon>, seed: Option<u64>) -> PyBattleState {
    let p1_team: Vec<Pokemon> = team1.into_iter().map(|p| p.inner).collect();
    let p2_team: Vec<Pokemon> = team2.into_iter().map(|p| p.inner).collect();
    let rng = match seed {
        Some(s) => SmallRng::seed_from_u64(s),
        None => SmallRng::from_os_rng(),
    };
    PyBattleState {
        inner: BattleState::new(PlayerState::new(p1_team), PlayerState::new(p2_team), rng),
    }
}

// ============================================================
// BATCH API
// ============================================================

/// Simulate N (action1, action2) pairs from a root state in parallel.
/// Returns list of (value, winner, is_over, p1_hp_frac, p2_hp_frac, p1_alive, p2_alive).
#[pyfunction]
#[pyo3(signature = (state, action_pairs, base_seed=0))]
fn batch_resolve(
    state: &PyBattleState,
    action_pairs: Vec<(u8, u8)>,
    base_seed: u64,
) -> Vec<(f32, Option<u8>, bool, f32, f32, u8, u8)> {
    let results = crate::batch::batch_resolve(&state.inner, &action_pairs, base_seed);
    results
        .into_iter()
        .map(|r| (r.value, r.winner, r.is_over, r.p1_hp_frac, r.p2_hp_frac, r.p1_alive, r.p2_alive))
        .collect()
}

/// 1-ply lookahead search entirely in Rust.
/// p1_actions: list of valid P1 action ints
/// opp_actions: list of (action_int, weight) from opponent model
/// Returns list of (action_int, expected_value) sorted best-first.
#[pyfunction]
#[pyo3(signature = (state, p1_actions, opp_actions, base_seed=0))]
fn search_1ply(
    state: &PyBattleState,
    p1_actions: Vec<u8>,
    opp_actions: Vec<(u8, f32)>,
    base_seed: u64,
) -> Vec<(u8, f32)> {
    crate::batch::search_1ply(&state.inner, &p1_actions, &opp_actions, base_seed)
}

/// 2-ply lookahead search entirely in Rust.
/// opp_actions_d1/d2: opponent weights for depth 1 and 2
/// Returns list of (action_int, expected_value) sorted best-first.
#[pyfunction]
#[pyo3(signature = (state, p1_actions, opp_actions_d1, opp_actions_d2, base_seed=0))]
fn search_2ply(
    state: &PyBattleState,
    p1_actions: Vec<u8>,
    opp_actions_d1: Vec<(u8, f32)>,
    opp_actions_d2: Vec<(u8, f32)>,
    base_seed: u64,
) -> Vec<(u8, f32)> {
    crate::batch::search_2ply(&state.inner, &p1_actions, &opp_actions_d1, &opp_actions_d2, base_seed)
}

/// 1-ply search with NN eval: Rust sims all branches, returns obs batch,
/// Python evals, Rust aggregates. Single FFI round trip.
/// evaluator_fn(obs_batch, mask_batch) -> (values, priors)
/// Returns list of (action_int, expected_value) sorted best-first.
#[pyfunction]
#[pyo3(signature = (state, p1_actions, opp_actions, evaluator_fn, base_seed=0))]
fn search_1ply_nn(
    py: Python<'_>,
    state: &PyBattleState,
    p1_actions: Vec<u8>,
    opp_actions: Vec<(u8, f32)>,
    evaluator_fn: &Bound<'_, pyo3::types::PyAny>,
    base_seed: u64,
) -> PyResult<Vec<(u8, f32)>> {
    let leaves = crate::batch_nn::search_1ply_collect(
        &state.inner, &p1_actions, &opp_actions, base_seed,
    );

    // separate terminal vs non-terminal leaves
    let needs_eval: Vec<usize> = leaves.iter().enumerate()
        .filter(|(_, l)| !l.is_terminal)
        .map(|(i, _)| i)
        .collect();

    let mut values = vec![0.0f32; leaves.len()];

    // set terminal values directly
    for (i, leaf) in leaves.iter().enumerate() {
        if leaf.is_terminal {
            values[i] = leaf.terminal_value;
        }
    }

    // batch NN eval for non-terminal leaves
    if !needs_eval.is_empty() {
        let obs_batch: Vec<Vec<f32>> = needs_eval.iter()
            .map(|&i| leaves[i].obs.to_vec())
            .collect();
        let mask_batch: Vec<Vec<f32>> = needs_eval.iter()
            .map(|&i| leaves[i].mask.iter().map(|&b| if b { 1.0 } else { 0.0 }).collect())
            .collect();

        let result = evaluator_fn.call1((obs_batch, mask_batch))?;
        let (nn_values, _priors): (Vec<f32>, Vec<Vec<f32>>) = result.extract()?;

        for (j, &i) in needs_eval.iter().enumerate() {
            values[i] = nn_values[j];
        }
    }

    Ok(crate::batch_nn::aggregate_1ply(&p1_actions, &leaves, &values))
}

/// 2-ply search with NN eval and per-state opponent prediction.
/// Matches the Python LookaheadAgent's behavior:
///   1. Rust sims all depth-1 branches
///   2. Python predicts opponent actions for each depth-1 result state
///   3. Rust sims all depth-2 branches with correct opp weights
///   4. Python evals all depth-2 leaves
///   5. Rust aggregates
///
/// evaluator_fn(obs_batch, mask_batch) -> (values, priors)
/// opp_predict_fn(obs_batch, mask_batch) -> list of [(action, prob), ...]
#[pyfunction]
#[pyo3(signature = (state, p1_actions, opp_actions_d1, evaluator_fn=None, opp_predict_fn=None, n_opp_samples=5, base_seed=0))]
fn search_2ply_nn(
    py: Python<'_>,
    state: &PyBattleState,
    p1_actions: Vec<u8>,
    opp_actions_d1: Vec<(u8, f32)>,
    evaluator_fn: Option<&Bound<'_, pyo3::types::PyAny>>,
    opp_predict_fn: Option<&Bound<'_, pyo3::types::PyAny>>,
    n_opp_samples: usize,
    base_seed: u64,
) -> PyResult<Vec<(u8, f32)>> {
    use crate::batch_nn::{decode_action, handle_forced_switches};
    use rand::SeedableRng;
    use rand::rngs::SmallRng;

    // ---- depth 1: simulate all (p1_action, opp_action) combos ----
    let d1_combos: Vec<(u8, u8, f32)> = p1_actions.iter()
        .flat_map(|&a1| opp_actions_d1.iter().map(move |&(a2, w)| (a1, a2, w)))
        .collect();

    struct D1Result {
        d1_action: u8,
        d1_opp_weight: f32,
        state: crate::battle::BattleState,
        is_terminal: bool,
        terminal_value: f32,
    }

    let d1_results: Vec<D1Result> = d1_combos.iter().enumerate().map(|(i, &(a1, a2, w))| {
        let mut s = state.inner.clone();
        s.rng = SmallRng::seed_from_u64(base_seed.wrapping_add(i as u64));
        crate::turn_engine::resolve_turn(&mut s, decode_action(a1), decode_action(a2));
        handle_forced_switches(&mut s);
        let is_terminal = s.is_over();
        let tv = if is_terminal {
            match s.winner { Some(1) => 1.0, Some(2) => -1.0, _ => 0.0 }
        } else { 0.0 };
        D1Result { d1_action: a1, d1_opp_weight: w, state: s, is_terminal, terminal_value: tv }
    }).collect();

    // ---- get opponent predictions for each non-terminal depth-1 state ----
    let non_terminal_d1: Vec<usize> = d1_results.iter().enumerate()
        .filter(|(_, r)| !r.is_terminal)
        .map(|(i, _)| i)
        .collect();

    // build obs for non-terminal d1 states
    let d1_opp_weights: Vec<Vec<(u8, f32)>> = if let Some(predict_fn) = opp_predict_fn {
        if !non_terminal_d1.is_empty() {
            let obs_batch: Vec<Vec<f32>> = non_terminal_d1.iter()
                .map(|&i| crate::obs::build_observation(&d1_results[i].state).to_vec())
                .collect();
            let mask_batch: Vec<Vec<f32>> = non_terminal_d1.iter()
                .map(|&i| {
                    let s = &d1_results[i].state;
                    s.p2.valid_action_mask_filtered(Some(&s.p1), true)
                        .iter().map(|&b| if b { 1.0 } else { 0.0 }).collect()
                })
                .collect();
            let result = predict_fn.call1((obs_batch, mask_batch))?;
            let predictions: Vec<Vec<(u8, f32)>> = result.extract()?;
            predictions
        } else {
            vec![]
        }
    } else {
        // no opp model: uniform weights for each state
        non_terminal_d1.iter().map(|&i| {
            let s = &d1_results[i].state;
            let mask = s.p2.valid_action_mask_filtered(Some(&s.p1), true);
            let valid: Vec<u8> = (0..10).filter(|&j| mask[j as usize]).collect();
            let w = 1.0 / valid.len().max(1) as f32;
            valid.into_iter().take(n_opp_samples).map(|a| (a, w)).collect()
        }).collect()
    };

    // ---- depth 2: for each non-terminal d1 state, expand with correct opp weights ----
    let mut all_d2_leaves: Vec<crate::batch_nn::Search2PlyLeaf> = Vec::new();

    // terminal d1 results go straight to leaves
    for r in &d1_results {
        if r.is_terminal {
            all_d2_leaves.push(crate::batch_nn::Search2PlyLeaf {
                d1_action: r.d1_action,
                d1_opp_weight: r.d1_opp_weight,
                d2_action: 255,
                d2_opp_weight: 1.0,
                obs: [0.0; crate::obs::OBS_SIZE],
                mask: [false; 10],
                is_terminal: true,
                terminal_value: r.terminal_value,
            });
        }
    }

    // non-terminal: expand depth 2
    for (nt_idx, &d1_idx) in non_terminal_d1.iter().enumerate() {
        let r = &d1_results[d1_idx];
        let d2_opp = if nt_idx < d1_opp_weights.len() { &d1_opp_weights[nt_idx] } else { &opp_actions_d1 };

        let d2_p1_mask = r.state.p1.valid_action_mask_filtered(Some(&r.state.p2), true);
        let d2_valid: Vec<u8> = (0..10).filter(|&j| d2_p1_mask[j as usize]).collect();

        if d2_valid.is_empty() {
            let val = crate::batch::evaluate_position(&r.state);
            all_d2_leaves.push(crate::batch_nn::Search2PlyLeaf {
                d1_action: r.d1_action, d1_opp_weight: r.d1_opp_weight,
                d2_action: 255, d2_opp_weight: 1.0,
                obs: [0.0; crate::obs::OBS_SIZE], mask: [false; 10],
                is_terminal: true, terminal_value: val,
            });
            continue;
        }

        let d2_seed = base_seed.wrapping_add((d1_idx as u64 + 1) * 1000003);
        for (j, &d2_a1) in d2_valid.iter().enumerate() {
            for (k, &(d2_a2, d2_w)) in d2_opp.iter().enumerate() {
                let mut d2_state = r.state.clone();
                d2_state.rng = SmallRng::seed_from_u64(
                    d2_seed.wrapping_add((j * d2_opp.len() + k) as u64),
                );
                crate::turn_engine::resolve_turn(
                    &mut d2_state, decode_action(d2_a1), decode_action(d2_a2),
                );
                handle_forced_switches(&mut d2_state);

                if d2_state.is_over() {
                    let tv = match d2_state.winner { Some(1) => 1.0, Some(2) => -1.0, _ => 0.0 };
                    all_d2_leaves.push(crate::batch_nn::Search2PlyLeaf {
                        d1_action: r.d1_action, d1_opp_weight: r.d1_opp_weight,
                        d2_action: d2_a1, d2_opp_weight: d2_w,
                        obs: [0.0; crate::obs::OBS_SIZE], mask: [false; 10],
                        is_terminal: true, terminal_value: tv,
                    });
                } else {
                    let obs = crate::obs::build_observation(&d2_state);
                    let mask = d2_state.p1.valid_action_mask_filtered(Some(&d2_state.p2), true);
                    all_d2_leaves.push(crate::batch_nn::Search2PlyLeaf {
                        d1_action: r.d1_action, d1_opp_weight: r.d1_opp_weight,
                        d2_action: d2_a1, d2_opp_weight: d2_w,
                        obs, mask,
                        is_terminal: false, terminal_value: 0.0,
                    });
                }
            }
        }
    }

    // ---- evaluate all depth-2 leaves ----
    let needs_eval: Vec<usize> = all_d2_leaves.iter().enumerate()
        .filter(|(_, l)| !l.is_terminal)
        .map(|(i, _)| i)
        .collect();

    let mut values = vec![0.0f32; all_d2_leaves.len()];
    for (i, leaf) in all_d2_leaves.iter().enumerate() {
        if leaf.is_terminal { values[i] = leaf.terminal_value; }
    }

    if !needs_eval.is_empty() {
        if let Some(eval_fn) = evaluator_fn {
            // NN evaluation
            let obs_batch: Vec<Vec<f32>> = needs_eval.iter()
                .map(|&i| all_d2_leaves[i].obs.to_vec()).collect();
            let mask_batch: Vec<Vec<f32>> = needs_eval.iter()
                .map(|&i| all_d2_leaves[i].mask.iter().map(|&b| if b { 1.0 } else { 0.0 }).collect())
                .collect();
            let result = eval_fn.call1((obs_batch, mask_batch))?;
            let (nn_values, _): (Vec<f32>, Vec<Vec<f32>>) = result.extract()?;
            for (j, &i) in needs_eval.iter().enumerate() { values[i] = nn_values[j]; }
        } else {
            // heuristic evaluation (no Python callback needed)
            for &i in &needs_eval {
                // reconstruct the state to evaluate -- we stored the obs but need
                // to use the heuristic which needs hp_frac and alive_count.
                // The obs has these at known positions, extract them directly:
                // obs[0] = my_hp_frac, obs[2] = opp_hp_frac (from active section)
                // But total_hp_frac and alive_count are in the global section.
                // Global starts at 407+378+252 = 1037
                // global[0]=turn, [1]=my_alive/6, [2]=opp_alive/6, [3]=my_hp_frac, [4]=opp_hp_frac
                let obs = &all_d2_leaves[i].obs;
                let my_hp = obs[1040]; // global[3]
                let opp_hp = obs[1041]; // global[4]
                let my_alive = obs[1038]; // global[1]
                let opp_alive = obs[1039]; // global[2]
                values[i] = (my_hp - opp_hp) * 0.6 + (my_alive - opp_alive) * 0.4;
            }
        }
    }

    Ok(crate::batch_nn::aggregate_2ply(&p1_actions, &all_d2_leaves, &values))
}

/// Evaluate a battle position (HP diff + alive diff heuristic)
#[pyfunction]
fn evaluate_position(state: &PyBattleState) -> f32 {
    crate::batch::evaluate_position(&state.inner)
}

/// Build observation vector (1052 floats) from a BattleState -- entirely in Rust
#[pyfunction]
fn build_observation(state: &PyBattleState) -> Vec<f32> {
    crate::obs::build_observation(&state.inner).to_vec()
}

/// Run full MCTS search, calling a Python evaluator for leaf batches.
/// The evaluator callable receives (obs_batch: list[list[f32]], mask_batch: list[list[float]])
/// and must return (values: list[float], priors: list[list[float]]).
/// This keeps the loop in Rust, crossing FFI only for inference callbacks.
#[pyfunction]
#[pyo3(signature = (state, n_simulations, evaluator_fn, seed=0, c_puct=1.5, max_batch=64, temperature=0.1, opp_weights=None))]
fn mcts_search(
    py: Python<'_>,
    state: &PyBattleState,
    n_simulations: u32,
    evaluator_fn: &Bound<'_, pyo3::types::PyAny>,
    seed: u64,
    c_puct: f32,
    max_batch: usize,
    temperature: f32,
    opp_weights: Option<Vec<(u8, f32)>>,
) -> PyResult<[f32; 10]> {
    let mut ctx = crate::search::MctsContext::new(
        state.inner.clone(), n_simulations, seed, c_puct,
    );
    if let Some(weights) = opp_weights {
        ctx.set_opp_weights(weights);
    }

    loop {
        let n_pending = ctx.run_until_eval_needed(max_batch);
        if n_pending == 0 {
            break;
        }

        // build obs + masks entirely in Rust
        let leaves = ctx.get_pending_leaves();
        let mut obs_batch: Vec<Vec<f32>> = Vec::with_capacity(n_pending);
        let mut mask_batch: Vec<Vec<f32>> = Vec::with_capacity(n_pending);

        for leaf in leaves {
            let obs = crate::obs::build_observation(&leaf.state);
            let mask = leaf.state.p1.valid_action_mask_filtered(Some(&leaf.state.p2), true);
            obs_batch.push(obs.to_vec());
            mask_batch.push(mask.iter().map(|&b| if b { 1.0f32 } else { 0.0 }).collect());
        }

        // call Python evaluator
        let result = evaluator_fn.call1((obs_batch, mask_batch))?;
        let (values, priors): (Vec<f32>, Vec<Vec<f32>>) = result.extract()?;

        // convert priors to fixed arrays
        let priors_arr: Vec<[f32; 10]> = priors
            .iter()
            .map(|p| {
                let mut arr = [0.0f32; 10];
                for (i, &v) in p.iter().enumerate().take(10) {
                    arr[i] = v;
                }
                arr
            })
            .collect();

        ctx.supply_evaluations(&values, &priors_arr);
    }

    Ok(ctx.get_action_probs(temperature))
}

/// Build observations for a batch of BattleStates (for MCTS leaf eval)
/// Returns list of 1052-element float lists
#[pyfunction]
fn build_observations_batch(states: &Bound<'_, pyo3::types::PyList>) -> PyResult<Vec<Vec<f32>>> {
    let mut result = Vec::with_capacity(states.len());
    for item in states.iter() {
        let state: PyRef<'_, PyBattleState> = item.extract()?;
        result.push(crate::obs::build_observation(&state.inner).to_vec());
    }
    Ok(result)
}

/// Type effectiveness lookup
#[pyfunction]
fn type_effectiveness(atk_type: &str, def_types: Vec<String>) -> PyResult<f32> {
    let atk = Type::from_str(atk_type)
        .ok_or_else(|| PyValueError::new_err(format!("Unknown type: {atk_type}")))?;
    let defs: Vec<Type> = def_types
        .iter()
        .filter_map(|s| Type::from_str(s))
        .collect();
    Ok(crate::types::combined_effectiveness(atk, &defs))
}

// ============================================================
// ACTION ENCODING
// ============================================================

/// Decode a gym action int (0-9) to engine Action.
/// 0-3: UseMove(0-3), 4-9: Switch(0-5), 10: Struggle, 11: Forfeit
fn decode_action(action: u8) -> Action {
    match action {
        0..=3 => Action::UseMove(action),
        4..=9 => Action::Switch(action - 4),
        10 => Action::Struggle,
        11 => Action::Forfeit,
        _ => Action::Struggle,
    }
}

/// Encode engine Action to gym action int
fn encode_action(action: Action) -> u8 {
    match action {
        Action::UseMove(i) => i,
        Action::Switch(i) => i + 4,
        Action::Struggle => 10,
        Action::Forfeit => 11,
    }
}

// ============================================================
// MCTS SEARCH
// ============================================================

#[pyclass(name = "MctsContext", skip_from_py_object)]
pub struct PyMctsContext {
    inner: crate::search::MctsContext,
}

#[pymethods]
impl PyMctsContext {
    #[new]
    #[pyo3(signature = (state, n_simulations, seed=0, c_puct=1.5))]
    fn new(state: &PyBattleState, n_simulations: u32, seed: u64, c_puct: f32) -> Self {
        PyMctsContext {
            inner: crate::search::MctsContext::new(
                state.inner.clone(), n_simulations, seed, c_puct,
            ),
        }
    }

    /// Run MCTS until a batch of leaves needs NN evaluation (or search is done).
    /// Returns number of leaves pending evaluation.
    #[pyo3(signature = (max_batch=16))]
    fn run_until_eval_needed(&mut self, max_batch: usize) -> usize {
        self.inner.run_until_eval_needed(max_batch)
    }

    /// Get state info for each pending leaf (for building observations).
    /// Returns list of dicts with fields needed by obs_builder.
    fn get_pending_leaf_states(&self) -> Vec<PyBattleState> {
        self.inner
            .get_pending_leaves()
            .iter()
            .map(|leaf| PyBattleState {
                inner: leaf.state.clone(),
            })
            .collect()
    }

    /// Supply NN evaluations for pending leaves.
    /// values: list of floats (P1 perspective, -1 to 1)
    /// priors: list of 10-element lists (policy priors over action space)
    fn supply_evaluations(&mut self, values: Vec<f32>, priors: Vec<[f32; 10]>) {
        self.inner.supply_evaluations(&values, &priors);
    }

    /// Use heuristic evaluation instead of NN (for testing / when no model available)
    fn supply_heuristic_evaluations(&mut self) {
        self.inner.supply_heuristic_evaluations();
    }

    /// Get search results: list of (action, visits, q_value) sorted by visits
    fn get_results(&self) -> Vec<(u8, u32, f32)> {
        self.inner
            .get_results()
            .into_iter()
            .map(|r| (r.action, r.visit_count, r.q_value))
            .collect()
    }

    /// Get visit count distribution as 10-element probability vector
    #[pyo3(signature = (temperature=1.0))]
    fn get_action_probs(&self, temperature: f32) -> [f32; 10] {
        self.inner.get_action_probs(temperature)
    }

    fn sims_completed(&self) -> u32 {
        self.inner.sims_completed()
    }

    /// Set opponent action weights for smarter expansion sampling.
    /// opp_weights: list of (action_int, probability)
    fn set_opp_weights(&mut self, weights: Vec<(u8, f32)>) {
        self.inner.set_opp_weights(weights);
    }
}

// ============================================================
// REGISTER
// ============================================================

pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyBattleState>()?;
    m.add_class::<PyPlayerState>()?;
    m.add_class::<PySideConditions>()?;
    m.add_class::<PyPokemon>()?;
    m.add_class::<PyMoveSlot>()?;
    m.add_class::<PyMoveTemplate>()?;
    m.add_class::<PyMoveMeta>()?;
    m.add_class::<PyEvent>()?;
    m.add_class::<PyDataStore>()?;
    m.add_function(wrap_pyfunction!(create_battle, m)?)?;
    m.add_function(wrap_pyfunction!(type_effectiveness, m)?)?;
    m.add_function(wrap_pyfunction!(batch_resolve, m)?)?;
    m.add_function(wrap_pyfunction!(search_1ply, m)?)?;
    m.add_function(wrap_pyfunction!(search_2ply, m)?)?;
    m.add_function(wrap_pyfunction!(search_1ply_nn, m)?)?;
    m.add_function(wrap_pyfunction!(search_2ply_nn, m)?)?;
    m.add_function(wrap_pyfunction!(evaluate_position, m)?)?;
    m.add_class::<PyMctsContext>()?;
    m.add_function(wrap_pyfunction!(build_observation, m)?)?;
    m.add_function(wrap_pyfunction!(build_observations_batch, m)?)?;
    m.add_function(wrap_pyfunction!(mcts_search, m)?)?;
    Ok(())
}
