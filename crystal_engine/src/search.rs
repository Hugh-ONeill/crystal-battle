// MCTS search with batched neural net evaluation
//
// Architecture:
//   1. Rust owns the tree, expands nodes via battle simulation
//   2. Leaf nodes that need NN evaluation are collected into a batch
//   3. Python runs batched inference and returns (value, policy_priors)
//   4. Rust backpropagates and continues expanding
//
// API (called from Python):
//   MctsContext::new(root_state, n_simulations)
//   MctsContext::run_until_eval_needed() -> Option<Vec<LeafState>>
//   MctsContext::supply_evaluations(values, priors)
//   MctsContext::get_action_probs() -> Vec<(action, visit_fraction)>

use rand::Rng;
use rand::SeedableRng;
use rand::rngs::SmallRng;

use crate::actions::Action;
use crate::battle::BattleState;
use crate::player::PlayerState;
use crate::turn_engine;

// ============================================================
// MCTS NODE
// ============================================================

struct MctsNode {
    // tree structure
    parent: Option<usize>,     // index into arena
    children: Vec<usize>,      // indices into arena
    action: Option<(u8, u8)>,  // (p1_action, p2_action) that led here

    // MCTS stats
    visit_count: u32,
    value_sum: f32,
    prior: f32,               // policy prior from NN (for P1's action)

    // state (only stored for expanded nodes)
    state: Option<BattleState>,
    is_terminal: bool,
    terminal_value: f32,

    // which P1 action this node represents (from parent's perspective)
    p1_action: u8,
}

impl MctsNode {
    fn q_value(&self) -> f32 {
        if self.visit_count == 0 {
            0.0
        } else {
            self.value_sum / self.visit_count as f32
        }
    }
}

// ============================================================
// MCTS CONTEXT
// ============================================================

pub struct MctsContext {
    arena: Vec<MctsNode>,
    root_idx: usize,
    root_state: BattleState,
    n_simulations: u32,
    sims_done: u32,
    rng: SmallRng,

    // PUCT hyperparams
    c_puct: f32,

    // dirichlet noise at root (alpha=0.3, epsilon=0.25 are AlphaZero defaults)
    dirichlet_alpha: f32,
    dirichlet_epsilon: f32,
    root_noise_applied: bool,

    // opponent action weights (from opponent model, or uniform if not provided)
    opp_weights: Option<Vec<(u8, f32)>>,

    // pending leaf evaluations
    pending_leaves: Vec<PendingLeaf>,
}

pub struct PendingLeaf {
    pub node_idx: usize,
    pub state: BattleState,
}

/// Info about a leaf node that Python needs to evaluate
pub struct LeafInfo {
    pub p1_hp_frac: f32,
    pub p2_hp_frac: f32,
    pub p1_alive: u8,
    pub p2_alive: u8,
    pub is_over: bool,
    pub winner: Option<u8>,
    pub turn: u16,
    pub weather: Option<String>,
    pub weather_turns: u8,
}

/// Result returned after search completes
pub struct SearchResult {
    pub action: u8,
    pub visit_count: u32,
    pub q_value: f32,
}

impl MctsContext {
    pub fn new(root_state: BattleState, n_simulations: u32, seed: u64, c_puct: f32) -> Self {
        let is_terminal = root_state.is_over();
        let terminal_value = if is_terminal {
            match root_state.winner {
                Some(1) => 1.0,
                Some(2) => -1.0,
                _ => 0.0,
            }
        } else {
            0.0
        };

        let root = MctsNode {
            parent: None,
            children: Vec::new(),
            action: None,
            visit_count: 0,
            value_sum: 0.0,
            prior: 1.0,
            state: Some(root_state.clone()),
            is_terminal,
            terminal_value,
            p1_action: 255, // sentinel
        };

        let mut arena = Vec::with_capacity(n_simulations as usize * 10);
        arena.push(root);

        MctsContext {
            arena,
            root_idx: 0,
            root_state,
            n_simulations,
            sims_done: 0,
            rng: SmallRng::seed_from_u64(seed),
            c_puct,
            dirichlet_alpha: 0.3,
            dirichlet_epsilon: 0.25,
            root_noise_applied: false,
            opp_weights: None,
            pending_leaves: Vec::new(),
        }
    }

    /// Set opponent action weights (from opponent model).
    /// Each entry is (action_int, probability). Used during expansion to
    /// weight which opponent action each child is simulated against.
    pub fn set_opp_weights(&mut self, weights: Vec<(u8, f32)>) {
        self.opp_weights = Some(weights);
    }

    /// Run MCTS iterations until we have a batch of leaves needing NN eval,
    /// or until all simulations are done.
    /// Returns the number of leaves needing evaluation.
    pub fn run_until_eval_needed(&mut self, max_batch: usize) -> usize {
        self.pending_leaves.clear();

        while self.sims_done < self.n_simulations && self.pending_leaves.len() < max_batch {
            let leaf_idx = self.select();

            // terminal node: backprop immediately
            if self.arena[leaf_idx].is_terminal {
                let value = self.arena[leaf_idx].terminal_value;
                self.backpropagate(leaf_idx, value);
                self.sims_done += 1;
                continue;
            }

            // already expanded with children: shouldn't happen in select,
            // but if it does, just backprop with current Q
            if !self.arena[leaf_idx].children.is_empty() {
                let value = self.arena[leaf_idx].q_value();
                self.backpropagate(leaf_idx, value);
                self.sims_done += 1;
                continue;
            }

            // leaf needs evaluation -- expand it and queue for NN eval
            let state = self.arena[leaf_idx].state.clone().unwrap();
            self.expand(leaf_idx, &state);
            self.pending_leaves.push(PendingLeaf {
                node_idx: leaf_idx,
                state,
            });
        }

        self.pending_leaves.len()
    }

    /// Supply NN evaluations for pending leaves.
    /// `values`: one value per pending leaf (from P1's perspective, -1 to 1)
    /// `priors`: one [f32; 10] policy prior per pending leaf (over action space)
    pub fn supply_evaluations(&mut self, values: &[f32], priors: &[[f32; 10]]) {
        assert_eq!(values.len(), self.pending_leaves.len());
        assert_eq!(priors.len(), self.pending_leaves.len());

        // collect all work items first to avoid borrow conflicts
        let work: Vec<(usize, f32, [f32; 10])> = self
            .pending_leaves
            .drain(..)
            .enumerate()
            .map(|(i, leaf)| (leaf.node_idx, values[i], priors[i]))
            .collect();

        for (node_idx, value, node_priors) in work {
            // set child priors -- collect child indices first
            let children: Vec<usize> = self.arena[node_idx].children.clone();
            for child_idx in &children {
                let p1_action = self.arena[*child_idx].p1_action as usize;
                if p1_action < 10 {
                    self.arena[*child_idx].prior = node_priors[p1_action];
                }
            }

            // apply Dirichlet noise to root children (once)
            if node_idx == self.root_idx && !self.root_noise_applied && self.dirichlet_epsilon > 0.0 {
                self.root_noise_applied = true;
                let noise = dirichlet_sample(children.len(), self.dirichlet_alpha, &mut self.rng);
                let eps = self.dirichlet_epsilon;
                for (i, &child_idx) in children.iter().enumerate() {
                    let p = self.arena[child_idx].prior;
                    self.arena[child_idx].prior = (1.0 - eps) * p + eps * noise[i];
                }
            }

            self.backpropagate(node_idx, value);
            self.sims_done += 1;
        }
    }

    /// Supply heuristic evaluations (no NN, use hp_diff + alive_diff)
    pub fn supply_heuristic_evaluations(&mut self) {
        let work: Vec<(usize, f32)> = self
            .pending_leaves
            .drain(..)
            .map(|leaf| {
                let value = crate::batch::evaluate_position(&leaf.state);
                (leaf.node_idx, value)
            })
            .collect();

        for (node_idx, value) in work {
            let children: Vec<usize> = self.arena[node_idx].children.clone();
            let n = children.len();
            if n > 0 {
                let uniform = 1.0 / n as f32;
                for child_idx in children {
                    self.arena[child_idx].prior = uniform;
                }
            }

            self.backpropagate(node_idx, value);
            self.sims_done += 1;
        }
    }

    /// Get pending leaf states for Python to build observations from
    pub fn get_pending_leaves(&self) -> &[PendingLeaf] {
        &self.pending_leaves
    }

    /// Get search results: (action, visits, q_value) sorted by visits descending
    pub fn get_results(&self) -> Vec<SearchResult> {
        let root = &self.arena[self.root_idx];
        let mut results: Vec<SearchResult> = root
            .children
            .iter()
            .map(|&ci| {
                let child = &self.arena[ci];
                SearchResult {
                    action: child.p1_action,
                    visit_count: child.visit_count,
                    q_value: child.q_value(),
                }
            })
            .collect();
        results.sort_by(|a, b| b.visit_count.cmp(&a.visit_count));
        results
    }

    /// Get visit count distribution as action probabilities (for training targets)
    pub fn get_action_probs(&self, temperature: f32) -> [f32; 10] {
        let root = &self.arena[self.root_idx];
        let mut counts = [0.0f32; 10];

        for &ci in &root.children {
            let child = &self.arena[ci];
            let a = child.p1_action as usize;
            if a < 10 {
                if temperature <= 0.01 {
                    counts[a] = child.visit_count as f32;
                } else {
                    counts[a] = (child.visit_count as f32).powf(1.0 / temperature);
                }
            }
        }

        let total: f32 = counts.iter().sum();
        if total > 0.0 {
            for c in counts.iter_mut() {
                *c /= total;
            }
        }
        counts
    }

    pub fn sims_completed(&self) -> u32 {
        self.sims_done
    }

    // ---- internal ----

    /// PUCT selection: walk down the tree picking the child with highest UCB
    fn select(&self) -> usize {
        let mut node_idx = self.root_idx;

        loop {
            let node = &self.arena[node_idx];

            // leaf: no children yet
            if node.children.is_empty() {
                return node_idx;
            }

            // terminal: return this node
            if node.is_terminal {
                return node_idx;
            }

            // pick child with highest PUCT score
            let parent_visits = node.visit_count as f32;
            let sqrt_parent = parent_visits.sqrt();
            let mut best_score = f32::NEG_INFINITY;
            let mut best_child = node.children[0];

            for &ci in &node.children {
                let child = &self.arena[ci];
                let q = child.q_value();
                let u = self.c_puct * child.prior * sqrt_parent / (1.0 + child.visit_count as f32);
                let score = q + u;
                if score > best_score {
                    best_score = score;
                    best_child = ci;
                }
            }

            node_idx = best_child;
        }
    }

    /// Expand a leaf node: create children for all valid (p1_action, opp_action) combos.
    /// We expand only P1 actions as children; opponent actions are sampled during simulation.
    fn expand(&mut self, node_idx: usize, state: &BattleState) {
        let p1_mask = state.p1.valid_action_mask_filtered(Some(&state.p2), true);
        let p2_mask = state.p2.valid_action_mask_filtered(Some(&state.p1), true);
        let p1_valid: Vec<u8> = (0..10).filter(|&i| p1_mask[i as usize]).collect();
        let p2_valid: Vec<u8> = (0..10).filter(|&i| p2_mask[i as usize]).collect();

        if p1_valid.is_empty() || p2_valid.is_empty() {
            return;
        }

        let uniform_prior = 1.0 / p1_valid.len() as f32;

        for &a1 in &p1_valid {
            // sample opponent action: use weighted sampling if opp_weights provided
            let a2 = self.sample_opp_action(&p2_valid);

            let mut child_state = state.clone();
            child_state.rng = SmallRng::seed_from_u64(self.rng.random());
            turn_engine::resolve_turn(
                &mut child_state,
                decode_action(a1),
                decode_action(a2),
            );
            handle_forced_switches(&mut child_state);

            let is_terminal = child_state.is_over();
            let terminal_value = if is_terminal {
                match child_state.winner {
                    Some(1) => 1.0,
                    Some(2) => -1.0,
                    _ => 0.0,
                }
            } else {
                0.0
            };

            let child = MctsNode {
                parent: Some(node_idx),
                children: Vec::new(),
                action: Some((a1, a2)),
                visit_count: 0,
                value_sum: 0.0,
                prior: uniform_prior, // overwritten by NN priors if supplied
                state: Some(child_state),
                is_terminal,
                terminal_value,
                p1_action: a1,
            };

            let child_idx = self.arena.len();
            self.arena.push(child);
            self.arena[node_idx].children.push(child_idx);
        }
    }

    /// Sample an opponent action, weighted by opp_weights if available
    fn sample_opp_action(&mut self, valid: &[u8]) -> u8 {
        if let Some(ref weights) = self.opp_weights {
            // weighted sampling: filter to valid actions, normalize, sample
            let mut candidates: Vec<(u8, f32)> = Vec::new();
            let mut total = 0.0f32;
            for &(a, w) in weights {
                if valid.contains(&a) && w > 0.001 {
                    candidates.push((a, w));
                    total += w;
                }
            }
            if !candidates.is_empty() && total > 0.0 {
                let roll: f32 = self.rng.random_range(0.0..total);
                let mut cumulative = 0.0;
                for &(a, w) in &candidates {
                    cumulative += w;
                    if roll < cumulative {
                        return a;
                    }
                }
                return candidates.last().unwrap().0;
            }
        }
        // fallback: uniform random
        valid[self.rng.random_range(0..valid.len())]
    }

    /// Backpropagate a value up the tree
    fn backpropagate(&mut self, mut node_idx: usize, value: f32) {
        loop {
            let node = &mut self.arena[node_idx];
            node.visit_count += 1;
            node.value_sum += value;

            if let Some(parent) = node.parent {
                node_idx = parent;
            } else {
                break;
            }
        }
    }
}

// ============================================================
// HELPERS
// ============================================================

fn decode_action(a: u8) -> Action {
    match a {
        0..=3 => Action::UseMove(a),
        4..=9 => Action::Switch(a - 4),
        10 => Action::Struggle,
        _ => Action::Struggle,
    }
}

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

/// Sample from a Dirichlet distribution using the gamma trick.
/// Dirichlet(alpha, ..., alpha) for `n` dimensions.
fn dirichlet_sample(n: usize, alpha: f32, rng: &mut SmallRng) -> Vec<f32> {
    if n == 0 {
        return vec![];
    }
    // gamma(alpha, 1) samples via Marsaglia-Tsang for alpha >= 1,
    // or rejection for alpha < 1. For simplicity, use the approximation:
    // gamma(alpha) ~ (U1^(1/alpha)) * (-ln(U2)) when alpha is small
    // But for alpha=0.3, a simpler approach: use exponential samples scaled
    let mut samples = Vec::with_capacity(n);
    let mut total = 0.0f32;
    for _ in 0..n {
        // gamma(alpha, 1) for small alpha: use U^(1/alpha) * Exp(1)
        let u: f32 = rng.random_range(0.001..1.0f32);
        let e: f32 = -rng.random_range(0.001..1.0f32).ln(); // Exp(1)
        let g = u.powf(1.0 / alpha) * e;
        samples.push(g);
        total += g;
    }
    if total > 0.0 {
        for s in &mut samples {
            *s /= total;
        }
    } else {
        let uniform = 1.0 / n as f32;
        for s in &mut samples {
            *s = uniform;
        }
    }
    samples
}
