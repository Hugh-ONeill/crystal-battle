# Crystal Battle RL Agent -- Architecture

## Current Observation Space (1052 padded, 1050 used)

```
active(405) + my_team(6x63=378) + opp_team(6x42=252) + global(15) + padding(2)
```

- **Active (405):** HP/speed (5), my 4 moves x 38 features (152), opp 4 moves x 36 (144),
  damage summaries + KO flags (8), types 2x17 defensive profile (34), status 2x7 one-hot (14),
  confusion/status turns + leech seed (6), protect consecutive (2),
  base stats atk/def/spa/spd x2 (8), status count (2), type effectiveness (4),
  multi-turn move costs (4), consecutive active turns (2), stat stages (7x2=14),
  matchup summaries (6)
- **My move features (38):** type one-hot(17), accuracy, ailment_type, effectiveness,
  pp_frac, priority, damage_class, ailment_chance, flinch_chance, drain, healing,
  damage_frac, is_boost, boost_atk, boost_def, boost_spd, is_debuff, debuff_magnitude,
  STAB, high_crit, multi_hit_avg, ailment_can_land
- **Opp move features (36):** same as my moves minus STAB/high_crit/multi_hit
- **Matchup summaries (6):** my_se_count, opp_se_count, i_am_walled, opp_is_walled,
  my_status_options, best_bench_advantage
- **My Team (6x63):** fixed slots, is_active, alive, hp, defensive_profile(17), dmg/opp_dmg,
  speed, status_onehot(7), pp, predicted_dmg, derived flags(5), defensive_eff,
  best_move_eff, spikes_entry, switch_in_cost, matchup_improvement, base_stats(4),
  move_type_coverage(17)
- **Opp Team (6x42):** is_active, alive, hp, defensive_profile(17), dmg/my_dmg, speed,
  status_onehot(7), pp, derived flags(5), base_stats(4), my_best_eff, move_type_count
- **Global (15):** turn, alive counts, HP fracs, spikes, screens, weather(3+turns)

### Type Encoding: Defensive Profiles

Type identity (one-hot) replaced with defensive type profile: each element is the
combined effectiveness of that attack type against the mon, normalized by /4.0.
Encodes vulnerability structure directly -- mons with similar weaknesses get similar
encodings, giving the network useful inductive bias for type reasoning.

### Status Landing Feature

Per-move `ailment_can_land` encodes whether the move's status effect can land on the
target (1.0=yes, 0.0=immune or already statused). Checks type immunities (Steel/Poison
immune to poison, Electric immune to paralysis, Fire immune to burn, Ice immune to freeze)
and whether the target already has a non-volatile status.

### Action Space (10 discrete)

```
0-3: use move slot 0-3
4-9: switch to team[0]-team[5] (active slot always masked)
```

Fixed team slots -- each mon keeps its index the entire match. Action masking disables
invalid moves (no PP, fainted mons, sleep-conditional moves). Sleep Talk/Snore masked
when user not asleep, Dream Eater masked when target not asleep.

### Key Design Decisions

- **Fixed team slots:** mons keep their team index throughout the match (no shifting).
  Action 4+i always means the same mon, giving the policy stable representations.
- **Shared move encoders:** same Linear(38->32) applied to each of the 4 my-move slots
  independently; same Linear(36->32) for opp-move slots. Eliminates move-slot position
  bias -- the network must evaluate moves by their features, not by which slot they occupy.
- **Shared team encoders:** same Linear applied to each team slot independently,
  so the model generalizes across team positions
- **Cross-attention:** active matchup queries the team --
  "given what I'm facing, which team mon matters most?"
- **Fainted-mon masking:** attention scores masked to -inf for fainted/empty slots
- **LSTM for temporal reasoning:** remembers opponent's previous moves, switches, revealed info
- **Actor-critic split:** shared LSTM feeds two separate MLP heads

## Current Architecture: v1 with Shared Move Encoders

```
                     Active Section (405)
     hp_speed(5) | my_moves(4x38) | opp_moves(4x36) | rest(104)
         |              |                |                |
         |        reshape(4,38)    reshape(4,36)          |
         |              |                |                |
         |        Linear(38->32)   Linear(36->32)         |
         |         shared x4        shared x4             |
         |              |                |                |
         |          mean pool        mean pool            |
         |           (32d)            (32d)               |
         |              |                |                |
         +----+---------+--------+-------+--------+------+
              |    Concat (5+32+32+104=173)               |
              +-------------------------------------------+
                                 |
                          Linear(173->256)
                              + ReLU
                                 |
                          active_enc (256d)
                                 |
                   +--- query ---+--- query ---+
                   |                           |
             my_team(6x64)              opp_team(6x64)
             cross-attention            cross-attention
             (single-head, d=64)        (single-head, d=64)
                   |                           |
             my_attn(64d)              opp_attn(64d)
                   |                           |
   active_enc(256) + my_move_flat(4x32=128) + my_attn(64) + opp_attn(64) + global(32)
                            |
                     Concat (256+128+64+64+32=544)
                            |
                      Linear(544->256) + ReLU
                            |
                     features (256d)
                            |
                       +----+----+
                       |  LSTM   |
                       |  256d   |
                       +----+----+
                            |
                     +------+------+
                     |             |
              Actor 256x256   Critic 256x256
              -> 10 actions   -> 1 value
              + action mask
```

Note: my_move_flat preserves per-slot embeddings (position-aware) for action routing,
while the active encoder uses mean-pooled moves (position-invariant) for matchup
understanding.

## Tested Variants

### v2: Multi-Head + Self-Attention (177k params)

Adds over v1:
- nn.MultiheadAttention(64, 4 heads) for cross-attention
- Self-attention within my team with LayerNorm + residual

Result: 43.5% peak vs MaxDamage. Underperformed v1.

### v3: Full Cross-Team Attention (211k params)

Adds over v2:
- Opp team self-attention
- Cross-team attention (my team -> opp team)
- 2-layer stacked architecture

Result: 47% peak vs MaxDamage. Underperformed v1.

### v4: Global-Conditioned Attention + FFN (324k params)

- 2-layer active encoder (405->256->128)
- Global conditioning: weather/screens injected into team representations before attention
- FFN after self-attention (standard transformer pattern)
- 2-head attention (lighter than v2/v3's 4-head)
- Both teams get self-attention

Not yet tested with current obs layout.

### Why v1 won

More parameters != better when training signal is limited. v1 provides the right
inductive bias ("attend to team given current matchup") without overwhelming the
optimizer. v2/v3's additional relationships are real in Pokemon but require more
data or pre-training to learn effectively.

## Training Setup

- **Algorithm:** MaskableRecurrentPPO (PPO + LSTM + action masking)
- **Reward:** shaped (HP diff delta * 0.1 + faint bonus 0.08 + terminal +1/-1)
  - Switch matchup reward tested (coef 0.02-0.05) but disabled: caused KL divergence
    explosion and entropy collapse, degrading overall performance
  - Status/setup bonuses tested but disabled: hurt critic convergence
- **Opponents:** phase 1 (random, 10% of steps) -> phase 2 (mixed MaxDmg/Smart) -> phase 3 (self-play)
  - `--skip-to-phase 2` recommended to avoid wasting steps on random
  - `--curriculum` mode available for graduated blend
- **Hyperparams:** lr 3e-4, gamma 0.99, ent_coef 0.02, clip 0.2, gae_lambda 0.95,
  n_steps 2048, n_epochs 10, 8 envs, net_arch 256x256
- **Self-play:** stochastic opponents (deterministic=False), pool of 20 snapshots
- **Eval:** 400 games per baseline, side-swapping (both_sides=True) to eliminate P1/P2 bias

### Reward Experiments

| Signal | Coefficient | Result |
|--------|------------|--------|
| HP diff delta | 0.1 | core signal, works well |
| Faint bonus | 0.08 | stable, helps KO prioritization |
| Switch matchup | 0.05 | KL explosion (0.13 vs 0.03), entropy collapse |
| Switch matchup | 0.02 | still destabilized, ~10% lower win rates than baseline |
| Status bonus | 0.0 | tested nonzero, hurt critic convergence |
| Setup bonus | 0.0 | tested nonzero, hurt critic convergence |

### P1/P2 Side Bias

Mirror match analysis (1600 games) found ~3-4% P2 advantage (Z=-2.85, statistically
significant). Side-swapping eval (`both_sides=True`) plays half games as each side to
eliminate this bias.

### Bot Strength Hierarchy (with engine fixes)

Measured via 400-game round-robin after engine fixes (run 40+):
```
Smart (69-71%) >> MaxDamage (51%) ~ Crystal (49%) >> Random (<4%)
```

## Engine Fixes (Run 40)

Major mechanics implemented mid-development that changed game dynamics:

1. **Charge/recharge moves**: Hyper Beam recharge, Solar Beam charge (instant in sun),
   Fly/Dig semi-invulnerability, Sky Attack/Skull Bash/Razor Wind charge
2. **Lock-in moves**: Thrash/Outrage/Petal Dance 2-3 turn lock + post-confusion
3. **Accuracy/evasion stages**: Sand Attack, Double Team etc. now affect hit chance
4. **Belly Drum HP cost**: Deducts 50% HP (was free +6 attack)
5. **Protect blocks status**: Toxic/Thunder Wave blocked by Protect
6. **Sandstorm Rock SpDef**: 1.5x SpDef for Rock types in sandstorm
7. **Hidden Power**: Type/power calculated from DVs per Gen 2 formula
8. **Team builder cleanup**: STAB in effective power, junk moves blacklisted,
   type coverage diversity, Dream Eater requires sleep move in moveset

## Imitation Learning (Run 45)

Pre-trained the full model (extractor + LSTM + policy head) on 5000 games of
SmartAgent play using sequential supervised learning:

- **Data**: 4715 game sequences, 104k (obs, action) pairs
- **Training**: Cross-entropy loss on full LSTM sequences (not individual frames)
- **Val accuracy**: 87.6% (vs 84.5% for feedforward-only pre-training)
- **Zero-shot performance**: 45% vs MaxDmg, 32% vs Smart (before any RL)
- **Key insight**: feedforward pre-training (no LSTM) failed at 3.5% vs MaxDmg
  because the untrained LSTM scrambled the pre-trained features

RL fine-tuning from this base reached plateau faster (~1.5M vs ~3M) but hit the
same ~50% Smart ceiling as pure RL runs.

## Opponent Modeling + Lookahead Search (Breakthrough)

Trained a separate opponent prediction model and used it for lookahead search:

### Opponent Model
- **Architecture**: Linear(OBS_SIZE->256->128) + LSTM(128) + Linear(128->10)
- **Training data**: 10k games of Smart/MaxDmg play, 280k (obs, opp_action) pairs
- **Val accuracy**: 83.4% (87-89% for moves, 64-74% for switches)

### Lookahead Results

| Depth | vs MaxDmg | vs Smart | Sims/turn |
|-------|-----------|----------|-----------|
| 0 (raw policy) | 45% | 32% | 1 |
| 1-ply | 69% | 61% | ~30 |
| 2-ply | 92% | 94% | ~900 |

**1-ply broke the 50% Smart ceiling.** 2-ply crushed both heuristic bots but is
too slow for training in Python.

### AlphaZero-Style Training Loop

```
repeat:
  1. play N games with 1-ply lookahead search (generates expert data)
  2. train policy to match search decisions (supervised on sequences)
  3. better policy -> better search -> better data -> repeat
```

This is the same approach that made AlphaGo/AlphaZero superhuman. The search
provides multi-turn reasoning that the raw policy can't learn from reward signals
alone, and training on search outputs distills that reasoning into the network.

## Run History

| Run | Architecture | Key Change              | Peak vs MaxDmg | Peak vs Smart | Notes |
|-----|-------------|--------------------------|----------------|---------------|-------|
| 10  | flat MLP    | opp move prediction obs  | 49%            | -             | plateaued ~46-48% |
| 13  | attn v1     | single-head cross-attn   | 52.5%          | -             | broke 50% MaxDmg ceiling |
| 17  | attn v1     | LSTM 256x2               | 53%            | -             | first recurrent model |
| 18  | attn v1     | OU tier constraint       | 65.5%          | -             | biggest jump, pure MaxDmg |
| 36  | attn v1     | 50/50 MaxDmg/Smart mix   | 55.7%          | 47.7%         | plateau at ~55% composite |
| 37  | attn v1     | switch matchup reward    | 45.5%          | 42.0%         | reward destabilized training |
| 40  | attn v1     | engine fixes + 256d ext  | 62.5%          | 50.7%         | engine fixes gave real bump |
| 41  | attn v1     | self-play at 2M          | 63.2%          | 52.7%         | self-play helped slightly |
| 42  | attn v1     | boost features + HP fix  | 63.0%          | 50.0%         | similar plateau |
| 43  | attn v1     | aggressive self-play     | -              | -             | early self-play, no improvement |
| 45  | imitation   | LSTM pre-train + RL      | 63.2%          | 52.7%         | faster to plateau, same ceiling |
| -   | + 1-ply     | lookahead search         | **69%**        | **61%**       | broke Smart ceiling |
| -   | + 2-ply     | deeper search            | **92%**        | **94%**       | crushes heuristic bots |

### PBT Results

Population-based training (6 members, 250k steps/gen, 400 eval games, fixed seed)
converged on HPs close to defaults: lr~3.5e-4, ent_coef~0.01, gamma=0.99, clip=0.15,
gae=0.9, epochs=8. HPs are not the bottleneck.

## Rust Engine

### Performance (measured)

| Search Depth | Sims/Turn | Python    | Rust       | Speedup | Practical?          |
|-------------|-----------|-----------|------------|---------|---------------------|
| 1-ply       | ~30       | ~50ms     | <1ms       | 280x    | Yes (training+eval) |
| 2-ply       | ~900      | ~1.5s     | ~6ms       | 511x    | Yes (training+eval) |
| 3-ply       | ~27,000   | ~45s      | ~180ms     | ~250x   | Eval only           |
| Native sim  | --        | 3.5k t/s  | 440k t/s   | 127x    | --                  |

### Phase 1 -- Core engine [DONE]

`crystal_engine/` Rust crate with PyO3 bindings (Python module: `crystal_engine_rs`).
14 modules, ~3400 LOC. Full Gen 2 battle sim: turn resolution, damage calc, all
status effects, weather, screens, spikes, multi-turn moves, protect, phazing.

Validated: type chart (5202 checks), stat calc, action masks (with immune filtering),
statistical win rate distribution. 11 Rust integration tests.

### Phase 2 -- Batch simulation [DONE]

Parallel simulation via rayon. Python API:
- `batch_resolve(state, action_pairs)` -- N sims in parallel from one root state
- `search_1ply(state, p1_actions, opp_actions)` -- full 1-ply search
- `search_2ply(state, p1_actions, opp_d1, opp_d2)` -- full 2-ply search
- `evaluate_position(state)` -- HP diff + alive diff heuristic

### Phase 3 -- MCTS with neural net evaluation [DONE]

`search.rs`: PUCT-based MCTS with iterative batched leaf evaluation.
Rust owns the tree and expands nodes via battle simulation. Leaf nodes
are batched and returned to Python for neural net inference, then values
and policy priors are propagated back into the tree.

Python-side components:
- `training/mcts_evaluator.py`: `MctsEvaluator` wraps the PPO model for
  batched (value, policy_prior) inference with zero-init LSTM
- `training/mcts_agent.py`: `MctsAgent` drives the MCTS loop per turn

Python API:
```python
mcts = ce.MctsContext(state, n_simulations=200, seed=0, c_puct=1.5)
while True:
    n = mcts.run_until_eval_needed(max_batch=32)
    if n == 0: break
    leaves = mcts.get_pending_leaf_states()
    values, priors = model.evaluate_batch(obs, masks)
    mcts.supply_evaluations(values, priors)
action_probs = mcts.get_action_probs(temperature=0.1)
```

Initial results (imitation_ppo weights, 100 sims, vs Smart):
- Heuristic eval: 12%
- NN eval: 22% (nearly 2x heuristic, but bottlenecked by FFI inference cost)

### Crate architecture

```
crystal_engine/
  src/
    lib.rs            module declarations + PyO3 entry point
    types.rs          17-type enum, compile-time type chart, Hidden Power
    moves.rs          MoveTemplate, MoveSlot, MoveMeta, Struggle
    actions.rs        Action enum (UseMove, Switch, Struggle, Forfeit)
    events.rs         27 event types as enum
    pokemon.rs        Pokemon struct, Gen 2 stat formula, volatiles
    status.rs         Status effects, prevention, confusion, residual damage
    stat_stages.rs    Stage multipliers, move stat effect table
    damage.rs         Gen 2 damage formula (integer math) + expected damage
    player.rs         PlayerState, SideConditions, valid_actions, action masks
    battle.rs         BattleState container, weather, winner check
    turn_engine.rs    resolve_turn() -- the full battle loop (~950 lines)
    batch.rs          batch_resolve, search_1ply, search_2ply (rayon parallel)
    search.rs         MCTS with PUCT selection + batched NN eval callbacks
    data.rs           JSON loader for pokemon.json / moves.json
    pybridge.rs       PyO3 bindings for all public types + functions
```
