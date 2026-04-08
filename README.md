# Crystal Battle

A competitive Pokemon Gen 2 OU battle agent built from scratch, combining reinforcement learning, Monte Carlo Tree Search, and neural network-guided search.

## Overview

Crystal Battle is a full-stack AI system for playing competitive Pokemon Crystal (Gen 2 OU tier). It includes:

- A **Python battle engine** implementing Gen 2 mechanics (damage calc, status, type chart, items, switching)
- A **Rust engine** (PyO3) for high-performance game simulation and search
- **Gymnasium environments** for RL training with action masking, observation encoding, and self-play
- **MCTS and expectiminimax search** with neural network policy priors (AlphaZero-style PUCT)
- A **Showdown integration layer** for playing on [Pokemon Showdown](https://pokemonshowdown.com/) servers
- **Training pipelines** for policy networks, value networks, imitation learning, and population-based training

## Architecture

```
crystal-battle/
  engine/           Python battle engine (turn resolution, damage, status, types)
  crystal_engine/   Rust engine rewrite (PyO3 bindings, ~10x faster)
  gym_env/          Gymnasium env wrapper (obs encoding, action masking, rewards)
  training/         RL training (PPO, imitation, search, transformers, PBT)
  showdown/         Showdown adapter (poke-env, MCTS player, feature extraction)
  tools/            Benchmarking, profiling, replay viewer
  data/             Pokemon/move JSON data
  tests/            Unit tests
```

### Search

Two search algorithms, both integrated with the Showdown bot:

- **MCTS** with optional PUCT policy priors -- width-based search, fast evaluation, good for real-time play. Policy network outputs bias exploration via AlphaZero-style UCB selection.
- **Expectiminimax** with iterative deepening and alpha-beta pruning -- depth-based search reaching depth 4-6 at 5s. Better strategic decisions (values Hypnosis, Spikes setup) but more conservative.

Both use a handcrafted evaluation function with matchup-aware boost scoring and type effectiveness checks.

### Policy Network Pipeline

1. **Data generation**: self-play MCTS games (Rust driver, rayon-parallelized) + human replay scraping from Showdown
2. **Feature extraction**: 579-dimensional state representation encoding active moves (type, power, accuracy, STAB, effectiveness, effect flags), per-pokemon stats with boost application, side conditions, and weather
3. **Training**: policy net learns to predict MCTS visit distributions (soft KL divergence targets)
4. **Inference**: ONNX export for Rust-side inference during MCTS (temperature-scaled softmax priors)

### RL Training

- MaskablePPO and MaskableRecurrentPPO (LSTM) via Stable Baselines 3
- 1052-dimensional observation space with defensive type profiles, move-level features, and matchup summaries
- Self-play curriculum with opponent pool
- Population-based training for hyperparameter search
- Imitation learning from search trajectories (Expert Iteration)

## Showdown Integration

The bot connects to Pokemon Showdown via [poke-env](https://github.com/hsahovic/poke-env) and plays Gen 2 OU:

```bash
# play against humans on a local Showdown server
python showdown/poke_engine_player.py --local --wait --search emm --search-ms 5000

# MCTS with policy priors
python showdown/poke_engine_player.py --local --wait --search mcts --search-ms 1000
```

Features:
- Translates Showdown game state to poke-engine's internal representation
- Opponent team prediction using Smogon usage statistics
- Intelligent forced switch scoring (type matchups, resistances, HP, status)
- Configurable search algorithm and time budget

## Results

- **Policy-guided MCTS**: 62% cross-team winrate against 10 GSC OU archetypes
- **Temperature scaling** (T=5.0) balances policy confidence vs search exploration -- wins 6/10 matchups vs 4/10 at T=1.0
- **Value network**: 88% accuracy predicting game outcomes from position (CPU-bound, pending GPU for real-time use)
- **Expectiminimax** reaches depth 4-6 in 5s, makes better strategic decisions than MCTS (uses Hypnosis, values Spikes) but occasionally too conservative

## Setup

```bash
# Python dependencies
pip install -r requirements.txt

# Rust engine (optional, for fast simulation)
cd crystal_engine && maturin develop --release

# poke-engine (optional, for MCTS/EMM search)
# see github.com/pmariglia/poke-engine
cd /path/to/poke-engine && pip install -e poke-engine-py
```

Requires Python 3.10+, Rust toolchain (for crystal_engine or poke-engine).

## Key Files

| File | Description |
|---|---|
| `showdown/poke_engine_player.py` | Main Showdown bot (MCTS/EMM search) |
| `showdown/features_v2.py` | 579-dim move-aware feature extraction |
| `showdown/policy_train.py` | Policy network training + ONNX export |
| `showdown/replay_to_training.py` | Showdown replay to training data converter |
| `showdown/scrape_replays.py` | Replay downloader from Showdown API |
| `training/train.py` | Main RL training loop (PPO + self-play) |
| `training/mcts_agent.py` | MCTS agent for RL evaluation |
| `training/rust_search_agent.py` | Rust-backed search agent |
| `gym_env/battle_env.py` | Gymnasium environment |
| `gym_env/obs_builder.py` | Observation space encoding (1052 dims) |
| `engine/turn_engine.py` | Python battle engine |
| `tools/replay.py` | Battle replay viewer |
