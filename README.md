# Crystal Battle

A competitive Pokemon battling agent. Monte Carlo Tree Search over a Rust battle
engine, statistical opponent modeling, in-battle set inference, and a live
[Pokemon Showdown](https://pokemonshowdown.com/) player, currently focused on
Gen 9 OU and Gen 9 Monotype.

The project grew out of from-scratch reinforcement learning experiments (own
Gen 2 engine, Gymnasium environments, PPO) and uses
[poke-engine](https://github.com/pmariglia/poke-engine) as its search backend.
Much of the current design (sampled opponent worlds, observational set
inference, searched team preview and forced switches) was arrived at
independently through loss analysis, and only afterwards turned out to mirror
[foul-play](https://github.com/pmariglia/foul-play), the winner of the
NeurIPS 2025 PokeAgent Challenge Gen 9 OU bracket, which shares that engine.
The study is deliberate now: stock foul-play is the benchmark every change is
measured against, in automated head-to-head series on a local Showdown server,
with its remaining architectural edges cataloged and worked through one by one.

## The live player (`showdown/gen9_player.py`)

A [poke-env](https://github.com/hsahovic/poke-env) player that plays real games
end to end: team preview, tera, choice locks, forced switches, timers.

**Full-fidelity state translation** (`showdown/gen9_translator.py`). Every turn
the poke-env battle state becomes a complete engine state: hazards, screens,
boosts, volatiles, weather, terrain, trick room, terastallization, and
choice-lock move restrictions for both sides. Validated against 200 real ladder
replays: 4,917/4,917 turns translate cleanly and every state is accepted by
search.

**Sampled opponent worlds.** Unknown opponent details are not guessed once and
trusted. Each turn the player samples K plausible completions of the opponent's
team from Smogon chaos statistics (sets, spreads, items, tera types, and the
unrevealed teammates via usage and teammate correlation), runs one MCTS per
world, and merges the visit distributions. The last world is deliberately
speed-pessimistic (fastest plausible spreads, Choice Scarf whenever it has
non-trivial usage) so the search hedges against scarf sweeps before any
evidence exists.

**Observational set inference** (`showdown/set_inference.py`). The player
updates its beliefs from what actually happens:

- Speed floors: if an opponent moves first at equal priority (guarded for
  trick room, boosts, paralysis, tailwind), its modeled set must be fast
  enough; contradictions promote Choice Scarf, then max-speed spreads.
- Speed ceilings: moving slower than modeled drops a wrongly inferred scarf,
  then clamps the raw stat, which also covers Iron Ball-class items.
- Damage brackets: a non-crit, boost-free hit that beats the modeled set's
  maximum roll proves a boosting item. The weakest consistent explanation is
  chosen: Life Orb, Expert Belt (boost confined to super-effective hits),
  a type item (boost confined to one move type), or Choice Band/Specs.

**Searched everything.** Team preview runs a 6x6 MCTS maximin over our leads
against the opponent's predicted leads. Forced switches (post-KO and pivot)
are searched like any other decision rather than picked by heuristic.

```bash
# spar against another bot or a human on a local Showdown server
.venv/bin/python showdown/gen9_player.py --local --username MyBot \
    --mode accept --format gen9ou --team showdown/teams/gen9ou_sample.txt \
    --search-ms 300 --set-samples 2

# connect to the PokeAgent Challenge ladder
.venv/bin/python showdown/gen9_player.py --server pokeagent \
    --username PAC-MyBot --password ... --mode ladder --format gen9ou
```

## Benchmarking against foul-play

`showdown/bench/` holds the automated A/B harness results: mirror-team series
against stock foul-play on a local server, equal search budgets, systemd-run
units, and a loss-trace analyzer (`showdown/loss_trace.py`) that aggregates
eval-score cliffs, faint attribution, tera timing, and first-blood stats across
hundreds of games to find what actually loses games.

Snapshot (2026-07-12): from 0% (first broken build) to a confirmed 29%
[23, 36] win rate over n=200 against the tournament-winning baseline, with the
remaining gap decomposed by loss-trace analysis into named, in-progress fixes.
Notable findings along the way, each worth a post-mortem of its own:

- Filling unrevealed opponent slots with fainted dummies made the engine
  believe it was always winning (eval +326 vs an honest -407) and cost every
  game; predicted fills fixed it.
- poke-env reports unrevealed items as the truthy string `"unknown_item"`,
  which silently made every opponent itemless in the engine state.
- Doubling the search budget changed nothing (compute is not the bottleneck);
  fixing what the search believes changed everything.

## Monotype research pipeline (`monotype/`)

A parallel research track targeting Gen 9 Monotype (the main-ladder deployment
format), built on round-robin self-play benches:

- Per-type opponent modeling from Smogon monotype stats: canonical sets,
  chaos priors, and a 1,425-species type map, all conditioned on the
  opponent's revealed monotype.
- A trained team-preview lead picker (32% top-1 on human replay leads vs 21%
  paste-order baseline) and a turn-level move prediction net (55% top-1).
- An exhaustive simultaneous-move minimax endgame solver with a node budget
  (stall endgames otherwise grow the memo without bound; this one OOM'd a
  desktop twice before being tamed).
- Team rosters engineered for the engine as pilot (`teams/teams_engine.txt`)
  with A/B rebuild tooling, blunder auditing, and ladder-weighted standings.
- Replay scraping and per-type composition statistics beyond what Smogon
  publishes (3,399 scraped gen9monotype replays).

The honest negative results are documented too: self-play policy distillation
hit a ~+10pp wall that more data, AlphaZero-style iteration, and value nets at
leaves all failed to break, which is what motivated the pivot to live play.

## Origins: RL from scratch (Gen 2)

The project began as a Gen 2 OU reinforcement learning agent and that stack
still works: a from-scratch
Python battle engine (`engine/`), a Rust rewrite (`crystal_engine/`), Gymnasium
environments with action masking (`gym_env/`), PPO and imitation training
(`training/`), and a Gen 2 Showdown bot (`showdown/poke_engine_player.py`) with
MCTS and expectiminimax search behind policy/value networks.

## Architecture

```
crystal-battle/
  showdown/         Gen 9 live player, translator, set inference, benches,
                    replay tooling, gen2 bot, net training pipelines
  monotype/         Monotype research: benches, nets, endgame solver,
                    chaos priors, team tooling, Smogon stats
  engine/           Python Gen 2 battle engine
  crystal_engine/   Rust Gen 2 engine (PyO3)
  gym_env/          Gymnasium env wrapper
  training/         RL training (PPO, imitation, PBT)
  tools/            Benchmarking, profiling, replay viewer
  data/             Pokemon/move JSON data
  tests/            Unit tests (translator suite runs real protocol traces)
```

## Setup

```bash
python -m venv .venv
.venv/bin/pip install -r requirements.txt

# poke-engine, built from source with gen9 features
# see github.com/pmariglia/poke-engine
cd /path/to/poke-engine && pip install -e poke-engine-py

# optional: local Showdown server for sparring
git clone https://github.com/smogon/pokemon-showdown
cd pokemon-showdown && npm ci && node pokemon-showdown start 8000 --no-security
```

Requires Python 3.11+, a Rust toolchain, and Node for the local server.

## Key files

| File | Description |
|---|---|
| `showdown/gen9_player.py` | Gen 9 live player (sampled-world MCTS) |
| `showdown/gen9_translator.py` | poke-env battle to engine state translation |
| `showdown/set_inference.py` | In-battle speed/damage set refinement |
| `showdown/chaos_stats.py` | Smogon chaos stats: prediction and sampling |
| `showdown/validate_gen9_translator.py` | Replay-driven translator validation |
| `showdown/loss_trace.py` | Cross-game loss pattern analyzer |
| `showdown/bench_monotype.py` | Monotype round-robin self-play bench |
| `monotype/endgame_solver.py` | Budgeted simultaneous-move minimax |
| `monotype/lead_picker.py` | Team preview lead selection (MCTS + net) |
| `monotype/chaos_priors.py` | Per-type opponent move priors |
| `showdown/poke_engine_player.py` | Gen 2 Showdown bot (MCTS/EMM) |
| `training/train.py` | Gen 2 RL training loop |

## Credits

- [poke-engine](https://github.com/pmariglia/poke-engine) and
  [foul-play](https://github.com/pmariglia/foul-play) by pmariglia: the search
  engine underneath everything, and the benchmark to beat.
- [poke-env](https://github.com/hsahovic/poke-env) for Showdown connectivity.
- [Smogon](https://www.smogon.com/stats/) usage statistics.
