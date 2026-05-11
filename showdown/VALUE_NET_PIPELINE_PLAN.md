# Gen9 Value-Net Data Pipeline — Implementation Plan

This is a self-contained spec for an agent to implement the gen9 value-net training pipeline.
The session that produced this plan got the scraper updated and a parser MVP working;
the remaining ~2-4 hours of work is state reconstruction, replay processing, training, and benching.

---

## Goal

Train a value network on gen9ou human replay data, then verify it improves MCTS play vs.
the v3 hand-coded eval. Concretely: produce an ONNX model loadable by `mcts_with_value` that,
at some α ∈ (0, 1], beats α=0 (pure hand-coded eval) by ≥5pp on the 4-matchup canonical bench.

Success means a working training pipeline + a value net that adds signal. Failure (no α improves
play) is also useful — it tells us the bottleneck is featurization or architecture, not data.

---

## Context: where things stand as of 2026-05-01

- **v3 hand-coded eval** is the current production: `engine-ref-gen9-v3` (poke-engine commit `84132bb`),
  hazard rebalance committed. Both `poke_engine` (dev) and `poke_engine_ref` are aligned at v3.
- **Pipeline stubs exist** but most are gen2-locked. The list below distinguishes what's already
  gen-agnostic, what's gen2-locked needing port, and what's missing.
- **Scrape complete**: `showdown/replays/gen9ou/` contains **2230 replays** at min-rating 1300
  (~25 turns/game avg → ~55k state-positions if 100% parse, realistically ~30-40k after
  filtering). Re-run scraper if more data is wanted: `python showdown/scrape_replays.py
  --format gen9ou --pages 200 --min-rating 1300 --download` will paginate further back in time.

### Inventory

**Already gen-agnostic, ready to use:**
- `showdown/value_train.py` — `ValueNet` MLP (3 layers × 256 hidden), trainer with `--features-v2`
- `showdown/features_v2.py` — `parse_state_v2()` featurizes engine state strings to 579-dim vectors
- `showdown/local_battle.py` — `build_pe_state_gen9(team1_str, team2_str)` builds engine states from
  Showdown-format team imports
- `poke_engine.mcts_with_value(state, value_net, duration_ms, alpha=...)` — Rust-side MCTS with
  value-net leaf eval, ONNX-loadable. Expects `value_net` as a `ValueNet` Python object (existing wrapper).

**Already updated for gen9 (this session):**
- `showdown/scrape_replays.py` — `--format` arg, default gen9ou. Downloads to `replays/<format>/`.
- `showdown/replay_parse_gen9.py` — MVP parser. Extracts per-turn voluntary actions
  (`p1_action`, `p2_action`), winner, revealed team specs (species, moves, item-if-shown,
  ability-if-shown, tera). **Validated** on a real replay (file gen9ou-2599174587.json gave
  25 turns with sensible actions).

**Gen2-locked, not used in the gen9 path (don't try to port):**
- `showdown/replay_to_training.py` — gen2 mechanics emulator. Skip entirely; we use poke-engine
  as the simulator instead.
- `showdown/chaos_stats.py` — loads `gen2ou_chaos.json` by default but the class accepts a path
  parameter; reusable with a gen9 chaos JSON.

**Missing, need to create or fetch:**
- `showdown/gen9ou_chaos.json` — Smogon usage statistics for gen9ou. **Fetch from**:
  `https://www.smogon.com/stats/<YYYY-MM>/chaos/gen9ou-1500.json`. Use the most recent month
  available (try `2026-04` first, then walk back month-by-month if 404). **Note**: the date in
  context says 2026-05-01 — Smogon stats publish monthly with ~1 month lag, so April 2026 is
  the expected freshest.
- `showdown/replay_to_training_gen9.py` — bulk processor (this plan's main deliverable).
- `showdown/bench_value_net.py` — α-sweep bench script.

---

## Implementation steps

### Step 1: Fetch gen9 chaos stats

```python
# Try each URL until one works:
import urllib.request, json
for ym in ["2026-04", "2026-03", "2026-02"]:
    url = f"https://www.smogon.com/stats/{ym}/chaos/gen9ou-1500.json"
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "Pokemon Bot Research"})
        with urllib.request.urlopen(req, timeout=15) as r:
            data = json.loads(r.read())
        with open("showdown/gen9ou_chaos.json", "w") as f:
            json.dump(data, f)
        print(f"saved {ym}")
        break
    except Exception as e:
        print(f"{ym}: {e}")
```

**Verify**: file exists, `data["info"]["metagame"]` is `"gen9ou"`, `len(data["data"])` is ~80-150
(number of distinct mons in usage stats).

### Step 2: Extend `chaos_stats.py` for gen9

Currently `ChaosStats.__init__` defaults to gen2 path. Update default-resolution logic so callers
can pass a format identifier:

```python
def __init__(self, path: str | Path | None = None, format: str = "gen2ou"):
    if path is None:
        path = Path(__file__).parent / f"{format}_chaos.json"
    ...
```

Don't break gen2 callers — leave existing constructor compatible. Test:
```python
cs = ChaosStats(format="gen9ou")
print(cs.pokemon["greattusk"].usage)  # should be a float in [0, 1]
print(list(cs.pokemon["greattusk"].moves.keys())[:5])  # top-5 moves by frequency
```

### Step 3: Implement `reconstruct_team`

Takes a `dict[str, PokemonReveal]` (from `replay_parse_gen9.parse_replay`) plus a `ChaosStats`
instance, returns a Showdown-format team import string suitable for `build_pe_state_gen9`.

**Signature**: `reconstruct_team(reveals: dict[str, PokemonReveal], chaos: ChaosStats) -> str`

**Logic**:
1. Dedup reveals by species. The parser currently has a known issue: team preview entries
   (species-keyed) and in-game switch entries (nickname-keyed) become separate dict entries for
   the same mon. Fix by post-processing: build species → reveal mapping, prefer in-game reveals
   (more info) when both exist, drop preview-only duplicates that match a nickname-keyed entry.
   Watch for forms: `Raichu-Alola` (preview) and `Raichu` (switch) refer to the same mon —
   normalize to base species + form suffix.
2. For each species in the deduped set:
   - **Moves**: take revealed moves, pad with top-frequency moves from chaos until ≥4. Use the
     chaos `moves` dict (top-K). If revealed has >4, drop excess (rare; happens on Choice locks
     where the engine sees 4 distinct moves used).
   - **Item**: if revealed → use it. Otherwise top-frequency item from chaos for this species.
   - **Ability**: if revealed → use it. Otherwise top-frequency ability.
   - **Tera type**: if revealed → use it. Otherwise default to "Stellar" or top tera-type from
     chaos if present (gen9 chaos JSON has `teraTypes` per mon).
   - **EVs**: chaos has `spreads` (top spreads with EV/nature). Use the top spread, or fall back
     to a balanced default `4 HP / 252 Atk / 252 Spe` for offensive mons,
     `252 HP / 252 Def / 4 SpD` for bulky.
3. Format as Showdown team import (one mon block per `\n\n`-separated section). Existing teams
   in `showdown/sample_teams_gen9.py` are the reference for format.

**Verify**: parse a known replay, reconstruct both teams, pass to `build_pe_state_gen9`, check
the resulting state has 6 mons per side and `pe.evaluate(state)` returns a finite number.

### Step 4: Implement engine-stepping replay processor

**Function**: `replay_to_trajectory(replay_json: dict, chaos: ChaosStats) -> list[tuple[str, float, int]]`
where each tuple is `(state_string, label, turns_remaining)`.

**Logic**:
1. `traj = parse_replay(replay_json)` — get parsed trajectory.
2. If `traj.aborted` or `traj.winner is None`, return `[]`.
3. Reconstruct both teams via Step 3.
4. Build initial state: `state = build_pe_state_gen9(team1_str, team2_str)`.
5. Convert winner to label: `label = 1.0 if winner == "p1" else (0.0 if winner == "p2" else 0.5)`.
6. Walk turns. For each turn `i`:
   - Extract `p1_action`, `p2_action` from `traj.turns[i]`. If either is None, **skip the game**
     (forfeit, partial parse, etc.) — but keep states recorded so far.
   - Translate actions to engine move strings. For type "move", use the move name. For type
     "switch", look up the species name → engine slot index. Engine wants `"switch <species>"`
     for switches.
   - `instructions = pe.generate_instructions(state, p1_str, p2_str)`. If this raises or returns
     empty, **skip the rest of this game**, return what we have so far.
   - Sample one instruction by `percentage`, `state = state.apply_instructions(chosen)`.
   - Record `state.to_string()` along with the label.
7. Compute `turns_remaining[i] = total_recorded - i - 1` for each recorded turn (used by trainer
   for gamma-discount of far-from-terminal labels).
8. Return list of `(state_str, label, turns_remaining)`.

**Edge cases to handle**:
- **Move name mismatch**: replay says "Hydro Steam" but engine has `hydrosteam`. Use the same
  name normalization the bench scripts do (`_strip_switch_prefix`, lowercase no-spaces).
- **Switch target ambiguity**: `{"type": "switch", "name": "Pelipper"}` is a species; engine
  needs `switch <species>` (or sometimes a slot index — check `bench_engines.py:85` for the
  exact format). The existing test in step 3 will surface format issues.
- **Tera move usage**: replay shows `|move|p1a: Kingambit|Kowtow Cleave|...` — and separately
  `|-terastallize|p1a: Kingambit|Fire`. The engine takes a separate `tera` action that gets
  bundled with the move. If the move name in poke-engine includes `-tera` suffix (like
  `kowtowcleave-tera`), use that form when terastallize fired this turn.
- **Forced switch after faint**: parser already filters these out via `force_switch` flags.
  If a turn has only one action (the surviving side's move), the other side's action is None
  and we skip.
- **Engine state divergence**: if a particular move resolves differently in the engine than the
  replay (rare but possible — e.g. damage rolls landing differently affecting whether a follow-up
  move could happen), we just trust the engine. The state we record is plausibly close to the
  real state, not identical. That's fine for value-net training.

**Verify**: process 10 replays, count successes (full trajectory recorded) vs. partial (stopped
mid-game). Aim for ≥70% full trajectories. If <50%, debug — likely a move-name mismatch issue.

### Step 5: Bulk-process replays into training pickle

**Script**: `showdown/replay_to_training_gen9.py`

```python
# Reads all replays in showdown/replays/gen9ou/, produces a pickle suitable for value_train.py
# Output format matches what gen_training_data.py produces (look at that for the schema).
import pickle, json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
from showdown.chaos_stats import ChaosStats
from showdown.replay_parse_gen9 import parse_replay
# ... (the function from step 4)

def main():
    chaos = ChaosStats(format="gen9ou")
    replay_dir = Path("showdown/replays/gen9ou")
    files = sorted(replay_dir.glob("*.json"))
    print(f"processing {len(files)} replays")
    all_states, all_labels, all_remaining = [], [], []
    success = errors = 0
    for f in files:
        try:
            data = json.load(open(f))
            traj = replay_to_trajectory(data, chaos)
            for s, l, r in traj:
                all_states.append(s)
                all_labels.append(l)
                all_remaining.append(r)
            if traj:
                success += 1
        except Exception as e:
            errors += 1
            if errors < 10:
                print(f"  {f.name}: {e}")
    print(f"success {success}/{len(files)}, errors {errors}")
    print(f"total turns: {len(all_states)}")
    with open("showdown/gen9ou_replay_data.pkl", "wb") as f:
        pickle.dump({"states": all_states, "labels": all_labels,
                     "turns_remaining": all_remaining}, f)
```

Match the data schema `value_train.py` expects — check `prepare_value_data` in that file
(`showdown/value_train.py:29-55`) for the exact format. Likely it expects a list-of-games
structure (each game has multiple turns); adjust accordingly.

**Verify**: `len(all_states)` should be ~30k-80k for ~2500 replays at ~25 turns avg with 50-70%
success rate.

### Step 6: Train ValueNet on the dataset

Use existing trainer:

```bash
.venv/bin/python showdown/value_train.py \
  --data showdown/gen9ou_replay_data.pkl \
  --model showdown/gen9_value_net.pt \
  --epochs 30 \
  --features-v2
```

**Verify**: training loss decreases monotonically, holdout BCE plateaus around 0.5-0.65 (pure
random would be 0.69 = ln(2); below that means the net is learning something).

Then export to ONNX. Look at how the existing policy net got exported (commit message mentions
"ONNX export ready for Rust"). Likely there's a `--onnx-out` flag or a separate export script.
If not, write one:

```python
import torch
from showdown.value_train import ValueNet
m = ValueNet(state_dim=579, hidden=256, n_layers=3)
m.load_state_dict(torch.load("showdown/gen9_value_net.pt"))
m.eval()
dummy = torch.randn(1, 579)
torch.onnx.export(m, dummy, "showdown/gen9_value_net.onnx",
                  input_names=["state"], output_names=["winprob"],
                  dynamic_axes={"state": {0: "batch"}})
```

### Step 7: Bench mcts_with_value at alpha sweep

**Script**: `showdown/bench_value_net.py` — variant of `bench_engines.py` where dev side uses
`mcts_with_value(state, value_net, duration_ms, alpha=ALPHA)` and ref side uses standard MCTS.

Run across the 4 canonical matchups (Mirror Sun 0v0, Sun-Stall 0v3, BO-Balance 1v2, Rain-TR 4v5)
at α ∈ {0.0, 0.3, 0.5, 0.7, 1.0}. α=0.0 should match v3 baseline (sanity check).

**Success criteria**: at some α, dev wins ≥55% across the 4-matchup average over multi-seed runs
(i.e. above the noise floor of ±3pp at typical n).

**Failure mode interpretation**:
- All α ≤ 50%: net is worse than heuristic. Check if data is corrupt (state strings mangled by
  our reconstruction), or net is undertrained, or featurization missing critical info (Tera state,
  Booster Energy timing, etc.).
- α=1.0 hugely worse than α=0.0 but α=0.3 better: net has signal but is being trusted too much.
  Lower α is the right answer for now.
- α=1.0 best: net is dominant; we're done with this phase.

---

## Known gotchas

1. **Tera state encoding** — verify `state.to_string()` includes terastallized status. Look at
   `parse_state_v2` to confirm the featurizer captures it.
2. **Booster Energy / Protosynthesis** — the +1-stat boost only fires under certain conditions
   (sun for Protosyn, Booster Energy held). The feature vector should reflect the active boost.
3. **Choice items locking** — if a mon used a move while holding Choice Scarf, the engine state
   has it choice-locked. Reconstructed teams that put Choice on a mon need to handle this; for
   simplicity, give Choice items only to mons revealed using a single move type, or use the
   `last_used_move` field correctly.
4. **Move-name normalization** — Showdown logs use display names ("Hydro Steam"), engine uses
   internal IDs (`hydrosteam`). Normalize consistently. Same for items ("Air Balloon" →
   `airballoon`).
5. **Forfeits and timeouts** — Showdown sometimes records a winner via forfeit before the game
   "naturally" ends. Game state at that point may be one-sided. Filter out games where one side
   has 0 fainted mons but loses (clear forfeit).
6. **Mons with multiple forms** (Ogerpon-Wellspring, Ogerpon-Hearthflame, etc.) — chaos stats
   have entries per form. Make sure form suffix is preserved in name normalization.
7. **The state-freeze bug** is still in the engine (existing memory note). If you hit a game
   where the engine state doesn't advance, just skip that turn forward.
8. **Random seed** — replays are deterministic outcomes; the engine's `apply_instructions`
   needs an instruction sampled by percentage. For training data, sample randomly per-turn — we
   want plausible trajectories, not byte-exact reproduction.

---

## What to do if things don't work

- **Parser handles <50% of replays**: most likely move-name normalization. Print a histogram of
  failed move names and add explicit mappings.
- **Reconstructed teams crash `build_pe_state_gen9`**: likely a tera-type or item-name mismatch.
  Print the failing team string + exception, narrow the offending field.
- **Training loss never goes below 0.69**: the labels might be inverted (label=1 should be P1 win,
  not P2). Or the featurizer is producing all-zeros / NaN.
- **α=0 doesn't match v3 baseline**: the value net is being mixed in even at α=0, or the
  ONNX export changed the model behavior. Verify with a quick `pe.evaluate(state)` comparison.

---

## Out of scope for this phase (Phase 2 candidates)

- Self-play data generation (AlphaZero loop). Only worth doing once the human-replay-trained
  net is competitive.
- Per-mon transformer architecture. Worth it after the MLP baseline is well-characterized.
- MCTS-distillation training targets (refined per-turn values from MCTS rollouts instead of
  outcome labels). Phase 2.
- Cron-scheduled scraping for longitudinal data accumulation. Manual scraping is fine for
  this phase.

---

## Files this plan touches

Created or significantly modified:
- `showdown/gen9ou_chaos.json` (new, downloaded)
- `showdown/chaos_stats.py` (extend for gen9 path)
- `showdown/replay_parse_gen9.py` (improve dedup, already exists as MVP)
- `showdown/replay_to_training_gen9.py` (new, the main deliverable)
- `showdown/gen9ou_replay_data.pkl` (new, training data output)
- `showdown/gen9_value_net.pt` (new, trained model)
- `showdown/gen9_value_net.onnx` (new, exported model)
- `showdown/bench_value_net.py` (new, α-sweep bench)

Reference / read-only:
- `showdown/value_train.py` — existing trainer
- `showdown/features_v2.py` — existing featurizer
- `showdown/local_battle.py` — `build_pe_state_gen9`
- `showdown/sample_teams_gen9.py` — team-import format reference
- `showdown/bench_engines.py` — bench harness skeleton to copy
- `poke-engine/poke-engine-py/src/lib.rs` (function `mcts_with_value`) — value-net binding
