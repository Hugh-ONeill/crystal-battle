---
name: bench-teams
description: Run the full monotype team-bench workflow — validate roster, round-robin bench, compare vs Smogon baseline. Use when the user wants to bench a teams file (e.g. "/bench-teams teams_engine.txt" or "bench v7"). Args: a teams file name/version, optionally games-per-direction and search-ms.
---

# Monotype team bench workflow

Run everything from the repo root (`~/Developer/grimoire/crystal-battle`) using `.venv/bin/python`. The bench script is `showdown/bench_monotype.py` (NOT in `monotype/`).

Resolve the args: a bare version like `v6` means `monotype/teams/teams_v6.txt`; `engine` means `monotype/teams/teams_engine.txt`. Default to `teams_engine.txt` if no file given. Default `--games 16 --search-ms 500` unless the user asked for a smoke run (then 2-4 games, 200ms, and prefix the command with `CLAUDE_SMOKE=1`).

## 1. Validate the roster

```
node ~/team-tools/validate_teams.js monotype/teams/<file> gen9monotype
```

Exit 0 = all clean. Exit 1 = at least one team has issues; show the problem lines and stop until fixed. Common trap: extra monotype bans and illegal move combos beyond standard Smogon (Bloodmoon Ursaluna, Annihilape, Bisharp+Knock Off, Blissey+Toxic, Porygon2+Stealth Rock, etc.).

## 2. Run the bench

Critical flags and gotchas:

- ALWAYS pass `--teams-file` explicitly; the script's default is the stale `teams_v3.txt`.
- ALWAYS pass `--lead-net monotype/lead_net.pt`. Without it the bench hard-leads each team's slot-1 paste mon, which confounds A/B comparisons. The net is near-instant; `--use-lead-picker` (MCTS variant) is slow, don't use it.
- No output flag exists: redirect stdout to `monotype/bench/<descriptive-name>.txt` (convention: name + games/pair + seed, e.g. `teams_engine_16pp_s42.txt`).
- Fix `--seed 42` for reproducibility; `--workers 22` on the 24-core desktop.
- Long runs (16 games/direction on 18 teams = 2448 games) must be wrapped in `systemd-inhibit --mode=block systemd-run --user --scope -p MemoryMax=20G --same-dir bash -c "..."` — a PreToolUse hook enforces this. Run it as a background task and report when it finishes.
- Do NOT add `--use-endgame-solver` by default; its A/B is still pending and it has an OOM history (node-budget-bounded now, but MemoryMax stays mandatory).
- Tera is always suppressed — that is correct for monotype, never "fix" it.

Canonical invocation:

```
systemd-inhibit --mode=block systemd-run --user --scope -p MemoryMax=20G --same-dir bash -c \
  ".venv/bin/python showdown/bench_monotype.py \
     --teams-file monotype/teams/<file> \
     --games 16 --search-ms 500 --workers 22 --seed 42 \
     --lead-net monotype/lead_net.pt \
     > monotype/bench/<name>.txt 2>&1"
```

## 3. Compare vs Smogon baseline

Raw winrate is NOT team strength; only the delta vs the Smogon type-matchup expectation is meaningful. Use the newest month under `monotype/smogon_stats/` (check `ls`, don't assume) and the 1500 bucket (the target ELO for this project; 1760+ is a thin volatile slice).

```
.venv/bin/python monotype/compare_bench_vs_smogon.py \
  --bench-raw monotype/bench/<name>.txt \
  --teams-file monotype/teams/<file> \
  --matchup-chart monotype/smogon_stats/<latest-YYYY-MM>/matchup/gen9monotype-matchup_chart-1500.txt \
  --verbose
```

## 4. Interpretation notes for the report

- Rank teams by delta-vs-expected, not raw win%.
- Discount any team carrying Zoroark: Illusion is still a silent no-op in poke-engine (asymmetric info, structurally deferred).
- Single-run month-to-month Smogon shifts are noise; only flag multi-month monotonic trends.
- `setup_and_die` leads are role-correct for screens/hazard setters, not blunders.
