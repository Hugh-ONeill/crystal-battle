# pool_hl manual additions

Teams copied into `pool_hl` after `curate_team_pool.py` regenerates it.

`pool_hl` is derived-and-gitignored: every run of the curator clears it and
rewrites it from the metamon high-ladder slice, so anything hand-added there
survives exactly until the next regen. This directory is the durable home.
Filenames are preserved verbatim on copy; keep the `NN_` prefix above the
curated range (curated entries number from `00`) so they sort last and never
collide.

Companion file: `../pool_hl_drops.json`, the removals side of the same state.

- `40_greattusk_bootsbal.txt` — hand-built anti-SR-chip Boots balance. Five of
  six on Heavy-Duty Boots, aimed squarely at the loss mechanism pinned on
  2026-07-28: 42-turn bleeds vs richwoman are ~30% hazard chip and Stealth Rock
  alone is 21% of it. Balance is also our best archetype vs that matchup (39%,
  against sand 36% / HO 29% / stall 25%), so this is the first team built to
  attack the identified mechanism rather than hoping search converts it.
