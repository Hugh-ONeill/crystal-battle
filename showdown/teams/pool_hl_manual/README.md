# pool_hl manual additions

Teams copied into `pool_hl` after `curate_team_pool.py` regenerates it.

`pool_hl` is derived-and-gitignored: every run of the curator clears it and
rewrites it from the metamon high-ladder slice, so anything hand-added there
survives exactly until the next regen. This directory is the durable home.
Filenames are preserved verbatim on copy; the curator copies every `*.txt`
here into `pool_hl` byte-for-byte, so a filename that appears three times here
lands three times in the rotation.

Companion file: `../pool_hl_drops.json`, the removals side of the same state.

## anti-hazard subpool (`ah*`) — the richwoman experiment (2026-07-29)

Four hand-built anti-SR-chip Boots-balance teams, aimed at the loss mechanism
pinned on 2026-07-28: our 42-turn bleeds vs richwoman (an fp-class opponent
that is ~half of all our ladder games) are ~30% hazard chip, Stealth Rock
alone 21% of it — death by SR on ~7 switch-ins/game, a clean attrition loss
with no blunder-collapse. Balance is also our best archetype vs that matchup
(39%, against sand 36% / HO 29% / stall 25%). The team is chosen *before* we
know the opponent, so we cannot bring these only vs richwoman — instead we
**weight the whole rotation** toward them and read the aggregate: since
richwoman is ~half our games, a 25%-of-pool anti-hazard slice puts anti-hazard
tools into ~25% of the richwoman games too. Boots-balance are strong
generalists, so this should not cost us vs the LLM bots we already beat 75–93%.

Each of the four distinct teams is present **3×** (`_v2`/`_v3` copies) so the
subpool is ~25% of the 47-team live pool (12 anti-hazard of 47). The `ah`
prefix is the aggregation key: `grep '^ah'` on the tally's team column pools
every anti-hazard game regardless of which of the four played; the `ah1..ah4`
stems split it per distinct team (so one bad team can't silently sink the
aggregate read).

- `ah1_greattusk_bootsbal` — Great Tusk / Zapdos / Slowking-Galar / Corviknight
  / Kingambit / Dragapult. Rapid Spin + Defog removal, Regen pivot, two
  breakers. (The original 2026-07-28 boots-balance team, renamed `40_`→`ah1_`.)
- `ah2_irontreads_tornt` — Iron Treads / Tornadus-T / Slowking / Kyurem /
  Dragonite / Iron Valiant. Rapid Spin + two Regenerator pivots (Torn-T,
  Slowking) + three breakers; offense-leaning.
- `ah3_cinderace_courtchange` — Cinderace / Corviknight / Toxapex / Landorus-T
  / Raging Bolt / Gholdengo. Cinderace **Court Change** flips their hazards
  back onto them (a direct answer to hazard-stack) + Corviknight Defog;
  Toxapex Haze anti-setup + Regen.
- `ah4_mandibuzz_magicguard` — Great Tusk / Mandibuzz / Amoonguss / Clefable /
  Iron Crown / Kingambit. Fat-balance: Rapid Spin + Defog, Clefable Magic Guard
  (hazard-immune wincon), Amoonguss Regen + Clear Smog anti-setup.

All four validate clean against `gen9ou` (the proxy validator; the server is
authoritative for the `gen9oulongtimer` custom bans — none use anything
revival/exotic, so low risk). Watch the next sessions: `ah`-vs-richwoman rate
against the ~25% non-`ah` baseline. If it doesn't beat ~25%, richwoman is a
true strength wall and the reweight comes back out (drop the `ah*_v2/_v3`
copies to de-weight, or the whole subpool via the drops list).
