# teams_v6.txt bench results — 2026-05-23

Full 18-team round-robin bench results vs the original v5 baseline.

## Bench config

- Teams file: `~/teams_v6.txt`
- Script: `~/Developer/grimoire/crystal-battle/showdown/bench_monotype.py`
- Games per direction: 4 (so 8 games per matchup with counterbalancing)
- Search: 200ms MCTS per turn (via poke-engine)
- Workers: 22 (out of 24 cores)
- Seed: 42
- Total games: ~1224 (153 pairs × 8)
- Tera-suppressed (monotype rule)

## v6 standings vs v5 baseline

| Rank | Team | v6 Win% | v5 Win% | Δ Win% | v5 Rank | Δ Rank |
|---|---|---|---|---|---|---|
| 1 | Supa Fitingu (Fighting) | 69.7% | 39.7% | +30.0 | 17 | +16 |
| 2 | RuPauls Dragon Race | 68.8% | 48.1% | +20.7 | 9 | +7 |
| 3 | Disco Inferno (Fire) | 68.2% | 46.0% | +22.2 | 11 | +8 |
| 4 | Heavy Metal (Steel) | 67.3% | 76.8% | -9.5 | 1 | -3 |
| 5 | Spooky Team (Ghost) | 65.0% | 65.7% | -0.7 | 4 | -1 |
| 6 | I Believe In Fairies | 59.0% | 47.2% | +11.8 | 10 | +4 |
| 7 | Rock Hard | 55.5% | 57.1% | -1.6 | 5 | -2 |
| 8 | Toxic Team (Poison) | 55.4% | 67.3% | -11.9 | 3 | -5 |
| 9 | Dork Team (Dark) | 54.2% | 43.8% | +10.4 | 14 | +5 |
| 10 | Grounded³ (Ground) | 52.3% | 52.9% | -0.6 | 7 | -3 |
| 11 | Stop Bugging Me (Bug) | 47.5% | 43.2% | +4.3 | 15 | +4 |
| 12 | Fly Away Now (Flying) | 44.2% | 48.5% | -4.3 | 8 | -4 |
| 13 | Brain Blast (Psychic) | 41.0% | 55.4% | -14.4 | 6 | -7 |
| 14 | Soft and Wet (Water) | 38.9% | 67.3% | -28.4 | 2 | -12 |
| 15 | Your Ass is Grass (Grass) | 31.5% | 45.6% | -14.1 | 12 | -3 |
| 16 | Totally Normal Team | 27.0% | 15.6% | +11.4 | 18 | +2 |
| 17 | Electric Boogaloo | 24.0% | 37.7% | -13.7 | 17 | 0 |
| 18 | Ice Ice Baby | 22.5% | 45.5% | -23.0 | 13 | -5 |

## Variance / balance

- v5 spread: 76.8% top → 15.6% bottom (61.2 pts)
- v6 spread: 69.7% top → 22.5% bottom (47.2 pts)
- v6 is a more balanced meta — worst teams improved dramatically, top teams compressed

## Key insights from per-team A/B benches

Each type also had a focused 8-team A/B bench (v6 vs v5 + 6 opponents) before this full bench. Results:

| Type | A/B Δ (v6 vs v5) | Pattern |
|---|---|---|
| Dragon | +38% | Choice-lock / wall meta refit; bench-friendly |
| Normal | +37% (TR build) | TR + Porygon2/Ditto pilots fine in MCTS |
| Fire | +23% (early), then sun rebuild | Specs Volcanion was a trap (19% wr at 1500+) |
| Steel | +25% | Iron Treads + Goodra-Hisui added Fairy/Ground resists |
| Ghost | +4% | Mimikyu + Ceruledge ≈ Spectrier + Sinistcha-M |
| Grass | +18% | Aurora Veil HO rebuild |
| Bug | +18% | Kleavor SR + Custap Araquanid + Lokix Tinted Lens |
| Rock | +6% | Set refits only (mons already meta) |
| Fairy | -3% (mirror v6 wins 75%) | Iron Val + Klefki added but lost Enamorus Earth Power |
| Fighting | tied (mirror 37.5% for v6) | Gallade + Sneasler + Urshifu redistributes matchups |
| Poison | -7% (within CI) | Haunter Eviolite + Overqwil rebuild |
| Ice | -3% (within CI) | Fixed Fire matchup (+32) but lost Fighting (-44) |
| Dark | -12% | Greninja + Darkrai meta (per replay-stats 71% wr) but MCTS pilots setup poorly |
| Electric | -26% | Pincurchin/Iron Hands/Raichu-A terrain engine; MCTS can't pilot |
| Water | -58% (worst gap) | Bulky stall (CurseDozo etc.); MCTS hates it but it's the 77% wr human meta |

## Methodology notes / patterns observed

1. **MCTS at 200ms pilots well**: choice-locked attackers (CB/Specs/Scarf), walls with immediate damage (Body Press, Stamina), priority moves (Sucker Punch, Bullet Punch).

2. **MCTS at 200ms pilots poorly**: setup sweepers requiring boost turn (NP Darkrai, CM Cresselia, Curse Dondozo), terrain-dependent abilities (Iron Hands Quark Drive, Raichu-A Surge Surfer), full bulky stall archetypes (Water v6).

3. **Replay-stats (1500+ ELO) and bench disagree most** when meta uses setup-heavy or terrain-engine teams. Replay-stats reflects human pilot skill; bench reflects MCTS-200ms skill ceiling.

4. **Pattern is reliable**: knowing the meta archetype before benching lets you predict whether bench will agree with replay-stats. Choice-lock heavy = bench will agree (Fighting/Dragon/Fire/Normal). Setup-heavy = bench will diverge negative (Dark/Electric/Water).

## File locations

- Team rosters: `~/teams_v6.txt` (18 teams)
- v5 baseline: `~/teams_v5.txt`
- Bench script: `~/Developer/grimoire/crystal-battle/showdown/bench_monotype.py`
- Replay-stats: `~/Developer/grimoire/crystal-battle/showdown/monotype_replay_stats.py`
- Species types: `~/Developer/grimoire/crystal-battle/showdown/species_types.json`
- Raw replay corpus: `~/Developer/grimoire/crystal-battle/showdown/replays/gen9monotype/` (3399 replays)
- Raw bench output: `/tmp/bench_v6_full_results.txt` (volatile)
