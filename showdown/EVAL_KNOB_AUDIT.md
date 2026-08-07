# Eval knob audit — which CB_EVAL_* toggles are actually live

Static audit, 2026-08-07, of `poke-engine/src/genx/evaluate.rs`. Motivation:
the 2026-07-22 eval bisect tested a list of terms at n=48 per arm, unpaired,
on one suite, and its own log lines say "noise-level, do not interpret". Before
spending box time retesting them with the paired rig, establish which knobs
change a computation at all. Two of the historical arms did not test what their
names suggest.

## DEAD — changes nothing, do not spend games on it

| knob | why |
|---|---|
| `hazards` | Switches `STEALTH_ROCK`→`STEALTH_ROCK_BASE` and `SPIKES`→`SPIKES_BASE`, but the pairs are **identical** (-10.0/-10.0 and -7.0/-7.0) since the 2026-07-23 revert of the monotype-era -15/-9 tuning. The code comment states it outright: "This makes the CB_EVAL_OFF=hazards knob inert (main == base) until someone retunes again." Any behavioural wiggle measured on this arm today is pure noise — which makes it a useful built-in NEGATIVE CONTROL for the screening rig. |

## INVERTED — `CB_EVAL_OFF` is meaningless, use `CB_EVAL_ON`

| knob | why |
|---|---|
| `threatv2` | `threatv2: !has_on("threatv2")` — default-OFF experimental term. Enable with `CB_EVAL_ON=threatv2`. Parked pending a recalibration that co-tunes the boost/speedtier couplings. |
| `weatherteam` | `weatherteam: !has_on("weatherteam")` — same pattern, enable with `CB_EVAL_ON=weatherteam`. |

## LIVE — 15 knobs that genuinely gate a computation

Split by KIND, because the two kinds deserve different treatment.

### Truth claims (statements about the game that are simply correct)
Default-ON on the stated principle "facts always-on, context weights
mode-gated". Retesting these asks "is our model right", not "is this weight
tuned".

| knob | what it gates |
|---|---|
| `unaware` | Zeroes atk/def/spa/spd boost credit into an Unaware active, mirroring `damage_calc`. Off = credit boosts the mechanic makes worthless. |
| `locks` | `CHOICE_LOCKED_STATUS` (-35) + `choice_on_wall` (-20). Un-parked 2026-07-24 after a 229-pair interleaved retest refuted the harm hypothesis (the accept-h0 that parked it was the Jul-23 level step "wearing a verdict costume"). |
| `synergy` | Guts/Flame Orb-family pairings, Black Sludge on a non-Poison, Rest-Talk rebate, Regenerator pending. |
| `supremeoverlord` | Fallen-ally threat scaling (1.0 + 0.1 × fainted allies). |
| `poisonheal` | `POISON_HEAL_STATUSED` 35.0 → 15.0 and drops `POISON_HEAL_PENDING`. |

### Context weights (tunable emphases with no truth value)
These are where "what was it supposed to do" is a design opinion, and where a
retest can legitimately move a default.

| knob | what it gates | magnitude | fires |
|---|---|---|---|
| `volatiles` | Ours scores Perish 1-4, Encore, Taunt, Disable, Torment, Heal Block, Octolock, Yawn, Salt Cure; OFF reverts to upstream's **three** (Leech Seed, Substitute, Confusion). The single biggest behavioural surface in the list. | SUBSTITUTE +40, LEECH_SEED -30, CONFUSION -20, others | whenever a volatile is up |
| `hopeless` | `HOPELESS_MATCHUP` when an active can threaten neither physically nor specially nor with status. | -50 (largest single constant) | narrow condition |
| `speedtier` | Speed-tier bonus for outspeeding, Trick-Room-reversed. | +20 | EVERY turn both actives live — highest frequency |
| `threat` | Forces the threat multipliers to 1.0 instead of `threat_vs`. NOTE: threat off implies hopeless unreachable (the condition tests threat == 0). | scales boosts | every turn |
| `tera` | `evaluate_tera_active` + USED_TERA. | TERA_STAB_AVAILABLE, TERA_WEAK_PENALTY | when tera relevant |
| `items` | Per-item pricing vs upstream's flat +10 for holding anything. | varies | every mon, every eval |
| `pp` | Recovery-PP depletion tax (doubled in stall mode). | RECOVERY_PP_VALUE | recovery movesets |
| `pending` | Wish / Future Sight scoring. | — | when pending |
| `weather` / `terrain` | Per-active weather and terrain scoring. | WEATHER_TYPE_BOOSTED 8, TERRAIN_TYPE_BOOSTED 5 | when field is up |
| (no knob) | **boost credit** — has no `eval_off` entry; added `CB_BOOST_SCALE` 2026-08-06 to scale it. | +30 atk/spe, +15 def/spd | when stages exist |

## Consequences for the retest program

1. Drop `hazards` from the list (dead), or keep it deliberately as the
   negative control — an arm that MUST show no behavioural delta. If a screen
   reports one, the screen is miscalibrated.
2. `threatv2` and `weatherteam` need `CB_EVAL_ON`, not `CB_EVAL_OFF`.
3. Prioritise by frequency × magnitude, not by name: `speedtier` fires every
   turn at +20, `volatiles` has the widest surface (13 statuses vs upstream's
   3), `hopeless` is the largest constant but fires rarely. Those three are the
   highest-value context-weight retests.
4. Truth claims are a different question from context weights and should not
   share a decision rule: a truth claim that costs winrate is a finding about
   the SEARCH (it is exploiting a fiction), not a licence to delete the fact.
5. `CB_EVAL_BASELINE=1` sets every knob at once AND disables stall mode, so it
   is a bundle, never an attribution. The 2026-07-22 bisect's `base` arm
   therefore cannot attribute anything to any single term.
