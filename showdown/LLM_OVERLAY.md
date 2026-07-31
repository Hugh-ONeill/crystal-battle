# LLM + RAG overlay over the MCTS — design

**Status: DESIGN ONLY (2026-07-31). Nothing here is implemented; nothing consumes
roles.json yet.** This file exists so the design survives context the way
roles.json does. Companion artifacts:

- `showdown/roles.json` — the knowledge the overlay injects (34 top-usage
  gen9ou species, human-reviewed; consumer-facing `fact` separated from
  review-only `provenance`)
- `showdown/roles_report.py` / `ROLES.md` — review rendering
- `showdown/roles_draft.py` — the offline RAG drafting pipeline (tier 1 in §4)
- `showdown/roles_screen.py` — the offline disagreement instrument that
  located the mid-game entry points
- grounded-rag `/retrieve` on `:8001` — the retrieval service (smogon +
  chaos corpora, ~100% faithfulness on its eval set)

---

## 1. Why an overlay, and why not an LLM player

Three blind spots are measured, not suspected:

1. **The eval is blind in the opening.** Desk-read Brier ≥ 0.25 on turns 1–9
   across four ladder sessions — at or worse than always-guessing-50/50 — and
   the early-signal study confirmed it out-of-sample (eval at T9: 0.2479;
   plain observables at T3: 0.1716; the opponent prior *known at preview*:
   0.1765). Preview-matrix flatness and T1-switch churn are the same
   phenomenon. Whatever fixes the opening, it will not come from the leaf
   eval.
2. **The mid-game screen found large role disagreements.** Rules derived from
   roles.json disagree with our actual play on chipped full-HP-entry setup
   (45/55, 82%), weather moves with weather down (917/1350, 68%), and early
   cleaner deployment (210/317, 66%). Disagreement is not improvement — but
   it marks where role knowledge and search diverge.
3. **Static opponent priors mismodel killers.** The loss traces show midgame
   snowballs driven by sets and habits the chaos marginals blur out
   (SD-Low-Kick Kingambit at 7.1% marginal was invisible until the slash
   overlay).

And one failure mode is equally measured: **the LLM must not play the game.**
The ladder's LLM bots broadcast their chain of thought; the declarations are
95.9% accurate to what they then click, articulate, well-reasoned — and
gem31fl loses 96% of its games to us. Reasoning aloud is not converting.
Closer to home, this campaign's signature lesson (five instances, archive
included) is that **belief accuracy ≠ belief utility**: accurate knowledge
injected at the wrong joint makes play worse.

Shape discipline, from the campaign record: the levers that failed all
injected play-style directly into move selection (move-net −51pp as pilot,
opp-net neutral, archive-as-eval worse than vague). The levers that worked
were vetoes, capability flags, discrete modes, role economics, and prior
data. Every channel in this design is in the second class.

## 2. The interface: emit the decision, not the verdict

The engine's regular output is a verdict: "Ice Beam, 61% of visits." A
verdict is unarguable — all an LLM can do is agree or second-guess it, and
second-guessing a search with a language model is the losing pattern above.

Instead the search emits its **root decision matrix**. Per determinized world
`w` (a sampled opponent set assignment), the root already contains our
actions × their replies → (Q, N). With K=4 worlds and ≤9 legal actions per
side, that is four ≤9×9 grids — small enough to prompt, rich enough to expose
the *assumptions* the verdict rests on:

- which rows (our actions) are robust across worlds vs. good in one world only
- which cells the verdict hinges on ("Ice Beam is best only if this slot
  lacks Sucker Punch")
- the engine's implied best reply per world
- which cells are barely explored (low-N cells are sampling noise and are
  flagged as such, or the LLM will treat an artifact as truth)

**The column-only discipline.** The LLM is only ever allowed to talk about
*columns* — which opponent sets exist (world weights) and which replies a
real opponent actually clicks (reply distributions). Those are exactly the
two things roles.json encodes (set `prevalence`; `deployment` patterns like
"waits for fallen allies before the cleaner enters"). The engine then
**re-solves the emitted matrix** under the adjusted column beliefs and picks
the best row itself. The LLM never assigns a value to an action; the
arithmetic stays in the engine. It cannot be confidently wrong about a row
because it has no channel to a row. (The one narrow exception, row *flags*,
is citation-validated and bounded — §5.3.)

**Zero-Rust Phase 1.** Because the overlay re-solves an *emitted* matrix, it
is a pure post-processor: search runs exactly as today, dumps the root
structure, and the re-solve is a small numpy pass in the player. No engine
surgery until a channel has already won a paired A/B as a post-processor.

**Preview is the same interface one level up.** The 6×6 preview maximin is
measured-flat (row-min spreads 0.01–0.12 at any depth) because maximin
aggregation destroys the conditional structure — every lead's *worst case*
is similar while the value *after their lead is revealed* is not. So the
preview emission is the **full 6×6 lead matrix**, not the row-mins: the LLM
sees the conditional values plus both dossiers, the scouting-book lead
profile (richwoman's leads are concentrated: Ting-Lu 22%), and the
canonical-slot-1 prior. Its output is again column beliefs — a distribution
over *their* lead — and a mixed lead distribution falls out mechanically.

### Emission sketch (per consulted turn)

```json
{
  "turn": 14,
  "worlds": [
    {"id": 0, "style": "curated", "weight": 0.25,
     "opp_sets": {"kingambit": {"item": "Black Glasses",
                                "ability": "Supreme Overlord",
                                "moves": ["swordsdance", "kowtowcleave",
                                          "suckerpunch", "lowkick"]}},
     "rows": ["icebeam", "voltswitch", "switch:tinglu"],
     "cols": ["suckerpunch", "lowkick", "switch:gholdengo"],
     "q": [[0.61, 0.34, 0.55], [0.48, 0.51, 0.47], [0.44, 0.29, 0.5]],
     "n": [[412, 88, 130], [95, 60, 41], [23, 3, 18]]}
  ],
  "verdict": {"action": "icebeam", "visit_share": 0.61, "margin": 0.07},
  "near_tie_pool": ["icebeam", "voltswitch"],
  "screen_rules_fired": ["entry_condition:full_hp:ceruledge"],
  "set_posterior": {"kingambit": [["sd-lowkick", 0.6], ["sd-ironhead", 0.3]]}
}
```

## 3. What the LLM sees

**Per-battle dossier, assembled once at team preview.** Both rosters'
roles.json entries — `fact`, tags, deployment, sets with prevalence,
sequences; only consumer-facing fields, provenance never ships — plus our
own full sets (we know them; the LLM should never infer our side). The
dossier is a **stable prompt prefix**: Ollama caches it, `keep_alive` keeps
the model warm, so per-turn cost is only the appendix tokens. Twelve species
at ~150 tokens each is ~2k tokens — nothing for the 3090.

**Per-turn appendix (small, changes every turn).**
- Board state: HP%, status, hazards, screens, weather, fainted counts, tera
  spent.
- The revealed ledger: which opponent moves / items / abilities have actually
  been seen this game.
- set_inference's current posterior per opponent mon — the engine's belief,
  labeled as belief.
- The matrix of §2, low-N cells flagged.
- Which role-screen rules fired this turn.

**Model.** gemma4:26b-a4b on the 3090, `think:false` (the 10× rule),
structured-output JSON. MCTS is pure CPU, so the GPU lane is free compute;
expected 1–3 s per call for a small schema.

**Prompt hygiene.** Opponent nicknames and battle chat are
attacker-controlled strings. Only canonicalized species/move/item identifiers
from our own translator ever enter the prompt — never raw opponent text.

## 4. When the RAG is consulted: three tiers, never in-turn

1. **Offline (bulk) — already running.** `roles_draft.py` → species-filtered
   multi-angle retrieval → human review → roles.json. This *is* the RAG
   consultation, moved to where mistakes are catchable: the wrong-species
   retrieval failure (a Great Tusk query returning a Deoxys-Speed passage) is
   a review-time annoyance offline and a poisoned prompt live.
2. **Match start.** At preview we know all 12 species and have seconds of
   headroom. Any opponent species missing from roles.json gets a one-shot
   species-filtered retrieve, auto-drafted into a fact marked *unverified*
   and weighted down. Cached per species, so it amortizes to nothing across
   a session.
3. **In-turn: never.** Latency, wrong-species risk, and nothing that happens
   mid-battle changes what Smogon says about a Pokemon. Chaos-stat lookups
   are local JSON reads — context assembly, not retrieval.

## 5. Output channels, ranked by restriction

Output is strict schema-validated JSON. Invalid JSON, an unresolvable
citation, or a missed deadline → **identity function**; the engine's original
pick stands. Every intervention is logged (turn, deltas, flipped-or-not,
cited rules) for post-hoc audit.

### 5.1 World reweighting (primary)

A posterior over the K emitted worlds, mixed with the engine's own weights
under a capped λ — the same shape as `--chaos-alpha`, so strength is one
A/B-able dial. Catches "you are searching a Choice Band world for a mon
whose revealed Low Kick makes it 80% the SD set." This composes with the
ranked-hypothesis world ladder sketched in TODO (world 1 = second-best
distinct curated candidate): the ladder diversifies the columns, the LLM
weights them.

### 5.2 Reply distribution (secondary)

Per world, a distribution over their columns, replacing implied-best-response
with predicted-actual-response at the root re-solve. Composes with the
move-net (55% top-1, already earmarked for opponent prediction only): the
net proposes, the LLM adjusts with role knowledge. The plumbing partially
exists — `mcts_with_priors` + `_aligned_opp_priors` were built for exactly
this shape — and the leaked-intent exploit (LLM bots declare their move in
chat, 95.9% accurate) is this channel's limiting case: a reply distribution
with near-certainty on one column. The loss-trace finding that static priors
mismodel our killers is exactly the gap this channel fills.

### 5.3 Row flags (tertiary, most restricted)

The one row-side channel, for future value the leaf eval structurally cannot
see: "entering Ceruledge chipped cancels the Sash plan" is true and invisible
to a within-horizon search. A flag must cite a roles.json rule id
(`species.field[.index]`); the citation is resolved mechanically against the
file and **a flag citing a rule that does not resolve is discarded
outright** — the hallucination guard is by construction, not by prompting.
Effect is a bounded Q-penalty on the flagged row, never a veto.

### Re-solve mechanics

Row value = Σ_w mix_λ(engine_w, llm_w) · Σ_c reply(w,c) · Q̃(w,r,c), where
Q̃ shrinks each cell toward its row's world-level mean in proportion to cell
N (low-N cells contribute little). Row flags subtract a capped δ. Argmax
picks the action; if it differs from the engine's verdict, that is a **flip**
and is logged as such. λ and δ start small (chaos-alpha experience says
≤0.25) and are the only tuning knobs.

## 6. Gating and latency

Not every turn — only where blindness or disagreement is measured:

- team preview and turns 1–3 (the Brier-blind window)
- turns where a role-screen rule fires (the 82/68/66% spots)
- near-tie roots (margin below ε)
- high cross-world row variance — the matrix's own "my answer depends on
  which set they have" signal

Estimated 15–20% of turns. The call runs **in parallel with the search's
clock budget**, fired when search starts; if the LLM misses the search
deadline the turn commits without it. We currently commit ~9.7 s before the
LLM opponents on average, so a 2–3 s parallel call costs no clock we
actually use. The overlay must never block the clock — we have already paid
for self-inflicted timeouts once.

## 7. Evaluation plan

The bar: five instances of belief-accuracy ≠ belief-utility. Nothing ships
on plausibility.

1. **Margin measurement first** (already in TODO — the missing half of the
   mid-game screen). Re-search a position pool with `--dump-states` and join
   ranked visit shares against the fired rules: do disagreements land on
   near-ties (cheap tie-break, low ceiling, low risk) or on confident
   positions (the re-solve matters; overrides have never won here)? The T1
   instrument already shows the answer varies by decision type — T1 switches
   average a 49pp margin with zero near-ties, stays are near-tied 34% of the
   time.
2. **Shadow mode.** Run the full overlay live but *apply nothing*: emit,
   call, re-solve, log the hypothetical flip. Free of game risk, and it
   yields the flip rate, per-channel deltas, and an auditable corpus of
   would-be interventions to blunder-audit before any of them is real.
3. **Per-channel paired A/B, fixed n, one channel at a time.** The existing
   rig (`par_series.sh --ab`, sign test on discordant pairs) is the arbiter;
   the ~15pp session noise floor is why whole-game winrate alone is blunt.
   With only ~15–20% of turns touched, power comes from conditioning on
   intervention games and auditing flips individually, not from the topline
   CI alone.
4. **Preview channel** is only testable with the randomized preview A/B
   (arm-alternation via AB_FLAG already supports it) — the observational
   lead screen came back null-and-underpowered and cannot say more.

## 8. Sequencing

| phase | what | cost | gate |
|---|---|---|---|
| 0 | margin measurement over a `--dump-states` pool | offline box time | decides tie-break vs re-solve framing |
| 1 | matrix emission + dossier assembly + shadow mode | logging only, no Rust | flip-rate + audit of would-be flips |
| 2 | preview channel live (lead distribution mix) | one LLM call/game | randomized preview A/B, fixed n |
| 3 | world-reweight channel live, gated turns | ~15–20% of turns | paired A/B vs shadow arm |
| 4 | reply-distribution channel | same | paired A/B; audit vs move-net-only |
| 5 | row flags | same | paired A/B; every flag citation audited |

Phases 3–5 ship one at a time; a combined arm only after singles have
individually survived.

## 9. Anti-goals

- The LLM never picks a move, never sees a channel to row values.
- No free text is consumed — free text goes to logs only.
- No in-turn retrieval, ever.
- The overlay never blocks the clock; identity on any failure.
- No fine-tuning until prompted Gemma demonstrably fails at schema
  discipline, faithfulness, or latency (the strongest real argument is
  latency, and only for per-turn calls). Facts always flow through
  retrieval, never into weights — meta goes stale monthly and baked
  knowledge fails confidently.

## 10. Open questions

- Shrinkage constant for low-N cells, and N_min below which a cell is
  ignored entirely.
- Initial λ / δ values (start ≤0.25 by chaos-alpha precedent) and whether
  they should differ per gate reason.
- Whether the reply distribution should eventually feed the search in-tree
  (a Rust change) after the post-hoc re-solve validates — deeper effect,
  much harder to audit.
- Dossier treatment of unverified auto-drafted facts (tier 2): weight-down
  factor, and whether they should be excluded from row-flag citations
  (leaning yes — flags should cite reviewed rules only).
- Team-level knowledge (sole-remover, entry-enabler gaps) is still not
  expressible per-species; the dossier assembler is the natural place to
  compute it per battle.
