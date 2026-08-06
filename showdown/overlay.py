#!/usr/bin/env python3
"""LLM overlay over the MCTS — Phase 1: SHADOW MODE. Emit, consult, log.
APPLIES NOTHING: the engine's move is already committed before the consult
thread even starts. Design: showdown/LLM_OVERLAY.md.

What shadow mode measures (the doc's sequencing step 2): the FLIP RATE — on
consulted turns, would the LLM's world reweighting, pushed through the real
merge (`_merge_mcts_results(weights=...)`, the same function the live
CB_WORLD_WEIGHTS dial uses), have changed the chosen move? Every consult is
logged with the full context needed to audit its flips afterwards, which is
what makes the later paired A/B affordable.

Column-only discipline (the doc's core rule): the LLM's output is a weight
vector over the ALREADY-SEARCHED worlds plus optional row flags that must
cite a roles.json rule (dropped mechanically when the citation does not
resolve). It never names a move to play; there is no channel for one.

Margin-measurement consequence (2026-07-31): roles disagreements sit on
20-45pp CONFIDENT margins, so row flags are recorded as logging only — at any
bounded penalty they cannot flip those decisions, and the campaign is 0-for-N
overriding them. If the shadow flip audit shows the world channel converging
on the same spots, that is the evidence the doc requires before anything goes
live.

Latency: the consult runs on a daemon thread AFTER the move is returned, so
it can never cost clock. think:false per the recorded 10x rule; keep_alive
holds Gemma warm; the per-battle dossier is a byte-stable prompt prefix so
ollama's prefix cache absorbs it after the first consult of a battle.
"""

from __future__ import annotations

import json
import os
import re
import threading
import time
import urllib.request
from pathlib import Path

HERE = Path(__file__).parent

OLLAMA = os.environ.get("CB_OVERLAY_OLLAMA", "http://127.0.0.1:11434/api/chat")
MODEL = os.environ.get("CB_OVERLAY_MODEL", "gemma4:26b-a4b-it-q4_K_M")
LOG = os.environ.get("CB_OVERLAY_LOG",
                     str(HERE / "overlay_shadow.jsonl"))
TIMEOUT_S = float(os.environ.get("CB_OVERLAY_TIMEOUT", "25"))
LAMBDAS = (0.25, 0.5, 1.0)      # blend strengths re-solved per consult

# Advocate world (user idea 2026-07-31): world reweighting redistributes
# EXISTING votes, and the margin measurement showed the starved actions have
# none to redistribute — the demanded setter switch drew ~zero visits in
# 184/455 weather disagreements. So when a roles rule nominates an action the
# merged search starved (<5% share), shadow runs ONE extra search over the
# world-0 state with our-side root priors concentrated on that action, deep
# enough to actually price its subtree. It does not pick the move — it
# subpoenas the search: the deep Q is logged next to the engine choice's Q
# and the verdict stays with the data. Runs post-commit on the consult
# thread, usually inside the opponent's thinking window.
ADVOCATE_MS = int(os.environ.get("CB_ADVOCATE_MS", "600"))
ADVOCATE_PRIOR = 0.75
STARVED_SHARE = 0.05

# LIVE mode (2026-08-01): apply the reweighted merge to the actual move, but
# ONLY when the LLM's world weights are EXTREME — the flip audit's evidence
# lives in the near-certain vectors ([0.999, 0.001]-shaped: "one of your
# worlds is contradicted by the reveals"), and the near-uniform flips priced
# as coin-flips at oracle depth. Fire rate at 0.8 is ~0.11 applied flips per
# game (measured, session 20260731_220522), so a 480-game A/B is a
# NON-REGRESSION gate + live validation, not a detection instrument — the
# conversion evidence keeps coming from flip_audit over accruing shadow data.
# Failure is always the identity: timeout, invalid JSON, or ollama down mean
# the engine's own choice plays, logged as such.
APPLY_MIN = float(os.environ.get("CB_OVERLAY_APPLY_MIN", "0.8"))
LIVE_TIMEOUT_S = float(os.environ.get("CB_OVERLAY_LIVE_TIMEOUT", "10"))

# consumer-facing roles fields only — provenance/review never enter a prompt
# ability/ability_split were MISSING here until 2026-08-02, so the dossier
# never showed an ability at all — while 89 of 114 entries carry `ability`
# and 23 carry a genuine `ability_split`. The shadow corpus caught it: the
# LLM cited `okidogi.ability` 18 times (its single most-cited path) for a
# species whose Toxic Chain / Guard Dog split decides the matchup, and got
# back "(no entry)". Citation audits are a channel for the model to report
# what the dossier is missing — read them that way.
_ENTRY_FIELDS = ("fact", "tags", "axis", "preserve", "deployment",
                 "lead_intent", "entry_condition", "value_curve", "resource",
                 "requires", "ability", "ability_split", "engine_blind",
                 "single_build_note")

_WEATHER = {"raindance": "rain", "primordialsea": "rain",
            "sunnyday": "sun", "desolateland": "sun",
            "sandstorm": "sand", "snow": "snow", "snowscape": "snow",
            "hail": "snow"}

SCHEMA = {
    "type": "object",
    "properties": {
        "world_weights": {
            "type": "object",
            "additionalProperties": {"type": "number"},
            "description": "world index -> relative weight, higher = search "
                           "this set assumption harder",
        },
        "worry": {"type": "string",
                  "description": "one sentence: the biggest thing the engine's "
                                 "assumptions might be missing (log only)"},
        "flags": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "row": {"type": "string"},
                    "rule": {"type": "string",
                             "description": "roles citation like "
                                            "'ceruledge.sequence' or "
                                            "'kingambit.value_curve'"},
                    "note": {"type": "string"},
                },
                "required": ["row", "rule"],
            },
        },
        "confidence": {"type": "number"},
        # Reply channel (§5.2), shadow build 2026-08-06 — the one overlay
        # channel still open after the phase-1 verdict killed world
        # reweighting. Still a COLUMN statement: predicting THEM, never
        # choosing for us. Scored post-hoc by reply_audit.py against the
        # opponent's actual click from the ladder log; the gate is beating
        # the engine's own implied-reply baseline, and only a passed gate
        # earns a re-solve design.
        "reply": {
            "type": "object",
            "additionalProperties": {"type": "number"},
            "description": "1-3 entries: action string from THEIR OPTIONS "
                           "(exact) -> probability the opponent actually "
                           "clicks it this turn",
        },
    },
    # worry became required 2026-08-01: schema-constrained decoding takes the
    # shortest valid path, so the optional field was emitted in 0 of 504
    # consults — and the worry stream is where both-worlds-wrong evidence
    # (the LLM-authored-world case) would show up. ~20 tokens per consult.
    # reply is required for the same reason (2026-08-06).
    "required": ["world_weights", "worry", "confidence", "reply"],
}

SYSTEM = (
    "You are the meta-knowledge overlay for a Pokemon Showdown gen9ou MCTS "
    "engine. The engine searched K sampled opponent-set worlds and shows you "
    "each world's assumed sets and per-action statistics. Your ONLY lever is "
    "world_weights: shift weight toward worlds whose set assumptions best "
    "match the meta knowledge and this game's reveals, away from worlds that "
    "contradict them. You may also flag an engine action that violates a "
    "cited roles rule (flags are logged, never applied). You never pick a "
    "move and you have no channel to do so. Hedge: when the reveals do not "
    "clearly separate the worlds, keep weights near-uniform — a confidently "
    "wrong weight vector is the one failure mode that matters. A flag's "
    "'rule' must be an EXISTING roles field cited as '<species>.<field>' "
    "exactly as shown in the team knowledge (e.g. 'kingambit.value_curve', "
    "'ceruledge.entry_condition') — invented rule names are discarded "
    "mechanically, so a flag without a real citation is wasted output. "
    "A `usually:` line under an opposing species gives its TYPICAL moves and "
    "items with usage shares, and the `set '<name>':` lines under it give the "
    "REAL BUILDS those moves belong to. Judge a world by whether its assumed "
    "moves form ONE of those builds — the marginals alone cannot tell you "
    "which moves go together (a mon showing 7 plausible moves has only 4 "
    "slots, and they are not interchangeable). A world assuming a combination "
    "that matches no listed build is the one to weight DOWN; cite this as "
    "'<species>.moves' or '<species>.item'. "
    "An `engine_blind` line on a species means THE SEARCH ITSELF CANNOT "
    "MODEL that mechanic, so every world's numbers are confidently wrong "
    "about it — weigh that far above anything the visit counts say. "
    "The BOARD block is the full field state with an epistemic contract: "
    "US lines are exact; on THEM lines, hp/status/boosts/faints and "
    "anything under `revealed:` are protocol fact, while everything under "
    "`assumed(world-0 sample):` is one world's GUESS — judge worlds by "
    "whether their guesses fit the revealed lines and the usage priors, "
    "and never treat an assumed value as a reveal. An axis marked `none` "
    "really is empty; do not invent state the board does not show. "
    "A BELIEFS block, when present, lists ITEM eliminations proven by this "
    "game's play (e.g. `not leftovers` after an unhealed upkeep, or "
    "`heavydutyboots` after a chip-free entry over hazards) — stronger than "
    "a usage prior, weaker than a reveal: weight DOWN any world whose "
    "assumed item BELIEFS has eliminated. "
    "Second output, `reply`: predict what the opponent ACTUALLY CLICKS this "
    "turn — 1-3 entries mapping actions from THEIR OPTIONS (copy the exact "
    "strings) to probabilities. The worlds' reply counts show what the "
    "search thinks is BEST for them; real opponents click habits — use "
    "deployment patterns, usage shares, this game's reveals, and what this "
    "pilot has done so far. Spread probability honestly when unsure; a "
    "single 1.0 entry is a claim you will be scored on. This too is a "
    "column statement: you are predicting them, never choosing for us. "
    "Respond only with the JSON."
)


def _canon_opt(s: str) -> str:
    """Canonicalize an engine option string (or the LLM's echo of one) for
    matching: 'Switch Heatran' == 'switch heatran', 'U-turn' == 'uturn',
    'Earthquake-Tera' == 'earthquake-tera'."""
    s = (s or "").strip().lower()
    if s.startswith("switch"):
        return "switch " + re.sub(r"[^a-z0-9]", "", s[6:])
    tera = s.endswith("-tera")
    if tera:
        s = s[:-5]
    return re.sub(r"[^a-z0-9]", "", s) + ("-tera" if tera else "")


def _norm(s: str) -> str:
    return re.sub(r"[^a-z0-9]", "", (s or "").lower())


def _load_roles() -> dict:
    try:
        return json.loads((HERE / "roles.json").read_text())["roles"]
    except Exception:
        return {}


def _entry_lines(key: str, e: dict) -> list[str]:
    bits = []
    for f in _ENTRY_FIELDS:
        v = e.get(f)
        if not v:
            continue
        if f == "fact":
            continue
        if isinstance(v, dict):
            v = ", ".join(f"{a} {b}" for a, b in v.items())
        elif isinstance(v, list):
            v = ", ".join(map(str, v))
        bits.append(f"{f}={v}")
    out = [f"{key}: {'; '.join(bits)}" if bits else f"{key}:"]
    if e.get("fact"):
        out.append(f"  {e['fact']}")
    for st in e.get("sets", []):
        pv = st.get("prevalence")
        pvs = f" ~{int(100 * pv)}%" if pv is not None else ""
        out.append(f"  set {st.get('name', '?')}{pvs}: "
                   f"{', '.join(st.get('tags', []))}")
        if st.get("fact"):
            out.append(f"    {st['fact']}")
    if e.get("sequence"):
        out.append("  the play: " + " -> ".join(e["sequence"]))
    return out


class OverlayShadow:
    """One instance per player process; per-battle dossier cached by tag."""

    def __init__(self, roles: dict | None = None):
        self.roles = roles if roles is not None else _load_roles()
        self._dossiers: dict[str, str] = {}
        self._lock = threading.Lock()
        from showdown.roles_screen import SETUP     # single rule encoding
        self._setup_moves = SETUP

    # ---- dossier (stable prompt prefix per battle) ----

    def _dossier(self, battle) -> str:
        tag = getattr(battle, "battle_tag", "?")
        d = self._dossiers.get(tag)
        if d is not None:
            return d
        lines = ["=== OUR TEAM (sets are exact) ==="]
        for mon in battle.team.values():
            sp = _norm(mon.species)
            moves = ",".join(mon.moves.keys())
            lines.append(f"{sp} @ {mon.item or '?'} [{mon.ability or '?'}] "
                         f"({moves})")
            e = self.roles.get(sp)
            if e:
                lines.extend("  " + ln for ln in _entry_lines(sp, e))
        lines.append("=== THEIR PREVIEW (roles knowledge) ===")
        # The FULL previewed roster, not just appeared mons (2026-08-04 —
        # same appeared-only blindness as the translator's phantom-roster
        # fill, one layer up): this dossier builds at the FIRST consult,
        # usually T1-3 when only their lead has appeared, and is then
        # byte-stable-cached for the battle — so every later-appearing mon
        # had no roles entry, no usage prior and no curated sets, which is
        # exactly what the model's `quagsire.moves`-style worry requests
        # were reporting. The roster key needs all six for the same reason:
        # a partial key can never match a sets_by_roster book entry.
        their_species: list[str] = []
        for mon in list(battle.opponent_team.values()) + list(
                getattr(battle, "teampreview_opponent_team", None) or []):
            sp = _norm(getattr(mon, "species", ""))   # _norm strips '-*'
            if sp and sp not in their_species:
                their_species.append(sp)
        rkey = "|".join(sorted(their_species))
        their = []
        for sp in their_species:
            their.append(sp)
            e = self.roles.get(sp)
            lines.extend(_entry_lines(sp, e) if e else [f"{sp}: (no entry)"])
            prior = self._usage_prior(sp)
            if prior:
                lines.append(prior)
            lines.extend(self._book_lines(battle, sp, rkey))
        lines.extend(self._team_level(battle, their))
        d = "\n".join(lines)
        if len(self._dossiers) > 8:        # one-game-per-process anyway
            self._dossiers.clear()
        self._dossiers[tag] = d
        return d

    _book_cache = None

    @classmethod
    def _book_lines(cls, battle, species: str, roster_key: str = ""):
        """What THIS opponent has actually been seen doing with this species.

        Strictly better than the meta prior when it exists: a set's real
        prevalence is a property of the FORMAT, the ELO and the PLAYER, not
        of the species (SubProtect Gliscor is 7% in gen9 OU, 17.6% in
        monotype Ground at 1500 and 8.6% by 1630 — the same mon three
        times). Against a known opponent the observed counts settle it, and
        bots in particular run stable sets. The book fed the translator's
        tier-0 inference from the start; the model was never shown it.

        Prefers the PER-ROSTER record — the same species on a different team
        of theirs is a different set (richwoman runs three distinct Ting-Lu
        builds) — and falls back to their species-level blend.
        """
        try:
            if cls._book_cache is None:
                import json as _j
                cls._book_cache = _j.loads(
                    (HERE / "scouting_book.json").read_text())
            prof = cls._book_cache.get(
                getattr(battle, "opponent_username", "") or "")
            if not prof:
                return []
            sets = {}
            if roster_key:
                sets = (prof.get("sets_by_roster") or {}).get(roster_key) or {}
            if not sets:
                sets = prof.get("sets") or {}
            entry = next((v for k, v in sets.items() if _norm(k) == species),
                         None)
            if not entry:
                return []
            mv = sorted((entry.get("moves") or {}).items(),
                        key=lambda kv: -kv[1])[:5]
            it = sorted((entry.get("items") or {}).items(),
                        key=lambda kv: -kv[1])[:2]
            if not mv:
                return []
            games = prof.get("games", 0)
            out = ("    SEEN from this opponent (" + str(games) + " games): "
                   + ", ".join(f"{m} x{c}" for m, c in mv))
            if it:
                out += " | " + ", ".join(f"{i} x{c}" for i, c in it)
            return [out]
        except Exception:
            return []

    _ps_cache = None

    @classmethod
    def _ps_sets(cls, species: str):
        """Curated whole builds: [(name, moves, item)]. The joint counterpart
        to the usage marginals."""
        try:
            if cls._ps_cache is None:
                import json as _j
                raw = _j.loads((HERE / "ps_sets_gen9.json").read_text())
                dex = raw.get("gen9ou", {}).get("dex", {})
                cls._ps_cache = {_norm(k): v for k, v in dex.items()}
            out = []
            for name, st in (cls._ps_cache.get(species) or {}).items():
                mv = [str(m) for m in (st.get("moves") or []) if not isinstance(m, list)]
                slashed = [m[0] for m in (st.get("moves") or []) if isinstance(m, list)]
                out.append((name, (mv + slashed)[:4], st.get("item") or "?"))
            return out
        except Exception:
            return []

    _chaos_cache = None

    @classmethod
    def _usage_prior(cls, species: str, n_moves: int = 5, n_items: int = 3):
        """What this species TYPICALLY runs, as a one-line prior.

        The dossier already shows each world's ASSUMED set, but never what is
        normal — so the model could see "world 0 says Choice Band, world 1
        says Toxic Orb" with no way to judge which is plausible, which is
        exactly the judgement its only lever (world_weights) requires. This
        is the missing half, and it was the model's own standing request:
        per-species `moves` and `item` were the top UNRESOLVED citations for
        weeks (ragingbolt.item 23, blissey.moves 21, greattusk.moves 15).

        Shares are DEFLATION-CORRECTED — this chaos dump's weighted counts run
        0.35-0.67 of Raw count per species, identically across the item,
        ability and move axes, so raw shares understate everything by ~2x.
        """
        try:
            if cls._chaos_cache is None:
                from showdown.chaos_stats import ChaosStats
                cls._chaos_cache = ChaosStats(format="gen9ou")
            st = cls._chaos_cache.pokemon.get(species)
            if st is None:
                return None
            items = sorted(st._items.items(), key=lambda kv: -kv[1])[:n_items]
            moves = sorted(st._moves.items(), key=lambda kv: -kv[1])[:n_moves]
            tot_i = sum(st._items.values()) or 1
            mv = ", ".join(f"{m} {p / tot_i:.0%}" for m, p in moves)
            it = ", ".join(f"{i} {p / tot_i:.0%}" for i, p in items)
            out = [f"  usually: {mv} | {it}"]
            # MARGINALS CANNOT SHOW WHICH MOVES GO TOGETHER. Gliscor is the
            # type case: protect 82 / earthquake 72 / knockoff 53 / toxic 43 /
            # swordsdance 33 / spikes 29 / facade 28 is SEVEN moves for four
            # slots, and nothing in that list says Swords Dance pairs with
            # Facade while Spikes pairs with Toxic — two different Pokemon.
            # The curated sets are the JOINT view, which is what lets the
            # model match a world's assumed moves to a real build instead of
            # scoring each move independently.
            cand = cls._ps_sets(species)
            for name, mv4, item in cand[:3]:
                out.append(f"    set '{name}': {', '.join(mv4)} @ {item}")
            return "\n".join(out)
        except Exception:
            return None

    def _team_level(self, battle, their: list[str]) -> list[str]:
        """Team-derived knowledge no per-species entry can carry.

        Three things the leaf eval structurally cannot compute: which of our
        mons is the SOLE holder of a role (so losing it is permanent, not
        incremental), which field resource each mon depends on and who
        provides it (its setter's death disables them), and — the timing
        signal — which of THEIR mons currently blank each of our wincons, so
        a wincon is described as live or as waiting on a specific removal.
        Degrades to nothing on any failure; the dossier must never break.
        """
        try:
            from showdown.team_roles import (analyze_roster, wincon_outlook,
                                             wincon_report)
            ours = [{"species": _norm(m.species),
                     "moves": list(m.moves.keys()),
                     "item": m.item or ""} for m in battle.team.values()]
            a = analyze_roster(ours, self.roles, name="ours")
            out = ["=== OUR TEAM STRUCTURE (derived; the eval cannot see this) ==="]
            for sp, roles_ in (a.get("sole") or {}).items():
                out.append(f"{sp} is our ONLY {', '.join(roles_)} — losing it "
                           f"forfeits that role for the rest of the game")
            for sp, toks in (a.get("provides") or {}).items():
                dep = (a.get("dependents") or {}).get(sp)
                if dep:
                    verb = "depends" if len(dep) == 1 else "depend"
                    out.append(f"{sp} provides {'/'.join(toks)}; "
                               f"{', '.join(dep)} {verb} on it and "
                               f"{'is' if len(dep) == 1 else 'are'} much "
                               f"weaker once it dies")
            for o in a.get("orphans") or []:
                out.append(f"{o['species']} needs {o['needs']} and NOBODY on "
                           f"our side provides it")
            rows = wincon_outlook(a, their, self.roles)
            body = wincon_report(rows)
            if body:
                out.append("--- wincon timing (a wincon is not live until the "
                           "answers to it are gone)")
                # keep one level of nesting: the sub-lines belong to the
                # wincon above them, and a flat list reads as unrelated claims
                out.extend(ln[2:] if ln.startswith("    ") else ln.strip()
                           for ln in body)
            return out if len(out) > 1 else []
        except Exception:
            return []

    # ---- gating ----

    def consult_reasons(self, battle, ranked, results) -> list[str]:
        reasons = []
        if battle.turn <= 3:
            reasons.append("opening")
        total = sum(r.visits for r in ranked) or 1
        if len(ranked) >= 2 and \
                (ranked[0].visits - ranked[1].visits) / total < 0.10:
            reasons.append("near-tie")
        tops = set()
        for res in results or []:
            side = getattr(res, "side_one", None) or []
            if side:
                tops.add(max(side, key=lambda r: r.visits).move_choice)
        if len(tops) > 1:
            reasons.append("world-disagreement")
        reasons.extend(self._role_reasons(battle, ranked))
        return reasons

    @staticmethod
    def _active_resources(battle) -> set[str]:
        """Canonical tokens for every field resource CURRENTLY up on our
        side. A resource can live in three different places — weather,
        fields (terrain and Trick Room), and our own side conditions
        (Tailwind, screens) — and the gate has to look in all of them."""
        out = set()
        for w in getattr(battle, "weather", {}) or {}:
            tok = _WEATHER.get(str(getattr(w, "name", w)).split(".")[-1].lower())
            if tok:
                out.add(tok)
        for f in getattr(battle, "fields", {}) or {}:
            name = str(getattr(f, "name", f)).split(".")[-1].lower()
            if name.endswith("terrain"):
                out.add(name.replace("_", ""))
            elif "trick" in name:
                out.add("trickroom")
        for c in getattr(battle, "side_conditions", {}) or {}:
            name = str(getattr(c, "name", c)).split(".")[-1].lower()
            if "tailwind" in name:
                out.add("tailwind")
            elif "screen" in name or "veil" in name or "reflect" in name:
                out.add("screens")
        return out

    def _role_reasons(self, battle, ranked) -> list[str]:
        out = []
        ours = {_norm(m.species): m for m in battle.team.values()}
        fallen = sum(1 for m in battle.team.values() if m.fainted)
        active = next((m for m in battle.team.values() if m.active), None)
        act_sp = _norm(active.species) if active else ""
        active_res = self._active_resources(battle)
        top = {r.move_choice for r in ranked[:4]}
        for sp, mon in ours.items():
            e = self.roles.get(sp) or {}
            res = e.get("resource")
            if res and not mon.fainted and sp != act_sp:
                # resource_tokens() normalises the PROSE the field actually
                # holds ("Psychic Terrain", "Trick Room turns", "snow +
                # Aurora Veil"). The old exact-string compare matched only
                # the five single-token values and so fired EVERY TURN for
                # the rest — a permanent false positive introduced when the
                # 2026-08-02 entries wrote prose into a machine-read field.
                from showdown.team_roles import resource_tokens
                want = resource_tokens(res)
                if want and not (want & active_res):
                    out.append(f"weather-down:{sp}")
            if e.get("value_curve") == "grows_with_own_faints" and \
                    fallen <= 1 and f"switch {sp}" in top:
                out.append(f"cleaner-early:{sp}")
        if active is not None and not active.fainted:
            e = self.roles.get(act_sp) or {}
            if e.get("entry_condition") == "full_hp" and \
                    (active.current_hp_fraction or 1.0) < 0.99 and \
                    any(c.split("-")[0] in self._setup_moves for c in top):
                out.append(f"chipped-setup:{act_sp}")
        return out

    # ---- emission ----

    @staticmethod
    def _emit_worlds(results, states) -> list[dict]:
        worlds = []
        for i, res in enumerate(results or []):
            w = {"id": i}
            side = getattr(res, "side_one", None) or []
            w["rows"] = [[r.move_choice, int(r.visits),
                          round(r.total_score / r.visits, 3) if r.visits else 0]
                         for r in sorted(side, key=lambda r: -r.visits)[:8]]
            opp = getattr(res, "side_two", None) or []
            if opp:
                w["their_replies"] = [
                    [r.move_choice, int(r.visits)]
                    for r in sorted(opp, key=lambda r: -r.visits)[:6]]
            if states is not None and i < len(states):
                sets = {}
                try:
                    for p in states[i].side_two.pokemon:
                        if getattr(p, "hp", 0) <= 0:
                            continue
                        sets[p.id] = {
                            "item": getattr(p, "item", "?"),
                            "ability": getattr(p, "ability", "?"),
                            "tera": getattr(p, "tera_type", "?"),
                            "moves": [m.id for m in getattr(p, "moves", [])
                                      if m.id != "none"],
                        }
                except Exception:
                    pass
                w["assumed_sets"] = sets
            worlds.append(w)
        return worlds

    @staticmethod
    def _appendix(battle) -> dict:
        def hp(m):
            return round(m.current_hp_fraction or 0.0, 2)
        opp_rev = {}
        for m in battle.opponent_team.values():
            opp_rev[_norm(m.species)] = {
                "hp": hp(m), "fainted": m.fainted,
                "revealed_moves": list(m.moves.keys()),
                "item": m.item if m.item and m.item != "unknown_item" else None,
                "ability": m.ability,
                "tera": str(m.tera_type).split(".")[-1].lower()
                if m.tera_type else None,
            }
        return {
            "turn": battle.turn,
            "weather": {str(k).split(".")[-1].lower(): v
                        for k, v in (getattr(battle, "weather", {}) or {}).items()},
            "our_side": {str(k).split(".")[-1].lower(): v for k, v in
                         (battle.side_conditions or {}).items()},
            "their_side": {str(k).split(".")[-1].lower(): v for k, v in
                           (battle.opponent_side_conditions or {}).items()},
            "our_mons": {_norm(m.species): {"hp": hp(m), "fainted": m.fainted,
                                            "active": bool(m.active)}
                         for m in battle.team.values()},
            "their_mons": opp_rev,
        }

    # ---- consult (daemon thread; never blocks the move) ----

    def _nominations(self, reasons, ranked) -> list[str]:
        """Actions a roles rule demands that the merged search starved."""
        total = sum(r.visits for r in ranked) or 1
        share = {r.move_choice: r.visits / total for r in ranked}
        out = []
        for r in reasons:
            if r.startswith("weather-down:"):
                act = f"switch {r.split(':', 1)[1]}"
                if share.get(act, 0.0) < STARVED_SHARE:
                    out.append(act)
        return out

    @staticmethod
    def _advocate_priors(options: list[str], action: str) -> list[float] | None:
        """Our-side root prior array with ADVOCATE_PRIOR on the nominated
        action, remainder uniform. None when the action isn't a root option
        (fainted setter, trapped, or a nomination bug) or is the only one."""
        if action not in options or len(options) < 2:
            return None
        rest = (1.0 - ADVOCATE_PRIOR) / (len(options) - 1)
        return [ADVOCATE_PRIOR if o == action else rest for o in options]

    def _advocate(self, state_str: str, action: str, engine_choice: str) -> dict:
        import poke_engine as pe
        out = {"action": action}
        try:
            warm = pe.monte_carlo_tree_search(pe.State.from_string(state_str), 1)
            opts = [r.move_choice for r in warm.side_one]
            s1 = self._advocate_priors(opts, action)
            if s1 is None:
                out["skip"] = "not-a-root-option"
                return out
            s2 = [1.0 / (len(warm.side_two) or 1)] * (len(warm.side_two) or 1)
            res = pe.monte_carlo_tree_search_with_priors(
                pe.State.from_string(state_str), s1, s2, ADVOCATE_MS)
            by = {r.move_choice: r for r in res.side_one}
            adv = by.get(action)
            eng = by.get(engine_choice)
            if adv and adv.visits:
                out["deep_visits"] = int(adv.visits)
                out["deep_q"] = round(adv.total_score / adv.visits, 4)
            if eng and eng.visits:
                out["engine_q_same_tree"] = round(
                    eng.total_score / eng.visits, 4)
            if "deep_q" in out and "engine_q_same_tree" in out:
                out["advocate_prefers"] = out["deep_q"] > out["engine_q_same_tree"]
        except Exception as e:
            out["error"] = repr(e)
        return out

    @staticmethod
    def _reply_options(results) -> list[str]:
        """Union of the opponent's searched actions across worlds, ordered
        by summed visit share — the candidate set the reply prediction is
        made (and validated) over. Full side_two, not the emission's top-6:
        the interesting misses are exactly the columns the search starved."""
        share: dict[str, float] = {}
        for res in results or []:
            side = getattr(res, "side_two", None) or []
            total = sum(r.visits for r in side) or 1
            for r in side:
                if r.move_choice.lower() in ("none", "no move", "nomove"):
                    continue
                share[r.move_choice] = (share.get(r.move_choice, 0.0)
                                        + r.visits / total)
        return [k for k, _ in
                sorted(share.items(), key=lambda kv: -kv[1])][:14]

    def _build_rec(self, battle, ranked, results, states, reasons,
                   obs=None) -> dict:
        rec = {
            "ts": time.time(),
            "tag": getattr(battle, "battle_tag", "?"),
            "turn": battle.turn,
            "reasons": reasons,
            "engine_choice": ranked[0].move_choice,
            "engine_margin": round(
                (ranked[0].visits - ranked[1].visits)
                / (sum(r.visits for r in ranked) or 1), 3)
            if len(ranked) >= 2 else 1.0,
            "worlds": self._emit_worlds(results, states),
            "appendix": self._appendix(battle),
            "reply_options": self._reply_options(results),
        }
        # full-board fact sheet (2026-08-04): replaces the appendix in the
        # PROMPT (strict superset — adds boosts, volatiles, PP, counters,
        # tera availability, and the revealed/assumed split the appendix
        # never had). The appendix stays in the rec because flip_audit and
        # friends read it; the sheet is logged too so audits can see exactly
        # what the model saw. Prompt change ⇒ flip rates before/after this
        # date are not comparable.
        try:
            if states:
                from showdown.state_sheet import render_sheet
                rec["sheet"] = render_sheet(states[0], battle=battle, obs=obs)
        except Exception:
            pass
        return rec

    def _log(self, rec) -> None:
        try:
            with self._lock, open(LOG, "a") as f:
                f.write(json.dumps(rec) + "\n")
        except Exception:
            pass

    def live_consult(self, battle, ranked, results, states, obs=None):
        """SYNCHRONOUS consult for live mode: returns the reweighted merged
        ranking to PLAY when the LLM clears the extreme-weight gate, else
        None (identity — the engine's own choice stands). Everything is
        logged in the shadow format with mode='live'."""
        reasons = self.consult_reasons(battle, ranked, results)
        if not reasons or len(results or []) < 2:
            return None
        rec = self._build_rec(battle, ranked, results, states, reasons,
                              obs=obs)
        rec["mode"] = "live"
        rec["applied"] = False
        t0 = time.monotonic()
        parsed = None
        try:
            raw = self._ask(self._dossier(battle), rec,
                            timeout=LIVE_TIMEOUT_S)
            rec["llm"] = raw
            parsed = self._validate(raw, len(rec["worlds"]),
                                    options=rec.get("reply_options"))
            rec["valid"] = parsed is not None
            if parsed:
                rec["reply_pred"] = parsed["reply"]
                rec["reply_dropped"] = parsed["reply_dropped"]
        except Exception as e:
            rec["error"] = repr(e)
        rec["latency_s"] = round(time.monotonic() - t0, 2)
        out = None
        if parsed:
            w = parsed["world_weights"]
            rec["llm_weights"] = [round(x, 3) for x in w]
            if max(w) >= APPLY_MIN:
                from showdown.gen9_player import _merge_mcts_results
                merged = _merge_mcts_results(results, weights=w)
                if merged:
                    rec["applied"] = True
                    rec["applied_top"] = merged[0].move_choice
                    rec["applied_flip"] = (merged[0].move_choice
                                           != ranked[0].move_choice)
                    if rec["applied_flip"] and states:
                        try:
                            rec["w0_state"] = states[0].to_string()
                        except Exception:
                            pass
                    out = merged
        self._log(rec)
        return out

    def maybe_consult(self, battle, ranked, results, states, obs=None):
        # obs is the translator's per-battle BattleObservations; the sheet
        # (and thus obs) is consumed HERE on the main thread, before the
        # daemon thread starts, so the translator mutating it next move
        # cannot race the render.
        reasons = self.consult_reasons(battle, ranked, results)
        if not reasons or len(results or []) < 2:
            return
        rec = self._build_rec(battle, ranked, results, states, reasons,
                              obs=obs)
        # CB_ADVOCATE_MS=0 disables the advocate entirely (shadow-path only —
        # live_consult never runs it). Retired on the ladder 2026-08-01: its
        # question is answered (95% confirms offline, 52/53 live) and its
        # thread-per-firing search bursts are the prime suspect for the
        # session RSS ratchet (glibc arena retention; 3 oom-kills in 12h).
        rec["nominations"] = (self._nominations(reasons, ranked)
                              if ADVOCATE_MS > 0 else [])
        # always snapshot world-0 (a few KB): flips store it so flip_audit
        # can oracle them WITHOUT a --dump-states pool join — the reweight
        # A/B's 39 played flips were unauditable for lack of this
        w0_str = None
        if states:
            try:
                w0_str = states[0].to_string()   # snapshot on the main thread
            except Exception:
                pass
        dossier = self._dossier(battle)
        snapshot = list(results)
        threading.Thread(target=self._consult_bg,
                         args=(rec, dossier, snapshot, w0_str),
                         daemon=True).start()

    def _consult_bg(self, rec, dossier, results, w0_str=None):
        t0 = time.monotonic()
        if w0_str and rec.get("nominations"):
            # advocate first: pure local CPU, no network dependency — an
            # ollama outage must not lose the starved-action measurement
            rec["advocate"] = [
                self._advocate(w0_str, act, rec["engine_choice"])
                for act in rec["nominations"][:2]]
        try:
            raw = self._ask(dossier, rec)
            rec["llm"] = raw
            rec["latency_s"] = round(time.monotonic() - t0, 2)
            parsed = self._validate(raw, len(rec["worlds"]),
                                    options=rec.get("reply_options"))
            rec["valid"] = parsed is not None
            if parsed:
                rec["flips"] = self._flips(parsed["world_weights"], results,
                                           rec["engine_choice"])
                rec["dropped_flags"] = parsed["dropped_flags"]
                rec["reply_pred"] = parsed["reply"]
                rec["reply_dropped"] = parsed["reply_dropped"]
                if w0_str and any(v.get("flip") for v in rec["flips"].values()
                                  if isinstance(v, dict)):
                    rec["w0_state"] = w0_str
        except Exception as e:
            rec["error"] = repr(e)
            rec["latency_s"] = round(time.monotonic() - t0, 2)
        self._log(rec)

    @staticmethod
    def _turn_message(rec) -> str:
        """The per-turn user message. When the fact sheet rendered, it IS
        the board (the appendix is its strict subset and sending both would
        spend the same tokens twice inside num_ctx=12288); the appendix JSON
        is the fallback so an import or render failure degrades to the old
        prompt rather than a stateless consult."""
        keys = ("turn", "reasons", "engine_choice", "engine_margin", "worlds")
        sheet = rec.get("sheet")
        if not sheet:
            keys += ("appendix",)
        turn_block = json.dumps({k: rec[k] for k in keys},
                                separators=(",", ":"))
        msg = ""
        if sheet:
            msg += f"{sheet}\n"
        msg += f"SEARCH:\n{turn_block}\n"
        opts = rec.get("reply_options") or []
        if opts:
            msg += f"THEIR OPTIONS (predict `reply` from these, exact " \
                   f"strings): {', '.join(opts)}\n"
        msg += "Weigh the worlds; flag at most 2 rows; predict their click."
        return msg

    def _ask(self, dossier, rec, timeout: float = TIMEOUT_S) -> dict:
        body = {
            "model": MODEL,
            "messages": [
                {"role": "system", "content": SYSTEM},
                {"role": "user", "content": dossier},
                {"role": "user", "content": self._turn_message(rec)},
            ],
            "stream": False,
            "think": False,
            "format": SCHEMA,
            "keep_alive": "30m",
            "options": {"temperature": 0.2, "num_ctx": 12288},
        }
        req = urllib.request.Request(
            OLLAMA, data=json.dumps(body).encode(),
            headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=timeout) as r:
            out = json.loads(r.read())
        return json.loads(out["message"]["content"])

    def _validate(self, raw: dict, n_worlds: int,
                  options: list | None = None) -> dict | None:
        try:
            ww = {int(k): float(v) for k, v in raw["world_weights"].items()
                  if 0 <= int(k) < n_worlds and float(v) > 0}
        except Exception:
            return None
        if not ww:
            return None
        weights = [ww.get(i, 0.001) for i in range(n_worlds)]
        s = sum(weights)
        weights = [w / s for w in weights]
        kept, dropped = [], 0
        for fl in raw.get("flags", []) or []:
            if self._rule_resolves(fl.get("rule", "")):
                kept.append(fl)
            else:
                dropped += 1
        raw["flags"] = kept
        # reply prediction: match the model's action strings against the
        # offered option set, drop what doesn't resolve, renormalize. A bad
        # or missing reply NEVER fails the whole consult (identity
        # philosophy) — it just logs reply=None with the drop count.
        reply, r_dropped = None, 0
        if options:
            canon = {_canon_opt(o): o for o in options}
            picked = {}
            try:
                for k, v in (raw.get("reply") or {}).items():
                    o = canon.get(_canon_opt(k))
                    if o is None or not float(v) > 0:
                        r_dropped += 1
                        continue
                    picked[o] = picked.get(o, 0.0) + float(v)
            except Exception:
                picked, r_dropped = {}, r_dropped + 1
            if picked:
                top = sorted(picked.items(), key=lambda kv: -kv[1])[:4]
                tot = sum(p for _, p in top)
                reply = {o: round(p / tot, 4) for o, p in top}
        return {"world_weights": weights, "dropped_flags": dropped,
                "reply": reply, "reply_dropped": r_dropped}

    def _rule_resolves(self, rule: str) -> bool:
        parts = _norm_path(rule)
        if not parts:
            return False
        e = self.roles.get(parts[0])
        if e is None:
            return False
        if len(parts) == 1 or parts[1] in e:
            return True
        # `X.ability` is the natural way to cite a species whose ability is
        # recorded as a SPLIT (63/37 Toxic Chain / Guard Dog); rejecting it
        # on the field name alone fails a semantically correct citation
        if parts[1] == "ability" and "ability_split" in e:
            return True
        # ...and once abilities became visible in the dossier (2026-08-02) the
        # model started citing them BY VALUE — `gholdengo.goodasgold` rather
        # than `gholdengo.ability`, 11 times in the first session. That names a
        # real fact in a real entry, so accept it rather than discard the
        # output on a naming convention it was never told.
        named = {_norm(e.get("ability") or "")} | {
            _norm(k) for k in (e.get("ability_split") or {})}
        if parts[1] in named - {""}:
            return True
        # `X.moves` / `X.item` were the model's TOP unresolved citations for
        # weeks. They are not roles fields — but as of 2026-08-03 the dossier
        # carries a per-species usage prior ("usually: ... | ..."), so those
        # citations now reference something real and are honoured.
        if parts[1] in ("moves", "move", "item", "items", "usage"):
            return self._usage_prior(parts[0]) is not None
        return False

    def _flips(self, llm_weights, results, engine_choice) -> dict:
        from showdown.gen9_player import _merge_mcts_results
        n = len(results)
        flips = {}
        for lam in LAMBDAS:
            blend = [lam * w + (1 - lam) * (1 / n) for w in llm_weights]
            merged = _merge_mcts_results(results, weights=blend)
            top = merged[0].move_choice if merged else None
            flips[str(lam)] = {"top": top, "flip": top != engine_choice}
        flips["llm_weights"] = [round(w, 3) for w in llm_weights]
        return flips


def _norm_path(rule: str) -> list[str]:
    parts = [p for p in re.split(r"[.\[\]]+", (rule or "").strip()) if p]
    return [_norm(parts[0])] + [p.lower() for p in parts[1:]] if parts else []
