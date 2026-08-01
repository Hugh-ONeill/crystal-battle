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
_ENTRY_FIELDS = ("fact", "tags", "axis", "preserve", "deployment",
                 "lead_intent", "entry_condition", "value_curve", "resource",
                 "requires", "single_build_note")

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
    },
    "required": ["world_weights", "confidence"],
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
    "Respond only with the JSON."
)


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
        for mon in battle.opponent_team.values():
            sp = _norm(mon.species)
            e = self.roles.get(sp)
            lines.extend(_entry_lines(sp, e) if e else [f"{sp}: (no entry)"])
        d = "\n".join(lines)
        if len(self._dossiers) > 8:        # one-game-per-process anyway
            self._dossiers.clear()
        self._dossiers[tag] = d
        return d

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

    def _role_reasons(self, battle, ranked) -> list[str]:
        out = []
        ours = {_norm(m.species): m for m in battle.team.values()}
        fallen = sum(1 for m in battle.team.values() if m.fainted)
        active = next((m for m in battle.team.values() if m.active), None)
        act_sp = _norm(active.species) if active else ""
        weather_now = None
        for w in getattr(battle, "weather", {}) or {}:
            weather_now = _WEATHER.get(str(w).split(".")[-1].lower())
        fields = {str(f).split(".")[-1].lower()
                  for f in (getattr(battle, "fields", {}) or {})}
        top = {r.move_choice for r in ranked[:4]}
        for sp, mon in ours.items():
            e = self.roles.get(sp) or {}
            res = e.get("resource")
            if res and not mon.fainted and sp != act_sp:
                up = (res == weather_now or
                      (res == "grassyterrain" and any("grassy" in f
                                                      for f in fields)))
                if not up:
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

    def _build_rec(self, battle, ranked, results, states, reasons) -> dict:
        return {
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
        }

    def _log(self, rec) -> None:
        try:
            with self._lock, open(LOG, "a") as f:
                f.write(json.dumps(rec) + "\n")
        except Exception:
            pass

    def live_consult(self, battle, ranked, results, states):
        """SYNCHRONOUS consult for live mode: returns the reweighted merged
        ranking to PLAY when the LLM clears the extreme-weight gate, else
        None (identity — the engine's own choice stands). Everything is
        logged in the shadow format with mode='live'."""
        reasons = self.consult_reasons(battle, ranked, results)
        if not reasons or len(results or []) < 2:
            return None
        rec = self._build_rec(battle, ranked, results, states, reasons)
        rec["mode"] = "live"
        rec["applied"] = False
        t0 = time.monotonic()
        parsed = None
        try:
            raw = self._ask(self._dossier(battle), rec,
                            timeout=LIVE_TIMEOUT_S)
            rec["llm"] = raw
            parsed = self._validate(raw, len(rec["worlds"]))
            rec["valid"] = parsed is not None
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
                    out = merged
        self._log(rec)
        return out

    def maybe_consult(self, battle, ranked, results, states):
        reasons = self.consult_reasons(battle, ranked, results)
        if not reasons or len(results or []) < 2:
            return
        rec = self._build_rec(battle, ranked, results, states, reasons)
        rec["nominations"] = self._nominations(reasons, ranked)
        w0_str = None
        if rec["nominations"] and states:
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
            parsed = self._validate(raw, len(rec["worlds"]))
            rec["valid"] = parsed is not None
            if parsed:
                rec["flips"] = self._flips(parsed["world_weights"], results,
                                           rec["engine_choice"])
                rec["dropped_flags"] = parsed["dropped_flags"]
        except Exception as e:
            rec["error"] = repr(e)
            rec["latency_s"] = round(time.monotonic() - t0, 2)
        self._log(rec)

    def _ask(self, dossier, rec, timeout: float = TIMEOUT_S) -> dict:
        turn_block = json.dumps(
            {k: rec[k] for k in ("turn", "reasons", "engine_choice",
                                 "engine_margin", "worlds", "appendix")},
            separators=(",", ":"))
        body = {
            "model": MODEL,
            "messages": [
                {"role": "system", "content": SYSTEM},
                {"role": "user", "content": dossier},
                {"role": "user", "content": f"TURN STATE:\n{turn_block}\n"
                 "Weigh the worlds; flag at most 2 rows."},
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

    def _validate(self, raw: dict, n_worlds: int) -> dict | None:
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
        return {"world_weights": weights, "dropped_flags": dropped}

    def _rule_resolves(self, rule: str) -> bool:
        parts = _norm_path(rule)
        if not parts:
            return False
        e = self.roles.get(parts[0])
        if e is None:
            return False
        return len(parts) == 1 or parts[1] in e

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
