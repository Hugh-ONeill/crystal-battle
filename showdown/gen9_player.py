# Gen 9 live player: poke-env Player driving poke-engine MCTS through the
# gen9 translator (showdown/gen9_translator.py).
#
# Primary target is gen9ou on the PokeAgent server (pokeagentshowdown.com,
# the living benchmark whose 2025 gen9 OU bracket was won by stock foul-play,
# this project's upstream). Also plays gen9monotype: the server never offers
# tera there (Terastal Clause), so "-tera" search choices simply fail to map
# and the next-best candidate is used.
#
# Usage (local sparring vs foul-play):
#   .venv/bin/python showdown/gen9_player.py --local --username CBGen9 \
#       --mode challenge --user-to-challenge FPSpar1 \
#       --format gen9ou --team teams/gen9ou_sample.txt --n-games 1
#
# PokeAgent (bot usernames should start with "PAC"):
#   .venv/bin/python showdown/gen9_player.py --server pokeagent \
#       --username PAC-Crystal9 --password ... --mode ladder --format gen9ou

from __future__ import annotations

import asyncio
import argparse
import json
import random
import re
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).parent.parent))

import poke_engine as pe
from poke_env.player import Player
from poke_env import AccountConfiguration, ServerConfiguration
from poke_env.ps_client.server_configuration import ShowdownServerConfiguration

from showdown.name_mapping import _normalize
from showdown.gen9_translator import Gen9Translator
from showdown.poke_engine_player import _score_switch_in
from showdown.airi_bridge import AiriBridge, DEFAULT_URL as AIRI_DEFAULT_URL
from showdown.beat_director import (Director, Event, ProtocolScanner,
                                    TurnContext)

# every numeric desk read (top line's avg MCTS score per searched decision)
# plus the game outcome, one JSONL line per game — the raw material for
# Brier/calibration scoring (showdown/brier_report.py). Default-on: the
# ledger should accrue from every game played, commentated or not.
DESK_LOG_DEFAULT = str(Path(__file__).parent / "desk_reads.jsonl")

LOCAL_SERVER = ServerConfiguration(
    "ws://localhost:8000/showdown/websocket",
    "https://play.pokemonshowdown.com/action.php?",
)

# Season 2 (2026) server — the season-1 pokeagentshowdown.com host is gone.
# Accounts are issued by the team portal at battling.pokeagentchallenge.com
# (Create Team -> generate named AI agent -> credentials), NOT the classic
# gear-icon Showdown registration. Auth endpoint below is the best guess for
# their login service — verify on first connect and adjust if login fails.
POKEAGENT_SERVER = ServerConfiguration(
    "wss://battling.pokeagentchallenge.com/showdown/websocket",
    "https://battling.pokeagentchallenge.com/action.php?",
)

SERVERS = {
    "local": LOCAL_SERVER,
    "pokeagent": POKEAGENT_SERVER,
    "showdown": ShowdownServerConfiguration,
}


_TIME_LEFT_RE = re.compile(r"has (\d+) sec")
_TIME_TOTAL_RE = re.compile(r"\|\s*(\d+) sec total")


_TIME_THIS_TURN_RE = re.compile(r"Time left: (\d+) sec this turn")


def _parse_clock(battle, username: str | None) -> tuple[int | None, int | None]:
    """(bank_seconds, per_turn_cap_seconds) from the server's |inactive|
    messages, or (None, None) when the timer is off/unseen.

    The server re-sends 'Time left: X sec this turn | Y sec total' every
    request while the timer runs (room-battle.ts line ~332), so the latest
    reading is authoritative — regeneration (+addPerTurn/turn) shows up in
    the next reading and never needs to be modeled. Per-player countdown
    warnings ('<user> has N seconds left.') reflect per-turn remaining;
    treated as a conservative bank reading when they're the freshest line.
    PokeAgent runs multiple timer variants (standard + extended for LLMs);
    the first |inactive| lines of a battle reveal which one applies."""
    replay = getattr(battle, "_replay_data", None)
    if not replay:
        return None, None
    bank = cap = None
    uname = (username or "").lower()
    for event in replay:
        if len(event) < 3 or event[1] != "inactive":
            continue
        # the timer text itself contains pipes ('... this turn | N sec
        # total | ...') and the protocol splits on '|', so the reading is
        # scattered across event[2:]; rejoin before parsing
        msg = "|".join(str(x) for x in event[2:])
        low = msg.lower()
        if uname and low.startswith(uname) and "sec" in low:
            m = _TIME_LEFT_RE.search(msg)
            if m:
                bank = int(m.group(1))
        elif "total" in low:
            m = _TIME_TOTAL_RE.search(msg)
            if m:
                bank = int(m.group(1))
            mc = _TIME_THIS_TURN_RE.search(msg)
            if mc:
                cap = int(mc.group(1))
    return bank, cap


def _time_left(battle, username: str | None) -> int | None:
    """Back-compat shim: just the bank."""
    return _parse_clock(battle, username)[0]


# hazard move -> (poke-env SideCondition name, max layers). Re-setting a
# hazard already at max is a no-op: it leaves the state (and eval) identical
# to the best line, so a flat-eval MCTS can't tell it apart and may spend the
# turn on it — measured as 19 consecutive Spikes (16 wasted) at 5s. The
# monotype bench fixed this with `_best_useful`; this is the live-player port.
_HAZARD_MAX = {"stealthrock": ("STEALTH_ROCK", 1), "spikes": ("SPIKES", 3),
               "toxicspikes": ("TOXIC_SPIKES", 2), "stickyweb": ("STICKY_WEB", 1)}


def _is_noop_hazard(move_id: str, battle) -> bool:
    """True if move_id sets a hazard already at max on the opponent's side."""
    hz = _HAZARD_MAX.get(move_id)
    if hz is None:
        return False
    name, cap = hz
    for cond, layers in battle.opponent_side_conditions.items():
        if cond.name == name:
            return layers >= cap
    return False


# dedicated status moves -> type/ability immunity check (ported from the
# monotype bench). Already-statused / behind-Substitute targets block all of
# these regardless of move. Ability checks are best-effort: the opponent's
# ability is often unrevealed, in which case only the type immunity applies.
_STATUS_IMMUNE = {
    "toxic":      lambda t, ab: bool(t & {"poison", "steel"}) or ab == "immunity",
    "willowisp":  lambda t, ab: "fire" in t or ab in {"waterveil", "waterbubble",
                                                       "thermalexchange", "comatose"},
    "thunderwave": lambda t, ab: "ground" in t or "electric" in t or ab == "limber",
    "spore":      lambda t, ab: "grass" in t or ab in {"insomnia", "vitalspirit",
                                                       "comatose", "sweetveil", "overcoat"},
    "sleeppowder": lambda t, ab: "grass" in t or ab in {"insomnia", "vitalspirit",
                                                        "comatose", "sweetveil", "overcoat"},
    "glare":      lambda t, ab: ab == "limber",
    "poisonpowder": lambda t, ab: bool(t & {"poison", "steel"}) or ab == "immunity",
}


def _is_noop_status(move_id: str, battle) -> bool:
    """True if move_id is a status move guaranteed to fail on the opponent's
    current active (already statused, behind a Sub, or type/ability immune)."""
    check = _STATUS_IMMUNE.get(move_id)
    opp = battle.opponent_active_pokemon
    if check is None or opp is None:
        return False
    from poke_env.battle.effect import Effect
    if Effect.SUBSTITUTE in (opp.effects or {}):
        return True
    if opp.status is not None:          # already has a major status
        return True
    types = {t.name.lower() for t in (opp.type_1, opp.type_2) if t}
    ability = (opp.ability or "").lower()
    return check(types, ability)


def _merge_mcts_results(results) -> list:
    """Combine side_one results from searches over different sampled
    opponent worlds: sum visits and scores per move_choice, rank by visits.
    A move that only looks good in one world loses to one that holds up
    across all of them."""
    merged: dict[str, SimpleNamespace] = {}
    for result in results:
        for r in result.side_one:
            m = merged.get(r.move_choice)
            if m is None:
                merged[r.move_choice] = SimpleNamespace(
                    move_choice=r.move_choice,
                    visits=r.visits, total_score=r.total_score)
            else:
                m.visits += r.visits
                m.total_score += r.total_score
    return sorted(merged.values(), key=lambda m: -m.visits)


_GEN9_DATA = None


def _gen9_data():
    global _GEN9_DATA
    if _GEN9_DATA is None:
        from poke_env.data import GenData
        _GEN9_DATA = GenData.from_gen(9)
    return _GEN9_DATA


def _move_display(move_id: str) -> str:
    """'makeitrain' -> 'Make It Rain'. Beats must carry display names: fed
    the raw id, the character free-associates (a real 'makeitrain' beat
    became 'an Iron Ball hit' on camera)."""
    entry = _gen9_data().moves.get(_normalize(move_id))
    return entry["name"] if entry and "name" in entry else move_id


def _species_display(species_id: str) -> str:
    """'tinglu' -> 'Ting-Lu'."""
    entry = _gen9_data().pokedex.get(_normalize(species_id))
    return entry["name"] if entry and "name" in entry else species_id


def _species_stats(display_name: str) -> tuple[int, int] | None:
    """Base (Atk, SpA) for the director's burn physical-vs-special split."""
    entry = _gen9_data().pokedex.get(_normalize(display_name))
    if entry and "baseStats" in entry:
        bs = entry["baseStats"]
        return bs.get("atk", 0), bs.get("spa", 0)
    return None


# inferred-item id -> (display name, the evidence that confirmed it). The
# damage brackets model several items as one multiplier (Expert Belt /
# Booster in the Life Orb bracket), so the phrasing hedges honestly where
# the inference does.
_BELIEF_PROSE = {
    "choicescarf": ("Choice Scarf", "it outsped what any legal set allows"),
    "choiceband": ("Choice Band", "it hit past its max damage roll"),
    "choicespecs": ("Choice Specs", "it hit past its max damage roll"),
    "lifeorb": ("Life Orb", "its damage ran over the itemless ceiling"),
    "expertbelt": ("Expert Belt",
                   "its super-effective hits ran over the ceiling"),
}


def _belief_prose(name_display: str, item_id: str) -> str:
    """The set-reveal cue for a confirmed inferred item — display name +
    the evidence chain, phrased so the analyst can cite it and the gremlin
    can claim it."""
    label, why = _BELIEF_PROSE.get(
        item_id, ("a boosting item", "its output exceeded its modeled set"))
    return (f"set inference confirms {name_display}'s {label}: {why}")


def _preview_order(lead_idx: int, n: int) -> str:
    """'/team 312456'-style order string: chosen lead first, rest in order."""
    rest = [i for i in range(1, n + 1) if i != lead_idx + 1]
    return "/team " + "".join(str(x) for x in [lead_idx + 1] + rest)


def _lead_pool(matrix, epsilon: float = 0.08) -> list[int]:
    """Lead indices whose maximin (worst-case row value) is within epsilon
    of the best. A deterministic maximin lead is optimally predictable — we
    measured 30/30 identical leads per series, a free read for the opponent.
    Sampling among near-ties keeps the choice sound but unreadable."""
    row_mins = [min(row) for row in matrix]
    best = max(row_mins)
    return [i for i, v in enumerate(row_mins) if v >= best - epsilon]


def _select_choice(mappable, rng, sample: bool = True, keep_ratio: float = 0.75):
    """Pick from visit-ranked mappable candidates. Argmax is exploitable:
    foul-play keeps every move >= 75% of its best and samples — same rule
    here. `mappable` is a non-empty list of (result, order) tuples sorted
    by visits descending."""
    if not sample or len(mappable) == 1:
        return mappable[0]
    top = mappable[0][0].visits
    pool = [m for m in mappable if m[0].visits >= keep_ratio * top]
    weights = [m[0].visits for m in pool]
    return rng.choices(pool, weights=weights)[0]


class Gen9PokeEnginePlayer(Player):
    """poke-env Player: translate -> poke-engine MCTS -> order.

    The MCTS result is a visit-ranked list of engine choices; we walk it
    best-first and play the first choice that maps onto something the
    server actually offered (available_moves / available_switches /
    can_tera). That one loop handles tera legality, choice lock, Encore,
    trapping, and disabled moves without format-specific branches.
    """

    def __init__(self, search_ms: int = 1000, set_source: str = "gen9ou",
                 team_paste: str | None = None, preview_search_ms: int = 80,
                 set_samples: int = 2, data_tiers: bool = True,
                 stochastic: bool = True, adaptive: bool = False,
                 escalate_ms: int = 2000, flat_threshold: float = 0.55,
                 clock_floor_s: int = 40, escalate_bank_s: float = 90.0,
                 spend_frac: float = 0.25, escalate_max_ms: int = 15000,
                 base_frac: float = 0.02, base_max_ms: int = 2000,
                 escalate_min_turn: int = 20, escalate_min_gap: int = 8,
                 value_net_path: str | None = None, value_alpha: float = 0.5,
                 value_batch: int = 32, verbose: bool = True,
                 airi_bridge: AiriBridge | None = None,
                 airi_min_interval: float = 20.0,
                 airi_min_swing: float = 0.10,
                 airi_turn_pace: float = 0.0,
                 desk_log_path: str | None = DESK_LOG_DEFAULT, **kwargs):
        super().__init__(**kwargs)
        # Brier ledger: numeric desk reads per battle, flushed with the
        # outcome at game end (None path disables)
        self._desk_log_path = desk_log_path
        self._desk_reads: dict[str, list] = {}
        self._translator = Gen9Translator(set_source=set_source,
                                          use_data_tiers=data_tiers)
        self._stochastic = stochastic
        self._choice_rng = random.Random()
        # concurrent search across sampled worlds (real parallelism once the
        # mcts binding's GIL release is built; harmless serialization before)
        self._search_pool = ThreadPoolExecutor(max_workers=max(1, set_samples))
        # optional learned leaf eval: mcts_with_value blends the value net
        # with the static eval by alpha (0=static, 1=pure net). ~3.7x fewer
        # iterations even batched, so this only wins where static-eval
        # blindness (flat stall positions) wastes plain MCTS's iterations.
        self._value_net = None
        if value_net_path:
            self._value_net = pe.ValueNet(value_net_path)
            print(f"loaded value net {value_net_path} "
                  f"(alpha={value_alpha}, batch={value_batch})")
        self._value_alpha = value_alpha
        self._value_batch = value_batch
        # adaptive search: probe at search_ms, escalate to escalate_ms in
        # flat (undecided) positions. In stall games flat is the NORM, so
        # escalating every flat turn blows the clock; a per-game bank of
        # extra seconds (self-tracked — the local server doesn't emit the
        # |inactive| timer messages) caps total spend and concentrates it on
        # the earliest/most-contested positions. The parsed server clock is
        # an additional safety when present.
        self._adaptive = adaptive
        self._escalate_ms = escalate_ms
        self._flat_threshold = flat_threshold
        self._clock_floor_s = clock_floor_s
        self._escalate_bank_s = escalate_bank_s
        # server-clock policy: spend this fraction of (bank - floor) per
        # escalation, hard-capped per world; self-paces to server regen
        self._spend_frac = spend_frac
        self._escalate_max_ms = escalate_max_ms
        # budget-by-clock (base search scales with the parsed bank)
        self._base_frac = base_frac
        self._base_max_ms = base_max_ms
        self._bank_used_s = 0.0
        # WHERE the bank is spent matters more than how much: greedy early
        # spending drained it by ~turn 30 (all on low-leverage opening
        # flatness), leaving nothing for the turn-100+ attrition grind that
        # IS the horizon problem. min_turn skips the opening; min_gap spaces
        # escalations so the budget stretches across the late game.
        self._escalate_min_turn = escalate_min_turn
        self._escalate_min_gap = escalate_min_gap
        self._last_escalate_turn = -999
        self._search_ms = search_ms
        self._team_paste = team_paste
        self._preview_search_ms = preview_search_ms
        # >1: search that many sampled opponent-set worlds per turn and merge
        # (chaos sources only; monotype canonical sets have no sampler yet)
        self._set_samples = set_samples if set_source not in (None, "monotype") else 1
        self._verbose = verbose
        self._last_tag: str | None = None
        # AIRI commentary bridge (optional): battle beats become input:text
        # events for the character. Momentum = the top line's avg MCTS score
        # (side_one win estimate); swings are measured against the last SENT
        # event, so a slow bleed still crosses the significance gate
        # eventually instead of vanishing turn by turn.
        self._airi = airi_bridge
        self._airi_turn_pace = airi_turn_pace
        self._airi_tag: str | None = None
        self._airi_last_sent = 0.0
        # beat pipeline: protocol -> scanner -> typed events -> director ->
        # composed beat text. All routing/gating logic lives in
        # beat_director (pure, offline-drivable — the gold-set eval runs
        # the same classes against replays); this class only adapts battle
        # objects into TurnContext and ships Decision.text to AIRI.
        self._scanner = ProtocolScanner()
        self._director = Director(min_interval=airi_min_interval,
                                  min_swing=airi_min_swing,
                                  stats_fn=_species_stats)
        # species -> last-announced inferred item, so a belief the search
        # adopts (set_inference.confirmed) becomes a one-time "that's a
        # Scarf" reveal beat instead of firing every turn it holds
        self._announced_beliefs: dict = {}

    async def teampreview(self, battle):
        """6x6 MCTS maximin over (our lead, their predicted lead) pairings —
        a fixed lead hands the opponent a free, certain counter-pick every
        game. Falls back to paste order on any failure."""
        if self._team_paste is None:
            return "/team 123456"
        try:
            from monotype.lead_picker import pick_leads
            opp_species = [m.species for m in battle.opponent_team.values()]
            opp_paste = self._translator.predicted_preview_paste(opp_species)
            loop = asyncio.get_event_loop()
            lead_idx, _, matrix = await loop.run_in_executor(
                None, lambda: pick_leads(self._team_paste, opp_paste,
                                         search_ms=self._preview_search_ms))
            pool = _lead_pool(matrix)
            if self._stochastic and len(pool) > 1:
                lead_idx = self._choice_rng.choice(pool)
            order = _preview_order(lead_idx, 6)
            if self._verbose:
                print(f"  preview: leading slot {lead_idx + 1} "
                      f"(pool of {len(pool)}) -> {order}")
            try:
                lead = list(battle.team.values())[lead_idx].species
            except Exception:
                lead = None
            self._airi_new_battle(battle, lead=lead)
            return order
        except Exception as e:
            if self._verbose:
                print(f"  preview pick failed ({e!r}); using paste order")
            self._airi_new_battle(battle)
            return "/team 123456"

    def _airi_new_battle(self, battle, lead: str | None = None):
        """Emit the match-start event once per battle and reset the beat
        pipeline. Safe to call from every decision point: no-op after the
        first call for a given battle tag."""
        if self._airi is None or battle.battle_tag == self._airi_tag:
            return
        self._airi_tag = battle.battle_tag
        self._airi_last_sent = 0.0
        self._scanner.reset()
        self._announced_beliefs = {}
        try:
            ours = [_species_display(p.species)
                    for p in battle.team.values()]
            theirs = [_species_display(p.species)
                      for p in battle.opponent_team.values()]
            text = self._director.match_start(
                battle.opponent_username, ours, theirs,
                lead=_species_display(lead) if lead else None)
            self._airi.send(text)
            self._airi_last_sent = time.monotonic()
        except Exception:
            pass

    async def _handle_battle_message(self, split_messages):
        """Defer to the base handler, then scrape dramatic protocol events
        for commentary. Best-effort: a scan error must never disturb play."""
        result = await super()._handle_battle_message(split_messages)
        if self._airi is not None:
            try:
                role = None
                if split_messages and split_messages[0]:
                    tag = split_messages[0][0].lstrip(">").strip()
                    b = self._battles.get(tag)
                    role = b.player_role if b else None
                self._director.observe(
                    self._scanner.scan(split_messages, role))
            except Exception:
                pass
        return result

    def _airi_note_switch(self, mon):
        """Record a forced-switch replacement as a ride-along highlight so
        it's named in the next beat, without forcing extra late-game chatter
        (the single-switch and heuristic paths otherwise return silently)."""
        if self._airi is None:
            return
        try:
            self._director.note(f"we send {_species_display(mon.species)} in",
                                side="us")
        except Exception:
            pass

    def _emit_belief_deltas(self):
        """Diff the search's confirmed set inferences against what we've
        already announced; each newly-adopted inferred item becomes a
        high-priority set_reveal beat ('that's a Scarf'). The delta fires
        once — reactions track genuine information gain, not every turn the
        belief holds. species keys are normalized ids; display them."""
        obs = getattr(self._translator, "_obs", None)
        if obs is None:
            return
        try:
            for species, item in list(obs.confirmed.items()):
                if self._announced_beliefs.get(species) == item:
                    continue
                self._announced_beliefs[species] = item
                self._director.observe([Event(
                    "belief_delta",
                    _belief_prose(_species_display(species), item),
                    side="them", notable=True,
                    data={"species": species, "item": item})])
        except Exception:
            pass

    def _airi_turn_event(self, battle, ranked, desc: str):
        """Adapt the decision point into a TurnContext and let the director
        decide. All gating (5s floor, significance, swing-vs-last-SENT) and
        beat text assembly live in beat_director; this method only extracts
        primitives from the battle object — display names throughout — and
        ships Decision.text when there is one."""
        if self._airi is None:
            return
        try:
            self._airi_new_battle(battle)  # formats without team preview
            self._emit_belief_deltas()
            top = ranked[0]
            value = top.total_score / max(1, top.visits)
            me = battle.active_pokemon
            opp = battle.opponent_active_pokemon

            def hp(p):
                return round(100 * (p.current_hp_fraction or 0))

            # desc is engine-speak ("makeitrain", "switch gliscor"); the
            # character must see display names or it free-associates
            tera_choice = desc.endswith(" (tera)")
            base = desc[:-7] if tera_choice else desc
            if base.startswith("switch "):
                choice_text = f"We switch to {_species_display(base[7:])}."
            else:
                choice_text = (f"We go for {_move_display(base)}"
                               f"{' and Terastallize' if tera_choice else ''}.")

            ctx = TurnContext(
                turn=battle.turn,
                value=value,
                elapsed=time.monotonic() - self._airi_last_sent,
                me_name=_species_display(me.species) if me else None,
                me_hp=hp(me) if me else None,
                me_status=(me.status.name.lower()
                           if me and me.status else None),
                opp_name=_species_display(opp.species) if opp else None,
                opp_hp=hp(opp) if opp else None,
                opp_status=(opp.status.name.lower()
                            if opp and opp.status else None),
                ours_fainted=frozenset(
                    _species_display(p.species)
                    for p in battle.team.values() if p.fainted),
                theirs_fainted=frozenset(
                    _species_display(p.species)
                    for p in battle.opponent_team.values() if p.fainted),
                choice_text=choice_text,
            )
            decision = self._director.decide(ctx)
            if decision.text:
                # structured side-channel for the caster: director beats
                # (persona/priority/register/handoff) + numeric HUD. Real
                # AIRI reads only data.text and ignores these fields.
                self._airi.send(
                    decision.text,
                    beats=[asdict(b) for b in decision.beats],
                    hud={"turn": ctx.turn, "value": round(ctx.value, 4),
                         "us": ctx.me_name, "us_hp": ctx.me_hp,
                         "them": ctx.opp_name, "them_hp": ctx.opp_hp,
                         "us_alive": 6 - len(ctx.ours_fainted),
                         "them_alive": 6 - len(ctx.theirs_fainted)})
                self._airi_last_sent = time.monotonic()
        except Exception:
            pass

    def _flush_desk_log(self, battle):
        """One JSONL line per finished game: every numeric desk read the
        search produced plus the outcome. Brier and calibration tables are
        computed offline (showdown/brier_report.py) — nothing here may ever
        disturb play."""
        if self._desk_log_path is None:
            return
        try:
            reads = self._desk_reads.pop(battle.battle_tag, [])
            if not reads:
                return
            result = ("tie" if battle.won is None
                      else "win" if battle.won else "loss")
            line = {
                "ts": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                "battle_tag": battle.battle_tag,
                "opponent": battle.opponent_username,
                "search_ms": self._search_ms,
                "set_samples": self._set_samples,
                "reads": reads,
                "result": result,
                "outcome": {"win": 1.0, "loss": 0.0, "tie": 0.5}[result],
            }
            with open(self._desk_log_path, "a") as f:
                f.write(json.dumps(line) + "\n")
        except Exception:
            pass

    def _battle_finished_callback(self, battle):
        self._flush_desk_log(battle)
        if self._airi is None:
            return
        try:
            outcome = ("TIE" if battle.won is None
                       else "WIN" if battle.won else "LOSS")
            ours_left = sum(1 for p in battle.team.values() if not p.fainted)
            theirs_left = sum(1 for p in battle.opponent_team.values()
                              if not p.fainted)
            text, beat = self._director.match_end(
                outcome, ours_left, theirs_left, battle.opponent_username)
            self._airi.send(
                text, beats=[asdict(beat)],
                hud={"turn": None, "value": {"WIN": 1.0, "LOSS": 0.0,
                                             "TIE": 0.5}[outcome],
                     "us_alive": ours_left, "them_alive": theirs_left})
        except Exception:
            pass

    async def choose_move(self, battle):
        """Broadcast pacing wrapper: the actual decision is unchanged, but
        when commentating we HOLD the chosen move for airi_turn_pace seconds
        before sending it. The engine otherwise resolves a turn (~3-5s) far
        faster than the character generates a line (~8s), so every comment
        lands 2-3 turns late and the spectator client pins it to wherever
        the animation happens to be. Holding each move ~one generation's
        worth of time makes turn N's line ready as turn N animates."""
        order = await self._choose_move_impl(battle)
        if self._airi is not None and self._airi_turn_pace > 0 and \
                not battle.finished:
            await asyncio.sleep(self._airi_turn_pace)
        return order

    async def _choose_move_impl(self, battle):
        if battle.battle_tag != self._last_tag:
            self._last_tag = battle.battle_tag
            self._translator.new_battle()
            self._bank_used_s = 0.0  # fresh escalation bank per game
            self._last_escalate_turn = -999

        # forced switch (post-KO / pivot): search it like any other decision —
        # the translator flags side_one.force_switch and (for KOs) leaves the
        # fainted active at slot 0, so MCTS returns replacement choices
        if battle.force_switch:
            if not battle.available_switches:
                return self.choose_default_move()
            if len(battle.available_switches) == 1:
                self._airi_note_switch(battle.available_switches[0])
                return self.create_order(battle.available_switches[0])
            try:
                loop = asyncio.get_event_loop()
                results = await loop.run_in_executor(
                    None, self._search_samples, battle)
                order = self._map_choice(_merge_mcts_results(results), battle)
                if order is not None:
                    return order
            except Exception as e:
                if self._verbose:
                    print(f"  T{battle.turn} force-switch search failed "
                          f"({e!r}); using heuristic")
            best = max(battle.available_switches,
                       key=lambda p: _score_switch_in(p, battle))
            self._airi_note_switch(best)
            return self.create_order(best)

        try:
            loop = asyncio.get_event_loop()
            search = self._adaptive_search if self._adaptive else self._search_samples
            results = await loop.run_in_executor(None, search, battle)
        except Exception as e:
            if self._verbose:
                print(f"  T{battle.turn} translate/search failed ({e!r}); "
                      f"choosing randomly")
            return self.choose_random_move(battle)

        ranked = _merge_mcts_results(results)
        order = self._map_choice(ranked, battle)
        if order is not None:
            return order
        if self._verbose:
            print(f"  T{battle.turn} no MCTS choice mapped to a legal order; "
                  f"choosing randomly")
        return self.choose_random_move(battle)

    def _base_budget_ms(self, battle) -> int:
        """Budget-by-clock: derive the BASE per-world search budget from the
        parsed server bank. We probed at a fixed 300ms while the long-timer
        queue hands us a 1500s bank and the oracle bench prices 2000ms at 86%
        deep-search agreement vs 70% at 300ms — free strength. Policy: spend
        base_frac (2%) of surplus above the floor per turn, clamped to
        [150ms, base_max_ms]; worlds run in parallel so wall ~= this budget.
        Self-pacing: as the bank drains the budget decays smoothly toward
        survival speed; with no timer visible, the configured search_ms is
        used unchanged (keeps timerless local play and pinned benches exact).
        """
        try:
            uname = self.username
        except Exception:
            uname = None
        bank, _cap = _parse_clock(battle, uname)
        if bank is None:
            return self._search_ms
        dyn = int((bank - self._clock_floor_s) * self._base_frac * 1000)
        return max(150, min(self._base_max_ms, dyn))

    def _search_samples(self, battle, search_ms: int | None = None,
                        use_value: bool | None = None) -> list:
        """One MCTS per sampled opponent world (K = set_samples). With K=1,
        the deterministic top-set translation is used, as before. The LAST
        world is speed-pessimistic (fastest spreads, scarf when plausible):
        speed-floor inference only triggers after a scarfer already outsped
        something, so one world hedges against the sweep pre-emptively.

        Translation must be serial (the translator mutates per-call instance
        state: rng/pessimism/prefer_ps/archetype/obs), but the searches are
        independent and the mcts binding releases the GIL (py.detach), so
        they run concurrently across cores when set_samples > 1. Identical
        results either way — only wall time differs."""
        ms = search_ms if search_ms is not None else self._base_budget_ms(battle)
        states = []
        for i in range(self._set_samples):
            rng = random.Random() if self._set_samples > 1 else None
            pessimistic = self._set_samples > 1 and i == self._set_samples - 1
            # world 0: curated PS joint sets; later worlds: chaos sampling.
            # PS sets in every world collapsed diversity (series 10) — some
            # species have a single curated candidate, so all worlds shared
            # the same confident wrong set
            prefer_ps = i == 0
            states.append(self._translator.translate(
                battle, rng=rng, speed_pessimistic=pessimistic,
                prefer_ps=prefer_ps))
        # value net defaults to on-when-loaded, but callers override: the
        # adaptive probe forces it OFF (fast plain MCTS), and only the
        # escalated deep-think turns it ON — spend the learned eval where a
        # human would pause and think, not on every routine turn
        if use_value is None:
            use_value = self._value_net is not None
        if len(states) == 1:
            return [self._search_one(states[0], ms, use_value)]
        return list(self._search_pool.map(
            lambda st: self._search_one(st, ms, use_value), states))

    def _search_one(self, state, ms: int, use_value: bool):
        """One world's search: value-net-guided leaf eval when requested and
        a net is loaded, else plain MCTS."""
        if not use_value or self._value_net is None:
            return pe.monte_carlo_tree_search(state, ms)
        return pe.monte_carlo_tree_search_with_value(
            state, self._value_net, ms,
            alpha=self._value_alpha, batch_size=self._value_batch)

    def _adaptive_search(self, battle) -> list:
        """Staged search that reinvests the timer bank where it matters.

        Fixed 300ms/decision leaves ~90% of the server clock unused in the
        long grindy games (stall/fat), which are exactly the positions where
        a flat static eval makes every move look equal and deeper search is
        most likely to break the tie. So: probe cheap; if the merged visit
        distribution is DECISIVE (one move dominates — a resolved tactic),
        keep it. If it's FLAT and the clock is healthy, re-search deep and
        use that instead. Sharp positions self-select out (they produce
        peaked distributions at the probe budget); quiet/attrition positions
        produce flat ones and get the extra thinking."""
        # probe is always fast plain MCTS — its 164k iters resolve tactics
        # cheaply; the value net's 3.7x throughput cost is reserved for the
        # escalated deep-think below
        base_ms = self._base_budget_ms(battle)
        if self._verbose and battle.turn <= 1:
            print(f"  T1 clock base budget {base_ms}ms/world", flush=True)
        probe = self._search_samples(battle, base_ms, use_value=False)
        merged = _merge_mcts_results(probe)
        total = sum(m.visits for m in merged) or 1
        top_share = merged[0].visits / total if merged else 1.0
        if top_share >= self._flat_threshold or len(merged) <= 1:
            return probe  # decisive — a resolved tactic, don't spend more

        # spend the bank on the late grind, not the opening: skip early turns
        # and space escalations so the budget reaches the turn-100+ attrition
        # that is the actual horizon problem
        if battle.turn < self._escalate_min_turn:
            return probe
        if battle.turn - self._last_escalate_turn < self._escalate_min_gap:
            return probe

        # budget policy. With the server clock visible (it re-sends an
        # authoritative bank reading every request while the timer runs),
        # spend a FRACTION OF SURPLUS above a floor each escalation: the bank
        # then self-paces to the server's regeneration (+addPerTurn/turn)
        # without ever modeling it — works unchanged on quick and extended
        # (LLM) timer variants. Worlds search in parallel, so wall-time per
        # escalation ~= the per-world budget.
        try:
            uname = self.username
        except Exception:
            uname = None
        bank, cap = _parse_clock(battle, uname)
        if bank is not None:
            surplus = bank - self._clock_floor_s
            if surplus <= 2:
                return probe
            allowed_s = surplus * self._spend_frac
            if cap:
                # per-turn cap bounds this whole turn incl. the probe already
                # spent; leave a margin for translation/overhead
                allowed_s = min(allowed_s,
                                cap - base_ms / 1000 - 5)
            budget = min(self._escalate_max_ms, int(round(allowed_s * 1000)))
        else:
            # no timer messages (local play with timer off): fall back to the
            # fixed self-imposed per-game bank
            if self._bank_used_s >= self._escalate_bank_s:
                return probe
            budget = self._escalate_ms
        if budget <= base_ms:
            return probe
        if self._verbose:
            print(f"  T{battle.turn} flat (top {top_share:.0%}), escalating "
                  f"{base_ms}->{budget}ms/world "
                  f"(bank {self._bank_used_s:.0f}/{self._escalate_bank_s:.0f}s)")
        if self._airi is not None and \
                time.monotonic() - self._airi_last_sent > 5.0:
            self._airi.send(
                f"[BATTLE T{battle.turn}] Critical position: no line stands "
                "out from the rest here. We're taking extra time to think "
                "this one through.")
        # escalated deep-think: this is the "human pauses to think" moment —
        # spend the timer bank AND the value net's learned eval here, where
        # the static eval is blindest (flat positions) and depth is wasted
        t0 = time.monotonic()
        deep = self._search_samples(battle, budget, use_value=True)
        self._bank_used_s += time.monotonic() - t0
        self._last_escalate_turn = battle.turn
        return deep

    def _map_choice(self, ranked, battle):
        """Collect every legal engine choice, then pick: probabilistic among
        near-ties (>=75% of the top's visits) when stochastic, else argmax."""
        moves_by_id = {_normalize(m.id): m for m in battle.available_moves}
        switches_by_id = {_normalize(p.species): p
                          for p in battle.available_switches}
        mappable = []
        for r in ranked:
            choice = r.move_choice
            if choice.startswith("switch "):
                target = switches_by_id.get(_normalize(choice[7:]))
                if target is not None:
                    mappable.append((r, self.create_order(target),
                                     f"switch {target.species}", None))
                continue
            tera = choice.endswith("-tera")
            move_id = _normalize(choice[:-5] if tera else choice)
            move = moves_by_id.get(move_id)
            if move is None:
                continue
            if tera and not battle.can_tera:
                continue  # engine explored tera we don't have; try next
            mappable.append((r, self.create_order(move, terastallize=tera),
                             move_id + (" (tera)" if tera else ""), move_id))
        if not mappable:
            return None
        # drop guaranteed no-op moves the flat-eval search can't distinguish:
        # already-maxed hazard re-sets, and status moves that can't land on the
        # current target (immune / already statused / behind Sub). Keep them
        # only if nothing else is legal.
        def _noop(m):
            return _is_noop_hazard(m[3], battle) or _is_noop_status(m[3], battle)
        useful = [m for m in mappable if not _noop(m)]
        mappable = useful or mappable
        mappable = [m[:3] for m in mappable]
        chosen_result, (order, desc) = _select_choice(
            [(m[0], (m[1], m[2])) for m in mappable],
            self._choice_rng, sample=self._stochastic)
        self._log_choice(battle, chosen_result, desc)
        if self._desk_log_path is not None:
            try:
                top = ranked[0]
                self._desk_reads.setdefault(battle.battle_tag, []).append(
                    (battle.turn,
                     round(top.total_score / max(1, top.visits), 4)))
            except Exception:
                pass
        self._airi_turn_event(battle, ranked, desc)
        return order

    def _log_choice(self, battle, r, desc: str):
        if self._verbose:
            print(f"  T{battle.turn}: {desc} "
                  f"(visits={r.visits}, "
                  f"avg_score={r.total_score / max(1, r.visits):.3f})")


async def main():
    parser = argparse.ArgumentParser(description="gen9 poke-engine live player")
    parser.add_argument("--local", action="store_true",
                        help="shorthand for --server local")
    parser.add_argument("--server", choices=list(SERVERS), default="local")
    parser.add_argument("--username", required=True)
    parser.add_argument("--password", default=os.environ.get("PS_PASSWORD"),
                        help="account password (or set PS_PASSWORD env — "
                             "preferred for units/wrappers so it stays out "
                             "of process listings)")
    parser.add_argument("--format", dest="fmt", default="gen9ou")
    parser.add_argument("--team", default=None,
                        help="path to a Showdown paste file (required for "
                             "team formats)")
    parser.add_argument("--search-ms", type=int, default=1000)
    parser.add_argument("--set-source", default=None,
                        help="opponent set inference source; defaults to "
                             "'monotype' for gen9monotype else the format name")
    parser.add_argument("--mode", choices=["accept", "challenge", "ladder"],
                        default="accept")
    parser.add_argument("--user-to-challenge", default=None)
    parser.add_argument("--n-games", type=int, default=1)
    parser.add_argument("--set-samples", type=int, default=2,
                        help="sampled opponent-set worlds searched per turn "
                             "(1 = deterministic top sets)")
    parser.add_argument("--data-tiers", choices=["on", "off"], default="on",
                        help="PS-curated + replay-observed set tiers; 'off' "
                             "reproduces the pure chaos config (ab9 baseline)")
    parser.add_argument("--stochastic", choices=["on", "off"], default="on",
                        help="sample moves among near-ties (75%% rule) and "
                             "leads among maximin near-ties; 'off' = argmax")
    parser.add_argument("--adaptive", choices=["on", "off"], default="off",
                        help="staged search: probe at --search-ms, escalate to "
                             "--escalate-ms in flat positions when clock allows")
    parser.add_argument("--escalate-ms", type=int, default=2000,
                        help="deep-search budget per world in flat positions")
    parser.add_argument("--base-frac", type=float, default=0.02,
                        help="budget-by-clock: fraction of bank surplus used "
                             "as the BASE per-world search budget each turn")
    parser.add_argument("--base-max-ms", type=int, default=2000,
                        help="cap on the clock-derived base budget (oracle "
                             "bench: 2000ms = 86%% deep-search agreement)")
    parser.add_argument("--spend-frac", type=float, default=0.25,
                        help="fraction of (bank - floor) spent per escalation "
                             "when the server clock is visible")
    parser.add_argument("--escalate-max-ms", type=int, default=15000,
                        help="hard cap on per-world escalated search time")
    parser.add_argument("--escalate-bank-s", type=float, default=90.0,
                        help="per-game budget of extra seconds for escalation")
    parser.add_argument("--escalate-min-turn", type=int, default=20,
                        help="don't escalate before this turn (skip the opening)")
    parser.add_argument("--escalate-min-gap", type=int, default=8,
                        help="min turns between escalations (spread the bank "
                             "across the late game instead of draining early)")
    parser.add_argument("--value-net", type=str, default=None,
                        help="path to a ValueNet ONNX for leaf eval "
                             "(mcts_with_value); omit for pure static eval")
    parser.add_argument("--value-alpha", type=float, default=0.5,
                        help="value-net blend weight (0=static, 1=pure net)")
    parser.add_argument("--value-batch", type=int, default=32,
                        help="leaf-eval batch size (throughput vs quality)")
    parser.add_argument("--log-level", type=int, default=30,
                        help="poke-env logger level (10=DEBUG shows protocol)")
    parser.add_argument("--airi", action="store_true",
                        help="commentate: forward battle beats + momentum to "
                             "a running AIRI desktop app as input:text events")
    parser.add_argument("--airi-url", default=AIRI_DEFAULT_URL,
                        help="AIRI server WebSocket endpoint")
    parser.add_argument("--airi-min-interval", type=float, default=20.0,
                        help="max seconds between routine commentary beats")
    parser.add_argument("--airi-min-swing", type=float, default=0.10,
                        help="win-estimate swing (0-1) that forces a beat")
    parser.add_argument("--airi-turn-pace", type=float, default=0.0,
                        help="hold each move this many seconds before sending "
                             "(broadcast pacing: keeps commentary in sync with "
                             "the battle animation; ~8 suits Gemma's latency)")
    parser.add_argument("--desk-log", default=DESK_LOG_DEFAULT,
                        help="JSONL file for desk-read calibration logging "
                             "(one line per game: reads + outcome; "
                             "'off' disables)")
    args = parser.parse_args()

    server = LOCAL_SERVER if args.local else SERVERS[args.server]
    team = Path(args.team).read_text() if args.team else None
    set_source = args.set_source or (
        "monotype" if args.fmt == "gen9monotype" else args.fmt)

    bridge = AiriBridge(url=args.airi_url) if args.airi else None

    player = Gen9PokeEnginePlayer(
        search_ms=args.search_ms,
        set_source=set_source,
        team_paste=team,
        set_samples=args.set_samples,
        data_tiers=args.data_tiers == "on",
        stochastic=args.stochastic == "on",
        adaptive=args.adaptive == "on",
        escalate_ms=args.escalate_ms,
        escalate_bank_s=args.escalate_bank_s,
        spend_frac=args.spend_frac,
        escalate_max_ms=args.escalate_max_ms,
        base_frac=args.base_frac,
        base_max_ms=args.base_max_ms,
        escalate_min_turn=args.escalate_min_turn,
        escalate_min_gap=args.escalate_min_gap,
        value_net_path=args.value_net,
        value_alpha=args.value_alpha,
        value_batch=args.value_batch,
        airi_bridge=bridge,
        airi_min_interval=args.airi_min_interval,
        airi_min_swing=args.airi_min_swing,
        airi_turn_pace=args.airi_turn_pace,
        desk_log_path=None if args.desk_log == "off" else args.desk_log,
        account_configuration=AccountConfiguration(args.username, args.password),
        server_configuration=server,
        battle_format=args.fmt,
        team=team,
        max_concurrent_battles=1,
        # NOTE: accept_open_team_sheet=True makes poke-env defer the team
        # preview reply until a showteam/rejection message that formats
        # without the OTS rule never send -> guaranteed timer loss
        log_level=args.log_level,
    )

    if bridge is not None:
        await bridge.start()

    if args.mode == "challenge":
        if not args.user_to_challenge:
            parser.error("--mode challenge requires --user-to-challenge")
        await player.send_challenges(args.user_to_challenge,
                                     n_challenges=args.n_games)
    elif args.mode == "ladder":
        await player.ladder(args.n_games)
    else:
        await player.accept_challenges(None, args.n_games)

    print(f"finished: {player.n_won_battles}W / "
          f"{player.n_lost_battles}L / {player.n_tied_battles}T")
    if bridge is not None:
        await asyncio.sleep(2.0)  # let the [RESULT] event flush
        await bridge.close()


if __name__ == "__main__":
    asyncio.run(main())
