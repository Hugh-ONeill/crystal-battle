# Gen 9 monotype state translator: poke-env Battle -> poke-engine State
#
# The gen2 translator (poke_engine_player.py) predates poke-engine exposing
# side conditions / boosts / volatiles as constructor args, so it dropped all
# mid-game side state. The current bindings accept everything, so this builds
# full-fidelity states: hazards, screens, boosts, volatiles, weather, terrain,
# trick room.
#
# Opponent inference: monotype's shared type plus the per-type Smogon moveset
# stats (monotype/canonical_sets.py) pin down likely sets far better than the
# gen2 "base stats + leftovers" guess. Revealed information always overrides
# the canonical fill. Unrevealed team slots stay as fainted dummies — filling
# them with usage predictions misled MCTS more than empty slots did (the
# search wastes visits switching into imagined threats; see the gen2
# translator's note).
#
# Known approximations (all bounded, revisit if traces show they matter):
#   - screen/weather/terrain turns-remaining are inferred from start turn +
#     default duration; extender items (Light Clay / Heat Rock / Icy Rock,
#     all monotype-legal) are assumed only when revealed on that side
#   - Rest sleep is translated as regular sleep (rest_turns unknown to
#     poke-env; affects wake timing in search only)
#   - substitute health = maxhp//4 (poke-env tracks presence, not HP)
#   - volatile durations (taunt/encore/confusion) are median estimates

from __future__ import annotations

import os
import random
import sys
import zlib
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import poke_engine as pe

from showdown.name_mapping import _normalize


def _fainted_mon(mon) -> "pe.Pokemon":
    """A fainted placeholder that REMEMBERS whether it terastallized.

    `pe.Pokemon.create_fainted()` is a blank dummy with terastallized=False,
    and the engine's `can_use_tera()` returns true unless SOME mon on the side
    carries the flag. So the moment a tera'd mon fainted, the engine believed
    tera was available again and offered `<move>-tera` twins at the root; MCTS
    spent visits on them and _map_choice then discarded every one as illegal
    (`if tera and not battle.can_tera: continue`). Pure wasted search, worst in
    the mid-late game where the budgets are largest.

    Same root cause as the endgame-solver gate bug, which was patched at the
    gate (reading battle.is_terastallized) rather than here at the source, so
    the SEARCH kept seeing a phantom tera. Applies to both sides: we also
    over-modelled the opponent's remaining tera.
    """
    if not getattr(mon, "is_terastallized", False):
        return pe.Pokemon.create_fainted()
    dummy = pe.Pokemon.create_fainted()
    return pe.Pokemon(id=dummy.id, hp=0, maxhp=max(1, dummy.maxhp),
                      terastallized=True)


def _slot_key(species: str) -> str:
    """Species key for canonical slot assignment. Team preview marks an
    undisclosed forme with a '-*' suffix (Zamazenta-*), which must resolve to
    the same slot as the revealed forme rather than claiming a seventh."""
    return _normalize(str(species).strip().removesuffix("-*"))
from showdown.local_battle import (
    parse_showdown_team, _calc_stat_modern, _NATURE_TABLE,
)
from monotype.chaos_priors import _detect_side_type


# ============================================================
# STATIC MAPPINGS
# ============================================================

# poke-env Weather enum name -> poke-engine Weather
_WEATHER_MAP = {
    "SUNNYDAY": pe.Weather.SUN,
    "DESOLATELAND": pe.Weather.HARSH_SUN,
    "RAINDANCE": pe.Weather.RAIN,
    "PRIMORDIALSEA": pe.Weather.HEAVY_RAIN,
    "SANDSTORM": pe.Weather.SAND,
    "HAIL": pe.Weather.HAIL,
    "SNOW": pe.Weather.SNOW,
    "SNOWSCAPE": pe.Weather.SNOW,
}

# weather -> item that extends it to 8 turns (Damp/Smooth Rock are banned in
# monotype but checking for them is harmless)
_WEATHER_ROCK = {
    pe.Weather.SUN: "heatrock",
    pe.Weather.RAIN: "damprock",
    pe.Weather.SAND: "smoothrock",
    pe.Weather.SNOW: "icyrock",
    pe.Weather.HAIL: "icyrock",
}

# poke-env Field enum name -> poke-engine Terrain
_TERRAIN_MAP = {
    "ELECTRIC_TERRAIN": pe.Terrain.ELECTRIC,
    "GRASSY_TERRAIN": pe.Terrain.GRASSY,
    "MISTY_TERRAIN": pe.Terrain.MISTY,
    "PSYCHIC_TERRAIN": pe.Terrain.PSYCHIC,
}

# poke-env Effect names (underscores stripped) that poke-engine's
# PokemonVolatileStatus understands. Unknown names default to NONE on the
# Rust side, so this allowlist keeps the set clean rather than guarding
# against crashes.
_VOLATILE_ALLOW = frozenset({
    # DISABLE is deliberately absent: poke-env doesn't expose which move got
    # disabled, and the bare volatile tells the engine nothing useful
    "AQUARING", "ATTRACT", "CHARGE", "CONFUSION", "CURSE", "DESTINYBOND",
    "EMBARGO", "ENCORE", "FLASHFIRE", "FOCUSENERGY", "GASTROACID",
    "GLAIVERUSH", "HEALBLOCK", "IMPRISON", "INGRAIN", "LEECHSEED",
    "MAGNETRISE", "MINIMIZE", "MUSTRECHARGE", "NIGHTMARE", "NORETREAT",
    "OCTOLOCK", "PARTIALLYTRAPPED", "PERISH1", "PERISH2", "PERISH3",
    "PERISH4", "POWERTRICK", "SALTCURE", "SLOWSTART", "SMACKDOWN",
    "STOCKPILE", "SUBSTITUTE", "SYRUPBOMB", "TARSHOT", "TAUNT",
    "TELEKINESIS", "YAWN",
})

_SLEEP_STATUS = "slp"

# items/abilities that lock the user into its last move until it switches
_CHOICE_LOCKERS = frozenset({"choiceband", "choicespecs", "choicescarf"})
_LOCKING_ABILITY = "gorillatactics"


def _clamp_turns(remaining: int) -> int:
    """A condition observed active has at least 1 turn left."""
    return max(1, remaining)


def _base_format(fmt: str | None) -> str | None:
    """Strip PokeAgent timer-variant suffixes so data lookups resolve.

    The ladder runs gen9ou under multiple timer queues (gen9oulongtimer /
    gen9oushorttimer for the LLM-viable long clock); these are mechanically
    identical to the base tier and share its data files (<base>_chaos.json,
    ps_sets, replay archetypes). Without this, set_source="gen9oulongtimer"
    sends the chaos fallback looking for a nonexistent gen9oulongtimer_chaos
    .json — it raised FileNotFoundError mid-game and dropped us to random
    moves the first time an opponent species missed the ps/replay tiers.
    Sentinels ("monotype", None) pass through untouched.
    """
    if not fmt:
        return fmt
    for suffix in ("longtimer", "shorttimer"):
        if fmt.endswith(suffix) and fmt != suffix:
            return fmt[: -len(suffix)]
    return fmt


# ============================================================
# TRANSLATOR
# ============================================================

# CB_PS_TERA=chaos: world 0 keeps the curated PS moves/item/spread but takes
# its UNREVEALED tera from a chaos-marginal draw instead of the PS set's
# single type. belief_calibration.py (2026-07-31, 239 games): the PS tera was
# the worst calibration cell measured — avg probability on the revealed tera
# 17% vs chaos 24%, and BLINDSIDED (revealed tera priced <=5%) 29.8% of the
# time vs chaos 10.2%. Moves/item stay curated because there PS wins. A/B'd
# per process like CB_MERGE_RAW; default unset = current behaviour.
_PS_TERA_SOURCE = os.environ.get("CB_PS_TERA", "")

# CB_TEAM_COND_OVERLAP=5|4: partial-core backoff for the replay archetype
# tier. Default 0 = OFF = today's exact-6-roster behaviour. See
# ReplaySetsIndex.partial_team_match for the coverage and divergence
# measurements; gated because world-composition changes are 0-for-3 and this
# needs its own paired A/B.
_TEAM_COND_OVERLAP = int(os.environ.get("CB_TEAM_COND_OVERLAP") or 0)


def _stable_tera_draw(dist: dict, battle_tag: str, species: str) -> str | None:
    """One probability-weighted draw from a tera marginal, seeded by
    (battle, species): stable for the whole battle — world 0 stays the
    reproducible world, no per-turn flapping — while varying across battles,
    which is what makes the hedge calibrated in aggregate. crc32, not
    hash(): hash() is salted per process and would silently decohere the
    zygote's forked workers from a restarted one."""
    if not dist:
        return None
    total = sum(dist.values())
    if total <= 0:
        return None
    seed = zlib.crc32(f"{battle_tag}:{species}".encode())
    r = random.Random(seed).random() * total
    acc = 0.0
    for t, w in sorted(dist.items()):     # sorted: dict order is not contract
        acc += w
        if r <= acc:
            return t
    return max(dist, key=lambda k: dist[k])


class Gen9Translator:
    """Translates poke-env Battle objects to full-fidelity gen9 States.

    One instance per battle (or call new_battle() between battles): the
    detected opponent monotype is cached after enough mons are revealed.

    `set_source` picks how unrevealed opponent set details are inferred:
      - "monotype": per-type Smogon canonical sets (monotype/canonical_sets),
        conditioned on the opponent's detected monotype
      - any other format string (e.g. "gen9ou"): Smogon chaos stats for that
        format (showdown/<format>_chaos.json via ChaosStats)
      - None: no inference; unrevealed details fall back to pokedex defaults
    """

    _canon_cache: dict[int, dict] = {}  # elo -> {type: {norm_species: mon dict}}
    _chaos_cache: dict[str, object] = {}  # format -> ChaosStats
    _pokedex = None
    _type_chart = None  # gen9 effectiveness chart (uppercase-type keyed)

    def __init__(self, elo_bucket: int = 1500, set_source: str | None = "monotype",
                 use_data_tiers: bool = True,
                 team_archive_index: str | None = None):
        self._elo = elo_bucket
        self._archive = None
        self._archive_team = None
        if team_archive_index:
            from showdown.team_archive import TeamArchive
            self._archive = TeamArchive(team_archive_index)
        # timer-variant ladder formats share their base tier's data files
        self._set_source = _base_format(set_source)
        # gates the PS-curated and replay-observed set tiers; off reproduces
        # the pure chaos-sampling config (the ab9 baseline) exactly
        self._use_data_tiers = use_data_tiers
        self._opp_type: str | None = None
        self._obs = None  # per-battle observational set refinement
        self._book: dict | None = None   # scouting profile for THIS opponent
        self._book_min_obs = 2
        # per-battle canonical slot ordering (see _slot_order)
        self._slots: dict[str, list[str]] = {}

    def new_battle(self):
        self._opp_type = None
        self._obs = None
        self._slots = {}
        self._archive_team = None

    def set_opponent_book(self, profile: dict | None, min_obs: int = 2):
        """Scouting profile for the CURRENT opponent (showdown/scouting_book
        .py output). Direct evidence from games against this exact username
        outranks every corpus statistic, so it becomes the top set tier —
        but only for fields we can actually observe (moves/item/ability/
        tera). EVs/nature stay with the statistical baseline: we never see
        an opponent's spread, and inventing one from a 2-game sample would
        be worse than the usage-stat prior."""
        self._book = profile or None
        self._book_min_obs = min_obs

    def _apply_book(self, s: dict | None, booked: dict | None) -> dict | None:
        """Overlay this opponent's observed fields onto a composed set.
        Applied to EVERY tier's output — the PS-curated tier returns early,
        so overlaying only at the end silently skipped the book whenever a
        curated set existed (which is most of the meta's common mons)."""
        if not s or not booked:
            return s
        if booked.get("moves"):
            s["moves"] = booked["moves"]
        if booked.get("item"):
            s["item"] = booked["item"]
        if booked.get("ability"):
            s["ability"] = booked["ability"]
        if booked.get("tera"):
            s["tera_type"] = booked["tera"]
        return s

    def _book_set(self, species: str, known_moves: tuple[str, ...] = ()):
        """Observed (moves, item, ability, tera) for species from this
        opponent, gated on having seen it enough times to be a pattern
        rather than a one-off. Returns None when unknown/too thin."""
        if not self._book:
            return None
        sets = (self._book.get("sets") or {})
        # book keys are display species ("Great Tusk"); translator asks with
        # normalized ids ("greattusk")
        entry = sets.get(species)
        if entry is None:
            for k, v in sets.items():
                if _normalize(k) == species:
                    entry = v
                    break
        if not entry:
            return None
        mv = entry.get("moves") or {}
        seen = max(mv.values()) if mv else 0
        if seen < self._book_min_obs:
            return None
        ranked = [m for m, _ in sorted(mv.items(), key=lambda kv: -kv[1])]
        # revealed moves this game always survive; book fills the rest
        moves = list(known_moves) + [m for m in ranked if m not in known_moves]
        top = lambda d: (max(d.items(), key=lambda kv: kv[1])[0]
                         if d else None)
        return {
            "moves": moves[:4],
            "item": top(entry.get("items") or {}),
            "ability": top(entry.get("abilities") or {}),
            "tera": top(entry.get("tera") or {}),
        }

    # ---- lazy shared data ----

    @classmethod
    def _dex(cls):
        if cls._pokedex is None:
            from poke_env.data.gen_data import GenData
            cls._pokedex = GenData.from_gen(9).pokedex
        return cls._pokedex

    @classmethod
    def _typechart(cls):
        if cls._type_chart is None:
            from poke_env.data.gen_data import GenData
            cls._type_chart = GenData.from_gen(9).type_chart
        return cls._type_chart

    def _worst_case_defensive_tera(self, battle) -> str | None:
        """The Tera type that most blunts our ACTIVE's offense — the defensive
        Tera a pessimistic opponent burns to wall us. Returns a lowercase
        engine type string, or None if our threat can't be read.

        Threat = the types of our active's damaging moves (same-type STAB
        already lives there); falls back to our active's own types when no
        damaging move is visible. We pick the pure defensive type T that
        MINIMIZES the worst (max) effectiveness of those attack types into T,
        so coverage is accounted for, not just the primary STAB; ties break
        toward the lower total exposure."""
        me = getattr(battle, "active_pokemon", None)
        if me is None:
            return None
        from poke_env.battle.pokemon_type import PokemonType
        atk_types = {mv.type for mv in me.moves.values()
                     if mv.type and mv.base_power and mv.base_power > 0}
        if not atk_types:
            atk_types = {t for t in (me.type_1, me.type_2) if t}
        if not atk_types:
            return None
        tc = self._typechart()
        skip = {PokemonType.THREE_QUESTION_MARKS, PokemonType.STELLAR}
        best_type = best_worst = best_sum = None
        for cand in PokemonType:
            if cand in skip:
                continue
            effs = [a.damage_multiplier(cand, type_chart=tc) for a in atk_types]
            worst, total = max(effs), sum(effs)
            if (best_worst is None or worst < best_worst
                    or (worst == best_worst and total < best_sum)):
                best_type, best_worst, best_sum = cand, worst, total
        return best_type.name.lower() if best_type else None

    def _canonical(self) -> dict[str, dict[str, dict]]:
        """{type: {normalized_species: parsed canonical mon dict}}"""
        cached = Gen9Translator._canon_cache.get(self._elo)
        if cached is not None:
            return cached
        from monotype.canonical_sets import build_canonical_sets
        index: dict[str, dict[str, dict]] = {}
        for mono_type, by_species in build_canonical_sets(self._elo).items():
            index[mono_type] = {}
            for species, paste in by_species.items():
                mons = parse_showdown_team(paste)
                if mons:
                    index[mono_type][_normalize(species)] = mons[0]
        Gen9Translator._canon_cache[self._elo] = index
        return index

    def _chaos(self):
        fmt = self._set_source
        cached = Gen9Translator._chaos_cache.get(fmt)
        if cached is None:
            from showdown.chaos_stats import ChaosStats
            cached = ChaosStats(format=fmt)
            Gen9Translator._chaos_cache[fmt] = cached
        return cached

    def _ps_index(self):
        if not self._use_data_tiers or self._set_source in (None, "monotype"):
            return None
        from showdown.ps_sets import get_index
        return get_index(self._set_source)

    def _replay_index(self):
        if not self._use_data_tiers or self._set_source in (None, "monotype"):
            return None
        from showdown.replay_sets import get_index
        return get_index(self._set_source)

    def _resolve_archetype(self, battle):
        """Match the opponent's previewed roster against the replay team
        archetype index (ladder players copy whole teams; a match predicts
        moves/tera for revealed AND unrevealed mons)."""
        self._archetype = None
        idx = self._replay_index()
        if idx is None:
            return
        species = [m.species for m in
                   getattr(battle, "teampreview_opponent_team", None) or []]
        if len(species) != 6 and len(battle.opponent_team) == 6:
            species = [m.species for m in battle.opponent_team.values()]
        if len(species) == 6:
            match = idx.team_match(species)
            # confidence gate: a roster seen twice could be one player's
            # home-brew; three-plus sightings means a real archetype
            if match is not None and match.get("count", 0) >= 3:
                self._archetype = match
            elif _TEAM_COND_OVERLAP:
                # PARTIAL-CORE BACKOFF (default OFF, CB_TEAM_COND_OVERLAP=5|4).
                # The exact-roster match fires on 70.5% of the rosters we
                # actually face; >=5 shared reaches 87.1% and >=4 reaches
                # 98.4% (measured 2026-08-02 over 730 booked games). Held
                # behind a flag because every world-composition change so far
                # is 0-for-3 on winrate and this needs its own paired A/B.
                self._archetype = idx.partial_team_match(
                    species, min_overlap=_TEAM_COND_OVERLAP)

    def _resolve_archive(self, battle):
        """Full-set team archive tier (team_archive.py): match the previewed
        roster against the metamon corpus and draw ONE correlated candidate
        team, filtered by everything revealed so far. Joint sets for
        revealed AND unrevealed mons — items/EVs included, which the replay
        archetype tier structurally cannot supply.

        WORLD 0 ONLY (v2): the v1 wiring fired in every world and was gated
        OUT at accept-h0 — both sampled worlds drew correlated archive
        candidates, so plausible-but-wrong details were wrong the SAME way
        in both worlds (unmergeable), and the speed-pessimistic hedge world
        silently lost its anti-scarf insurance. Keying on _prefer_ps makes
        the archive replace exactly the curated-PS world and nothing else —
        the series-10 one-joint-world-plus-one-chaos-world rule, applied at
        this tier."""
        self._archive_team = None
        arch = getattr(self, "_archive", None)
        if arch is None or not getattr(self, "_prefer_ps", True):
            return
        species = [m.species for m in
                   getattr(battle, "teampreview_opponent_team", None) or []]
        if len(species) != 6 and len(battle.opponent_team) == 6:
            species = [m.species for m in battle.opponent_team.values()]
        if len(species) != 6:
            return
        revealed = {}
        for mon in battle.opponent_team.values():
            obs = {"moves": {_normalize(m.id) for m in mon.moves.values()}}
            item = getattr(mon, "item", None)
            if item and item not in ("unknown_item",):
                obs["item"] = _normalize(item)
            ability = getattr(mon, "ability", None)
            if ability:
                obs["ability"] = _normalize(ability)
            revealed[_normalize(mon.species)] = obs
        try:
            # book-weighted selection (2026-07-30): the archive knows this
            # opponent (95% best-candidate accuracy) but a blind draw threw
            # that away at 69% — see TeamArchive._book_score. Falls back to
            # the blind draw for an opponent we have no history on.
            book_sets = (self._book or {}).get("sets") if self._book else None
            # CB_ARCHIVE_SELECT=blind reproduces the pre-2026-07-31 draw (first
            # consistent candidate) so the selection component can be ABLATED:
            # the +9.2pp A/B bundled archive tier + selection + gate, and only
            # selection obviously transfers to a booked ladder opponent.
            if os.environ.get("CB_ARCHIVE_SELECT", "book").strip().lower() == "blind":
                self._archive_team = arch.sample(species, revealed, rng=self._rng)
            else:
                self._archive_team = arch.sample_booked(
                    species, revealed, book_sets, rng=self._rng)
        except Exception:
            self._archive_team = None   # advisory tier; never fail a translation

    def _archive_covers(self, known_moves: tuple[str, ...]) -> bool:
        """Which mons the archive tier is allowed to answer for.

        `unrevealed` (default): only mons that have shown NOTHING yet. That is
        where a correlated roster match is pure gain — we would otherwise be
        guessing from marginals — and it is the shape the 2026-07-23 shelving
        itself proposed after v1's whole-team override lost three gates. Once a
        mon has revealed a move, the observation-filtered PS/chaos tiers can
        answer it without betting the rest of its set on one archive variant,
        which caps the correlated-wrong exposure that killed those gates
        (top-1 selection is ~83% right on revealed moves, so ~1 in 6 wrong,
        and wrong across six mons at once when the pick is bad).

        `all` reproduces v1 for A/B purposes. Env: CB_ARCHIVE_MODE.
        """
        mode = getattr(self, "_archive_mode", None)
        if mode is None:
            mode = os.environ.get("CB_ARCHIVE_MODE", "unrevealed").strip().lower()
            self._archive_mode = mode
        return True if mode == "all" else not known_moves

    def _opp_set(self, species: str, known_moves: tuple[str, ...] = (),
                 known_item: str | None = None,
                 known_ability: str | None = None) -> dict | None:
        """Inferred set for an opponent species: same dict shape as
        parse_showdown_team (nature/evs/ivs/item/ability/moves) plus an
        optional 'tera_type'. None when the source has nothing. When a
        sampling rng is active (translate(..., rng=...)), the set is drawn
        from the distributions instead of taking the top values.

        Tier 1 is the curated PS full-set database (ps_sets.py): joint sets
        filtered by every observation, so item/spread/move correlations
        survive. Chaos-stat marginals are the fallback tier."""
        if self._set_source == "monotype":
            if self._opp_type is None:
                return None
            return self._canonical().get(self._opp_type, {}).get(species)
        if self._set_source is None:
            return None

        booked = self._book_set(species, known_moves)

        # CONSTRAINT LAYER: everything the observations have PROVEN this mon
        # cannot hold, applied to every tier below (curated, archive, chaos)
        # rather than checked ad hoc per tier — the scattered version is how
        # the Choice Band Gliscor survived a game's worth of contrary
        # evidence. Revealed items still win over all of it.
        disproven = frozenset()
        if self._obs is not None:
            disproven = self._obs.forbidden(species)
            if _normalize(species) in self._obs.choice_disproven:
                disproven = disproven | _CHOICE_LOCKERS

        # full-set archive tier: a correlated whole-team match beats every
        # per-species source; the book (this exact opponent's observed
        # behaviour) still overlays it
        arch_team = getattr(self, "_archive_team", None)
        if arch_team is not None and self._archive_covers(known_moves):
            s = arch_team.get(_normalize(species))
            if s is not None:
                return self._apply_book(dict(s), booked)

        if getattr(self, "_prefer_ps", True):
            ps_cand = self._ps_candidate(species, known_moves, known_item,
                                         known_ability,
                                         exclude_items=disproven)
            if ps_cand is not None:
                if _PS_TERA_SOURCE == "chaos":
                    # tera only: the PS set's moves/item are its strength,
                    # its single tera is its worst-measured cell (see the
                    # _PS_TERA_SOURCE note). Book (this opponent's own
                    # observed tera) still overlays below; a revealed tera
                    # always wins later in _tera_fields.
                    # getattr: predicted_preview_paste calls _opp_set before
                    # the first translate() has stashed a battle tag
                    stats = self._chaos().pokemon.get(species)
                    t = _stable_tera_draw(
                        getattr(stats, "_tera_types", None) or {},
                        getattr(self, "_battle_tag", ""), species) \
                        if stats else None
                    if t:
                        ps_cand = dict(ps_cand, tera_type=t)
                return self._apply_book(ps_cand, booked)

        stats = self._chaos().pokemon.get(species)
        if stats is None:
            return None
        from showdown.chaos_stats import _COHERENCE_ON, incompatible_items
        rng = getattr(self, "_rng", None)
        if rng is not None:
            sampled = stats.sample_set(
                rng, known_moves=known_moves,
                speed_pessimistic=getattr(self, "_speed_pess", False),
                known_item=known_item, known_ability=known_ability,
                exclude_items=disproven)
            nature, evs = sampled["nature"], sampled["evs"]
            item, ability = sampled["item"], sampled["ability"]
            moves, tera = sampled["moves"], sampled["tera_type"]
        else:
            spread = stats.top_spread()
            nature, evs = spread if spread else (
                "Serious", dict.fromkeys(("hp", "atk", "def",
                                          "spa", "spd", "spe"), 85))
            ability = _normalize(known_ability) if known_ability \
                else stats.top_ability()
            if known_item:
                item = _normalize(known_item)
            else:
                item = stats.top_item(
                    exclude=incompatible_items(known_moves) | disproven) \
                    or "none"
                if _COHERENCE_ON and ability == "poisonheal":
                    item = "toxicorb"
            moves, tera = stats.top_moves(4), stats.top_tera_type()

        # tier 2: joint moveset fragments (and teras) actually observed in
        # ladder replays beat chaos-composed marginals; archetype-matched
        # data (this exact 6-mon team) beats species-level data. Items stay
        # with the upper tiers — choice items are invisible in replay logs.
        replay_idx = self._replay_index()
        if replay_idx is not None:
            team = getattr(self, "_archetype", None)
            # item conditioning rides the same gate as the partial backoff.
            # Visibility-aware by construction (see conditioned_item): it only
            # re-allocates the mass chaos gives to items replays can SEE, so
            # P(Choice item) is untouched — the naive version would have
            # deleted choice-lock modelling, 14.06% of real usage vs 0.09% of
            # replay observations.
            if _TEAM_COND_OVERLAP and team is not None and known_item is None:
                cs = self._chaos()
                shares = {}
                for st in cs.pokemon.values():
                    for i, p in st._items.items():
                        shares[i] = shares.get(i, 0.0) + p * st.usage
                probs = dict(stats._items)
                alt = replay_idx.conditioned_item(species, probs, team,
                                                  shares, rng)
                if alt:
                    item = alt
            frag = replay_idx.pick_moves(species, known_moves, team=team,
                                         rng=rng)
            if frag:
                pad = [m for m in moves if m not in frag]
                moves = (list(frag) + pad)[:4]
                replay_tera = replay_idx.pick_tera(species, team, rng)
                if replay_tera:
                    tera = replay_tera

        # tier 0 (highest): what THIS opponent actually did with this species
        # in our own past games — a username-specific observation beats any
        # corpus statistic (the baselines run remarkably stable sets).
        return self._apply_book({
            "nature": nature.capitalize(),
            "evs": evs,
            "ivs": dict.fromkeys(("hp", "atk", "def", "spa", "spd", "spe"), 31),
            "item": item,
            "ability": ability,
            "moves": moves,
            "tera_type": tera,
        }, booked)

    def _ps_candidate(self, species: str, known_moves: tuple[str, ...],
                      known_item: str | None, known_ability: str | None,
                      exclude_items: frozenset = frozenset()) -> dict | None:
        """Pick a curated full set consistent with all observations, or None
        to fall through to chaos. Mirrors foul-play's tier semantics: always
        used when nothing is revealed; with reveals, the sampler keeps 25%
        chaos draws for diversity."""
        ps = self._ps_index()
        if ps is None:
            return None
        floor = self._obs.speed_floor.get(species) if self._obs else None
        cands = ps.consistent(species, known_moves=known_moves,
                              known_item=known_item,
                              known_ability=known_ability,
                              speed_floor=floor)
        # An item ruled out does NOT rule out the SET. Most builds tolerate
        # several items that shift them subtly (Kingambit's Leftovers vs
        # Black Glasses vs Air Balloon are the same set), so dropping the
        # candidate would discard its moveset and spread — the very things
        # the curated tier exists to supply — over one refuted axis. Prefer
        # candidates whose item survives; if NONE do, keep them and re-draw
        # the item instead of falling through to a fully chaos-composed set.
        redraw_item = False
        if exclude_items:
            ok = [c for c in cands
                  if _normalize(c.get("item") or "") not in exclude_items]
            if ok:
                cands = ok
            else:
                redraw_item = True
        # confidence gate: with nothing revealed, an editorial dex set is
        # only trusted if the ladder corpus corroborates it — the suite A/B
        # showed uncorroborated curated sets cost more than chaos sampling
        # (legacy -16pp) while corroborated ones pay hugely (fat +24pp)
        if not (known_moves or known_item or known_ability):
            replay_idx = self._replay_index()
            if replay_idx is not None:
                cands = [c for c in cands
                         if replay_idx.corroborates(species, c["moves"])]
        if not cands:
            return None
        rng = getattr(self, "_rng", None)
        if rng is None:
            # deterministic: the usage composite carries the higher weight,
            # else the most prominent dex set
            cand = max(cands, key=lambda c: c["weight"])
        elif getattr(self, "_speed_pess", False):
            cand = max(cands, key=lambda c: c["spe_stat"] *
                       (1.5 if c["item"] == "choicescarf" else 1.0))
        elif known_moves and rng.random() >= 0.75:
            return None  # occasional chaos draw keeps the worlds diverse
        else:
            cand = rng.choices(cands, weights=[c["weight"] for c in cands])[0]

        item = cand["item"]
        if redraw_item and known_item is None:
            stats = self._chaos().pokemon.get(species)
            pool = {i: p for i, p in (stats._items if stats else {}).items()
                    if i not in exclude_items}
            if pool:
                rng2 = rng or random.Random(0)
                item = rng2.choices(list(pool),
                                    weights=list(pool.values()))[0]
            else:
                item = "none"
        if (getattr(self, "_speed_pess", False) and known_item is None
                and item != "choicescarf"):
            stats = self._chaos().pokemon.get(species)
            if stats is not None and stats._items.get("choicescarf", 0) >= 0.02:
                item = "choicescarf"
        return {
            "nature": cand["nature"],
            "evs": cand["evs"],
            "ivs": cand["ivs"],
            "item": item,
            "ability": cand["ability"],
            "moves": cand["moves"],
            "tera_type": cand["tera_type"],
        }

    # ---- team preview ----

    _EV_KEYS = (("hp", "HP"), ("atk", "Atk"), ("def", "Def"),
                ("spa", "SpA"), ("spd", "SpD"), ("spe", "Spe"))

    def predicted_preview_paste(self, species_list) -> str:
        """Showdown paste of predicted sets for the opponent's previewed
        species — feeds the 6x6 lead maximin (monotype/lead_picker) at
        team preview time."""
        if self._set_source == "monotype" and self._opp_type is None:
            self._opp_type = _detect_side_type(
                tuple(_normalize(s) for s in species_list))
        blocks = []
        for species in species_list:
            norm = _normalize(species)
            canon = self._opp_set(norm) or {}
            lines = [f"{species} @ {canon.get('item') or 'leftovers'}"]
            if canon.get("ability"):
                lines.append(f"Ability: {canon['ability']}")
            evs = canon.get("evs") or {}
            ev_str = " / ".join(f"{evs[k]} {label}"
                                for k, label in self._EV_KEYS if evs.get(k))
            if ev_str:
                lines.append(f"EVs: {ev_str}")
            lines.append(f"{canon.get('nature', 'Serious')} Nature")
            for mid in (canon.get("moves") or [])[:4]:
                lines.append(f"- {mid}")
            blocks.append("\n".join(lines))
        return "\n\n".join(blocks)

    # ---- entry point ----

    def translate(self, battle, rng=None, speed_pessimistic=False,
                  prefer_ps=True, tera_pessimistic=False) -> pe.State:
        """Build a State for search. With `rng`, opponent unknowns (sets and
        unrevealed species) are SAMPLED instead of taking the deterministic
        most-likely values — callers run one search per sampled world and
        combine (see gen9_player). `speed_pessimistic` makes the sampled sets
        worst-case on speed (fastest spreads, scarf when plausible).

        `prefer_ps` gates the curated-set tier. Series 10 showed why this is
        per-world: PS dex sets are single-candidate for some species (only a
        bulky Rain Setter Pelipper exists, the real one was Specs), so using
        them in EVERY world collapses diversity and both worlds share the
        same confident wrong set. One PS world + one chaos world keeps the
        joint-set quality without losing the tail coverage."""
        self._rng = rng
        self._speed_pess = speed_pessimistic
        self._prefer_ps = prefer_ps
        self._tera_pess = tera_pessimistic
        self._battle_tag = getattr(battle, "battle_tag", "") or ""
        if tera_pessimistic:
            # worst-case world: the opponent's active still holds Tera and
            # burns it DEFENSIVELY into the type that best walls our active.
            # Legal only if no opponent mon has tera'd yet (once per battle);
            # is_terastallized survives a faint, so a spent-then-fainted tera
            # still closes the option (mirrors _both_teras_spent).
            self._opp_tera_available = not any(
                getattr(m, "is_terastallized", False)
                for m in battle.opponent_team.values())
            self._pess_tera_type = (
                self._worst_case_defensive_tera(battle)
                if self._opp_tera_available else None)
        else:
            self._opp_tera_available = False
            self._pess_tera_type = None
        if self._set_source is not None:
            if self._obs is None:
                from showdown.set_inference import BattleObservations
                self._obs = BattleObservations()
            try:
                self._obs.update(battle)
            except Exception:
                pass  # refinement is advisory; never fail a translation
        self._resolve_archetype(battle)
        self._resolve_archive(battle)
        side_one = self._my_side(battle)
        side_two = self._opp_side(battle)
        weather, weather_turns = self._weather(battle)
        terrain, terrain_turns = self._terrain(battle)
        trick_room, tr_turns = self._trick_room(battle)
        return pe.State(
            side_one=side_one, side_two=side_two,
            weather=weather, weather_turns_remaining=weather_turns,
            terrain=terrain, terrain_turns_remaining=terrain_turns,
            trick_room=trick_room, trick_room_turns_remaining=tr_turns,
            team_preview=False,
        )

    # ---- globals ----

    def _revealed_items(self, battle) -> set[str]:
        items = set()
        for mon in list(battle.team.values()) + list(battle.opponent_team.values()):
            if mon.item:
                items.add(_normalize(mon.item))
        return items

    def _weather(self, battle) -> tuple[pe.Weather, int]:
        if not battle.weather:
            return pe.Weather.NONE, 0
        weather_enum, start_turn = next(iter(battle.weather.items()))
        weather = _WEATHER_MAP.get(weather_enum.name, pe.Weather.NONE)
        if weather == pe.Weather.NONE:
            return pe.Weather.NONE, 0
        # DURATION IS ITSELF THE EVIDENCE. The extender rocks are silent
        # items, so waiting for one to be "revealed" never fired for an
        # opponent and we modelled every weather as 5 turns — against a
        # Ninetales running Heat Rock on 94% of sets that is three turns of
        # sun we think we have waited out and have not. A weather still up
        # after 5 elapsed turns CANNOT be the 5-turn version, so the rock is
        # proven. Measured on our own logs (554 runs, re-sets counted
        # separately since Drought re-fires on every switch-in): durations
        # are bimodal at 4 and 7 elapsed turns exactly as the mechanic
        # predicts, and 23% of runs prove an extender.
        elapsed = battle.turn - start_turn
        rock = _WEATHER_ROCK.get(weather)
        duration = 5
        if elapsed >= 5 or (rock and rock in self._revealed_items(battle)):
            duration = 8
        return weather, _clamp_turns(duration - elapsed)

    def _terrain(self, battle) -> tuple[pe.Terrain, int]:
        for field, start_turn in battle.fields.items():
            terrain = _TERRAIN_MAP.get(field.name)
            if terrain is not None:
                elapsed = battle.turn - start_turn
                duration = 8 if (
                    elapsed >= 5
                    or "terrainextender" in self._revealed_items(battle)
                ) else 5
                return terrain, _clamp_turns(duration - elapsed)
        return pe.Terrain.NONE, 0

    def _trick_room(self, battle) -> tuple[bool, int]:
        for field, start_turn in battle.fields.items():
            if field.name == "TRICK_ROOM":
                return True, _clamp_turns(5 - (battle.turn - start_turn))
        return False, 0

    # ---- side-level state ----

    def _side_conditions(self, battle, conditions, side_mons,
                         active) -> pe.SideConditions:
        turn = battle.turn
        side_items = {_normalize(m.item) for m in side_mons if m.item}
        # Light Clay is silent too, so the same duration proof applies: a
        # screen still standing after 5 elapsed turns proves the Clay. Uses
        # the OLDEST standing screen on this side, since one Clay extends
        # them all.
        screen_starts = [v for c, v in conditions.items()
                         if c.name in ("REFLECT", "LIGHT_SCREEN",
                                       "AURORA_VEIL")]
        clay_proven = any(turn - v >= 5 for v in screen_starts)
        screen_duration = 8 if ("lightclay" in side_items or clay_proven) else 5

        kwargs: dict[str, int] = {}
        for cond, value in conditions.items():
            name = cond.name
            if name == "SPIKES":
                kwargs["spikes"] = value
            elif name == "TOXIC_SPIKES":
                kwargs["toxic_spikes"] = value
            elif name == "STEALTH_ROCK":
                kwargs["stealth_rock"] = 1
            elif name == "STICKY_WEB":
                kwargs["sticky_web"] = 1
            elif name == "REFLECT":
                kwargs["reflect"] = _clamp_turns(screen_duration - (turn - value))
            elif name == "LIGHT_SCREEN":
                kwargs["light_screen"] = _clamp_turns(screen_duration - (turn - value))
            elif name == "AURORA_VEIL":
                kwargs["aurora_veil"] = _clamp_turns(screen_duration - (turn - value))
            elif name == "TAILWIND":
                kwargs["tailwind"] = _clamp_turns(4 - (turn - value))
            elif name == "SAFEGUARD":
                kwargs["safeguard"] = _clamp_turns(5 - (turn - value))
            elif name == "MIST":
                kwargs["mist"] = _clamp_turns(5 - (turn - value))

        if active is not None:
            if active.status is not None and active.status.name == "TOX":
                kwargs["toxic_count"] = active.status_counter
            if active.protect_counter:
                kwargs["protect"] = active.protect_counter
        return pe.SideConditions(**kwargs)

    def _active_volatiles(self, active) -> tuple[set, pe.VolatileStatusDurations]:
        """(volatile_statuses, durations) for one active."""
        vols: set[str] = set()
        durs: dict[str, int] = {}
        if active is None:
            return vols, pe.VolatileStatusDurations()
        for effect, count in active.effects.items():
            name = effect.name.replace("_", "")
            if name not in _VOLATILE_ALLOW:
                continue
            vols.add(name.lower())
            # poke-env counts turns an effect has been active; the engine
            # wants turns remaining. Best-effort for the turn-limited ones.
            if name == "TAUNT":
                durs["taunt"] = _clamp_turns(3 - count)
            elif name == "ENCORE":
                durs["encore"] = _clamp_turns(3 - count)
            elif name == "CONFUSION":
                durs["confusion"] = 2  # actual remaining is hidden (1-4)
            elif name == "YAWN":
                durs["yawn"] = 1
            elif name == "SLOWSTART":
                durs["slowstart"] = _clamp_turns(5 - count)
        return vols, pe.VolatileStatusDurations(**durs)

    def _boost_kwargs(self, active) -> dict[str, int]:
        if active is None:
            return {}
        boosts = active.boosts or {}
        return {
            "attack_boost": boosts.get("atk", 0),
            "defense_boost": boosts.get("def", 0),
            "special_attack_boost": boosts.get("spa", 0),
            "special_defense_boost": boosts.get("spd", 0),
            "speed_boost": boosts.get("spe", 0),
            "accuracy_boost": boosts.get("accuracy", 0),
            "evasion_boost": boosts.get("evasion", 0),
        }

    def _slot_order(self, battle, side_key, mons) -> list[str]:
        """Canonical species -> slot ordering for one side, FIXED for the
        whole battle.

        The engine identifies the active mon by `active_index`, so the array
        does not need the active first. Keeping slots stable is what lets
        cross-turn tree reuse survive a switch: a retained subtree's
        MoveChoice::Switch(i) is a SLOT INDEX, so if the array permuted
        between turns, slot i would silently denote a different Pokemon and
        every switch line in the reused tree would be wrong. Measured on real
        ladder games, 61% of turns contain a switch, so an active-first array
        made reuse impossible on most turns.

        Opponent order is seeded from team preview (all six species are known
        from turn 0); our own side is seeded from the request's team order.
        Anything unseen is appended on first sight, so a missing preview only
        costs stability, never correctness."""
        order = self._slots.get(side_key)
        if order is None:
            order = []
            if side_key == "opp":
                for p in (getattr(battle, "teampreview_opponent_team", None) or []):
                    sp = _slot_key(getattr(p, "species", "") or "")
                    if sp and sp not in order:
                        order.append(sp)
            self._slots[side_key] = order
        for mon in mons:
            sp = _slot_key(getattr(mon, "species", "") or "")
            if not sp or sp in order:
                continue
            # a revealed forme may not match its preview entry verbatim
            # (preview shows an undisclosed forme as "Zamazenta-*"): adopt
            # that slot when exactly one base-name candidate is free
            base = sp.split("-")[0] if "-" in sp else sp
            cands = [i for i, o in enumerate(order)
                     if o == base or o.split("-")[0] == base]
            if len(cands) == 1:
                order[cands[0]] = sp
            elif len(order) < 6:
                order.append(sp)
        return order

    def _assemble_side(self, battle, mons, active, conditions,
                       build_one, side_key, force_switch=False,
                       force_trapped=False, fill=None) -> pe.Side:
        """Place mons in STABLE per-battle slots, pad to 6, attach side-level
        state and point `active_index` at the active mon's slot.
        `fill` supplies predicted mons for unrevealed slots; remaining
        slots become fainted dummies."""
        # the slot survives a faint (it's what active_index points at), but
        # boosts/volatiles die with the mon and must not be attributed to
        # whoever replaces it
        active_species = _slot_key(getattr(active, "species", "") or "") \
            if active is not None else ""
        if active is not None and active.fainted:
            active = None

        order = self._slot_order(battle, side_key, mons)
        slots: list = [None] * 6

        def _place(obj, species):
            idx = order.index(species) if species in order else None
            if idx is None or idx >= 6 or slots[idx] is not None:
                idx = next((i for i, s in enumerate(slots) if s is None), None)
            if idx is not None:
                slots[idx] = obj
            return idx

        for mon in mons:
            sp = _slot_key(getattr(mon, "species", "") or "")
            _place(_fainted_mon(mon) if mon.fainted else build_one(mon), sp)
        for predicted in (fill or []):
            if all(s is not None for s in slots):
                break
            _place(predicted, _slot_key(getattr(predicted, "id", "") or ""))
        pokemon = [s if s is not None else pe.Pokemon.create_fainted()
                   for s in slots]

        # a fainted active keeps its slot: the engine reads hp<=0 at
        # active_index and offers the replacement switches itself, which is
        # why the old "fainted dummy at slot 0" hack is gone
        active_slot = order.index(active_species) \
            if active_species in order and order.index(active_species) < 6 else 0

        vols, durs = self._active_volatiles(active)
        sub_health = 0
        if active is not None and "substitute" in vols:
            # poke-env tracks sub presence, not HP; use the engine-side maxhp
            # (opponent poke-env HP is normalized to /100)
            sub_health = max(1, pokemon[active_slot].maxhp // 4)

        # last_used_move feeds the engine's Encore re-routing, Fake Out /
        # First Impression legality, and choice-lock continuation. poke-env
        # clears last_move on switch-out, so a known one is from this stint.
        last_used_move = "move:none"
        if active is not None:
            last = active.last_move
            if last is not None:
                lid = _normalize(last.id)
                for i, mv in enumerate(pokemon[active_slot].moves):
                    if mv.id == lid:
                        last_used_move = f"move:{i}"
                        break
            elif active.first_turn:
                last_used_move = "switch:0"  # just switched in (Fake Out live)
        # the engine panics if ENCORE is set without a real move slot;
        # without a known last move, dropping the volatile beats a panic
        if "encore" in vols and not (last_used_move.startswith("move:")
                                     and last_used_move != "move:none"):
            vols.discard("encore")

        return pe.Side(
            pokemon=pokemon[:6],
            active_index=str(active_slot),
            side_conditions=self._side_conditions(battle, conditions, mons, active),
            volatile_statuses=vols,
            volatile_status_durations=durs,
            substitute_health=sub_health,
            last_used_move=last_used_move,
            force_switch=force_switch,
            force_trapped=force_trapped,
            **self._boost_kwargs(active),
        )

    def _my_side(self, battle) -> pe.Side:
        # poke-env doesn't parse the request's per-mon teraType; mine it from
        # the raw request so our own un-tera'd mons carry their real tera type
        self._own_tera = {}
        request = getattr(battle, "last_request", None) or {}
        for pkmn in (request.get("side") or {}).get("pokemon", []):
            tera = pkmn.get("teraType")
            if tera:
                self._own_tera[_normalize(pkmn["ident"][4:])] = tera.lower()

        # the request is authoritative for what our active can do THIS turn
        # (choice lock, Taunt, Disable, Encore, no PP). Marking the missing
        # moves disabled carries that restriction into multi-turn search.
        self._own_available = {_normalize(m.id) for m in battle.available_moves}
        self._my_built = {}  # species -> pe.Pokemon, for damage inference
        return self._assemble_side(
            battle,
            mons=list(battle.team.values()),
            active=battle.active_pokemon,
            conditions=battle.side_conditions,
            build_one=self._my_pokemon,
            side_key="me",
            force_switch=bool(battle.force_switch),
            force_trapped=bool(battle.trapped),
        )

    def _opp_side(self, battle) -> pe.Side:
        opp_mons = list(battle.opponent_team.values())
        if self._set_source == "monotype" and self._opp_type is None and opp_mons:
            self._opp_type = _detect_side_type(
                tuple(_normalize(m.species) for m in opp_mons))
        return self._assemble_side(
            battle,
            mons=opp_mons,
            active=battle.opponent_active_pokemon,
            conditions=battle.opponent_side_conditions,
            build_one=self._opp_pokemon,
            side_key="opp",
            fill=self._predicted_fill(opp_mons),
        )

    def _predicted_fill(self, opp_mons) -> list:
        """Predicted mons for the opponent's unrevealed slots.

        Fainted-dummy fill (the gen2 approach) is catastrophic for live play:
        the engine's eval reads empty slots as fainted, so the search believes
        the game is nearly won from turn 1 and plays with unearned aggression
        (measured 0-10 vs foul-play with ~0.98 mid-game evals). Chaos-stats
        team prediction, teammate-correlated with what's been revealed, keeps
        the eval honest.
        """
        n_fill = 6 - len(opp_mons)
        if n_fill <= 0 or self._set_source in (None, "monotype"):
            # TODO monotype: fill from per-type replay teammate stats
            return []
        try:
            from showdown.chaos_stats import RevealedMon
            revealed = {_normalize(m.species): RevealedMon(_normalize(m.species))
                        for m in opp_mons}
            rng = getattr(self, "_rng", None)
            if rng is not None:
                species = self._chaos().sample_team(revealed, n_fill, rng)
            else:
                species = [_normalize(p.species) for p in
                           self._chaos().predict_team(revealed, n_fill=n_fill)]
        except Exception:
            return []
        return [self._predicted_pokemon(sp) for sp in species]

    def _predicted_pokemon(self, species: str) -> pe.Pokemon:
        """Full-HP engine mon for a predicted (never-revealed) species."""
        entry = self._dex().get(species, {})
        bs = entry.get("baseStats", {})
        canon = self._opp_set(species) or {}
        nature_pair = _NATURE_TABLE.get(canon.get("nature", "Serious"))

        def mult(stat: str) -> float:
            if nature_pair is None:
                return 1.0
            if stat == nature_pair[0]:
                return 1.1
            if stat == nature_pair[1]:
                return 0.9
            return 1.0

        evs = canon.get("evs") or dict.fromkeys(
            ("hp", "atk", "def", "spa", "spd", "spe"), 85)
        ivs = canon.get("ivs") or dict.fromkeys(
            ("hp", "atk", "def", "spa", "spd", "spe"), 31)

        def calc(stat: str, is_hp: bool = False) -> int:
            return _calc_stat_modern(bs.get(stat, 80), ivs[stat], evs[stat],
                                     100, mult(stat), is_hp)

        maxhp = calc("hp", is_hp=True)
        moves = [pe.Move(id=m, pp=16) for m in (canon.get("moves") or [])[:4]]
        while len(moves) < 4:
            moves.append(pe.Move(id="none", pp=0))
        types = [t.lower() for t in entry.get("types", ["Normal"])]
        while len(types) < 2:
            types.append("typeless")
        types = tuple(types[:2])
        ability = canon.get("ability") or _normalize(
            str(entry.get("abilities", {}).get("0", "noability")))
        return pe.Pokemon(
            id=species, level=100,
            hp=maxhp, maxhp=maxhp,
            attack=calc("atk"), defense=calc("def"),
            special_attack=calc("spa"), special_defense=calc("spd"),
            speed=calc("spe"),
            types=types, base_types=types,
            ability=ability, base_ability=ability,
            item=canon.get("item", "none") or "none",
            weight_kg=self._weight(species),
            moves=moves[:4],
            terastallized=False,
            tera_type=canon.get("tera_type") or types[0],
        )

    # ---- pokemon-level state ----

    @staticmethod
    def _status_fields(mon) -> dict:
        status = "none"
        sleep_turns = 0
        if mon.status is not None:
            status = mon.status.name.lower()  # binding accepts showdown short forms
            if status == _SLEEP_STATUS:
                sleep_turns = min(mon.status_counter, 3)
        return {"status": status, "sleep_turns": sleep_turns}

    def _types(self, mon) -> tuple[str, str]:
        """Base types. poke-env's type_1/type_2 reflect terastallization and
        temporary type changes; the engine wants base types (it applies
        tera_type itself from the terastallized flag), so prefer the pokedex.
        Temporary types (Soak etc.) are knowingly dropped."""
        entry = self._dex().get(_normalize(mon.species), {})
        types = [t.lower() for t in entry.get("types", [])]
        if not types:
            if mon.type_1:
                types.append(mon.type_1.name.lower())
            if mon.type_2:
                types.append(mon.type_2.name.lower())
        if not types:
            types = ["normal"]
        while len(types) < 2:
            types.append("typeless")
        return tuple(types[:2])

    @staticmethod
    def _tera_fields(mon, fallback: str) -> dict:
        """terastallized/tera_type kwargs for pe.Pokemon. `fallback` is used
        when the tera type isn't known (opponent hasn't tera'd yet)."""
        revealed = mon.tera_type.name.lower() if mon.tera_type else None
        return {
            "terastallized": bool(mon.is_terastallized),
            "tera_type": revealed or fallback,
        }

    def _weight(self, species_norm: str) -> float:
        return float(self._dex().get(species_norm, {}).get("weightkg", 50.0))

    def _my_pokemon(self, mon) -> pe.Pokemon:
        species = _normalize(mon.species)
        entry = self._dex().get(species, {})

        stats = mon.stats or {}
        if any(stats.get(k) is None for k in ("atk", "def", "spa", "spd", "spe")):
            # stats come from the request; fall back to neutral 85-EV estimates
            bs = entry.get("baseStats", {})
            stats = {k: _calc_stat_modern(bs.get(k, 80), 31, 85, mon.level, 1.0, False)
                     for k in ("atk", "def", "spa", "spd", "spe")}

        move_ids = [_normalize(mid) for mid in mon.moves]
        available = getattr(self, "_own_available", set())
        # only restrict the active mon, and only when the request's available
        # moves overlap its known moves (a struggle-only request would
        # otherwise disable everything)
        restrict = bool(mon.active) and bool(available & set(move_ids))
        moves = []
        for mid, move_obj in zip(move_ids, mon.moves.values()):
            moves.append(pe.Move(id=mid, pp=max(0, move_obj.current_pp),
                                 disabled=restrict and mid not in available))
        while len(moves) < 4:
            moves.append(pe.Move(id="none", pp=0))

        ability = _normalize(mon.ability) if mon.ability else \
            _normalize(str(entry.get("abilities", {}).get("0", "noability")))
        types = self._types(mon)
        maxhp = mon.max_hp or 100
        tera_fallback = getattr(self, "_own_tera", {}).get(species, types[0])

        built = pe.Pokemon(
            id=species, level=mon.level,
            hp=mon.current_hp or 0, maxhp=maxhp,
            attack=stats["atk"], defense=stats["def"],
            special_attack=stats["spa"], special_defense=stats["spd"],
            speed=stats["spe"],
            types=types, base_types=types,
            ability=ability, base_ability=ability,
            item=_normalize(mon.item) if mon.item else "none",
            weight_kg=self._weight(species),
            moves=moves[:4],
            **self._tera_fields(mon, tera_fallback),
            **self._status_fields(mon),
        )
        if hasattr(self, "_my_built"):
            self._my_built[species] = built  # exact stats for damage inference
        return built

    def _opp_pokemon(self, mon) -> pe.Pokemon:
        species = _normalize(mon.species)
        entry = self._dex().get(species, {})
        bs = entry.get("baseStats", {})

        # poke-env item semantics: "unknown_item" sentinel = never revealed
        # (truthy!), None/"" = revealed to be gone (knocked off / consumed)
        raw_item = mon.item
        revealed_item_id = None
        if raw_item and _normalize(raw_item) != "unknownitem":
            revealed_item_id = _normalize(raw_item)
        elif raw_item and self._obs is not None:
            # still the unknown sentinel, but the protocol may have named it
            # in a [from] tag poke-env ignores ("poisoned by Toxic Orb") —
            # sentinel-only so a knocked-off item is never resurrected
            revealed_item_id = self._obs.revealed_item.get(_normalize(species))
        revealed_ability = _normalize(mon.ability) if mon.ability else None
        if revealed_ability is None and self._obs is not None:
            revealed_ability = self._obs.revealed_ability.get(
                _normalize(species))

        known_move_ids = tuple(_normalize(m) for m in mon.moves)
        canon = self._opp_set(
            species, known_moves=known_move_ids,
            known_item=revealed_item_id, known_ability=revealed_ability)

        # stats: canonical spread when we have one, neutral 85s otherwise
        if canon is not None:
            nature_pair = _NATURE_TABLE.get(canon.get("nature", "Serious"))

            def mult(stat: str) -> float:
                if nature_pair is None:
                    return 1.0
                if stat == nature_pair[0]:
                    return 1.1
                if stat == nature_pair[1]:
                    return 0.9
                return 1.0

            evs, ivs = canon["evs"], canon["ivs"]

            def calc(stat: str, is_hp: bool = False) -> int:
                return _calc_stat_modern(bs.get(stat, 80), ivs[stat], evs[stat],
                                         mon.level, mult(stat), is_hp)
        else:
            def calc(stat: str, is_hp: bool = False) -> int:
                return _calc_stat_modern(bs.get(stat, 80), 31, 85,
                                         mon.level, 1.0, is_hp)

        maxhp = calc("hp", is_hp=True)

        # revealed item/ability beat the canonical guess
        if revealed_item_id:
            item = revealed_item_id
            item_known = True
        elif raw_item is None or raw_item == "":
            item = "none"
            item_known = True
        else:
            item = canon["item"] if canon is not None else "none"
            item_known = False
        if mon.ability:
            ability = _normalize(mon.ability)
        elif canon is not None and canon.get("ability"):
            ability = canon["ability"]
        else:
            ability = _normalize(str(entry.get("abilities", {}).get("0", "noability")))

        # observational refinement (set_inference.py) — inferred details only
        revealed_item = item_known
        spe_stat = calc("spe")
        if self._obs is not None and not revealed_item:
            from showdown.chaos_stats import incompatible_items
            choice_vetoed = (
                "choicescarf" in incompatible_items(known_move_ids)
                or species in self._obs.choice_disproven)
            # speed floor: they outsped something our model says they can't
            if self._obs.scarf_needed(species, spe_stat, item):
                # CHEAPEST EXPLANATION FIRST: full Speed investment before any
                # item claim. Our canonical spread for a bulky mon carries
                # little Speed, so "they invested" explains most floors — and
                # a wrong Scarf is not a cosmetic error, it tells the search
                # the target is CHOICE-LOCKED. Only when max investment still
                # cannot reach the floor is an item actually required.
                max_spe = _calc_stat_modern(bs.get("spe", 80), 31, 252,
                                            mon.level, 1.1, False)
                if self._obs.max_speed_suffices(species, max_spe) \
                        or choice_vetoed:
                    # a revealed status move rules the scarf claim out too
                    # (Booster Energy or investment explains the floor
                    # without asserting a choice lock)
                    spe_stat = max_spe
                else:
                    item = "choicescarf"
                    self._obs.confirmed[species] = "choicescarf"
                    if self._obs.max_speed_needed(species, spe_stat):
                        spe_stat = max_spe
            # speed ceiling: drop a wrongly-inferred scarf / clamp the stat
            clamp = self._obs.speed_clamp(species, spe_stat, item)
            if clamp is not None:
                spe_stat, clamped_item = clamp
                if clamped_item != item:
                    item = (canon or {}).get("item") or "none"
                    if item == "choicescarf":
                        item = "none"
            # damage bracket: probe with current belief; a beyond-max-roll
            # hit upgrades to the weakest item that explains it
            probe = pe.Pokemon(
                id=species, level=mon.level, hp=maxhp, maxhp=maxhp,
                attack=calc("atk"), defense=calc("def"),
                special_attack=calc("spa"), special_defense=calc("spd"),
                speed=spe_stat, types=self._types(mon),
                base_types=self._types(mon),
                ability=ability, base_ability=ability, item=item,
                weight_kg=self._weight(species),
            )
            upgrade = self._obs.damage_item_upgrade(
                species, probe, getattr(self, "_my_built", {}),
                known_moves=known_move_ids)
            if upgrade:
                item = upgrade
                self._obs.confirmed[species] = upgrade
            # negative evidence: walked in over Stealth Rock for free ->
            # Heavy-Duty Boots (only when Magic Guard can't explain it, and
            # never over a damage-item upgrade the same mon just proved)
            elif self._obs.boots_inferred(species):
                item = "heavydutyboots"
                self._obs.confirmed[species] = "heavydutyboots"

        # active choice lock: last_move is cleared on switch-out, so a known
        # last move on a choice-locked holder pins everything else. This is
        # consistent with `item` even when the item is only inferred — the
        # search state holds that item either way.
        locked_move = None
        if (bool(mon.active) and mon.last_move is not None
                and (item in _CHOICE_LOCKERS or ability == _LOCKING_ABILITY)):
            locked_move = _normalize(mon.last_move.id)

        # moves: revealed first (PP as observed), canonical fill for the rest
        moves = []
        seen = set()
        for move_id, move_obj in mon.moves.items():
            mid = _normalize(move_id)
            seen.add(mid)
            moves.append(pe.Move(id=mid, pp=max(0, move_obj.current_pp),
                                 disabled=locked_move is not None and mid != locked_move))
        if canon is not None:
            for mid in canon["moves"]:
                if len(moves) >= 4:
                    break
                if mid not in seen:
                    seen.add(mid)
                    moves.append(pe.Move(id=mid, pp=16,
                                         disabled=locked_move is not None))
        while len(moves) < 4:
            moves.append(pe.Move(id="none", pp=0))

        types = self._types(mon)
        tera_fallback = (canon or {}).get("tera_type") or types[0]
        tera_flds = self._tera_fields(mon, tera_fallback)
        # tera-pessimistic world: force the ACTIVE opponent to have already
        # tera'd into its worst-case defensive type against our active. Never
        # touches reserves (only one mon can hold a live tera) nor a mon that
        # already tera'd (keep the observed type).
        if (getattr(self, "_tera_pess", False) and bool(mon.active)
                and getattr(self, "_opp_tera_available", False)
                and getattr(self, "_pess_tera_type", None)
                and not tera_flds["terastallized"]):
            tera_flds = {"terastallized": True,
                         "tera_type": self._pess_tera_type}
        return pe.Pokemon(
            id=species, level=mon.level,
            hp=max(1, round(mon.current_hp_fraction * maxhp)), maxhp=maxhp,
            attack=calc("atk"), defense=calc("def"),
            special_attack=calc("spa"), special_defense=calc("spd"),
            speed=spe_stat,
            types=types, base_types=types,
            ability=ability, base_ability=ability,
            item=item,
            weight_kg=self._weight(species),
            moves=moves[:4],
            **tera_flds,
            **self._status_fields(mon),
        )


# ============================================================
# CHOICE MAPPING (engine move_choice -> poke-env order args)
# ============================================================

def parse_engine_choice(move_choice: str) -> tuple[str, str]:
    """Split an MCTS move_choice into ("switch"|"move", normalized id).

    poke-engine emits switches as "switch <species>" and moves as the bare
    move id. The caller matches the id against battle.available_switches /
    available_moves and builds the poke-env order.
    """
    if move_choice.startswith("switch "):
        return "switch", _normalize(move_choice[7:])
    return "move", _normalize(move_choice)
