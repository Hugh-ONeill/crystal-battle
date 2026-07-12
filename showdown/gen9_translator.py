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

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import poke_engine as pe

from showdown.name_mapping import _normalize
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


# ============================================================
# TRANSLATOR
# ============================================================

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

    def __init__(self, elo_bucket: int = 1500, set_source: str | None = "monotype"):
        self._elo = elo_bucket
        self._set_source = set_source
        self._opp_type: str | None = None
        self._obs = None  # per-battle observational set refinement

    def new_battle(self):
        self._opp_type = None
        self._obs = None

    # ---- lazy shared data ----

    @classmethod
    def _dex(cls):
        if cls._pokedex is None:
            from poke_env.data.gen_data import GenData
            cls._pokedex = GenData.from_gen(9).pokedex
        return cls._pokedex

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
        if self._set_source in (None, "monotype"):
            return None
        from showdown.ps_sets import get_index
        return get_index(self._set_source)

    def _replay_index(self):
        if self._set_source in (None, "monotype"):
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
            self._archetype = idx.team_match(species)

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

        if getattr(self, "_prefer_ps", True):
            ps_cand = self._ps_candidate(species, known_moves, known_item,
                                         known_ability)
            if ps_cand is not None:
                return ps_cand

        stats = self._chaos().pokemon.get(species)
        if stats is None:
            return None
        rng = getattr(self, "_rng", None)
        if rng is not None:
            sampled = stats.sample_set(
                rng, known_moves=known_moves,
                speed_pessimistic=getattr(self, "_speed_pess", False))
            nature, evs = sampled["nature"], sampled["evs"]
            item, ability = sampled["item"], sampled["ability"]
            moves, tera = sampled["moves"], sampled["tera_type"]
        else:
            spread = stats.top_spread()
            nature, evs = spread if spread else (
                "Serious", dict.fromkeys(("hp", "atk", "def",
                                          "spa", "spd", "spe"), 85))
            item, ability = stats.top_item() or "none", stats.top_ability()
            moves, tera = stats.top_moves(4), stats.top_tera_type()

        # tier 2: joint moveset fragments (and teras) actually observed in
        # ladder replays beat chaos-composed marginals; archetype-matched
        # data (this exact 6-mon team) beats species-level data. Items stay
        # with the upper tiers — choice items are invisible in replay logs.
        replay_idx = self._replay_index()
        if replay_idx is not None:
            team = getattr(self, "_archetype", None)
            frag = replay_idx.pick_moves(species, known_moves, team=team,
                                         rng=rng)
            if frag:
                pad = [m for m in moves if m not in frag]
                moves = (list(frag) + pad)[:4]
                replay_tera = replay_idx.pick_tera(species, team, rng)
                if replay_tera:
                    tera = replay_tera
        return {
            "nature": nature.capitalize(),
            "evs": evs,
            "ivs": dict.fromkeys(("hp", "atk", "def", "spa", "spd", "spe"), 31),
            "item": item,
            "ability": ability,
            "moves": moves,
            "tera_type": tera,
        }

    def _ps_candidate(self, species: str, known_moves: tuple[str, ...],
                      known_item: str | None,
                      known_ability: str | None) -> dict | None:
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
                  prefer_ps=True) -> pe.State:
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
        if self._set_source is not None:
            if self._obs is None:
                from showdown.set_inference import BattleObservations
                self._obs = BattleObservations()
            try:
                self._obs.update(battle)
            except Exception:
                pass  # refinement is advisory; never fail a translation
        self._resolve_archetype(battle)
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
        duration = 5
        rock = _WEATHER_ROCK.get(weather)
        if rock and rock in self._revealed_items(battle):
            duration = 8
        return weather, _clamp_turns(duration - (battle.turn - start_turn))

    def _terrain(self, battle) -> tuple[pe.Terrain, int]:
        for field, start_turn in battle.fields.items():
            terrain = _TERRAIN_MAP.get(field.name)
            if terrain is not None:
                duration = 8 if "terrainextender" in self._revealed_items(battle) else 5
                return terrain, _clamp_turns(duration - (battle.turn - start_turn))
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
        screen_duration = 8 if "lightclay" in side_items else 5

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

    def _assemble_side(self, battle, mons, active, conditions,
                       build_one, force_switch=False, force_trapped=False,
                       fill=None) -> pe.Side:
        """Order mons active-first, pad to 6, attach side-level state.
        `fill` supplies predicted mons for unrevealed slots; remaining
        slots become fainted dummies."""
        if active is not None and active.fainted:
            # boosts/volatiles die with the mon; don't attribute them to
            # whoever ends up in slot 0
            active = None

        pokemon = []
        if active is None and force_switch:
            # replacement choice after a KO: the engine must see a FAINTED
            # active at slot 0, or it treats the first bench mon as already
            # on the field and never offers it as the replacement
            pokemon.append(pe.Pokemon.create_fainted())
        if active is not None:
            pokemon.append(build_one(active))
        for mon in mons:
            if mon is active:
                continue
            if mon.fainted:
                pokemon.append(pe.Pokemon.create_fainted())
            else:
                pokemon.append(build_one(mon))
        for predicted in (fill or []):
            if len(pokemon) >= 6:
                break
            pokemon.append(predicted)
        while len(pokemon) < 6:
            pokemon.append(pe.Pokemon.create_fainted())

        vols, durs = self._active_volatiles(active)
        sub_health = 0
        if active is not None and "substitute" in vols:
            # poke-env tracks sub presence, not HP; use the engine-side maxhp
            # (opponent poke-env HP is normalized to /100)
            sub_health = max(1, pokemon[0].maxhp // 4)

        # last_used_move feeds the engine's Encore re-routing, Fake Out /
        # First Impression legality, and choice-lock continuation. poke-env
        # clears last_move on switch-out, so a known one is from this stint.
        last_used_move = "move:none"
        if active is not None:
            last = active.last_move
            if last is not None:
                lid = _normalize(last.id)
                for i, mv in enumerate(pokemon[0].moves):
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
        revealed_ability = _normalize(mon.ability) if mon.ability else None

        canon = self._opp_set(
            species, known_moves=tuple(_normalize(m) for m in mon.moves),
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
            # speed floor: they outsped something our model says they can't
            if self._obs.scarf_needed(species, spe_stat, item):
                item = "choicescarf"
                if self._obs.max_speed_needed(species, spe_stat):
                    spe_stat = _calc_stat_modern(bs.get("spe", 80), 31, 252,
                                                 mon.level, 1.1, False)
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
                species, probe, getattr(self, "_my_built", {}))
            if upgrade:
                item = upgrade

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
            **self._tera_fields(mon, tera_fallback),
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
