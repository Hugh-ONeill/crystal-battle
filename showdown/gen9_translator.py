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

    def new_battle(self):
        self._opp_type = None

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

    def _opp_set(self, species: str) -> dict | None:
        """Inferred set for an opponent species: same dict shape as
        parse_showdown_team (nature/evs/ivs/item/ability/moves) plus an
        optional 'tera_type'. None when the source has nothing."""
        if self._set_source == "monotype":
            if self._opp_type is None:
                return None
            return self._canonical().get(self._opp_type, {}).get(species)
        if self._set_source is None:
            return None
        stats = self._chaos().pokemon.get(species)
        if stats is None:
            return None
        spread = stats.top_spread()
        nature, evs = spread if spread else ("Serious",
                                             dict.fromkeys(("hp", "atk", "def",
                                                            "spa", "spd", "spe"), 85))
        return {
            "nature": nature.capitalize(),
            "evs": evs,
            "ivs": dict.fromkeys(("hp", "atk", "def", "spa", "spd", "spe"), 31),
            "item": stats.top_item() or "none",
            "ability": stats.top_ability(),
            "moves": stats.top_moves(4),
            "tera_type": stats.top_tera_type(),
        }

    # ---- entry point ----

    def translate(self, battle) -> pe.State:
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
                       build_one, force_switch=False, force_trapped=False) -> pe.Side:
        """Order mons active-first, pad to 6, attach side-level state."""
        if active is not None and active.fainted:
            # boosts/volatiles die with the mon; don't attribute them to
            # whoever ends up in slot 0
            active = None

        pokemon = []
        if active is not None:
            pokemon.append(build_one(active))
        for mon in mons:
            if mon is active:
                continue
            if mon.fainted:
                pokemon.append(pe.Pokemon.create_fainted())
            else:
                pokemon.append(build_one(mon))
        while len(pokemon) < 6:
            pokemon.append(pe.Pokemon.create_fainted())

        vols, durs = self._active_volatiles(active)
        sub_health = 0
        if active is not None and "substitute" in vols:
            # poke-env tracks sub presence, not HP; use the engine-side maxhp
            # (opponent poke-env HP is normalized to /100)
            sub_health = max(1, pokemon[0].maxhp // 4)

        # the engine requires last_used_move to be a real move slot whenever
        # ENCORE is set (it re-routes the choice to that slot); without a
        # known last move, dropping the volatile beats a Rust panic
        last_used_move = "move:none"
        if "encore" in vols:
            last_idx = None
            last = active.last_move if active is not None else None
            if last is not None:
                lid = _normalize(last.id)
                for i, mv in enumerate(pokemon[0].moves):
                    if mv.id == lid:
                        last_idx = i
                        break
            if last_idx is not None:
                last_used_move = f"move:{last_idx}"
            else:
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

        moves = []
        for move_id, move_obj in mon.moves.items():
            moves.append(pe.Move(id=_normalize(move_id),
                                 pp=max(0, move_obj.current_pp)))
        while len(moves) < 4:
            moves.append(pe.Move(id="none", pp=0))

        ability = _normalize(mon.ability) if mon.ability else \
            _normalize(str(entry.get("abilities", {}).get("0", "noability")))
        types = self._types(mon)
        maxhp = mon.max_hp or 100
        tera_fallback = getattr(self, "_own_tera", {}).get(species, types[0])

        return pe.Pokemon(
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

    def _opp_pokemon(self, mon) -> pe.Pokemon:
        species = _normalize(mon.species)
        entry = self._dex().get(species, {})
        bs = entry.get("baseStats", {})

        canon = self._opp_set(species)

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

        # moves: revealed first (PP as observed), canonical fill for the rest
        moves = []
        seen = set()
        for move_id, move_obj in mon.moves.items():
            mid = _normalize(move_id)
            seen.add(mid)
            moves.append(pe.Move(id=mid, pp=max(0, move_obj.current_pp)))
        if canon is not None:
            for mid in canon["moves"]:
                if len(moves) >= 4:
                    break
                if mid not in seen:
                    seen.add(mid)
                    moves.append(pe.Move(id=mid, pp=16))
        while len(moves) < 4:
            moves.append(pe.Move(id="none", pp=0))

        # revealed item/ability beat the canonical guess
        if mon.item:
            item = _normalize(mon.item)
        elif mon.item == "":  # knocked off / consumed, poke-env keeps ""
            item = "none"
        else:
            item = canon["item"] if canon is not None else "none"
        if mon.ability:
            ability = _normalize(mon.ability)
        elif canon is not None and canon.get("ability"):
            ability = canon["ability"]
        else:
            ability = _normalize(str(entry.get("abilities", {}).get("0", "noability")))

        types = self._types(mon)
        tera_fallback = (canon or {}).get("tera_type") or types[0]
        return pe.Pokemon(
            id=species, level=mon.level,
            hp=max(1, round(mon.current_hp_fraction * maxhp)), maxhp=maxhp,
            attack=calc("atk"), defense=calc("def"),
            special_attack=calc("spa"), special_defense=calc("spd"),
            speed=calc("spe"),
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
