# translate poke-env Battle state to our engine's BattleState
#
# handles partial information: opponent team is revealed incrementally,
# unrevealed slots are filled with dummy fainted pokemon

from __future__ import annotations

import random
from typing import Any

from engine.battle_state import BattleState
from engine.data_loader import DataStore
from engine.move import MoveSlot, MoveTemplate
from engine.player_state import PlayerState, SideConditions
from engine.pokemon import Pokemon, PokemonSpecies, calc_stat, PERFECT_DV
from engine.types import TypeChart

import random as _random

from .name_mapping import (
    NameMapper, _normalize, STATUS_FROM_SHOWDOWN, WEATHER_FROM_SHOWDOWN, STAT_FROM_SHOWDOWN,
)


# dummy species for unrevealed opponent slots
_DUMMY_SPECIES = PokemonSpecies(
    id=0, name="Unknown", types=["normal"],
    base_stats={"hp": 1, "attack": 1, "defense": 1,
                "special_attack": 1, "special_defense": 1, "speed": 1},
    learnset=[],
)

# a zero-power status move placeholder for unknown move slots
_DUMMY_MOVE = MoveTemplate(
    id=0, name="Unknown", type="normal", power=0,
    accuracy=100, pp=10, priority=0, damage_class="status",
)

# common GSC OU Pokemon with standard sets for filling unrevealed opponent slots
# (species_id, [move_ids], item)
# derived from Smogon sample teams -- the most common 8 Pokemon
_COMMON_OU_POOL = [
    (143, [38, 174, 214, 156], "leftovers"),   # Snorlax: Double-Edge, Curse, Sleep Talk, Rest
    (91,  [191, 57, 92, 153], "leftovers"),    # Cloyster: Spikes, Surf, Toxic, Explosion
    (145, [87, 237, 156, 214], "leftovers"),   # Zapdos: Thunder, HP, Rest, Sleep Talk
    (76,  [89, 153, 46, 229], "leftovers"),    # Golem: EQ, Explosion, Roar, Rapid Spin
    (103, [79, 94, 202, 153], "leftovers"),    # Exeggutor: Sleep Powder, Psychic, Giga Drain, Explosion
    (94,  [223, 87, 8, 153], "leftovers"),     # Gengar: Dynamic Punch, Thunder, Ice Punch, Explosion
    (248, [157, 228, 89, 46], "leftovers"),    # Tyranitar: Rock Slide, Pursuit, EQ, Roar
    (34,  [89, 58, 87, 168], "leftovers"),     # Nidoking: EQ, Ice Beam, Thunder, Thief
]


class StateTranslator:
    """Translates poke-env Battle objects to our engine BattleState."""

    def __init__(self, data: DataStore | None = None, type_chart: TypeChart | None = None):
        self._data = data or DataStore()
        self._tc = type_chart or TypeChart.load()
        self._mapper = NameMapper(self._data)

        # stable team ordering: maps poke-env pokemon identifier -> team slot index
        # rebuilt each battle
        self._my_team_order: dict[str, int] = {}
        self._opp_team_order: dict[str, int] = {}
        self._opp_next_slot: int = 0

    def new_battle(self):
        """Reset state for a new battle."""
        self._my_team_order = {}
        self._opp_team_order = {}
        self._opp_next_slot = 0
        self._opp_fill_rng = _random.Random(42)
        self._opp_fill_cache: list[Pokemon] | None = None

    def translate(self, battle) -> BattleState:
        """Convert a poke-env Battle to our engine BattleState."""
        p1 = self._translate_my_side(battle)
        p2 = self._translate_opp_side(battle)
        weather, weather_turns = self._translate_weather(battle)

        return BattleState(
            p1=p1, p2=p2,
            turn=battle.turn,
            weather=weather,
            weather_turns=weather_turns,
            rng=random.Random(),
        )

    # ============================================================
    # OUR SIDE (full information)
    # ============================================================

    def _translate_my_side(self, battle) -> PlayerState:
        team_mons = list(battle.team.values())

        # establish stable ordering on first call
        if not self._my_team_order:
            for i, mon in enumerate(team_mons):
                self._my_team_order[mon.species] = i

        # build team in stable order
        team = [None] * len(team_mons)
        active_idx = 0
        for mon in team_mons:
            idx = self._my_team_order.get(mon.species, len(team) - 1)
            pokemon = self._translate_pokemon(mon, is_opponent=False)
            team[idx] = pokemon
            if mon.active:
                active_idx = idx

        # fill any None gaps
        team = [p if p is not None else self._make_dummy_fainted() for p in team]

        side = self._translate_side_conditions(battle.side_conditions)

        return PlayerState(team=team, active_index=active_idx, side=side)

    # ============================================================
    # OPPONENT SIDE (partial information)
    # ============================================================

    def _translate_opp_side(self, battle) -> PlayerState:
        opp_mons = list(battle.opponent_team.values())

        # assign stable slot indices to revealed pokemon
        for mon in opp_mons:
            if mon.species not in self._opp_team_order:
                self._opp_team_order[mon.species] = self._opp_next_slot
                self._opp_next_slot += 1

        # fill unrevealed slots with realistic common OU Pokemon
        revealed_species = {self._mapper.species_id(mon.species) for mon in opp_mons}
        fill_mons = self._get_fill_pokemon(revealed_species)

        team: list[Pokemon] = [None] * 6  # type: ignore
        active_idx = 0
        for mon in opp_mons:
            idx = self._opp_team_order[mon.species]
            team[idx] = self._translate_pokemon(mon, is_opponent=True)
            if mon.active:
                active_idx = idx

        # fill empty slots with common OU mons
        fill_idx = 0
        for i in range(6):
            if team[i] is None:
                if fill_idx < len(fill_mons):
                    team[i] = fill_mons[fill_idx]
                    fill_idx += 1
                else:
                    team[i] = self._make_dummy_alive()

        side = self._translate_side_conditions(battle.opponent_side_conditions)

        return PlayerState(team=team, active_index=active_idx, side=side)

    # ============================================================
    # POKEMON TRANSLATION
    # ============================================================

    def _translate_pokemon(self, poke_mon, is_opponent: bool) -> Pokemon:
        """Convert a poke-env Pokemon to our engine Pokemon."""
        species_id = self._mapper.species_id(poke_mon.species)
        if species_id is None:
            return self._make_dummy_fainted()

        pkmn_data = self._mapper.pokemon_data(species_id)
        species = PokemonSpecies.from_dict(pkmn_data)

        # move slots
        move_slots = []
        for move_id_str, poke_move in poke_mon.moves.items():
            mid = self._mapper.move_id(move_id_str)
            if mid is not None:
                mdata = self._mapper.move_data(mid)
                tmpl = MoveTemplate.from_dict(mdata)
                pp = poke_move.current_pp if not is_opponent else tmpl.pp
                move_slots.append(MoveSlot(template=tmpl, current_pp=pp))

        # pad to 4 slots with dummies if we don't know all moves
        while len(move_slots) < 4:
            if is_opponent:
                # guess a STAB move for unknown slots
                stub = self._guess_move(species)
                move_slots.append(MoveSlot(template=stub))
            else:
                move_slots.append(MoveSlot(template=_DUMMY_MOVE))

        # hp
        if is_opponent:
            max_hp = calc_stat(species.base_stats["hp"], is_hp=True)
            current_hp = max(0, int(poke_mon.current_hp_fraction * max_hp))
        else:
            # for our side, poke-env gives exact HP
            current_hp = poke_mon.current_hp or 0

        # status
        status = None
        if poke_mon.status is not None:
            status = STATUS_FROM_SHOWDOWN.get(poke_mon.status.name)

        # stat stages
        stat_stages = {
            "attack": 0, "defense": 0,
            "special_attack": 0, "special_defense": 0,
            "speed": 0, "accuracy": 0, "evasion": 0,
        }
        for sd_key, engine_key in STAT_FROM_SHOWDOWN.items():
            if sd_key in poke_mon.boosts:
                stat_stages[engine_key] = poke_mon.boosts[sd_key]

        # item
        item = None
        if not is_opponent and poke_mon.item:
            item = _normalize(poke_mon.item)
        elif is_opponent:
            item = "leftovers"  # safe default for gen2 OU

        pokemon = Pokemon(
            species=species,
            move_slots=move_slots,
            current_hp=current_hp,
            item=item,
            status=status,
            stat_stages=stat_stages,
        )
        return pokemon

    def _guess_move(self, species: PokemonSpecies) -> MoveTemplate:
        """Guess a reasonable move for an unrevealed opponent slot.

        Picks the highest-power STAB move from the species' learnset.
        """
        best_id = None
        best_power = 0
        for mid in species.learnset:
            mdata = self._data.moves.get(mid)
            if mdata is None or mdata["power"] == 0:
                continue
            power = mdata["power"]
            if mdata["type"] in species.types:
                power = int(power * 1.5)  # STAB bonus for ranking
            if power > best_power:
                best_power = power
                best_id = mid

        if best_id is not None:
            return MoveTemplate.from_dict(self._data.moves[best_id])
        return _DUMMY_MOVE

    def _get_fill_pokemon(self, revealed_species: set[int | None]) -> list[Pokemon]:
        """Build a list of common OU Pokemon to fill unrevealed opponent slots.

        Excludes species already revealed. Cached per battle so the guesses
        are consistent across turns.
        """
        if self._opp_fill_cache is not None:
            # filter out any that have since been revealed
            return [p for p in self._opp_fill_cache
                    if p.species.id not in revealed_species]

        # pick from common pool, excluding revealed species
        pool = [entry for entry in _COMMON_OU_POOL
                if entry[0] not in revealed_species]
        self._opp_fill_rng.shuffle(pool)

        fill = []
        for species_id, move_ids, item in pool[:5]:
            pkmn_data = self._mapper.pokemon_data(species_id)
            if pkmn_data is None:
                continue
            species = PokemonSpecies.from_dict(pkmn_data)
            move_slots = []
            for mid in move_ids:
                mdata = self._mapper.move_data(mid)
                if mdata:
                    move_slots.append(MoveSlot(template=MoveTemplate.from_dict(mdata)))
            while len(move_slots) < 4:
                move_slots.append(MoveSlot(template=_DUMMY_MOVE))
            mon = Pokemon(species=species, move_slots=move_slots, item=item)
            fill.append(mon)

        self._opp_fill_cache = fill
        return fill

    def _make_dummy_fainted(self) -> Pokemon:
        """Create a dummy fainted pokemon for unrevealed team slots."""
        p = Pokemon(species=_DUMMY_SPECIES, move_slots=[MoveSlot(template=_DUMMY_MOVE)])
        p.current_hp = 0
        return p

    def _make_dummy_alive(self) -> Pokemon:
        """Create a dummy alive pokemon for unrevealed opponent slots.

        Assumes full health so the search doesn't overestimate our advantage.
        """
        return Pokemon(species=_DUMMY_SPECIES, move_slots=[MoveSlot(template=_DUMMY_MOVE)])

    # ============================================================
    # SIDE CONDITIONS & WEATHER
    # ============================================================

    def _translate_side_conditions(self, conditions) -> SideConditions:
        side = SideConditions()
        for cond, turn in conditions.items():
            name = cond.name if hasattr(cond, "name") else str(cond)
            if "SPIKES" in name.upper():
                side.spikes = True
            elif "REFLECT" in name.upper():
                side.reflect_turns = 3  # assume midpoint
            elif "LIGHT_SCREEN" in name.upper() or "LIGHTSCREEN" in name.upper():
                side.light_screen_turns = 3
        return side

    def _translate_weather(self, battle) -> tuple[str | None, int]:
        if not battle.weather:
            return None, 0
        for w, turn in battle.weather.items():
            name = w.name if hasattr(w, "name") else str(w)
            weather = WEATHER_FROM_SHOWDOWN.get(name.upper())
            if weather:
                return weather, max(1, battle.turn - turn)
        return None, 0

    # ============================================================
    # ACTION MAPPING (engine action -> poke-env order)
    # ============================================================

    def action_to_order(self, action_int: int, battle, player_cls):
        """Convert our engine action int to a poke-env BattleOrder.

        action_int: 0-3 = move slot, 4-9 = switch to team[i]
        player_cls: the Player instance (for create_order)
        """
        if action_int < 4:
            # find the matching move in available_moves
            target_move = self._find_move_by_slot(action_int, battle)
            if target_move is not None:
                return player_cls.create_order(target_move)
            # fallback: first available move
            if battle.available_moves:
                return player_cls.create_order(battle.available_moves[0])
            return player_cls.choose_random_move(battle)
        else:
            team_index = action_int - 4
            target_mon = self._find_switch_target(team_index, battle)
            if target_mon is not None:
                return player_cls.create_order(target_mon)
            # fallback: first available switch
            if battle.available_switches:
                return player_cls.create_order(battle.available_switches[0])
            return player_cls.choose_random_move(battle)

    def _find_move_by_slot(self, slot_idx: int, battle):
        """Find the poke-env Move matching our move slot index."""
        if not battle.active_pokemon:
            return None
        # get our engine's view of move ordering
        active_moves = list(battle.active_pokemon.moves.values())
        if slot_idx < len(active_moves):
            target_id = active_moves[slot_idx].id
            for move in battle.available_moves:
                if move.id == target_id:
                    return move
        return None

    def _find_switch_target(self, team_index: int, battle):
        """Find the poke-env Pokemon matching our team index."""
        # reverse lookup: team_index -> species name
        species = None
        for sp, idx in self._my_team_order.items():
            if idx == team_index:
                species = sp
                break
        if species is None:
            return None
        for mon in battle.available_switches:
            if mon.species == species:
                return mon
        return None
