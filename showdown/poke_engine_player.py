# poke-env Player powered by poke-engine MCTS search
# uses poke-engine for accurate Gen 2 battle simulation with items
#
# Usage:
#   python showdown/poke_engine_player.py --local --search-ms 1000

from __future__ import annotations

import asyncio
import argparse
import concurrent.futures
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import poke_engine as pe
from poke_env.player import Player
from poke_env import AccountConfiguration, ServerConfiguration

from showdown.name_mapping import NameMapper, _normalize, STATUS_FROM_SHOWDOWN, STAT_FROM_SHOWDOWN
from showdown.usage_stats import predict_opponent_team, get_likely_moveset, get_likely_item
from showdown.chaos_stats import ChaosStats, RevealedMon


# ============================================================
# SERVER CONFIGURATIONS
# ============================================================

LOCAL_SERVER = ServerConfiguration(
    "ws://localhost:8000/showdown/websocket",
    "https://play.pokemonshowdown.com/action.php?",
)

POKEAGENT_SERVER = ServerConfiguration(
    "wss://pokeagentshowdown.com/showdown/websocket",
    "https://play.pokemonshowdown.com/action.php?",
)


# ============================================================
# STATE TRANSLATION (poke-env -> poke-engine)
# ============================================================

class PokeEngineTranslator:
    """Translates poke-env Battle objects to poke-engine State."""

    def __init__(self):
        self._my_team_order: dict[str, int] = {}
        self._opp_team_order: dict[str, int] = {}
        self._opp_next_slot: int = 0

    def new_battle(self):
        self._my_team_order = {}
        self._opp_team_order = {}
        self._opp_next_slot = 0

    def translate(self, battle) -> pe.State:
        """Convert poke-env Battle to poke-engine State."""
        side_one = self._translate_my_side(battle)
        side_two = self._translate_opp_side(battle)

        weather = pe.Weather.NONE
        if battle.weather:
            for w in battle.weather:
                wname = w.name if hasattr(w, "name") else str(w)
                wname = wname.upper()
                if "SUN" in wname:
                    weather = pe.Weather.SUN
                elif "RAIN" in wname:
                    weather = pe.Weather.RAIN
                elif "SAND" in wname:
                    weather = pe.Weather.SAND
                elif "HAIL" in wname:
                    weather = pe.Weather.HAIL
            weather_turns = 5  # approximate
        else:
            weather_turns = 0

        return pe.State(
            side_one=side_one, side_two=side_two,
            weather=weather, weather_turns_remaining=weather_turns,
            terrain=pe.Terrain.NONE, terrain_turns_remaining=0,
            trick_room=False, trick_room_turns_remaining=0,
            team_preview=False,
        )

    def _translate_my_side(self, battle) -> pe.Side:
        team_mons = list(battle.team.values())
        if not self._my_team_order:
            for i, mon in enumerate(team_mons):
                self._my_team_order[mon.species] = i

        # active pokemon must be at index 0 (active_index is read-only)
        active_mon = None
        bench = []
        for mon in team_mons:
            if mon.active:
                active_mon = mon
            else:
                bench.append(mon)

        pokemon = []
        if active_mon:
            pokemon.append(self._translate_pokemon(active_mon, is_opponent=False))
        for mon in bench:
            pokemon.append(self._translate_pokemon(mon, is_opponent=False))

        while len(pokemon) < 6:
            pokemon.append(pe.Pokemon.create_fainted())

        side = pe.Side(pokemon=pokemon[:6])
        # note: boosts and side conditions are read-only on Side object
        # they default to 0 -- for accurate mid-game state we'd need from_string()
        return side

    def _translate_opp_side(self, battle) -> pe.Side:
        opp_mons = list(battle.opponent_team.values())

        # active first, bench after
        active_mon = None
        bench = []
        revealed_names = set()
        for mon in opp_mons:
            revealed_names.add(_normalize(mon.species))
            if mon.active:
                active_mon = mon
            else:
                bench.append(mon)

        pokemon = []
        if active_mon:
            pokemon.append(self._translate_pokemon(active_mon, is_opponent=True))
        for mon in bench:
            pokemon.append(self._translate_pokemon(mon, is_opponent=True))

        # unrevealed slots: fainted dummies
        # (usage-predicted fills hurt MCTS -- the wrong predictions mislead search
        # more than empty slots do, because MCTS explores switching into them)
        while len(pokemon) < 6:
            pokemon.append(pe.Pokemon.create_fainted())

        side = pe.Side(pokemon=pokemon[:6])
        return side

    _gen2_pokedex = None

    def _build_predicted_pokemon(self, species_norm: str) -> pe.Pokemon:
        """Build a poke-engine Pokemon from usage stats prediction."""
        if PokeEngineTranslator._gen2_pokedex is None:
            from poke_env.data.gen_data import GenData
            PokeEngineTranslator._gen2_pokedex = GenData.from_gen(2).pokedex
        pokedex = PokeEngineTranslator._gen2_pokedex

        # get base stats from poke-env's pokedex
        poke_data = pokedex.get(species_norm, {})
        base_stats = poke_data.get("baseStats", {})

        # compute gen2 stats at level 100 with perfect DVs
        def calc(base, is_hp=False):
            core = ((base + 15) * 2 + 64) * 100 // 100
            return core + 110 if is_hp else core + 5

        hp_stat = calc(base_stats.get("hp", 80), is_hp=True)
        atk_stat = calc(base_stats.get("atk", 80))
        def_stat = calc(base_stats.get("def", 80))
        spa_stat = calc(base_stats.get("spa", 80))
        spd_stat = calc(base_stats.get("spd", 80))
        spe_stat = calc(base_stats.get("spe", 80))

        # types
        types_list = poke_data.get("types", ["Normal"])
        types = tuple(t.lower() for t in types_list)
        if len(types) < 2:
            types = (types[0], "typeless")

        # moves from usage stats
        move_names = get_likely_moveset(species_norm)
        moves = []
        for mname in move_names[:4]:
            moves.append(pe.Move(id=mname, pp=16))
        while len(moves) < 4:
            moves.append(pe.Move(id="splash", pp=1))

        item = get_likely_item(species_norm)

        return pe.Pokemon(
            id=species_norm, level=100,
            hp=hp_stat, maxhp=hp_stat,
            attack=atk_stat, defense=def_stat,
            special_attack=spa_stat, special_defense=spd_stat,
            speed=spe_stat,
            types=types,
            ability="noability",
            item=item,
            status="none",
            moves=moves,
        )

    def _translate_pokemon(self, poke_mon, is_opponent: bool) -> pe.Pokemon:
        species = _normalize(poke_mon.species)
        if poke_mon.fainted:
            return pe.Pokemon.create_fainted()

        # moves
        moves = []
        for move_id, move_obj in poke_mon.moves.items():
            mid = _normalize(move_id)
            pp = move_obj.current_pp if not is_opponent else move_obj.max_pp
            moves.append(pe.Move(id=mid, pp=pp))
        while len(moves) < 4:
            moves.append(pe.Move(id="splash", pp=1))

        # stats
        if is_opponent:
            # estimate stats from base stats at level 100
            # poke-env gives base_stats, we need computed stats
            bs = poke_mon.base_stats
            hp_stat = ((bs.get("hp", 80) + 15) * 2 + 64) + 110
            atk_stat = ((bs.get("atk", 80) + 15) * 2 + 64) + 5
            def_stat = ((bs.get("def", 80) + 15) * 2 + 64) + 5
            spa_stat = ((bs.get("spa", 80) + 15) * 2 + 64) + 5
            spd_stat = ((bs.get("spd", 80) + 15) * 2 + 64) + 5
            spe_stat = ((bs.get("spe", 80) + 15) * 2 + 64) + 5
            current_hp = max(1, int(poke_mon.current_hp_fraction * hp_stat))
        else:
            hp_stat = poke_mon.max_hp or 300
            current_hp = poke_mon.current_hp or 0
            stats = poke_mon.stats or {}
            atk_stat = stats.get("atk", 200)
            def_stat = stats.get("def", 200)
            spa_stat = stats.get("spa", 200)
            spd_stat = stats.get("spd", 200)
            spe_stat = stats.get("spe", 200)

        # types
        types = []
        if poke_mon.type_1:
            types.append(poke_mon.type_1.name.lower())
        if poke_mon.type_2:
            types.append(poke_mon.type_2.name.lower())
        if not types:
            types = ["normal"]
        while len(types) < 2:
            types.append("typeless")

        # status -- poke-engine needs full lowercase names
        _STATUS_MAP = {
            "BRN": "burn", "PAR": "paralysis", "SLP": "sleep",
            "FRZ": "freeze", "PSN": "poison", "TOX": "toxic",
        }
        status = "none"
        if poke_mon.status:
            status = _STATUS_MAP.get(poke_mon.status.name, "none")

        # item (opponent: guess leftovers, us: use actual)
        item = "none"
        if not is_opponent:
            if poke_mon.item:
                item = _normalize(poke_mon.item)
        else:
            item = "leftovers"  # safe default for gen2 OU

        return pe.Pokemon(
            id=species, level=100,
            hp=current_hp, maxhp=hp_stat,
            attack=atk_stat, defense=def_stat,
            special_attack=spa_stat, special_defense=spd_stat,
            speed=spe_stat,
            types=tuple(types),
            ability="noability",
            item=item,
            status=status,
            moves=moves[:4],
        )


# ============================================================
# POKE-ENGINE SEARCH PLAYER
# ============================================================

class PokeEnginePlayer(Player):
    """poke-env Player using poke-engine MCTS for decisions."""

    def __init__(self, search_ms: int = 1000, **kwargs):
        super().__init__(**kwargs)
        self._translator = PokeEngineTranslator()
        self._search_ms = search_ms
        self._executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)

    async def choose_move(self, battle):
        if battle.turn <= 1:
            self._translator.new_battle()

        # forced switch: pick best available
        if battle.force_switch:
            if battle.available_switches:
                best = max(battle.available_switches,
                           key=lambda p: p.current_hp_fraction)
                return self.create_order(best)
            return self.choose_default_move()

        # translate state for poke-engine
        try:
            pe_state = self._translator.translate(battle)
            # run MCTS in thread pool so async event loop stays responsive
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(
                self._executor,
                pe.monte_carlo_tree_search,
                pe_state, self._search_ms,
            )
        except Exception as e:
            print(f"  Search error: {e}")
            return self.choose_random_move(battle)

        # find best move from MCTS results
        best = max(result.side_one, key=lambda x: x.visits)
        move_choice = best.move_choice

        # map poke-engine move choice back to poke-env order
        return self._map_move_to_order(move_choice, battle)

    def _map_move_to_order(self, move_choice: str, battle):
        """Map poke-engine move name to poke-env BattleOrder."""
        choice_norm = _normalize(move_choice)

        # check if it's a switch
        if choice_norm.startswith("switch"):
            # poke-engine returns "switch <pokemon_id>"
            # extract the pokemon name
            parts = move_choice.split()
            if len(parts) >= 2:
                target_species = _normalize(parts[1])
                for mon in battle.available_switches:
                    if _normalize(mon.species) == target_species:
                        return self.create_order(mon)
            # fallback
            if battle.available_switches:
                return self.create_order(battle.available_switches[0])

        # it's a move -- match by normalized name
        for move in battle.available_moves:
            if _normalize(move.id) == choice_norm:
                return self.create_order(move)

        # fuzzy fallback
        for move in battle.available_moves:
            if choice_norm in _normalize(move.id) or _normalize(move.id) in choice_norm:
                return self.create_order(move)

        # give up, pick random
        return self.choose_random_move(battle)


# ============================================================
# MULTI-SAMPLE SEARCH PLAYER
# ============================================================

class MultiSamplePlayer(Player):
    """poke-env Player that runs MCTS over multiple sampled opponent teams.

    For each decision, generates N plausible opponent teams (varying the
    unrevealed slots using usage stats), runs MCTS on each, and picks
    the move with the best average performance across all samples.
    """

    def __init__(self, search_ms: int = 500, n_samples: int = 3, **kwargs):
        super().__init__(**kwargs)
        self._translator = PokeEngineTranslator()
        self._search_ms = search_ms
        self._n_samples = n_samples
        self._executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)
        self._rng = __import__("random").Random(42)
        self._chaos = ChaosStats()

    async def choose_move(self, battle):
        if battle.turn <= 1:
            self._translator.new_battle()

        if battle.force_switch:
            if battle.available_switches:
                best = max(battle.available_switches,
                           key=lambda p: p.current_hp_fraction)
                return self.create_order(best)
            return self.choose_default_move()

        try:
            # build our side (same every sample)
            my_side = self._translator._translate_my_side(battle)

            # get revealed opponent mons and build RevealedMon objects
            opp_mons = list(battle.opponent_team.values())
            revealed = {}
            active_mon = None
            bench_mons = []
            for mon in opp_mons:
                norm = _normalize(mon.species)
                known_moves = [_normalize(m) for m in mon.moves.keys()]
                known_item = _normalize(mon.item) if mon.item else None
                revealed[norm] = RevealedMon(
                    norm, known_moves=known_moves,
                    known_item=known_item, hp_frac=mon.current_hp_fraction,
                )
                pe_mon = self._translator._translate_pokemon(mon, is_opponent=True)
                if mon.active:
                    active_mon = pe_mon
                else:
                    bench_mons.append(pe_mon)

            n_revealed = (1 if active_mon else 0) + len(bench_mons)
            n_fill = 6 - n_revealed

            # generate multiple opponent team samples
            states = []
            for _ in range(self._n_samples):
                opp_pokemon = []
                if active_mon:
                    opp_pokemon.append(active_mon)
                opp_pokemon.extend(bench_mons)

                if n_fill > 0:
                    fill = self._sample_fill(revealed, n_fill)
                    opp_pokemon.extend(fill)

                while len(opp_pokemon) < 6:
                    opp_pokemon.append(pe.Pokemon.create_fainted())

                opp_side = pe.Side(pokemon=opp_pokemon[:6])

                weather = pe.Weather.NONE
                weather_turns = 0
                if battle.weather:
                    for w in battle.weather:
                        wname = w.name if hasattr(w, "name") else str(w)
                        wname = wname.upper()
                        if "SUN" in wname:
                            weather = pe.Weather.SUN
                        elif "RAIN" in wname:
                            weather = pe.Weather.RAIN
                        elif "SAND" in wname:
                            weather = pe.Weather.SAND
                    weather_turns = 5

                state = pe.State(
                    side_one=my_side, side_two=opp_side,
                    weather=weather, weather_turns_remaining=weather_turns,
                    terrain=pe.Terrain.NONE, terrain_turns_remaining=0,
                    trick_room=False, trick_room_turns_remaining=0,
                    team_preview=False,
                )
                states.append(state)

            # run MCTS on each sample
            loop = asyncio.get_event_loop()
            move_scores = {}  # move_choice -> (total_score, total_visits)

            for state in states:
                try:
                    result = await loop.run_in_executor(
                        self._executor,
                        pe.monte_carlo_tree_search,
                        state, self._search_ms // self._n_samples,
                    )
                    for r in result.side_one:
                        key = r.move_choice
                        if key not in move_scores:
                            move_scores[key] = [0.0, 0]
                        move_scores[key][0] += r.total_score
                        move_scores[key][1] += r.visits
                except Exception as e:
                    pass  # skip failed samples

            if not move_scores:
                return self.choose_random_move(battle)

            # pick move with highest average score (total_score / visits)
            best_move = max(move_scores.keys(),
                            key=lambda k: move_scores[k][0] / max(move_scores[k][1], 1))

        except Exception as e:
            print(f"  Multi-sample error: {e}")
            return self.choose_random_move(battle)

        return self._map_move_to_order(best_move, battle)

    def _sample_fill(self, revealed: dict[str, RevealedMon], n_fill: int) -> list:
        """Sample unrevealed Pokemon from chaos stats with randomization.

        Each call produces a slightly different team by:
        1. Getting top candidates from chaos stats (weighted by teammates)
        2. Randomly dropping 1-2 from the top and pulling from deeper in the pool
        3. Varying movesets using chaos move probabilities
        """
        predicted = self._chaos.predict_team(revealed, n_fill=n_fill + 4)
        candidates = [p for p in predicted]

        # randomize: swap some top picks with deeper ones
        if len(candidates) > n_fill:
            # randomly drop 0-2 from top and shuffle
            n_drop = self._rng.randint(0, min(2, len(candidates) - n_fill))
            if n_drop > 0:
                drop_indices = self._rng.sample(range(min(n_fill, len(candidates))), n_drop)
                candidates = [c for i, c in enumerate(candidates) if i not in drop_indices]

        fill = []
        for pred in candidates[:n_fill]:
            try:
                # use chaos stats for moveset (with some randomization)
                mon = self._translator._build_predicted_pokemon(pred.species)
                fill.append(mon)
            except Exception:
                continue
        return fill

    def _map_move_to_order(self, move_choice: str, battle):
        """Map poke-engine move name to poke-env BattleOrder."""
        choice_norm = _normalize(move_choice)

        if choice_norm.startswith("switch"):
            parts = move_choice.split()
            if len(parts) >= 2:
                target_species = _normalize(parts[1])
                for mon in battle.available_switches:
                    if _normalize(mon.species) == target_species:
                        return self.create_order(mon)
            if battle.available_switches:
                return self.create_order(battle.available_switches[0])

        for move in battle.available_moves:
            if _normalize(move.id) == choice_norm:
                return self.create_order(move)

        for move in battle.available_moves:
            if choice_norm in _normalize(move.id) or _normalize(move.id) in choice_norm:
                return self.create_order(move)

        return self.choose_random_move(battle)


# ============================================================
# CLI
# ============================================================

async def main():
    parser = argparse.ArgumentParser(description="poke-engine Search Bot")
    parser.add_argument("--local", action="store_true")
    parser.add_argument("--server", type=str, default="local",
                        choices=["local", "pokeagent", "showdown"])
    parser.add_argument("--username", type=str, default="PokeEngBot")
    parser.add_argument("--password", type=str, default="")
    parser.add_argument("--format", type=str, default="gen2ou")
    parser.add_argument("--search-ms", type=int, default=1000)
    parser.add_argument("--n-games", type=int, default=10)
    parser.add_argument("--n-samples", type=int, default=1,
                        help="Number of opponent team samples (1=single, 3+=multi-sample)")
    parser.add_argument("--challenge", type=str, default=None)
    parser.add_argument("--vs-smart", action="store_true",
                        help="Play against SmartAgent wrapper")
    args = parser.parse_args()

    if args.local or args.server == "local":
        server_config = LOCAL_SERVER
    elif args.server == "pokeagent":
        server_config = POKEAGENT_SERVER
    else:
        from poke_env import ShowdownServerConfiguration
        server_config = ShowdownServerConfiguration

    from showdown.sample_teams import SAMPLE_TEAMS
    import random as rng_mod
    team_str = rng_mod.choice(SAMPLE_TEAMS)
    print(f"Search time: {args.search_ms}ms per turn\n")

    if args.n_samples > 1:
        player = MultiSamplePlayer(
            search_ms=args.search_ms,
            n_samples=args.n_samples,
            account_configuration=AccountConfiguration(args.username, args.password),
            server_configuration=server_config,
            battle_format=args.format,
            team=team_str,
        )
        print(f"Multi-sample mode: {args.n_samples} samples\n")
    else:
        player = PokeEnginePlayer(
            search_ms=args.search_ms,
            account_configuration=AccountConfiguration(args.username, args.password),
            server_configuration=server_config,
            battle_format=args.format,
            team=team_str,
        )

    if args.vs_smart:
        from showdown.player import HeuristicPlayer
        opp = HeuristicPlayer(
            agent_type="smart",
            account_configuration=AccountConfiguration("SmartOpp2", ""),
            server_configuration=server_config,
            battle_format=args.format,
            team=rng_mod.choice(SAMPLE_TEAMS),
        )
        print(f"Playing {args.n_games} games vs SmartAgent...")
        await player.battle_against(opp, n_battles=args.n_games)
    elif args.challenge:
        await player.send_challenges(args.challenge, n_challenges=args.n_games)
    else:
        from poke_env.player import RandomPlayer
        opp = RandomPlayer(
            account_configuration=AccountConfiguration("RandOpp2", ""),
            server_configuration=server_config,
            battle_format=args.format,
            team=rng_mod.choice(SAMPLE_TEAMS),
        )
        print(f"Playing {args.n_games} games vs RandomPlayer...")
        await player.battle_against(opp, n_battles=args.n_games)

    wins = sum(1 for b in player.battles.values() if b.won)
    total = len(player.battles)
    print(f"\nResults: {wins}/{total} ({wins/total*100:.1f}%)")


if __name__ == "__main__":
    asyncio.run(main())
