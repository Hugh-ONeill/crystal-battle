# poke-env Player that uses our search engine for decisions
#
# Usage:
#   # local testing (requires local Showdown server)
#   python showdown/player.py --local --depth 2 --n-games 10
#
#   # connect to PokeAgent server
#   python showdown/player.py --server pokeagent --username PAC-Crystal --password xxx

from __future__ import annotations

import asyncio
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from poke_env.player import Player
from poke_env import AccountConfiguration, ServerConfiguration

from engine.types import TypeChart
from engine.data_loader import DataStore
from engine.actions import Switch, UseMove, Struggle

from showdown.state_translator import StateTranslator
from showdown.team_export import team_to_showdown


# ============================================================
# SERVER CONFIGURATIONS
# ============================================================

POKEAGENT_SERVER = ServerConfiguration(
    "wss://pokeagentshowdown.com/showdown/websocket",
    "https://play.pokemonshowdown.com/action.php?",
)

LOCAL_SERVER = ServerConfiguration(
    "ws://localhost:8000/showdown/websocket",
    "https://play.pokemonshowdown.com/action.php?",
)


# ============================================================
# SEARCH PLAYER
# ============================================================

class SearchPlayer(Player):
    """poke-env Player that uses our lookahead search for decisions."""

    def __init__(self, search_depth: int = 2, **kwargs):
        super().__init__(**kwargs)
        self._data = DataStore()
        self._tc = TypeChart.load()
        self._translator = StateTranslator(self._data, self._tc)
        self._depth = search_depth

        # import search components
        from training.lookahead import LookaheadAgent
        from training.opponent_model import OpponentPredictor
        from training.baselines import SmartAgent
        import torch

        # use SmartAgent as the base policy for value estimates
        self._smart = SmartAgent(self._tc, seed=42)

        # opponent model (if available)
        opp_model_path = Path(__file__).parent.parent / "opp_model.pt"
        self._opp_model = OpponentPredictor()
        if opp_model_path.exists():
            self._opp_model.load_state_dict(
                torch.load(opp_model_path, map_location="cpu", weights_only=True))
            print(f"  Loaded opponent model from {opp_model_path}")
        else:
            print(f"  No opponent model found at {opp_model_path}, using random predictions")
        self._opp_model.eval()

        self._agent: LookaheadAgent | None = None

    def choose_move(self, battle):
        """Called by poke-env each turn to pick an action."""
        # reset translator on new battle
        if battle.turn <= 1:
            self._translator.new_battle()

        # translate poke-env state to our engine
        engine_state = self._translator.translate(battle)

        # use heuristic search
        action_int = self._heuristic_search(engine_state, battle)

        # convert back to poke-env order
        return self._translator.action_to_order(action_int, battle, self)

    def _heuristic_search(self, state, battle) -> int:
        """Run recursive lookahead search on the translated state."""
        from training.lookahead import (
            _int_to_action, _copy_battle, _handle_forced_switches_simple,
            _sample_top_actions,
        )
        from engine.turn_engine import resolve_turn
        from gym_env.obs_builder import build_observation
        import numpy as np

        p1 = state.p1
        p2 = state.p2
        mask = np.array(p1.valid_action_mask(p2, type_chart=self._tc), dtype=bool)
        valid = [i for i in range(10) if mask[i]]

        if not valid:
            return 0
        if len(valid) == 1:
            return valid[0]

        # predict opponent actions
        obs = build_observation(
            p1, p2, self._tc, turn=state.turn,
            weather=state.weather, weather_turns=state.weather_turns,
        )
        opp_probs, _ = self._opp_model.predict_single(obs, None)
        opp_mask = np.array(p2.valid_action_mask(p1, type_chart=self._tc), dtype=bool)
        opp_probs = opp_probs * opp_mask
        s = opp_probs.sum()
        if s > 0:
            opp_probs /= s
        else:
            opp_probs = opp_mask.astype(np.float32)
            if opp_probs.sum() > 0:
                opp_probs /= opp_probs.sum()

        action_values = self._search(
            state, opp_probs, valid, depth=self._depth,
        )
        return max(action_values, key=action_values.get)

    def _search(self, state, opp_probs, valid_actions, depth):
        """Recursive expectimax search."""
        from training.lookahead import (
            _int_to_action, _copy_battle, _handle_forced_switches_simple,
            _sample_top_actions,
        )
        from engine.turn_engine import resolve_turn
        from gym_env.obs_builder import build_observation
        import numpy as np

        opp_actions = _sample_top_actions(opp_probs, 3)
        action_values = {}

        for my_action_int in valid_actions:
            my_action = _int_to_action(my_action_int, state.p1)
            total_value = 0.0
            total_weight = 0.0

            for opp_action_int, opp_weight in opp_actions:
                opp_action = _int_to_action(opp_action_int, state.p2)
                sim = _copy_battle(state)

                try:
                    resolve_turn(sim, my_action, opp_action, self._tc)
                    _handle_forced_switches_simple(sim, self._tc)
                except Exception:
                    continue

                if depth <= 1 or sim.is_over:
                    value = self._evaluate(sim)
                else:
                    # recurse deeper
                    sim_mask = np.array(
                        sim.p1.valid_action_mask(sim.p2, type_chart=self._tc),
                        dtype=bool,
                    )
                    sim_valid = [i for i in range(10) if sim_mask[i]]

                    if not sim_valid:
                        value = self._evaluate(sim)
                    else:
                        sim_obs = build_observation(
                            sim.p1, sim.p2, self._tc, turn=sim.turn,
                            weather=sim.weather, weather_turns=sim.weather_turns,
                        )
                        sim_opp_probs, _ = self._opp_model.predict_single(sim_obs, None)
                        sim_opp_mask = np.array(
                            sim.p2.valid_action_mask(sim.p1, type_chart=self._tc),
                            dtype=bool,
                        )
                        sim_opp_probs = sim_opp_probs * sim_opp_mask
                        s = sim_opp_probs.sum()
                        if s > 0:
                            sim_opp_probs /= s
                        else:
                            sim_opp_probs = sim_opp_mask.astype(np.float32)
                            if sim_opp_probs.sum() > 0:
                                sim_opp_probs /= sim_opp_probs.sum()

                        child_values = self._search(
                            sim, sim_opp_probs, sim_valid, depth=depth - 1,
                        )
                        value = max(child_values.values()) if child_values else 0.0

                total_value += value * opp_weight
                total_weight += opp_weight

            action_values[my_action_int] = (
                total_value / total_weight if total_weight > 0 else -1.0
            )

        return action_values

    def _evaluate(self, state) -> float:
        """Evaluate a battle position using rich Gen 2 heuristic."""
        from showdown.evaluate import evaluate
        return evaluate(state, self._tc)


# ============================================================
# HEURISTIC PLAYER (wraps SmartAgent / MaxDamageAgent)
# ============================================================

class HeuristicPlayer(Player):
    """poke-env Player that uses our heuristic agents (Smart/MaxDmg)."""

    def __init__(self, agent_type: str = "smart", **kwargs):
        super().__init__(**kwargs)
        self._data = DataStore()
        self._tc = TypeChart.load()
        self._translator = StateTranslator(self._data, self._tc)

        from training.baselines import SmartAgent, MaxDamageAgent
        if agent_type == "smart":
            self._agent = SmartAgent(self._tc, seed=42)
        else:
            self._agent = MaxDamageAgent(self._tc)

    def choose_move(self, battle):
        if battle.turn <= 1:
            self._translator.new_battle()

        engine_state = self._translator.translate(battle)
        action = self._agent.act(engine_state.p1, engine_state.p2)

        # convert engine Action to action int
        from engine.actions import UseMove, Switch, Struggle
        if isinstance(action, UseMove):
            action_int = action.slot_index
        elif isinstance(action, Switch):
            action_int = 4 + action.team_index
        elif isinstance(action, Struggle):
            action_int = 0
        else:
            action_int = 0

        return self._translator.action_to_order(action_int, battle, self)


# ============================================================
# CLI
# ============================================================

async def main():
    parser = argparse.ArgumentParser(description="Crystal Battle Showdown Bot")
    parser.add_argument("--local", action="store_true", help="Use local Showdown server")
    parser.add_argument("--server", type=str, default="local",
                        choices=["local", "pokeagent", "showdown"],
                        help="Server to connect to")
    parser.add_argument("--username", type=str, default="CrystalBot")
    parser.add_argument("--password", type=str, default="")
    parser.add_argument("--depth", type=int, default=1)
    parser.add_argument("--n-games", type=int, default=10)
    parser.add_argument("--format", type=str, default="gen2randombattle",
                        help="Battle format (gen2ou, gen2randombattle, etc.)")
    parser.add_argument("--challenge", type=str, default=None,
                        help="Challenge a specific user instead of laddering")
    args = parser.parse_args()

    if args.local or args.server == "local":
        server_config = LOCAL_SERVER
    elif args.server == "pokeagent":
        server_config = POKEAGENT_SERVER
    else:
        from poke_env import ShowdownServerConfiguration
        server_config = ShowdownServerConfiguration

    # build a team (only for non-random formats)
    team_str = None
    if "random" not in args.format:
        from showdown.sample_teams import SAMPLE_TEAMS
        import random as rng_mod
        team_str = rng_mod.choice(SAMPLE_TEAMS)
        print(f"Team:\n{team_str}\n")

    player = SearchPlayer(
        search_depth=args.depth,
        account_configuration=AccountConfiguration(args.username, args.password),
        server_configuration=server_config,
        battle_format=args.format,
        team=team_str,
    )

    if args.challenge:
        print(f"Challenging {args.challenge}...")
        await player.send_challenges(args.challenge, n_challenges=args.n_games)
    elif args.local or args.server == "local":
        # local mode: create a random opponent and play against it
        from poke_env.player import RandomPlayer
        opponent = RandomPlayer(
            account_configuration=AccountConfiguration("RandomOpp", ""),
            server_configuration=server_config,
            battle_format=args.format,
        )
        print(f"Playing {args.n_games} games vs RandomPlayer...")
        await player.battle_against(opponent, n_battles=args.n_games)
    else:
        print(f"Laddering {args.n_games} games...")
        await player.ladder(args.n_games)

    # print results
    wins = sum(1 for b in player.battles.values() if b.won)
    total = len(player.battles)
    print(f"\nResults: {wins}/{total} ({wins/total*100:.1f}%)" if total else "No games played")


if __name__ == "__main__":
    asyncio.run(main())
