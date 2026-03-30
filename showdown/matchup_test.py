# full matchup matrix: MCTS bot vs SmartAgent across all team combinations
#
# Usage:
#   .venv/bin/python showdown/matchup_test.py --games-per 2

import asyncio
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from poke_env import AccountConfiguration, ServerConfiguration
from showdown.sample_teams import SAMPLE_TEAMS
from showdown.poke_engine_player import PokeEnginePlayer
from showdown.player import HeuristicPlayer

LOCAL = ServerConfiguration(
    "ws://localhost:8000/showdown/websocket",
    "https://play.pokemonshowdown.com/action.php?",
)

NAMES = [
    "NidoGengar", "NidoMachamp", "JynxGengar", "VapExplosion",
    "ParaMarowak", "HeraExplosion", "DblElectric", "ThiefExegg",
    "AlaExplosion", "MoltExegg",
]


async def run_matchup(bot_team, opp_team, bot_id, n_games, search_ms, timeout_per,
                      opp_type="smart"):
    from poke_env.player import RandomPlayer as PokeEnvRandom
    bot = PokeEnginePlayer(
        search_ms=search_ms,
        account_configuration=AccountConfiguration(f"MB{bot_id}", ""),
        server_configuration=LOCAL,
        battle_format="gen2ou",
        team=bot_team,
    )
    if opp_type == "random":
        opp = PokeEnvRandom(
            account_configuration=AccountConfiguration(f"MO{bot_id}", ""),
            server_configuration=LOCAL,
            battle_format="gen2ou",
            team=opp_team,
        )
    else:
        opp = HeuristicPlayer(
            agent_type=opp_type,
            account_configuration=AccountConfiguration(f"MO{bot_id}", ""),
            server_configuration=LOCAL,
            battle_format="gen2ou",
            team=opp_team,
        )
    try:
        await asyncio.wait_for(
            bot.battle_against(opp, n_battles=n_games),
            timeout=timeout_per * n_games,
        )
        wins = sum(1 for b in bot.battles.values() if b.won)
        total = len(bot.battles)
        return wins, total
    except asyncio.TimeoutError:
        wins = sum(1 for b in bot.battles.values() if b.won)
        total = len(bot.battles)
        return wins, total


async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--games-per", type=int, default=2)
    parser.add_argument("--search-ms", type=int, default=500)
    parser.add_argument("--timeout-per", type=int, default=50,
                        help="timeout per game in seconds")
    parser.add_argument("--opp", type=str, default="smart",
                        choices=["smart", "maxdmg", "random"])
    args = parser.parse_args()

    n_teams = len(SAMPLE_TEAMS)
    results = {}
    match_id = 0

    # header
    print(f"{'':14s}", end="")
    for j in range(n_teams):
        print(f" {NAMES[j]:>12s}", end="")
    print("   AVG")
    print("-" * (14 + 13 * n_teams + 6))

    for i in range(n_teams):
        row_wins = 0
        row_total = 0
        print(f"{NAMES[i]:14s}", end="", flush=True)

        for j in range(n_teams):
            wins, total = await run_matchup(
                SAMPLE_TEAMS[i], SAMPLE_TEAMS[j],
                match_id, args.games_per, args.search_ms, args.timeout_per,
                opp_type=args.opp,
            )
            match_id += 1
            results[(i, j)] = (wins, total)
            row_wins += wins
            row_total += total

            if total > 0:
                pct = wins / total * 100
                print(f" {pct:11.0f}%", end="", flush=True)
            else:
                print(f" {'?':>12s}", end="", flush=True)

        avg = row_wins / row_total * 100 if row_total > 0 else 0
        print(f" {avg:5.0f}%")

    # column averages
    print(f"{'AVG':14s}", end="")
    total_wins = 0
    total_games = 0
    for j in range(n_teams):
        col_wins = sum(results.get((i, j), (0, 0))[0] for i in range(n_teams))
        col_total = sum(results.get((i, j), (0, 0))[1] for i in range(n_teams))
        total_wins += col_wins
        total_games += col_total
        if col_total > 0:
            print(f" {col_wins/col_total*100:11.0f}%", end="")
        else:
            print(f" {'?':>12s}", end="")
    print(f" {total_wins/total_games*100:5.0f}%" if total_games > 0 else "")


if __name__ == "__main__":
    asyncio.run(main())
