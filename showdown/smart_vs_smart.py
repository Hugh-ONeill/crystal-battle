# Smart vs Smart matchup matrix: baseline team strength
# shows which teams are inherently strong regardless of bot quality
#
# Usage:
#   .venv/bin/python showdown/smart_vs_smart.py --games-per 3

import asyncio
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from poke_env import AccountConfiguration, ServerConfiguration
from showdown.sample_teams import SAMPLE_TEAMS
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


async def run_matchup(team_a, team_b, mid, n_games, timeout_per,
                      p1_type="smart", p2_type="smart"):
    from poke_env.player import RandomPlayer as PokeEnvRandom
    pre1 = p1_type[0].upper() + "A"
    pre2 = p2_type[0].upper() + "B"
    p1 = HeuristicPlayer(
        agent_type=p1_type,
        account_configuration=AccountConfiguration(f"{pre1}{mid}", ""),
        server_configuration=LOCAL,
        battle_format="gen2ou",
        team=team_a,
    ) if p1_type != "random" else PokeEnvRandom(
        account_configuration=AccountConfiguration(f"{pre1}{mid}", ""),
        server_configuration=LOCAL,
        battle_format="gen2ou",
        team=team_a,
    )
    p2 = HeuristicPlayer(
        agent_type=p2_type,
        account_configuration=AccountConfiguration(f"{pre2}{mid}", ""),
        server_configuration=LOCAL,
        battle_format="gen2ou",
        team=team_b,
    ) if p2_type != "random" else PokeEnvRandom(
        account_configuration=AccountConfiguration(f"{pre2}{mid}", ""),
        server_configuration=LOCAL,
        battle_format="gen2ou",
        team=team_b,
    )
    try:
        await asyncio.wait_for(
            p1.battle_against(p2, n_battles=n_games),
            timeout=timeout_per * n_games,
        )
        wins = sum(1 for b in p1.battles.values() if b.won)
        total = len(p1.battles)
        return wins, total
    except asyncio.TimeoutError:
        wins = sum(1 for b in p1.battles.values() if b.won)
        total = len(p1.battles)
        return wins, total


async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--games-per", type=int, default=3)
    parser.add_argument("--timeout-per", type=int, default=20)
    parser.add_argument("--p1", type=str, default="smart",
                        choices=["smart", "maxdmg", "random"])
    parser.add_argument("--p2", type=str, default="smart",
                        choices=["smart", "maxdmg", "random"])
    args = parser.parse_args()

    n_teams = len(SAMPLE_TEAMS)
    results = {}
    mid = 0

    print(f"{args.p1} vs {args.p2} -- {args.games_per} games per matchup\n")
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
            if i == j:
                # mirror match: skip, assume 50%
                print(f" {'50':>11s}%", end="", flush=True)
                row_wins += args.games_per // 2
                row_total += args.games_per
                mid += 1
                continue

            wins, total = await run_matchup(
                SAMPLE_TEAMS[i], SAMPLE_TEAMS[j],
                mid, args.games_per, args.timeout_per,
                p1_type=args.p1, p2_type=args.p2,
            )
            mid += 1
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

    # column averages (how often each team LOSES as opponent)
    print(f"\n{'Def AVG':14s}", end="")
    for j in range(n_teams):
        col_wins = sum(results.get((i, j), (0, 0))[0] for i in range(n_teams) if i != j)
        col_total = sum(results.get((i, j), (0, 0))[1] for i in range(n_teams) if i != j)
        if col_total > 0:
            print(f" {col_wins/col_total*100:11.0f}%", end="")
        else:
            print(f" {'?':>12s}", end="")
    print()


if __name__ == "__main__":
    asyncio.run(main())
