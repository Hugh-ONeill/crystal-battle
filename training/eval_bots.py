#!/usr/bin/env python3
# Evaluate heuristic bots against each other (no trained model needed)
# Usage: python training/eval_bots.py [--n-games 1000]

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import random

from engine.battle_state import BattleState
from engine.player_state import PlayerState
from engine.turn_engine import resolve_forced_switches, resolve_turn
from engine.types import TypeChart
from engine.data_loader import DataStore
from engine.actions import Switch

from training.baselines import MaxDamageAgent, RandomAgent, SmartAgent
from gym_env.team_builder import build_team

MAX_TURNS = 200


def play_game(
    p1_agent, p2_agent, data: DataStore, tc: TypeChart, rng: random.Random,
) -> int | None:
    """Play a single game, return winner (1 or 2) or None for draw."""
    team1 = build_team(data, rng=rng)
    team2 = build_team(data, rng=rng)
    battle = BattleState(
        p1=PlayerState(team=team1),
        p2=PlayerState(team=team2),
        rng=random.Random(rng.randint(0, 2**32)),
    )

    for _ in range(MAX_TURNS):
        if battle.is_over:
            break

        a1 = p1_agent.act(battle.p1, battle.p2)
        a2 = p2_agent.act(battle.p2, battle.p1)
        events = resolve_turn(battle, a1, a2, tc)

        # forced switches
        sw1, sw2 = None, None
        if battle.p1.must_switch:
            sw1_action = p1_agent.act(battle.p1, battle.p2)
            sw1 = sw1_action if isinstance(sw1_action, Switch) else None
            if sw1 is None:
                for i, p in enumerate(battle.p1.team):
                    if i != battle.p1.active_index and not p.is_fainted:
                        sw1 = Switch(team_index=i)
                        break
        if battle.p2.must_switch:
            sw2_action = p2_agent.act(battle.p2, battle.p1)
            sw2 = sw2_action if isinstance(sw2_action, Switch) else None
            if sw2 is None:
                for i, p in enumerate(battle.p2.team):
                    if i != battle.p2.active_index and not p.is_fainted:
                        sw2 = Switch(team_index=i)
                        break

        if sw1 or sw2:
            resolve_forced_switches(battle, sw1, sw2)

    return battle.winner


def main():
    parser = argparse.ArgumentParser(description="Evaluate bots vs each other")
    parser.add_argument("--n-games", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    tc = TypeChart.load()
    data = DataStore()
    rng = random.Random(args.seed)

    matchups = [
        ("Smart", "MaxDamage", SmartAgent(tc, seed=1), MaxDamageAgent(tc)),
        ("Smart", "Random", SmartAgent(tc, seed=1), RandomAgent(seed=2)),
        ("MaxDamage", "Random", MaxDamageAgent(tc), RandomAgent(seed=2)),
    ]

    for p1_name, p2_name, p1_agent, p2_agent in matchups:
        wins = {1: 0, 2: 0, None: 0}
        for i in range(args.n_games):
            game_rng = random.Random(rng.randint(0, 2**32))
            winner = play_game(p1_agent, p2_agent, data, tc, game_rng)
            wins[winner] = wins.get(winner, 0) + 1

        p1_wr = wins[1] / args.n_games
        p2_wr = wins[2] / args.n_games
        draws = wins[None] / args.n_games
        print(f"{p1_name} vs {p2_name}: "
              f"{p1_name} {p1_wr:.1%} | {p2_name} {p2_wr:.1%} | draws {draws:.1%} "
              f"({args.n_games} games)")


if __name__ == "__main__":
    main()
