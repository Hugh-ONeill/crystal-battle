#!/usr/bin/env python3
# Record and replay Pokemon Crystal battles
# Record: from training.replay import record_games
# Replay: python training/replay.py replays/game_001.json

from __future__ import annotations

import json
import sys
from dataclasses import asdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import gymnasium
import numpy as np

import gym_env  # noqa: F401
from engine.actions import Switch, Struggle, UseMove
from training.evaluate import evaluate_vs_baseline
from training.baselines import MaxDamageAgent

REPLAY_DIR = Path(__file__).parent.parent / "replays"


# ============================================================
# RECORDING
# ============================================================
def record_games(model, n_games: int = 5, baseline: str = "max_damage",
                 out_dir: Path | None = None) -> list[Path]:
    """Play games and save replays as JSON."""
    out_dir = out_dir or REPLAY_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    env = gymnasium.make("CrystalBattle-v1", opponent_policy=None)
    inner = env.unwrapped

    # set up opponent
    if baseline == "max_damage":
        from training.baselines import MaxDamageAgent
        agent = MaxDamageAgent()
        opp_fn = lambda opp, p1, rng, **kw: agent.act(opp, p1)
    elif baseline == "smart":
        from training.baselines import SmartAgent
        agent = SmartAgent()
        opp_fn = lambda opp, p1, rng, **kw: agent.act(opp, p1)
    else:
        import random as _random
        rng = _random.Random(42)
        opp_fn = lambda opp, p1, brng, **kw: rng.choice(opp.valid_actions(p1))

    inner._opponent_policy = opp_fn

    is_recurrent = hasattr(model.policy, "lstm_actor")
    saved = []

    for game_idx in range(n_games):
        obs, info = env.reset()
        game_log = {
            "game": game_idx,
            "baseline": baseline,
            "p1_team": _serialize_team(inner._battle.p1),
            "p2_team": _serialize_team(inner._battle.p2),
            "turns": [],
        }

        if is_recurrent:
            lstm_states = None
            episode_start = np.ones((1,), dtype=bool)

        done = False
        while not done:
            # snapshot pre-turn state
            turn_data = {
                "turn": inner._battle.turn,
                "p1_active": _serialize_active(inner._battle.p1),
                "p2_active": _serialize_active(inner._battle.p2),
                "weather": inner._battle.weather,
            }

            # get agent action
            mask = info.get("action_mask", None)
            if is_recurrent:
                action, lstm_states = model.predict(
                    obs, state=lstm_states, episode_start=episode_start,
                    deterministic=True, action_masks=mask,
                )
                episode_start = np.zeros((1,), dtype=bool)
            else:
                action, _ = model.predict(obs, deterministic=True, action_masks=mask)

            action_int = int(action)
            turn_data["p1_action"] = _describe_action(action_int, inner._battle.p1)

            obs, reward, terminated, truncated, info = env.step(action_int)
            done = terminated or truncated

            # capture events from the turn
            # events are on the battle object's last_events if we store them
            # since we don't have direct event access, reconstruct from state changes
            turn_data["p1_active_after"] = _serialize_active(inner._battle.p1)
            turn_data["p2_active_after"] = _serialize_active(inner._battle.p2)
            turn_data["reward"] = float(reward)

            game_log["turns"].append(turn_data)

        # outcome
        winner = info.get("winner", 0)
        game_log["winner"] = "p1" if winner == 1 else "p2" if winner == 2 else "draw"
        game_log["total_turns"] = len(game_log["turns"])
        game_log["p1_remaining"] = inner._battle.p1.alive_count
        game_log["p2_remaining"] = inner._battle.p2.alive_count

        path = out_dir / f"game_{game_idx:03d}.json"
        with open(path, "w") as f:
            json.dump(game_log, f, indent=2)
        saved.append(path)
        result = "WIN" if winner == 1 else "LOSS" if winner == 2 else "DRAW"
        print(f"  Game {game_idx}: {result} in {game_log['total_turns']} turns -> {path.name}")

    env.close()
    return saved


def _serialize_team(player) -> list[dict]:
    team = []
    for p in player.team:
        team.append({
            "name": p.species.name,
            "types": p.types,
            "hp": f"{p.current_hp}/{p.max_hp}",
            "moves": [s.template.name for s in p.move_slots],
        })
    return team


def _serialize_active(player) -> dict:
    p = player.active
    return {
        "name": p.species.name,
        "hp": f"{p.current_hp}/{p.max_hp}",
        "hp_pct": round(p.hp_frac * 100, 1),
        "status": p.status,
        "moves": [
            {"name": s.template.name, "pp": f"{s.current_pp}/{s.template.pp}",
             "type": s.template.type, "power": s.template.power}
            for s in p.move_slots
        ],
    }


def _describe_action(action_int: int, player) -> dict:
    if action_int < 4:
        slot = player.active.move_slots[action_int]
        return {
            "type": "move",
            "slot": action_int,
            "name": slot.template.name,
            "move_type": slot.template.type,
            "power": slot.template.power,
            "category": slot.template.damage_class,
        }
    else:
        team_idx = action_int - 4
        if team_idx < len(player.team):
            target = player.team[team_idx]
            return {
                "type": "switch",
                "to": target.species.name,
                "to_hp_pct": round(target.hp_frac * 100, 1),
            }
        return {"type": "switch", "to": "???"}


# ============================================================
# TERMINAL VIEWER
# ============================================================
def replay_game(path: Path, speed: str = "step"):
    """Replay a recorded game in the terminal."""
    with open(path) as f:
        game = json.load(f)

    print("=" * 60)
    print(f"  REPLAY: Game {game['game']} vs {game['baseline']}")
    print(f"  Result: {game['winner'].upper()} in {game['total_turns']} turns")
    print("=" * 60)

    # show teams
    print("\n  P1 Team (Agent):")
    for p in game["p1_team"]:
        types = "/".join(p["types"])
        moves = ", ".join(p["moves"])
        print(f"    {p['name']:12s} [{types:16s}] {moves}")

    print("\n  P2 Team (Opponent):")
    for p in game["p2_team"]:
        types = "/".join(p["types"])
        moves = ", ".join(p["moves"])
        print(f"    {p['name']:12s} [{types:16s}] {moves}")

    print("\n" + "-" * 60)

    for turn in game["turns"]:
        p1 = turn["p1_active"]
        p2 = turn["p2_active"]
        act = turn["p1_action"]
        p1_after = turn["p1_active_after"]
        p2_after = turn["p2_active_after"]

        # turn header
        weather = f" [{turn['weather']}]" if turn.get("weather") else ""
        print(f"\n  Turn {turn['turn']}{weather}")
        print(f"    P1: {p1['name']:12s} {p1['hp']:>10s} ({p1['hp_pct']:5.1f}%)"
              f"  {p1['status'] or ''}")
        print(f"    P2: {p2['name']:12s} {p2['hp']:>10s} ({p2['hp_pct']:5.1f}%)"
              f"  {p2['status'] or ''}")

        # action
        if act["type"] == "move":
            cat = act["category"][:3].upper()
            power = f"pow:{act['power']}" if act["power"] else "status"
            print(f"    >> {act['name']} [{act['move_type']}/{cat}] {power}")
        else:
            print(f"    >> Switch to {act['to']} ({act.get('to_hp_pct', '?')}% HP)")

        # outcome
        p1_delta = p1_after["hp_pct"] - p1["hp_pct"]
        p2_delta = p2_after["hp_pct"] - p2["hp_pct"]

        changes = []
        if p2_delta != 0:
            changes.append(f"P2: {p2_delta:+.1f}%")
        if p1_delta != 0:
            changes.append(f"P1: {p1_delta:+.1f}%")
        if p1_after["name"] != p1["name"]:
            changes.append(f"P1 now: {p1_after['name']}")
        if p2_after["name"] != p2["name"]:
            changes.append(f"P2 now: {p2_after['name']}")
        if p1_after.get("status") and not p1.get("status"):
            changes.append(f"P1 got {p1_after['status']}")
        if p2_after.get("status") and not p2.get("status"):
            changes.append(f"P2 got {p2_after['status']}")

        if changes:
            print(f"       {' | '.join(changes)}")

        r = turn["reward"]
        if abs(r) > 0.01:
            print(f"       reward: {r:+.3f}")

        if speed == "step":
            try:
                input("       [enter]")
            except (EOFError, KeyboardInterrupt):
                print("\n  (skipping to end)")
                speed = "auto"

    print("\n" + "=" * 60)
    print(f"  RESULT: {game['winner'].upper()}")
    print(f"  P1 remaining: {game['p1_remaining']}")
    print(f"  P2 remaining: {game['p2_remaining']}")
    print("=" * 60)


# ============================================================
# CLI
# ============================================================
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Battle replay viewer")
    parser.add_argument("file", nargs="?", help="Replay JSON file to view")
    parser.add_argument("--record", action="store_true",
                        help="Record games from latest model")
    parser.add_argument("--model", type=str, default="crystal_battle_ppo",
                        help="Model path for recording")
    parser.add_argument("--n-games", type=int, default=5)
    parser.add_argument("--baseline", type=str, default="max_damage")
    parser.add_argument("--auto", action="store_true",
                        help="Auto-advance instead of step-by-step")
    args = parser.parse_args()

    if args.record:
        from training.maskable_recurrent_ppo import MaskableRecurrentPPO
        print(f"Loading model from {args.model}...")
        model = MaskableRecurrentPPO.load(args.model, device="cpu")
        print(f"Recording {args.n_games} games vs {args.baseline}...")
        paths = record_games(model, n_games=args.n_games, baseline=args.baseline)
        print(f"\nSaved {len(paths)} replays to {REPLAY_DIR}/")
    elif args.file:
        replay_game(Path(args.file), speed="auto" if args.auto else "step")
    else:
        # list available replays
        if REPLAY_DIR.exists():
            replays = sorted(REPLAY_DIR.glob("*.json"))
            if replays:
                print(f"Available replays in {REPLAY_DIR}/:")
                for r in replays:
                    with open(r) as f:
                        g = json.load(f)
                    print(f"  {r.name}: vs {g['baseline']} -> {g['winner']} "
                          f"({g['total_turns']} turns)")
            else:
                print("No replays found. Use --record to create some.")
        else:
            print("No replays found. Use --record to create some.")
