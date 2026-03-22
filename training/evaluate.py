# Evaluate trained agent vs baselines

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import random

import numpy as np

from engine.actions import Struggle, Switch, UseMove
from engine.types import TypeChart
from training.baselines import MaxDamageAgent, RandomAgent, SmartAgent
from training.crystal_ai import CrystalAIAgent, AI_CHAMPION


def _make_baseline(baseline: str, tc: TypeChart, seed: int):
    if baseline == "random":
        return RandomAgent(seed=seed)
    elif baseline == "max_damage":
        return MaxDamageAgent(type_chart=tc)
    elif baseline == "smart":
        return SmartAgent(type_chart=tc, seed=seed)
    elif baseline == "crystal_ai":
        return CrystalAIAgent(type_chart=tc, layers=AI_CHAMPION, seed=seed)
    else:
        raise ValueError(f"Unknown baseline: {baseline}")


def _action_to_int(action) -> int:
    """Convert engine Action to gym action int."""
    if isinstance(action, UseMove):
        return action.slot_index
    elif isinstance(action, Switch):
        return 4 + action.team_index
    else:  # Struggle
        return 0


def _eval_as_p1(model, baseline_agent, n_games: int, seed: int) -> dict:
    """Standard eval: model plays as P1."""
    import gym_env  # noqa: F401
    import gymnasium

    def opponent_policy(opp_state, p1_state, rng, **kwargs):
        return baseline_agent.act(opp_state, p1_state)

    env = gymnasium.make("CrystalBattle-v1", opponent_policy=opponent_policy)
    is_recurrent = hasattr(model.policy, "lstm_actor")

    wins = losses = draws = total_turns = 0

    for i in range(n_games):
        obs, info = env.reset(seed=seed + i)
        done = False
        lstm_state = None
        episode_start = np.array([True])
        while not done:
            mask = info["action_mask"]
            if is_recurrent:
                action, lstm_state = model.predict(
                    obs, state=lstm_state, episode_start=episode_start,
                    deterministic=True, action_masks=mask,
                )
                episode_start = np.array([False])
            else:
                action, _ = model.predict(obs, deterministic=True, action_masks=mask)
            obs, reward, terminated, truncated, info = env.step(int(action))
            done = terminated or truncated

        winner = info.get("winner")
        total_turns += info.get("turns", 0)
        if winner == 1:
            wins += 1
        elif winner == 2:
            losses += 1
        else:
            draws += 1

    env.close()
    return {"wins": wins, "losses": losses, "draws": draws, "turns": total_turns}


def _eval_as_p2(model, baseline_agent, n_games: int, seed: int) -> dict:
    """Swapped eval: model plays as P2 (opponent), baseline drives P1."""
    import gym_env  # noqa: F401
    import gymnasium
    from gym_env.obs_builder import build_observation

    tc = TypeChart.load()
    is_recurrent = hasattr(model.policy, "lstm_actor")

    # model acts as P2 opponent policy
    lstm_state = [None]
    episode_start = [np.array([True])]

    def model_as_opponent(opp_state, p1_state, rng, **kwargs):
        turn = kwargs.get("turn", 0)
        weather = kwargs.get("weather")
        weather_turns = kwargs.get("weather_turns", 0)

        # build obs from P2's perspective
        obs = build_observation(
            opp_state, p1_state, tc, turn,
            weather=weather, weather_turns=weather_turns,
        )
        mask = np.array(opp_state.valid_action_mask(p1_state, type_chart=tc), dtype=bool)

        if is_recurrent:
            action, lstm_state[0] = model.predict(
                obs, state=lstm_state[0], episode_start=episode_start[0],
                deterministic=True, action_masks=mask,
            )
            episode_start[0] = np.array([False])
        else:
            action, _ = model.predict(obs, deterministic=True, action_masks=mask)

        action_int = int(action)
        # decode to engine action
        if action_int < 4:
            active = opp_state.active
            if not active.has_any_pp():
                return Struggle()
            if action_int < len(active.move_slots) and active.move_slots[action_int].has_pp:
                return UseMove(slot_index=action_int)
            for j, slot in enumerate(active.move_slots):
                if slot.has_pp:
                    return UseMove(slot_index=j)
            return Struggle()
        else:
            return Switch(team_index=action_int - 4)

    env = gymnasium.make("CrystalBattle-v1", opponent_policy=model_as_opponent)

    wins = losses = draws = total_turns = 0

    for i in range(n_games):
        obs, info = env.reset(seed=seed + i)
        lstm_state[0] = None
        episode_start[0] = np.array([True])
        done = False

        while not done:
            # baseline drives P1
            battle = env.unwrapped._battle
            p1_action = baseline_agent.act(battle.p1, battle.p2)
            action_int = _action_to_int(p1_action)
            obs, reward, terminated, truncated, info = env.step(action_int)
            done = terminated or truncated

        winner = info.get("winner")
        total_turns += info.get("turns", 0)
        # agent is P2, so winner==2 means agent wins
        if winner == 2:
            wins += 1
        elif winner == 1:
            losses += 1
        else:
            draws += 1

    env.close()
    return {"wins": wins, "losses": losses, "draws": draws, "turns": total_turns}


def evaluate_vs_baseline(
    model,
    baseline: str = "random",
    n_games: int = 100,
    seed: int = 0,
    both_sides: bool = True,
) -> dict:
    """Evaluate a trained model against a baseline opponent.

    If both_sides=True, plays half the games as P1 and half as P2 to
    eliminate side bias.
    """
    tc = TypeChart.load()
    agent = _make_baseline(baseline, tc, seed)

    if both_sides:
        n_half = n_games // 2
        r1 = _eval_as_p1(model, agent, n_half, seed)
        # fresh baseline for P2 games to avoid state leakage
        agent2 = _make_baseline(baseline, tc, seed + 10000)
        r2 = _eval_as_p2(model, agent2, n_games - n_half, seed + 10000)
        wins = r1["wins"] + r2["wins"]
        losses = r1["losses"] + r2["losses"]
        draws = r1["draws"] + r2["draws"]
        total_turns = r1["turns"] + r2["turns"]
    else:
        r = _eval_as_p1(model, agent, n_games, seed)
        wins, losses, draws, total_turns = r["wins"], r["losses"], r["draws"], r["turns"]

    return {
        "wins": wins,
        "losses": losses,
        "draws": draws,
        "win_rate": wins / n_games,
        "avg_turns": total_turns / n_games,
    }


def print_evaluation(results: dict, baseline: str) -> None:
    print(f"\n--- vs {baseline} ---")
    print(f"  Win rate: {results['win_rate']:.1%} "
          f"({results['wins']}W / {results['losses']}L / {results['draws']}D)")
    print(f"  Avg turns: {results['avg_turns']:.1f}")


if __name__ == "__main__":
    from sb3_contrib import MaskablePPO

    model_path = sys.argv[1] if len(sys.argv) > 1 else "crystal_battle_ppo"
    print(f"Loading model from {model_path}...")
    model = MaskablePPO.load(model_path)

    for baseline in ["random", "max_damage", "smart", "crystal_ai"]:
        results = evaluate_vs_baseline(model, baseline=baseline, n_games=400)
        print_evaluation(results, baseline)
