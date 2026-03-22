# Evaluate agent with each curated team template
# Usage: .venv/bin/python training/eval_templates.py [model_path]

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import gymnasium
import numpy as np

from engine.data_loader import DataStore
from engine.types import TypeChart
from gym_env.team_builder import TEMPLATES, _make_pokemon, get_tier
from training.baselines import MaxDamageAgent

TEMPLATE_NAMES = [
    "Paralysis Spread",
    "Sleep-and-Sweep",
    "Toxic Stall",
    "Ho-Oh Offense",
    "Ghost Disruption",
    "Jynx Psychic Sweep",
    "Physical Blitz",
    "Venusaur Sleep Stall",
    "CurseLax",
    "Rain Dance",
    "Spikes Stacking",
    "SD Sweepers",
    "Baton Pass Chain",
    "CurseMilk",
    "Raikou Offense",
    "Nidoking Mixed",
]


def build_template_team(data: DataStore, template_idx: int):
    """Build a team from a specific template index."""
    template = TEMPLATES[template_idx]
    return [_make_pokemon(data, sid, mids) for sid, mids in template]


def evaluate_template(model, template_idx: int, n_games: int = 100, seed: int = 0):
    """Evaluate agent using a specific team template vs MaxDamage."""
    import gym_env  # noqa: F401

    tc = TypeChart.load()
    data = DataStore()
    md = MaxDamageAgent(type_chart=tc)

    def opponent_policy(opp_state, p1_state, rng, **kwargs):
        return md.act(opp_state, p1_state)

    env = gymnasium.make("CrystalBattle-v1", opponent_policy=opponent_policy)
    is_recurrent = hasattr(model.policy, "lstm_actor")

    wins = 0
    total_turns = 0

    for i in range(n_games):
        team = build_template_team(data, template_idx)
        obs, info = env.reset(seed=seed + i, options={"p1_team": team})
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
        turns = info.get("turns", 0)
        total_turns += turns
        if winner == 1:
            wins += 1

    env.close()
    return {"win_rate": wins / n_games, "wins": wins, "n_games": n_games,
            "avg_turns": total_turns / n_games}


def main():
    model_path = sys.argv[1] if len(sys.argv) > 1 else "crystal_battle_ppo"
    print(f"Loading model from {model_path}...")

    from training.maskable_recurrent_ppo import MaskableRecurrentPPO
    model = MaskableRecurrentPPO.load(model_path, device="cpu")

    n_games = 200

    # ---- Baseline: random teams (normal eval) ----
    from training.evaluate import evaluate_vs_baseline
    baseline = evaluate_vs_baseline(model, baseline="max_damage", n_games=n_games)
    print(f"\n{'Template':<25s} {'Win%':>6s} {'AvgT':>6s}  {'W/L':>8s}")
    print("-" * 50)
    print(f"{'(random teams)':<25s} {baseline['win_rate']:>5.1%} {baseline['avg_turns']:>6.1f}"
          f"  {baseline['wins']:>3d}/{baseline['losses']:>3d}")
    print("-" * 50)

    # ---- Per-template eval ----
    results = []
    for idx, name in enumerate(TEMPLATE_NAMES):
        has_uber = any(get_tier(sid) == "uber" for sid, _ in TEMPLATES[idx])
        if has_uber:
            print(f"{name:<25s}  -- skipped (contains ubers) --")
            continue
        r = evaluate_template(model, idx, n_games=n_games)
        results.append((name, r))
        print(f"{name:<25s} {r['win_rate']:>5.1%} {r['avg_turns']:>6.1f}"
              f"  {r['wins']:>3d}/{n_games - r['wins']:>3d}")

    print("-" * 50)
    best = max(results, key=lambda x: x[1]["win_rate"])
    worst = min(results, key=lambda x: x[1]["win_rate"])
    print(f"\n  Best:  {best[0]} ({best[1]['win_rate']:.1%})")
    print(f"  Worst: {worst[0]} ({worst[1]['win_rate']:.1%})")


if __name__ == "__main__":
    main()
