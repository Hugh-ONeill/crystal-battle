# evaluate search quality using the current PPO model's value network
# tests raw policy vs search-enhanced play
#
# Usage:
#   python training/eval_search.py --model imitation_ppo --depth 2 --games 100

from __future__ import annotations

import argparse
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

from engine.types import TypeChart
from engine.data_loader import DataStore
from training.baselines import SmartAgent, MaxDamageAgent
from training.maskable_recurrent_ppo import MaskableRecurrentPPO
from training.evaluate import evaluate_vs_baseline


def eval_with_rust_search(model_path: str, n_games: int = 100, depth: int = 2,
                          device: str = "cpu"):
    """Evaluate model with Rust search using its own value network."""
    import crystal_engine_rs as ce
    from training.opponent_model import OpponentPredictor
    from training.rust_search_agent import (
        build_obs_from_rust, _RustPlayerAdapter,
        _get_opp_actions, _handle_forced_switches_rs,
        _first_alive_bench_idx, _py_action_to_int
    )
    from gym_env.team_builder import build_team

    tc = TypeChart.load()
    data = DataStore()
    rs_data = ce.DataStore(str(Path(__file__).parent.parent / "data"))

    opp_model = OpponentPredictor()
    opp_model.load_state_dict(
        torch.load("opp_model.pt", map_location=device, weights_only=True))
    opp_model.eval()

    smart = SmartAgent(tc, seed=99)
    maxdmg = MaxDamageAgent(tc)

    for bl_name, bl_agent in [("max_damage", maxdmg), ("smart", smart)]:
        wins = 0
        for i in range(n_games):
            t1 = build_team(data, rng=random.Random(i + 100), tier="ou")
            t2 = build_team(data, rng=random.Random(i + 200), tier="ou")

            rs_t1 = [rs_data.build_pokemon(m.species.id,
                      [s.template.id for s in m.move_slots]) for m in t1]
            rs_t2 = [rs_data.build_pokemon(m.species.id,
                      [s.template.id for s in m.move_slots]) for m in t2]
            rs_battle = ce.create_battle(rs_t1, rs_t2, seed=i + 300)

            opp_hidden = None
            for turn in range(100):
                if rs_battle.is_over:
                    break

                obs = build_obs_from_rust(rs_battle, tc)
                p1_mask = rs_battle.p1.valid_action_mask(rs_battle.p2,
                                                          filter_immune=True)
                valid = [j for j in range(10) if p1_mask[j]]

                if not valid:
                    break

                if len(valid) == 1:
                    best = valid[0]
                    _, opp_hidden = opp_model.predict_single(obs, opp_hidden)
                else:
                    opp_actions = _get_opp_actions(
                        rs_battle, obs, opp_model, opp_hidden, 5)
                    _, opp_hidden = opp_model.predict_single(obs, opp_hidden)

                    if depth >= 2:
                        ranked = ce.search_2ply(
                            rs_battle, valid, opp_actions, opp_actions,
                            base_seed=turn * 1000)
                    else:
                        ranked = ce.search_1ply(
                            rs_battle, valid, opp_actions,
                            base_seed=turn * 1000)
                    best = ranked[0][0] if ranked else valid[0]

                p2_adapter = _RustPlayerAdapter(rs_battle.p2)
                p1_adapter = _RustPlayerAdapter(rs_battle.p1)
                p2_action = bl_agent.act(p2_adapter, p1_adapter)
                a2_int = _py_action_to_int(p2_action)

                rs_battle.resolve_turn(best, a2_int)
                _handle_forced_switches_rs(
                    rs_battle, lambda my, opp: bl_agent.act(my, opp))

                if rs_battle.p1.must_switch:
                    sw = _first_alive_bench_idx(rs_battle.p1)
                    if sw is not None:
                        rs_battle.resolve_forced_switches(sw, None)

            if rs_battle.winner == 1:
                wins += 1

        print(f"  Rust {depth}-ply vs {bl_name:12s}: "
              f"{wins}/{n_games} ({wins / n_games * 100:.1f}%)")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="imitation_ppo")
    parser.add_argument("--depth", type=int, default=2)
    parser.add_argument("--games", type=int, default=100)
    parser.add_argument("--device", type=str, default="cpu")
    args = parser.parse_args()

    # raw policy eval
    print("Raw policy:")
    model = MaskableRecurrentPPO.load(args.model, device=args.device)
    for bl in ("max_damage", "smart"):
        r = evaluate_vs_baseline(model, bl, n_games=args.games, seed=42,
                                  both_sides=False)
        print(f"  vs {bl:12s}: {r['win_rate']:.1%}")

    # search eval
    print(f"\nWith Rust {args.depth}-ply search:")
    t0 = time.time()
    eval_with_rust_search(args.model, n_games=args.games, depth=args.depth,
                          device=args.device)
    print(f"  Time: {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
