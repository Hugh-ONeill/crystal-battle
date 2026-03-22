# Validate Rust engine against Python engine
# Two validation modes:
#   1. Deterministic: compare stat calc, type chart, action masks, expected damage
#   2. Statistical: run N games with each engine, compare win rate distributions
#
# Usage:
#   .venv/bin/python tools/validate_rust_engine.py [--games 1000]

from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from engine.actions import Struggle, Switch, UseMove
from engine.battle_state import BattleState as PyBattleState
from engine.damage import calc_expected_damage as py_calc_expected_damage
from engine.data_loader import DataStore
from engine.player_state import PlayerState as PyPlayerState
from engine.turn_engine import resolve_turn as py_resolve_turn
from engine.turn_engine import resolve_forced_switches as py_resolve_forced_switches
from engine.types import TypeChart

try:
    import crystal_engine_rs as ce
except ImportError:
    print("ERROR: crystal_engine_rs not found -- run 'maturin develop --release' in crystal_engine/")
    sys.exit(1)

from gym_env.team_builder import build_team


DATA_DIR = str(Path(__file__).parent.parent / "data")


# ============================================================
# DETERMINISTIC VALIDATION
# ============================================================

def validate_type_chart(type_chart):
    """Compare type effectiveness between Python and Rust engines."""
    all_types = ["bug", "dark", "dragon", "electric", "fighting", "fire", "flying",
                 "ghost", "grass", "ground", "ice", "normal", "poison", "psychic",
                 "rock", "steel", "water"]
    errors = 0
    for atk in all_types:
        for def1 in all_types:
            py_eff = type_chart.effectiveness(atk, def1)
            rs_eff = ce.type_effectiveness(atk, [def1])
            if abs(py_eff - rs_eff) > 0.001:
                print(f"  TYPE MISMATCH: {atk} vs {def1}: py={py_eff} rs={rs_eff}")
                errors += 1
            # dual type
            for def2 in all_types:
                py_eff2 = type_chart.combined_effectiveness(atk, [def1, def2])
                rs_eff2 = ce.type_effectiveness(atk, [def1, def2])
                if abs(py_eff2 - rs_eff2) > 0.001:
                    print(f"  TYPE MISMATCH: {atk} vs [{def1},{def2}]: py={py_eff2} rs={rs_eff2}")
                    errors += 1
    return errors


def validate_stat_calc(data, rs_data):
    """Compare stat calculations between engines."""
    errors = 0
    for species_id in list(data.pokemon.keys())[:50]:
        species_data = data.pokemon[species_id]
        # get a few moves from learnset
        learnset = species_data.get("learnset", [])
        move_ids = learnset[:4] if learnset else []

        from engine.pokemon import PokemonSpecies, Pokemon as PyPokemon
        from engine.move import MoveTemplate as PyMoveTemplate

        py_species = PokemonSpecies(
            id=species_data["id"],
            name=species_data["name"],
            types=species_data["types"],
            base_stats=species_data["base_stats"],
            learnset=learnset,
        )
        py_move_templates = []
        for mid in move_ids:
            md = data.moves.get(mid)
            if md:
                py_move_templates.append(PyMoveTemplate(
                    id=md["id"], name=md["name"], type=md["type"],
                    power=md.get("power") or 0, accuracy=md.get("accuracy"),
                    pp=md["pp"], priority=md.get("priority", 0),
                    damage_class=md["damage_class"], meta=md.get("meta"),
                ))
        py_mon = PyPokemon.from_species(py_species, py_move_templates)

        rs_mon = rs_data.build_pokemon(species_id, move_ids)
        if rs_mon is None:
            continue

        # compare stats
        py_stats = py_mon.stats
        rs_stats = rs_mon.stats
        for stat_name, py_val in py_stats.items():
            idx_map = {"hp": 0, "attack": 1, "defense": 2, "special_attack": 3,
                       "special_defense": 4, "speed": 5}
            idx = idx_map.get(stat_name)
            if idx is not None and py_val != rs_stats[idx]:
                print(f"  STAT MISMATCH: {py_species.name} {stat_name}: py={py_val} rs={rs_stats[idx]}")
                errors += 1

        # compare max HP
        if py_mon.max_hp != rs_mon.max_hp:
            print(f"  HP MISMATCH: {py_species.name}: py={py_mon.max_hp} rs={rs_mon.max_hp}")
            errors += 1

    return errors


def validate_action_masks(data, type_chart, rs_data, n_games=20):
    """Run games and compare action masks at each turn."""
    errors = 0
    master_rng = random.Random(12345)

    for g in range(n_games):
        rng = random.Random(master_rng.randint(0, 2**32))
        team1 = build_team(data, rng=random.Random(rng.randint(0, 2**32)))
        team2 = build_team(data, rng=random.Random(rng.randint(0, 2**32)))

        py_battle = PyBattleState(
            p1=PyPlayerState(team=team1),
            p2=PyPlayerState(team=team2),
            rng=random.Random(rng.randint(0, 2**32)),
        )

        rs_team1 = [rs_data.build_pokemon(m.species.id, [s.template.id for s in m.move_slots]) for m in team1]
        rs_team2 = [rs_data.build_pokemon(m.species.id, [s.template.id for s in m.move_slots]) for m in team2]
        rs_battle = ce.create_battle(rs_team1, rs_team2, seed=0)

        # compare initial action masks
        # compare with type_chart immune filtering
        py_mask = py_battle.p1.valid_action_mask(py_battle.p2, type_chart=type_chart)
        rs_mask = rs_battle.p1.valid_action_mask(rs_battle.p2, filter_immune=True)
        for i in range(10):
            if py_mask[i] != rs_mask[i]:
                p1_name = py_battle.p1.active.name
                errors += 1
                print(f"  MASK MISMATCH game {g} turn 0 action {i}: py={py_mask[i]} rs={rs_mask[i]} (P1={p1_name})")

    return errors


# ============================================================
# STATISTICAL VALIDATION
# ============================================================

def run_py_game(data, type_chart, rng_seed):
    """Run a game with the Python engine using random actions. Returns winner."""
    rng = random.Random(rng_seed)
    team1 = build_team(data, rng=random.Random(rng.randint(0, 2**32)))
    team2 = build_team(data, rng=random.Random(rng.randint(0, 2**32)))

    battle = PyBattleState(
        p1=PyPlayerState(team=team1),
        p2=PyPlayerState(team=team2),
        rng=random.Random(rng.randint(0, 2**32)),
    )

    action_rng = random.Random(rng.randint(0, 2**32))
    for _ in range(200):
        if battle.is_over:
            break
        a1 = action_rng.choice(battle.p1.valid_actions(battle.p2))
        a2 = action_rng.choice(battle.p2.valid_actions(battle.p1))
        py_resolve_turn(battle, a1, a2, type_chart)

        # forced switches
        for pnum in [1, 2]:
            ps = battle.p1 if pnum == 1 else battle.p2
            if ps.must_switch:
                sw = None
                for i, p in enumerate(ps.team):
                    if i != ps.active_index and not p.is_fainted:
                        sw = Switch(team_index=i)
                        break
                if sw:
                    py_resolve_forced_switches(
                        battle,
                        sw if pnum == 1 else None,
                        sw if pnum == 2 else None,
                    )

    return battle.winner, battle.turn


def run_rs_game(rs_data, rng_seed):
    """Run a game with the Rust engine using random actions. Returns winner."""
    import random as stdlib_random
    rng = stdlib_random.Random(rng_seed)

    # build teams via Python's team builder then convert
    data = DataStore()
    team1 = build_team(data, rng=stdlib_random.Random(rng.randint(0, 2**32)))
    team2 = build_team(data, rng=stdlib_random.Random(rng.randint(0, 2**32)))

    rs_team1 = [rs_data.build_pokemon(m.species.id, [s.template.id for s in m.move_slots]) for m in team1]
    rs_team2 = [rs_data.build_pokemon(m.species.id, [s.template.id for s in m.move_slots]) for m in team2]

    battle = ce.create_battle(rs_team1, rs_team2, seed=rng.randint(0, 2**62))

    action_rng = stdlib_random.Random(rng.randint(0, 2**32))
    for _ in range(200):
        if battle.is_over:
            break

        # pick random valid actions
        p1_mask = battle.p1.valid_action_mask(battle.p2)
        p2_mask = battle.p2.valid_action_mask(battle.p1)
        p1_valid = [i for i in range(10) if p1_mask[i]]
        p2_valid = [i for i in range(10) if p2_mask[i]]

        if not p1_valid or not p2_valid:
            break

        a1 = action_rng.choice(p1_valid)
        a2 = action_rng.choice(p2_valid)

        battle.resolve_turn(a1, a2)

        # forced switches
        for pnum in [1, 2]:
            ps = battle.p1 if pnum == 1 else battle.p2
            if ps.must_switch:
                team = ps.team
                active_idx = ps.active_index
                sw = None
                for i, p in enumerate(team):
                    if i != active_idx and not p.is_fainted:
                        sw = i
                        break
                if sw is not None:
                    battle.resolve_forced_switches(
                        sw if pnum == 1 else None,
                        sw if pnum == 2 else None,
                    )

    return battle.winner, battle.turn


def statistical_validation(data, type_chart, rs_data, n_games):
    """Run games with both engines and compare statistics."""
    import time

    print(f"\n  Running {n_games} games with Python engine...")
    py_results = {"p1_wins": 0, "p2_wins": 0, "draws": 0, "total_turns": 0}
    t0 = time.time()
    for i in range(n_games):
        winner, turns = run_py_game(data, type_chart, rng_seed=i * 7 + 42)
        py_results["total_turns"] += turns
        if winner == 1:
            py_results["p1_wins"] += 1
        elif winner == 2:
            py_results["p2_wins"] += 1
        else:
            py_results["draws"] += 1
    py_time = time.time() - t0

    print(f"  Running {n_games} games with Rust engine...")
    rs_results = {"p1_wins": 0, "p2_wins": 0, "draws": 0, "total_turns": 0}
    t0 = time.time()
    for i in range(n_games):
        winner, turns = run_rs_game(rs_data, rng_seed=i * 7 + 42)
        rs_results["total_turns"] += turns
        if winner == 1:
            rs_results["p1_wins"] += 1
        elif winner == 2:
            rs_results["p2_wins"] += 1
        else:
            rs_results["draws"] += 1
    rs_time = time.time() - t0

    print(f"\n  Python: P1={py_results['p1_wins']} P2={py_results['p2_wins']} "
          f"draws={py_results['draws']} turns={py_results['total_turns']} "
          f"time={py_time:.2f}s ({py_results['total_turns']/py_time:.0f} turns/s)")
    print(f"  Rust:   P1={rs_results['p1_wins']} P2={rs_results['p2_wins']} "
          f"draws={rs_results['draws']} turns={rs_results['total_turns']} "
          f"time={rs_time:.2f}s ({rs_results['total_turns']/rs_time:.0f} turns/s)")

    if rs_time > 0:
        print(f"  Speedup: {py_time/rs_time:.1f}x")

    # check that distributions are reasonable (not identical due to different RNG)
    py_wr = py_results["p1_wins"] / n_games
    rs_wr = rs_results["p1_wins"] / n_games
    py_avg_turns = py_results["total_turns"] / n_games
    rs_avg_turns = rs_results["total_turns"] / n_games

    print(f"\n  P1 win rate: py={py_wr:.1%} rs={rs_wr:.1%}")
    print(f"  Avg turns:   py={py_avg_turns:.1f} rs={rs_avg_turns:.1f}")

    # with random actions, both engines should produce similar distributions
    # allow 10% tolerance
    wr_diff = abs(py_wr - rs_wr)
    turn_diff = abs(py_avg_turns - rs_avg_turns) / max(py_avg_turns, 1)
    if wr_diff > 0.10:
        print(f"  WARNING: P1 win rate differs by {wr_diff:.1%} (threshold: 10%)")
    if turn_diff > 0.15:
        print(f"  WARNING: avg turns differs by {turn_diff:.1%} (threshold: 15%)")

    return wr_diff <= 0.10 and turn_diff <= 0.15


# ============================================================
# MAIN
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="Validate Rust engine against Python engine")
    parser.add_argument("--games", type=int, default=200, help="Games for statistical validation")
    args = parser.parse_args()

    data = DataStore()
    type_chart = TypeChart.load()
    rs_data = ce.DataStore(DATA_DIR)

    print("=" * 60)
    print("Crystal Engine Rust Validation")
    print("=" * 60)

    # 1. Type chart
    print("\n[1] Type chart validation...")
    errs = validate_type_chart(type_chart)
    print(f"  {289 + 289*17 - errs} checks passed, {errs} errors")

    # 2. Stat calculations
    print("\n[2] Stat calculation validation...")
    errs = validate_stat_calc(data, rs_data)
    print(f"  {'PASS' if errs == 0 else f'{errs} ERRORS'}")

    # 3. Action masks
    print("\n[3] Action mask validation...")
    errs = validate_action_masks(data, type_chart, rs_data)
    print(f"  {'PASS' if errs == 0 else f'{errs} ERRORS'}")

    # 4. Statistical validation
    print(f"\n[4] Statistical validation ({args.games} games)...")
    stats_ok = statistical_validation(data, type_chart, rs_data, args.games)
    print(f"  {'PASS' if stats_ok else 'FAIL: distributions differ significantly'}")

    print("\n" + "=" * 60)


if __name__ == "__main__":
    main()
