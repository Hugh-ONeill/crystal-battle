# Benchmark: Python lookahead vs Rust batch search
#
# Usage:
#   .venv/bin/python tools/bench_search.py

from __future__ import annotations

import copy
import random
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from engine.actions import Struggle, Switch, UseMove
from engine.battle_state import BattleState as PyBattleState
from engine.player_state import PlayerState as PyPlayerState
from engine.turn_engine import resolve_turn as py_resolve_turn
from engine.turn_engine import resolve_forced_switches as py_resolve_forced_switches
from engine.types import TypeChart

import crystal_engine_rs as ce

from gym_env.team_builder import build_team
from engine.data_loader import DataStore

DATA_DIR = str(Path(__file__).parent.parent / "data")


def build_game(data, type_chart, rs_data, seed):
    """Build matched Python + Rust battle states."""
    rng = random.Random(seed)
    team1 = build_team(data, rng=random.Random(rng.randint(0, 2**32)))
    team2 = build_team(data, rng=random.Random(rng.randint(0, 2**32)))

    py_battle = PyBattleState(
        p1=PyPlayerState(team=team1),
        p2=PyPlayerState(team=team2),
        rng=random.Random(rng.randint(0, 2**32)),
    )

    rs_team1 = [rs_data.build_pokemon(m.species.id, [s.template.id for s in m.move_slots]) for m in team1]
    rs_team2 = [rs_data.build_pokemon(m.species.id, [s.template.id for s in m.move_slots]) for m in team2]
    rs_battle = ce.create_battle(rs_team1, rs_team2, seed=rng.randint(0, 2**62))

    return py_battle, rs_battle


def py_1ply_search(battle, type_chart, n_opp_samples=3):
    """Python 1-ply search (simplified: uniform opponent)."""
    p1_mask = battle.p1.valid_action_mask(battle.p2, type_chart=type_chart)
    p2_mask = battle.p2.valid_action_mask(battle.p1, type_chart=type_chart)
    p1_valid = [i for i in range(10) if p1_mask[i]]
    p2_valid = [i for i in range(10) if p2_mask[i]]

    if not p1_valid:
        return 0, 0

    # uniform opponent weights
    opp_weight = 1.0 / max(len(p2_valid), 1)
    opp_actions = [(a, opp_weight) for a in p2_valid[:n_opp_samples]]

    action_values = {}
    sims = 0
    for a1 in p1_valid:
        total_val = 0.0
        total_w = 0.0
        for a2, w in opp_actions:
            sim = copy.deepcopy(battle)
            p1_action = _int_to_py_action(a1, sim.p1)
            p2_action = _int_to_py_action(a2, sim.p2)
            py_resolve_turn(sim, p1_action, p2_action, type_chart)
            _handle_forced_switches(sim)
            val = _evaluate(sim)
            total_val += val * w
            total_w += w
            sims += 1
        action_values[a1] = total_val / total_w if total_w > 0 else -1.0

    best = max(action_values, key=action_values.get)
    return best, sims


def py_2ply_search(battle, type_chart, n_opp_samples=3):
    """Python 2-ply search."""
    p1_mask = battle.p1.valid_action_mask(battle.p2, type_chart=type_chart)
    p2_mask = battle.p2.valid_action_mask(battle.p1, type_chart=type_chart)
    p1_valid = [i for i in range(10) if p1_mask[i]]
    p2_valid = [i for i in range(10) if p2_mask[i]]

    if not p1_valid:
        return 0, 0

    opp_weight = 1.0 / max(len(p2_valid), 1)
    opp_d1 = [(a, opp_weight) for a in p2_valid[:n_opp_samples]]

    action_values = {}
    sims = 0
    for a1 in p1_valid:
        total_val = 0.0
        total_w = 0.0
        for a2, w in opp_d1:
            sim = copy.deepcopy(battle)
            py_resolve_turn(sim, _int_to_py_action(a1, sim.p1), _int_to_py_action(a2, sim.p2), type_chart)
            _handle_forced_switches(sim)
            sims += 1

            if sim.is_over:
                val = _evaluate(sim)
            else:
                # depth 2: do 1-ply from here
                _, d2_sims = py_1ply_search(sim, type_chart, n_opp_samples)
                val = _evaluate(sim)  # simplified: just eval the state
                sims += d2_sims

            total_val += val * w
            total_w += w
        action_values[a1] = total_val / total_w if total_w > 0 else -1.0

    best = max(action_values, key=action_values.get)
    return best, sims


def _int_to_py_action(action_int, player):
    if action_int < 4:
        if not player.active.has_any_pp():
            return Struggle()
        if action_int < len(player.active.move_slots) and player.active.move_slots[action_int].has_pp:
            return UseMove(slot_index=action_int)
        for j, slot in enumerate(player.active.move_slots):
            if slot.has_pp:
                return UseMove(slot_index=j)
        return Struggle()
    return Switch(team_index=action_int - 4)


def _handle_forced_switches(battle):
    sw1 = sw2 = None
    if battle.p1.must_switch:
        for i, p in enumerate(battle.p1.team):
            if i != battle.p1.active_index and not p.is_fainted:
                sw1 = Switch(team_index=i); break
    if battle.p2.must_switch:
        for i, p in enumerate(battle.p2.team):
            if i != battle.p2.active_index and not p.is_fainted:
                sw2 = Switch(team_index=i); break
    if sw1 or sw2:
        py_resolve_forced_switches(battle, sw1, sw2)


def _evaluate(battle):
    if battle.is_over:
        return 1.0 if battle.winner == 1 else (-1.0 if battle.winner == 2 else 0.0)
    my_hp = battle.p1.total_hp_frac
    opp_hp = battle.p2.total_hp_frac
    my_alive = battle.p1.alive_count / len(battle.p1.team)
    opp_alive = battle.p2.alive_count / len(battle.p2.team)
    return (my_hp - opp_hp) * 0.6 + (my_alive - opp_alive) * 0.4


def main():
    data = DataStore()
    type_chart = TypeChart.load()
    rs_data = ce.DataStore(DATA_DIR)

    n_positions = 50
    n_opp_samples = 3
    positions = []

    print("Building test positions...")
    for i in range(n_positions):
        py_b, rs_b = build_game(data, type_chart, rs_data, seed=i * 13 + 7)
        # advance a few turns to get interesting positions
        for _ in range(random.randint(2, 8)):
            if py_b.is_over:
                break
            acts = py_b.p1.valid_actions(py_b.p2)
            opp_acts = py_b.p2.valid_actions(py_b.p1)
            if not acts or not opp_acts:
                break
            a1 = random.choice(acts)
            a2 = random.choice(opp_acts)
            py_resolve_turn(py_b, a1, a2, type_chart)
            _handle_forced_switches(py_b)
        if not py_b.is_over:
            positions.append((py_b, rs_b))

    print(f"Got {len(positions)} non-terminal positions\n")

    # ---- 1-ply benchmark ----
    print("=" * 60)
    print("1-PLY SEARCH")
    print("=" * 60)

    # Python
    total_sims_py = 0
    t0 = time.time()
    for py_b, _ in positions:
        _, sims = py_1ply_search(py_b, type_chart, n_opp_samples)
        total_sims_py += sims
    py_1ply_time = time.time() - t0

    # Rust
    total_sims_rs = 0
    t0 = time.time()
    for _, rs_b in positions:
        p1_mask = rs_b.p1.valid_action_mask(rs_b.p2, filter_immune=True)
        p2_mask = rs_b.p2.valid_action_mask(rs_b.p1, filter_immune=True)
        p1_valid = [i for i in range(10) if p1_mask[i]]
        p2_valid = [i for i in range(10) if p2_mask[i]]
        if not p1_valid:
            continue
        opp_w = 1.0 / max(len(p2_valid), 1)
        opp_actions = [(a, opp_w) for a in p2_valid[:n_opp_samples]]
        total_sims_rs += len(p1_valid) * len(opp_actions)
        ce.search_1ply(rs_b, p1_valid, opp_actions)
    rs_1ply_time = time.time() - t0

    print(f"  Python: {total_sims_py:,} sims in {py_1ply_time:.3f}s = {total_sims_py/py_1ply_time:,.0f} sims/s")
    print(f"  Rust:   {total_sims_rs:,} sims in {rs_1ply_time:.3f}s = {total_sims_rs/rs_1ply_time:,.0f} sims/s")
    print(f"  Speedup: {py_1ply_time/rs_1ply_time:.1f}x")

    # ---- 2-ply benchmark ----
    print(f"\n{'=' * 60}")
    print("2-PLY SEARCH")
    print("=" * 60)

    # Python
    total_sims_py = 0
    t0 = time.time()
    for py_b, _ in positions[:20]:  # fewer positions for 2-ply
        _, sims = py_2ply_search(py_b, type_chart, n_opp_samples)
        total_sims_py += sims
    py_2ply_time = time.time() - t0

    # Rust
    total_sims_rs = 0
    t0 = time.time()
    for _, rs_b in positions[:20]:
        p1_mask = rs_b.p1.valid_action_mask(rs_b.p2, filter_immune=True)
        p2_mask = rs_b.p2.valid_action_mask(rs_b.p1, filter_immune=True)
        p1_valid = [i for i in range(10) if p1_mask[i]]
        p2_valid = [i for i in range(10) if p2_mask[i]]
        if not p1_valid:
            continue
        opp_w = 1.0 / max(len(p2_valid), 1)
        opp_d1 = [(a, opp_w) for a in p2_valid[:n_opp_samples]]
        opp_d2 = opp_d1  # same weights for depth 2
        total_sims_rs += len(p1_valid) * len(opp_d1) * (len(p1_valid) * len(opp_d2))
        ce.search_2ply(rs_b, p1_valid, opp_d1, opp_d2)
    rs_2ply_time = time.time() - t0

    print(f"  Python: {total_sims_py:,} sims in {py_2ply_time:.3f}s = {total_sims_py/py_2ply_time:,.0f} sims/s")
    print(f"  Rust:   {total_sims_rs:,} sims in {rs_2ply_time:.3f}s = {total_sims_rs/rs_2ply_time:,.0f} sims/s")
    print(f"  Speedup: {py_2ply_time/rs_2ply_time:.1f}x")

    # ---- batch_resolve benchmark ----
    print(f"\n{'=' * 60}")
    print("BATCH RESOLVE (raw throughput)")
    print("=" * 60)

    # pick one position, blast it with 10k sims
    _, rs_b = positions[0]
    p1_mask = rs_b.p1.valid_action_mask(rs_b.p2, filter_immune=True)
    p2_mask = rs_b.p2.valid_action_mask(rs_b.p1, filter_immune=True)
    p1_valid = [i for i in range(10) if p1_mask[i]]
    p2_valid = [i for i in range(10) if p2_mask[i]]

    # generate 10k random action pairs
    n_batch = 10000
    pairs = [(random.choice(p1_valid), random.choice(p2_valid)) for _ in range(n_batch)]

    t0 = time.time()
    results = ce.batch_resolve(rs_b, pairs)
    batch_time = time.time() - t0

    print(f"  {n_batch:,} sims in {batch_time:.3f}s = {n_batch/batch_time:,.0f} sims/s")


if __name__ == "__main__":
    main()
