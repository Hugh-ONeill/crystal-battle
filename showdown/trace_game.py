#!/usr/bin/env python3
"""Turn-by-turn trace of one engine-vs-engine game (dev=P1, ref=P2).

Usage:
  .venv/bin/python showdown/trace_game.py --gen 9 --team1 4 --team2 5 \
      --search-ms 300 --seed 4000 --max-turns 80
"""

from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import poke_engine as pe_dev
import poke_engine_ref as pe_ref

from showdown.local_battle import build_pe_state, build_pe_state_gen9
from showdown.sample_teams import SAMPLE_TEAMS
from showdown.sample_teams_gen9 import SAMPLE_TEAMS_GEN9


def _strip_switch_prefix(m: str) -> str:
    return m[7:] if m.startswith("switch ") else m


def _normalize_no_move(m: str) -> str:
    return "none" if m == "No Move" else m


def _fmt_active(side, label):
    p = side.pokemon[int(side.active_index)]
    boosts = []
    if side.attack_boost: boosts.append(f"atk{side.attack_boost:+d}")
    if side.special_attack_boost: boosts.append(f"spa{side.special_attack_boost:+d}")
    if side.speed_boost: boosts.append(f"spe{side.speed_boost:+d}")
    if side.defense_boost: boosts.append(f"def{side.defense_boost:+d}")
    if side.special_defense_boost: boosts.append(f"spd{side.special_defense_boost:+d}")
    vols = sorted(side.volatile_statuses) if side.volatile_statuses else []
    extras = []
    if side.wish and side.wish[0]:
        extras.append(f"wish({side.wish[0]})")
    if side.future_sight and side.future_sight[0]:
        extras.append(f"fs({side.future_sight[0]})")
    tera = ",tera" if p.terastallized else ""
    return (f"{label}={p.id}({p.hp}/{p.maxhp},spe={p.speed}"
            + (f",{','.join(boosts)}" if boosts else "")
            + (f",st={p.status}" if p.status and p.status != "none" else "")
            + (f",vol={'+'.join(vols)}" if vols else "")
            + (f",{','.join(extras)}" if extras else "")
            + tera + ")")


def _fmt_field(state):
    parts = []
    if state.weather and state.weather != "none":
        parts.append(f"wx={state.weather}({state.weather_turns_remaining})")
    if state.terrain and state.terrain != "none":
        parts.append(f"ter={state.terrain}({state.terrain_turns_remaining})")
    if state.trick_room:
        parts.append(f"TR({state.trick_room_turns_remaining})")
    return " ".join(parts) or "-"


def trace_game(team1, team2, search_ms, max_turns, gen, seed):
    random.seed(seed)
    builder = build_pe_state_gen9 if gen == 9 else build_pe_state
    state = builder(team1, team2)
    prev_str = ""
    stuck_turns = 0

    for turn in range(1, max_turns + 1):
        s1_alive = sum(1 for p in state.side_one.pokemon if p.hp > 0)
        s2_alive = sum(1 for p in state.side_two.pokemon if p.hp > 0)
        if s1_alive == 0:
            print(f"[T{turn}] P2 wins"); return 2
        if s2_alive == 0:
            print(f"[T{turn}] P1 wins"); return 1

        s_str = state.to_string()
        try:
            ev_dev = pe_dev.evaluate(pe_dev.State.from_string(s_str))
            ev_ref = pe_ref.evaluate(pe_ref.State.from_string(s_str))
        except Exception:
            ev_dev = ev_ref = float("nan")

        print(f"\n[T{turn}] {_fmt_field(state)} | dev_eval={ev_dev:+.1f} ref_eval={ev_ref:+.1f}")
        print(f"   {_fmt_active(state.side_one, 'P1')}")
        print(f"   {_fmt_active(state.side_two, 'P2')}")

        try:
            ds = pe_dev.State.from_string(s_str)
            r1 = pe_dev.monte_carlo_tree_search(ds, duration_ms=search_ms)
            top1 = sorted(r1.side_one, key=lambda x: x.visits, reverse=True)[:3]
            p1_move = top1[0].move_choice
            p1_top = ", ".join(f"{m.move_choice}({m.visits})" for m in top1)
        except Exception as e:
            print(f"   P1 MCTS error: {e}"); break

        try:
            rs = pe_ref.State.from_string(s_str)
            r2 = pe_ref.monte_carlo_tree_search(rs, duration_ms=search_ms)
            top2 = sorted(r2.side_two, key=lambda x: x.visits, reverse=True)[:3]
            p2_move = top2[0].move_choice
            p2_top = ", ".join(f"{m.move_choice}({m.visits})" for m in top2)
        except Exception as e:
            print(f"   P2 MCTS error: {e}"); break

        print(f"   P1 picks: {p1_move}    [top3: {p1_top}]")
        print(f"   P2 picks: {p2_move}    [top3: {p2_top}]")

        p1_move = _normalize_no_move(p1_move)
        p2_move = _normalize_no_move(p2_move)
        if p1_move == "No Move" and p2_move == "No Move":
            print(f"[T{turn}] both No Move"); break

        try:
            instructions = pe_dev.generate_instructions(
                state, _strip_switch_prefix(p1_move), _strip_switch_prefix(p2_move),
            )
        except Exception as e:
            print(f"   resolve error: {e}"); break
        if not instructions:
            break

        roll = random.random() * 100
        cum = 0.0
        chosen = instructions[0]
        for inst in instructions:
            cum += inst.percentage
            if roll <= cum:
                chosen = inst
                break

        state = state.apply_instructions(chosen)

        cur_str = state.to_string()
        if cur_str == prev_str:
            stuck_turns += 1
            if stuck_turns >= 3:
                print(f"[T{turn}] STATE FROZEN — engine no-op edge case (declaring draw)")
                return 0
        else:
            stuck_turns = 0
        prev_str = cur_str

    print(f"[end] turn limit / break"); return 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gen", type=int, default=9, choices=[2, 9])
    ap.add_argument("--team1", type=int, default=4)
    ap.add_argument("--team2", type=int, default=5)
    ap.add_argument("--search-ms", type=int, default=300)
    ap.add_argument("--max-turns", type=int, default=120)
    ap.add_argument("--seed", type=int, default=4000)
    args = ap.parse_args()

    teams = SAMPLE_TEAMS_GEN9 if args.gen == 9 else SAMPLE_TEAMS
    print(f"=== gen{args.gen} team{args.team1} vs team{args.team2}, seed={args.seed}, "
          f"search={args.search_ms}ms ===")
    trace_game(teams[args.team1], teams[args.team2],
               args.search_ms, args.max_turns, args.gen, args.seed)


if __name__ == "__main__":
    main()
