#!/usr/bin/env python3
"""
Turn-by-turn trace of one monotype matchup: same MCTS-vs-MCTS loop as
bench_monotype.py but logs field/active/eval/top-K-visits/chosen-move per turn.

Tera moves are suppressed (monotype rule), matching the bench.

Usage:
  .venv/bin/python monotype/trace_match.py \\
      --teams-file monotype/teams/teams_v6.txt \\
      --p1 "I Believe In Fairies" --p2 "RuPauls Dragon Race" \\
      --search-ms 200 --seed 4242 --top-k 5 \\
      --out monotype/traces/fairy_vs_dragon.txt
"""

from __future__ import annotations

import argparse
import random
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import poke_engine as pe

from showdown.local_battle import build_pe_state_gen9
from monotype.heuristics import recovery_override
from monotype.lead_picker import (
    pick_leads, pick_leads_net, load_lead_net,
    reorder_team, split_team_body, species_of,
)
from monotype.chaos_priors import active_move_probs, default_priors
from monotype.move_net_infer import load_move_net, predict_active_move_probs


_HEADER_RE = re.compile(r"^=== \[gen9monotype\] (.+?) ===\s*$", re.M)


def load_teams(path: Path) -> dict[str, str]:
    text = path.read_text()
    parts = _HEADER_RE.split(text)
    out = {}
    for i in range(1, len(parts), 2):
        name = parts[i].strip()
        body = parts[i + 1].strip()
        if body:
            out[name] = body
    return out


def best_non_tera(side_results):
    nt = [x for x in side_results if not x.move_choice.endswith("-tera")]
    pool = nt if nt else list(side_results)
    return max(pool, key=lambda x: x.visits).move_choice


def topk_non_tera(side_results, k):
    nt = [x for x in side_results if not x.move_choice.endswith("-tera")]
    pool = nt if nt else list(side_results)
    return sorted(pool, key=lambda x: x.visits, reverse=True)[:k]


def fmt_active(side, label):
    p = side.pokemon[int(side.active_index)]
    boosts = []
    for stat, attr in [("atk", "attack_boost"), ("spa", "special_attack_boost"),
                       ("def", "defense_boost"), ("spd", "special_defense_boost"),
                       ("spe", "speed_boost"), ("acc", "accuracy_boost"),
                       ("eva", "evasion_boost")]:
        v = getattr(side, attr, 0)
        if v:
            boosts.append(f"{stat}{v:+d}")
    vols = sorted(side.volatile_statuses) if side.volatile_statuses else []
    extras = []
    if side.wish and side.wish[0]:
        extras.append(f"wish({side.wish[0]})")
    if side.future_sight and side.future_sight[0]:
        extras.append(f"fs({side.future_sight[0]})")
    sc = []
    for cond in ["stealth_rock", "spikes", "toxic_spikes", "sticky_web",
                 "reflect", "light_screen", "aurora_veil"]:
        v = getattr(side.side_conditions, cond, 0)
        if v:
            sc.append(f"{cond}={v}")
    return (f"{label}={p.id}({p.hp}/{p.maxhp}"
            + (f",{','.join(boosts)}" if boosts else "")
            + (f",st={p.status}" if p.status and p.status != "none" else "")
            + (f",vol={'+'.join(vols)}" if vols else "")
            + (f",{','.join(extras)}" if extras else "")
            + ")"
            + (f"  hazards[{','.join(sc)}]" if sc else ""))


def fmt_field(state):
    parts = []
    if state.weather and state.weather != "none":
        parts.append(f"wx={state.weather}({state.weather_turns_remaining})")
    if state.terrain and state.terrain != "none":
        parts.append(f"ter={state.terrain}({state.terrain_turns_remaining})")
    if state.trick_room:
        parts.append(f"TR({state.trick_room_turns_remaining})")
    if getattr(state, "gravity", False):
        parts.append(f"grav({getattr(state, 'gravity_turns_remaining', '?')})")
    return " ".join(parts) or "-"


def fmt_team(side, label):
    """One-line team status (HP / status for all 6)."""
    parts = []
    for i, p in enumerate(side.pokemon):
        marker = "*" if i == side.active_index else " "
        hp_pct = (p.hp / p.maxhp * 100) if p.maxhp else 0
        st = f",{p.status[:3]}" if p.status and p.status != "none" else ""
        parts.append(f"{marker}{p.id[:12]}({hp_pct:.0f}%{st})")
    return f"{label}: " + " | ".join(parts)


def _strip_switch(m): return m[7:] if m.startswith("switch ") else m
def _norm_no_move(m): return "none" if m == "No Move" else m


def trace_match(t1_body, t2_body, *, search_ms, max_turns, seed, top_k, out,
                use_heal_heuristic=False, use_lead_picker=False,
                lead_pick_ms=100, lead_net_path=None,
                show_chaos_priors=False, move_net_path=None):
    random.seed(seed)
    if use_lead_picker or lead_net_path:
        if lead_net_path:
            net = load_lead_net(lead_net_path)
            lp1, lp2, _ = pick_leads_net(t1_body, t2_body, net)
            picker_kind = f"net ({lead_net_path})"
        else:
            lp1, lp2, _ = pick_leads(t1_body, t2_body, search_ms=lead_pick_ms)
            picker_kind = f"MCTS {lead_pick_ms}ms/cell"
        b1, b2 = split_team_body(t1_body), split_team_body(t2_body)
        out_line = (f"# lead picker [{picker_kind}]: "
                    f"P1={species_of(b1[lp1])} (was {species_of(b1[0])}), "
                    f"P2={species_of(b2[lp2])} (was {species_of(b2[0])})")
        print(out_line)
        if out:
            out.write(out_line + "\n")
        if lp1:
            t1_body = reorder_team(t1_body, lp1)
        if lp2:
            t2_body = reorder_team(t2_body, lp2)
    state = build_pe_state_gen9(t1_body, t2_body)

    def w(line=""):
        print(line)
        if out:
            out.write(line + "\n")

    w(f"# seed={seed} search_ms={search_ms} max_turns={max_turns} "
      f"heal_heuristic={use_heal_heuristic}")
    w(fmt_team(state.side_one, "P1"))
    w(fmt_team(state.side_two, "P2"))

    prev_str = ""
    stuck = 0
    prev_hp1: int | None = None
    prev_hp2: int | None = None
    prev_id1: str | None = None
    prev_id2: str | None = None
    for turn in range(1, max_turns + 1):
        s1_alive = sum(1 for p in state.side_one.pokemon if p.hp > 0)
        s2_alive = sum(1 for p in state.side_two.pokemon if p.hp > 0)
        if s1_alive == 0:
            w(f"\n[T{turn}] P2 wins ({s2_alive} alive)"); return 2
        if s2_alive == 0:
            w(f"\n[T{turn}] P1 wins ({s1_alive} alive)"); return 1

        s_str = state.to_string()
        try:
            ev = pe.evaluate(pe.State.from_string(s_str))
        except Exception:
            ev = float("nan")

        w(f"\n[T{turn}] field={fmt_field(state)}  eval={ev:+.1f}  (alive P1={s1_alive} P2={s2_alive})")
        w(f"   {fmt_active(state.side_one, 'P1')}")
        w(f"   {fmt_active(state.side_two, 'P2')}")

        try:
            r1 = pe.monte_carlo_tree_search(pe.State.from_string(s_str), duration_ms=search_ms)
            t1 = topk_non_tera(r1.side_one, top_k)
            p1_move = t1[0].move_choice if t1 else "none"
            t1s = ", ".join(f"{m.move_choice}({m.visits})" for m in t1)
        except Exception as e:
            w(f"   P1 MCTS error: {e}"); break
        try:
            r2 = pe.monte_carlo_tree_search(pe.State.from_string(s_str), duration_ms=search_ms)
            t2 = topk_non_tera(r2.side_two, top_k)
            p2_move = t2[0].move_choice if t2 else "none"
            t2s = ", ".join(f"{m.move_choice}({m.visits})" for m in t2)
        except Exception as e:
            w(f"   P2 MCTS error: {e}"); break

        # Heuristic override (after MCTS, before applying instructions).
        if use_heal_heuristic:
            ov1 = recovery_override(state.side_one, r1.side_one, prev_hp1, prev_id1)
            if ov1 is not None and ov1 != p1_move:
                w(f"   [heuristic] P1 override: {p1_move} -> {ov1}")
                p1_move = ov1
            ov2 = recovery_override(state.side_two, r2.side_two, prev_hp2, prev_id2)
            if ov2 is not None and ov2 != p2_move:
                w(f"   [heuristic] P2 override: {p2_move} -> {ov2}")
                p2_move = ov2

        a1 = state.side_one.pokemon[int(state.side_one.active_index)]
        a2 = state.side_two.pokemon[int(state.side_two.active_index)]
        prev_hp1, prev_id1 = a1.hp, a1.id
        prev_hp2, prev_id2 = a2.hp, a2.id

        w(f"   P1 picks: {p1_move}    [{t1s}]")
        w(f"   P2 picks: {p2_move}    [{t2s}]")

        if show_chaos_priors:
            p1_priors = active_move_probs(state.side_one)
            p2_priors = active_move_probs(state.side_two)
            if p1_priors:
                top = sorted(p1_priors.items(), key=lambda kv: -kv[1])[:3]
                w("   P1 chaos prior: " + ", ".join(f"{m}({p*100:.0f}%)" for m, p in top))
            if p2_priors:
                top = sorted(p2_priors.items(), key=lambda kv: -kv[1])[:3]
                w("   P2 chaos prior: " + ", ".join(f"{m}({p*100:.0f}%)" for m, p in top))

        if move_net_path:
            mb = load_move_net(move_net_path)
            p1_pred = predict_active_move_probs(state, state.side_one, mb)
            p2_pred = predict_active_move_probs(state, state.side_two, mb)
            if p1_pred:
                top = sorted(p1_pred.items(), key=lambda kv: -kv[1])
                w("   P1 movenet:    " + ", ".join(f"{m}({p*100:.0f}%)" for m, p in top))
            if p2_pred:
                top = sorted(p2_pred.items(), key=lambda kv: -kv[1])
                w("   P2 movenet:    " + ", ".join(f"{m}({p*100:.0f}%)" for m, p in top))

        p1_move = _norm_no_move(p1_move)
        p2_move = _norm_no_move(p2_move)
        if p1_move == "No Move" and p2_move == "No Move":
            w(f"[T{turn}] both no-move; draw"); return 0

        try:
            insts = pe.generate_instructions(state, _strip_switch(p1_move), _strip_switch(p2_move))
        except Exception as e:
            w(f"   resolve error: {e}"); break
        if not insts:
            w(f"   no instructions returned"); break

        roll = random.random() * 100
        cum = 0.0
        chosen = insts[0]
        for inst in insts:
            cum += inst.percentage
            if roll <= cum:
                chosen = inst
                break

        # short summary of what happened
        w(f"   -> applied branch p={chosen.percentage:.1f}% ({len(chosen.instruction_list)} ops)")
        state = state.apply_instructions(chosen)

        cur = state.to_string()
        if cur == prev_str:
            stuck += 1
            if stuck >= 3:
                w(f"[T{turn}] state frozen; draw"); return 0
        else:
            stuck = 0
        prev_str = cur

    w(f"[end] turn limit"); return 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--teams-file", required=True, type=Path)
    ap.add_argument("--p1", required=True, help="P1 team name from paste")
    ap.add_argument("--p2", required=True, help="P2 team name from paste")
    ap.add_argument("--search-ms", type=int, default=200)
    ap.add_argument("--max-turns", type=int, default=80)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--top-k", type=int, default=5)
    ap.add_argument("--out", type=Path, default=None, help="also write trace to file")
    ap.add_argument("--use-heal-heuristic", action="store_true",
                    help="enable the low-HP recovery override")
    ap.add_argument("--use-lead-picker", action="store_true",
                    help="pick leads via 6x6 MCTS-eval maximin")
    ap.add_argument("--lead-pick-ms", type=int, default=100,
                    help="MCTS budget per cell when picking leads")
    ap.add_argument("--lead-net", type=str, default=None,
                    help="path to trained LeadPickerNet .pt; supersedes --use-lead-picker")
    ap.add_argument("--show-chaos-priors", action="store_true",
                    help="print Smogon chaos top-3 expected moves per turn")
    ap.add_argument("--move-net", type=str, default=None,
                    help="path to trained MoveNet .pt; per-turn move-prediction "
                         "alongside MCTS picks (state-conditioned vs chaos)")
    args = ap.parse_args()

    teams = load_teams(args.teams_file)
    if args.p1 not in teams:
        print(f"team not found: {args.p1!r}\nknown: {list(teams)}", file=sys.stderr); sys.exit(2)
    if args.p2 not in teams:
        print(f"team not found: {args.p2!r}\nknown: {list(teams)}", file=sys.stderr); sys.exit(2)

    out_fh = open(args.out, "w") if args.out else None
    try:
        result = trace_match(teams[args.p1], teams[args.p2],
                             search_ms=args.search_ms, max_turns=args.max_turns,
                             seed=args.seed, top_k=args.top_k, out=out_fh,
                             use_heal_heuristic=args.use_heal_heuristic,
                             use_lead_picker=args.use_lead_picker,
                             lead_pick_ms=args.lead_pick_ms,
                             lead_net_path=args.lead_net,
                             show_chaos_priors=args.show_chaos_priors,
                             move_net_path=args.move_net)
    finally:
        if out_fh:
            out_fh.close()
    print(f"\nresult: {result} ({'P1' if result==1 else 'P2' if result==2 else 'draw'})")


if __name__ == "__main__":
    main()
