#!/usr/bin/env python3
"""Play one bench-style game and print per-turn diagnostics.

For each turn we log:
  - Each side's alive count and HP fractions
  - Dev (value + policy at given α) and ref (plain MCTS) move choices
  - Dev side's value-net prediction for the current state
  - Top-N visits per side for both engines
  - The chosen instructions and side outcome

Useful for finding the specific decisions the value net is making that
lose the game vs plain MCTS.

Usage:
  .venv/bin/python showdown/inspect_game.py \\
    --value-net showdown/value_net_deepset_filter.onnx \\
    --policy-net showdown/policy_net_az.pt \\
    --alpha 0.5 --t1-idx 0 --t2-idx 0 --seed 1000 --dev-side 1
"""

from __future__ import annotations

import argparse
import math
import random
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))
import poke_engine as pe
from showdown.local_battle import build_pe_state_gen9
from showdown.sample_teams_gen9 import SAMPLE_TEAMS_GEN9
from showdown.features_v3 import parse_state_v3
from showdown.bench_value_net import PolicyOnnx


def _strip_switch(m: str) -> str:
    return m[7:] if m.startswith("switch ") else m


def _norm_none(m: str) -> str:
    return "none" if m == "No Move" else m


def fmt_top(results, n=3):
    sorted_r = sorted(results, key=lambda x: -x.visits)
    bits = []
    total = sum(x.visits for x in results)
    for x in sorted_r[:n]:
        pct = x.visits / max(total, 1) * 100
        bits.append(f"{x.move_choice}({pct:.0f}%)")
    return " | ".join(bits)


def hp_summary(side):
    bits = []
    for i, p in enumerate(side.pokemon):
        if p.hp > 0:
            frac = p.hp / max(p.maxhp, 1)
            tag = "*" if i == int(side.active_index) else " "
            bits.append(f"{tag}{p.id[:6]}:{int(frac*100):3d}%")
        else:
            bits.append(f" {p.id[:6]}:KO  ")
    return "  ".join(bits)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--value-net", required=True)
    ap.add_argument("--policy-net", required=True)
    ap.add_argument("--alpha", type=float, default=0.5)
    ap.add_argument("--t1-idx", type=int, default=0)
    ap.add_argument("--t2-idx", type=int, default=0)
    ap.add_argument("--seed", type=int, default=1000)
    ap.add_argument("--dev-side", type=int, default=1, choices=[1, 2])
    ap.add_argument("--search-ms", type=int, default=300)
    ap.add_argument("--max-turns", type=int, default=120)
    ap.add_argument("--residual", action="store_true")
    args = ap.parse_args()

    random.seed(args.seed)
    print(f"loading value: {args.value_net}")
    vn = pe.ValueNet(args.value_net)
    print(f"loading policy: {args.policy_net}")
    pn = PolicyOnnx(args.policy_net)
    print(f"matchup: team{args.t1_idx} vs team{args.t2_idx}  "
          f"dev=p{args.dev_side}  α={args.alpha}  residual={args.residual}")

    t1 = SAMPLE_TEAMS_GEN9[args.t1_idx]
    t2 = SAMPLE_TEAMS_GEN9[args.t2_idx]
    state = build_pe_state_gen9(t1, t2)

    # We approximate dev's V_p1 prediction by doing a 10ms search with α=1
    # (model-only at leaves) and reading the visit-weighted score. Cheap and
    # avoids loading the value net's torch structure here.
    def dev_v_proxy(s):
        # 10 ms search at α=1 — model output dominates
        r = pe.mcts_with_value(s, vn, 10, alpha=1.0, residual=args.residual)
        tot_visits = sum(x.visits for x in r.s1)
        tot_score = sum(x.total_score for x in r.s1)
        return tot_score / max(tot_visits, 1)

    prev_str = ""
    stuck = 0
    for turn in range(1, args.max_turns + 1):
        s1_alive = sum(1 for p in state.side_one.pokemon if p.hp > 0)
        s2_alive = sum(1 for p in state.side_two.pokemon if p.hp > 0)
        if s1_alive == 0 or s2_alive == 0:
            winner = 2 if s1_alive == 0 else 1
            print(f"\n→ GAME END turn {turn}: p{winner} wins  "
                  f"(s1={s1_alive}, s2={s2_alive})")
            return

        print(f"\n── Turn {turn} ──")
        print(f"  p1: {hp_summary(state.side_one)}")
        print(f"  p2: {hp_summary(state.side_two)}")
        print(f"  dev V_p1 estimate: {dev_v_proxy(state):.3f}")

        s1_priors, s2_priors = pn.priors(state.to_string())
        # Show top-3 policy priors per side
        def top_priors(p, side, k=3):
            idx = np.argsort(p)[::-1][:k]
            return " ".join(f"a{i}={p[i]:.2f}" for i in idx)
        print(f"  policy: p1 [{top_priors(s1_priors, state.side_one)}]"
              f"  p2 [{top_priors(s2_priors, state.side_two)}]")

        # dev search
        dev_r = pe.mcts_with_value(state, vn, args.search_ms,
                                    s1_priors=s1_priors, s2_priors=s2_priors,
                                    alpha=args.alpha, residual=args.residual)
        # ref search
        ref_r = pe.monte_carlo_tree_search(state, duration_ms=args.search_ms)

        if args.dev_side == 1:
            p1_choice = max(dev_r.s1, key=lambda x: x.visits).move_choice
            p2_choice = max(ref_r.side_two, key=lambda x: x.visits).move_choice
            print(f"  p1 dev: {fmt_top(dev_r.s1)}")
            print(f"  p2 ref: {fmt_top(ref_r.side_two)}")
        else:
            p1_choice = max(ref_r.side_one, key=lambda x: x.visits).move_choice
            p2_choice = max(dev_r.s2, key=lambda x: x.visits).move_choice
            print(f"  p1 ref: {fmt_top(ref_r.side_one)}")
            print(f"  p2 dev: {fmt_top(dev_r.s2)}")

        print(f"  → p1={p1_choice}   p2={p2_choice}")
        p1_choice = _norm_none(p1_choice); p2_choice = _norm_none(p2_choice)
        if p1_choice == "none" and p2_choice == "none":
            print("  both pass; terminating"); return

        p1_clean = _strip_switch(p1_choice); p2_clean = _strip_switch(p2_choice)
        try:
            insts = pe.generate_instructions(state, p1_clean, p2_clean)
        except Exception as e:
            print(f"  generate_instructions failed: {e}"); return
        if not insts:
            print("  no instructions; terminating"); return

        roll = random.random() * 100
        cum = 0.0
        chosen = insts[0]
        for inst in insts:
            cum += inst.percentage
            if roll <= cum:
                chosen = inst; break
        state = state.apply_instructions(chosen)

        cur = state.to_string()
        if cur == prev_str:
            stuck += 1
            if stuck >= 3:
                print(f"\n→ STUCK at turn {turn}; declaring draw"); return
        else:
            stuck = 0
        prev_str = cur

    print(f"\n→ MAX TURNS ({args.max_turns}); draw")


if __name__ == "__main__":
    main()
