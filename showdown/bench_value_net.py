#!/usr/bin/env python3
"""Bench mcts_with_value (dev) vs plain monte_carlo_tree_search (ref) at α-sweep.

Step 7 of the gen9 value-net pipeline. dev side calls
`pe.mcts_with_value(state, value_net, ms, alpha=A)`; ref side calls
`pe.monte_carlo_tree_search(state, ms)`. Counterbalanced halves so first-mover
bias cancels.

Success criteria (from VALUE_NET_PIPELINE_PLAN.md):
  At some α, dev wins ≥55% across the 4-matchup average over multi-seed runs.
  α=0.0 is a sanity check (should match v3 baseline ~50%).

Usage:
  .venv/bin/python showdown/bench_value_net.py \\
      --value-net showdown/gen9_value_net.onnx \\
      --games 10 --search-ms 300

The 4 canonical matchups per the plan:
  Mirror Sun (0 vs 0), Sun-Stall (0 vs 3), BO-Balance (1 vs 2), Rain-TR (4 vs 5).
"""

from __future__ import annotations

import argparse
import random
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

import poke_engine as pe

from showdown.local_battle import build_pe_state_gen9
from showdown.sample_teams_gen9 import SAMPLE_TEAMS_GEN9
from showdown.features_v3 import parse_state_v3


_FLIP_CACHE: dict[str, str] = {}


def _flipped_state_str(s: str) -> str:
    cached = _FLIP_CACHE.get(s)
    if cached is not None:
        return cached
    parts = s.split("/")
    if len(parts) < 2:
        return s
    out = "/".join([parts[1], parts[0]] + parts[2:])
    if len(_FLIP_CACHE) > 4096:
        _FLIP_CACHE.clear()
    _FLIP_CACHE[s] = out
    return out


class PolicyOnnx:
    """Policy net wrapper: state → 9-dim softmax priors. Despite the name, it
    loads from a .pt checkpoint via PyTorch (no onnxruntime dependency).

    p1 priors come from the raw state; p2 priors come from the side-flipped
    state (the policy net was trained from p1's POV with side-flip augment).
    """

    def __init__(self, path: str):
        import torch
        from showdown.policy_train import PolicyNet
        ckpt = torch.load(path, weights_only=True, map_location="cpu")
        self.model = PolicyNet(state_dim=ckpt["state_dim"],
                               hidden=ckpt["hidden"],
                               n_layers=ckpt.get("n_layers", 3))
        self.model.load_state_dict(ckpt["model"])
        self.model.eval()
        self._torch = torch

    def _logits(self, feats: np.ndarray) -> np.ndarray:
        with self._torch.no_grad():
            x = self._torch.from_numpy(feats[None, :].astype(np.float32))
            return self.model(x)[0].numpy()

    @staticmethod
    def _softmax(x: np.ndarray) -> np.ndarray:
        x = x - x.max()
        e = np.exp(x)
        return e / e.sum()

    def priors(self, state_str: str) -> tuple[list[float], list[float]]:
        s1 = self._softmax(self._logits(parse_state_v3(state_str)))
        s2 = self._softmax(self._logits(parse_state_v3(_flipped_state_str(state_str))))
        return s1.tolist(), s2.tolist()


class PolicyFromQNet:
    """Adapter: use a Q-net's per-action Q values as policy priors via softmax.

    The Q-net was trained with a policy aux loss that pushed softmax(Q) toward
    visit_dist, so this should be similar to a dedicated policy net — but
    with a per-action *value* signal also influencing the rankings. If the
    value loss pulled some Q values down (e.g., Rapid Spin in no-hazard
    states), the resulting priors will be slightly less biased toward those
    bad actions than the pure-visit-distillation policy net.
    """

    def __init__(self, value_net):
        self.value_net = value_net  # a pe.ValueNet (must be a Q-net .material)

    @staticmethod
    def _softmax(x: np.ndarray) -> np.ndarray:
        # The Q values are already in [0,1] from sigmoid; convert back to
        # logits via inverse-sigmoid before softmax so the temperature
        # interpretation is consistent.
        x = np.clip(x, 1e-6, 1.0 - 1e-6)
        z = np.log(x / (1.0 - x))
        z = z - z.max()
        e = np.exp(z)
        return e / e.sum()

    def priors(self, state_str: str) -> tuple[list[float], list[float]]:
        from showdown.local_battle import build_pe_state_gen9
        # We have a state_str but need a pe.State to query the Q-net.
        # The value_net.predict_q method takes a pe.State directly.
        # Reconstruct from the string via pe.State.from_string if available,
        # else parse via the standard path.
        state = pe.State.from_string(state_str)
        s1_q = np.asarray(self.value_net.predict_q(state), dtype=np.float32)
        # Side-flip for p2 perspective.
        flipped_str = _flipped_state_str(state_str)
        state_flipped = pe.State.from_string(flipped_str)
        s2_q = np.asarray(self.value_net.predict_q(state_flipped), dtype=np.float32)
        return self._softmax(s1_q).tolist(), self._softmax(s2_q).tolist()


CANONICAL_MATCHUPS = [
    ("Mirror Sun", 0, 0),
    ("Sun-Stall", 0, 3),
    ("BO-Balance", 1, 2),
    ("Rain-TR", 4, 5),
]

# BO-locked eval: dev side is always team 1 (Bulky Offense). The two
# holdouts (Sun=0, KingambitHO=9) are kept out of training data; BO-Balance
# is an in-distribution sanity check so a holdout win at the cost of
# training-pool regression is visible.
BO_HOLDOUT_MATCHUPS = [
    ("BO-Sun", 1, 0),
    ("BO-GambitHO", 1, 9),
    ("BO-Balance", 1, 2),
]

MATCHUP_SETS = {
    "general": CANONICAL_MATCHUPS,
    "bo_holdout": BO_HOLDOUT_MATCHUPS,
}


def _strip_switch_prefix(m: str) -> str:
    return m[7:] if m.startswith("switch ") else m


def _normalize_no_move(m: str) -> str:
    return "none" if m == "No Move" else m


def _dev_pick(state: pe.State, value_net, search_ms: int, alpha: float, side: str,
              policy_net: "PolicyOnnx | None" = None, residual: bool = False,
              batch_size: int = 1) -> str:
    """Run mcts_with_value (+ optional policy priors), return top-visits move."""
    s1_priors = s2_priors = None
    if policy_net is not None:
        s1_priors, s2_priors = policy_net.priors(state.to_string())
    r = pe.mcts_with_value(state, value_net, search_ms,
                           s1_priors=s1_priors, s2_priors=s2_priors,
                           alpha=alpha, residual=residual, batch_size=batch_size)
    side_results = r.s1 if side == "p1" else r.s2
    return max(side_results, key=lambda x: x.visits).move_choice


def _ref_pick(state: pe.State, search_ms: int, side: str) -> str:
    """Run plain MCTS, return the highest-visit move_choice for given side."""
    r = pe.monte_carlo_tree_search(state, duration_ms=search_ms)
    side_results = r.side_one if side == "p1" else r.side_two
    return max(side_results, key=lambda x: x.visits).move_choice


def play_value_vs_ref_game(team1: str, team2: str, value_net,
                           search_ms: int, alpha: float, dev_side: int,
                           max_turns: int = 120,
                           policy_net: "PolicyOnnx | None" = None,
                           residual: bool = False,
                           batch_size: int = 1) -> int:
    """One game. dev_side=1 → dev (value-net) plays p1; dev_side=2 → dev plays p2.
    Returns 1 if p1 wins, 2 if p2 wins, 0 on draw/timeout."""
    state = build_pe_state_gen9(team1, team2)
    prev_str = ""
    stuck_turns = 0

    for _ in range(max_turns):
        s1_alive = sum(1 for p in state.side_one.pokemon if p.hp > 0)
        s2_alive = sum(1 for p in state.side_two.pokemon if p.hp > 0)
        if s1_alive == 0:
            return 2
        if s2_alive == 0:
            return 1

        try:
            if dev_side == 1:
                p1_move = _dev_pick(state, value_net, search_ms, alpha, "p1", policy_net, residual, batch_size)
                p2_move = _ref_pick(state, search_ms, "p2")
            else:
                p1_move = _ref_pick(state, search_ms, "p1")
                p2_move = _dev_pick(state, value_net, search_ms, alpha, "p2", policy_net, residual, batch_size)
        except Exception as e:
            print(f"\n  search error: {e}")
            break

        p1_move = _normalize_no_move(p1_move)
        p2_move = _normalize_no_move(p2_move)
        if p1_move == "none" and p2_move == "none":
            break
        p1_clean = _strip_switch_prefix(p1_move)
        p2_clean = _strip_switch_prefix(p2_move)

        try:
            instructions = pe.generate_instructions(state, p1_clean, p2_clean)
        except Exception as e:
            print(f"\n  resolve error: {e}")
            break
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
                return 0
        else:
            stuck_turns = 0
        prev_str = cur_str

    return 0


def run_matchup(team1: str, team2: str, value_net, search_ms: int,
                alpha: float, n_games: int, seed_base: int | None,
                policy_net: "PolicyOnnx | None" = None,
                residual: bool = False,
                batch_size: int = 1) -> tuple[int, int, int]:
    """Run n_games counterbalanced halves. Returns (dev_wins, dev_losses, draws)."""
    dev_w = dev_l = draws = 0
    for i in range(n_games):
        if seed_base is not None:
            random.seed(seed_base + i)
        # Half A: dev controls p1
        r = play_value_vs_ref_game(team1, team2, value_net, search_ms, alpha,
                                   dev_side=1, policy_net=policy_net,
                                   residual=residual, batch_size=batch_size)
        if r == 1: dev_w += 1
        elif r == 2: dev_l += 1
        else: draws += 1
        print("." if r == 1 else ("x" if r == 2 else "-"), end="", flush=True)

        if seed_base is not None:
            random.seed(seed_base + i)
        # Half B: dev controls p2
        r = play_value_vs_ref_game(team1, team2, value_net, search_ms, alpha,
                                   dev_side=2, policy_net=policy_net,
                                   residual=residual, batch_size=batch_size)
        if r == 2: dev_w += 1
        elif r == 1: dev_l += 1
        else: draws += 1
        print("." if r == 2 else ("x" if r == 1 else "-"), end="", flush=True)
    print()
    return dev_w, dev_l, draws


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--value-net", type=str, default="showdown/gen9_value_net.onnx")
    ap.add_argument("--policy-net", type=str, default=None,
                    help="optional policy net ONNX (priors via PUCT in mcts_with_value)")
    ap.add_argument("--games", type=int, default=10,
                    help="games per matchup per half (total per matchup = 2x)")
    ap.add_argument("--search-ms", type=int, default=300)
    ap.add_argument("--alphas", type=str, default="0.0,0.3,0.5,0.7,1.0",
                    help="comma-separated α values to sweep")
    ap.add_argument("--seed", type=int, default=1000)
    ap.add_argument("--residual", action="store_true",
                    help="treat the value net as a centered residual on top of "
                         "sigmoid(eval/SCALE); leaf = clamp(h + α(2v-1), 0, 1).")
    ap.add_argument("--batch-size", type=int, default=1,
                    help="batched MCTS with virtual loss; K rollouts per "
                         "value-net call. 1=sequential.")
    ap.add_argument("--qnet-as-policy", type=str, default=None,
                    help="path to a Q-net .material file used as policy "
                         "(softmax of per-action Q vector). Replaces --policy-net.")
    ap.add_argument("--matchup-set", choices=list(MATCHUP_SETS.keys()),
                    default="general",
                    help="'general' = legacy 4 mixed matchups; "
                         "'bo_holdout' = BO vs Sun/GambitHO holdouts + "
                         "BO-Balance in-distribution sanity.")
    args = ap.parse_args()
    matchups = MATCHUP_SETS[args.matchup_set]

    alphas = [float(a) for a in args.alphas.split(",")]

    print(f"loading value net: {args.value_net}")
    value_net = pe.ValueNet(args.value_net)
    policy_net = None
    if args.qnet_as_policy:
        print(f"loading Q-net as policy: {args.qnet_as_policy}")
        qnet_for_policy = pe.ValueNet(args.qnet_as_policy)
        policy_net = PolicyFromQNet(qnet_for_policy)
    elif args.policy_net:
        print(f"loading policy net: {args.policy_net}")
        policy_net = PolicyOnnx(args.policy_net)
    print(f"alphas: {alphas}, {args.games}/half × {len(matchups)} matchups "
          f"× {len(alphas)} alphas = "
          f"{2 * args.games * len(matchups) * len(alphas)} total games "
          f"at {args.search_ms}ms ({args.matchup_set})")
    print()

    t0 = time.time()
    # alpha → matchup_name → (w, l, d)
    results: dict[float, dict[str, tuple[int, int, int]]] = {a: {} for a in alphas}

    for alpha in alphas:
        print(f"=== α = {alpha} ===")
        for name, t1_idx, t2_idx in matchups:
            print(f"  {name} ({t1_idx} vs {t2_idx}): ", end="", flush=True)
            t1 = SAMPLE_TEAMS_GEN9[t1_idx]
            t2 = SAMPLE_TEAMS_GEN9[t2_idx]
            wld = run_matchup(t1, t2, value_net, args.search_ms, alpha,
                              args.games, args.seed, policy_net=policy_net,
                              residual=args.residual,
                              batch_size=args.batch_size)
            results[alpha][name] = wld
            w, l, d = wld
            total = w + l
            pct = w / total * 100 if total > 0 else 0
            print(f"     {w}W {l}L {d}D ({pct:.1f}%)")
        print()

    elapsed = time.time() - t0
    print()
    print(f"=== summary (n={2 * args.games} per matchup, {elapsed:.0f}s total) ===")
    print(f"{'alpha':>6}  " + "  ".join(f"{name:>12}" for name, _, _ in matchups) + "  |   avg")
    for alpha in alphas:
        cells = []
        wins = total = 0
        for name, _, _ in matchups:
            w, l, d = results[alpha][name]
            t = w + l
            wins += w
            total += t
            pct = w / t * 100 if t > 0 else 0
            cells.append(f"{pct:>11.1f}%")
        avg = wins / total * 100 if total > 0 else 0
        marker = " *" if avg >= 55.0 else ""
        print(f"  {alpha:>4.2f}  " + "  ".join(cells) + f"  |  {avg:>5.1f}%{marker}")
    print()
    print("* = ≥55% (plan's success threshold)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
