#!/usr/bin/env python3
"""Self-play data generator for value-net + policy-net training.

Generates training data by playing engine-vs-engine MCTS games. Each turn we
record:
  * the state string (input features)
  * the visit-weighted V for side one (value-net target, in [0, 1])
  * the 9-dim visit distribution for each side (policy-net targets)

Schema:
    [(winner_int, [(state_str, mcts_v_p1, s1_pi9, s2_pi9), ...]), ...]

s1_pi9 / s2_pi9 are length-9 numpy float32 arrays summing to 1, indexed by
poke_engine's policy action space: 0..=3 = active mon's moves (Move/MoveTera/
MoveMega all map to the same slot), 4..=8 = the 5 non-active teammates.

value_train.py reads field [1] (the V); policy_train.py reads fields [2]/[3].

Usage:
  .venv/bin/python showdown/selfplay_gen.py \\
      --games 500 --search-ms 300 --workers 22 \\
      --output showdown/gen9_selfplay_data.pkl

Iter 0 uses plain MCTS (hand-coded v3 eval at leaves). Future iterations can
pass --value-net to drive search via mcts_with_value(state, vn, ms, alpha=A).
"""

from __future__ import annotations

import argparse
import multiprocessing as mp
import os
import pickle
import random
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))


def _strip_switch_prefix(m: str) -> str:
    return m[7:] if m.startswith("switch ") else m


def _normalize_no_move(m: str) -> str:
    return "none" if m == "No Move" else m


def _mcts_value_from_result(side_results) -> float:
    """Visit-weighted V from a side's MCTS results."""
    total_visits = sum(x.visits for x in side_results)
    total_score = sum(x.total_score for x in side_results)
    return total_score / max(total_visits, 1)


def _visit_distribution(side_results) -> "np.ndarray":
    """9-dim normalized visit distribution over the policy action space.

    MoveTera/MoveMega entries share their slot with the underlying Move
    (engine policy_idx is the same), so visits naturally sum into the
    correct slot.
    """
    import numpy as np
    pi = np.zeros(9, dtype=np.float32)
    total = 0
    for x in side_results:
        idx = int(x.policy_idx)
        if 0 <= idx < 9:
            pi[idx] += x.visits
            total += x.visits
    if total > 0:
        pi /= total
    return pi


def _play_game(args: tuple):
    """One self-play game.

    Returns: (winner_int, [(state_str, v_p1, s1_pi9, s2_pi9), ...]).

    winner_int: +1 if p1 won, -1 if p2 won, 0 for draw/timeout.
    State is recorded BEFORE the actions for that turn are applied.
    """
    (seed, team1_idx, team2_idx, search_ms, max_turns,
     value_net_path, alpha, policy_net_path,
     early_temp, early_temp_turns) = args
    random.seed(seed)

    # Imports local to worker (forkserver clones with modules imported in parent
    # but engine handles must be re-bound for some PyO3 patterns).
    import poke_engine as pe
    from showdown.local_battle import build_pe_state_gen9
    from showdown.sample_teams_gen9 import SAMPLE_TEAMS_GEN9

    team1 = SAMPLE_TEAMS_GEN9[team1_idx]
    team2 = SAMPLE_TEAMS_GEN9[team2_idx]

    value_net = None
    use_value = value_net_path is not None
    if use_value:
        value_net = pe.ValueNet(value_net_path)

    policy_net = None
    use_policy = policy_net_path is not None
    if use_policy:
        from showdown.bench_value_net import PolicyOnnx
        policy_net = PolicyOnnx(policy_net_path)

    state = build_pe_state_gen9(team1, team2)
    states_recorded: list = []
    prev_str = ""
    stuck_turns = 0

    def _sample_move(results, temperature: float):
        """Pick a move from MCTS results. temperature=0 → argmax visits;
        temperature>0 → sample from (visits ** (1/τ))-weighted distribution.
        Adds early-game diversity so 10k self-play games don't all look the
        same (deterministic argmax with a fixed policy produces highly
        correlated trajectories)."""
        if temperature <= 0.0:
            return max(results, key=lambda x: x.visits)
        weights = [(max(1, x.visits)) ** (1.0 / temperature) for x in results]
        total = sum(weights)
        if total <= 0:
            return max(results, key=lambda x: x.visits)
        roll = random.random() * total
        cum = 0.0
        for x, w in zip(results, weights):
            cum += w
            if roll <= cum:
                return x
        return results[-1]

    for turn_idx in range(max_turns):
        s1_alive = sum(1 for p in state.side_one.pokemon if p.hp > 0)
        s2_alive = sum(1 for p in state.side_two.pokemon if p.hp > 0)
        if s1_alive == 0:
            return -1, states_recorded
        if s2_alive == 0:
            return 1, states_recorded

        # Run search and pull both V_p1 and best moves for each side.
        try:
            s1_priors = s2_priors = None
            if use_policy:
                s1_priors, s2_priors = policy_net.priors(state.to_string())
            if use_value:
                r = pe.mcts_with_value(state, value_net, search_ms,
                                       s1_priors=s1_priors, s2_priors=s2_priors,
                                       alpha=alpha)
                s1_results = r.s1
                s2_results = r.s2
            elif use_policy:
                r = pe.monte_carlo_tree_search_with_priors(
                    state, s1_priors, s2_priors, duration_ms=search_ms)
                s1_results = r.side_one
                s2_results = r.side_two
            else:
                r = pe.monte_carlo_tree_search(state, duration_ms=search_ms)
                s1_results = r.side_one
                s2_results = r.side_two
        except Exception:
            break

        v_p1 = _mcts_value_from_result(s1_results)
        s1_pi = _visit_distribution(s1_results)
        s2_pi = _visit_distribution(s2_results)
        states_recorded.append((state.to_string(), v_p1, s1_pi, s2_pi))

        # Early turns use temperature sampling for diversity; later turns
        # use argmax (deterministic best play) so endgames stay clean.
        cur_temp = early_temp if turn_idx < early_temp_turns else 0.0
        p1_move = _sample_move(s1_results, cur_temp).move_choice
        p2_move = _sample_move(s2_results, cur_temp).move_choice
        p1_move = _normalize_no_move(p1_move)
        p2_move = _normalize_no_move(p2_move)
        if p1_move == "none" and p2_move == "none":
            break
        p1_clean = _strip_switch_prefix(p1_move)
        p2_clean = _strip_switch_prefix(p2_move)

        try:
            instructions = pe.generate_instructions(state, p1_clean, p2_clean)
        except Exception:
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
                return 0, states_recorded
        else:
            stuck_turns = 0
        prev_str = cur_str

    return 0, states_recorded


def main() -> int:
    default_workers = max(1, (os.cpu_count() or 4) - 2)
    ap = argparse.ArgumentParser()
    ap.add_argument("--games", type=int, default=500,
                    help="number of self-play games to generate")
    ap.add_argument("--search-ms", type=int, default=300,
                    help="MCTS budget per turn")
    ap.add_argument("--workers", type=int, default=default_workers)
    ap.add_argument("--max-turns", type=int, default=120)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--n-teams", type=int, default=10,
                    help="use first N teams from SAMPLE_TEAMS_GEN9 for matchup sampling")
    ap.add_argument("--value-net", type=str, default=None,
                    help="if set, drive both sides via mcts_with_value with this ONNX")
    ap.add_argument("--alpha", type=float, default=1.0,
                    help="α for mcts_with_value when --value-net is set")
    ap.add_argument("--policy-net", type=str, default=None,
                    help="if set, feed PUCT priors from this policy net (.pt) "
                         "into search. Combine with --value-net for AZ-style; "
                         "alone, uses monte_carlo_tree_search_with_priors.")
    ap.add_argument("--output", type=str, default="showdown/gen9_selfplay_data.pkl")
    ap.add_argument("--early-temp", type=float, default=1.0,
                    help="temperature for move sampling in the first "
                         "--early-temp-turns turns (0 = argmax)")
    ap.add_argument("--early-temp-turns", type=int, default=10,
                    help="number of early turns that use temperature "
                         "sampling; later turns use argmax")
    args = ap.parse_args()

    rng = random.Random(args.seed)
    tasks = []
    for g in range(args.games):
        seed = args.seed + g
        t1 = rng.randrange(args.n_teams)
        t2 = rng.randrange(args.n_teams)
        tasks.append((seed, t1, t2, args.search_ms, args.max_turns,
                      args.value_net, args.alpha, args.policy_net,
                      args.early_temp, args.early_temp_turns))

    print(f"generating {args.games} self-play games "
          f"({args.search_ms} ms × ~25 turns × 2 sides ≈ "
          f"{args.games * args.search_ms * 25 * 2 / 1000 / args.workers / 60:.1f} min "
          f"on {args.workers} workers)")
    if args.value_net and args.policy_net:
        print(f"  driver: mcts_with_value+priors (value={args.value_net}, "
              f"policy={args.policy_net}, alpha={args.alpha})")
    elif args.value_net:
        print(f"  driver: mcts_with_value({args.value_net}, alpha={args.alpha})")
    elif args.policy_net:
        print(f"  driver: mcts_with_priors({args.policy_net})")
    else:
        print(f"  driver: plain MCTS (hand-coded v3 eval)")

    t0 = time.time()
    results: list[tuple[int, list[tuple[str, float]]]] = []
    n_turns_total = 0
    win_p1 = win_p2 = draws = 0
    progress_step = max(1, args.games // 25)

    if args.workers <= 1:
        for i, task in enumerate(tasks):
            r = _play_game(task)
            results.append(r)
            n_turns_total += len(r[1])
            if r[0] > 0: win_p1 += 1
            elif r[0] < 0: win_p2 += 1
            else: draws += 1
            if (i + 1) % progress_step == 0:
                dt = time.time() - t0
                rate = (i + 1) / dt
                eta = (args.games - i - 1) / rate
                print(f"  [{i+1}/{args.games}] {rate:.1f} g/s  "
                      f"turns={n_turns_total}  eta={eta/60:.1f} min")
    else:
        with mp.Pool(processes=args.workers) as pool:
            for i, r in enumerate(pool.imap_unordered(_play_game, tasks, chunksize=2)):
                results.append(r)
                n_turns_total += len(r[1])
                if r[0] > 0: win_p1 += 1
                elif r[0] < 0: win_p2 += 1
                else: draws += 1
                if (i + 1) % progress_step == 0:
                    dt = time.time() - t0
                    rate = (i + 1) / dt
                    eta = (args.games - i - 1) / rate
                    print(f"  [{i+1}/{args.games}] {rate:.1f} g/s  "
                          f"turns={n_turns_total}  eta={eta/60:.1f} min", flush=True)

    elapsed = time.time() - t0
    print()
    print(f"done in {elapsed:.0f}s ({elapsed/60:.1f} min)")
    print(f"  games: {len(results)}, total turns: {n_turns_total}")
    print(f"  outcomes: p1={win_p1}, p2={win_p2}, draws={draws}")

    # V distribution from search-derived labels (sanity)
    all_v = [t[1] for _, turns in results for t in turns]
    if all_v:
        import statistics
        n_dec = sum(1 for v in all_v if v < 0.05 or v > 0.95)
        n_unc = sum(1 for v in all_v if 0.4 < v < 0.6)
        print(f"  V: mean={statistics.mean(all_v):.3f}, "
              f"std={statistics.stdev(all_v) if len(all_v) > 1 else 0:.3f}, "
              f"min={min(all_v):.3f}, max={max(all_v):.3f}")
        print(f"  decisive (<0.05/>0.95): {n_dec} ({100*n_dec/len(all_v):.1f}%)")
        print(f"  uncertain (0.4-0.6):    {n_unc} ({100*n_unc/len(all_v):.1f}%)")

    # Filter empty trajectories
    nonempty = [r for r in results if len(r[1]) >= 2]
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "wb") as f:
        pickle.dump(nonempty, f)
    print(f"  wrote {out_path} ({out_path.stat().st_size / 1e6:.1f} MB, "
          f"{len(nonempty)} games kept)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
