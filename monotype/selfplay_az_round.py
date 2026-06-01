#!/usr/bin/env python3
"""
One round of peer-self-play AZ iteration for monotype.

Each invocation runs ONE round:
  1. Generate self-play games with `--prior-policy` baked in as priors for
     BOTH sides (peer self-play — the asymmetry fix vs BO's r1/r2 failures).
  2. Train a fresh policy net on the generated data.
  3. Run TWO mini-benches:
     - new policy vs PRIOR policy (head-to-head). >50% means AZ compounded.
     - new policy vs plain MCTS (sanity — does it still beat the original).

Chain rounds manually:
  round 1: --prior-policy monotype/selfplay/disco/policy_5000.pt \\
           --out-dir monotype/selfplay/disco_r1/
  round 2: --prior-policy monotype/selfplay/disco_r1/policy.pt \\
           --out-dir monotype/selfplay/disco_r2/
  ...

Why peer self-play: the canary run had only one side use the policy as priors
(the locked team) and opp side was plain MCTS. That lets the policy learn to
beat plain MCTS — capping lift at "what 3000ms search already finds." With
both sides using priors, training distribution shifts to "beat policy-aware
opponents," which has a higher ceiling. The BO experiment tried this at 1k
games per round and didn't compound; we know 1k is below the floor for monotype
(canary's ckpt 500 lifted -4.5pp), so a 5k-per-round retest is the proper test.
"""

from __future__ import annotations

import argparse
import json
import random
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).parent
ROOT = HERE.parent


def _run(cmd: list[str], label: str):
    print(f"\n[{label}] $ {' '.join(cmd)}", flush=True)
    t0 = time.time()
    proc = subprocess.run(cmd, capture_output=False)
    if proc.returncode != 0:
        raise RuntimeError(f"[{label}] failed (exit {proc.returncode})")
    print(f"[{label}] OK ({time.time()-t0:.0f}s)", flush=True)


def head_to_head_bench(*, p1_policy: Path, p2_policy: Path | None,
                       teams_file: str, p1_idx: int, opp_indices: list[int],
                       n_games: int, bench_ms: int) -> dict:
    """P1 uses p1_policy as priors; P2 uses p2_policy (or plain MCTS if None).
    Both sides at bench_ms. Alternates which side faces which opponent."""
    sys.path.insert(0, str(ROOT))
    import poke_engine as pe
    from showdown.bench_value_net import PolicyOnnx
    from showdown.local_battle import build_pe_state_gen9
    from showdown.bench_monotype import load_teams, _best_non_tera, _strip_switch_prefix, _normalize_no_move

    teams = load_teams(Path(teams_file))
    team_bodies = [body for _name, body in teams]
    p1_pol = PolicyOnnx(str(p1_policy))
    p2_pol = PolicyOnnx(str(p2_policy)) if p2_policy else None

    wins = losses = draws = 0
    for opp_idx in opp_indices:
        for i in range(n_games):
            # alternate sides
            if i % 2 == 0:
                t1, t2 = team_bodies[p1_idx], team_bodies[opp_idx]
                p1_is_new = True
            else:
                t1, t2 = team_bodies[opp_idx], team_bodies[p1_idx]
                p1_is_new = False
            random.seed((p1_idx * 1000 + opp_idx) * 100 + i)
            state = build_pe_state_gen9(t1, t2)
            prev = ""; stuck = 0; result = 0
            for _ in range(120):
                s1a = sum(1 for p in state.side_one.pokemon if p.hp > 0)
                s2a = sum(1 for p in state.side_two.pokemon if p.hp > 0)
                if s1a == 0:
                    result = 2; break
                if s2a == 0:
                    result = 1; break
                try:
                    # Side using the "new" policy gets p1_pol priors;
                    # other side gets p2_pol priors (or plain MCTS).
                    new_side_idx = 1 if p1_is_new else 2
                    s_str = state.to_string()
                    if new_side_idx == 1:
                        s1p, s2p = p1_pol.priors(s_str)
                        r1 = pe.monte_carlo_tree_search_with_priors(
                            pe.State.from_string(s_str), s1p, s2p, duration_ms=bench_ms)
                        if p2_pol is not None:
                            s1q, s2q = p2_pol.priors(s_str)
                            r2 = pe.monte_carlo_tree_search_with_priors(
                                pe.State.from_string(s_str), s1q, s2q, duration_ms=bench_ms)
                        else:
                            r2 = pe.monte_carlo_tree_search(
                                pe.State.from_string(s_str), duration_ms=bench_ms)
                    else:
                        if p2_pol is not None:
                            s1q, s2q = p2_pol.priors(s_str)
                            r1 = pe.monte_carlo_tree_search_with_priors(
                                pe.State.from_string(s_str), s1q, s2q, duration_ms=bench_ms)
                        else:
                            r1 = pe.monte_carlo_tree_search(
                                pe.State.from_string(s_str), duration_ms=bench_ms)
                        s1p, s2p = p1_pol.priors(s_str)
                        r2 = pe.monte_carlo_tree_search_with_priors(
                            pe.State.from_string(s_str), s1p, s2p, duration_ms=bench_ms)
                    p1m = _normalize_no_move(_best_non_tera(r1.side_one))
                    p2m = _normalize_no_move(_best_non_tera(r2.side_two))
                except Exception:
                    break
                if p1m == "No Move" and p2m == "No Move":
                    break
                try:
                    insts = pe.generate_instructions(state, _strip_switch_prefix(p1m), _strip_switch_prefix(p2m))
                except Exception:
                    break
                if not insts:
                    break
                roll = random.random() * 100; cum = 0.0; chosen = insts[0]
                for inst in insts:
                    cum += inst.percentage
                    if roll <= cum:
                        chosen = inst; break
                state = state.apply_instructions(chosen)
                cur = state.to_string()
                if cur == prev:
                    stuck += 1
                    if stuck >= 3:
                        break
                else:
                    stuck = 0
                prev = cur

            new_won = (result == 1 and p1_is_new) or (result == 2 and not p1_is_new)
            new_lost = (result == 2 and p1_is_new) or (result == 1 and not p1_is_new)
            if new_won:
                wins += 1
            elif new_lost:
                losses += 1
            else:
                draws += 1
    decided = wins + losses
    wr = (wins / decided) if decided else 0.0
    return {"wins": wins, "losses": losses, "draws": draws,
            "decided": decided, "winrate": wr,
            "lift_vs_50pct": (wr - 0.5) * 100.0}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prior-policy", type=Path, required=True,
                    help="path to .pt of the policy from the previous round "
                         "(used as priors during self-play AND as p2 in h2h)")
    ap.add_argument("--teams-file", type=str, required=True)
    ap.add_argument("--lock-team-idx", type=int, required=True)
    ap.add_argument("--opp-pool", type=str, required=True)
    ap.add_argument("--games", type=int, default=5000)
    ap.add_argument("--chunk-size", type=int, default=1000,
                    help="generate self-play in resumable chunks of this size; "
                         "a restart re-runs only the unfinished chunk")
    ap.add_argument("--train-ms", type=int, default=3000)
    ap.add_argument("--bench-ms", type=int, default=200)
    ap.add_argument("--bench-games", type=int, default=8)
    ap.add_argument("--bench-opps", type=str, required=True,
                    help="comma-sep opp indices for both head-to-head and sanity benches")
    ap.add_argument("--workers", type=int, default=22)
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--seed", type=int, default=2042)
    ap.add_argument("--out-dir", type=Path, required=True)
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    data_pkl = args.out_dir / f"data_{args.games}.pkl"
    new_policy_pt = args.out_dir / "policy.pt"

    # Step 1: peer self-play with prior policy as priors for both sides.
    # Generate in resumable chunks so a machine restart loses at most one
    # chunk (selfplay_gen only writes its .pkl at the very end, so a single
    # 5000-game call would lose everything on interrupt). Each chunk writes
    # its own pkl and is skipped on resume if already present.
    import pickle as _pickle
    chunk = max(1, args.chunk_size)
    n_chunks = (args.games + chunk - 1) // chunk
    chunk_pkls = []
    for ci in range(n_chunks):
        n_this = min(chunk, args.games - ci * chunk)
        chunk_pkl = args.out_dir / f"chunk_{ci:02d}_{n_this}.pkl"
        chunk_pkls.append(chunk_pkl)
        if chunk_pkl.exists() and chunk_pkl.stat().st_size > 0:
            print(f"[chunk {ci+1}/{n_chunks}] {chunk_pkl.name} exists, skipping")
            continue
        _run([
            ".venv/bin/python", "showdown/selfplay_gen.py",
            "--teams-file", args.teams_file,
            "--lock-team-idx", str(args.lock_team_idx),
            "--opp-pool", args.opp_pool,
            "--games", str(n_this),
            "--search-ms", str(args.train_ms),
            "--workers", str(args.workers),
            # distinct seed window per chunk so games don't duplicate
            "--seed", str(args.seed + ci * chunk),
            "--policy-net", str(args.prior_policy),
            "--output", str(chunk_pkl),
        ], label=f"peer-selfplay chunk {ci+1}/{n_chunks} ({n_this}g@{args.train_ms}ms)")

    # Merge chunks into the round's data pkl
    merged = []
    for cp in chunk_pkls:
        with open(cp, "rb") as f:
            merged.extend(_pickle.load(f))
    with open(data_pkl, "wb") as f:
        _pickle.dump(merged, f)
    print(f"merged {len(merged)} games from {n_chunks} chunks into {data_pkl}")

    # Step 2: train fresh policy on this data
    _run([
        ".venv/bin/python", "showdown/policy_train.py",
        "--data", str(data_pkl),
        "--model", str(new_policy_pt),
        "--epochs", str(args.epochs),
    ], label=f"train(e{args.epochs})")

    opp_indices = [int(x) for x in args.bench_opps.split(",")]

    # Step 3a: head-to-head — new policy vs prior policy
    print(f"\n=== h2h bench: new vs prior ({len(opp_indices)} opps x {args.bench_games} games)")
    t0 = time.time()
    h2h = head_to_head_bench(
        p1_policy=new_policy_pt, p2_policy=args.prior_policy,
        teams_file=args.teams_file, p1_idx=args.lock_team_idx,
        opp_indices=opp_indices, n_games=args.bench_games, bench_ms=args.bench_ms,
    )
    h2h["bench_time_sec"] = time.time() - t0
    with open(args.out_dir / "result_h2h.json", "w") as f:
        json.dump(h2h, f, indent=2)
    print(f"  h2h: {h2h['wins']}W {h2h['losses']}L {h2h['draws']}D, "
          f"winrate {h2h['winrate']*100:.1f}%, lift {h2h['lift_vs_50pct']:+.1f}pp")

    # Step 3b: sanity — new policy vs plain MCTS
    print(f"\n=== sanity bench: new vs plain MCTS")
    t0 = time.time()
    sanity = head_to_head_bench(
        p1_policy=new_policy_pt, p2_policy=None,
        teams_file=args.teams_file, p1_idx=args.lock_team_idx,
        opp_indices=opp_indices, n_games=args.bench_games, bench_ms=args.bench_ms,
    )
    sanity["bench_time_sec"] = time.time() - t0
    with open(args.out_dir / "result_sanity.json", "w") as f:
        json.dump(sanity, f, indent=2)
    print(f"  sanity: {sanity['wins']}W {sanity['losses']}L {sanity['draws']}D, "
          f"winrate {sanity['winrate']*100:.1f}%, lift {sanity['lift_vs_50pct']:+.1f}pp")

    print(f"\n=== AZ round summary ===")
    print(f"  vs prior policy:  lift {h2h['lift_vs_50pct']:+.1f}pp  (does AZ compound?)")
    print(f"  vs plain MCTS:    lift {sanity['lift_vs_50pct']:+.1f}pp  (does it still beat baseline?)")
    print(f"  new policy saved to {new_policy_pt}")


if __name__ == "__main__":
    main()
