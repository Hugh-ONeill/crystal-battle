#!/usr/bin/env python3
"""
Checkpoint-driven self-play distillation for a single monotype team.

For each checkpoint in `--checkpoints`:
  1. Generate self-play games (cumulative target minus already-generated)
  2. Train a policy net on the cumulative .pkl
  3. Run a mini-bench: net-as-PUCT-priors vs plain MCTS, both at bench-ms
  4. Print lift; optionally stop early if it's clearly not working

Usage (one canary team):
  .venv/bin/python monotype/selfplay_checkpoints.py \\
      --teams-file monotype/teams/teams_v6.txt \\
      --lock-team-idx 17 \\
      --opp-pool 0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16 \\
      --checkpoints 500,1000,2500,5000 \\
      --train-ms 3000 --bench-ms 200 \\
      --bench-games 100 \\
      --out-dir monotype/selfplay/disco/

Outputs per checkpoint:
  monotype/selfplay/disco/data_<N>.pkl
  monotype/selfplay/disco/policy_<N>.pt + .onnx
  monotype/selfplay/disco/result_<N>.json
"""

from __future__ import annotations

import argparse
import json
import os
import pickle
import random
import subprocess
import sys
import time
from pathlib import Path


HERE = Path(__file__).parent
ROOT = HERE.parent


def _run(cmd: list[str], log_prefix: str = ""):
    """Run a subprocess, streaming stdout/stderr with a label."""
    print(f"\n[{log_prefix}] $ {' '.join(cmd)}", flush=True)
    t0 = time.time()
    proc = subprocess.run(cmd, capture_output=False)
    elapsed = time.time() - t0
    if proc.returncode != 0:
        raise RuntimeError(f"[{log_prefix}] subprocess failed (exit {proc.returncode}) after {elapsed:.0f}s")
    print(f"[{log_prefix}] OK ({elapsed:.0f}s)", flush=True)
    return elapsed


def generate_more_games(*, teams_file: str, lock_idx: int, opp_pool: str,
                        n_new_games: int, train_ms: int, workers: int,
                        seed_start: int, out_pkl: Path):
    cmd = [
        ".venv/bin/python", "showdown/selfplay_gen.py",
        "--teams-file", teams_file,
        "--lock-team-idx", str(lock_idx),
        "--opp-pool", opp_pool,
        "--games", str(n_new_games),
        "--search-ms", str(train_ms),
        "--workers", str(workers),
        "--seed", str(seed_start),
        "--output", str(out_pkl),
    ]
    return _run(cmd, log_prefix=f"selfplay({n_new_games}g@{train_ms}ms)")


def merge_pkls(parts: list[Path], out: Path):
    """Concatenate the per-batch self-play .pkl trajectory lists into one file."""
    merged = []
    for p in parts:
        with open(p, "rb") as f:
            merged.extend(pickle.load(f))
    with open(out, "wb") as f:
        pickle.dump(merged, f)
    return len(merged)


def train_policy(*, data_pkl: Path, out_pt: Path, epochs: int):
    cmd = [
        ".venv/bin/python", "showdown/policy_train.py",
        "--data", str(data_pkl),
        "--model", str(out_pt),
        "--epochs", str(epochs),
    ]
    return _run(cmd, log_prefix=f"train(e{epochs})")


def mini_bench(*, policy_onnx: Path, teams_file: str, p1_idx: int,
               opp_indices: list[int], n_games: int, bench_ms: int) -> dict:
    """Run a mini-bench: P1 (using policy as priors) vs each opp (plain MCTS).
    Both sides at bench_ms. Returns {wins, losses, draws, lift_vs_50}.

    Uses the inline-imported PolicyOnnx + monte_carlo_tree_search_with_priors
    for P1, and plain MCTS for P2 (the "opponent we want to beat").
    """
    sys.path.insert(0, str(ROOT))
    import poke_engine as pe
    from showdown.bench_value_net import PolicyOnnx
    from showdown.local_battle import build_pe_state_gen9
    from showdown.bench_monotype import load_teams, _best_non_tera, _strip_switch_prefix, _normalize_no_move

    teams = load_teams(Path(teams_file))
    team_bodies = [body for _name, body in teams]
    policy = PolicyOnnx(str(policy_onnx))

    wins = losses = draws = 0
    for opp_idx in opp_indices:
        for i in range(n_games):
            # alternate sides
            if i % 2 == 0:
                t1, t2 = team_bodies[p1_idx], team_bodies[opp_idx]
                p1_uses_policy = True
            else:
                t1, t2 = team_bodies[opp_idx], team_bodies[p1_idx]
                p1_uses_policy = False
            random.seed((p1_idx * 1000 + opp_idx) * 100 + i)
            state = build_pe_state_gen9(t1, t2)
            prev_str = ""
            stuck = 0
            result = 0
            for turn in range(120):
                s1_alive = sum(1 for p in state.side_one.pokemon if p.hp > 0)
                s2_alive = sum(1 for p in state.side_two.pokemon if p.hp > 0)
                if s1_alive == 0:
                    result = 2; break
                if s2_alive == 0:
                    result = 1; break

                # P1 uses policy priors when we're playing our side; the
                # other side plays plain MCTS.
                try:
                    if p1_uses_policy:
                        s1p, s2p = policy.priors(state.to_string())
                        r1 = pe.monte_carlo_tree_search_with_priors(
                            pe.State.from_string(state.to_string()),
                            s1p, s2p, duration_ms=bench_ms)
                        r2 = pe.monte_carlo_tree_search(
                            pe.State.from_string(state.to_string()), duration_ms=bench_ms)
                    else:
                        r1 = pe.monte_carlo_tree_search(
                            pe.State.from_string(state.to_string()), duration_ms=bench_ms)
                        s1p, s2p = policy.priors(state.to_string())
                        r2 = pe.monte_carlo_tree_search_with_priors(
                            pe.State.from_string(state.to_string()),
                            s1p, s2p, duration_ms=bench_ms)
                    p1m = _normalize_no_move(_best_non_tera(r1.side_one))
                    p2m = _normalize_no_move(_best_non_tera(r2.side_two))
                except Exception:
                    break
                if p1m == "No Move" and p2m == "No Move":
                    break
                try:
                    insts = pe.generate_instructions(state,
                                                    _strip_switch_prefix(p1m),
                                                    _strip_switch_prefix(p2m))
                except Exception:
                    break
                if not insts:
                    break
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
                        break
                else:
                    stuck = 0
                prev_str = cur

            # who is "policy net side"?
            if p1_uses_policy:
                if result == 1:
                    wins += 1
                elif result == 2:
                    losses += 1
                else:
                    draws += 1
            else:
                if result == 2:
                    wins += 1
                elif result == 1:
                    losses += 1
                else:
                    draws += 1

    decided = wins + losses
    winrate = (wins / decided) if decided else 0.0
    return {
        "wins": wins, "losses": losses, "draws": draws,
        "decided": decided,
        "winrate": winrate,
        "lift_vs_50pct": (winrate - 0.5) * 100.0,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--teams-file", type=str, required=True)
    ap.add_argument("--lock-team-idx", type=int, required=True)
    ap.add_argument("--opp-pool", type=str, required=True,
                    help="comma-separated team indices for opp pool")
    ap.add_argument("--checkpoints", type=str, default="500,1000,2500,5000",
                    help="cumulative-game targets per checkpoint")
    ap.add_argument("--train-ms", type=int, default=3000)
    ap.add_argument("--bench-ms", type=int, default=200)
    ap.add_argument("--bench-games", type=int, default=100,
                    help="games per checkpoint mini-bench (spread across opp pool)")
    ap.add_argument("--bench-opps", type=str, default=None,
                    help="comma-sep subset of opp pool for mini-bench (default: full pool)")
    ap.add_argument("--workers", type=int, default=22)
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--early-stop-lift", type=float, default=-100.0,
                    help="if mini-bench lift_vs_50 < this for ckpt >=1k games, stop")
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    ckpts = [int(x) for x in args.checkpoints.split(",")]
    ckpts.sort()
    opp_pool_list = [int(x) for x in args.opp_pool.split(",")]
    if args.bench_opps:
        bench_opps = [int(x) for x in args.bench_opps.split(",")]
    else:
        bench_opps = opp_pool_list

    print(f"=== checkpoint plan: {ckpts}")
    print(f"=== train_ms={args.train_ms}, bench_ms={args.bench_ms}")
    print(f"=== lock team={args.lock_team_idx}, opp pool size={len(opp_pool_list)}")
    print(f"=== mini-bench: {len(bench_opps)} opps x {args.bench_games} games each")
    print(f"=== output dir: {args.out_dir}")

    prior_total = 0
    batch_pkls: list[Path] = []
    summary = []
    for ckpt in ckpts:
        n_new = ckpt - prior_total
        if n_new <= 0:
            print(f"\n[ckpt {ckpt}] already covered, skipping batch gen")
        else:
            batch_pkl = args.out_dir / f"batch_{prior_total}_{ckpt}.pkl"
            generate_more_games(
                teams_file=args.teams_file,
                lock_idx=args.lock_team_idx,
                opp_pool=args.opp_pool,
                n_new_games=n_new,
                train_ms=args.train_ms,
                workers=args.workers,
                seed_start=args.seed + prior_total,
                out_pkl=batch_pkl,
            )
            batch_pkls.append(batch_pkl)
            prior_total = ckpt

        cum_pkl = args.out_dir / f"cum_{ckpt}.pkl"
        n_traj = merge_pkls(batch_pkls, cum_pkl)
        print(f"  merged {n_traj} games into {cum_pkl}")

        policy_pt = args.out_dir / f"policy_{ckpt}.pt"
        train_policy(data_pkl=cum_pkl, out_pt=policy_pt, epochs=args.epochs)

        print(f"\n[ckpt {ckpt}] running mini-bench "
              f"({len(bench_opps)} opps x {args.bench_games} games @ {args.bench_ms}ms)...")
        t0 = time.time()
        result = mini_bench(
            policy_onnx=policy_pt,
            teams_file=args.teams_file,
            p1_idx=args.lock_team_idx,
            opp_indices=bench_opps,
            n_games=args.bench_games,
            bench_ms=args.bench_ms,
        )
        bench_time = time.time() - t0
        result["ckpt_games"] = ckpt
        result["bench_time_sec"] = bench_time
        summary.append(result)

        res_path = args.out_dir / f"result_{ckpt}.json"
        with open(res_path, "w") as f:
            json.dump(result, f, indent=2)

        lift = result["lift_vs_50pct"]
        print(f"\n[ckpt {ckpt}] {result['wins']}W {result['losses']}L {result['draws']}D, "
              f"winrate {result['winrate']*100:.1f}%, "
              f"lift vs 50% = {lift:+.1f}pp  ({bench_time:.0f}s)")

        if ckpt >= 1000 and lift < args.early_stop_lift:
            print(f"\n*** early stop: lift {lift:.1f} < threshold {args.early_stop_lift} ***")
            break

    print("\n=== scaling curve ===")
    for r in summary:
        print(f"  ckpt={r['ckpt_games']:>5}  winrate={r['winrate']*100:5.1f}%  "
              f"lift={r['lift_vs_50pct']:+5.1f}pp  ({r['decided']} decided, {r['draws']} draws)")

    final = args.out_dir / "summary.json"
    with open(final, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nsummary saved to {final}")


if __name__ == "__main__":
    main()
