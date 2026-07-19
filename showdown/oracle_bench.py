#!/usr/bin/env python3
"""
Decision-quality oracle bench: score search configs by agreement with a
deep-search oracle on a fixed bank of stored positions — minutes offline
instead of hours of live games.

Method:
  bank    — N mid-game positions sampled from a holdout pickle, phase-
            stratified; positions with <3 root options are dropped (forced
            moves measure nothing)
  oracle  — TWO independent deep searches per position. Where their argmaxes
            agree the position is STABLE and top-1 agreement is meaningful;
            where they disagree the oracle itself is noise, so those
            positions only contribute to the soft metric
  metrics — top-1 agreement with the oracle on stable positions (overall,
            by phase, by oracle-flatness), plus mean "oracle approval": the
            oracle's visit share of the candidate's chosen move on ALL
            positions (near-ties don't punish)

Scope note: candidates are ENGINE-level deciders (budget, value net at
leaves). The bench sees translated full-information states, so it measures
the search+eval stack, not the belief/translation stack (tiers, sampling).

The oracle shares the engine's eval, so this measures how well a config
approximates deep search — not absolute correctness. Errors the eval itself
makes at every depth are invisible here; that axis is eval_calibration.py's.

Usage:
  .venv/bin/python showdown/oracle_bench.py \
      --holdout <scratch>/holdout_data.pkl --n 240 --oracle-ms 8000 \
      --candidate mcts:300 --candidate mcts:2000 \
      --candidate value:showdown/value_net_gen9_v3.onnx:0.5:2000:32
"""

from __future__ import annotations

import argparse
import pickle
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

import poke_engine as pe
from showdown.eval_calibration import load_rows

FLAT_SHARE = 0.45  # top move below this visit share = flat (house convention)


def build_bank(holdout_path: str, n: int, rng) -> list[tuple[str, int, int]]:
    """Phase-stratified sample of (state_str, turn_idx, game_len)."""
    rows = load_rows(holdout_path)
    early = [r for r in rows if r[2] < 8]
    mid = [r for r in rows if 8 <= r[2] < 20]
    late = [r for r in rows if r[2] >= 20]
    bank = []
    for pool, share in ((early, 0.30), (mid, 0.40), (late, 0.30)):
        k = min(int(n * share), len(pool))
        idx = rng.choice(len(pool), size=k, replace=False)
        bank.extend((pool[i][0], pool[i][2], pool[i][3]) for i in idx)
    return bank


def _dist(result) -> dict[str, int]:
    return {m.move_choice: m.visits for m in result.side_one}


def _top(dist: dict[str, int]) -> tuple[str, float]:
    tot = sum(dist.values()) or 1
    mv = max(dist, key=dist.get)
    return mv, dist[mv] / tot


def oracle_pass(bank, oracle_ms: int, workers: int, cache: Path):
    """Two independent deep searches per position; cached (expensive)."""
    if cache.exists():
        print(f"oracle cache hit: {cache}")
        return pickle.load(open(cache, "rb"))

    def probe(state_str):
        d1 = _dist(pe.monte_carlo_tree_search(
            pe.State.from_string(state_str), oracle_ms))
        d2 = _dist(pe.monte_carlo_tree_search(
            pe.State.from_string(state_str), oracle_ms))
        return d1, d2

    out = []
    t0 = time.time()
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for i, (row, (d1, d2)) in enumerate(
                zip(bank, pool.map(lambda r: probe(r[0]), bank))):
            merged = {k: d1.get(k, 0) + d2.get(k, 0)
                      for k in set(d1) | set(d2)}
            if len(merged) < 3:
                out.append(None)  # forced/trivial — excluded
                continue
            m1, _ = _top(d1)
            m2, _ = _top(d2)
            top_mv, top_share = _top(merged)
            out.append({
                "merged": merged, "top": top_mv, "top_share": top_share,
                "stable": m1 == m2, "flat": top_share < FLAT_SHARE,
            })
            if i % 40 == 0 and i:
                print(f"  oracle {i}/{len(bank)} ({time.time()-t0:.0f}s)")
    pickle.dump(out, open(cache, "wb"))
    return out


def parse_candidate(spec: str):
    parts = spec.split(":")
    if parts[0] == "mcts":
        ms = int(parts[1])
        return spec, ("mcts", ms)
    if parts[0] == "value":
        path, alpha, ms, batch = parts[1], float(parts[2]), int(parts[3]), \
            int(parts[4])
        return spec, ("value", path, alpha, ms, batch)
    raise ValueError(f"unknown candidate spec: {spec}")


def candidate_pass(bank, mode, workers: int) -> list[str | None]:
    if mode[0] == "mcts":
        _, ms = mode

        def decide(state_str):
            d = _dist(pe.monte_carlo_tree_search(
                pe.State.from_string(state_str), ms))
            return _top(d)[0] if d else None

        with ThreadPoolExecutor(max_workers=workers) as pool:
            return list(pool.map(lambda r: decide(r[0]), bank))

    # value candidates run SERIAL: mcts_with_value still holds the GIL
    # (its PyRef<ValueNet> isn't Ungil-safe; see the fork binding notes)
    _, path, alpha, ms, batch = mode
    vn = pe.ValueNet(path)
    out = []
    for r in bank:
        d = _dist(pe.monte_carlo_tree_search_with_value(
            pe.State.from_string(r[0]), vn, ms, alpha=alpha,
            batch_size=batch))
        out.append(_top(d)[0] if d else None)
    return out


def report(bank, oracle, results: dict[str, list]):
    strata = {
        "all": lambda r, o: True,
        "early (t<8)": lambda r, o: r[1] < 8,
        "mid (8-19)": lambda r, o: 8 <= r[1] < 20,
        "late (t>=20)": lambda r, o: r[1] >= 20,
        "oracle-flat": lambda r, o: o["flat"],
        "oracle-decisive": lambda r, o: not o["flat"],
    }
    valid = [(i, o) for i, o in enumerate(oracle) if o is not None]
    stable = [(i, o) for i, o in valid if o["stable"]]
    print(f"\nbank: {len(bank)} sampled, {len(valid)} usable (>=3 options), "
          f"{len(stable)} oracle-stable "
          f"({100*len(stable)/max(1,len(valid)):.0f}% stability = "
          f"agreement ceiling)")

    print(f"\nTOP-1 AGREEMENT with oracle (stable positions only):")
    print(f"{'stratum':<18}" + "".join(f"{k:>16}" for k in results))
    for name, pred in strata.items():
        line = f"{name:<18}"
        for spec, moves in results.items():
            hits = [moves[i] == o["top"] for i, o in stable if pred(bank[i], o)]
            line += (f"{100*np.mean(hits):>15.0f}% "
                     if hits else f"{'--':>16}")
        print(line)

    print(f"\nMEAN ORACLE APPROVAL (oracle visit share of chosen move, "
          f"all usable positions):")
    line = f"{'':<18}"
    for spec, moves in results.items():
        appr = []
        for i, o in valid:
            mv = moves[i]
            tot = sum(o["merged"].values()) or 1
            appr.append(o["merged"].get(mv, 0) / tot)
        line += f"{100*np.mean(appr):>15.0f}% "
    print(line)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--holdout", required=True)
    ap.add_argument("--n", type=int, default=240)
    ap.add_argument("--oracle-ms", type=int, default=8000)
    ap.add_argument("--workers", type=int, default=5)
    ap.add_argument("--candidate", action="append", default=[],
                    help="mcts:<ms> or value:<onnx>:<alpha>:<ms>:<batch>")
    ap.add_argument("--cache-dir", default=None)
    args = ap.parse_args()

    cache_dir = Path(args.cache_dir) if args.cache_dir \
        else Path(args.holdout).parent
    rng = np.random.default_rng(11)
    bank = build_bank(args.holdout, args.n, rng)
    print(f"bank built: {len(bank)} positions")

    cache = cache_dir / f"oracle_{len(bank)}_{args.oracle_ms}ms.pkl"
    oracle = oracle_pass(bank, args.oracle_ms, args.workers, cache)

    specs = args.candidate or ["mcts:300"]
    results = {}
    for spec in specs:
        name, mode = parse_candidate(spec)
        t0 = time.time()
        results[name] = candidate_pass(bank, mode, args.workers)
        print(f"candidate {name}: decided {len(bank)} positions "
              f"in {time.time()-t0:.0f}s")

    report(bank, oracle, results)


if __name__ == "__main__":
    main()
