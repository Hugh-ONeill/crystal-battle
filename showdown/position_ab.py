"""Position-level eval A/B: the game-mix-free instrument for eval-term
attribution.

The 2026-07-22 bisect showed arm-level behavioral metrics can't attribute
eval effects — between-run comparisons inherit game-mix variance (~±2.5pp on
a CONSTANT policy). This instrument removes the denominator problem: a FIXED
pool of positions (collected via gen9_player --dump-states) is searched under
each CB_EVAL_OFF config in its own subprocess (the flags are OnceLock-cached
per process), and the CHOSEN MOVES are diffed position by position.

MCTS is stochastic, so `ext` runs TWICE and the ext-vs-ext flip rate is the
instrument's own noise floor — a config matters only where its flip rate and
category shifts clear that floor.

Usage:
  position_ab.py POOL.jsonl --configs base,hazards,hopeless,hazards+hopeless
      [--ms 300] [--limit N] [--threads 8]
  ('ext' + its rerun are always included; '+' in a config = comma in
   CB_EVAL_OFF, e.g. hazards+hopeless)
"""
import argparse
import json
import os
import subprocess
import sys
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path


def worker(pool_path, ms, threads, limit):
    import poke_engine as pe
    recs = [json.loads(l) for l in open(pool_path)][:limit or None]

    def search(rec):
        state = pe.State.from_string(rec["state"])
        res = pe.monte_carlo_tree_search(state, ms)
        best = max(res.side_one, key=lambda r: r.visits)
        return {"choice": best.move_choice, "visits": best.visits}

    with ThreadPoolExecutor(max_workers=threads) as pool:
        out = list(pool.map(search, recs))
    json.dump(out, sys.stdout)


def category(choice):
    sys.path.insert(0, str(Path(__file__).parent))
    from behavior_compare import classify, mid
    if choice.startswith("switch"):
        return "switch"
    return classify(mid(choice))


def run_config(name, env_base, env_off, args):
    env = {**os.environ}
    env.pop("CB_EVAL_BASELINE", None)
    env.pop("CB_EVAL_OFF", None)
    if env_base:
        env["CB_EVAL_BASELINE"] = env_base
    if env_off:
        env["CB_EVAL_OFF"] = env_off
    cmd = [sys.executable, __file__, args.pool, "--worker",
           "--ms", str(args.ms), "--threads", str(args.threads)]
    if args.limit:
        cmd += ["--limit", str(args.limit)]
    r = subprocess.run(cmd, capture_output=True, text=True, env=env)
    if r.returncode != 0:
        sys.exit(f"worker for {name} failed: {r.stderr[-500:]}")
    return json.loads(r.stdout)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("pool")
    ap.add_argument("--configs", default="base")
    ap.add_argument("--ms", type=int, default=300)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--worker", action="store_true")
    args = ap.parse_args()

    if args.worker:
        worker(args.pool, args.ms, args.threads, args.limit)
        return

    n_pool = sum(1 for _ in open(args.pool))
    n = min(n_pool, args.limit) if args.limit else n_pool
    print(f"pool: {n} positions from {args.pool}, {args.ms}ms searches")

    configs = [("ext", "", ""), ("ext-rerun", "", "")]
    for c in args.configs.split(","):
        c = c.strip()
        if not c or c in ("ext", "ext-rerun"):
            continue
        if c == "base":
            configs.append(("base", "1", ""))
        else:
            configs.append((c, "", c.replace("+", ",")))

    results = {}
    for name, eb, eo in configs:
        results[name] = run_config(name, eb, eo, args)
        print(f"  searched under {name} (OFF='{eo or eb and 'ALL' or ''}')")

    ref = results["ext"]
    print(f"\n{'config':18s} {'flip%':>6s}  category deltas vs ext "
          f"(only |Δ| >= 1 shown)")
    ref_cat = Counter(category(r["choice"]) for r in ref)
    for name, _, _ in configs:
        if name == "ext":
            continue
        cur = results[name]
        flips = sum(1 for a, b in zip(ref, cur) if a["choice"] != b["choice"])
        cat = Counter(category(r["choice"]) for r in cur)
        deltas = {k: cat[k] - ref_cat[k] for k in set(cat) | set(ref_cat)}
        dstr = "  ".join(f"{k}{v:+d}" for k, v in
                         sorted(deltas.items(), key=lambda kv: -abs(kv[1]))
                         if abs(v) >= 1)
        tag = "  <- NOISE FLOOR" if name == "ext-rerun" else ""
        print(f"{name:18s} {100 * flips / len(ref):5.1f}%  {dstr}{tag}")

    # biggest qualitative flips for the first non-control config
    interesting = [c for c, _, _ in configs if c not in ("ext", "ext-rerun")]
    if interesting:
        name = interesting[0]
        recs = [json.loads(l) for l in open(args.pool)][:args.limit or None]
        flips = [(recs[i], ref[i]["choice"], results[name][i]["choice"])
                 for i in range(len(ref))
                 if ref[i]["choice"] != results[name][i]["choice"]]
        print(f"\nsample flips under {name} (up to 10):")
        for rec, a, b in flips[:10]:
            print(f"  {rec['tag']} T{rec['turn']}: {a} -> {b}")


if __name__ == "__main__":
    main()
