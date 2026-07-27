#!/usr/bin/env python3
"""Position-level STOCK-vs-FORK decision diff (the "comparison A" instrument).

Searches a FIXED pool of positions under both engines and diffs the chosen
moves position-by-position:
  - fork  : this repo's poke_engine  (CB custom eval + mechanics)   [CB venv]
  - stock : upstream poke_engine 0.0.47 — what foul-play searches with [fp venv]

Same state strings feed both (fork-dumped states parse in upstream, verified).
This is the position-level analog of the fork-vs-foul-play WINRATE comparison:
it shows WHERE and HOW the two engines decide differently in practice, and in
particular whether the fork leans housekeeping (recovery/protect/hazard/switch)
where stock leans pressure (attack/boost/pivot).

CAVEAT — reads honestly: this is the TOTAL behavioral difference between the
engines. It conflates the eval divergence with the MECHANICS divergence; a flip
can be either. (Comparison B, full-vs-CB_EVAL_BASELINE via position_ab.py,
isolates the eval at the cost of comparing against a near-blind searcher.)

fork-vs-fork-rerun is the stochastic-MCTS noise floor — a shift matters only
where it clears that floor.

Usage:
  eval_stock_diff.py POOL.jsonl [--ms 300] [--limit N] [--threads 8]
  eval_stock_diff.py POOL.jsonl --worker --out FILE   (internal)
"""
import argparse
import json
import os
import subprocess
import sys
from collections import Counter
from pathlib import Path

FP_PY = "/home/wiz/Developer/grimoire/foul-play/.venv/bin/python"

# rough offense/defense split for a one-line "does the fork lean housekeeping?"
# summary. pivot (U-turn/Volt) is momentum -> pressure; hazard is setup-for-
# -later -> housekeeping. Fuzzy by design; the raw category table is the truth.
PRESSURE = {"attack", "boost", "pivot"}
HOUSEKEEP = {"recovery", "protect", "hazard", "removal",
             "status_infl", "other_status", "phaze", "switch"}


def worker(pool_path, ms, threads, limit, out):
    """Pure-engine leg: imports only poke_engine (present in BOTH venvs under
    that name), searches every position, writes the chosen moves in order.
    Result goes to a FILE so engine import-warnings on stdout can't corrupt it."""
    import poke_engine as pe
    from concurrent.futures import ThreadPoolExecutor
    states = [json.loads(l)["state"] for l in open(pool_path)][:limit or None]

    def search(s):
        try:
            r = pe.monte_carlo_tree_search(pe.State.from_string(s), ms)
            return max(r.side_one, key=lambda x: x.visits).move_choice
        except Exception as e:
            return f"__ERR__ {e!r}"

    with ThreadPoolExecutor(max_workers=threads) as pool:
        moves = list(pool.map(search, states))
    json.dump(moves, open(out, "w"))


def category(choice):
    if choice.startswith("switch"):
        return "switch"
    if choice.startswith("__ERR__") or choice in ("No Move", "none"):
        return "unknown"
    sys.path.insert(0, str(Path(__file__).parent))
    from behavior_compare import classify, mid
    return classify(mid(choice))


def run_leg(name, python_exe, args, tmp):
    out = str(tmp / f"{name}.json")
    cmd = [python_exe, __file__, args.pool, "--worker", "--out", out,
           "--ms", str(args.ms), "--threads", str(args.threads)]
    if args.limit:
        cmd += ["--limit", str(args.limit)]
    # fresh env; upstream leg must not inherit CB_* eval knobs
    env = {k: v for k, v in os.environ.items() if not k.startswith("CB_")}
    r = subprocess.run(cmd, env=env, capture_output=True, text=True)
    if r.returncode != 0:
        sys.exit(f"leg {name} failed:\n{r.stderr[-800:]}")
    return json.load(open(out))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("pool")
    ap.add_argument("--ms", type=int, default=300)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--worker", action="store_true")
    ap.add_argument("--out")
    args = ap.parse_args()

    if args.worker:
        worker(args.pool, args.ms, args.threads, args.limit, args.out)
        return

    if not Path(FP_PY).exists():
        sys.exit(f"stock engine python not found: {FP_PY}")
    n_pool = sum(1 for _ in open(args.pool))
    n = min(n_pool, args.limit) if args.limit else n_pool
    print(f"pool: {n} positions from {args.pool}, {args.ms}ms searches, "
          f"{args.threads} threads")
    print("legs: fork (this repo) | fork-rerun (noise floor) | "
          "stock (upstream 0.0.47, fp venv)\n")

    tmp = Path(os.environ.get("TMPDIR", "/tmp")) / f"evaldiff_{os.getpid()}"
    tmp.mkdir(parents=True, exist_ok=True)
    fork = run_leg("fork", sys.executable, args, tmp)
    print("  fork done")
    rerun = run_leg("fork-rerun", sys.executable, args, tmp)
    print("  fork-rerun done")
    stock = run_leg("stock", FP_PY, args, tmp)
    print("  stock done\n")

    m = min(len(fork), len(rerun), len(stock))
    fork, rerun, stock = fork[:m], rerun[:m], stock[:m]
    recs = [json.loads(l) for l in open(args.pool)][:m]

    stock_flips = sum(1 for a, b in zip(fork, stock) if a != b)
    noise = sum(1 for a, b in zip(fork, rerun) if a != b)
    print(f"flip rate  fork-vs-STOCK : {100*stock_flips/m:5.1f}%  ({stock_flips}/{m})")
    print(f"flip rate  fork-vs-rerun : {100*noise/m:5.1f}%  ({noise}/{m})  <- MCTS noise floor\n")

    # category distributions side by side
    fc = Counter(category(x) for x in fork)
    sc = Counter(category(x) for x in stock)
    keys = sorted(set(fc) | set(sc), key=lambda k: -(fc[k] + sc[k]))
    print(f"{'category':13s} {'stock':>6s} {'fork':>6s} {'Δ(fork-stock)':>14s}")
    for k in keys:
        print(f"{k:13s} {sc[k]:6d} {fc[k]:6d} {fc[k]-sc[k]:+14d}")
    fp_press = sum(fc[k] for k in PRESSURE); sp_press = sum(sc[k] for k in PRESSURE)
    fp_house = sum(fc[k] for k in HOUSEKEEP); sp_house = sum(sc[k] for k in HOUSEKEEP)
    print(f"\n  pressure   moves: stock {sp_press:4d}  fork {fp_press:4d}  ({fp_press-sp_press:+d})")
    print(f"  housekeep  moves: stock {sp_house:4d}  fork {fp_house:4d}  ({fp_house-sp_house:+d})")

    # transition table on DISCORDANT positions: when they disagree, what does
    # each pick? this is the direct 'stock attacks, fork recovers' readout.
    trans = Counter()
    for a, b in zip(stock, fork):          # a=stock choice, b=fork choice
        if a != b:
            trans[(category(a), category(b))] += 1
    print(f"\ntop stock->fork category transitions on the {stock_flips} discordant positions:")
    for (sca, fca), c in trans.most_common(12):
        arrow = "  (stock pressure -> fork housekeep)" if sca in PRESSURE and fca in HOUSEKEEP \
            else "  (stock housekeep -> fork pressure)" if sca in HOUSEKEEP and fca in PRESSURE \
            else ""
        print(f"  {sca:13s} -> {fca:13s}  x{c}{arrow}")

    print("\nsample discordant positions (up to 12):")
    shown = 0
    for i, (a, b) in enumerate(zip(stock, fork)):
        if a != b:
            print(f"  {recs[i].get('tag','?')} T{recs[i].get('turn','?')}: "
                  f"stock={a} | fork={b}")
            shown += 1
            if shown >= 12:
                break


if __name__ == "__main__":
    main()
