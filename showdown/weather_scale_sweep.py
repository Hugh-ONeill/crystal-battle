#!/usr/bin/env python3
"""Find the smallest CB_WEATHERTEAM_SCALE that actually flips the decisions.

The first weatherab A/B moved NOTHING behaviorally (setter-switch decline 63%
vs 64% at scale 1.0) — the term's ~7-25 eval points are a rounding error
against 20-45pp visit-share margins. Instead of guessing a bigger constant
and spending another 480-game A/B per guess, this sweep re-searches the
KNOWN weather-disagreement positions (margin_screen join over a --dump-states
pool) under increasing scales and reports how many now choose a roles-
demanded setter switch. The paired A/B is then spent ONCE, at the knee.

Env is read once per process (OnceLock), so each scale runs in its own
subprocess; searches thread inside it (the mcts binding releases the GIL).

  .venv/bin/python showdown/weather_scale_sweep.py \
      --pool showdown/bench/marginpool_states.jsonl \
      --logs 'showdown/bench/marginpool_L*_ours.log' \
      --scales 1 2 4 8 --ms 600
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))

WORKER = r"""
import json, sys
from concurrent.futures import ThreadPoolExecutor
import poke_engine as pe

cases = [json.loads(l) for l in open(sys.argv[1])]
ms = int(sys.argv[2])

def run(case):
    try:
        res = pe.monte_carlo_tree_search(
            pe.State.from_string(case["state"]), ms)
        side = sorted(res.side_one, key=lambda r: -r.visits)
        top = side[0].move_choice if side else None
        return top in case["targets"]
    except Exception:
        return None

with ThreadPoolExecutor(max_workers=4) as ex:
    got = list(ex.map(run, cases))
ok = [g for g in got if g is not None]
print(json.dumps({"n": len(ok), "compliant": sum(ok),
                  "errors": len(got) - len(ok)}))
"""


def collect_cases(pool: str, logglob: str) -> list[dict]:
    from roles_screen import games, opinions
    index = {}
    for line in open(pool):
        try:
            r = json.loads(line)
        except json.JSONDecodeError:
            continue
        index[(r["tag"], r["turn"])] = r
    cases = []
    for g in games(sorted(glob.glob(logglob))):
        for op in opinions(g):
            if not op["rule"].startswith("our weather") or not op["disagreed"]:
                continue
            rec = index.get((g["tag"], op["turn"]))
            if rec:
                cases.append({"state": rec["state"],
                              "targets": list(op["targets"]),
                              "engine_choice": rec["choice"]})
    return cases


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pool", required=True)
    ap.add_argument("--logs", required=True)
    ap.add_argument("--scales", nargs="+", type=float, default=[1, 2, 4, 8])
    ap.add_argument("--ms", type=int, default=600)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    cases = collect_cases(args.pool, args.logs)
    if args.limit:
        cases = cases[: args.limit]
    print(f"{len(cases)} weather-disagreement positions "
          f"(engine declined a demanded setter switch at {args.ms}ms-class "
          f"search)")

    scratch = tempfile.NamedTemporaryFile(
        "w", suffix=".jsonl", delete=False,
        dir=os.environ.get("TMPDIR") or None)
    for c in cases:
        scratch.write(json.dumps(c) + "\n")
    scratch.close()

    py = sys.executable
    print(f"  {'scale':>6s} {'searched':>9s} {'now-compliant':>14s} {'rate':>6s}")
    for scale in [0.0] + args.scales:      # 0.0 = term off, the control row
        env = dict(os.environ)
        env.pop("CB_EVAL_ON", None)
        env.pop("CB_WEATHERTEAM_SCALE", None)
        label = "off"
        if scale > 0:
            env["CB_EVAL_ON"] = "weatherteam"
            env["CB_WEATHERTEAM_SCALE"] = str(scale)
            label = f"{scale:g}"
        out = subprocess.run([py, "-c", WORKER, scratch.name, str(args.ms)],
                             capture_output=True, text=True, env=env,
                             cwd=str(HERE.parent))
        try:
            r = json.loads(out.stdout.strip().splitlines()[-1])
        except Exception:
            print(f"  {label:>6s}  worker failed: {out.stderr.strip()[:120]}")
            continue
        rate = r["compliant"] / max(1, r["n"])
        print(f"  {label:>6s} {r['n']:9d} {r['compliant']:14d} {100*rate:5.1f}%"
              + (f"  ({r['errors']} errors)" if r["errors"] else ""))
    os.unlink(scratch.name)
    print("\n  Pick the knee: the smallest scale where a real fraction of the "
          "demanded\n  switches flip — then spend the paired A/B once, there. "
          "A scale that flips\n  ~everything is an override, not an eval "
          "correction; prefer partial flips.")


if __name__ == "__main__":
    main()
