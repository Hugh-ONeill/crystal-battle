"""Deep-vs-shallow search counterfactual on dumped positions.

WHY (2026-07-23): the tempo ledger reframed the fp gap — at the current
config we hold early parity and lose the LONG games. Two rival explanations
for the late bleed: the search is time-starved at the live 300ms/world
budget (the payoff of pressure lines sits past its horizon), or the eval
genuinely prefers housekeeping even with unlimited time. This tool separates
them: re-search fixed dumped positions at the live budget and at ~10x, and
look at WHERE deep search abandons the shallow choice and in which category
direction. Both budgets run twice — each arm's rerun flip rate is its own
noise floor, and a deep floor that stays high marks genuinely ambiguous
positions rather than starvation.

Strata: late positions (turn >= 30) from long losses are the accused; late
positions from long wins and early positions from all games are contrasts
(is any effect loss-specific, late-specific, or global?).

Usage:
  deep_shallow.py stratify POOL.jsonl --logs "showdown/bench/poolrun_L*_ours.log" \
      --out-dir showdown/bench [--late-turn 30] [--long-game 41] [--cap 40]
  deep_shallow.py run showdown/bench/ds_*.jsonl [--shallow-ms 300] \
      [--deep-ms 3000] [--threads 8]
"""
import argparse
import glob
import json
import re
import subprocess
import sys
from collections import Counter
from pathlib import Path

from position_ab import category

ROOM_RE = re.compile(r">(battle-gen9ou-\d+)")


def game_info(log_paths):
    """tag -> {'winner': 'cb'|'fp', 'turns': int} from bench ours-logs."""
    info = {}
    for f in log_paths:
        cur = None
        for line in open(f, errors="replace"):
            m = ROOM_RE.search(line)
            if m:
                cur = m.group(1)
                info.setdefault(cur, {"winner": None, "turns": 0})
            i = line.find("|")
            if i < 0 or cur is None:
                continue
            parts = line.rstrip("\n").split("|")[1:]
            if not parts:
                continue
            if parts[0] == "turn":
                info[cur]["turns"] = max(info[cur]["turns"], int(parts[1]))
            elif parts[0] == "win" and len(parts) >= 2:
                info[cur]["winner"] = ("cb" if parts[1].startswith("CBGen9")
                                       else "fp")
    return {t: v for t, v in info.items() if v["winner"]}


def stride_cap(recs, cap):
    """At most `cap` records, evenly strided so one marathon game can't
    dominate a stratum."""
    if len(recs) <= cap:
        return recs
    step = len(recs) / cap
    return [recs[int(i * step)] for i in range(cap)]


def stratify(args):
    recs = [json.loads(l) for l in open(args.pool)]
    info = game_info(sorted(f for p in args.logs for f in glob.glob(p)))
    missing = {r["tag"] for r in recs} - set(info)
    if missing:
        print(f"WARNING: {len(missing)} tags with no decided outcome dropped")
    strata = {"ds_late_loss": lambda r, g: (g["winner"] == "fp"
                                            and g["turns"] >= args.long_game
                                            and r["turn"] >= args.late_turn),
              "ds_late_win": lambda r, g: (g["winner"] == "cb"
                                           and g["turns"] >= args.long_game
                                           and r["turn"] >= args.late_turn),
              "ds_early": lambda r, g: r["turn"] <= args.early_turn}
    for name, pred in strata.items():
        by_tag = {}
        for r in recs:
            g = info.get(r["tag"])
            if g and pred(r, g):
                by_tag.setdefault(r["tag"], []).append(r)
        out = [r for tag in sorted(by_tag)
               for r in stride_cap(sorted(by_tag[tag], key=lambda r: r["turn"]),
                                   args.cap)]
        path = Path(args.out_dir) / f"{name}.jsonl"
        path.write_text("".join(json.dumps(r) + "\n" for r in out))
        print(f"{name}: {len(out)} positions from {len(by_tag)} games "
              f"-> {path}")


def search_arm(pool_path, ms, threads):
    cmd = [sys.executable, str(Path(__file__).parent / "position_ab.py"),
           str(pool_path), "--worker", "--ms", str(ms),
           "--threads", str(threads)]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        sys.exit(f"worker failed for {pool_path} @{ms}ms: {r.stderr[-500:]}")
    return [x["choice"] for x in json.loads(r.stdout)]


PRESSURE = {"attack", "boost", "status_infl", "pivot"}


def grp(cat):
    return "pressure" if cat in PRESSURE else "housekeep"


def run(args):
    for pool in args.pools:
        recs = [json.loads(l) for l in open(pool)]
        n = len(recs)
        print(f"\n=== {Path(pool).stem}: {n} positions "
              f"({args.shallow_ms}ms vs {args.deep_ms}ms) ===")
        arms = {}
        for name, ms in (("shallow", args.shallow_ms),
                         ("shallow2", args.shallow_ms),
                         ("deep", args.deep_ms), ("deep2", args.deep_ms)):
            arms[name] = search_arm(pool, ms, args.threads)

        def flips(a, b):
            return [i for i in range(n) if arms[a][i] != arms[b][i]]

        sfl, dfl = flips("shallow", "shallow2"), flips("deep", "deep2")
        sd = flips("shallow", "deep")
        print(f"  noise floors: shallow {100 * len(sfl) / n:.1f}%, "
              f"deep {100 * len(dfl) / n:.1f}%")
        print(f"  shallow-vs-deep flips: {len(sd)}/{n} "
              f"({100 * len(sd) / n:.1f}%)")
        trans = Counter((category(arms["shallow"][i]), category(arms["deep"][i]))
                        for i in sd)
        gt = Counter((grp(a), grp(b)) for (a, b), c in trans.items()
                     for _ in range(c) if grp(a) != grp(b))
        print(f"  housekeep->pressure {gt[('housekeep', 'pressure')]}, "
              f"pressure->housekeep {gt[('pressure', 'housekeep')]} "
              f"(rest of flips stay within group)")
        for (a, b), c in trans.most_common(8):
            print(f"    {a:12s} -> {b:12s} {c}")
        for i in sd[:args.samples]:
            print(f"    e.g. {recs[i]['tag']} T{recs[i]['turn']}: "
                  f"{arms['shallow'][i]} -> {arms['deep'][i]}")


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("stratify")
    s.add_argument("pool")
    s.add_argument("--logs", nargs="+", required=True)
    s.add_argument("--out-dir", default="showdown/bench")
    s.add_argument("--late-turn", type=int, default=30)
    s.add_argument("--long-game", type=int, default=41)
    s.add_argument("--early-turn", type=int, default=12)
    s.add_argument("--cap", type=int, default=40)
    r = sub.add_parser("run")
    r.add_argument("pools", nargs="+")
    r.add_argument("--shallow-ms", type=int, default=300)
    r.add_argument("--deep-ms", type=int, default=3000)
    r.add_argument("--threads", type=int, default=8)
    r.add_argument("--samples", type=int, default=8)
    args = ap.parse_args()
    (stratify if args.cmd == "stratify" else run)(args)


if __name__ == "__main__":
    main()
