#!/usr/bin/env python3
"""Margin measurement: WHEN the roles rules disagree with the search, is the
search near-tied or confident? The missing half of roles_screen.py.

roles_screen showed the roles knowledge disagrees with real play a lot
(chipped full-HP setup 82%, weather-down 68%, early cleaner 66%) — but
disagreement is only actionable if we know what acting on it would override.
Near-tie: the rule is a cheap tie-break, the one control-flow hook this
campaign considers defensible. Confident: acting means overriding a decided
search, which this campaign has never won doing. Precedent says the answer
varies by decision type (T1 switches: 49pp average margin, zero near-ties;
T1 stays: 34% near-tied), so it must be measured per rule, not assumed.

Needs a position pool recorded with `gen9_player --dump-states POOL.jsonl`
AFTER 2026-07-31 (older pools lack the "ranked" field — only the winner was
recorded, which is exactly the number that can't give a margin). Joined to
the same run's ours-logs by (battle tag, turn); the rule encoding is imported
from roles_screen so it exists exactly once.

The margin priced is COMPLIANCE cost, not top1-vs-top2: the chosen action's
visit share minus the best rule-compliant alternative's (veto: best candidate
outside the vetoed set; demand: best candidate inside the demanded set). That
is the share the rule is asking us to walk away from.

  .venv/bin/python showdown/margin_screen.py \
      --pool showdown/bench/marginpool_states.jsonl \
      --logs 'showdown/bench/marginpool_L*_ours.log'
"""
from __future__ import annotations
import argparse, glob, json, statistics, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from roles_screen import games, opinions

NEAR, CONF = 0.05, 0.10   # share-gap labels: <5pp near-tie, >=10pp confident


def base(choice: str) -> str:
    return choice[:-5] if choice.endswith("-tera") else choice


def compliance_margin(op: dict, ranked: list) -> float | str:
    """chosen share minus best rule-compliant share, or a skip reason.

    Sanity-checks that the join agrees with the log parse: on a veto
    disagreement the pool's winner must itself be a vetoed choice (we
    disagreed BY picking it); on a demand disagreement it must not be a
    demanded one. A mismatch means tag/turn joined two different decisions —
    count it, never price it."""
    total = sum(v for _, v in ranked)
    if total <= 0:
        return "empty"
    share = {c: v / total for c, v in ranked}
    chosen = ranked[0][0]
    hit = base(chosen) in op["targets"] or chosen in op["targets"]
    if op["mode"] == "veto":
        if not hit:
            return "join-mismatch"
        alts = [share[c] for c, _ in ranked
                if base(c) not in op["targets"] and c not in op["targets"]]
    else:
        if hit:
            return "join-mismatch"
        alts = [share[c] for c, _ in ranked
                if base(c) in op["targets"] or c in op["targets"]]
    if not alts:
        return "no-alternative"   # e.g. demanded setter fainted/not an option
    return share[chosen] - max(alts)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pool", required=True)
    ap.add_argument("--logs", required=True,
                    help="glob for the SAME run's ours-logs (quote it)")
    args = ap.parse_args()

    index: dict[tuple, list] = {}
    n_pool = n_old = 0
    with open(args.pool) as f:
        for line in f:
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            n_pool += 1
            if "ranked" not in rec:
                n_old += 1
                continue
            index[(rec["tag"], rec["turn"])] = rec["ranked"]
    if n_old:
        print(f"  WARNING: {n_old}/{n_pool} pool records predate the 'ranked' "
              f"field and cannot be priced", file=sys.stderr)

    stats: dict = {}   # rule -> dict(margins=[], missed, skipped{reason}, agree_margins=[])
    n_games = 0
    for g in games(sorted(glob.glob(args.logs))):
        n_games += 1
        for op in opinions(g):
            st = stats.setdefault(op["rule"], dict(
                margins=[], agree=0, missed=0, skipped={}))
            if not op["disagreed"]:
                st["agree"] += 1
                continue
            ranked = index.get((g["tag"], op["turn"]))
            if ranked is None:      # forced switch, preview entry, or lost turn
                st["missed"] += 1
                continue
            m = compliance_margin(op, ranked)
            if isinstance(m, str):
                st["skipped"][m] = st["skipped"].get(m, 0) + 1
            else:
                st["margins"].append(m)

    print(f"  COMPLIANCE MARGIN WHERE ROLES RULES DISAGREE WITH THE SEARCH")
    print(f"  pool: {len(index)} priced positions; logs: {n_games} games\n")
    print(f"  {'rule':56s} {'n':>4s} {'p25':>6s} {'p50':>6s} {'p75':>6s} "
          f"{'<5pp':>5s} {'>=10pp':>7s}")
    for rule, st in sorted(stats.items(), key=lambda kv: -len(kv[1]["margins"])):
        ms = sorted(st["margins"])
        if ms:
            q = statistics.quantiles(ms, n=4) if len(ms) >= 4 else [ms[0], ms[len(ms)//2], ms[-1]]
            near = sum(1 for m in ms if m < NEAR) / len(ms)
            conf = sum(1 for m in ms if m >= CONF) / len(ms)
            print(f"  {rule:56s} {len(ms):4d} {q[0]:6.3f} {q[1]:6.3f} {q[2]:6.3f} "
                  f"{100*near:4.0f}% {100*conf:6.0f}%")
        else:
            print(f"  {rule:56s} {0:4d}      -      -      -     -       -")
        extras = [f"agreements {st['agree']}", f"unjoined {st['missed']}"]
        extras += [f"{k} {v}" for k, v in st["skipped"].items()]
        print(f"    ({', '.join(extras)})")
    print(f"\n  <5pp share gap = near-tie (tie-break territory); "
          f">=10pp = the search was confident\n  (override territory — "
          f"the campaign is 0-for-N overriding those)")


if __name__ == "__main__":
    main()
