"""Full-game lead sweep — the reliable lead picker.

Turn-1 MCTS root value does NOT differentiate leads (see lead_optimizer.py:
it picked Gholdengo for Steel, contradicting the +11pp Heatran lead-test).
Only full-game win rate measures lead value. So: for each team, take the net's
top-k leads as a *candidate filter* (it contains the best lead but can't rank
it — Heatran was Steel's 3rd) plus the current slot-1, and play full games of
each candidate vs the field (opponents at their current slot-1). Pick the lead
with the best win%.

Reuses bench_monotype.play_one with explicit lead indices, so it's the exact
same engine — just restricted to candidate-vs-field games (no field-vs-field).

  .venv/bin/python -m monotype.lead_sweep --teams-file monotype/teams/teams_v7.txt \
      --search-ms 500 --games 4 --top-k 3 --out-teams monotype/teams/teams_v7_leadopt.txt
"""
from __future__ import annotations

import argparse
import random
import re
import sys
import time
from pathlib import Path
import multiprocessing as mp

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np

from showdown.bench_monotype import play_one
from monotype.lead_picker import (split_team_body, reorder_team, species_of,
                                  load_lead_net, pick_leads_net)


def parse_teams(path):
    text = Path(path).read_text()
    parts = re.split(r'(?m)^=== \[gen9monotype\] (.+?) ===\s*$', text)
    return [(parts[i].strip(), parts[i + 1].strip()) for i in range(1, len(parts), 2)]


_T = None
_MS = 500
_MAXT = 120
def _init(teams, ms, maxt):
    global _T, _MS, _MAXT
    _T, _MS, _MAXT = teams, ms, maxt


def _game(task):
    """Play one full game; return (ti, li, outcome) with outcome 1=win 0=loss -1=draw for team ti."""
    ti, li, oi, direction, seed = task
    random.seed(seed)
    if direction == 0:  # team ti is side_one, leading li; opp leads slot-1
        r = play_one(_T[ti][1], _T[oi][1], _MS, max_turns=_MAXT, lead_p1=li, lead_p2=0)
        return (ti, li, 1 if r == 1 else (0 if r == 2 else -1))
    else:               # team ti is side_two
        r = play_one(_T[oi][1], _T[ti][1], _MS, max_turns=_MAXT, lead_p1=0, lead_p2=li)
        return (ti, li, 1 if r == 2 else (0 if r == 1 else -1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--teams-file", required=True)
    ap.add_argument("--search-ms", type=int, default=500)
    ap.add_argument("--games", type=int, default=4, help="games per direction (x2 counterbalanced)")
    ap.add_argument("--workers", type=int, default=22)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--top-k", type=int, default=3)
    ap.add_argument("--max-turns", type=int, default=120)
    ap.add_argument("--lead-net", default="monotype/lead_net.pt")
    ap.add_argument("--out-teams", default=None, help="write reordered (best-lead-first) teams here")
    args = ap.parse_args()

    teams = parse_teams(args.teams_file)
    n = len(teams)
    mons = [split_team_body(b) for _, b in teams]
    names = [[species_of(m) for m in t] for t in mons]

    # candidate leads per team: net top-k (filter) UNION current slot-1 (baseline)
    net = load_lead_net(args.lead_net)
    cands = []
    for ti in range(n):
        acc = np.zeros(len(mons[ti]))
        for oi in range(n):
            if oi == ti:
                continue
            _, _, sc = pick_leads_net(teams[ti][1], teams[oi][1], net)
            acc += np.array(sc[0][:len(mons[ti])])
        top = [int(x) for x in np.argsort(-acc)[:min(args.top_k, len(mons[ti]))]]
        if 0 not in top:
            top.append(0)
        cands.append(top)

    tasks = []
    for ti in range(n):
        for li in cands[ti]:
            for oi in range(n):
                if oi == ti:
                    continue
                for gi in range(args.games):
                    seed = args.seed + gi
                    tasks.append((ti, li, oi, 0, seed))
                    tasks.append((ti, li, oi, 1, seed))

    print(f"=== lead sweep: {n} teams, "
          f"{sum(len(c) for c in cands)} (team,lead) candidates, "
          f"{len(tasks)} games @ {args.search_ms}ms, {args.workers} workers ===", flush=True)
    t0 = time.time()
    rec = {}  # (ti, li) -> [w, l, d]
    with mp.Pool(args.workers, initializer=_init,
                 initargs=(teams, args.search_ms, args.max_turns)) as pool:
        for k, (ti, li, o) in enumerate(pool.imap_unordered(_game, tasks, chunksize=2)):
            s = rec.setdefault((ti, li), [0, 0, 0])
            s[0 if o == 1 else 1 if o == 0 else 2] += 1
            if (k + 1) % max(1, len(tasks) // 20) == 0:
                print(f"  [{k+1}/{len(tasks)}] {time.time()-t0:.0f}s", flush=True)

    print(f"\nDone in {time.time()-t0:.0f}s\n")
    hdr = f"{'team':26} {'CURRENT lead':14} {'win%':>6}   {'BEST lead':14} {'win%':>6}  Δpp"
    print(hdr); print("-" * len(hdr))
    new_bodies = []
    changes = []
    for ti in range(n):
        def wr(li):
            w, l, d = rec[(ti, li)]
            return 100 * w / (w + l) if (w + l) else 0.0
        cur_wr = wr(0)
        best_li = max(cands[ti], key=wr)
        best_wr = wr(best_li)
        tag = "" if best_li == 0 else f"   <-- {names[ti][0]}→{names[ti][best_li]}"
        print(f"{teams[ti][0]:26} {names[ti][0]:14} {cur_wr:5.1f}%   "
              f"{names[ti][best_li]:14} {best_wr:5.1f}%  {best_wr-cur_wr:+5.1f}{tag}")
        new_bodies.append(reorder_team(teams[ti][1], best_li))
        if best_li != 0:
            changes.append((teams[ti][0], names[ti][0], names[ti][best_li], best_wr - cur_wr))

    print(f"\n{len(changes)} teams improve by changing slot-1 lead:")
    for nm, cur, best, dv in sorted(changes, key=lambda r: -r[3]):
        print(f"  {nm:26} {cur:14} -> {best:14} (+{dv:.1f}pp)")

    if args.out_teams:
        out = "\n".join(f"=== [gen9monotype] {teams[ti][0]} ===\n\n{new_bodies[ti]}\n"
                        for ti in range(n)) + "\n"
        Path(args.out_teams).write_text(out)
        print(f"\nwrote reordered teams -> {args.out_teams}")


if __name__ == "__main__":
    main()
