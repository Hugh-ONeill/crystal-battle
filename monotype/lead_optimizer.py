"""Per-team slot-1 lead optimizer.

The bench (and team-order play) hard-leads slot 1, and the imitation lead net
is unreliable at choosing OUR lead (it ranks Steel's Heatran 3rd and leads
Archaludon, an 11pp blunder). So pick each team's best *fixed* lead by search:
for every candidate lead, average the turn-1 MCTS root value across the field
(opponents leading their current slot 1), and take the argmax. Cheap because
the opponent's lead is fixed — 6 x (N-1) evals per team, not 36 per matchup.

Output: per team, current slot-1 vs search-suggested lead (and the net's pick
for contrast), so reorders can be eyeballed before committing.

  .venv/bin/python -m monotype.lead_optimizer --teams-file monotype/teams/teams_v7.txt --search-ms 500
"""
from __future__ import annotations

import argparse
import re
import sys
import time
from pathlib import Path
import multiprocessing as mp

sys.path.insert(0, str(Path(__file__).parent.parent))

import poke_engine as pe
from showdown.local_battle import build_pe_state_gen9
from monotype.lead_picker import (split_team_body, reorder_team, species_of,
                                  _root_value_from_mcts, load_lead_net, pick_leads_net)


def parse_teams(path):
    text = Path(path).read_text()
    parts = re.split(r'(?m)^=== \[gen9monotype\] (.+?) ===\s*$', text)
    return [(parts[i].strip(), parts[i + 1].strip()) for i in range(1, len(parts), 2)]


_TEAMS = None
def _init(teams):
    global _TEAMS
    _TEAMS = teams


def _eval(args):
    ti, li, oi, ms = args
    try:
        state = build_pe_state_gen9(reorder_team(_TEAMS[ti][1], li), _TEAMS[oi][1])
        r = pe.monte_carlo_tree_search(state, duration_ms=ms)
        return (ti, li, _root_value_from_mcts(r))
    except Exception:
        return (ti, li, 0.0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--teams-file", required=True)
    ap.add_argument("--search-ms", type=int, default=500)
    ap.add_argument("--workers", type=int, default=22)
    ap.add_argument("--lead-net", default="monotype/lead_net.pt")
    args = ap.parse_args()

    teams = parse_teams(args.teams_file)
    n = len(teams)
    mons = [split_team_body(b) for _, b in teams]
    names = [[species_of(m) for m in t] for t in mons]

    # net's avg top-1 lead per team (over the field), for contrast
    net = load_lead_net(args.lead_net)
    net_top = []
    for ti in range(n):
        import numpy as np
        acc = np.zeros(len(mons[ti]))
        for oi in range(n):
            if oi == ti:
                continue
            _, _, sc = pick_leads_net(teams[ti][1], teams[oi][1], net)
            acc += np.array(sc[0][:len(mons[ti])])
        net_top.append(names[ti][int(acc.argmax())])

    # MCTS: avg root value of each candidate lead vs the field
    tasks = [(ti, li, oi, args.search_ms)
             for ti in range(n) for li in range(len(mons[ti])) for oi in range(n) if oi != ti]
    print(f"=== lead-optimizer: {n} teams, {len(tasks)} MCTS evals "
          f"@ {args.search_ms}ms, {args.workers} workers ===", flush=True)
    t0 = time.time()
    agg = {}  # (ti, li) -> [sum, count]
    with mp.Pool(args.workers, initializer=_init, initargs=(teams,)) as pool:
        for k, (ti, li, val) in enumerate(pool.imap_unordered(_eval, tasks, chunksize=4)):
            s = agg.setdefault((ti, li), [0.0, 0])
            s[0] += val
            s[1] += 1
            if (k + 1) % max(1, len(tasks) // 10) == 0:
                print(f"  [{k+1}/{len(tasks)}] {time.time()-t0:.0f}s", flush=True)

    print(f"\nDone in {time.time()-t0:.0f}s\n")
    hdr = f"{'team':26} {'cur slot-1':13} {'net pick':13} {'SEARCH best':13}  Δvalue  change?"
    print(hdr)
    print("-" * len(hdr))
    reorders = []
    for ti in range(n):
        vals = {li: agg[(ti, li)][0] / agg[(ti, li)][1] for li in range(len(mons[ti]))}
        best_li = max(vals, key=vals.get)
        cur = names[ti][0]
        best = names[ti][best_li]
        dv = vals[best_li] - vals[0]
        change = "" if best_li == 0 else f"  <-- {cur}→{best}"
        if best_li != 0:
            reorders.append((teams[ti][0], cur, best, best_li, dv))
        print(f"{teams[ti][0]:26} {cur:13} {net_top[ti]:13} {best:13}  {dv:+6.1f}{change}")

    print(f"\n{len(reorders)} teams would change slot-1 lead:")
    for nm, cur, best, _, dv in sorted(reorders, key=lambda r: -r[4]):
        print(f"  {nm:26} {cur:13} -> {best:13} (+{dv:.1f} root value)")


if __name__ == "__main__":
    main()
