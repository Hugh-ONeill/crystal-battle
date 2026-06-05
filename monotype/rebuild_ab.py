"""A/B a rebuilt team (and variants) against the field, with lead-net leads.

Compares one team's variants (current + rebuilds) vs the rest of teams_engine,
using the lead net to pick each side's lead per matchup (no slot-1 confound).

  .venv/bin/python -m monotype.rebuild_ab
"""
from __future__ import annotations
import random, re, time
from pathlib import Path
import multiprocessing as mp

from showdown.bench_monotype import play_one
from monotype.lead_picker import load_lead_net, pick_leads_net

REPLACED = "Stop Bugging Me"
VARIANTS = [
    ("current", None),
    ("HO", "monotype/bench/_bug_ho.txt"),
]


def parse(p):
    t = Path(p).read_text()
    s = re.split(r'(?m)^=== \[gen9monotype\] (.+?) ===\s*$', t)
    return [(s[i].strip(), s[i + 1].strip()) for i in range(1, len(s), 2)]


_V = _F = _L0 = _L1 = None
_MS = 500
def _init(V, F, L0, L1, ms):
    global _V, _F, _L0, _L1, _MS
    _V, _F, _L0, _L1, _MS = V, F, L0, L1, ms


def _game(task):
    vi, oi, direction, seed = task
    random.seed(seed)
    if direction == 0:
        lv, lo = _L0[(vi, oi)]
        r = play_one(_V[vi], _F[oi], _MS, lead_p1=lv, lead_p2=lo)
        return (vi, 1 if r == 1 else (0 if r == 2 else -1))
    lo, lv = _L1[(vi, oi)]
    r = play_one(_F[oi], _V[vi], _MS, lead_p1=lo, lead_p2=lv)
    return (vi, 1 if r == 2 else (0 if r == 1 else -1))


def main():
    teams = parse("monotype/teams/teams_engine.txt")
    tdict = dict(teams)
    field = [b for n, b in teams if n != REPLACED]
    V = []
    for lab, fp in VARIANTS:
        body = tdict[REPLACED] if fp is None else parse(fp)[0][1]
        V.append((lab, body))
    bodies = [b for _, b in V]

    print(f"=== rebuild A/B: {len(V)} variants of {REPLACED} vs {len(field)} field "
          f"(lead-net leads) ===", flush=True)
    net = load_lead_net("monotype/lead_net.pt")
    L0, L1 = {}, {}
    for vi in range(len(V)):
        for oi in range(len(field)):
            lv, lo, _ = pick_leads_net(bodies[vi], field[oi], net)
            L0[(vi, oi)] = (lv, lo)
            lo2, lv2, _ = pick_leads_net(field[oi], bodies[vi], net)
            L1[(vi, oi)] = (lo2, lv2)

    GAMES = 4
    tasks = [(vi, oi, d, 42 + g) for vi in range(len(V)) for oi in range(len(field))
             for g in range(GAMES) for d in (0, 1)]
    rec = {vi: [0, 0, 0] for vi in range(len(V))}
    t0 = time.time(); done = 0
    with mp.Pool(22, initializer=_init, initargs=(bodies, field, L0, L1, 500)) as pool:
        for vi, o in pool.imap_unordered(_game, tasks, chunksize=2):
            rec[vi][0 if o == 1 else 1 if o == 0 else 2] += 1
            done += 1
            if done % max(1, len(tasks) // 10) == 0:
                print(f"  [{done}/{len(tasks)}] {time.time()-t0:.0f}s", flush=True)
    print(f"\nDone in {time.time()-t0:.0f}s\n")
    for vi, (lab, _) in enumerate(V):
        w, l, d = rec[vi]
        wr = 100 * w / (w + l) if (w + l) else 0
        print(f"  {lab:12} {wr:5.1f}%   {w}W/{l}L/{d}D")


if __name__ == "__main__":
    main()
