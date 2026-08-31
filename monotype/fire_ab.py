"""Focused Fire retool A/B: current Disco Inferno (two leads) vs Torkoal retool,
each vs the field. Uses play_one directly (variant-vs-field only). Run with -m so
forkserver can import the module in workers (heredoc/stdin breaks on py3.14)."""
from __future__ import annotations
import random, re, time
from pathlib import Path
import multiprocessing as mp

from showdown.bench_monotype import play_one
from monotype.lead_picker import reorder_team


def parse(p):
    t = Path(p).read_text()
    s = re.split(r'(?m)^=== \[gen9monotype\] (.+?) ===\s*$', t)
    return [(s[i].strip(), s[i + 1].strip()) for i in range(1, len(s), 2)]


_V = _F = None
_MS = 500
_MAXT = 120
def _init(V, F, ms, maxt):
    global _V, _F, _MS, _MAXT
    _V, _F, _MS, _MAXT = V, F, ms, maxt


def _game(task):
    vi, oi, d, seed = task
    random.seed(seed)
    vb, ob = _V[vi], _F[oi]
    if d == 0:
        r = play_one(vb, ob, _MS, max_turns=_MAXT, lead_p1=0, lead_p2=0)
        return (vi, 1 if r == 1 else (0 if r == 2 else -1))
    r = play_one(ob, vb, _MS, max_turns=_MAXT, lead_p1=0, lead_p2=0)
    return (vi, 1 if r == 2 else (0 if r == 1 else -1))


def main():
    teams = parse("monotype/teams/teams_engine.txt")
    disco = [b for n, b in teams if n == "Disco Inferno"][0]
    field = [b for n, b in teams if n != "Disco Inferno"]
    retool = parse("monotype/bench/_fire_retool.txt")[0][1]
    labels = ["cur-Heatran-lead", "cur-Ogerpon-lead", "retool-Torkoal-lead"]
    V = [reorder_team(disco, 0), reorder_team(disco, 5), retool]
    GAMES = 4
    tasks = [(vi, oi, d, 42 + g) for vi in range(3) for oi in range(len(field))
             for g in range(GAMES) for d in (0, 1)]
    print(f"=== fire A/B: 3 variants x {len(field)} field x {GAMES*2} = {len(tasks)} games ===", flush=True)
    rec = {vi: [0, 0, 0] for vi in range(3)}
    t0 = time.time(); done = 0
    with mp.Pool(22, initializer=_init, initargs=(V, field, 500, 120)) as pool:
        for vi, o in pool.imap_unordered(_game, tasks, chunksize=2):
            rec[vi][0 if o == 1 else 1 if o == 0 else 2] += 1
            done += 1
            if done % max(1, len(tasks) // 10) == 0:
                print(f"  [{done}/{len(tasks)}] {time.time()-t0:.0f}s", flush=True)
    print(f"\nDone in {time.time()-t0:.0f}s\n")
    for vi in range(3):
        w, l, d = rec[vi]
        wr = 100 * w / (w + l) if (w + l) else 0
        print(f"  {labels[vi]:22} {wr:5.1f}%   {w}W/{l}L/{d}D")


if __name__ == "__main__":
    main()
