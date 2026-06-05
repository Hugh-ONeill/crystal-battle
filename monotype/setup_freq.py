"""How often does the engine actually use setup in self-play?

Counts offensive setup-move uses vs total move-decisions across dev-vs-dev games,
to put a hard number on "frequently enough to matter".

  .venv/bin/python -m monotype.setup_freq
"""
from __future__ import annotations
import random, time
from pathlib import Path
import multiprocessing as mp

import poke_engine as pe
from showdown.bench_monotype import (build_pe_state_gen9, _best_useful,
                                     _strip_switch_prefix, _normalize_no_move)
from monotype.blunder_audit import parse, _norm

SETUP = {"swordsdance", "dragondance", "quiverdance", "nastyplot", "calmmind",
         "bulkup", "victorydance", "bellydrum", "irondefense", "curse", "trailblaze"}


def scan(t1, t2, seed, maxt=90):
    random.seed(seed)
    st = build_pe_state_gen9(t1, t2)
    setups = moves = turns = 0
    prev = ""; stuck = 0
    for _ in range(maxt):
        if sum(p.hp > 0 for p in st.side_one.pokemon) == 0 or sum(p.hp > 0 for p in st.side_two.pokemon) == 0:
            break
        s = st.to_string()
        try:
            r1 = pe.monte_carlo_tree_search(pe.State.from_string(s), duration_ms=500)
            r2 = pe.monte_carlo_tree_search(pe.State.from_string(s), duration_ms=500)
        except Exception:
            break
        p1 = _best_useful(r1.side_one, st.side_two.side_conditions)
        p2 = _best_useful(r2.side_two, st.side_one.side_conditions)
        for sd, (mv, me) in {1: (p1, st.side_one), 2: (p2, st.side_two)}.items():
            if me.force_switch or mv == "No Move":
                continue
            turns += 1
            if mv.startswith("switch "):
                continue
            moves += 1
            if _norm(mv) in SETUP:
                setups += 1
        p1 = _normalize_no_move(p1); p2 = _normalize_no_move(p2)
        if p1 == "No Move" and p2 == "No Move":
            break
        try:
            ins = pe.generate_instructions(st, _strip_switch_prefix(p1), _strip_switch_prefix(p2))
        except Exception:
            break
        if not ins:
            break
        roll = random.random() * 100; cum = 0.0; ch = ins[0]
        for i in ins:
            cum += i.percentage
            if roll <= cum:
                ch = i; break
        st = st.apply_instructions(ch)
        cs = st.to_string()
        if cs == prev:
            stuck += 1
            if stuck >= 3:
                break
        else:
            stuck = 0
        prev = cs
    return (setups, moves, turns, 1 if setups else 0)


_T = None
def _init(T):
    global _T
    _T = T


def _run(task):
    i, j, seed = task
    return scan(_T[i][1], _T[j][1], seed)


def main():
    teams = parse("monotype/teams/teams_engine.txt")
    n = len(teams)
    pairs = [(i, (i + k) % n) for i in range(n) for k in (2, 7, 12)]
    tasks = [(i, j, 400 + i) for (i, j) in pairs]
    print(f"=== setup frequency: {len(tasks)} self-play games @ 500ms ===", flush=True)
    S = M = T = G = 0
    t0 = time.time(); done = 0
    with mp.Pool(22, initializer=_init, initargs=(teams,)) as pool:
        for s, m, t, g in pool.imap_unordered(_run, tasks):
            S += s; M += m; T += t; G += g
            done += 1
    print(f"\nDone in {time.time()-t0:.0f}s over {len(tasks)} games\n")
    print(f"  setup moves used:        {S}  ({S/len(tasks):.2f} per game)")
    print(f"  as % of all move-clicks: {100*S/max(1,M):.1f}%  (vs {M} attacking/status moves)")
    print(f"  as % of all decisions:   {100*S/max(1,T):.1f}%  (incl. switches; {T} total decisions)")
    print(f"  games with >=1 setup:    {G}/{len(tasks)}  ({100*G/len(tasks):.0f}%)")


if __name__ == "__main__":
    main()
