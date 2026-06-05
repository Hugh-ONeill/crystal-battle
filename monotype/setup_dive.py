"""Deep-dive on setup_and_die: when a mon sets up then faints WITHOUT using the
boost, capture the visit distribution + eval at the setup decision. Tells us
whether the engine confidently chose setup (dominant visits = real judgment
error) or it was a flat-eval near-tie (same pathology as wasted hazards).

  .venv/bin/python -m monotype.setup_dive
"""
from __future__ import annotations
import random, re, time
from pathlib import Path
import multiprocessing as mp

import poke_engine as pe
from showdown.bench_monotype import (build_pe_state_gen9, _best_useful,
                                     _strip_switch_prefix, _normalize_no_move)
from monotype.blunder_audit import parse, SETUP, STATUS, _norm


def _vis(side_res, k=4):
    tot = sum(x.visits for x in side_res) or 1
    xs = sorted(side_res, key=lambda x: -x.visits)[:k]
    return ", ".join(f"{x.move_choice}({100*x.visits/tot:.0f}%)" for x in xs)


def dive_game(t1, t2, seed, maxt=80):
    random.seed(seed)
    st = build_pe_state_gen9(t1, t2)
    recs = []
    watch = {1: None, 2: None}   # [mon, turn, visits, eval, used]
    for turn in range(maxt):
        if sum(p.hp > 0 for p in st.side_one.pokemon) == 0 or sum(p.hp > 0 for p in st.side_two.pokemon) == 0:
            break
        s = st.to_string()
        try:
            r1 = pe.monte_carlo_tree_search(pe.State.from_string(s), duration_ms=500)
            r2 = pe.monte_carlo_tree_search(pe.State.from_string(s), duration_ms=500)
            ev = float(pe.evaluate(st))
        except Exception:
            break
        p1 = _best_useful(r1.side_one, st.side_two.side_conditions)
        p2 = _best_useful(r2.side_two, st.side_one.side_conditions)
        info = {1: (p1, r1.side_one, st.side_one), 2: (p2, r2.side_two, st.side_two)}
        for sd in (1, 2):
            mv, res, me = info[sd]
            if me.force_switch or mv.startswith("switch ") or mv == "No Move":
                continue
            nm = _norm(mv); act = me.pokemon[int(me.active_index)]
            if nm in SETUP:
                watch[sd] = [act.id, turn, _vis(res), ev if sd == 1 else -ev, False]
            elif nm not in STATUS and watch[sd] and watch[sd][0] == act.id:
                watch[sd][4] = True   # booster attacked -> boost used
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
        for sd in (1, 2):
            if watch[sd]:
                mon, t0, vs, e0, used = watch[sd]
                sside = st.side_one if sd == 1 else st.side_two
                alive = any(p.id == mon and p.hp > 0 for p in sside.pokemon)
                if not alive:
                    if not used:
                        recs.append((mon, t0, e0, vs))   # setup-and-die, boost unused
                    watch[sd] = None
                elif turn - t0 >= 2:
                    watch[sd] = None
    return recs


_T = None
def _init(T):
    global _T
    _T = T


def _run(task):
    i, j, seed = task
    return dive_game(_T[i][1], _T[j][1], seed)


def main():
    teams = parse("monotype/teams/teams_engine.txt")
    n = len(teams)
    # matchups likely to feature setup sweepers vs offense
    pairs = [(i, (i + k) % n) for i in range(n) for k in (3, 7, 13)]
    tasks = [(i, j, 100 + (i * 7 + j)) for (i, j) in pairs]
    print(f"=== setup_and_die deep-dive: {len(tasks)} games @ 500ms ===", flush=True)
    recs = []
    t0 = time.time(); done = 0
    with mp.Pool(22, initializer=_init, initargs=(teams,)) as pool:
        for r in pool.imap_unordered(_run, tasks):
            recs.extend(r)
            done += 1
            if done % max(1, len(tasks) // 10) == 0:
                print(f"  [{done}/{len(tasks)}] {time.time()-t0:.0f}s", flush=True)
    print(f"\nDone in {time.time()-t0:.0f}s — {len(recs)} setup-and-die (boost unused) instances\n")
    # was the setup move dominant in visits, or a flat near-tie?
    import re as _re
    dom = flat = 0
    for mon, turn, ev, vs in recs:
        top_pct = int(_re.search(r"\((\d+)%\)", vs).group(1)) if _re.search(r"\((\d+)%\)", vs) else 0
        tag = "CONFIDENT" if top_pct >= 45 else "flat-tie"
        if top_pct >= 45: dom += 1
        else: flat += 1
        print(f"  {mon:16} eval@setup={ev:+7.1f}  top={top_pct:2d}%  [{tag}]  {vs}")
    print(f"\n  summary: {dom} confident-setup-then-died  vs  {flat} flat-eval-near-tie  (of {len(recs)})")


if __name__ == "__main__":
    main()
