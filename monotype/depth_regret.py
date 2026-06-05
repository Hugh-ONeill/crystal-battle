"""Depth-regret blunder finder — the engine as its own oracle.

At each decision, compare a shallow pick to a deep search. Flag a "regret
blunder" when the deep search CONFIDENTLY prefers a different move (deep_best
visit-share > 30%) that the shallow pick barely got (< 8%). Caught with no
opponent/oracle needed.

Runs three configs against the SAME deep oracle to test the levers:
  500ms        : baseline
  1000ms       : does doubling search reduce regret?
  500ms+heal   : does the heal-heuristic kill the recovery-theme regrets?

  .venv/bin/python -m monotype.depth_regret
"""
from __future__ import annotations
import random, time
from pathlib import Path
import multiprocessing as mp

import poke_engine as pe
from showdown.bench_monotype import (build_pe_state_gen9, _best_useful,
                                     _strip_switch_prefix, _normalize_no_move)
from monotype.heuristics import recovery_override
from monotype.blunder_audit import parse

DEEP_MS = 2500
MAX_REGRET_TURN = 16
RECOVERY = {"recover", "roost", "softboiled", "slackoff", "moonlight", "wish",
            "lifedew", "strengthsap", "rest", "painsplit"}


def _share(res, mc):
    tot = sum(x.visits for x in res) or 1
    for x in res:
        if x.move_choice == mc:
            return x.visits / tot
    return 0.0


def regret_game(t1, t2, seed, shallow_ms, use_heal, maxt=70):
    random.seed(seed)
    st = build_pe_state_gen9(t1, t2)
    flags = []
    prev = ""; stuck = 0
    ph = {1: None, 2: None}; pid = {1: None, 2: None}
    for turn in range(maxt):
        if sum(p.hp > 0 for p in st.side_one.pokemon) == 0 or sum(p.hp > 0 for p in st.side_two.pokemon) == 0:
            break
        s = st.to_string()
        try:
            r1 = pe.monte_carlo_tree_search(pe.State.from_string(s), duration_ms=shallow_ms)
            r2 = pe.monte_carlo_tree_search(pe.State.from_string(s), duration_ms=shallow_ms)
        except Exception:
            break
        p1 = _best_useful(r1.side_one, st.side_two.side_conditions)
        p2 = _best_useful(r2.side_two, st.side_one.side_conditions)
        if use_heal:
            o1 = recovery_override(st.side_one, r1.side_one, ph[1], pid[1])
            o2 = recovery_override(st.side_two, r2.side_two, ph[2], pid[2])
            if o1: p1 = o1
            if o2: p2 = o2
        if turn < MAX_REGRET_TURN:
            try:
                d1 = pe.monte_carlo_tree_search(pe.State.from_string(s), duration_ms=DEEP_MS)
                d2 = pe.monte_carlo_tree_search(pe.State.from_string(s), duration_ms=DEEP_MS)
            except Exception:
                d1 = d2 = None
            for sd, (pick, dres, ocond, me) in {
                    1: (p1, d1.side_one if d1 else None, st.side_two.side_conditions, st.side_one),
                    2: (p2, d2.side_two if d2 else None, st.side_one.side_conditions, st.side_two)}.items():
                if dres is None or me.force_switch or pick == "No Move":
                    continue
                deep_best = _best_useful(dres, ocond)
                if deep_best == pick:
                    continue
                if _share(dres, deep_best) > 0.30 and _share(dres, pick) < 0.08:
                    is_rec = deep_best.replace(" ", "").lower() in RECOVERY
                    flags.append((me.pokemon[int(me.active_index)].id, pick, deep_best, is_rec))
        # snapshot prev (for next turn's heal override), then advance via shallow
        for sd, side in ((1, st.side_one), (2, st.side_two)):
            if side.active_index is not None:
                a = side.pokemon[int(side.active_index)]
                ph[sd], pid[sd] = a.hp, a.id
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
    return flags


_T = None
_CFG = None
def _init(T, cfg):
    global _T, _CFG
    _T, _CFG = T, cfg


def _run(task):
    i, j, seed = task
    shallow_ms, use_heal = _CFG
    return regret_game(_T[i][1], _T[j][1], seed, shallow_ms, use_heal)


def main():
    teams = parse("monotype/teams/teams_engine.txt")
    n = len(teams)
    pairs = [(i, (i + k) % n) for i in range(n) for k in (4, 9)]   # 36 matchups
    tasks = [(i, j, 200 + i) for (i, j) in pairs]
    configs = [("500ms", (500, False)), ("1000ms", (1000, False)), ("500ms+heal", (500, True))]
    print(f"=== depth-regret levers: {len(tasks)} games/config, deep={DEEP_MS}ms, turns<{MAX_REGRET_TURN} ===\n", flush=True)
    results = {}
    for name, cfg in configs:
        t0 = time.time(); flags = []
        with mp.Pool(20, initializer=_init, initargs=(teams, cfg)) as pool:
            for fl in pool.imap_unordered(_run, tasks):
                flags.extend(fl)
        rec = sum(1 for *_, isr in flags if isr)
        results[name] = (len(flags), rec)
        print(f"  {name:12} {len(flags):3d} regret ({len(flags)/len(tasks):.2f}/game), "
              f"{rec:3d} recovery-theme ({rec/len(tasks):.2f}/game)   [{time.time()-t0:.0f}s]", flush=True)
    print("\n  lever read:")
    b = results["500ms"][0]
    print(f"    search:  500ms {b} -> 1000ms {results['1000ms'][0]}  "
          f"({100*(b-results['1000ms'][0])/max(1,b):+.0f}% regret)")
    rb = results["500ms"][1]
    print(f"    heal:    recovery-regret {rb} -> {results['500ms+heal'][1]}  "
          f"({100*(rb-results['500ms+heal'][1])/max(1,rb):+.0f}%)")


if __name__ == "__main__":
    main()
