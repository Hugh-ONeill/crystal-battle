"""Eval blind-spot net — status moves into immunities.

`calculate_damage` can't see these (status deals 0 damage either way), so if the
engine's eval doesn't model the immunity it'll happily click a move that CANNOT
work — and BOTH search depths share the error, so depth-regret misses it too.
This is a direct test of whether the engine has those blind spots.

Flags a dedicated status move (Toxic / Will-O-Wisp / Thunder Wave / Spore /
Sleep Powder) chosen against a target that is type/ability/already-status/sub
immune to it.

  .venv/bin/python -m monotype.eval_blindspot
"""
from __future__ import annotations
import random, time
from collections import Counter
from pathlib import Path
import multiprocessing as mp

import poke_engine as pe
from showdown.bench_monotype import (build_pe_state_gen9, _best_useful,
                                     _strip_switch_prefix, _normalize_no_move)
from monotype.blunder_audit import parse, _norm


def status_immune(move, tgt):
    """Return a reason string if `move` (a status move) cannot affect `tgt`, else None."""
    t = set(tgt.types)
    ab = (tgt.ability or "").lower()
    if (getattr(tgt, "substitute_health", 0) or 0) > 0:
        return "behind Substitute"
    cur = str(tgt.status).lower()
    if cur not in ("none", "status.none", ""):
        return f"already {cur}"
    if move == "toxic":
        if t & {"poison", "steel"}:
            return "poison-immune type"
        if ab == "immunity":
            return "ability Immunity"
    elif move == "willowisp":
        if "fire" in t:
            return "Fire (burn-immune)"
        if ab in {"waterveil", "waterbubble", "thermalexchange", "comatose"}:
            return f"ability {ab}"
    elif move == "thunderwave":
        if "ground" in t:
            return "Ground (Electric-immune)"
        if "electric" in t:
            return "Electric (para-immune)"
        if ab == "limber":
            return "ability Limber"
    elif move in ("spore", "sleeppowder"):
        if "grass" in t:
            return "Grass (powder-immune)"
        if ab in {"insomnia", "vitalspirit", "comatose", "sweetveil", "overcoat"}:
            return f"ability {ab}"
        if (tgt.item or "").lower() == "safetygoggles":
            return "Safety Goggles"
    return None


STATUS_MOVES = {"toxic", "willowisp", "thunderwave", "spore", "sleeppowder"}


def scan_game(t1, t2, seed, maxt=80):
    random.seed(seed)
    st = build_pe_state_gen9(t1, t2)
    flags = []
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
        oa1 = st.side_one.pokemon[int(st.side_one.active_index)] if st.side_one.active_index is not None else None
        oa2 = st.side_two.pokemon[int(st.side_two.active_index)] if st.side_two.active_index is not None else None
        p1 = _best_useful(r1.side_one, st.side_two.side_conditions, oa2)
        p2 = _best_useful(r2.side_two, st.side_one.side_conditions, oa1)
        for sd, (mv, me, opp) in {1: (p1, st.side_one, st.side_two),
                                  2: (p2, st.side_two, st.side_one)}.items():
            if me.force_switch or mv.startswith("switch ") or mv == "No Move":
                continue
            nm = _norm(mv)
            if nm in STATUS_MOVES:
                tgt = opp.pokemon[int(opp.active_index)]
                reason = status_immune(nm, tgt)
                if reason:
                    flags.append((me.pokemon[int(me.active_index)].id, nm, tgt.id, reason))
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
def _init(T):
    global _T
    _T = T


def _run(task):
    i, j, seed = task
    return scan_game(_T[i][1], _T[j][1], seed)


def main():
    teams = parse("monotype/teams/teams_engine.txt")
    n = len(teams)
    # bias toward matchups where these statuses meet immunities (Steel/Poison/
    # Fire/Ground/Grass/Electric teams), but sweep broadly
    pairs = [(i, (i + k) % n) for i in range(n) for k in (1, 5, 9, 13)]
    tasks = [(i, j, 300 + i) for (i, j) in pairs]
    print(f"=== eval blind-spot (status-into-immune): {len(tasks)} self-play games @ 500ms ===", flush=True)
    flags = []
    t0 = time.time(); done = 0
    with mp.Pool(22, initializer=_init, initargs=(teams,)) as pool:
        for fl in pool.imap_unordered(_run, tasks):
            flags.extend(fl)
            done += 1
            if done % max(1, len(tasks) // 10) == 0:
                print(f"  [{done}/{len(tasks)}] {time.time()-t0:.0f}s", flush=True)
    print(f"\nDone in {time.time()-t0:.0f}s — {len(flags)} status-into-immune ({len(flags)/len(tasks):.2f}/game)\n")
    by = Counter(f"{mv} -> {reason}" for _, mv, _, reason in flags)
    for k, c in by.most_common():
        ex = next(f"{u}->{t}" for u, mv, t, r in flags if f"{mv} -> {r}" == k)
        print(f"  {c:3d}x  {k:34}  e.g. {ex}")


if __name__ == "__main__":
    main()
