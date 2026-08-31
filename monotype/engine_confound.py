"""Self-play confound probe: re-run a known result with the OPPONENT piloted by
a different engine (poke_engine_ref) instead of the same engine (self-mirror).

Our side (P1) is always pe_dev. Opponent (P2) is dev (self-mirror baseline) or
ref (mirror broken). If the forced-setup Δ and the Ogerpon>Heatran lead gap hold
when the opponent is a *different* engine, the conclusions aren't an artifact of
the engine playing its identical self. (Caveat: ref is still an engine — shared
blind spots vs humans need Showdown, not this.)

  .venv/bin/python -m monotype.engine_confound
"""
from __future__ import annotations
import random, re, time
from pathlib import Path
import multiprocessing as mp

import poke_engine as pe_dev
import poke_engine_ref as pe_ref
from showdown.bench_monotype import (build_pe_state_gen9, reorder_team, _best_non_tera,
                                     _strip_switch_prefix, _normalize_no_move)
from monotype.lead_picker import split_team_body, species_of


def parse(p):
    t = Path(p).read_text()
    s = re.split(r'(?m)^=== \[gen9monotype\] (.+?) ===\s*$', t)
    return [(s[i].strip(), s[i + 1].strip()) for i in range(1, len(s), 2)]


def lead_with(body, sp):
    names = [species_of(b) for b in split_team_body(body)]
    return reorder_team(body, names.index(sp))


def _eng(eid):
    return pe_dev if eid == "dev" else pe_ref


def play_dual(team1, team2, search_ms, p1_eng, p2_eng, forced_p1, max_turns=120):
    """P1 uses p1_eng (+ forced_p1 overrides), P2 uses p2_eng. State owned by pe_dev."""
    state = build_pe_state_gen9(team1, team2)
    prev = ""
    stuck = 0
    for turn in range(max_turns):
        if sum(p.hp > 0 for p in state.side_one.pokemon) == 0:
            return 2
        if sum(p.hp > 0 for p in state.side_two.pokemon) == 0:
            return 1
        s = state.to_string()
        try:
            r1 = p1_eng.monte_carlo_tree_search(p1_eng.State.from_string(s), duration_ms=search_ms)
            r2 = p2_eng.monte_carlo_tree_search(p2_eng.State.from_string(s), duration_ms=search_ms)
        except Exception:
            return 0
        p1 = _best_non_tera(r1.side_one)
        p2 = _best_non_tera(r2.side_two)
        if turn in forced_p1 and forced_p1[turn] in {x.move_choice for x in r1.side_one}:
            p1 = forced_p1[turn]
        p1 = _normalize_no_move(p1)
        p2 = _normalize_no_move(p2)
        if p1 == "No Move" and p2 == "No Move":
            return 0
        try:
            ins = pe_dev.generate_instructions(state, _strip_switch_prefix(p1), _strip_switch_prefix(p2))
        except Exception:
            return 0
        if not ins:
            return 0
        roll = random.random() * 100
        cum = 0.0
        chosen = ins[0]
        for i in ins:
            cum += i.percentage
            if roll <= cum:
                chosen = i
                break
        state = state.apply_instructions(chosen)
        cs = state.to_string()
        if cs == prev:
            stuck += 1
            if stuck >= 3:
                return 0
        else:
            stuck = 0
        prev = cs
    return 0


_COND = _FIELD = None
_MS = 500
def _init(C, F, ms):
    global _COND, _FIELD, _MS
    _COND, _FIELD, _MS = C, F, ms


def _game(task):
    ci, opp, oi, seed = task
    random.seed(seed)
    _, body, forced = _COND[ci]
    r = play_dual(body, _FIELD[oi], _MS, pe_dev, _eng(opp), forced)  # P1 always dev
    return (ci, opp, 1 if r == 1 else (0 if r == 2 else -1))


def main():
    teams = parse("monotype/teams/teams_engine.txt")
    disco = [b for n, b in teams if n == "Disco Inferno"][0]
    field = [b for n, b in teams if n != "Disco Inferno"]
    heatran = lead_with(disco, "Heatran")
    ogerpon = lead_with(disco, "Ogerpon-Hearthflame")
    COND = [
        ("Heatran free",     heatran, {}),
        ("Heatran forceSR",  heatran, {0: "stealthrock"}),
        ("Ogerpon free",     ogerpon, {}),
    ]
    GAMES = 6
    tasks = [(ci, opp, oi, 42 + g)
             for ci in range(len(COND)) for opp in ("dev", "ref")
             for oi in range(len(field)) for g in range(GAMES)]
    print(f"=== engine-confound: {len(COND)} conditions x (dev/ref opp) x {len(field)} field x "
          f"{GAMES} = {len(tasks)} games ===", flush=True)
    rec = {(ci, opp): [0, 0, 0] for ci in range(len(COND)) for opp in ("dev", "ref")}
    t0 = time.time(); done = 0
    with mp.Pool(22, initializer=_init, initargs=(COND, field, 500)) as pool:
        for ci, opp, o in pool.imap_unordered(_game, tasks, chunksize=2):
            rec[(ci, opp)][0 if o == 1 else 1 if o == 0 else 2] += 1
            done += 1
            if done % max(1, len(tasks) // 20) == 0:
                print(f"  [{done}/{len(tasks)}] {time.time()-t0:.0f}s", flush=True)
    print(f"\nDone in {time.time()-t0:.0f}s\n")
    def wr(ci, opp):
        w, l, _ = rec[(ci, opp)]
        return 100 * w / (w + l) if (w + l) else 0.0
    print(f"  {'condition':18} {'vs DEV(self)':>13} {'vs REF(other)':>14}")
    for ci, (lab, _, _) in enumerate(COND):
        print(f"  {lab:18} {wr(ci,'dev'):11.1f}% {wr(ci,'ref'):13.1f}%")
    print(f"\n  forceSR Δ vs free:   dev {wr(1,'dev')-wr(0,'dev'):+.1f}   ref {wr(1,'ref')-wr(0,'ref'):+.1f}")
    print(f"  Ogerpon-Heatran gap: dev {wr(2,'dev')-wr(0,'dev'):+.1f}   ref {wr(2,'ref')-wr(0,'ref'):+.1f}")


if __name__ == "__main__":
    main()
