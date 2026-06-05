"""Forced-opening experiment.

The engine plays greedily: forcing a non-preferred LEAD just makes it switch to
its wanted breaker (a free turn for the opponent). This tests the other side of
that: FORCE P1's move on chosen turns (e.g. make the lead set up / set rocks
instead of switching), then let MCTS play freely, and compare win% vs free play.
Answers "is the engine's greedy switch actually better than committing to the
intended setup line, or does it just not see the plan?"

Run with -m (forkserver needs the module importable):
  .venv/bin/python -m monotype.forced_play
"""
from __future__ import annotations
import random, re, time
from pathlib import Path
import multiprocessing as mp

import poke_engine as pe
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


def play_forced(team1, team2, search_ms, forced_p1, max_turns=120):
    """Play one game; force P1's move on turns in forced_p1 {turn: move_choice}.
    Returns 1 if P1 wins, 2 if P2, 0 draw/error."""
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
            r1 = pe.monte_carlo_tree_search(pe.State.from_string(s), duration_ms=search_ms)
            r2 = pe.monte_carlo_tree_search(pe.State.from_string(s), duration_ms=search_ms)
        except Exception:
            return 0
        p1 = _best_non_tera(r1.side_one)
        p2 = _best_non_tera(r2.side_two)
        if turn in forced_p1:
            fm = forced_p1[turn]
            if fm in {x.move_choice for x in r1.side_one}:
                p1 = fm   # forced move is legal this turn; commit to it
        p1 = _normalize_no_move(p1)
        p2 = _normalize_no_move(p2)
        if p1 == "No Move" and p2 == "No Move":
            return 0
        try:
            ins = pe.generate_instructions(state, _strip_switch_prefix(p1), _strip_switch_prefix(p2))
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


# (display, team name, setup-lead species, forced move) — verified legal moves
SCRIPTS = [
    ("Steel  ", "Heavy Metal",                "Heatran",         "stealthrock"),
    ("Dark   ", "Dork Team",                  "Ting-Lu",         "stealthrock"),
    ("Fairy  ", "I Believe In Fairies",       "Tinkaton",        "stealthrock"),
    ("Ground ", "Grounded Grounded Grounded", "Hippowdon",       "stealthrock"),
    ("Flying ", "Fly Away Now",               "Gliscor",         "stealthrock"),
    ("Poison ", "Toxic Team",                 "Glimmora",        "stealthrock"),
    ("Ice    ", "Ice Ice Baby",               "Ninetales-Alola", "auroraveil"),
    ("BugWeb ", "Stop Bugging Me",            "Araquanid",       "stickyweb"),
    ("BugRock", "Stop Bugging Me",            "Kleavor",         "stoneaxe"),
    ("Rock   ", "Rock Hard",                  "Glimmora",        "spikes"),
    ("Normal ", "Totally Normal Team",        "Porygon2",        "trickroom"),
]

_VAR = None
_MS = 500
def _init(V, ms):
    global _VAR, _MS
    _VAR, _MS = V, ms


def _game(task):
    vi, oi, seed = task
    random.seed(seed)
    _, vb, forced, field = _VAR[vi]
    r = play_forced(vb, field[oi], _MS, forced)   # test team always P1
    return (vi, 1 if r == 1 else (0 if r == 2 else -1))


def main():
    teams = parse("monotype/teams/teams_v7.txt")
    tdict = dict(teams)
    VAR = []  # (label, lead_body, forced_dict, field_bodies)
    for disp, tname, lead, move in SCRIPTS:
        body = lead_with(tdict[tname], lead)
        field = [b for n, b in teams if n != tname]
        VAR.append((f"{disp} {lead[:11]:11} free ", body, {}, field))
        VAR.append((f"{disp} {lead[:11]:11} FORCE", body, {0: move}, field))
    GAMES = 6
    nfield = len(VAR[0][3])
    tasks = [(vi, oi, 42 + g) for vi in range(len(VAR)) for oi in range(nfield) for g in range(GAMES)]
    print(f"=== forced-setup batch: {len(SCRIPTS)} teams x (free/FORCE) x {nfield} field x "
          f"{GAMES} = {len(tasks)} games ===", flush=True)
    rec = {vi: [0, 0, 0] for vi in range(len(VAR))}
    t0 = time.time(); done = 0
    with mp.Pool(22, initializer=_init, initargs=(VAR, 500)) as pool:
        for vi, o in pool.imap_unordered(_game, tasks, chunksize=2):
            rec[vi][0 if o == 1 else 1 if o == 0 else 2] += 1
            done += 1
            if done % max(1, len(tasks) // 20) == 0:
                print(f"  [{done}/{len(tasks)}] {time.time()-t0:.0f}s", flush=True)
    print(f"\nDone in {time.time()-t0:.0f}s\n")
    print(f"  {'team / lead':30} {'free':>6} {'FORCE':>7}   Δpp")
    for i in range(0, len(VAR), 2):
        fw, fl, _ = rec[i]
        gw, gl, _ = rec[i + 1]
        fr = 100 * fw / (fw + fl) if (fw + fl) else 0
        gr = 100 * gw / (gw + gl) if (gw + gl) else 0
        lab = VAR[i][0].replace(" free ", "")
        flag = "  <-- forcing helps" if gr - fr >= 4 else ("  >> forcing hurts" if gr - fr <= -4 else "")
        print(f"  {lab:30} {fr:5.1f}% {gr:6.1f}%   {gr-fr:+5.1f}{flag}")


if __name__ == "__main__":
    main()
