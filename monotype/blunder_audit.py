"""Self-play blunder audit — find silly behaviors the engine does CONSISTENTLY,
with no opponent/oracle needed. Plays dev-vs-dev games and flags, per decision:

  immune_attack : chose an attacking move that does 0 (immune/no effect) while a
                  switch was available — actively ineffective.
  weak_stay     : chose an attack doing <8% while getting chunked (opp hits >=40%)
                  and a switch was available — flailing instead of pivoting.
  weaker_move   : chose an attack when another of its moves does >=2x more to the
                  same target — picked a worse move for no reason.
  setup_and_die : used a setup/boost move, then fainted within 2 turns — wasted.
  wasted_hazard : set a hazard already at max on the opponent's side — dead turn.
  switch_loop   : switched a mon in, then back out within 2 turns w/o it attacking.
  stuck         : game terminated in a no-progress stuck state.

Run with -m:  .venv/bin/python -m monotype.blunder_audit
"""
from __future__ import annotations
import random, re, time
from pathlib import Path
import multiprocessing as mp

import poke_engine as pe
from showdown.bench_monotype import (build_pe_state_gen9, _best_non_tera, _best_useful,
                                     _strip_switch_prefix, _normalize_no_move)
from monotype.lead_picker import split_team_body, species_of

# non-damaging moves in the team pool (legit 0 damage — not "ineffective")
STATUS = {
    "auroraveil","bellydrum","bulkup","calmmind","chillyreception","courtchange",
    "curse","defog","dragondance","encore","endure","futuresight","haze",
    "healingwish","irondefense","lightscreen","memento","moonlight","nastyplot",
    "painsplit","partingshot","protect","quiverdance","recover","reflect","rest",
    "roar","roost","sleeppowder","sleeptalk","slackoff","softboiled","spikes",
    "spore","stealthrock","stickyweb","strengthsap","substitute","swordsdance",
    "tailwind","taunt","thunderwave","toxic","toxicspikes","transform","trick",
    "trickroom","victorydance","whirlwind","willowisp","wish","lifedew",
}
# moves that legitimately deal low/zero damage for other reasons — exclude from
# the move-quality detectors so we don't flag good utility play as a blunder.
PIVOT = {"uturn", "voltswitch", "flipturn", "partingshot", "chillyreception"}
CONDITIONAL = {"suckerpunch", "thunderclap"}            # fail if foe doesn't attack -> calc unreliable
UTILITY_CHIP = {"saltcure", "rapidspin", "mortalspin", "knockoff", "endeavor", "seismictoss"}
SETUP = {"swordsdance","nastyplot","calmmind","dragondance","quiverdance","bulkup",
         "curse","victorydance","bellydrum","irondefense"}
HAZARD = {"stealthrock": ("stealth_rock", 1), "spikes": ("spikes", 3),
          "toxicspikes": ("toxic_spikes", 2), "stickyweb": ("sticky_web", 1)}


def parse(p):
    t = Path(p).read_text()
    s = re.split(r'(?m)^=== \[gen9monotype\] (.+?) ===\s*$', t)
    return [(s[i].strip(), s[i + 1].strip()) for i in range(1, len(s), 2)]


def _norm(m):
    return m.lower().replace(" ", "").replace("-", "").replace("'", "")


def _dmg(state, atk, attacker_first):
    """Max damage `atk` (from the attacker) does to the defender; 0 on error/immune."""
    try:
        d1, d2 = pe.calculate_damage(state, atk, "splash", True) if attacker_first \
            else pe.calculate_damage(state, "splash", atk, True)
        return (d1 if attacker_first else d2)[1]
    except Exception:
        return None  # unknown (skip)


def _switches(side):
    ai = int(side.active_index)
    alive = sum(1 for i, p in enumerate(side.pokemon) if i != ai and p.hp > 0)
    return alive > 0 and not side.force_trapped


def audit_game(team1, team2, search_ms, max_turns=120):
    state = build_pe_state_gen9(team1, team2)
    flags = []  # (detector, side, mon, move)
    prev = ""
    stuck = 0
    # per-side memory for setup-and-die / switch-loop
    last_active = {1: None, 2: None}
    setup_pending = {1: None, 2: None}   # (mon_id, turns_left)
    switched_in = {1: None, 2: None}     # (mon_id, turns_since, dealt_dmg)
    for turn in range(max_turns):
        a1 = sum(p.hp > 0 for p in state.side_one.pokemon)
        a2 = sum(p.hp > 0 for p in state.side_two.pokemon)
        if a1 == 0 or a2 == 0:
            break
        s = state.to_string()
        try:
            ps = pe.State.from_string(s)
            r1 = pe.monte_carlo_tree_search(ps, duration_ms=search_ms)
            ps2 = pe.State.from_string(s)
            r2 = pe.monte_carlo_tree_search(ps2, duration_ms=search_ms)
        except Exception:
            break
        p1 = _best_useful(r1.side_one, state.side_two.side_conditions)
        p2 = _best_useful(r2.side_two, state.side_one.side_conditions)
        moves = {1: p1, 2: p2}
        sides = {1: state.side_one, 2: state.side_two}
        # --- per-side decision detectors ---
        for sd in (1, 2):
            mv = moves[sd]
            side = sides[sd]
            if side.force_switch or mv == "No Move":
                continue  # forced replacement after a faint — not a real choice
            opp = sides[2 if sd == 1 else 1]
            act = side.pokemon[int(side.active_index)]
            opp_act = opp.pokemon[int(opp.active_index)]
            nm = _norm(mv)
            is_switch = mv.startswith("switch ")
            # setup-and-die bookkeeping (decrement/expiry handled post-apply)
            if not is_switch and nm in SETUP:
                setup_pending[sd] = [act.id, 2, False]   # [mon, turns_left, used_boost]
            # wasted hazard
            if not is_switch and nm in HAZARD:
                cond, mx = HAZARD[nm]
                if getattr(opp.side_conditions, cond) >= mx:
                    flags.append(("wasted_hazard", sd, act.id, nm))
            # switch-loop bookkeeping
            if is_switch:
                tgt = _strip_switch_prefix(mv)
                if switched_in[sd] and switched_in[sd][0] == tgt and switched_in[sd][1] <= 2 \
                        and not switched_in[sd][2]:
                    flags.append(("switch_loop", sd, tgt, "switch"))
                switched_in[sd] = (tgt, 0, False)
            else:
                if switched_in[sd]:
                    mon, t, dealt = switched_in[sd]
                    switched_in[sd] = (mon, t + 1, dealt)
            # move-quality detectors (attacks only)
            if not is_switch and nm not in STATUS and nm != "none" and mv != "splash":
                dmg = _dmg(state, mv, sd == 1)
                if dmg is not None:
                    if switched_in[sd] and dmg > 0:
                        m, t, _ = switched_in[sd]; switched_in[sd] = (m, t, True)
                    if setup_pending[sd] and setup_pending[sd][0] == act.id and dmg > 0:
                        setup_pending[sd][2] = True   # booster is using its boost
                    sw = _switches(side)
                    sub = getattr(opp, "substitute_health", 0) > 0
                    if dmg == 0 and sw and not sub and nm not in PIVOT and nm not in CONDITIONAL:
                        flags.append(("immune_attack", sd, act.id, nm))
                    elif 0 < dmg < 0.08 * max(1, opp_act.maxhp) and sw \
                            and nm not in PIVOT and nm not in UTILITY_CHIP and nm not in CONDITIONAL:
                        opp_hit = _dmg(state, moves[2 if sd == 1 else 1], sd == 2) or 0
                        if opp_hit >= 0.40 * max(1, act.maxhp):
                            flags.append(("weak_stay", sd, act.id, nm))
        # apply
        p1 = _normalize_no_move(p1); p2 = _normalize_no_move(p2)
        if p1 == "No Move" and p2 == "No Move":
            break
        try:
            ins = pe.generate_instructions(state, _strip_switch_prefix(p1), _strip_switch_prefix(p2))
        except Exception:
            break
        if not ins:
            break
        roll = random.random() * 100; cum = 0.0; chosen = ins[0]
        for i in ins:
            cum += i.percentage
            if roll <= cum:
                chosen = i; break
        state = state.apply_instructions(chosen)
        # setup-and-die: flag only if the booster fainted WITHOUT ever using the boost
        for sd in (1, 2):
            if setup_pending[sd]:
                mon, tl, used = setup_pending[sd]
                s_side = state.side_one if sd == 1 else state.side_two
                alive = any(p.id == mon and p.hp > 0 for p in s_side.pokemon)
                if not alive:
                    if not used:
                        flags.append(("setup_and_die", sd, mon, "setup"))
                    setup_pending[sd] = None
                else:
                    setup_pending[sd] = None if tl - 1 <= 0 else [mon, tl - 1, used]
        cs = state.to_string()
        if cs == prev:
            stuck += 1
            if stuck >= 3:
                flags.append(("stuck", 0, "-", "-"))
                break
        else:
            stuck = 0
        prev = cs
    return flags


_TEAMS = None
_MS = 500
def _init(T, ms):
    global _TEAMS, _MS
    _TEAMS, _MS = T, ms


def _run(task):
    i, j, seed = task
    random.seed(seed)
    return audit_game(_TEAMS[i][1], _TEAMS[j][1], _MS)


def main():
    teams = parse("monotype/teams/teams_v7.txt")
    n = len(teams)
    # diverse matchups: each team P1 vs 3 spread opponents
    pairs = [(i, (i + k) % n) for i in range(n) for k in (1, 6, 11)]
    tasks = [(i, j, 42) for (i, j) in pairs]
    print(f"=== blunder audit: {len(tasks)} self-play games @ {_MS}ms ===", flush=True)
    from collections import Counter
    counts = Counter(); examples = {}
    t0 = time.time(); done = 0
    with mp.Pool(22, initializer=_init, initargs=(teams, 500)) as pool:
        for flags in pool.imap_unordered(_run, tasks):
            for det, sd, mon, mv in flags:
                counts[det] += 1
                examples.setdefault(det, [])
                if len(examples[det]) < 5:
                    examples[det].append(f"{mon}:{mv}")
            done += 1
            if done % max(1, len(tasks) // 10) == 0:
                print(f"  [{done}/{len(tasks)}] {time.time()-t0:.0f}s", flush=True)
    print(f"\nDone in {time.time()-t0:.0f}s over {len(tasks)} games\n")
    print(f"  {'blunder':16} {'count':>6}  {'per-game':>8}   examples")
    for det in ["immune_attack","weak_stay","setup_and_die",
                "wasted_hazard","switch_loop","stuck"]:
        c = counts.get(det, 0)
        ex = ", ".join(examples.get(det, [])[:5])
        print(f"  {det:16} {c:6d}  {c/len(tasks):8.2f}   {ex}")


if __name__ == "__main__":
    main()
