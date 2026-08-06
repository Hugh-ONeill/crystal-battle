#!/usr/bin/env python3
"""Preregistered read for the Scarf-Imposter-Ditto draft screen
(showdown/bench/dtscreen0806_REGISTER.txt).

Mechanism first, winrate second — the register's order, because winrate at
this n cannot decide anything against the ~15pp noise floor while the
mechanism questions have crisp answers:

  (a) deploy rate      transforms per game (does the engine use the mon at all?)
  (b) boosted-copy     share of transforms where the target held >= +1 stages
                       (the revenge-killer case the whole plan rests on)
  (c) conversion       opponent's active faints within one turn of a boosted
                       copy, before our Ditto leaves the field
  (d) blindness        fp selects a setup move while our healthy Ditto sits on
                       the bench — the signature of a search that structurally
                       cannot price the copy (upstream poke-engine has no
                       Imposter implementation; verified 2026-08-06)
  (e) health           tracebacks / quarantines / stuck games

Sides: we are p2 (CBGen9*), foul-play is p1. Boost state is tracked from the
protocol per side and reset on switch/drag/faint, so (b) reflects what the
target actually held at copy time.

Usage:
  .venv/bin/python showdown/ditto_screen_read.py 'showdown/bench/dtscreen0806b_L*_ours.log'
"""

import glob
import re
import sys
from collections import defaultdict

SETUP = {"swordsdance", "dragondance", "nastyplot", "calmmind", "quiverdance",
         "bulkup", "irondefense", "shellsmash", "agility", "curse",
         "victorydance", "tidyup", "workup", "growth", "honeclaws",
         "coil", "bellydrum", "clangoroussoul", "noretreat", "tailglow"}

TEAMS = {
    "dt1_ho": {"glimmora", "ironmoth", "ceruledge", "dragapult", "kingambit"},
    "dt2_balance": {"tinglu", "dragapult", "cinderace", "corviknight",
                    "ragingbolt"},
    "dt3_bulky": {"samurotthisui", "landorustherian", "pecharunt", "scizor",
                  "zamazenta"},
}


def norm(s):
    return re.sub(r"[^a-z0-9]", "", (s or "").lower())


def wilson(k, n, z=1.96):
    if not n:
        return (0.0, 0.0, 0.0)
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * ((p * (1 - p) / n + z * z / (4 * n * n)) ** 0.5) / d
    return (p, max(0.0, c - h), min(1.0, c + h))


def read(paths):
    games = {}
    for path in sorted(paths):
        room = None
        st = None
        for line in open(path, errors="replace"):
            m = re.search(r">(battle-gen9ou-\d+)", line)
            if m:
                room = m.group(1)
                st = games.setdefault(room, {
                    "turn": 0, "mons": set(), "winner": None, "us": None,
                    "tf": 0, "tf_boosted": 0, "converted": 0,
                    "setup_while_benched": 0, "ditto_out": False,
                    "ditto_alive": True, "boosts": {"p1": 0, "p2": 0},
                    "pending": None, "opp_setups": 0, "expired": 0,
                    "ditto_died_first": 0, "setup_avail": 0, "moves_avail": 0,
                    "setup_unavail": 0, "moves_unavail": 0,
                })
                continue
            if room is None or not line.startswith("|"):
                continue
            p = line.rstrip("\n").split("|")
            if len(p) < 2:
                continue
            kind = p[1]

            if kind == "turn":
                # Imposter fires on SWITCH-IN, so Ditto does not act until the
                # NEXT turn — closing the window at the first turn marker (the
                # obvious-looking choice) measures zero by construction. The
                # window runs until Ditto leaves the field or 3 turns elapse.
                if st["pending"] is not None:
                    st["pending"] -= 1
                    if st["pending"] <= 0:
                        st["pending"] = None
                        st["expired"] += 1
                st["turn"] = max(st["turn"], int(p[2]) if p[2].isdigit() else 0)
            elif kind in ("switch", "drag"):
                side = p[2][:2]
                st["boosts"][side] = 0
                species = norm(p[3].split(",")[0]) if len(p) > 3 else ""
                if side == "p2":
                    st["mons"].add(species)
                    st["ditto_out"] = (species == "ditto")
                if st["pending"] and side == "p2":
                    st["pending"] = None      # our Ditto left; window closed
            elif kind in ("-boost", "-unboost"):
                if len(p) > 4 and p[4].lstrip("-").isdigit():
                    amt = int(p[4]) * (1 if kind == "-boost" else -1)
                    st["boosts"][p[2][:2]] += amt
            elif kind in ("-clearboost", "-clearallboost", "-setboost"):
                st["boosts"]["p1"] = st["boosts"]["p2"] = 0
            elif kind == "-transform":
                if p[2][:2] != "p2":
                    continue
                st["tf"] += 1
                if st["boosts"]["p1"] >= 1:
                    st["tf_boosted"] += 1
                    st["pending"] = 3
            elif kind == "move":
                if p[2][:2] == "p1":
                    setup = norm(p[3]) in SETUP
                    # BASE RATE is required to read (d): Ditto is benched most
                    # of the game anyway, so a raw count of setups-while-
                    # benched says nothing. The blindness claim is that fp's
                    # setup rate is UNCHANGED by whether a live Imposter is
                    # sitting in the back.
                    avail = st["ditto_alive"] and not st["ditto_out"]
                    if avail:
                        st["moves_avail"] += 1
                        st["setup_avail"] += setup
                    else:
                        st["moves_unavail"] += 1
                        st["setup_unavail"] += setup
                    if setup:
                        st["opp_setups"] += 1
                        if avail:
                            st["setup_while_benched"] += 1
            elif kind == "faint":
                side = p[2][:2]
                st["boosts"][side] = 0
                if side == "p1" and st["pending"]:
                    st["converted"] += 1
                    st["pending"] = None
                if side == "p2" and st["ditto_out"]:
                    st["ditto_alive"] = False
                    if st["pending"]:
                        st["ditto_died_first"] += 1
                    st["pending"] = None
            elif kind == "win":
                st["winner"] = p[2].strip()
            elif kind == "tie":
                st["winner"] = "tie"

    for g in games.values():
        best, score = "?", 0
        for name, roster in TEAMS.items():
            ov = len(g["mons"] & roster)
            if ov > score:
                best, score = name, ov
        g["team"] = best
    return {k: v for k, v in games.items() if v["winner"]}


def main():
    pattern = sys.argv[1] if len(sys.argv) > 1 else \
        "showdown/bench/dtscreen0806b_L*_ours.log"
    games = read(glob.glob(pattern))
    n = len(games)
    tf = sum(g["tf"] for g in games.values())
    tfb = sum(g["tf_boosted"] for g in games.values())
    conv = sum(g["converted"] for g in games.values())
    setups = sum(g["opp_setups"] for g in games.values())
    swb = sum(g["setup_while_benched"] for g in games.values())
    print(f"decided games: {n}\n")
    print("PRIMARY (mechanism)")
    print(f"  (a) deploy     {tf} transforms = {tf/n:.2f}/game; "
          f"{sum(1 for g in games.values() if g['tf']):d} games "
          f"({sum(1 for g in games.values() if g['tf'])/n:.0%}) used Ditto")
    p, lo, hi = wilson(tfb, tf)
    print(f"  (b) boosted    {tfb}/{tf} copies onto a boosted target = "
          f"{p:.1%} [{lo:.1%}, {hi:.1%}]")
    p, lo, hi = wilson(conv, tfb)
    print(f"  (c) conversion {conv}/{tfb} boosted copies -> opp faint in the "
          f"window = {p:.1%} [{lo:.1%}, {hi:.1%}]")
    died = sum(g["ditto_died_first"] for g in games.values())
    exp = sum(g["expired"] for g in games.values())
    print(f"      of the {tfb} boosted copies: {conv} converted, "
          f"{died} our Ditto died first, {exp} window expired")
    sa = sum(g["setup_avail"] for g in games.values())
    ma = sum(g["moves_avail"] for g in games.values())
    su = sum(g["setup_unavail"] for g in games.values())
    mu = sum(g["moves_unavail"] for g in games.values())
    pa, la, ha = wilson(sa, ma)
    pu, lu, hu = wilson(su, mu)
    print(f"  (d) blindness  fp setup rate with a live Ditto BENCHED: "
          f"{sa}/{ma} = {pa:.2%} [{la:.2%}, {ha:.2%}]")
    print(f"                 ...with Ditto out or dead:               "
          f"{su}/{mu} = {pu:.2%} [{lu:.2%}, {hu:.2%}]")
    print(f"                 (equal rates = fp does not price the copy; "
          f"raw count was {swb}/{setups} = {swb/max(setups,1):.1%})")
    print("\nSECONDARY (screening only — cannot decide at this n)")
    for t in sorted(TEAMS):
        sub = [g for g in games.values() if g["team"] == t]
        w = sum(1 for g in sub if (g["winner"] or "").startswith("CBGen9"))
        p, lo, hi = wilson(w, len(sub))
        d = [g for g in sub if g["tf"]]
        dw = sum(1 for g in d if (g["winner"] or "").startswith("CBGen9"))
        print(f"  {t:12s} {w:3d}/{len(sub):3d} = {p:5.1%} [{lo:.1%}, {hi:.1%}]"
              f"   | games where Ditto deployed: {dw}/{len(d)} = "
              f"{dw/max(len(d),1):.1%}")
    w = sum(1 for g in games.values()
            if (g["winner"] or "").startswith("CBGen9"))
    p, lo, hi = wilson(w, n)
    print(f"  {'POOLED':12s} {w:3d}/{n:3d} = {p:5.1%} [{lo:.1%}, {hi:.1%}]")


if __name__ == "__main__":
    main()
