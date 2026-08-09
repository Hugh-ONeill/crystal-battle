#!/usr/bin/env python3
"""Suspect #3 final call: outcome-join the LADDER maximin/voting shadow.

Mirrors the 2026-08-05 bench-corpus read (DONE.md) so the two verdicts are
directly comparable:
  - decision-level fire rate in eventual wins vs losses (maximin + voting)
  - game-level: winrate of >=1-fire games vs fire-free games
  - per-opponent split (richwoman is the geometry the bench can't produce)
Bench verdict was NO-GO lean: maximin fired MORE in wins (2.37% vs 1.72%;
fire-games 49.3% vs 41.4%) — worst-case would overrule in positions we
already win. Gate: if the ladder agrees, close suspect #3, remove drop-in.
"""

import glob
import json
import math
import re
from collections import Counter, defaultdict

ME = "PAC-Crystal"


def wilson(w, n, z=1.96):
    if not n:
        return (0.0, 0.0, 1.0)
    p = w / n
    d = 1 + z * z / n
    c = p + z * z / (2 * n)
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return (p, (c - h) / d, (c + h) / d)


def parse_outcomes(paths):
    """room -> (won: bool, opponent: str)"""
    sides = defaultdict(dict)
    winner, room = {}, None
    for path in paths:
        for line in open(path, errors="replace"):
            m = re.search(r">(battle-\S+)", line)
            if m:
                room = m.group(1)
                continue
            if room is None:
                continue
            m = re.search(r"\|player\|(p[12])\|([^|]+)\|", line)
            if m:
                sides[room][m.group(1)] = m.group(2)
                continue
            m = re.search(r"\|win\|([^|\n]+)", line)
            if m:
                winner[room] = m.group(1).strip()
    out = {}
    for room, w in winner.items():
        names = set(sides[room].values())
        if ME not in names:
            continue
        opp = next((n for n in names if n != ME), "?")
        out[room] = (w == ME, opp)
    return out


def main():
    outcomes = parse_outcomes(
        sorted(glob.glob("showdown/bench/overnight_*_ladder.log")))

    recs = [json.loads(l) for l in
            open("showdown/bench/maximin_shadow.jsonl")]
    rooms = {r["tag"] for r in recs}
    joined = {t for t in rooms if t in outcomes}
    print(f"shadow records: {len(recs)} across {len(rooms)} games; "
          f"outcome-joined {len(joined)} games "
          f"({len(joined)/len(rooms):.0%})")

    # decision-level and game-level, per channel
    for chan, key in (("MAXIMIN", "fired"), ("VOTING", "vote_fired")):
        dec = Counter()          # (won, fired) -> n
        per_game = defaultdict(int)
        for r in recs:
            o = outcomes.get(r["tag"])
            if not o:
                continue
            dec[(o[0], bool(r[key]))] += 1
            per_game[r["tag"]] += bool(r[key])
        w_rate = dec[(True, True)] / max(dec[(True, True)] +
                                         dec[(True, False)], 1)
        l_rate = dec[(False, True)] / max(dec[(False, True)] +
                                          dec[(False, False)], 1)
        fire_g = [t for t in joined if per_game[t] > 0]
        free_g = [t for t in joined if per_game[t] == 0]
        fw = sum(outcomes[t][0] for t in fire_g)
        nw = sum(outcomes[t][0] for t in free_g)
        p1, lo1, hi1 = wilson(fw, len(fire_g))
        p2, lo2, hi2 = wilson(nw, len(free_g))
        print(f"\n{chan}:")
        print(f"  decision-level fire rate: wins "
              f"{dec[(True, True)]}/{dec[(True, True)]+dec[(True, False)]}"
              f" = {w_rate:.2%}  |  losses "
              f"{dec[(False, True)]}/"
              f"{dec[(False, True)]+dec[(False, False)]} = {l_rate:.2%}")
        print(f"  game-level: fire-games {fw}/{len(fire_g)} = "
              f"{p1:.1%} [{lo1:.1%},{hi1:.1%}]  |  fire-free {nw}/"
              f"{len(free_g)} = {p2:.1%} [{lo2:.1%},{hi2:.1%}]")

    # per-opponent maximin split
    print("\nper-opponent (maximin):")
    by_opp = defaultdict(lambda: Counter())
    for r in recs:
        o = outcomes.get(r["tag"])
        if not o:
            continue
        c = by_opp[o[1]]
        c["dec"] += 1
        c["fire"] += bool(r["fired"])
        c["vfire"] += bool(r["vote_fired"])
    games_opp = defaultdict(lambda: [0, 0])
    for t in joined:
        won, opp = outcomes[t]
        games_opp[opp][0] += 1
        games_opp[opp][1] += won
    for opp, c in sorted(by_opp.items(), key=lambda kv: -kv[1]["dec"]):
        g, gw = games_opp[opp]
        print(f"  {opp:16s} games {gw}W-{g-gw}L | decisions {c['dec']:5d} "
              f"| maximin {c['fire']/c['dec']:.2%} "
              f"| voting {c['vfire']/c['dec']:.2%}")

    # richwoman decision-level win/loss split (the geometry question)
    rw = Counter()
    for r in recs:
        o = outcomes.get(r["tag"])
        if not o or o[1] != "richwoman":
            continue
        rw[(o[0], bool(r["fired"]), bool(r["vote_fired"]))] += 1
    if rw:
        for lab, idx in (("maximin", 1), ("voting", 2)):
            wn = sum(v for k, v in rw.items() if k[0] and k[idx])
            wt = sum(v for k, v in rw.items() if k[0])
            ln = sum(v for k, v in rw.items() if not k[0] and k[idx])
            lt = sum(v for k, v in rw.items() if not k[0])
            print(f"  richwoman {lab}: wins {wn}/{wt} = "
                  f"{wn/max(wt,1):.2%} | losses {ln}/{lt} = "
                  f"{ln/max(lt,1):.2%}")

    # fire-turn median, for the bench comparison (bench median: 13)
    turns = sorted(r["turn"] for r in recs
                   if r["fired"] and r["tag"] in joined)
    if turns:
        print(f"\nmaximin fires: {len(turns)}, median turn "
              f"{turns[len(turns)//2]}")


if __name__ == "__main__":
    main()
