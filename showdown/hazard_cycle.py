"""Hazard-cycle ledger: attribute WHY one side bleeds more entry-hazard HP.

WHY (2026-07-23): the breadth test's Blissey-stall mirror (brd_stallB, 3W-27L)
showed CB taking ~2x fp's hazard damage on a Regenerator/cycle archetype.
Hazard bleed factors as (entries) x (dirty when entering) x (cost per entry,
which depends on WHICH mon cycles). Each factor points at a different fix
(switch-tax vs removal timing vs cycler choice), so measure before tuning.

Per side, per game set:
  entries        total switch-ins (and the voluntary "hard" subset)
  dirty%         share of turns with SR or Spikes active against the side
  hazard dmg     total entry-hazard HP lost (in mons), split per mon
  SR-unpaid%     entries under SR that took no SR damage (Boots/Magic Guard)
  removals       removal clicks; what they cleared (own/theirs/nothing)
  restick<=5     cleared own-side hazards re-set within 5 turns

Limitations: spikes grounding can't be read from the log (flying mons skip
spikes legitimately), so unpaid% is computed against SR only; Court Change is
not modeled.

Usage:
  hazard_cycle.py "showdown/bench/brd_stallB_L*_ours.log" [more patterns]
"""
import argparse
import glob
from collections import Counter, defaultdict

try:
    from showdown.tempo_ledger import _frac, _ident
except ImportError:
    from tempo_ledger import _frac, _ident

REMOVAL_MOVES = {"Defog", "Rapid Spin", "Tidy Up", "Mortal Spin"}
DMG_HAZARDS = ("Stealth Rock", "Spikes")


def _cond(raw):
    return raw.replace("move: ", "").strip()


def games_from_log(path):
    """Yield per-game dicts with the hazard-cycle event stream digested."""
    roles, g = {}, None
    hp = {}
    haz = {"cb": Counter(), "fp": Counter()}   # side -> condition counts

    def new_game():
        return dict(winner=None, turns=0,
                    entries=Counter(), hard=Counter(),
                    dirty_turns=Counter(), sr_turns=Counter(),
                    hazdmg=Counter(), hazdmg_mon=defaultdict(float),
                    sr_entries=Counter(), sr_paid=Counter(),
                    removals=Counter(),
                    set_at=[], cleared_at=[],
                    _moved=set(), _fainted=set(), _turn=0,
                    _last_entry=None)

    for line in open(path, errors="replace"):
        i = line.find("|")
        if i < 0:
            continue
        parts = line.rstrip("\n").split("|")[1:]
        if not parts:
            continue
        tag = parts[0]
        if tag == "init" and len(parts) > 1 and parts[1] == "battle":
            if g and g["winner"]:
                yield g
            g = new_game()
            roles, hp = {}, {}
            haz = {"cb": Counter(), "fp": Counter()}
        if g is None:
            continue
        if tag == "player" and len(parts) >= 3 and parts[2]:
            roles[parts[1]] = "cb" if parts[2].startswith("CBGen9") else "fp"
        elif tag == "turn":
            g["_turn"] = g["turns"] = int(parts[1])
            g["_moved"], g["_fainted"] = set(), set()
            for side in ("cb", "fp"):
                if haz[side]["Stealth Rock"] or haz[side]["Spikes"]:
                    g["dirty_turns"][side] += 1
                if haz[side]["Stealth Rock"]:
                    g["sr_turns"][side] += 1
        elif tag == "-sidestart" and len(parts) >= 3:
            side = roles.get(parts[1][:2])
            if side:
                c = _cond(parts[2])
                haz[side][c] += 1
                if c in DMG_HAZARDS:
                    g["set_at"].append((side, c, g["_turn"]))
        elif tag == "-sideend" and len(parts) >= 3:
            side = roles.get(parts[1][:2])
            if side:
                c = _cond(parts[2])
                if haz[side][c]:
                    haz[side][c] = 0
                    if c in DMG_HAZARDS:
                        g["cleared_at"].append((side, c, g["_turn"]))
        elif tag == "move" and len(parts) >= 3:
            who = _ident(parts[1])
            if who and who[0] in roles:
                side = roles[who[0]]
                g["_moved"].add(side)
                if parts[2] in REMOVAL_MOVES:
                    g["removals"][side] += 1
        elif tag in ("switch", "drag", "replace") and len(parts) >= 4:
            who = _ident(parts[1])
            if not who or who[0] not in roles:
                continue
            f = _frac(parts[3])
            if f is not None:
                hp[who] = f
            side = roles[who[0]]
            if g["_turn"] >= 1:
                g["entries"][side] += 1
                if (tag == "switch" and side not in g["_moved"]
                        and side not in g["_fainted"]):
                    g["hard"][side] += 1
                if haz[side]["Stealth Rock"]:
                    g["sr_entries"][side] += 1
                    g["_last_entry"] = (who, side)
        elif tag == "-damage" and len(parts) >= 3:
            who = _ident(parts[1])
            if not who or who[0] not in roles:
                continue
            f = _frac(parts[2])
            if f is None:
                continue
            delta = hp.get(who, 1.0) - f
            hp[who] = f
            src = next((h for h in DMG_HAZARDS if f"[from] {h}" in line), None)
            if src and delta > 0:
                side = roles[who[0]]
                g["hazdmg"][side] += delta
                g["hazdmg_mon"][(side, who[1])] += delta
                if src == "Stealth Rock" and g["_last_entry"] == (who, side):
                    g["sr_paid"][side] += 1
        elif tag == "faint" and len(parts) >= 2:
            who = _ident(parts[1])
            if who and who[0] in roles:
                hp[who] = 0.0
                g["_fainted"].add(roles[who[0]])
        elif tag == "win" and len(parts) >= 2:
            g["winner"] = "cb" if parts[1].startswith("CBGen9") else "fp"
    if g and g["winner"]:
        yield g


def restick(games, window=5):
    """(re-set within `window` turns, total clears) of own-side hazards,
    per side."""
    out = {s: [0, 0] for s in ("cb", "fp")}
    for x in games:
        for side, cond, t in x["cleared_at"]:
            out[side][1] += 1
            if any(s == side and c == cond and t < ts <= t + window
                   for s, c, ts in x["set_at"]):
                out[side][0] += 1
    return out


def report(games):
    n = len(games)
    cbw = sum(1 for x in games if x["winner"] == "cb")
    print(f"{n} decided games, CB {cbw}W-{n - cbw}L "
          f"({100 * cbw / max(1, n):.1f}%)")
    tot_turns = sum(x["turns"] for x in games)
    print(f"{tot_turns} total turns")
    hdr = f"  {'':24s}{'CB':>10s}{'FP':>10s}"
    print(hdr)

    def row(label, f, fmt="{:10.2f}"):
        print(f"  {label:24s}"
              + "".join(fmt.format(f(s)) for s in ("cb", "fp")))

    row("entries/game", lambda s: sum(x["entries"][s] for x in games) / n)
    row("  hard switches/game", lambda s: sum(x["hard"][s] for x in games) / n)
    row("dirty-turn %", lambda s: 100 * sum(x["dirty_turns"][s] for x in games)
        / max(1, tot_turns))
    row("  SR-up %", lambda s: 100 * sum(x["sr_turns"][s] for x in games)
        / max(1, tot_turns))
    row("hazard dmg (mons)/game",
        lambda s: sum(x["hazdmg"][s] for x in games) / n)
    row("entries under SR/game",
        lambda s: sum(x["sr_entries"][s] for x in games) / n)
    row("  SR-unpaid % (boots)",
        lambda s: 100 * (1 - sum(x["sr_paid"][s] for x in games)
                         / max(1, sum(x["sr_entries"][s] for x in games))))
    row("removal clicks/game", lambda s: sum(x["removals"][s] for x in games) / n)
    row("own-clears/game",
        lambda s: sum(1 for x in games for sd, c, t in x["cleared_at"]
                      if sd == s) / n)
    rs = restick(games)
    row("  re-set within 5t %",
        lambda s: 100 * rs[s][0] / max(1, rs[s][1]))

    print("\n  top hazard bleeders (mons of HP across set):")
    mons = Counter()
    for x in games:
        for k, v in x["hazdmg_mon"].items():
            mons[k] += v
    for (side, mon), v in mons.most_common(8):
        print(f"    {side} {mon:14s} {v:5.2f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("patterns", nargs="+")
    args = ap.parse_args()
    files = sorted(f for p in args.patterns for f in glob.glob(p))
    games = [x for f in files for x in games_from_log(f)]
    if not games:
        print("no decided games")
        return
    print(f"parsed {len(files)} logs")
    report(games)


if __name__ == "__main__":
    main()
