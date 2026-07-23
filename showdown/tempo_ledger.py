"""Turn-level tempo ledger from bench logs (CB vs foul-play).

WHY (2026-07-23): the behavioral autopsy showed fp buys pressure and CB buys
housekeeping, and loss traces say we lose short games to midgame snowballs —
but "we lose early" was inferred from first-blood stats and game length.
This tool makes it a curve: cumulative team-HP differential at every turn
(unseen mons count as full), so we can see WHERE the gap opens, then autopsy
what the eventual loser clicked in the turns immediately before its deficit
crossed the snowball threshold. All metrics are within-game side-vs-side,
the only comparison basis that has survived the game-mix-variance lessons.

Usage:
  python showdown/tempo_ledger.py "showdown/bench/retunegate_L*_ours.log"

HP tracking is per-mon keyed by protocol ident nickname; p1 percent HP and
p2 absolute HP both normalize to fractions. Known approximation: Zoroark's
Illusion can briefly double-count a disguised mon until |replace| lands.
"""
import argparse
import glob
import re
from collections import Counter, defaultdict

try:
    from showdown.behavior_compare import classify, mid
except ImportError:
    from behavior_compare import classify, mid

HP_RE = re.compile(r"^(\d+)/(\d+)")


def _frac(tok):
    """HP token -> fraction, or None if unparseable. '0 fnt' -> 0.0."""
    tok = tok.strip()
    if tok.startswith("0 fnt") or tok == "0":
        return 0.0
    m = HP_RE.match(tok)
    return int(m.group(1)) / int(m.group(2)) if m else None


def _ident(raw):
    """'p1a: Ting-Lu' -> ('p1', 'Ting-Lu'); None for non-slot idents."""
    if len(raw) < 5 or raw[0] != "p" or raw[3] != ":":
        return None
    return raw[:2], raw.split(": ", 1)[1]


def games_from_log(path, team_size=6):
    """Yield per-game dicts: winner, turns, diffs (turn -> cb-fp team HP at
    START of that turn, in mons), end_diff, actions (turn -> [(side, kind,
    val)]; switch = hard switches only), hard_sw / dosi / faints Counters."""
    role, g = {}, None
    hp = {}
    moved, fainted, entered = set(), set(), {}

    def total(side):
        mine = [v for (s, _), v in hp.items() if s == side]
        return sum(mine) + (team_size - len(mine)) * 1.0

    def diff():
        cb = next((p for p, r in role.items() if r == "cb"), "p2")
        fp = "p1" if cb == "p2" else "p2"
        return total(cb) - total(fp)

    for line in open(path, errors="replace"):
        i = line.find("|")
        if i < 0:
            continue
        parts = line.rstrip("\n").split("|")[1:]
        if not parts:
            continue
        tag = parts[0]
        if tag == "init" and len(parts) > 1 and parts[1] == "battle":
            if g and g.get("winner"):
                yield g
            g = dict(winner=None, turn=0, turns=0, diffs={}, end_diff=0.0,
                     actions=defaultdict(list), hard_sw=Counter(),
                     dosi=Counter(), faints=Counter())
            role, hp = {}, {}
            moved, fainted, entered = set(), set(), {}
        if g is None:
            continue
        if tag == "player" and len(parts) >= 3 and parts[2]:
            role[parts[1]] = "cb" if parts[2].startswith("CBGen9") else "fp"
        elif tag == "turn":
            g["turn"] = g["turns"] = int(parts[1])
            g["diffs"][g["turn"]] = diff()
            moved, fainted, entered = set(), set(), {}
        elif tag == "move" and len(parts) >= 3:
            who = _ident(parts[1])
            if who and who[0] in role:
                side = role[who[0]]
                moved.add(side)
                g["actions"][g["turn"]].append((side, "move", mid(parts[2])))
        elif tag in ("switch", "drag", "replace") and len(parts) >= 4:
            who = _ident(parts[1])
            if not who or who[0] not in role:
                continue
            f = _frac(parts[3])
            if f is not None:
                hp[who] = f
            side = role[who[0]]
            # a hard switch is a chosen action: not a pivot follow-up (side
            # already moved this turn), not a post-faint replacement, not a
            # phaze drag, not the turn-0 lead entry
            if (tag == "switch" and g["turn"] >= 1
                    and side not in moved and side not in fainted):
                g["hard_sw"][side] += 1
                g["actions"][g["turn"]].append((side, "switch", who[1]))
                entered[who] = g["turn"]
        elif tag in ("-damage", "-heal") and len(parts) >= 3:
            who = _ident(parts[1])
            if who and who[0] in role:
                f = _frac(parts[2])
                if f is not None:
                    hp[who] = f
        elif tag == "-sethp":
            rest = parts[1:]
            for j in range(0, len(rest) - 1, 2):
                who = _ident(rest[j])
                f = _frac(rest[j + 1]) if who else None
                if who and f is not None:
                    hp[who] = f
        elif tag == "faint" and len(parts) >= 2:
            who = _ident(parts[1])
            if who and who[0] in role:
                hp[who] = 0.0
                side = role[who[0]]
                fainted.add(side)
                g["faints"][side] += 1
                if entered.get(who) == g["turn"]:
                    g["dosi"][side] += 1
        elif tag == "win" and len(parts) >= 2:
            g["winner"] = "cb" if parts[1].startswith("CBGen9") else "fp"
            g["end_diff"] = diff()
    if g and g.get("winner"):
        yield g


def onset_turn(game, side, thresh):
    """First turn whose START-of-turn snapshot has `side` down >= thresh
    mons of team HP, or None. The crossing happened DURING the prior turn."""
    sign = 1.0 if side == "cb" else -1.0
    for t in sorted(game["diffs"]):
        if sign * game["diffs"][t] <= -thresh:
            return t
    return None


def _mix(counter):
    tot = max(1, sum(counter.values()))
    return {k: 100.0 * v / tot for k, v in counter.items()}, tot


def report(games, thresh=1.0, window=2):
    n = len(games)
    cbw = sum(1 for g in games if g["winner"] == "cb")
    print(f"{n} decided games, CB {cbw}W-{n - cbw}L ({100 * cbw / n:.1f}%)")
    sane = sum(1 for g in games
               if (g["end_diff"] > 0) == (g["winner"] == "cb"))
    print(f"ledger sanity: winner ahead on HP at game end in {sane}/{n}")
    wl = [g["turns"] for g in games if g["winner"] == "cb"]
    ll = [g["turns"] for g in games if g["winner"] == "fp"]
    print(f"avg length: CB wins {sum(wl) / max(1, len(wl)):.1f} turns, "
          f"CB losses {sum(ll) / max(1, len(ll)):.1f}")

    print("\n=== winrate by game length ===")
    for lo, hi in ((1, 15), (16, 25), (26, 40), (41, 999)):
        b = [g for g in games if lo <= g["turns"] <= hi]
        w = sum(1 for g in b if g["winner"] == "cb")
        tag = f"{lo}-{hi if hi < 999 else '+'}"
        pct = f"{100 * w / len(b):5.1f}%" if b else "    -"
        print(f"  {tag:>6s}: {w:3d}W-{len(b) - w:3d}L  {pct}")

    print("\n=== HP differential (CB minus FP, in mons; mean over games "
          "still running at that turn) ===")
    marks = (3, 5, 8, 12, 16, 20, 30)
    print(f"  {'':8s}" + "".join(f"  t{t:<5d}" for t in marks) + "  end")
    for label, side in (("wins", "cb"), ("losses", "fp")):
        row = f"  {label:8s}"
        for t in marks:
            vals = [g["diffs"][t] for g in games
                    if g["winner"] == side and t in g["diffs"]]
            row += (f"  {sum(vals) / len(vals):+5.2f} " if vals
                    else "     -  ")
        ends = [g["end_diff"] for g in games if g["winner"] == side]
        row += f" {sum(ends) / len(ends):+5.2f}"
        print(row + f"   (n@t8={sum(1 for g in games if g['winner'] == side and 8 in g['diffs'])})")

    print(f"\n=== deficit onset (first start-of-turn down >= {thresh:.1f} "
          "mons) ===")
    first_down_loses = comeback = Counter()
    for g in games:
        on = {s: onset_turn(g, s, thresh) for s in ("cb", "fp")}
        firsts = [s for s in ("cb", "fp") if on[s] is not None]
        if firsts:
            first = min(firsts, key=lambda s: on[s])
            first_down_loses[g["winner"] != first] += 1
        for s in ("cb", "fp"):
            if on[s] is not None and g["winner"] == s:
                comeback[s] += 1
    fd = sum(first_down_loses.values())
    if fd:
        print(f"  first side down a mon loses: "
              f"{100 * first_down_loses[True] / fd:.0f}% ({fd} games)")
    print(f"  comebacks (crossed but won): CB {comeback['cb']}, "
          f"FP {comeback['fp']}")
    for label, side in (("CB losses", "cb"), ("FP losses", "fp")):
        loser_games = [g for g in games if g["winner"] != side]
        ons = [onset_turn(g, side, thresh) for g in loser_games]
        ons = [o for o in ons if o is not None]
        if not ons:
            continue
        ons.sort()
        by10 = sum(1 for o in ons if o <= 10)
        print(f"  {label}: onset median t{ons[len(ons) // 2]}, "
              f"mean t{sum(ons) / len(ons):.1f}, "
              f"by t10 in {100 * by10 / len(ons):.0f}% "
              f"({len(ons)}/{len(loser_games)} crossed)")

    print(f"\n=== pre-onset autopsy: loser's actions in the {window} turns "
          "before crossing, vs that side's all-turns baseline ===")
    pre, base = defaultdict(Counter), defaultdict(Counter)
    for g in games:
        for t, evs in g["actions"].items():
            for side, kind, val in evs:
                cat = "hard_switch" if kind == "switch" else classify(val)
                base[side][cat] += 1
        loser = "cb" if g["winner"] == "fp" else "fp"
        on = onset_turn(g, loser, thresh)
        if on is None:
            continue
        for t in range(max(1, on - window), on):
            for side, kind, val in g["actions"].get(t, ()):
                if side != loser:
                    continue
                cat = "hard_switch" if kind == "switch" else classify(val)
                pre[loser][cat] += 1
    cats = sorted(set().union(*pre.values(), *base.values()),
                  key=lambda c: -sum(b[c] for b in base.values()))
    header = "  ".join(f"{s}-pre  {s}-base" for s in ("cb", "fp"))
    print(f"  {'per 100':14s}  {header}")
    mixes = {(s, k): _mix(d[s]) for s in ("cb", "fp")
             for k, d in (("pre", pre), ("base", base))}
    for c in cats:
        row = f"  {c:14s}"
        for s in ("cb", "fp"):
            row += (f"  {mixes[(s, 'pre')][0].get(c, 0):5.1f}"
                    f"  {mixes[(s, 'base')][0].get(c, 0):6.1f}")
        print(row)
    print(f"  (pre-window n: cb={mixes[('cb', 'pre')][1]}, "
          f"fp={mixes[('fp', 'pre')][1]})")

    print("\n=== died on hard switch-in (same turn) ===")
    for s in ("cb", "fp"):
        d, h = sum(g["dosi"][s] for g in games), sum(g["hard_sw"][s] for g in games)
        print(f"  {s}: {d} deaths / {h} hard switches "
              f"({100 * d / max(1, h):.1f}%), {d / n:.2f} per game")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("patterns", nargs="+")
    ap.add_argument("--onset-mons", type=float, default=1.0)
    ap.add_argument("--window", type=int, default=2)
    args = ap.parse_args()
    files = sorted(f for p in args.patterns for f in glob.glob(p))
    games = [g for f in files for g in games_from_log(f)]
    if not games:
        print("no decided games found")
        return
    print(f"parsed {len(files)} logs")
    report(games, thresh=args.onset_mons, window=args.window)


if __name__ == "__main__":
    main()
