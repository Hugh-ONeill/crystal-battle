# Loss-trace analysis over the foul-play A/B series logs.
#
# Parses our player's logs (poke-env INFO protocol lines + our per-turn
# choice prints) into per-game records, then aggregates blunder signals
# across losses: score collapses (the eval cliff where a game was lost),
# what we chose at/right before the cliff, which of our mons die first and
# to what, tera timing, first-KO correlation, and game-length splits.
#
# Same philosophy as monotype/blunder_audit.py: objective detectors over
# many games beat staring at single replays.
#
# Usage:
#   .venv/bin/python showdown/loss_trace.py showdown/bench/ab*_ours.log
#   .venv/bin/python showdown/loss_trace.py --collapse-examples 3 <logs...>

from __future__ import annotations

import argparse
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

OUR_NAME = "CBGen9"

_ROOM_RE = re.compile(r">(battle-\w+-\d+)")
_CHOICE_RE = re.compile(
    r"^\s+T(\d+): (.+?) \(visits=(\d+),\s*avg_score=([\d.eE+-]+)\)")
_PLAYER_RE = re.compile(r"\|player\|(p[12])\|([^|]+)\|")
_TURN_RE = re.compile(r"\|turn\|(\d+)")
_MOVE_RE = re.compile(r"\|move\|(p[12])a: ([^|]+)\|([^|]+)\|")
_FAINT_RE = re.compile(r"\|faint\|(p[12])a: (.+)$")
_TERA_RE = re.compile(r"\|-terastallize\|(p[12])a: ([^|]+)\|(\w+)")
_WIN_RE = re.compile(r"\|win\|(.+)$")


def parse_games(paths: list[Path], our_name: str = OUR_NAME) -> list[dict]:
    games = []
    for path in paths:
        cur = None
        room = None
        for line in path.read_text(errors="replace").splitlines():
            m = _ROOM_RE.search(line)
            if m and m.group(1) != room:
                room = m.group(1)
                cur = {"room": room, "file": path.name, "players": {},
                       "choices": [], "faints": [], "teras": {}, "turn": 0,
                       "last_move": {}, "winner": None}
                games.append(cur)
            if cur is None:
                continue
            cm = _CHOICE_RE.match(line)
            if cm:
                cur["choices"].append((int(cm.group(1)), cm.group(2),
                                       int(cm.group(3)), float(cm.group(4))))
                continue
            pm = _PLAYER_RE.search(line)
            if pm:
                cur["players"][pm.group(1)] = pm.group(2).strip()
            tm = _TURN_RE.search(line)
            if tm:
                cur["turn"] = int(tm.group(1))
            mm = _MOVE_RE.search(line)
            if mm:
                cur["last_move"][mm.group(1)] = (mm.group(2).strip(),
                                                 mm.group(3).strip())
            fm = _FAINT_RE.search(line)
            if fm:
                side = fm.group(1)
                other = "p2" if side == "p1" else "p1"
                killer = cur["last_move"].get(other, ("?", "?"))
                cur["faints"].append((side, fm.group(2).strip(),
                                      cur["turn"], killer))
            xm = _TERA_RE.search(line)
            if xm and xm.group(1) not in cur["teras"]:
                cur["teras"][xm.group(1)] = (cur["turn"], xm.group(2).strip(),
                                             xm.group(3))
            wm = _WIN_RE.search(line)
            if wm:
                cur["winner"] = wm.group(1).strip()
    # keep only completed games with choices
    out = []
    for g in games:
        if g["winner"] is None or not g["choices"]:
            continue
        g["our_role"] = next((r for r, n in g["players"].items()
                              if n == our_name), None)
        if g["our_role"] is None:
            continue
        g["opp_role"] = "p2" if g["our_role"] == "p1" else "p1"
        g["opp_name"] = g["players"].get(g["opp_role"], "?")
        g["we_won"] = g["winner"] == our_name
        out.append(g)
    return out


def biggest_collapse(g: dict):
    """(turn, drop, action_at, action_before, score_before, score_after) of
    the largest single-step avg_score drop."""
    best = None
    ch = g["choices"]
    for i in range(1, len(ch)):
        drop = ch[i - 1][3] - ch[i][3]
        if best is None or drop > best[1]:
            best = (ch[i][0], drop, ch[i][1], ch[i - 1][1],
                    ch[i - 1][3], ch[i][3])
    return best


def main():
    parser = argparse.ArgumentParser(description="A/B loss-trace analysis")
    parser.add_argument("logs", nargs="+")
    parser.add_argument("--name", default=OUR_NAME,
                        help="our player name in the logs (ladder runs as "
                             "PAC-Crystal, local benches as CBGen9)")
    parser.add_argument("--opponent", default=None,
                        help="restrict analysis to games vs this opponent "
                             "(e.g. LLM-gem3f) — for per-matchup loss-tracing")
    parser.add_argument("--collapse-examples", type=int, default=0,
                        help="print the N worst collapse timelines")
    args = parser.parse_args()

    games = parse_games([Path(p) for p in args.logs], args.name)
    if args.opponent:
        games = [g for g in games if g["opp_name"] == args.opponent]
        print(f"[filtered to opponent {args.opponent}]")
    wins = [g for g in games if g["we_won"]]
    losses = [g for g in games if not g["we_won"]]
    n = len(games)
    wr = len(wins) / n if n else 0.0
    se = (wr * (1 - wr) / n) ** 0.5 if n else 0.0
    print(f"parsed {n} games as {args.name}: {len(wins)}W / {len(losses)}L "
          f"= {wr:.1%} (±{1.96 * se:.1%} 95% CI)\n")

    # per-opponent standings — on a thin baseline pool this IS the read
    opp = defaultdict(lambda: [0, 0])
    for g in games:
        opp[g["opp_name"]][0 if g["we_won"] else 1] += 1
    print("per-opponent standings:")
    for name, (w, l) in sorted(opp.items(), key=lambda kv: -(kv[1][0] + kv[1][1])):
        tot = w + l
        print(f"  {w:>3}W {l:>3}L  {w / tot:>5.1%}  {name}")
    print()

    # game length
    def lengths(gs):
        return [g["choices"][-1][0] for g in gs] or [0]
    lw, ll = lengths(wins), lengths(losses)
    print(f"avg game length: wins {sum(lw)/len(lw):.1f} turns, "
          f"losses {sum(ll)/len(ll):.1f} turns")

    # first KO correlation
    first_ko_us = first_ko_them = 0
    fk_win, fk_loss = 0, 0
    for g in games:
        if not g["faints"]:
            continue
        side = g["faints"][0][0]
        ours = side == g["our_role"]
        if ours:
            first_ko_us += 1
            if g["we_won"]:
                fk_win += 1
        else:
            first_ko_them += 1
    for g in losses:
        if g["faints"] and g["faints"][0][0] != g["our_role"]:
            fk_loss += 1
    print(f"first faint is OURS in {first_ko_us}/{len(games)} games "
          f"(we still won {fk_win} of those); "
          f"we drew first blood in {fk_loss} of our {len(losses)} losses")

    # which of our mons dies first in losses, and to what
    first_death = Counter()
    killers = Counter()
    for g in losses:
        our_faints = [f for f in g["faints"] if f[0] == g["our_role"]]
        if our_faints:
            first_death[our_faints[0][1]] += 1
        for _, victim, _, (kmon, kmove) in our_faints:
            killers[f"{kmon} / {kmove}"] += 1
    print("\nfirst of OUR mons to die (losses):")
    for name, c in first_death.most_common(6):
        print(f"  {c:>3}  {name}")
    print("\ntop killers of our mons (losses):")
    for name, c in killers.most_common(8):
        print(f"  {c:>3}  {name}")

    # tera usage
    def tera_stats(gs, label):
        ours_t, theirs_t, ours_none = [], [], 0
        for g in gs:
            if g["our_role"] in g["teras"]:
                ours_t.append(g["teras"][g["our_role"]][0])
            else:
                ours_none += 1
            if g["opp_role"] in g["teras"]:
                theirs_t.append(g["teras"][g["opp_role"]][0])
        avg = lambda x: sum(x) / len(x) if x else float("nan")
        print(f"  {label}: we tera'd in {len(ours_t)}/{len(gs)} "
              f"(avg turn {avg(ours_t):.1f}); "
              f"they tera'd in {len(theirs_t)}/{len(gs)} "
              f"(avg turn {avg(theirs_t):.1f})")
    print("\ntera timing:")
    tera_stats(wins, "wins  ")
    tera_stats(losses, "losses")

    # score collapses in losses
    print("\nscore collapses (largest single-turn avg_score drop per loss):")
    at_actions, before_actions = Counter(), Counter()
    drops = []
    collapses = []
    for g in losses:
        c = biggest_collapse(g)
        if c is None:
            continue
        collapses.append((c[1], g, c))
        drops.append(c[1])
        norm = lambda a: ("switch" if a.startswith("switch")
                          else ("tera" if a.endswith("(tera)") else a))
        at_actions[norm(c[2])] += 1
        before_actions[norm(c[3])] += 1
    if drops:
        print(f"  mean max-drop {sum(drops)/len(drops):.2f}; "
              f">0.4 cliffs in {sum(1 for d in drops if d > 0.4)}/{len(drops)} losses")
    print("  action at the cliff:")
    for name, c in at_actions.most_common(6):
        print(f"    {c:>3}  {name}")
    print("  action just before the cliff:")
    for name, c in before_actions.most_common(6):
        print(f"    {c:>3}  {name}")

    if args.collapse_examples:
        collapses.sort(key=lambda x: -x[0])
        print(f"\n=== {args.collapse_examples} worst collapses ===")
        for drop, g, c in collapses[: args.collapse_examples]:
            print(f"\n--- {g['file']} {g['room']} (winner {g['winner']}) ---")
            print(f"  cliff: T{c[0]} {c[4]:.2f}->{c[5]:.2f} "
                  f"after [{c[3]}] then [{c[2]}]")
            for t, a, v, s in g["choices"]:
                marker = " <-- cliff" if t == c[0] else ""
                print(f"    T{t:>2} {s:.2f}  {a}{marker}")
            for side, name, t, (km, kv) in g["faints"]:
                who = "US " if side == g["our_role"] else "OPP"
                print(f"    faint T{t:>2} {who} {name:<14} last hit: {km} / {kv}")


if __name__ == "__main__":
    main()
