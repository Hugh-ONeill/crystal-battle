"""Stall-mechanics audit of specific marathon games.

WHY (2026-07-23): the tempo ledger says we lose the LONG games and the
deep-vs-shallow counterfactual cleared search depth, leaving long-horizon
stall mechanics (PP economy, Toxic clocks, recovery wars) as prime suspects
— resources MCTS cannot price because their payoff sits 50+ turns out. This
tool audits named games from bench logs: per-mon move-use vs PP caps (Sleep
Talk-called moves excluded from PP), damage taken by source, healing by
source with pre-click HP for recovery moves, Toxic infliction, Struggle
events, and the faint timeline with killer attribution.

Usage:
  stall_audit.py --logs "showdown/bench/poolrun_L*_ours.log" \
      battle-gen9ou-3604 battle-gen9ou-3615 battle-gen9ou-3623
"""
import argparse
import glob
import re
from collections import Counter, defaultdict

try:
    from showdown.deep_shallow import ROOM_RE
    from showdown.tempo_ledger import _frac, _ident
except ImportError:
    from deep_shallow import ROOM_RE
    from tempo_ledger import _frac, _ident

# max PP (base*8/5) for the moves that decide stall wars
PP_CAP = {"Recover": 8, "Roost": 8, "Moonlight": 8, "Soft-Boiled": 8,
          "Slack Off": 8, "Synthesis": 8, "Morning Sun": 8, "Rest": 8,
          "Protect": 16, "Substitute": 16, "Toxic": 16, "Sleep Talk": 16}
RECOVERY = {"Recover", "Roost", "Moonlight", "Soft-Boiled", "Slack Off",
            "Synthesis", "Morning Sun", "Rest", "Wish"}


def dmg_class(line):
    m = re.search(r"\[from\] ([^|]+)", line)
    if not m:
        return "direct"
    src = m.group(1).strip()
    if src in ("psn", "tox"):
        return "poison"
    if src == "brn":
        return "burn"
    if src in ("Stealth Rock", "Spikes", "move: Stealth Rock", "move: Spikes"):
        return "hazards"
    if src.startswith("item:") or src.startswith("ability:"):
        return src
    return src if ":" in src else src.lower()


def audit_game(lines, room):
    roles, hp = {}, {}
    turn = 0
    g = dict(room=room, winner=None, turns=0,
             moves=defaultdict(Counter), called=defaultdict(Counter),
             dmg=defaultdict(Counter), heal=defaultdict(Counter),
             heal_pre=defaultdict(list), tox_given=Counter(),
             struggle=[], faints=[], last_dmg={}, last_move={})
    for line in lines:
        i = line.find("|")
        if i < 0:
            continue
        parts = line.rstrip("\n").split("|")[1:]
        if not parts:
            continue
        tag = parts[0]
        if tag == "player" and len(parts) >= 3 and parts[2]:
            roles[parts[1]] = "cb" if parts[2].startswith("CBGen9") else "fp"
        elif tag == "turn":
            turn = int(parts[1])
            g["turns"] = turn
        elif tag == "move" and len(parts) >= 3:
            who = _ident(parts[1])
            if not who or who[0] not in roles:
                continue
            side, mon = roles[who[0]], who[1]
            if "[from]move: Sleep Talk" in line or "[from] move: Sleep Talk" in line:
                g["called"][(side, mon)][parts[2]] += 1
            else:
                g["moves"][(side, mon)][parts[2]] += 1
            g["last_move"][who] = parts[2]
            if parts[2] == "Struggle":
                g["struggle"].append((turn, side, mon))
        elif tag in ("switch", "drag", "replace") and len(parts) >= 4:
            who = _ident(parts[1])
            f = _frac(parts[3]) if who else None
            if who and f is not None:
                hp[who] = f
        elif tag == "-damage" and len(parts) >= 3:
            who = _ident(parts[1])
            if not who or who[0] not in roles:
                continue
            f = _frac(parts[2])
            if f is None:
                continue
            delta = hp.get(who, 1.0) - f
            hp[who] = f
            cls = dmg_class(line)
            if delta > 0:
                g["dmg"][roles[who[0]]][cls] += delta
                g["last_dmg"][who] = cls
        elif tag == "-heal" and len(parts) >= 3:
            who = _ident(parts[1])
            if not who or who[0] not in roles:
                continue
            f = _frac(parts[2])
            if f is None:
                continue
            old = hp.get(who, 1.0)
            hp[who] = f
            src = dmg_class(line)   # same [from] convention
            if src == "direct":     # no [from]: the mon's own move this turn
                mv = g["last_move"].get(who, "?")
                src = f"move: {mv}"
                if mv in RECOVERY:
                    g["heal_pre"][roles[who[0]]].append(old)
            if f > old:
                g["heal"][roles[who[0]]][src] += f - old
        elif tag == "-status" and len(parts) >= 3:
            who = _ident(parts[1])
            if who and who[0] in roles and parts[2] == "tox":
                # credit the INFLICTING side (the opponent of the receiver)
                g["tox_given"]["fp" if roles[who[0]] == "cb" else "cb"] += 1
        elif tag == "faint" and len(parts) >= 2:
            who = _ident(parts[1])
            if who and who[0] in roles:
                hp[who] = 0.0
                g["faints"].append((turn, roles[who[0]], who[1],
                                    g["last_dmg"].get(who, "?")))
        elif tag == "win" and len(parts) >= 2:
            g["winner"] = "cb" if parts[1].startswith("CBGen9") else "fp"
    return g


def room_lines(files, room):
    out = []
    for f in files:
        cur = None
        for line in open(f, errors="replace"):
            m = ROOM_RE.search(line)
            if m:
                cur = m.group(1)
            if cur == room:
                out.append(line)
    return out


def show(g):
    print(f"\n=== {g['room']}: {g['winner']} wins in {g['turns']} turns ===")
    print("  move usage (PP-relevant, called-by-Sleep-Talk in parens):")
    for side in ("cb", "fp"):
        for (s, mon), cnt in sorted(g["moves"].items()):
            if s != side:
                continue
            bits = []
            for mv, c in cnt.most_common():
                cap = PP_CAP.get(mv)
                mark = "!" if cap and c >= cap else ""
                called = g["called"][(s, mon)].get(mv, 0)
                bits.append(f"{mv} x{c}{mark}"
                            + (f"(+{called})" if called else ""))
            print(f"    {side} {mon:12s} " + ", ".join(bits))
    print("  damage taken by source (in mons of HP):")
    for side in ("cb", "fp"):
        tot = sum(g["dmg"][side].values())
        d = "  ".join(f"{k}:{v:.2f}" for k, v in
                      g["dmg"][side].most_common(7))
        print(f"    {side} total {tot:.2f}  {d}")
    print("  healing gained by source (in mons of HP):")
    for side in ("cb", "fp"):
        tot = sum(g["heal"][side].values())
        h = "  ".join(f"{k}:{v:.2f}" for k, v in
                      g["heal"][side].most_common(6))
        pre = g["heal_pre"][side]
        avg = 100 * sum(pre) / len(pre) if pre else 0
        print(f"    {side} total {tot:.2f}  {h}  "
              f"(recovery clicks: {len(pre)}, avg pre-click HP {avg:.0f}%)")
    print(f"  toxics landed: cb {g['tox_given']['cb']}, "
          f"fp {g['tox_given']['fp']}")
    if g["struggle"]:
        st = Counter((s, m) for _, s, m in g["struggle"])
        first = g["struggle"][0]
        print(f"  struggle: first {first[1]} {first[2]} T{first[0]}; " +
              ", ".join(f"{s} {m} x{c}" for (s, m), c in st.items()))
    print("  faint timeline (turn, side, mon, killer):")
    for t, s, m, k in g["faints"]:
        print(f"    T{t:3d} {s} {m:12s} <- {k}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("rooms", nargs="+")
    ap.add_argument("--logs", nargs="+", required=True)
    args = ap.parse_args()
    files = sorted(f for p in args.logs for f in glob.glob(p))
    games = []
    for room in args.rooms:
        lines = room_lines(files, room)
        if not lines:
            print(f"{room}: not found in logs")
            continue
        g = audit_game(lines, room)
        games.append(g)
        show(g)

    if len(games) > 1:
        print(f"\n=== aggregate over {len(games)} games ===")
        for side in ("cb", "fp"):
            dmg = Counter()
            heal = Counter()
            for g in games:
                dmg.update(g["dmg"][side])
                heal.update(g["heal"][side])
            print(f"  {side} damage taken {sum(dmg.values()):.2f} mons: "
                  + "  ".join(f"{k}:{v:.2f}" for k, v in dmg.most_common(6)))
            print(f"  {side} healing     {sum(heal.values()):.2f} mons: "
                  + "  ".join(f"{k}:{v:.2f}" for k, v in heal.most_common(5)))
        tox = Counter()
        for g in games:
            tox.update(g["tox_given"])
        print(f"  toxics landed: cb {tox['cb']}, fp {tox['fp']}")


if __name__ == "__main__":
    main()
