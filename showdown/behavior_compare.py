"""Behavioral comparison of CB vs foul-play from mirror-match bench logs.

Every game is a mirror (same team both sides), so any difference in move
usage is pure policy. Parses the full battle protocol out of ours.log
(both players' actions are in it) and aggregates per-side metrics.
"""
import glob
import re
import sys
from collections import Counter, defaultdict

from poke_env.data import GenData

MOVES = GenData.from_gen(9).moves

PIVOTS = {"uturn", "voltswitch", "flipturn", "partingshot", "chillyreception",
          "teleport", "batonpass", "shedtail"}
HAZARDS = {"stealthrock", "spikes", "toxicspikes", "stickyweb"}
REMOVAL = {"rapidspin", "defog", "tidyup", "mortalspin", "courtchange"}
PROTECT = {"protect", "detect", "banefulbunker", "silktrap", "burningbulwark",
           "spikyshield", "kingsshield", "obstruct"}
PHAZE = {"whirlwind", "roar", "dragontail", "circlethrow"}

def mid(name):
    return re.sub(r"[^a-z0-9]", "", name.lower())

def classify(move_id):
    m = MOVES.get(move_id)
    if move_id in HAZARDS: return "hazard"
    if move_id in REMOVAL: return "removal"
    if move_id in PIVOTS: return "pivot"
    if move_id in PROTECT: return "protect"
    if move_id in PHAZE: return "phaze"
    if m is None: return "unknown"
    if m.get("category") != "Status": return "attack"
    if m.get("heal") or "heal" in m.get("flags", {}): return "recovery"
    b = m.get("boosts")
    if b and m.get("target") == "self" and any(v > 0 for v in b.values()):
        return "boost"
    # setup that boosts via volatile/other (e.g. DD handled above; BU too)
    if move_id in {"dragondance", "bulkup", "calmmind", "swordsdance",
                   "nastyplot", "quiverdance", "irondefense", "curse",
                   "agility", "shellsmash", "victorydance", "shiftgear"}:
        return "boost"
    if m.get("status") or m.get("volatileStatus") in {"confusion"}:
        return "status_infl"
    return "other_status"

P = re.compile(r"^\|")

def games_from_log(path):
    """Yield per-game event dicts parsed from one ours.log."""
    role = {}          # 'p1'/'p2' -> 'fp'|'cb'
    g = None
    for line in open(path, errors="replace"):
        # protocol lines can be continuation lines or embedded after INFO -
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
            g = dict(turn=0, events=[], winner=None, turns=0,
                     first_blood=None, tera={}, faints=Counter())
            role = {}
        if g is None:
            continue
        if tag == "player" and len(parts) >= 3 and parts[2]:
            role[parts[1]] = "cb" if parts[2].startswith("CBGen9") else "fp"
        elif tag == "turn":
            g["turn"] = int(parts[1]); g["turns"] = g["turn"]
        elif tag == "move" and len(parts) >= 3:
            side = role.get(parts[1][:2])
            if side:
                g["events"].append((g["turn"], side, "move", mid(parts[2])))
        elif tag in ("switch", "drag") and len(parts) >= 2:
            side = role.get(parts[1][:2])
            if side:
                g["events"].append((g["turn"], side, tag, parts[2].split(",")[0]))
        elif tag == "faint":
            side = role.get(parts[1][:2])
            if side:
                g["faints"][side] += 1
                if g["first_blood"] is None:
                    # first blood CREDIT goes to the opponent of the fainter
                    g["first_blood"] = ("cb" if side == "fp" else "fp", g["turn"])
        elif tag == "-terastallize":
            side = role.get(parts[1][:2])
            if side and side not in g["tera"]:
                g["tera"][side] = g["turn"]
        elif tag in ("-fail", "-immune", "-miss") and len(parts) >= 2:
            side = role.get(parts[1][:2])
            if side:
                g["events"].append((g["turn"], side, tag, ""))
        elif tag == "cant" and len(parts) >= 3:
            side = role.get(parts[1][:2])
            if side:
                g["events"].append((g["turn"], side, "cant", parts[2]))
        elif tag == "-status" and len(parts) >= 3:
            side = role.get(parts[1][:2])   # side RECEIVING the status
            if side:
                g["events"].append((g["turn"], side, "gotstatus", parts[2]))
        elif tag == "win":
            g["winner"] = "cb" if parts[1].startswith("CBGen9") else "fp"
    if g and g.get("winner"):
        yield g


def main(patterns):
    files = [f for p in patterns for f in glob.glob(p)]
    games = [g for f in files for g in games_from_log(f)]
    print(f"{len(games)} decided games from {len(files)} logs")
    cbw = sum(1 for g in games if g["winner"] == "cb")
    print(f"CB record: {cbw}W-{len(games)-cbw}L "
          f"({100*cbw/len(games):.1f}%)  avg turns {sum(g['turns'] for g in games)/len(games):.0f}")

    # --- move category mix per side (and by phase) ---
    cat = defaultdict(Counter)      # side -> category counter
    phase_cat = defaultdict(Counter)  # (side, phase) -> counter
    misc = defaultdict(Counter)     # side -> fails/immunes/cant etc.
    for g in games:
        for turn, side, kind, val in g["events"]:
            ph = "early(1-10)" if turn <= 10 else "mid(11-30)" if turn <= 30 else "late(31+)"
            if kind == "move":
                c = classify(val)
                cat[side][c] += 1
                phase_cat[(side, ph)][c] += 1
            elif kind == "switch":
                cat[side]["hard_switch"] += 1
                phase_cat[(side, ph)]["hard_switch"] += 1
            elif kind in ("-fail", "-immune", "-miss"):
                misc[side][kind] += 1
            elif kind == "cant":
                misc[side]["turns_lost(cant)"] += 1
            elif kind == "gotstatus":
                misc[side][f"got_{val}"] += 1

    cats = sorted(set(cat["cb"]) | set(cat["fp"]),
                  key=lambda c: -(cat["cb"][c] + cat["fp"][c]))
    tot = {s: sum(cat[s].values()) for s in ("cb", "fp")}
    print(f"\n=== action mix (per 100 actions) ===   CB(n={tot['cb']})  FP(n={tot['fp']})")
    for c in cats:
        print(f"  {c:14s}  {100*cat['cb'][c]/tot['cb']:6.2f}  {100*cat['fp'][c]/tot['fp']:6.2f}")

    print("\n=== action mix by phase (per 100 actions in phase) ===")
    for ph in ("early(1-10)", "mid(11-30)", "late(31+)"):
        t = {s: max(1, sum(phase_cat[(s, ph)].values())) for s in ("cb", "fp")}
        keys = sorted(set(phase_cat[("cb", ph)]) | set(phase_cat[("fp", ph)]),
                      key=lambda c: -(phase_cat[("cb", ph)][c] + phase_cat[("fp", ph)][c]))
        row = "  ".join(f"{c}:{100*phase_cat[('cb',ph)][c]/t['cb']:.1f}/{100*phase_cat[('fp',ph)][c]/t['fp']:.1f}"
                        for c in keys[:8])
        print(f"  {ph:12s} (cb/fp) {row}")

    print("\n=== per-game incident rates ===        CB      FP")
    n = len(games)
    for k in sorted(set(misc["cb"]) | set(misc["fp"])):
        print(f"  {k:20s}  {misc['cb'][k]/n:6.2f}  {misc['fp'][k]/n:6.2f}")

    # --- first blood ---
    fb = Counter(g["first_blood"][0] for g in games if g["first_blood"])
    fbturn = defaultdict(list)
    for g in games:
        if g["first_blood"]:
            fbturn[g["first_blood"][0]].append(g["first_blood"][1])
    print(f"\n=== first blood ===  CB {fb['cb']} ({100*fb['cb']/n:.0f}%)  FP {fb['fp']}")
    fbw = sum(1 for g in games if g["first_blood"] and g["first_blood"][0] == g["winner"])
    print(f"  first-blood side wins the game: {100*fbw/n:.0f}%")

    # --- tera timing ---
    for side in ("cb", "fp"):
        used = [g["tera"][side] for g in games if side in g["tera"]]
        by_out = {o: [g["tera"][side] for g in games
                      if side in g["tera"] and g["winner"] == o] for o in ("cb", "fp")}
        won = by_out[side]; lost = by_out["fp" if side == "cb" else "cb"]
        print(f"  {side} tera: used in {len(used)}/{n} games, mean turn "
              f"{sum(used)/max(1,len(used)):.0f} "
              f"(when winning {sum(won)/max(1,len(won)):.0f}, "
              f"when losing {sum(lost)/max(1,len(lost)):.0f})")


if __name__ == "__main__":
    main(sys.argv[1:])
