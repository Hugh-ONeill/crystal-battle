#!/usr/bin/env python3
"""Mid-game screen: does the roles knowledge disagree with what we actually did?

Before wiring any consumer (LLM or rule), find out whether the annotations in
roles.json have an OPINION on real decisions and whether that opinion differs
from the move MCTS chose. If they never disagree, the knowledge is inert and no
delivery mechanism helps; if they disagree constantly on positions MCTS was
confident about, acting on them means overriding a confident search, which this
campaign has repeatedly lost doing.

Reads finished logs — no MCTS runs, no LLM. Each rule below is a direct
encoding of a field already in roles.json, so a rule firing means the file
would have said something at that moment.

Importable: margin_screen.py reuses games() + opinions() so the rule encoding
exists exactly once — a drifted copy would silently measure different rules.

  .venv/bin/python showdown/roles_screen.py
"""
from __future__ import annotations
import glob, json, re, sys
from pathlib import Path

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE.parent))
def n(s): return re.sub(r"[^a-z0-9]", "", (s or "").lower())

R = json.loads((HERE / "roles.json").read_text())["roles"]
SETUP = {"dragondance","swordsdance","calmmind","nastyplot","bulkup","irondefense","curse","quiverdance"}
WEATHER_MOVE = {"raindance":"rain","sunnyday":"sun","sandstorm":"sand","snowscape":"snow","chillyreception":"snow"}

# We are PAC-Crystal on the ladder and CBGen9L<lane><arm> on the bench; rooms
# are battle-gen9oulongtimer-N there, battle-gen9ou-N on the local server.
US_RE = re.compile(r"PAC-Crystal|CBGen9\w*")
TAG_RE = re.compile(r">(battle-gen9ou\w*-\d+)")

def games(paths):
    for p in paths:
        cur=None; st={}
        for line in Path(p).read_text(errors="replace").split("\n"):
            m=TAG_RE.search(line)
            if m:
                cur=m.group(1)
                st.setdefault(cur, dict(tag=cur,us=None,ours=[],active=None,hp={},
                                        fallen=0,weather=None,turn=0,events=[]))
            if cur is None: continue
            g=st[cur]; s=line.strip()
            pm=re.match(r"\|player\|(p[12])\|([^|]+)", s)
            if pm and US_RE.fullmatch(pm.group(2).strip()): g["us"]=pm.group(1)
            km=re.match(r"\|poke\|(p[12])\|([^,|]+)", s)
            if km and km.group(1)==g["us"]: g["ours"].append(n(km.group(2)))
            if s.startswith("|turn|"): g["turn"]=int(s.split("|")[2])
            wm=re.match(r"\|-weather\|(\w+)", s)
            if wm: g["weather"]=None if wm.group(1)=="none" else n(wm.group(1))
            # species may carry a gender/form suffix ("Kingambit, M") before the
            # HP field — matching only up to the comma then demanding a pipe
            # silently dropped every gendered mon's switch-in
            sw=re.match(r"\|(?:switch|drag)\|(p[12])a: [^|]*\|([^,|]+)[^|]*\|(\d+)/(\d+)", s)
            if sw and sw.group(1)==g["us"]:
                g["active"]=n(sw.group(2))
                g["hp"][g["active"]]=(int(sw.group(3)), int(sw.group(4)))
                g["events"].append(("switch", g["turn"], g["active"], dict(fallen=g["fallen"])))
            dm=re.match(r"\|-damage\|(p[12])a: [^|]*\|(\d+)/(\d+)", s)
            if dm and dm.group(1)==g["us"] and g["active"]:
                g["hp"][g["active"]]=(int(dm.group(2)), int(dm.group(3)))
            fm=re.match(r"\|faint\|(p[12])a:", s)
            if fm and fm.group(1)==g["us"]:
                g["fallen"]+=1
                # the fainting mon is whoever is active; the faint line itself
                # carries only the nickname, so record the tracked active
                if g["active"]: g.setdefault("dead",set()).add(g["active"])
            tm=re.match(r"^  T(\d+): (\S+)", line)
            if tm and g["active"]:
                cur_hp=g["hp"].get(g["active"],(100,100))
                g["events"].append(("move", int(tm.group(1)), n(tm.group(2)),
                                    dict(active=g["active"], hp=cur_hp[0]/max(1,cur_hp[1]),
                                         fallen=g["fallen"], weather=g["weather"], ours=list(g["ours"]),
                                         # snapshot AT EVENT TIME: opinions()
                                         # runs post-game, and the end-state
                                         # dead set would anachronistically
                                         # filter setters that died later
                                         dead=set(g.get("dead",set())))))
        for g in st.values():
            if g["us"] and g["ours"]: yield g

def opinions(g):
    """Every moment a roles rule has an opinion on this game, as
    dict(rule, turn, disagreed, mode, targets):
      mode "veto"   — targets are engine-speak choice strings the rule says
                      NOT to pick ("-tera" suffix counts as its base move)
      mode "demand" — targets are the choices the rule says TO pick
    The (mode, targets) pair is what margin_screen.py prices against the
    ranked visit list; the screen's own tally only reads rule/disagreed."""
    for kind, turn, what, ctx in g["events"]:
        # RULE 1 — cleaner deployment: roles marks Kingambit value_curve
        # grows_with_own_faints and lead_intent avoid, i.e. hold it back.
        if kind=="switch" and what=="kingambit" and R.get("kingambit"):
            yield dict(rule="hold the cleaner (Kingambit) until allies have fallen",
                       turn=turn, disagreed=ctx["fallen"] <= 1,
                       mode="veto", targets=("switch kingambit",))
        # RULE 2 — entry_condition full_hp: a setup plan that needs an
        # undamaged entry is dead once the mon is chipped.
        if kind=="move" and what in SETUP:
            e=R.get(ctx["active"])
            if e and e.get("entry_condition")=="full_hp":
                yield dict(rule="don't start a full_hp setup plan while chipped",
                           turn=turn, disagreed=ctx["hp"] < 0.99,
                           mode="veto", targets=tuple(SETUP))
        # RULE 3 — weather resource: a setter on our team with its field down.
        if kind=="move":
            # ALIVE setters only: the 2026-07-31 advocate sweep exposed that
            # this rule fired on 184 fainted-setter turns (the entire
            # "no-alternative" bucket) — a demand nobody could satisfy
            dead=ctx.get("dead",set())
            setters=[m for m in ctx["ours"]
                     if (R.get(m,{}) or {}).get("resource") and m not in dead]
            if setters and ctx["active"] not in setters:
                want={R[m]["resource"] for m in setters}
                if ctx["weather"] not in want:
                    # roles says the field is a team resource; the repair is to
                    # switch the setter in (its ability sets the field on entry)
                    yield dict(rule="our weather/terrain is down while its setter is alive",
                               turn=turn, disagreed=not what.startswith("switch"),
                               mode="demand", targets=tuple(f"switch {m}" for m in setters))

def main():
    fire = {}   # rule -> [times it had an opinion, times our play disagreed]
    for g in games(sorted(glob.glob(str(HERE/"bench"/"overnight_2026*_ladder.log")))):
        for op in opinions(g):
            e=fire.setdefault(op["rule"],[0,0]); e[0]+=1; e[1]+=op["disagreed"]
    print(f"  RULE COVERAGE AND DISAGREEMENT (490-game log corpus)\n")
    print(f"  {'rule':56s} {'fired':>7s} {'disagreed':>10s} {'rate':>6s}")
    for k,(c,dis) in sorted(fire.items(), key=lambda kv:-kv[1][0]):
        print(f"  {k:56s} {c:7d} {dis:10d} {100*dis/max(1,c):5.0f}%")

if __name__ == "__main__":
    main()
