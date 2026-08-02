#!/usr/bin/env python3
"""Team-level role derivation — the pass roles.json structurally cannot do.

WHY THIS IS NOT MORE ANNOTATION. roles.json is per-species by design, and
two of its most decision-relevant properties are not species properties at
all, they are properties of a species IN A TEAM:

  SOLE-X       Corviknight's hazard removal is worth far more when it is the
               team's ONLY remover — lose it and the hazards stay up for the
               rest of the game. The same mon alongside a second remover is
               merely useful. `preserve` cannot express that, because the
               answer changes per team.
  ENTRY        A setup-window mon's condition (`entry_condition`) may be
               unreachable or trivially restorable depending on who else is
               on the side: a slow pivot brings it in without taking a hit,
               a Healing Wish user RESTORES a spent full_hp condition.
  RESOURCE     Venusaur `requires` sun; whether that is satisfiable is a
               question about its teammates. A weather abuser with no setter
               on its own side is a BUILD ERROR, and nothing per-species can
               see it.

So this is a DERIVATION over (roster x roles.json), not a new hand-written
layer: it stays correct as roles.json grows, and it audits our own pool for
build errors as a side effect.

  .venv/bin/python showdown/team_roles.py showdown/teams/pool_hl/*.txt
  .venv/bin/python showdown/team_roles.py --json showdown/teams/pool_hl_manual/tr1_*.txt
"""

from __future__ import annotations

import argparse
import glob
import json
import re
import sys
from pathlib import Path

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE.parent))

from showdown.local_battle import parse_showdown_team  # noqa: E402


def _norm(s: str) -> str:
    return re.sub(r"[^a-z0-9]", "", (s or "").lower())


# Roles whose loss is structural rather than incremental: if the team has
# exactly one holder, that mon's preserve value is understated by its entry.
CRITICAL_ROLES = ("hazard-remover", "hazard-setter", "weather-setter",
                  "terrain-setter", "screens-setter", "trick-room-setter",
                  "tailwind-setter", "cleric", "status-absorber",
                  "anti-setup", "spinblocker", "trapper")

# `requires`/`resource` are free prose (they are read by humans and an LLM),
# so normalise to canonical field tokens. A provider token satisfies a
# consumer token when they are equal, or when the consumer asked for the
# GENERIC form ("terrain") and the provider sets a specific one.
_RESOURCE_PATTERNS = (
    (r"\brain\b", "rain"),
    (r"\bsun\b|sunny", "sun"),
    (r"\bsand\b|sandstorm", "sand"),
    (r"\bsnow\b|hail", "snow"),
    (r"grassy ?terrain|grassyterrain", "grassyterrain"),
    (r"psychic ?terrain", "psychicterrain"),
    (r"electric ?terrain", "electricterrain"),
    (r"misty ?terrain", "mistyterrain"),
    (r"\bterrain\b", "terrain"),          # generic — any terrain will do
    (r"trick ?room", "trickroom"),
    (r"tailwind", "tailwind"),
    (r"aurora ?veil|screens?", "screens"),
)
_SPECIFIC_TERRAINS = {"grassyterrain", "psychicterrain",
                      "electricterrain", "mistyterrain"}


def resource_tokens(text: str) -> set[str]:
    """Canonical field-resource tokens named in a free-text field."""
    if not text:
        return set()
    low = text.lower()
    out = {tok for pat, tok in _RESOURCE_PATTERNS if re.search(pat, low)}
    if out & _SPECIFIC_TERRAINS:
        out.discard("terrain")           # specific beats the generic reading
    return out


def satisfies(provided: set[str], needed: str) -> bool:
    if needed in provided:
        return True
    return needed == "terrain" and bool(provided & _SPECIFIC_TERRAINS)


# Moves that manufacture or restore another mon's entry condition. The
# measured ranking (490 games) said 58% of setups came off a HARD SWITCH and
# only 6% off a slow pivot — so a team carrying these is unusual, and worth
# surfacing rather than assuming.
SLOW_PIVOTS = {"uturn", "voltswitch", "flipturn", "partingshot",
               "chillyreception", "teleport", "shedtail"}
RESTORERS = {"healingwish": "restores a spent full_hp condition outright",
             "lunardance": "restores HP, PP and status completely",
             "wish": "passes a heal, partially restoring full_hp"}


def load_roles() -> dict:
    return json.loads((HERE / "roles.json").read_text())["roles"]


def analyze(team_path: str, roles: dict) -> dict:
    """Analyse a team PASTE on disk."""
    return analyze_roster(parse_showdown_team(Path(team_path).read_text()),
                          roles, name=Path(team_path).stem)


def analyze_roster(mons: list[dict], roles: dict, name: str = "team") -> dict:
    """Analyse a roster given as [{species, moves, item}, ...].

    Split out from analyze() so a LIVE consumer (the overlay dossier) can
    pass battle.team directly instead of round-tripping through a paste.
    """
    roster = []
    for m in mons:
        sp = _norm(m.get("species", ""))
        roster.append({"species": sp,
                       "moves": {_norm(x) for x in m.get("moves") or []},
                       "item": _norm(m.get("item") or ""),
                       "entry": roles.get(sp)})

    known = [r for r in roster if r["entry"]]
    tags_of = lambda r: set(r["entry"].get("tags") or [])

    # --- sole holders of a critical role
    sole, coverage = {}, {}
    for role in CRITICAL_ROLES:
        holders = [r["species"] for r in known if role in tags_of(r)]
        coverage[role] = holders
        if len(holders) == 1:
            sole.setdefault(holders[0], []).append(role)

    # --- field-resource chains: who provides, who needs, who is orphaned
    provides = {}
    for r in known:
        toks = resource_tokens(r["entry"].get("resource", ""))
        if toks:
            provides[r["species"]] = toks
    all_provided = set().union(*provides.values()) if provides else set()

    needs, orphans, dependents = {}, [], {}
    for r in known:
        for tok in resource_tokens(r["entry"].get("requires", "")):
            needs.setdefault(r["species"], set()).add(tok)
            src = [sp for sp, toks in provides.items() if satisfies(toks, tok)]
            if src:
                for s in src:
                    dependents.setdefault(s, set()).add(r["species"])
            else:
                orphans.append({"species": r["species"], "needs": tok})

    # --- entry economics
    pivots = [r["species"] for r in roster if r["moves"] & SLOW_PIVOTS]
    restorers = [{"species": r["species"], "move": mv, "effect": eff}
                 for r in roster for mv, eff in RESTORERS.items()
                 if mv in r["moves"]]
    gated = [{"species": r["species"],
              "condition": r["entry"]["entry_condition"]}
             for r in known if r["entry"].get("entry_condition")]

    return {
        "team": name,
        "roster": [r["species"] for r in roster],
        "unknown": [r["species"] for r in roster if not r["entry"]],
        "sole": sole,
        "coverage": coverage,
        "missing_roles": [k for k, v in coverage.items() if not v],
        "provides": {k: sorted(v) for k, v in provides.items()},
        "needs": {k: sorted(v) for k, v in needs.items()},
        "dependents": {k: sorted(v) for k, v in dependents.items()},
        "orphans": orphans,
        "slow_pivots": pivots,
        "restorers": restorers,
        "entry_gated": gated,
    }


# A wincon is not a wincon until the answers to it are gone — the user's
# definition (2026-08-02): "wincons aren't really wincons until they get the
# right opportunity". That opportunity is mostly the disappearance of the
# specific mons that blank the plan, which is derivable from tags on BOTH
# sides. Each blocker class defeats a setup plan a different way, so they are
# reported separately rather than as one count.
BLOCKER_CLASSES = {
    "anti-setup": "ignores or erases the boosts (Unaware/Haze), so setting up "
                  "gains nothing against it",
    "trapper": "prevents the switch, so the setup mon cannot leave once caught",
    "priority-attacker": "moves first regardless of the Speed boost, so a "
                         "boosted sweeper can still be revenge-killed",
}


def wincon_outlook(a: dict, their_species: list[str], roles: dict) -> list[dict]:
    """For each of our wincons, which of THEIR mons blank it, and what has to
    happen before the window opens."""
    out = []
    ours = [r for r in a["roster"] if roles.get(r)]
    for sp in ours:
        tags = set(roles[sp].get("tags") or [])
        if "wincon" not in tags and "setup-sweeper" not in tags:
            continue
        blockers = []
        for opp in their_species:
            e = roles.get(_norm(opp))
            if not e:
                continue
            for cls in BLOCKER_CLASSES:
                if cls in (e.get("tags") or []):
                    blockers.append({"species": _norm(opp), "class": cls})
        entry = roles[sp]
        out.append({
            "species": sp,
            "is_wincon": "wincon" in tags,
            "entry_condition": entry.get("entry_condition"),
            "sole_wincon": None,          # filled by the caller
            "blockers": blockers,
        })
    n_wincon = sum(1 for w in out if w["is_wincon"])
    for w in out:
        w["sole_wincon"] = w["is_wincon"] and n_wincon == 1
    return out


def wincon_report(rows: list[dict]) -> list[str]:
    out = []
    for w in rows:
        if not w["blockers"] and not w["is_wincon"]:
            continue
        label = "WINCON" if w["is_wincon"] else "setup"
        sole = "  [the team's ONLY wincon — trading it trades the game plan]" \
            if w["sole_wincon"] else ""
        out.append(f"  {label} {w['species']}{sole}")
        if not w["blockers"]:
            out.append("    window is OPEN: nothing on their side blanks it")
            continue
        by = {}
        for b in w["blockers"]:
            by.setdefault(b["class"], []).append(b["species"])
        for cls, who in by.items():
            out.append(f"    blocked by {', '.join(sorted(set(who)))} "
                       f"({cls}: {BLOCKER_CLASSES[cls]})")
        names = sorted({b["species"] for b in w["blockers"]})
        listed = names[0] if len(names) == 1 else \
            ", ".join(names[:-1]) + " and " + names[-1]
        out.append(f"    window opens once {listed} "
                   f"{'is' if len(names) == 1 else 'are'} gone or too weak to act")
    return out


def report(a: dict) -> str:
    out = [f"=== {a['team']}  ({', '.join(a['roster'])})"]
    if a["unknown"]:
        out.append(f"  no roles entry: {', '.join(a['unknown'])}")
    for o in a["orphans"]:
        # An unentered teammate may well be the provider — Pincurchin sets
        # Electric Terrain but had no entry, so Hawlucha read as orphaned.
        # Never call that a build error; it is a COVERAGE gap in roles.json.
        if a["unknown"]:
            out.append(f"  ?  {o['species']} requires {o['needs']} and no "
                       f"ENTERED teammate provides it — unverifiable while "
                       f"{', '.join(a['unknown'])} lack entries")
        else:
            out.append(f"  !! BUILD ERROR: {o['species']} requires "
                       f"{o['needs']} and NO teammate provides it")
    for sp, toks in a["provides"].items():
        dep = a["dependents"].get(sp)
        if dep:
            out.append(f"  {sp} provides {'/'.join(toks)} -> "
                       f"{len(dep)} teammate(s) depend on it: {', '.join(dep)}"
                       f"  [its death disables them]")
    for sp, roles_ in a["sole"].items():
        out.append(f"  SOLE {', '.join(roles_)}: {sp}  [preserve understated "
                   f"by its own entry — the team has no second one]")
    if a["entry_gated"]:
        for g in a["entry_gated"]:
            ways = []
            others = [p for p in a["slow_pivots"] if p != g["species"]]
            if others:                      # a mon is not its own way in
                ways.append("slow pivot: " + ", ".join(others))
            ways += [f"{r['move']} ({r['species']})" for r in a["restorers"]]
            out.append(f"  entry-gated {g['species']} ({g['condition']}) — "
                       + ("ways in: " + "; ".join(ways) if ways
                          else "NO enabler on this team: hard switch only"))
    # only absences that actually cost games: lacking removal means hazards
    # stay up for the whole game, lacking a setter means we never collect the
    # chip our own attrition plan assumes. The rest (trapper, cleric,
    # tailwind) are legitimately absent from most builds and were pure noise.
    for role in ("hazard-remover", "hazard-setter"):
        if not a["coverage"].get(role):
            out.append(f"  no {role} on this team")
    return "\n".join(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("teams", nargs="+", help="team paste files (globs ok)")
    ap.add_argument("--json", action="store_true")
    ap.add_argument("--vs", metavar="SPECIES",
                    help="comma-separated opposing species (or a team file): "
                         "report which of them blank each of our wincons")
    ap.add_argument("--errors-only", action="store_true",
                    help="only teams with a build error (orphaned resource)")
    args = ap.parse_args()

    roles = load_roles()
    paths = [p for t in args.teams for p in sorted(glob.glob(t))] or args.teams
    results = [analyze(p, roles) for p in paths]
    if args.errors_only:
        results = [r for r in results if r["orphans"]]
    if args.json:
        print(json.dumps(results, indent=1))
        return
    their = []
    if args.vs:
        if Path(args.vs).exists():
            their = [_norm(m.get("species", "")) for m in
                     parse_showdown_team(Path(args.vs).read_text())]
        else:
            their = [_norm(x) for x in args.vs.split(",")]
    for r in results:
        print(report(r))
        if their:
            lines = wincon_report(wincon_outlook(r, their, roles))
            if lines:
                print("  -- wincon outlook vs " + ", ".join(their))
                print("\n".join(lines))
    hard = [r for r in results if r["orphans"] and not r["unknown"]]
    soft = [r for r in results if r["orphans"] and r["unknown"]]
    print(f"\n{len(results)} teams analysed, {len(hard)} with a confirmed "
          f"build error, {len(soft)} unverifiable (unentered teammates)")


if __name__ == "__main__":
    main()
