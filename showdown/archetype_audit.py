"""A team named for a mechanic must CONTAIN that mechanic.

WHY THIS EXISTS. On 2026-08-03 the whole webs sub-family — web2, web3 and the
curated 10_araquanid — was found to carry ZERO Sticky Web. They had been built
2026-08-01 and the defining move was simply never included. Every one of them
VALIDATED AS LEGAL, because legality cannot express "this team is not what its
name claims", and round 3's reading of "web2 45.7% vs web3 34.3%" was therefore
measuring two ordinary bulky-offense teams.

The check is cheap and mechanical: map a filename prefix to the ability or move
that defines the archetype, then assert some mon on the roster provides it.
Nothing here is clever — the bug survived weeks precisely because nobody wrote
the obvious assertion.

ALSO CHECKED: anti-synergies, where a team carries something its own strategy
disables. Psychic Terrain blocks priority moves against grounded targets on
BOTH sides, so a Psychic Surge team running Sucker Punch or Grassy Glide has
paid slots for moves it has switched off itself. That is the same class of
defect — a team that is not what it claims — just harder to see.

Usage: archetype_audit.py [dir ...]     (default: every showdown/teams/*/)
Exit status is 1 if any team fails, so it can gate a bench.
"""
import glob
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from showdown.local_battle import parse_showdown_team


def _norm(s):
    return re.sub(r"[^a-z0-9]", "", str(s).lower())


# filename prefix -> (human label, abilities that satisfy it, moves that satisfy it)
# A team satisfies the rule if it provides EITHER the ability or the move: snow
# can come from Snow Warning or from clicking Snowscape, and both are real
# builds.
ARCHETYPES = [
    ("web",    "Sticky Web",      set(),                 {"stickyweb"}),
    ("snow",   "snow",            {"snowwarning"},       {"snowscape", "hail"}),
    ("sun",    "sun",             {"drought", "orichalcumpulse"}, {"sunnyday"}),
    ("rain",   "rain",            {"drizzle"},           {"raindance"}),
    ("sand",   "sand",            {"sandstream", "sandspit"}, {"sandstorm"}),
    ("troom",  "Trick Room",      set(),                 {"trickroom"}),
    ("psytr",  "Psychic Terrain + Trick Room",
                                  {"psychicsurge"},      {"trickroom"}),
    ("eterr",  "Electric Terrain", {"electricsurge"},    {"electricterrain"}),
    ("gterr",  "Grassy Terrain",  {"grassysurge"},       {"grassyterrain"}),
    ("mist",   "Misty Terrain",   {"mistysurge"},        {"mistyterrain"}),
    ("tr",     "a terrain",       {"grassysurge", "psychicsurge", "electricsurge",
                                   "mistysurge"},
                                  {"grassyterrain", "psychicterrain",
                                   "electricterrain", "mistyterrain"}),
    ("screens", "screens",        set(),                 {"reflect", "lightscreen",
                                                          "auroraveil"}),
    ("ho",     "screens or hazards",
                                  set(),                 {"reflect", "lightscreen",
                                                          "auroraveil", "stealthrock",
                                                          "spikes", "stickyweb"}),
    ("stall",  "recovery",        {"regenerator", "poisonheal"},
                                  {"recover", "roost", "softboiled", "slackoff",
                                   "synthesis", "moonlight", "morningsun",
                                   "shoreup", "rest", "wish", "painsplit",
                                   "strengthsap"}),
]

# Psychic Terrain blocks priority against grounded targets on BOTH sides, so
# these are dead weight on a Psychic Surge team rather than a legality problem.
PRIORITY_MOVES = {
    "suckerpunch", "grassyglide", "aquajet", "iceshard", "bulletpunch",
    "extremespeed", "shadowsneak", "machpunch", "quickattack", "thunderclap",
    "iceshard", "vacuumwave", "watershuriken", "jetpunch", "upperhand",
}


# Species whose EVERY curated set carries a defining move — if a team runs one
# of these and none of that move, the mon is off-role. This is the check that
# would have caught the live one: pool_hl/10_araquanid was named for its
# SPECIES, not its archetype, so no name rule could ever have flagged it, and
# it sat in the ladder pool with no Sticky Web. Derived from the curated
# corpus rather than hand-listed, so it stays true as the corpus moves.
def _off_role_species(threshold=0.90):
    """species -> defining moves that ~every real set of it carries.

    Judged from CHAOS USAGE, not from the curated corpus. The first version of
    this used "every curated set carries it", which is trivially true for any
    species the corpus lists once — it flagged Blissey for lacking Stealth Rock
    and Skarmory for lacking Spikes, both perfectly ordinary builds. Real usage
    separates them cleanly: Sticky Web is on 96% of Araquanid, while Stealth
    Rock is on 60% of Blissey, Spikes 79% of Skarmory, Stealth Rock 73% of
    Iron Treads. Only a near-universal move means the mon is off-role without
    it.
    """
    import json
    here = os.path.dirname(os.path.abspath(__file__))
    DEFINING = {"stickyweb", "trickroom", "auroraveil", "raindance", "sunnyday",
                "sandstorm", "snowscape", "memento", "healingwish"}
    try:
        data = json.load(open(os.path.join(here, "gen9ou_chaos.json")))["data"]
    except Exception:
        return {}
    out = {}
    for sp, e in data.items():
        mv = e.get("Moves") or {}
        total = sum(mv.values()) / 4.0
        if total <= 0:
            continue
        hits = {_norm(k) for k, v in mv.items()
                if _norm(k) in DEFINING and v / total >= threshold}
        if hits:
            out[_norm(sp)] = hits
    return out


_OFF_ROLE = None


def audit(path):
    """-> (ok, label, list of problem strings)"""
    base = os.path.basename(path)[:-4]
    try:
        mons = parse_showdown_team(open(path).read())
    except Exception as exc:
        return False, "?", [f"could not parse: {exc.__class__.__name__}"]
    abilities = {_norm(m.get("ability")) for m in mons}
    moves = {_norm(x) for m in mons for x in (m.get("moves") or [])}

    problems = []
    label = None
    for prefix, name, need_ab, need_mv in ARCHETYPES:
        if not re.match(rf"^(\d+_)?{prefix}\d*_", base) and not base.startswith(prefix):
            continue
        label = name
        if not (need_ab & abilities) and not (need_mv & moves):
            want = " or ".join(sorted(need_ab | need_mv))
            problems.append(f"claims {name!r} but no mon provides it (need: {want})")
        break

    global _OFF_ROLE
    if _OFF_ROLE is None:
        _OFF_ROLE = _off_role_species()
    for m in mons:
        need = _OFF_ROLE.get(_norm(m.get("species")))
        if not need:
            continue
        has = {_norm(x) for x in (m.get("moves") or [])}
        if not (need & has):
            problems.append(
                f"{m.get('species')} is off-role: ~every real set carries "
                f"{'/'.join(sorted(need))} and this one carries none")

    if "psychicsurge" in abilities:
        clash = PRIORITY_MOVES & moves
        if clash:
            problems.append(
                "Psychic Surge blocks priority for BOTH sides, so these are "
                f"self-disabled: {', '.join(sorted(clash))}")
    problems.extend(_harvest_defects(mons))
    return not problems, label or "-", problems


# HARVEST-DEFECT CLASSES (2026-08-04). Most of the pool was scraped from
# high-ladder replays, not built — and real ladder teams carry real
# sloppiness plus scraper artifacts. Found live: 32_primarina's Enamorus
# ran CALM MIND ON A CHOICE SCARF (a dead slot; the mon is locked the
# moment it boosts) in the pool for weeks. These are set-level
# contradictions, mechanical to check and invisible to legality.

def _status_moves():
    """Move ids whose category is Status, from the dex the belief system
    already loads. Trick/Switcheroo excluded — see the choice rule."""
    from showdown.set_inference import _moves_data
    return {mid for mid, e in _moves_data().items()
            if (e.get("category") or "").lower() == "status"}


_CHOICE_ITEMS = {"choiceband", "choicespecs", "choicescarf"}
_TRICKS = {"trick", "switcheroo"}
# self-fainting support moves are clicked once and the lock never matters —
# Scarf Healing Wish Enamorus is a real set, not a defect (learned from the
# rw_proxy build, 2026-08-04)
_SELF_FAINT = {"healingwish", "lunardance", "memento"}


def _harvest_defects(mons) -> list[str]:
    out = []
    try:
        status = _status_moves()
    except Exception:
        return out
    for m in mons:
        sp = m.get("species", "?")
        item = _norm(m.get("item"))
        mv = [_norm(x) for x in (m.get("moves") or [])]
        dead_av = sorted(set(mv) & status - _TRICKS)
        dead_choice = sorted(set(dead_av) - _SELF_FAINT)
        if item in _CHOICE_ITEMS and dead_choice and not (set(mv) & _TRICKS):
            # a Trick set legitimately carries status to use AFTER tricking
            # the item away; without Trick, a choiced status move is a slot
            # the mon can only click by accepting a lock into a non-attack.
            # Self-faint support (Healing Wish class) is exempt: clicked
            # once, the lock never matters. AV is NOT exempt — the vest
            # forbids selecting ANY status move, self-faint included.
            out.append(f"{sp}: {item} with status move(s) "
                       f"{'/'.join(dead_choice)} and no Trick — dead slot(s)")
        if item == "assaultvest" and dead_av:
            out.append(f"{sp}: Assault Vest forbids selecting status "
                       f"move(s) {'/'.join(dead_av)} — unusable slot(s)")
    return out


def main(dirs):
    files = []
    for d in dirs:
        files += sorted(glob.glob(os.path.join(d, "*.txt")))
    checked = failed = 0
    for f in files:
        ok, label, problems = audit(f)
        if label == "-" and ok:
            continue          # not an archetype-named team; nothing claimed
        checked += 1
        if not ok:
            failed += 1
            print(f"FAIL  {os.path.relpath(f)}")
            for p in problems:
                print(f"      {p}")
    print(f"\n{checked} archetype-named teams checked, {failed} failed"
          f" (of {len(files)} team files)")
    return 1 if failed else 0


if __name__ == "__main__":
    args = sys.argv[1:] or sorted(glob.glob(
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "teams", "*")))
    sys.exit(main([a for a in args if os.path.isdir(a)]))
