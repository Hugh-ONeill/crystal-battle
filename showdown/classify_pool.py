#!/usr/bin/env python3
"""Classify each team paste in a pool dir into a coarse archetype, for the
`ladder tally arch` view (pools the per-team n so the winrate stops being pure
noise). Prints `<team_basename>\t<archetype>` per file.

Rules are deliberately TRANSPARENT and auditable — `ladder tally arch` echoes
the archetype->teams mapping so a misclassification is visible and this file is
the one place to retune it. Weather is ability-definitive (unambiguous); the
stall / hyper-offense / balance split is threshold heuristics (fuzzier).
"""
import glob
import os
import re
import sys

# normalized (letters+digits only) move / ability tokens
RECOVERY = {"recover", "roost", "slackoff", "softboiled", "wish", "synthesis",
            "morningsun", "moonlight", "shoreup", "rest", "milkdrink",
            "strengthsap", "junglehealing", "lunarblessing"}
HEAL_ABILITY = {"regenerator", "poisonheal"}
SETUP = {"swordsdance", "nastyplot", "dragondance", "calmmind", "quiverdance",
         "bulkup", "shellsmash", "agility", "growth", "victorydance", "tidyup",
         "clangoroussoul", "noretreat", "coil", "workup", "takeheart",
         "filletaway"}


def norm(s):
    return re.sub(r"[^a-z0-9]", "", s.lower())


def classify(text):
    mons = [b for b in re.split(r"\n\s*\n", text.strip()) if b.strip()]
    abils, moves_per, allmoves = [], [], set()
    for b in mons:
        ab, mv = "", []
        for ln in b.splitlines():
            low = ln.strip().lower()
            if low.startswith("ability:"):
                ab = norm(ln.split(":", 1)[1])
            elif ln.strip().startswith("- "):
                mv.append(norm(ln.strip()[2:]))
        abils.append(ab)
        moves_per.append(mv)
        allmoves.update(mv)
    A = set(abils)
    # weather: ability-definitive
    if "drizzle" in A:
        return "rain"
    if "drought" in A or "orichalcumpulse" in A:
        return "sun"
    if "sandstream" in A or "sandspit" in A:
        return "sand"
    if "snowwarning" in A:
        return "snow"
    if "trickroom" in allmoves:
        return "trickroom"
    # heuristic: recovery-carrier count -> stall; setup density -> hyper offense
    recov = sum(1 for i, mv in enumerate(moves_per)
                if (set(mv) & RECOVERY) or abils[i] in HEAL_ABILITY)
    if recov >= 3:
        return "stall/fat"
    setup = sum(1 for mv in moves_per if set(mv) & SETUP)
    if setup >= 3 and recov <= 1:
        return "hyper offense"
    return "balance"


def main():
    pool = sys.argv[1] if len(sys.argv) > 1 else "showdown/teams/pool_hl"
    for f in sorted(glob.glob(os.path.join(pool, "*.txt"))):
        name = os.path.basename(f)[:-4]
        # ah* = the hand-built anti-hazard subpool (pool_hl_manual). They are
        # recovery-heavy by DESIGN so the rules read them as stall/fat, which
        # buried the experiment inside that row; the prefix is the experiment's
        # aggregation key, so it is the archetype.
        if name.startswith("ah"):
            print(f"{name}\tanti-hazard")
            continue
        try:
            arch = classify(open(f, errors="ignore").read())
        except Exception:
            arch = "?"
        print(f"{name}\t{arch}")


if __name__ == "__main__":
    main()
