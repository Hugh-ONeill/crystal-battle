#!/usr/bin/env python3
"""Reorder a team paste so slot 1 holds a plausible lead.

WHY. Paste order encodes a human's intent about how a team is piloted, and our
hand-built teams use it that way — 44% carry a lead-ish mon in slot 1 and none
carry one the role file marks lead_intent:avoid. The metamon-CURATED teams do
not: only 12% have a lead-ish mon first and 50% have one the file does not even
recognise as lead-relevant, because their order is an artifact of replay
reconstruction rather than anyone's choice. That makes slot 1 unusable as a
signal across the pool, and it also means the bench paths that hard-lead slot 1
start those teams on an arbitrary mon.

This picks a lead by explicit, auditable rules and rotates that mon to the
front. Composition is untouched — same six, same sets, same legality — so it
cannot change what a team IS, only where it starts.

  .venv/bin/python showdown/lead_order.py showdown/teams/pool_hl --apply
"""
from __future__ import annotations
import argparse, json, re, sys
from pathlib import Path

HERE = Path(__file__).parent
def n(s): return re.sub(r"[^a-z0-9]", "", (s or "").lower())

ROLES = json.loads((HERE / "roles.json").read_text())["roles"]
LEADISH = {"suicide-lead", "hazard-setter", "screens-setter", "weather-setter",
           "terrain-setter", "lead"}
HAZARD = {"stealthrock", "spikes", "toxicspikes", "stickyweb"}
SCREEN = {"reflect", "lightscreen", "auroraveil"}
FIELD_ABILITY = {"drizzle", "drought", "sandstream", "snowwarning", "grassysurge",
                 "orichalcumpulse", "electricsurge", "psychicsurge", "mistysurge"}


def blocks(text: str) -> list[str]:
    return [b for b in re.split(r"\n\s*\n", text.strip()) if b.strip()]


def parse(block: str) -> dict:
    lines = block.strip().split("\n")
    head = lines[0]
    species = n(head.split("@")[0].split("(")[0])
    ability = ""
    moves = set()
    for l in lines[1:]:
        t = l.strip()
        if t.lower().startswith("ability:"):
            ability = n(t.split(":", 1)[1])
        elif t.startswith("- "):
            moves.add(n(t[2:]))
    return {"species": species, "ability": ability, "moves": moves}


def base_speed(species: str) -> int:
    try:
        from poke_env.data.gen_data import GenData
        e = GenData.from_gen(9).pokedex.get(species) or {}
        return (e.get("baseStats") or {}).get("spe", 60)
    except Exception:
        return 60


def lead_score(mon: dict) -> tuple[int, str]:
    """Higher is a better lead. Reason string is for the audit trail."""
    e = ROLES.get(mon["species"]) or {}
    tags = set(e.get("tags", [])) | {t for s in e.get("sets", []) for t in s.get("tags", [])}
    if e.get("lead_intent") == "avoid":
        return (-10, "role says avoid leading it")
    if e.get("lead_intent") == "strong":
        return (100, "role marks lead_intent strong")
    if mon["ability"] in FIELD_ABILITY:
        return (90, f"sets a field on entry ({mon['ability']})")
    if tags & LEADISH:
        return (80, f"lead-ish role ({', '.join(sorted(tags & LEADISH))})")
    if mon["moves"] & SCREEN:
        return (70, "carries screens")
    if mon["moves"] & HAZARD:
        # a slow bulky wall that happens to carry Stealth Rock is a support
        # piece, not a lead — it wants to come in later, not start the game
        haz = ", ".join(sorted(mon["moves"] & HAZARD))
        if "wall" in tags and base_speed(mon["species"]) < 70:
            return (40, f"sets hazards ({haz}) but is a slow wall")
        return (60, f"sets hazards ({haz})")
    if "uturn" in mon["moves"] or "voltswitch" in mon["moves"] or "flipturn" in mon["moves"]:
        return (30, "pivot move, can scout and leave")
    return (0, "no lead signal")


def reorder(text: str) -> tuple[str, int, str]:
    bs = blocks(text)
    parsed = [parse(b) for b in bs]
    scored = [(lead_score(m), base_speed(m["species"]), i) for i, m in enumerate(parsed)]
    (best, reason), _spe, idx = max(scored, key=lambda x: (x[0][0], x[1], -x[2]))
    if best <= 0:
        # nothing on the team has a lead case; reshuffling on the speed
        # tiebreak alone would be arbitrary, so leave the order as found
        return text, 0, "no mon has a lead signal — left as-is"
    if idx == 0:
        return text, 0, f"already leads correctly ({reason})"
    out = [bs[idx]] + bs[:idx] + bs[idx + 1:]
    return "\n\n".join(out) + "\n", idx, reason


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("pool")
    ap.add_argument("--apply", action="store_true")
    ap.add_argument("--only", default=r"^\d\d_", help="regex on filename; default = curated teams only")
    args = ap.parse_args()
    pat = re.compile(args.only)
    moved = kept = 0
    for f in sorted(Path(args.pool).glob("*.txt")):
        if not pat.match(f.name):
            continue
        text = f.read_text()
        new, idx, reason = reorder(text)
        first_old = parse(blocks(text)[0])["species"]
        first_new = parse(blocks(new)[0])["species"]
        if idx == 0:
            kept += 1
            continue
        moved += 1
        print(f"  {f.name:34s} {first_old:18s} -> {first_new:18s}  ({reason})")
        if args.apply:
            f.write_text(new)
    print(f"\n  {moved} reordered, {kept} already fine"
          + ("" if args.apply else "   [dry run — pass --apply]"))


if __name__ == "__main__":
    main()
