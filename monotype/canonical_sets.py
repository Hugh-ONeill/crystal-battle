"""
Parse Smogon's gen9monotype moveset files into canonical Showdown paste
blocks indexed by (species, type).

Use case: replay data only gives species rosters, but the lead-pick
featurizer wants full pastes (item/ability/moves drive role flags). For
each (species, monotype-type), we pick the most-common ability + item +
spread + 4 most-common moves from the Smogon stats and assemble a paste.

Build the index once with `build_canonical_sets()`; cache the result.

  cs = build_canonical_sets()
  paste_block = cs["fire"]["Heatran"]
"""

from __future__ import annotations

import re
from pathlib import Path

HERE = Path(__file__).parent
SMOGON_STATS_DIR = HERE / "smogon_stats" / "2026-04" / "moveset"

# 18 canonical type files Smogon publishes per monotype bracket.
MONOTYPE_TYPES = (
    "bug", "dark", "dragon", "electric", "fairy", "fighting", "fire",
    "flying", "ghost", "grass", "ground", "ice", "normal", "poison",
    "psychic", "rock", "steel", "water",
)


_SECTION_HEADERS = {"Raw count", "Abilities", "Items", "Spreads", "Moves",
                    "Teammates", "Checks and Counters"}
_PCT_LINE_RE = re.compile(r"\|\s*(.+?)\s+(\d+\.\d+)%\s*\|")
_SPECIES_HEADER_RE = re.compile(r"^\|\s*([A-Za-z0-9\-\.]+(?:\s[A-Za-z0-9\-\.]+)*)\s+\|$")
_DIVIDER_RE = re.compile(r"^\+-+\+$")


def parse_moveset_file(path: Path) -> dict[str, dict]:
    """Parse one gen9monotype-mono<type>-<elo>.txt into a per-species dict.

    Each species value: {abilities, items, spreads, moves} where each is
    a list of (name, pct) pairs sorted by pct desc.
    """
    text = path.read_text()
    lines = text.splitlines()

    species_data: dict[str, dict] = {}
    cur_species: str | None = None
    cur_section: str | None = None

    for ln in lines:
        if _DIVIDER_RE.match(ln):
            continue
        if not ln.startswith("|"):
            continue
        inner = ln.strip().strip("|").strip()

        # Section header: matches one of the well-known section names.
        if inner in _SECTION_HEADERS:
            cur_section = inner
            continue

        # Stat-preamble lines under a species header (e.g. "Raw count: 6962",
        # "Avg. weight: 0.59", "Viability Ceiling: 89"). Skip cleanly.
        if ":" in inner and "%" not in inner:
            continue

        # Species header: single name in the cell, no percentage. Must come
        # *between* sections (cur_section gets reset on the next section
        # header, but we accept this match unconditionally because the line
        # shape is unambiguous — section headers were caught above).
        if "%" not in inner:
            m = _SPECIES_HEADER_RE.match(ln.strip())
            if m:
                name = m.group(1).strip()
                # Guard against stray non-species single-word lines
                if name not in _SECTION_HEADERS:
                    cur_species = name
                    if cur_species not in species_data:
                        species_data[cur_species] = {
                            "abilities": [], "items": [], "spreads": [],
                            "moves": [],
                        }
                    cur_section = None
                    continue

        # Percentage line — "| Name PCT% |" possibly with extra (...) after.
        m = _PCT_LINE_RE.match(ln)
        if not m or cur_species is None or cur_section is None:
            continue
        name, pct = m.group(1).strip(), float(m.group(2))
        if name == "Other":
            continue
        if cur_section == "Abilities":
            species_data[cur_species]["abilities"].append((name, pct))
        elif cur_section == "Items":
            species_data[cur_species]["items"].append((name, pct))
        elif cur_section == "Spreads":
            species_data[cur_species]["spreads"].append((name, pct))
        elif cur_section == "Moves":
            species_data[cur_species]["moves"].append((name, pct))

    return species_data


def _spread_to_nature_evs(spread_str: str) -> tuple[str, list[int]]:
    """'Modest:252/0/0/252/4/0' -> ('Modest', [252,0,0,252,4,0])."""
    nat, evs_str = spread_str.split(":")
    evs = [int(x) for x in evs_str.split("/")]
    return nat.strip(), evs


def _canonical_paste_block(species: str, info: dict) -> str:
    """Assemble a Showdown paste block from the species' top stats."""
    if not info["moves"]:
        return ""  # garbage entry; caller should fall back

    ability = info["abilities"][0][0] if info["abilities"] else "No Ability"
    item = info["items"][0][0] if info["items"] else ""
    spread = info["spreads"][0][0] if info["spreads"] else "Hardy:0/0/0/0/0/0"
    nature, evs = _spread_to_nature_evs(spread)
    # Showdown EV order: HP / Atk / Def / SpA / SpD / Spe
    ev_parts = []
    labels = ("HP", "Atk", "Def", "SpA", "SpD", "Spe")
    for lbl, v in zip(labels, evs):
        if v > 0:
            ev_parts.append(f"{v} {lbl}")
    ev_line = " / ".join(ev_parts) if ev_parts else "0 HP"

    moves = [m for m, _ in info["moves"][:4]]
    while len(moves) < 4:
        moves.append("Tackle")  # filler; shouldn't happen for real species

    item_part = f" @ {item}" if item else ""
    lines = [
        f"{species}{item_part}",
        f"Ability: {ability}",
        f"EVs: {ev_line}",
        f"{nature} Nature",
    ]
    lines += [f"- {m}" for m in moves]
    return "\n".join(lines)


def build_canonical_sets(elo_bucket: int = 1500) -> dict[str, dict[str, str]]:
    """Return {type_lower: {species: paste_block}}.

    Picks each species' top ability/item/spread + 4 most-common moves per
    monotype type bracket at the given ELO.
    """
    out: dict[str, dict[str, str]] = {}
    for t in MONOTYPE_TYPES:
        path = SMOGON_STATS_DIR / f"gen9monotype-mono{t}-{elo_bucket}.txt"
        if not path.exists():
            out[t] = {}
            continue
        species_data = parse_moveset_file(path)
        per_type: dict[str, str] = {}
        for species, info in species_data.items():
            block = _canonical_paste_block(species, info)
            if block:
                per_type[species] = block
        out[t] = per_type
    return out


if __name__ == "__main__":
    cs = build_canonical_sets()
    total = sum(len(v) for v in cs.values())
    print(f"Built canonical sets for {len(cs)} types, {total} (type, species) entries")
    for t in ("fire", "steel", "water"):
        print(f"\n=== {t}: {len(cs[t])} species ===")
        # Show a sample
        sample = next(iter(cs[t].items()), None)
        if sample:
            species, block = sample
            print(f"  sample [{species}]:")
            for line in block.splitlines():
                print(f"    {line}")
