#!/usr/bin/env python3
"""
Build tier-2 opponent set data from the scraped replay corpus.

Extracts, per replay side: the previewed roster, each mon's revealed moves,
items (direct reveals plus [from] item: attributions), abilities, and tera
types. Aggregates two indexes:

  species: per-species moveset FRAGMENTS with counts (what was actually seen
           together in one game), plus item/ability/tera counters
  teams:   team-archetype index keyed by the sorted 6-species roster — ladder
           players copy whole teams, so matching a preview against known
           archetypes predicts the sets of everything, revealed or not

Output: showdown/gen9ou_replay_sets.json (see replay_sets.py for the reader).

Usage:
  .venv/bin/python showdown/build_replay_sets.py --format gen9ou
"""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter, defaultdict
from pathlib import Path

HERE = Path(__file__).parent


def _normalize(name: str) -> str:
    return re.sub(r"[^a-z0-9]", "", name.lower())


def _species_from_details(details: str) -> str:
    return _normalize(details.split(",")[0])


_SUBJECT_RE = re.compile(r"^(p[12])[a-c]?: (.*)$")


def parse_replay(log: str):
    """Yield (side, species, moves_set, item, ability, tera) per rostered mon."""
    roster: dict[str, set[str]] = {"p1": set(), "p2": set()}
    nick: dict[tuple[str, str], str] = {}
    moves = defaultdict(set)
    items: dict[tuple[str, str], str] = {}
    abilities: dict[tuple[str, str], str] = {}
    teras: dict[tuple[str, str], str] = {}

    def subject(field: str) -> tuple[str, str] | None:
        m = _SUBJECT_RE.match(field)
        if not m:
            return None
        side, nickname = m.group(1), m.group(2).strip()
        species = nick.get((side, nickname))
        return (side, species) if species else None

    for line in log.split("\n"):
        if not line.startswith("|"):
            continue
        parts = line.split("|")
        if len(parts) < 3:
            continue
        kind = parts[1]
        if kind == "poke" and len(parts) >= 4:
            roster[parts[2]].add(_species_from_details(parts[3]))
        elif kind in ("switch", "drag") and len(parts) >= 4:
            m = _SUBJECT_RE.match(parts[2])
            if m:
                side, nickname = m.group(1), m.group(2).strip()
                species = _species_from_details(parts[3])
                nick[(side, nickname)] = species
                roster[side].add(species)  # no-preview edge; usually a no-op
        elif kind == "move" and len(parts) >= 4:
            key = subject(parts[2])
            if key:
                moves[key].add(_normalize(parts[3]))
        elif kind in ("-item", "-enditem") and len(parts) >= 4:
            key = subject(parts[2])
            if key and not any("[from] move:" in p for p in parts):
                items.setdefault(key, _normalize(parts[3]))
        elif kind == "-terastallize" and len(parts) >= 4:
            key = subject(parts[2])
            if key:
                teras[key] = parts[3].lower()
        elif kind == "-ability" and len(parts) >= 4:
            key = subject(parts[2])
            if key:
                abilities.setdefault(key, _normalize(parts[3]))
        if "[from] item:" in line or "[from] ability:" in line:
            frm_item = re.search(r"\[from\] item: ([^|]+)", line)
            frm_ab = re.search(r"\[from\] ability: ([^|]+)", line)
            of = re.search(r"\[of\] (p[12][a-c]?: [^|]+)", line)
            holder = subject(of.group(1)) if of else subject(parts[2])
            if holder:
                if frm_item:
                    items.setdefault(holder, _normalize(frm_item.group(1)))
                if frm_ab:
                    abilities.setdefault(holder, _normalize(frm_ab.group(1)))

    for side in ("p1", "p2"):
        for species in roster[side]:
            key = (side, species)
            yield (side, species, moves.get(key, set()),
                   items.get(key), abilities.get(key), teras.get(key))


def iter_replays(replay_dir: Path | None, jsonl_paths: list[str], fmt: str):
    """Yield (replay_id, log) across scraped-dir JSONs and JSONL dumps
    (metamon), deduplicated by replay id (first seen wins)."""
    seen: set[str] = set()
    if replay_dir is not None and replay_dir.exists():
        for path in sorted(replay_dir.glob(f"{fmt}-*.json")):
            try:
                data = json.loads(path.read_text())
            except Exception:
                continue
            rid = str(data.get("id") or path.stem)
            if rid in seen:
                continue
            seen.add(rid)
            yield rid, data.get("log") or ""
    for jp in jsonl_paths:
        with open(jp) as f:
            for line in f:
                try:
                    data = json.loads(line)
                except Exception:
                    continue
                rid = str(data.get("id") or "")
                if not rid or rid in seen:
                    continue
                seen.add(rid)
                yield rid, data.get("log") or ""


def main():
    parser = argparse.ArgumentParser(description="Build replay set data")
    parser.add_argument("--format", default="gen9ou")
    parser.add_argument("--min-team-count", type=int, default=2,
                        help="drop team archetypes seen fewer times")
    parser.add_argument("--jsonl", action="append", default=[],
                        help="additional replay JSONL dump(s), e.g. the "
                             "metamon filter output; merged with the scraped "
                             "dir, deduped by replay id")
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    replay_dir = HERE / "replays" / args.format
    print(f"parsing replays from {replay_dir} + {len(args.jsonl)} jsonl dump(s)")

    species_agg: dict[str, dict] = defaultdict(lambda: {
        "games": 0, "movesets": Counter(), "items": Counter(),
        "abilities": Counter(), "teras": Counter()})
    team_agg: dict[str, dict] = defaultdict(lambda: {
        "count": 0, "mons": defaultdict(lambda: {
            "movesets": Counter(), "items": Counter(), "teras": Counter()})})

    parsed = 0
    for rid, log in iter_replays(replay_dir, args.jsonl, args.format):
        try:
            if "|poke|" not in log:
                continue
            results = list(parse_replay(log))  # parse ONCE per replay
            side_mons = defaultdict(list)
            for side, sp, mv, item, ability, tera in results:
                side_mons[side].append(sp)
                agg = species_agg[sp]
                agg["games"] += 1
                if mv:
                    agg["movesets"][tuple(sorted(mv))] += 1
                if item:
                    agg["items"][item] += 1
                if ability:
                    agg["abilities"][ability] += 1
                if tera:
                    agg["teras"][tera] += 1
            for side, mons in side_mons.items():
                if len(mons) != 6:
                    continue
                key = "|".join(sorted(mons))
                team = team_agg[key]
                team["count"] += 1
                for sd, sp, mv, item, _, tera in results:
                    # match on SIDE too: shared species across both teams
                    # used to cross-pollute archetype movesets
                    if sd != side or sp not in mons:
                        continue
                    entry = team["mons"][sp]
                    if mv:
                        entry["movesets"][tuple(sorted(mv))] += 1
                    if item:
                        entry["items"][item] += 1
                    if tera:
                        entry["teras"][tera] += 1
            parsed += 1
            if parsed % 25000 == 0:
                print(f"  parsed {parsed} "
                      f"({len(team_agg)} archetypes so far)", flush=True)
        except Exception:
            continue
    print(f"parsed {parsed} replays, {len(species_agg)} species, "
          f"{len(team_agg)} team archetypes")

    def counter_out(c: Counter, top: int):
        return [[k if isinstance(k, str) else list(k), v]
                for k, v in c.most_common(top)]

    out = {"format": args.format, "replays": parsed, "species": {}, "teams": {}}
    for sp, agg in species_agg.items():
        out["species"][sp] = {
            "games": agg["games"],
            "movesets": counter_out(agg["movesets"], 25),
            "items": counter_out(agg["items"], 10),
            "abilities": counter_out(agg["abilities"], 5),
            "teras": counter_out(agg["teras"], 8),
        }
    kept = 0
    for key, team in team_agg.items():
        if team["count"] < args.min_team_count:
            continue
        kept += 1
        out["teams"][key] = {
            "count": team["count"],
            "mons": {sp: {
                "movesets": counter_out(e["movesets"], 5),
                "items": counter_out(e["items"], 3),
                "teras": counter_out(e["teras"], 3),
            } for sp, e in team["mons"].items()},
        }
    dest = Path(args.out) if args.out else HERE / f"{args.format}_replay_sets.json"
    dest.write_text(json.dumps(out))
    print(f"kept {kept} archetypes (count >= {args.min_team_count}); "
          f"wrote {dest} ({dest.stat().st_size // 1024} KB)")


if __name__ == "__main__":
    main()
