#!/usr/bin/env python3
"""
Per-type team-composition + set-usage stats from gen9monotype replays.

Reads replay JSONs under showdown/replays/gen9monotype/, detects each side's
mono-type via species-types lookup, and aggregates:
  - top 6-mon team compositions per type
  - per-mon move-usage frequencies (within that type's teams only)
  - per-mon items revealed during play
  - winner skew (which mons / comps are over-represented in winning sides)

This separates "Ceruledge on Ghost teams" from "Ceruledge on Fire teams",
something Smogon's aggregate chaos JSON can't do.

Usage:
  .venv/bin/python showdown/monotype_replay_stats.py --type Fire
  .venv/bin/python showdown/monotype_replay_stats.py --type Fire --min-rating 1700
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

HERE = Path(__file__).parent
SPECIES_TYPES_PATH = HERE / "species_types.json"
REPLAYS_DIR = HERE / "replays" / "gen9monotype"


def load_species_types() -> dict[str, list[str]]:
    return json.load(open(SPECIES_TYPES_PATH))


def detect_team_type(team: list[str], species_types: dict[str, list[str]]) -> str | None:
    """Find the type shared by all 6 mons. Returns None if no single type works."""
    if not team:
        return None
    type_sets = []
    for sp in team:
        types = species_types.get(sp)
        if not types:
            # Try base species (e.g. "Urshifu-Rapid-Strike" without subform suffix)
            base = sp.split("-")[0]
            types = species_types.get(base)
        if not types:
            return None
        type_sets.append(set(types))
    common = set.intersection(*type_sets)
    if len(common) == 1:
        return next(iter(common))
    if len(common) > 1:
        # Multiple types shared — pick deterministically. Rare in monotype.
        return sorted(common)[0]
    return None


def parse_replay(data: dict, species_types: dict) -> dict | None:
    """Extract structured info from one replay JSON. Returns None if unparseable."""
    log = data.get("log", "")
    if not log:
        return None

    p1_team: list[str] = []
    p2_team: list[str] = []
    p1_rating: int | None = None
    p2_rating: int | None = None
    p1_name: str = ""
    p2_name: str = ""
    winner: str | None = None
    # nickname -> species (from teampreview reveals and switches)
    nick_to_species: dict[str, str] = {}
    # species -> set of moves used
    moves_used: dict[str, set[str]] = defaultdict(set)
    # species -> set of items revealed
    items_revealed: dict[str, set[str]] = defaultdict(set)
    # species -> set of abilities revealed
    abilities_revealed: dict[str, set[str]] = defaultdict(set)

    for line in log.split("\n"):
        if not line.startswith("|"):
            continue
        parts = line.split("|")
        if len(parts) < 2:
            continue
        cmd = parts[1]

        if cmd == "player" and len(parts) >= 6:
            # |player|p1|username|avatar|rating
            side = parts[2]
            name = parts[3]
            try:
                rating = int(parts[5]) if parts[5] else None
            except (ValueError, IndexError):
                rating = None
            if side == "p1":
                p1_name, p1_rating = name, rating
            elif side == "p2":
                p2_name, p2_rating = name, rating

        elif cmd == "poke" and len(parts) >= 4:
            # |poke|p1|Species, gender|item-flag (rare)
            side = parts[2]
            species = parts[3].split(",")[0].strip()
            if side == "p1":
                p1_team.append(species)
            elif side == "p2":
                p2_team.append(species)

        elif cmd in ("switch", "drag") and len(parts) >= 4:
            # |switch|p1a: Nickname|Species, gender|HP
            slot_info = parts[2]  # e.g. "p1a: Nickname"
            species = parts[3].split(",")[0].strip()
            if ":" in slot_info:
                nick = slot_info.split(":", 1)[1].strip()
                nick_to_species[f"{slot_info.split(':')[0].strip()}|{nick}"] = species

        elif cmd == "move" and len(parts) >= 4:
            # |move|p1a: Nickname|Move|target
            slot_info = parts[2]
            move = parts[3].strip()
            if ":" in slot_info:
                slot = slot_info.split(":")[0].strip()
                nick = slot_info.split(":", 1)[1].strip()
                species = nick_to_species.get(f"{slot}|{nick}")
                if species:
                    moves_used[species].add(move)

        elif cmd == "-item" and len(parts) >= 4:
            # |-item|p1a: Nickname|ItemName|[from] move: Trick
            slot_info = parts[2]
            item = parts[3].strip()
            if ":" in slot_info:
                slot = slot_info.split(":")[0].strip()
                nick = slot_info.split(":", 1)[1].strip()
                species = nick_to_species.get(f"{slot}|{nick}")
                if species and item:
                    items_revealed[species].add(item)

        elif cmd == "-enditem" and len(parts) >= 4:
            # Berry consumed, item Knocked Off, etc.
            slot_info = parts[2]
            item = parts[3].strip()
            if ":" in slot_info:
                slot = slot_info.split(":")[0].strip()
                nick = slot_info.split(":", 1)[1].strip()
                species = nick_to_species.get(f"{slot}|{nick}")
                if species and item:
                    items_revealed[species].add(item)

        elif cmd == "-ability" and len(parts) >= 4:
            slot_info = parts[2]
            ability = parts[3].strip()
            if ":" in slot_info:
                slot = slot_info.split(":")[0].strip()
                nick = slot_info.split(":", 1)[1].strip()
                species = nick_to_species.get(f"{slot}|{nick}")
                if species and ability:
                    abilities_revealed[species].add(ability)

        elif cmd == "win" and len(parts) >= 3:
            winner = parts[2].strip()

    if len(p1_team) != 6 or len(p2_team) != 6:
        return None

    p1_type = detect_team_type(p1_team, species_types)
    p2_type = detect_team_type(p2_team, species_types)

    return {
        "p1_team": p1_team, "p1_type": p1_type, "p1_rating": p1_rating, "p1_name": p1_name,
        "p2_team": p2_team, "p2_type": p2_type, "p2_rating": p2_rating, "p2_name": p2_name,
        "winner": winner,
        "moves_used": {k: sorted(v) for k, v in moves_used.items()},
        "items_revealed": {k: sorted(v) for k, v in items_revealed.items()},
        "abilities_revealed": {k: sorted(v) for k, v in abilities_revealed.items()},
    }


def aggregate(replays: list[dict], target_type: str, min_rating: int,
              max_rating: int | None = None) -> dict:
    """Aggregate per-type stats from parsed replays.

    Rating filter is half-open: min_rating <= r < max_rating (if max_rating set).
    """
    teams_seen = Counter()  # frozenset of 6 mons -> count
    teams_won = Counter()   # frozenset of 6 mons -> wins
    mon_freq = Counter()    # species -> appearances in target-type teams
    mon_wins = Counter()    # species -> wins for that mon's team
    mon_moves = defaultdict(Counter)  # species -> Counter(move -> times seen on that mon)
    mon_items = defaultdict(Counter)
    mon_abilities = defaultdict(Counter)
    n_sides = 0  # total target-type sides seen

    for r in replays:
        for side in ("p1", "p2"):
            if r[f"{side}_type"] != target_type:
                continue
            rating = r[f"{side}_rating"] or 0
            if rating < min_rating:
                continue
            if max_rating is not None and rating >= max_rating:
                continue
            team = r[f"{side}_team"]
            name = r[f"{side}_name"]
            team_key = frozenset(team)
            teams_seen[team_key] += 1
            n_sides += 1
            won = r["winner"] == name
            if won:
                teams_won[team_key] += 1
            for mon in team:
                mon_freq[mon] += 1
                if won:
                    mon_wins[mon] += 1
                for mv in r["moves_used"].get(mon, []):
                    mon_moves[mon][mv] += 1
                for it in r["items_revealed"].get(mon, []):
                    mon_items[mon][it] += 1
                for ab in r["abilities_revealed"].get(mon, []):
                    mon_abilities[mon][ab] += 1

    return {
        "n_sides": n_sides,
        "teams_seen": teams_seen,
        "teams_won": teams_won,
        "mon_freq": mon_freq,
        "mon_wins": mon_wins,
        "mon_moves": mon_moves,
        "mon_items": mon_items,
        "mon_abilities": mon_abilities,
    }


def print_report(agg: dict, target_type: str, min_rating: int,
                 max_rating: int | None = None, top_teams: int = 10):
    n = agg["n_sides"]
    if max_rating is not None:
        range_str = f"{min_rating} <= rating < {max_rating}"
    else:
        range_str = f"rating >= {min_rating}"
    print(f"\n=== {target_type} Monotype Stats ({range_str}, {n} sides) ===\n")
    if n == 0:
        print("No qualifying sides found.")
        return

    print(f"## Top {top_teams} team compositions")
    for team, count in agg["teams_seen"].most_common(top_teams):
        wins = agg["teams_won"][team]
        pct = wins / count * 100 if count else 0
        members = sorted(team)
        print(f"  [{count}x, {wins}W ({pct:.0f}%)] {' / '.join(members)}")

    print(f"\n## Per-mon usage (in {target_type} teams)")
    print(f"  {'mon':<28} {'usage%':>7} {'winrate%':>9} (count, wins)")
    for mon, count in agg["mon_freq"].most_common(20):
        usage = count / n * 100
        wins = agg["mon_wins"][mon]
        wr = wins / count * 100 if count else 0
        print(f"  {mon:<28} {usage:>6.1f}% {wr:>8.1f}% ({count}, {wins})")

    print(f"\n## Per-mon top moves/items/abilities (top 12 mons)")
    for mon, count in agg["mon_freq"].most_common(12):
        moves = agg["mon_moves"][mon].most_common(6)
        items = agg["mon_items"][mon].most_common(3)
        abilities = agg["mon_abilities"][mon].most_common(2)
        print(f"\n  {mon}  ({count} appearances)")
        if moves:
            print("    moves: " + ", ".join(f"{m}({c})" for m, c in moves))
        if items:
            print("    items: " + ", ".join(f"{i}({c})" for i, c in items))
        if abilities:
            print("    abilities: " + ", ".join(f"{a}({c})" for a, c in abilities))


def main():
    parser = argparse.ArgumentParser(description="Per-type monotype replay stats")
    parser.add_argument("--type", required=True, help="Target type, e.g. Fire, Dark, Steel")
    parser.add_argument("--min-rating", type=int, default=1500)
    parser.add_argument("--max-rating", type=int, default=None,
                        help="exclusive upper bound; combined with --min-rating gives [min, max)")
    parser.add_argument("--top-teams", type=int, default=10)
    parser.add_argument("--replays-dir", default=str(REPLAYS_DIR))
    args = parser.parse_args()

    replays_dir = Path(args.replays_dir)
    if not replays_dir.exists():
        print(f"No replays directory at {replays_dir}", file=sys.stderr)
        sys.exit(1)

    species_types = load_species_types()
    print(f"Loaded {len(species_types)} species types")

    files = list(replays_dir.glob("*.json"))
    print(f"Parsing {len(files)} replays...")

    parsed = []
    for f in files:
        try:
            data = json.load(open(f))
            r = parse_replay(data, species_types)
            if r:
                parsed.append(r)
        except Exception as e:
            print(f"  {f.name}: {e}", file=sys.stderr)

    print(f"  {len(parsed)} valid replays parsed")

    # type distribution sanity check
    type_counts = Counter()
    for r in parsed:
        for s in ("p1", "p2"):
            t = r[f"{s}_type"]
            if t:
                type_counts[t] += 1
            else:
                type_counts["UNKNOWN"] += 1
    print(f"\nType distribution across all sides:")
    for t, c in type_counts.most_common():
        print(f"  {t}: {c}")

    agg = aggregate(parsed, args.type, args.min_rating, args.max_rating)
    print_report(agg, args.type, args.min_rating, args.max_rating, args.top_teams)


if __name__ == "__main__":
    main()
