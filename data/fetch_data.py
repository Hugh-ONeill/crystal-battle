#!/usr/bin/env python3
# Fetch veekun CSV data and emit bundled JSON for Gen 2 Crystal
# Usage: python data/fetch_data.py

import csv
import io
import json
import os
import urllib.request
from pathlib import Path

BASE_URL = "https://raw.githubusercontent.com/veekun/pokedex/master/pokedex/data/csv"

CSV_FILES = [
    "pokemon_species",
    "pokemon_stats",
    "pokemon_types",
    "pokemon_moves",
    "moves",
    "move_names",
    "types",
    "type_efficacy",
    "type_names",
    "move_meta",
    "pokemon_species_names",
]

# ============================================================
# GEN 2 CONSTANTS
# ============================================================

GEN2_MAX_SPECIES = 251
GEN2_MAX_MOVE = 251
CRYSTAL_VERSION_GROUP = 7

# type ids from veekun
FAIRY_TYPE_ID = 18

# gen 2 physical/special split is by TYPE, not per-move
PHYSICAL_TYPE_IDS = {1, 2, 3, 4, 5, 6, 7, 8, 9}   # normal..steel
SPECIAL_TYPE_IDS = {10, 11, 12, 13, 14, 15, 16, 17}  # fire..dark

# gen 2 corrections to type chart
TYPE_CHART_OVERRIDES = {
    (8, 9): 50,   # ghost -> steel: not-very-effective
    (17, 9): 50,  # dark -> steel: not-very-effective
}

# stat id -> name mapping (veekun)
STAT_NAMES = {1: "hp", 2: "attack", 3: "defense", 4: "special_attack", 5: "special_defense", 6: "speed"}


def fetch_csv(name: str, cache_dir: Path) -> list[dict]:
    cache_path = cache_dir / f"{name}.csv"
    if cache_path.exists():
        text = cache_path.read_text()
    else:
        url = f"{BASE_URL}/{name}.csv"
        print(f"  downloading {name}.csv ...")
        resp = urllib.request.urlopen(url)
        text = resp.read().decode("utf-8")
        cache_path.write_text(text)
    return list(csv.DictReader(io.StringIO(text)))


def build_type_map(types_rows: list[dict], type_names_rows: list[dict]) -> dict:
    """Map type_id -> type name (english), excluding Fairy."""
    names = {}
    for row in type_names_rows:
        if int(row["local_language_id"]) == 9:  # english
            names[int(row["type_id"])] = row["name"].lower()

    result = {}
    for row in types_rows:
        tid = int(row["id"])
        if tid == FAIRY_TYPE_ID or tid >= 10000:  # exclude fairy + shadow types
            continue
        if tid in names:
            result[tid] = names[tid]
    return result


def build_type_chart(efficacy_rows: list[dict], type_map: dict) -> dict:
    """17x17 type effectiveness chart with Gen 2 corrections."""
    chart = {}
    for row in efficacy_rows:
        atk = int(row["damage_type_id"])
        dfn = int(row["target_type_id"])
        factor = int(row["damage_factor"])

        if atk not in type_map or dfn not in type_map:
            continue

        # apply gen 2 overrides
        if (atk, dfn) in TYPE_CHART_OVERRIDES:
            factor = TYPE_CHART_OVERRIDES[(atk, dfn)]

        atk_name = type_map[atk]
        dfn_name = type_map[dfn]
        chart.setdefault(atk_name, {})[dfn_name] = factor / 100.0

    return chart


def build_moves(move_rows: list[dict], move_names_rows: list[dict],
                move_meta_rows: list[dict], type_map: dict) -> dict:
    """Build move data dict keyed by move_id."""
    # english names
    names = {}
    for row in move_names_rows:
        if int(row["local_language_id"]) == 9:
            names[int(row["move_id"])] = row["name"]

    # meta (ailment, drain, healing, etc.)
    meta = {}
    for row in move_meta_rows:
        mid = int(row["move_id"])
        meta[mid] = {
            "ailment_id": int(row["meta_ailment_id"]),
            "min_hits": int(row["min_hits"]) if row["min_hits"] else None,
            "max_hits": int(row["max_hits"]) if row["max_hits"] else None,
            "drain": int(row["drain"]),
            "healing": int(row["healing"]),
            "crit_rate": int(row["crit_rate"]),
            "ailment_chance": int(row["ailment_chance"]),
            "flinch_chance": int(row["flinch_chance"]),
            "stat_chance": int(row["stat_chance"]),
        }

    moves = {}
    for row in move_rows:
        mid = int(row["id"])
        if mid > GEN2_MAX_MOVE:
            continue

        type_id = int(row["type_id"]) if row["type_id"] else None
        if type_id is not None and type_id not in type_map:
            continue

        power = int(row["power"]) if row["power"] else 0
        accuracy = int(row["accuracy"]) if row["accuracy"] else None
        pp = int(row["pp"]) if row["pp"] else 0
        priority = int(row["priority"]) if row["priority"] else 0

        # gen 2: damage class determined by type, not per-move
        if power > 0 and type_id is not None:
            if type_id in PHYSICAL_TYPE_IDS:
                damage_class = "physical"
            else:
                damage_class = "special"
        elif power == 0:
            damage_class = "status"
        else:
            damage_class = "status"

        moves[mid] = {
            "id": mid,
            "name": names.get(mid, f"move_{mid}"),
            "type": type_map.get(type_id, "normal") if type_id else "normal",
            "type_id": type_id,
            "power": power,
            "accuracy": accuracy,
            "pp": pp,
            "priority": priority,
            "damage_class": damage_class,
            "meta": meta.get(mid),
        }

    return moves


def build_pokemon(species_rows, species_names_rows, stats_rows,
                  types_rows, moves_rows, move_data, type_map):
    """Build pokemon data dict keyed by species_id."""
    # english names
    names = {}
    for row in species_names_rows:
        if int(row["local_language_id"]) == 9:
            sid = int(row["pokemon_species_id"])
            names[sid] = row["name"]

    # base stats
    stats = {}
    for row in stats_rows:
        pid = int(row["pokemon_id"])
        if pid > GEN2_MAX_SPECIES:
            continue
        stat_id = int(row["stat_id"])
        if stat_id in STAT_NAMES:
            stats.setdefault(pid, {})[STAT_NAMES[stat_id]] = int(row["base_stat"])

    # types
    pokemon_types = {}
    for row in types_rows:
        pid = int(row["pokemon_id"])
        if pid > GEN2_MAX_SPECIES:
            continue
        type_id = int(row["type_id"])
        if type_id not in type_map:
            continue
        pokemon_types.setdefault(pid, []).append(type_map[type_id])

    # crystal learnsets (version_group_id=4)
    learnsets = {}
    for row in moves_rows:
        pid = int(row["pokemon_id"])
        if pid > GEN2_MAX_SPECIES:
            continue
        vg = int(row["version_group_id"])
        if vg != CRYSTAL_VERSION_GROUP:
            continue
        mid = int(row["move_id"])
        if mid > GEN2_MAX_MOVE:
            continue
        if mid not in move_data:
            continue
        learnsets.setdefault(pid, set()).add(mid)

    pokemon = {}
    for row in species_rows:
        sid = int(row["id"])
        if sid > GEN2_MAX_SPECIES:
            continue
        if sid not in stats:
            continue

        pokemon[sid] = {
            "id": sid,
            "name": names.get(sid, f"pokemon_{sid}"),
            "base_stats": stats[sid],
            "types": pokemon_types.get(sid, ["normal"]),
            "learnset": sorted(learnsets.get(sid, [])),
        }

    return pokemon


def main():
    data_dir = Path(__file__).parent
    cache_dir = data_dir / ".cache"
    cache_dir.mkdir(exist_ok=True)

    print("fetching veekun CSVs...")
    csvs = {}
    for name in CSV_FILES:
        csvs[name] = fetch_csv(name, cache_dir)

    print("building type map...")
    type_map = build_type_map(csvs["types"], csvs["type_names"])
    print(f"  {len(type_map)} types: {', '.join(sorted(type_map.values()))}")

    print("building type chart...")
    type_chart = build_type_chart(csvs["type_efficacy"], type_map)
    out = data_dir / "type_chart.json"
    out.write_text(json.dumps(type_chart, indent=2, sort_keys=True))
    print(f"  wrote {out}")

    print("building moves...")
    move_data = build_moves(csvs["moves"], csvs["move_names"], csvs["move_meta"], type_map)
    # output as list sorted by id
    moves_list = sorted(move_data.values(), key=lambda m: m["id"])
    out = data_dir / "moves.json"
    out.write_text(json.dumps(moves_list, indent=2))
    print(f"  wrote {out} ({len(moves_list)} moves)")

    print("building pokemon...")
    pokemon_data = build_pokemon(
        csvs["pokemon_species"], csvs["pokemon_species_names"],
        csvs["pokemon_stats"], csvs["pokemon_types"],
        csvs["pokemon_moves"], move_data, type_map
    )
    pokemon_list = sorted(pokemon_data.values(), key=lambda p: p["id"])
    out = data_dir / "pokemon.json"
    out.write_text(json.dumps(pokemon_list, indent=2))
    print(f"  wrote {out} ({len(pokemon_list)} pokemon)")


if __name__ == "__main__":
    main()
