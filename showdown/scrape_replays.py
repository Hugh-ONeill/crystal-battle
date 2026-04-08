#!/usr/bin/env python3
# scrape gen2ou replays from Pokemon Showdown
#
# Usage:
#   .venv/bin/python showdown/scrape_replays.py --pages 50 --min-rating 1200

import argparse
import json
import os
import time
from pathlib import Path

import urllib.request

SEARCH_URL = "https://replay.pokemonshowdown.com/search.json?format=gen2ou"
REPLAY_URL = "https://replay.pokemonshowdown.com/{}.json"
OUT_DIR = Path(__file__).parent / "replays"


def fetch_json(url: str) -> dict | list:
    req = urllib.request.Request(url, headers={"User-Agent": "Pokemon Bot Research"})
    with urllib.request.urlopen(req, timeout=15) as resp:
        return json.loads(resp.read())


def scrape_replay_list(pages: int = 50, min_rating: int = 0) -> list[dict]:
    """Paginate through search results, collecting replay metadata."""
    all_replays = []
    before = None

    for page in range(pages):
        url = SEARCH_URL
        if before:
            url += f"&before={before}"

        try:
            data = fetch_json(url)
        except Exception as e:
            print(f"  page {page + 1}: fetch error: {e}")
            break

        if not data:
            print(f"  page {page + 1}: no more results")
            break

        # filter by rating
        rated = [r for r in data if r.get("rating") and r["rating"] >= min_rating]
        # also include tournament replays (no rating but high quality)
        tours = [r for r in data if "smogtours" in r.get("id", "")]
        combined = {r["id"]: r for r in rated + tours}
        all_replays.extend(combined.values())

        total_rated = len([r for r in data if r.get("rating")])
        print(f"  page {page + 1}: {len(data)} results, {total_rated} rated, "
              f"{len(combined)} kept (total: {len(all_replays)})")

        # paginate using oldest uploadtime
        before = min(r["uploadtime"] for r in data)
        time.sleep(0.5)

    return all_replays


def download_replays(replays: list[dict], out_dir: Path) -> int:
    """Download full replay logs."""
    out_dir.mkdir(parents=True, exist_ok=True)
    downloaded = 0

    for i, r in enumerate(replays):
        rid = r["id"]
        out_path = out_dir / f"{rid}.json"
        if out_path.exists():
            downloaded += 1
            continue

        try:
            data = fetch_json(REPLAY_URL.format(rid))
            with open(out_path, "w") as f:
                json.dump(data, f)
            downloaded += 1
            if (i + 1) % 20 == 0:
                print(f"  downloaded {i + 1}/{len(replays)}")
            time.sleep(0.3)
        except Exception as e:
            print(f"  {rid}: error: {e}")

    return downloaded


def parse_replay_log(log: str) -> list[dict]:
    """Parse a Showdown replay log into per-turn action records.

    Returns list of dicts with:
      turn: int
      p1_action: str (move name or "switch:Species")
      p2_action: str
      p1_pokemon: str (active species)
      p2_pokemon: str (active species)
    """
    turns = []
    current_turn = 0
    p1_active = ""
    p2_active = ""
    p1_action = ""
    p2_action = ""
    p1_team = []
    p2_team = []

    for line in log.split("\n"):
        parts = line.split("|")
        if len(parts) < 2:
            continue

        cmd = parts[1]

        if cmd == "turn":
            # save previous turn
            if current_turn > 0 and (p1_action or p2_action):
                turns.append({
                    "turn": current_turn,
                    "p1_action": p1_action,
                    "p2_action": p2_action,
                    "p1_pokemon": p1_active,
                    "p2_pokemon": p2_active,
                })
            current_turn = int(parts[2])
            p1_action = ""
            p2_action = ""

        elif cmd == "switch" or cmd == "drag":
            # |switch|p1a: Nickname|Species, Gender|HP/maxHP
            if len(parts) >= 4:
                player_slot = parts[2].strip()
                species_info = parts[3].strip()
                species = species_info.split(",")[0].strip()

                if player_slot.startswith("p1"):
                    if current_turn > 0:
                        p1_action = f"switch:{species}"
                    p1_active = species
                    if species not in p1_team:
                        p1_team.append(species)
                elif player_slot.startswith("p2"):
                    if current_turn > 0:
                        p2_action = f"switch:{species}"
                    p2_active = species
                    if species not in p2_team:
                        p2_team.append(species)

        elif cmd == "move":
            # |move|p1a: Nickname|Move Name|target
            if len(parts) >= 4:
                player_slot = parts[2].strip()
                move_name = parts[3].strip()

                if player_slot.startswith("p1"):
                    p1_action = move_name
                elif player_slot.startswith("p2"):
                    p2_action = move_name

    # save last turn
    if current_turn > 0 and (p1_action or p2_action):
        turns.append({
            "turn": current_turn,
            "p1_action": p1_action,
            "p2_action": p2_action,
            "p1_pokemon": p1_active,
            "p2_pokemon": p2_active,
        })

    return turns


def main():
    parser = argparse.ArgumentParser(description="Scrape Gen 2 OU replays")
    parser.add_argument("--pages", type=int, default=50)
    parser.add_argument("--min-rating", type=int, default=1200)
    parser.add_argument("--download", action="store_true",
                        help="Download full replay logs")
    parser.add_argument("--parse", action="store_true",
                        help="Parse downloaded replays and show stats")
    args = parser.parse_args()

    if args.parse:
        replay_dir = OUT_DIR
        if not replay_dir.exists():
            print("No replays downloaded yet. Run with --download first.")
            return

        files = list(replay_dir.glob("*.json"))
        print(f"Parsing {len(files)} replays...")

        total_turns = 0
        total_games = 0
        for f in files:
            with open(f) as fp:
                data = json.load(fp)
            turns = parse_replay_log(data.get("log", ""))
            if turns:
                total_games += 1
                total_turns += len(turns)

        print(f"  {total_games} valid games, {total_turns} total turns")
        print(f"  avg {total_turns / max(total_games, 1):.1f} turns/game")
        return

    print(f"Scraping gen2ou replays (min rating: {args.min_rating})...")
    replays = scrape_replay_list(pages=args.pages, min_rating=args.min_rating)
    print(f"\nFound {len(replays)} replays")

    if replays:
        ratings = [r["rating"] for r in replays if r.get("rating")]
        if ratings:
            print(f"Rating range: {min(ratings)}-{max(ratings)}")
        tours = [r for r in replays if "smogtours" in r["id"]]
        print(f"Tournament replays: {len(tours)}")

    if args.download and replays:
        print(f"\nDownloading {len(replays)} replays...")
        n = download_replays(replays, OUT_DIR)
        print(f"Downloaded {n} replays to {OUT_DIR}")


if __name__ == "__main__":
    main()
