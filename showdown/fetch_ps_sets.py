#!/usr/bin/env python3
"""
Fetch Pokemon Showdown's curated set database (dex analysis sets + usage
composites) for gen 9 and vendor it as showdown/ps_sets_gen9.json.

These are complete, curated sets (moves/item/ability/nature/EVs/IVs/tera),
several per species, for every gen9 format including gen9ou and gen9monotype.
They are the tier-1 source for opponent set inference (see ps_sets.py):
joint sets preserve the correlations that chaos-stat marginals lose.

Usage:
  .venv/bin/python showdown/fetch_ps_sets.py
"""

from __future__ import annotations

import json
from pathlib import Path
from urllib.request import Request, urlopen

URL = "https://play.pokemonshowdown.com/data/sets/gen9.json"
DEST = Path(__file__).parent / "ps_sets_gen9.json"


def main():
    req = Request(URL, headers={"User-Agent": "crystal-battle set fetcher"})
    with urlopen(req, timeout=30) as r:
        data = json.loads(r.read())
    DEST.write_text(json.dumps(data))
    formats = {k: len(v.get("dex", {})) for k, v in data.items()}
    print(f"wrote {DEST} ({DEST.stat().st_size // 1024} KB)")
    print("dex species per format:", formats)


if __name__ == "__main__":
    main()
