#!/usr/bin/env python3
"""
Fetch Smogon gen9monotype usage / moveset / matchup-chart files for a month.

Pulls into  monotype/smogon_stats/<YYYY-MM>/{usage,moveset,matchup}/
  - usage/gen9monotype-mono<type>-<elo>.txt        (18 types x 4 elos)
  - moveset/gen9monotype-mono<type>-<elo>.txt      (18 types x 4 elos)
  - matchup/gen9monotype-matchup_chart-<elo>.txt   (4 elos)

Usage:
  .venv/bin/python monotype/smogon_stats/fetch_smogon_stats.py --month 2026-04
  .venv/bin/python monotype/smogon_stats/fetch_smogon_stats.py --month 2026-04 --elos 1500 1760
"""

from __future__ import annotations

import argparse
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

TYPES = [
    "bug", "dark", "dragon", "electric", "fairy", "fighting", "fire",
    "flying", "ghost", "grass", "ground", "ice", "normal", "poison",
    "psychic", "rock", "steel", "water",
]
ELOS = [0, 1500, 1630, 1760]
BASE = "https://www.smogon.com/stats"
UA = "crystal-battle monotype stats fetcher (https://github.com/local)"


def fetch(url: str, dest: Path, timeout: float = 30.0) -> tuple[str, str, int]:
    if dest.exists() and dest.stat().st_size > 0:
        return (url, "skip", dest.stat().st_size)
    try:
        req = Request(url, headers={"User-Agent": UA})
        with urlopen(req, timeout=timeout) as r:
            data = r.read()
    except HTTPError as e:
        return (url, f"http{e.code}", 0)
    except (URLError, TimeoutError) as e:
        return (url, f"err:{e}", 0)
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_bytes(data)
    return (url, "ok", len(data))


def build_tasks(month: str, elos: list[int], out_root: Path) -> list[tuple[str, Path]]:
    tasks: list[tuple[str, Path]] = []
    for elo in elos:
        url = f"{BASE}/{month}/monotype/matchupcharts/gen9monotype-matchup_chart-{elo}.txt"
        tasks.append((url, out_root / "matchup" / f"gen9monotype-matchup_chart-{elo}.txt"))
    for t in TYPES:
        for elo in elos:
            fn = f"gen9monotype-mono{t}-{elo}.txt"
            tasks.append((f"{BASE}/{month}/monotype/{fn}", out_root / "usage" / fn))
            tasks.append((f"{BASE}/{month}/monotype/moveset/{fn}", out_root / "moveset" / fn))
    return tasks


def main():
    p = argparse.ArgumentParser(description="Fetch Smogon gen9monotype stats")
    p.add_argument("--month", required=True, help="YYYY-MM, e.g. 2026-04")
    p.add_argument("--elos", nargs="+", type=int, default=ELOS)
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--out-root", default=None,
                   help="defaults to monotype/smogon_stats/<month>/ next to this script")
    args = p.parse_args()

    here = Path(__file__).parent
    out_root = Path(args.out_root) if args.out_root else here / args.month

    tasks = build_tasks(args.month, args.elos, out_root)
    print(f"=== fetching {len(tasks)} files for {args.month} -> {out_root} ===")
    t0 = time.time()
    ok = skip = err = 0
    total_bytes = 0
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(fetch, url, dest): (url, dest) for url, dest in tasks}
        for fut in as_completed(futs):
            url, status, nbytes = fut.result()
            if status == "ok":
                ok += 1
                total_bytes += nbytes
            elif status == "skip":
                skip += 1
            else:
                err += 1
                print(f"  ! {status}  {url}", file=sys.stderr)
    elapsed = time.time() - t0
    print(f"\ndone in {elapsed:.1f}s: {ok} fetched, {skip} skipped, {err} errors, "
          f"{total_bytes/1024:.0f} KB")
    if err:
        sys.exit(1)


if __name__ == "__main__":
    main()
