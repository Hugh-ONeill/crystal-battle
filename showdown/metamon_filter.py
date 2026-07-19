#!/usr/bin/env python3
"""
Filter the metamon-raw-replays parquet shards down to a gen9ou JSONL that
build_replay_sets.py / replay_to_training_gen9.py can consume.

Row schema matches Showdown replay JSON: {"id", "formatid", "rating", "log"}.
Defaults: gen9ou, uploaded since 2025-07-01 (meta drift — gen9ou spans
2022-2026 in the dump), numeric rating >= 1300 (~352k replays of the 2.04M).

Usage:
  .venv/bin/python showdown/metamon_filter.py \
      --shards ~/Developer/grimoire/metamon-data/raw-replays/data \
      --out ~/Developer/grimoire/metamon-data/gen9ou_recent_1300.jsonl
"""

from __future__ import annotations

import argparse
import datetime
import glob
import json
import os
from pathlib import Path

import pyarrow.parquet as pq


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shards", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--format", default="gen9ou")
    ap.add_argument("--since", default="2025-07-01")
    ap.add_argument("--min-rating", type=int, default=1300)
    args = ap.parse_args()

    y, m, d = map(int, args.since.split("-"))
    cut = int(datetime.datetime(y, m, d,
                                tzinfo=datetime.timezone.utc).timestamp())

    paths = sorted(glob.glob(os.path.join(
        os.path.expanduser(args.shards), "*.parquet")))
    kept = seen = 0
    with open(os.path.expanduser(args.out), "w") as out:
        for p in paths:
            t = pq.ParquetFile(p).read(
                columns=["id", "formatid", "rating", "uploadtime", "log"])
            ids = t.column("id").to_pylist()
            fmts = t.column("formatid").to_pylist()
            rats = t.column("rating").to_pylist()
            uts = t.column("uploadtime").to_pylist()
            logs = t.column("log").to_pylist()
            for i in range(len(ids)):
                seen += 1
                if fmts[i] != args.format or uts[i] < cut:
                    continue
                r = rats[i]
                if not r or r == "None" or int(r) < args.min_rating:
                    continue
                out.write(json.dumps({
                    "id": ids[i], "formatid": fmts[i],
                    "rating": int(r), "log": logs[i]}) + "\n")
                kept += 1
            print(f"  {Path(p).name}: kept so far {kept}", flush=True)
    print(f"done: kept {kept}/{seen} -> {args.out} "
          f"({os.path.getsize(os.path.expanduser(args.out))/1e9:.2f} GB)")


if __name__ == "__main__":
    main()
