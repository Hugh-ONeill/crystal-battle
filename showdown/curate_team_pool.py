#!/usr/bin/env python3
"""
Curate a gen9ou team-rotation pool from the metamon team slices.

Two sources, two roles:
  competitive  human-made Smogon sample teams — coherent sets, small, some
               stale (pre-Tera-Blast-ban). The safe own-play default.
  hl_05_26     High-Ladder May-2026 teams reconstructed from replays. Large,
               current-legal, but reconstruction can yield strategically
               incoherent sets. Cores REPEAT across the slice, so instance
               count = high-ladder popularity = a battle-tested signal; we
               keep the most-popular legal unique cores.

Every candidate is validated against the local pokemon-showdown checkout
(node showdown/validate_teams.js) — the ruleset moved under these teams
(Tera Blast is now banned in OU), so stale sets MUST be filtered, not trusted.

Output: two flat dirs of Showdown-paste .txt files that ladder_session.sh
rotates over via `shuf`, plus a pool_manifest.json recording provenance and
the exact validator checkout.

Usage:
  .venv/bin/python showdown/curate_team_pool.py \
      --ps ~/Developer/grimoire/pokemon-showdown \
      --teams ~/Developer/grimoire/metamon-data/teams \
      --hl-size 40
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from collections import Counter
from pathlib import Path

HERE = Path(__file__).parent


def _core_key(species: list[str]) -> str:
    return "|".join(sorted(s.lower().replace(" ", "") for s in species))


def _short_hash(s: str) -> str:
    return hashlib.sha1(s.encode()).hexdigest()[:8]


def _lead(text: str) -> str:
    """First mon's species name from a paste (the line before its stats).
    Handles 'Name @ Item', 'Nick (Species) @ Item', 'Name (M) @ Item'."""
    first = text.strip().split("\n", 1)[0]
    name = first.split("@")[0].strip()
    # a trailing (...) is either a gender tag (M/F) or the real species when
    # the mon is nicknamed; (M)/(F) are dropped, anything else is the species
    if name.endswith(")") and "(" in name:
        inside = name[name.rindex("(") + 1:-1].strip()
        name = name[:name.rindex("(")].strip() if inside in ("M", "F") else inside
    return name or "Unknown"


def validate_dir(ps: Path, files: list[Path], fmt: str) -> list[dict]:
    """Run the batch node validator over `files`, return parsed results."""
    if not files:
        return []
    proc = subprocess.run(
        ["node", str(HERE / "validate_teams.js"), fmt],
        input="\n".join(str(f) for f in files),
        capture_output=True, text=True, cwd=str(ps),
    )
    out = []
    for line in proc.stdout.splitlines():
        line = line.strip()
        if line.startswith("{"):
            out.append(json.loads(line))
    return out


def write_pool(dest: Path, entries: list[tuple[Path, list[str]]]) -> None:
    """Write (file, species) entries as NN_lead_hash.txt, clearing dest."""
    dest.mkdir(parents=True, exist_ok=True)
    for old in dest.glob("*.txt"):
        old.unlink()
    for i, (src, species) in enumerate(entries):
        text = src.read_text()
        lead = _lead(text).lower().replace(" ", "").replace("-", "")
        name = f"{i:02d}_{lead}_{_short_hash(_core_key(species))}.txt"
        (dest / name).write_text(text)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ps", required=True, help="pokemon-showdown checkout")
    ap.add_argument("--teams", required=True, help="metamon teams/ dir")
    ap.add_argument("--fmt", default="gen9ou")
    ap.add_argument("--hl-size", type=int, default=40,
                    help="top-N most-popular legal hl cores to keep")
    ap.add_argument("--out", default=str(HERE / "teams"))
    args = ap.parse_args()

    ps = Path(args.ps).expanduser()
    teams = Path(args.teams).expanduser()
    out = Path(args.out)

    ver = subprocess.run(["git", "-C", str(ps), "log", "-1",
                          "--format=%h %ci"], capture_output=True, text=True
                         ).stdout.strip()

    manifest = {"format": args.fmt, "validator_checkout": ver, "sources": {}}

    # --- competitive: keep every legal team, dedup by core ---
    comp_files = sorted((teams / f"competitive_{args.fmt}" / args.fmt)
                        .glob("team_*." + args.fmt + "_team"))
    comp_res = validate_dir(ps, comp_files, args.fmt)
    comp_entries, seen = [], set()
    comp_legal = 0
    for r in comp_res:
        if not r["ok"]:
            continue
        comp_legal += 1
        key = _core_key(r["species"])
        if key in seen:
            continue
        seen.add(key)
        comp_entries.append((Path(r["path"]), r["species"]))
    write_pool(out / "pool_competitive", comp_entries)
    manifest["sources"]["competitive"] = {
        "candidates": len(comp_res), "legal": comp_legal,
        "unique_cores_kept": len(comp_entries),
        "role": "own-play default (human sets)"}

    # --- hl: validate all, rank legal cores by popularity, keep top-N ---
    hl_files = sorted((teams / f"hl_05_26_{args.fmt}" / args.fmt)
                      .glob("team_*." + args.fmt + "_team"))
    hl_res = validate_dir(ps, hl_files, args.fmt)
    core_count: Counter = Counter()
    core_rep: dict[str, tuple[Path, list[str]]] = {}
    hl_legal = 0
    for r in hl_res:
        if not r["ok"]:
            continue
        hl_legal += 1
        key = _core_key(r["species"])
        core_count[key] += 1
        core_rep.setdefault(key, (Path(r["path"]), r["species"]))
    top = core_count.most_common(args.hl_size)
    hl_entries = [core_rep[k] for k, _ in top]
    write_pool(out / "pool_hl", hl_entries)
    manifest["sources"]["hl_05_26"] = {
        "candidates": len(hl_res), "legal": hl_legal,
        "unique_legal_cores": len(core_count),
        "kept_top_by_popularity": len(hl_entries),
        "top_core_instances": [c for _, c in top[:5]],
        "role": "large current-meta rotation (reconstructed, popularity-ranked)"}

    (out / "pool_manifest.json").write_text(json.dumps(manifest, indent=2))

    # --- summary ---
    print(f"validator: {ver}")
    print(f"competitive: {comp_legal}/{len(comp_res)} legal -> "
          f"{len(comp_entries)} unique cores  (pool_competitive)")
    print(f"hl_05_26:    {hl_legal}/{len(hl_res)} legal, "
          f"{len(core_count)} unique cores -> top {len(hl_entries)} by "
          f"popularity  (pool_hl)")
    if top:
        print(f"  most-popular legal core seen {top[0][1]}x; "
              f"#{len(top)} seen {top[-1][1]}x")
    leads = Counter(_lead((out / 'pool_hl' / f.name).read_text())
                    for f in sorted((out / 'pool_hl').glob('*.txt')))
    print("  hl pool lead spread: " +
          ", ".join(f"{k}:{v}" for k, v in leads.most_common(8)))
    print(f"manifest: {out / 'pool_manifest.json'}")


if __name__ == "__main__":
    main()
