#!/usr/bin/env python3
"""
Build supervised training data for the lead-pick net from gen9monotype replays.

For each parseable replay produces one labeled example:
  X_p1     (6, MON_DIM) features of P1's team
  X_p2     (6, MON_DIM) features of P2's team
  y_p1     int — index of P1's chosen lead (0..5 in P1's team order)
  y_p2     int — index of P2's chosen lead (0..5 in P2's team order)
  winner   int — 1 if P1 won, 2 if P2 won

Team pastes are assembled from `canonical_sets` lookups keyed on
(team-type, species). Replays with missing winners, unparseable rosters,
or species absent from canonical sets are skipped (logged to stderr).

Usage:
  .venv/bin/python monotype/extract_lead_training_data.py \\
      --replays-dir showdown/replays/gen9monotype \\
      --out monotype/lead_train_data.npz
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from monotype.canonical_sets import build_canonical_sets
from monotype.featurizer_lead_preview import (
    MON_DIM, featurize_team_from_state_side,
)
from showdown.local_battle import build_pe_state_gen9

HERE = Path(__file__).parent
SPECIES_TYPES_PATH = HERE.parent / "showdown" / "species_types.json"


def load_species_types() -> dict[str, list[str]]:
    return json.load(open(SPECIES_TYPES_PATH))


def detect_team_type(team: list[str], species_types: dict[str, list[str]]) -> str | None:
    """Return the single type shared by all 6 mons, or None."""
    if not team:
        return None
    type_sets = []
    for sp in team:
        types = species_types.get(sp)
        if not types:
            base = sp.split("-")[0]
            types = species_types.get(base)
        if not types:
            return None
        type_sets.append(set(types))
    common = set.intersection(*type_sets)
    if not common:
        return None
    if len(common) == 1:
        return next(iter(common))
    return sorted(common)[0]


def parse_replay(data: dict) -> dict | None:
    """Extract roster + leads + winner from one replay JSON."""
    log = data.get("log", "")
    if not log:
        return None

    p1_team: list[str] = []
    p2_team: list[str] = []
    p1_lead: str | None = None
    p2_lead: str | None = None
    p1_name: str | None = None
    p2_name: str | None = None
    winner: str | None = None

    for line in log.split("\n"):
        if not line.startswith("|"):
            continue
        parts = line.split("|")
        if len(parts) < 2:
            continue
        cmd = parts[1]
        if cmd == "player" and len(parts) >= 4:
            side = parts[2]
            name = parts[3]
            if side == "p1":
                p1_name = name
            elif side == "p2":
                p2_name = name
        elif cmd == "poke" and len(parts) >= 4:
            side = parts[2]
            species = parts[3].split(",")[0].strip()
            if side == "p1":
                p1_team.append(species)
            elif side == "p2":
                p2_team.append(species)
        elif cmd in ("switch", "drag") and len(parts) >= 4:
            slot = parts[2].split(":")[0].strip()
            species = parts[3].split(",")[0].strip()
            if slot == "p1a" and p1_lead is None:
                p1_lead = species
            elif slot == "p2a" and p2_lead is None:
                p2_lead = species
        elif cmd == "win" and len(parts) >= 3:
            winner = parts[2].strip()

    if (len(p1_team) != 6 or len(p2_team) != 6
            or not p1_lead or not p2_lead or not winner):
        return None

    # Map winner name -> side
    if winner == p1_name:
        winner_side = 1
    elif winner == p2_name:
        winner_side = 2
    else:
        return None

    # Lead must appear in roster
    if p1_lead not in p1_team or p2_lead not in p2_team:
        return None

    return {
        "p1_team": p1_team, "p2_team": p2_team,
        "p1_lead_idx": p1_team.index(p1_lead),
        "p2_lead_idx": p2_team.index(p2_lead),
        "winner": winner_side,
    }


def roster_to_paste(roster: list[str], team_type: str,
                    canonical_sets: dict[str, dict[str, str]]) -> str | None:
    """Assemble a 6-mon paste body using canonical sets for the given type.
    Returns None if any species is missing from the canonical set."""
    blocks = []
    type_key = team_type.lower()
    cs_type = canonical_sets.get(type_key, {})
    for sp in roster:
        block = cs_type.get(sp)
        if not block:
            # try base-form match (e.g. Urshifu vs Urshifu-Rapid-Strike)
            base = sp.split("-")[0]
            for k, v in cs_type.items():
                if k == base or k.split("-")[0] == base:
                    block = v
                    break
        if not block:
            return None
        blocks.append(block)
    return "\n\n".join(blocks)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--replays-dir", type=Path,
                    default=Path("showdown/replays/gen9monotype"))
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--limit", type=int, default=None,
                    help="cap on number of replays processed (debug)")
    args = ap.parse_args()

    print(f"=== loading canonical sets ===")
    cs = build_canonical_sets()
    total_species = sum(len(v) for v in cs.values())
    print(f"  {len(cs)} types, {total_species} (type, species) entries")

    species_types = load_species_types()

    files = sorted(args.replays_dir.glob("*.json"))
    if args.limit:
        files = files[:args.limit]
    print(f"=== processing {len(files)} replays ===")

    X1_list, X2_list = [], []
    y1_list, y2_list = [], []
    winner_list = []
    skipped = {"unparseable": 0, "type_detect": 0, "canonical_miss": 0,
               "featurize_fail": 0}

    for k, fp in enumerate(files):
        try:
            data = json.load(open(fp))
        except Exception:
            skipped["unparseable"] += 1
            continue
        rec = parse_replay(data)
        if rec is None:
            skipped["unparseable"] += 1
            continue
        p1_type = detect_team_type(rec["p1_team"], species_types)
        p2_type = detect_team_type(rec["p2_team"], species_types)
        if not p1_type or not p2_type:
            skipped["type_detect"] += 1
            continue
        p1_paste = roster_to_paste(rec["p1_team"], p1_type, cs)
        p2_paste = roster_to_paste(rec["p2_team"], p2_type, cs)
        if p1_paste is None or p2_paste is None:
            skipped["canonical_miss"] += 1
            continue
        try:
            state = build_pe_state_gen9(p1_paste, p2_paste)
            X1 = featurize_team_from_state_side(state.side_one)
            X2 = featurize_team_from_state_side(state.side_two)
        except Exception:
            skipped["featurize_fail"] += 1
            continue
        X1_list.append(X1)
        X2_list.append(X2)
        y1_list.append(rec["p1_lead_idx"])
        y2_list.append(rec["p2_lead_idx"])
        winner_list.append(rec["winner"])
        if (k + 1) % 500 == 0:
            print(f"  [{k+1}/{len(files)}] kept {len(X1_list)}, "
                  f"skipped {sum(skipped.values())}", flush=True)

    if not X1_list:
        print("ERROR: no examples extracted")
        sys.exit(1)

    X1 = np.stack(X1_list)
    X2 = np.stack(X2_list)
    y1 = np.array(y1_list, dtype=np.int64)
    y2 = np.array(y2_list, dtype=np.int64)
    win = np.array(winner_list, dtype=np.int64)

    np.savez_compressed(args.out, X_p1=X1, X_p2=X2, y_p1=y1, y_p2=y2, winner=win)
    print(f"\n=== saved {len(X1)} examples to {args.out} ===")
    print(f"  shapes: X_p1 {X1.shape}, X_p2 {X2.shape}, y_p1 {y1.shape}")
    print(f"  skipped: {skipped}")


if __name__ == "__main__":
    main()
