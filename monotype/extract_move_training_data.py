#!/usr/bin/env python3
"""
Extract turn-level move-prediction training data from gen9monotype replays.

For each `|move|` event in the replay log we record:
  - the actor side's active species + monotype
  - the opponent active species + monotype
  - the move chosen, mapped to an index into the actor's canonical 4-move set
  - the actor's current HP fraction at the time of the move (light state ftr)
  - the opponent's current HP fraction

Skipped:
  - Events where the actor or opponent active aren't in our canonical sets
  - Moves not in the actor's canonical 4-move set (off-meta sets)
  - Switch events (V1 only predicts moves, not switches)

Output shape (npz):
  actor_features   (N, MON_DIM)        — actor's active features
  opp_features     (N, MON_DIM)        — opp active features
  candidate_moves  (N, 4)              — move-id one-hot indices for actor's 4 moves
  hp_actor         (N,)                — actor HP fraction at action time
  hp_opp           (N,)                — opp HP fraction
  team_type_actor  (N,)                — int 0-17, actor's monotype
  team_type_opp    (N,)                — int 0-17, opp monotype
  y                (N,)                — chosen move index (0..3 into candidate_moves)
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from monotype.canonical_sets import MONOTYPE_TYPES, build_canonical_sets
from monotype.chaos_priors import _detect_side_type, _norm_species
from monotype.featurizer_lead_preview import (
    MON_DIM, TYPE_INDEX, TYPE_ORDER, featurize_team_from_state_side,
)
from showdown.local_battle import build_pe_state_gen9

HERE = Path(__file__).parent
SPECIES_TYPES_PATH = HERE.parent / "showdown" / "species_types.json"

TYPE_NAME_TO_IDX = {t: i for i, t in enumerate(MONOTYPE_TYPES)}


def _norm_move_id(name: str) -> str:
    return re.sub(r"[^a-z0-9]", "", name.lower())


def extract_moves_from_block(block: str) -> list[str]:
    """Extract the 4 move IDs from a canonical paste block (`- Move Name`)."""
    out = []
    for ln in block.splitlines():
        ln = ln.strip()
        if ln.startswith("- "):
            out.append(_norm_move_id(ln[2:]))
    return out[:4]


def parse_replay_for_move_events(data: dict) -> dict | None:
    """Walk a replay log and emit one record per |move| event with the
    minimal state needed to predict the actor's move."""
    log = data.get("log", "")
    if not log:
        return None

    p1_team: list[str] = []
    p2_team: list[str] = []
    # current active species per side (updated on |switch| / |drag|)
    active = {"p1": None, "p2": None}
    # current HP fraction per side (updated on |switch|, |-damage|, |-heal|)
    hp = {"p1": 1.0, "p2": 1.0}

    events: list[dict] = []

    for line in log.split("\n"):
        if not line.startswith("|"):
            continue
        parts = line.split("|")
        if len(parts) < 2:
            continue
        cmd = parts[1]

        if cmd == "poke" and len(parts) >= 4:
            side = parts[2]
            species = parts[3].split(",")[0].strip()
            if side == "p1":
                p1_team.append(species)
            elif side == "p2":
                p2_team.append(species)

        elif cmd in ("switch", "drag") and len(parts) >= 5:
            slot_full = parts[2]
            side = slot_full.split("a")[0]  # "p1a: Nick" -> "p1"
            if side not in ("p1", "p2"):
                continue
            species = parts[3].split(",")[0].strip()
            active[side] = species
            # field 4 is HP like "100/100" or "62/100 brn"
            hp_part = parts[4].split()[0].split("/")
            if len(hp_part) == 2:
                try:
                    hp[side] = int(hp_part[0]) / max(1, int(hp_part[1]))
                except ValueError:
                    pass

        elif cmd in ("-damage", "-heal", "-sethp") and len(parts) >= 4:
            slot_full = parts[2]
            side = slot_full.split("a")[0]
            if side not in ("p1", "p2"):
                continue
            hp_part = parts[3].split()[0].split("/")
            if len(hp_part) == 2:
                try:
                    hp[side] = int(hp_part[0]) / max(1, int(hp_part[1]))
                except ValueError:
                    pass

        elif cmd == "move" and len(parts) >= 4:
            slot_full = parts[2]
            actor = slot_full.split("a")[0]
            if actor not in ("p1", "p2"):
                continue
            opp = "p2" if actor == "p1" else "p1"
            move_name = parts[3].strip()
            if active[actor] is None or active[opp] is None:
                continue
            events.append({
                "actor": actor,
                "actor_species": active[actor],
                "opp_species": active[opp],
                "move": _norm_move_id(move_name),
                "hp_actor": hp[actor],
                "hp_opp": hp[opp],
            })

    if len(p1_team) != 6 or len(p2_team) != 6 or not events:
        return None
    return {
        "p1_team": p1_team, "p2_team": p2_team,
        "events": events,
    }


def load_species_types() -> dict[str, list[str]]:
    return json.load(open(SPECIES_TYPES_PATH))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--replays-dir", type=Path,
                    default=Path("showdown/replays/gen9monotype"))
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    print(f"=== loading canonical sets ===")
    cs = build_canonical_sets()

    species_types = load_species_types()

    # Build moves-per-canonical-block dict: cs[type][species] -> [move_id, ...]
    canonical_moves: dict[str, dict[str, list[str]]] = {}
    for t, sp_blocks in cs.items():
        canonical_moves[t] = {sp: extract_moves_from_block(b)
                              for sp, b in sp_blocks.items()}

    files = sorted(args.replays_dir.glob("*.json"))
    if args.limit:
        files = files[:args.limit]
    print(f"=== processing {len(files)} replays ===")

    actor_feats_list, opp_feats_list = [], []
    cand_moves_list = []  # move-id strings, we'll convert to indices at end
    hp_actor_list, hp_opp_list = [], []
    tta_list, tto_list = [], []
    y_list = []

    skipped = {"unparseable": 0, "type_detect": 0, "canonical_miss": 0,
               "move_not_in_set": 0, "featurize_fail": 0}

    for k, fp in enumerate(files):
        try:
            data = json.load(open(fp))
        except Exception:
            skipped["unparseable"] += 1
            continue
        rec = parse_replay_for_move_events(data)
        if rec is None:
            skipped["unparseable"] += 1
            continue

        # Detect each side's monotype
        # Note: replay |poke| lines use display names ("Goodra-Hisui");
        # _detect_side_type expects normalized ids — so use display-name lookup.
        def detect_from_display(team: list[str]) -> str | None:
            sets = []
            for sp in team:
                t = species_types.get(sp)
                if not t:
                    base = sp.split("-")[0]
                    t = species_types.get(base)
                if not t:
                    return None
                sets.append(set(x.lower() for x in t))
            common = set.intersection(*sets)
            if not common:
                return None
            return sorted(common)[0] if len(common) > 1 else next(iter(common))

        p1_type = detect_from_display(rec["p1_team"])
        p2_type = detect_from_display(rec["p2_team"])
        if not p1_type or not p2_type:
            skipped["type_detect"] += 1
            continue

        # Build paste from canonical sets (for actor/opp featurization
        # via the existing lead-preview featurizer — gives us the same
        # featurization shape as inference time).
        from monotype.extract_lead_training_data import roster_to_paste
        p1_paste = roster_to_paste(rec["p1_team"], p1_type, cs)
        p2_paste = roster_to_paste(rec["p2_team"], p2_type, cs)
        if p1_paste is None or p2_paste is None:
            skipped["canonical_miss"] += 1
            continue

        try:
            state = build_pe_state_gen9(p1_paste, p2_paste)
            p1_feats = featurize_team_from_state_side(state.side_one)
            p2_feats = featurize_team_from_state_side(state.side_two)
        except Exception:
            skipped["featurize_fail"] += 1
            continue

        for ev in rec["events"]:
            actor_team = rec["p1_team"] if ev["actor"] == "p1" else rec["p2_team"]
            opp_team = rec["p2_team"] if ev["actor"] == "p1" else rec["p1_team"]
            actor_type = p1_type if ev["actor"] == "p1" else p2_type
            opp_type = p2_type if ev["actor"] == "p1" else p1_type
            actor_feats_all = p1_feats if ev["actor"] == "p1" else p2_feats
            opp_feats_all = p2_feats if ev["actor"] == "p1" else p1_feats

            # Find actor active in their team
            actor_idx = None
            for i, sp in enumerate(actor_team):
                if sp == ev["actor_species"] or sp.split("-")[0] == ev["actor_species"].split("-")[0]:
                    actor_idx = i
                    break
            opp_idx = None
            for i, sp in enumerate(opp_team):
                if sp == ev["opp_species"] or sp.split("-")[0] == ev["opp_species"].split("-")[0]:
                    opp_idx = i
                    break
            if actor_idx is None or opp_idx is None:
                continue

            # Look up actor's canonical 4-move set
            ct = canonical_moves.get(actor_type, {})
            moves = ct.get(ev["actor_species"])
            if not moves:
                # try base form
                base = ev["actor_species"].split("-")[0]
                for k2, v2 in ct.items():
                    if k2 == base or k2.split("-")[0] == base:
                        moves = v2
                        break
            if not moves or len(moves) < 4:
                continue
            if ev["move"] not in moves:
                skipped["move_not_in_set"] += 1
                continue
            move_idx = moves.index(ev["move"])

            actor_feats_list.append(actor_feats_all[actor_idx])
            opp_feats_list.append(opp_feats_all[opp_idx])
            cand_moves_list.append(moves)
            hp_actor_list.append(ev["hp_actor"])
            hp_opp_list.append(ev["hp_opp"])
            tta_list.append(TYPE_NAME_TO_IDX.get(actor_type, -1))
            tto_list.append(TYPE_NAME_TO_IDX.get(opp_type, -1))
            y_list.append(move_idx)

        if (k + 1) % 500 == 0:
            print(f"  [{k+1}/{len(files)}] events kept {len(y_list)}, "
                  f"skipped {sum(skipped.values())}", flush=True)

    print(f"\n=== extracted {len(y_list)} move events from {len(files)} replays ===")
    print(f"  skipped: {skipped}")

    # Build a global move-id vocabulary
    all_moves: set[str] = set()
    for ms in cand_moves_list:
        all_moves.update(ms)
    vocab = sorted(all_moves)
    move_to_idx = {m: i for i, m in enumerate(vocab)}
    print(f"  move vocabulary size: {len(vocab)}")

    # Convert candidate moves to int indices
    cand_idx_arr = np.zeros((len(cand_moves_list), 4), dtype=np.int32)
    for i, ms in enumerate(cand_moves_list):
        for j in range(4):
            cand_idx_arr[i, j] = move_to_idx[ms[j]] if j < len(ms) else -1

    np.savez_compressed(
        args.out,
        actor_features=np.stack(actor_feats_list).astype(np.float32),
        opp_features=np.stack(opp_feats_list).astype(np.float32),
        candidate_moves=cand_idx_arr,
        hp_actor=np.array(hp_actor_list, dtype=np.float32),
        hp_opp=np.array(hp_opp_list, dtype=np.float32),
        team_type_actor=np.array(tta_list, dtype=np.int64),
        team_type_opp=np.array(tto_list, dtype=np.int64),
        y=np.array(y_list, dtype=np.int64),
        move_vocab=np.array(vocab, dtype=object),
    )
    print(f"=== saved to {args.out} ===")


if __name__ == "__main__":
    main()
