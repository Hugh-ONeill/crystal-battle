#!/usr/bin/env python3
"""
V2 move-prediction extractor — same as v1 but adds dynamic state context
(boosts / status / weather / terrain / hazards / screens) per turn.

Same retention rate as v1; new dataset key:
  state_features   (N, STATE_DIM=53) — actor-relative state context

The replay-log walker handles: |switch|/|drag|, |-damage|/|-heal|/|-sethp|,
|-status|/|-curestatus|, |-boost|/|-unboost|/|-clearboost|/|-clearallboost|,
|-weather|, |-fieldstart|/|-fieldend|, |-sidestart|/|-sideend|.

Usage:
  .venv/bin/python monotype/extract_move_training_data_v2.py \\
      --replays-dir showdown/replays/gen9monotype \\
      --out monotype/move_train_data_v2.npz
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
from monotype.extract_lead_training_data import roster_to_paste
from monotype.extract_move_training_data import (
    TYPE_NAME_TO_IDX, extract_moves_from_block, _norm_move_id,
)
from monotype.featurizer_lead_preview import featurize_team_from_state_side
from monotype.featurizer_move_state import (
    BOOST_STATS, STATE_DIM, TurnState, featurize_turn_state,
    normalize_status, normalize_terrain, normalize_weather,
)
from showdown.local_battle import build_pe_state_gen9


HERE = Path(__file__).parent
SPECIES_TYPES_PATH = HERE.parent / "showdown" / "species_types.json"


class _ReplayStateTracker:
    """Maintains live battle state derived from a Showdown log."""

    def __init__(self):
        self.active = {"p1": None, "p2": None}
        self.hp = {"p1": 1.0, "p2": 1.0}
        self.boosts = {
            "p1": {s: 0 for s in BOOST_STATS},
            "p2": {s: 0 for s in BOOST_STATS},
        }
        # Status tracked per (side, species) so a returning mon keeps its status
        self.status_per_mon: dict[tuple[str, str], str] = {}
        self.weather = "none"
        self.terrain = "none"
        self.trick_room = False
        self.tailwind = {"p1": False, "p2": False}
        self.hazards = {
            "p1": {"sr": False, "spikes": 0, "tspikes": 0, "web": False},
            "p2": {"sr": False, "spikes": 0, "tspikes": 0, "web": False},
        }
        self.screens = {
            "p1": {"reflect": False, "lightscreen": False, "auroraveil": False},
            "p2": {"reflect": False, "lightscreen": False, "auroraveil": False},
        }

    def snapshot(self) -> TurnState:
        ts = TurnState()
        ts.boosts_p1 = dict(self.boosts["p1"])
        ts.boosts_p2 = dict(self.boosts["p2"])
        ts.status_p1 = self.status_per_mon.get(("p1", self.active["p1"]), "none")
        ts.status_p2 = self.status_per_mon.get(("p2", self.active["p2"]), "none")
        ts.weather = self.weather
        ts.terrain = self.terrain
        ts.trick_room = self.trick_room
        ts.tailwind_p1 = self.tailwind["p1"]
        ts.tailwind_p2 = self.tailwind["p2"]
        for s in ("p1", "p2"):
            h = self.hazards[s]
            sc = self.screens[s]
            if s == "p1":
                ts.sr_p1, ts.spikes_p1, ts.tspikes_p1, ts.web_p1 = h["sr"], h["spikes"], h["tspikes"], h["web"]
                ts.reflect_p1, ts.lightscreen_p1, ts.auroraveil_p1 = sc["reflect"], sc["lightscreen"], sc["auroraveil"]
            else:
                ts.sr_p2, ts.spikes_p2, ts.tspikes_p2, ts.web_p2 = h["sr"], h["spikes"], h["tspikes"], h["web"]
                ts.reflect_p2, ts.lightscreen_p2, ts.auroraveil_p2 = sc["reflect"], sc["lightscreen"], sc["auroraveil"]
        return ts

    @staticmethod
    def _side_of(slot: str) -> str | None:
        """'p1a: Nick' -> 'p1'; '|p1: PlayerName' -> 'p1'."""
        slot = slot.strip()
        if slot.startswith("p1"):
            return "p1"
        if slot.startswith("p2"):
            return "p2"
        return None

    @staticmethod
    def _hp_frac(field: str) -> float | None:
        parts = field.split()[0].split("/")
        if len(parts) != 2:
            return None
        try:
            return int(parts[0]) / max(1, int(parts[1]))
        except ValueError:
            return None

    def handle_line(self, parts: list[str]):
        if len(parts) < 2:
            return
        cmd = parts[1]

        if cmd in ("switch", "drag") and len(parts) >= 5:
            side = self._side_of(parts[2])
            if side is None:
                return
            species = parts[3].split(",")[0].strip()
            self.active[side] = species
            f = self._hp_frac(parts[4])
            if f is not None:
                self.hp[side] = f
            # Switching wipes the incoming mon's boosts (treated as fresh)
            for s in BOOST_STATS:
                self.boosts[side][s] = 0

        elif cmd in ("-damage", "-heal", "-sethp") and len(parts) >= 4:
            side = self._side_of(parts[2])
            if side is None:
                return
            f = self._hp_frac(parts[3])
            if f is not None:
                self.hp[side] = f

        elif cmd == "-status" and len(parts) >= 4:
            side = self._side_of(parts[2])
            if side is None or self.active[side] is None:
                return
            self.status_per_mon[(side, self.active[side])] = normalize_status(parts[3])

        elif cmd in ("-curestatus", "-end") and len(parts) >= 3:
            side = self._side_of(parts[2])
            if side is None or self.active[side] is None:
                return
            # |-end| can also signal volatile-end events; ignore unless status-like
            if cmd == "-end":
                return
            self.status_per_mon.pop((side, self.active[side]), None)

        elif cmd in ("-boost", "-unboost") and len(parts) >= 5:
            side = self._side_of(parts[2])
            if side is None:
                return
            stat = parts[3].strip().lower()
            try:
                amount = int(parts[4])
            except ValueError:
                return
            if cmd == "-unboost":
                amount = -amount
            if stat in self.boosts[side]:
                cur = self.boosts[side][stat]
                self.boosts[side][stat] = max(-6, min(6, cur + amount))

        elif cmd == "-setboost" and len(parts) >= 5:
            side = self._side_of(parts[2])
            if side is None:
                return
            stat = parts[3].strip().lower()
            try:
                amount = int(parts[4])
            except ValueError:
                return
            if stat in self.boosts[side]:
                self.boosts[side][stat] = max(-6, min(6, amount))

        elif cmd == "-clearboost" and len(parts) >= 3:
            side = self._side_of(parts[2])
            if side is None:
                return
            for s in BOOST_STATS:
                self.boosts[side][s] = 0

        elif cmd == "-clearallboost":
            for side in ("p1", "p2"):
                for s in BOOST_STATS:
                    self.boosts[side][s] = 0

        elif cmd == "-weather" and len(parts) >= 3:
            # "[upkeep]" lines keep weather the same; "none" ends it
            if "[upkeep]" in "|".join(parts):
                return
            name = parts[2].strip()
            self.weather = normalize_weather(name) if name.lower() != "none" else "none"

        elif cmd == "-fieldstart" and len(parts) >= 3:
            payload = parts[2].lower()
            if "trick room" in payload or "trickroom" in payload:
                self.trick_room = True
            else:
                t = normalize_terrain(payload.replace("move:", "").replace("ability:", "").strip())
                if t != "none":
                    self.terrain = t

        elif cmd == "-fieldend" and len(parts) >= 3:
            payload = parts[2].lower()
            if "trick room" in payload or "trickroom" in payload:
                self.trick_room = False
            else:
                # any terrain end resets to none (we only track one at a time)
                self.terrain = "none"

        elif cmd == "-sidestart" and len(parts) >= 4:
            side = self._side_of(parts[2])
            payload = parts[3].lower()
            if side is None:
                return
            payload = payload.replace("move:", "").strip()
            if "stealth rock" in payload or "stealthrock" in payload:
                self.hazards[side]["sr"] = True
            elif "spikes" in payload and "toxic" not in payload:
                self.hazards[side]["spikes"] = min(3, self.hazards[side]["spikes"] + 1)
            elif "toxic spikes" in payload or "toxicspikes" in payload:
                self.hazards[side]["tspikes"] = min(2, self.hazards[side]["tspikes"] + 1)
            elif "sticky web" in payload or "stickyweb" in payload:
                self.hazards[side]["web"] = True
            elif "reflect" in payload:
                self.screens[side]["reflect"] = True
            elif "light screen" in payload or "lightscreen" in payload:
                self.screens[side]["lightscreen"] = True
            elif "aurora veil" in payload or "auroraveil" in payload:
                self.screens[side]["auroraveil"] = True
            elif "tailwind" in payload:
                self.tailwind[side] = True

        elif cmd == "-sideend" and len(parts) >= 4:
            side = self._side_of(parts[2])
            payload = parts[3].lower()
            if side is None:
                return
            payload = payload.replace("move:", "").strip()
            if "stealth rock" in payload or "stealthrock" in payload:
                self.hazards[side]["sr"] = False
            elif "spikes" in payload and "toxic" not in payload:
                self.hazards[side]["spikes"] = 0
            elif "toxic spikes" in payload or "toxicspikes" in payload:
                self.hazards[side]["tspikes"] = 0
            elif "sticky web" in payload or "stickyweb" in payload:
                self.hazards[side]["web"] = False
            elif "reflect" in payload:
                self.screens[side]["reflect"] = False
            elif "light screen" in payload or "lightscreen" in payload:
                self.screens[side]["lightscreen"] = False
            elif "aurora veil" in payload or "auroraveil" in payload:
                self.screens[side]["auroraveil"] = False
            elif "tailwind" in payload:
                self.tailwind[side] = False


def parse_replay_for_move_events(data: dict) -> dict | None:
    """Walk a replay log emitting one record per |move| event."""
    log = data.get("log", "")
    if not log:
        return None

    p1_team: list[str] = []
    p2_team: list[str] = []
    tracker = _ReplayStateTracker()
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

        elif cmd == "move" and len(parts) >= 4:
            slot = parts[2]
            actor = slot.split("a")[0] if "a:" in slot else slot.split(":")[0].strip()
            if actor not in ("p1", "p2"):
                continue
            opp = "p2" if actor == "p1" else "p1"
            if tracker.active[actor] is None or tracker.active[opp] is None:
                continue
            events.append({
                "actor": actor,
                "actor_species": tracker.active[actor],
                "opp_species": tracker.active[opp],
                "move": _norm_move_id(parts[3].strip()),
                "hp_actor": tracker.hp[actor],
                "hp_opp": tracker.hp[opp],
                "state_features": featurize_turn_state(tracker.snapshot(), actor),
            })
            tracker.handle_line(parts)
        else:
            tracker.handle_line(parts)

    if len(p1_team) != 6 or len(p2_team) != 6 or not events:
        return None
    return {"p1_team": p1_team, "p2_team": p2_team, "events": events}


def detect_type(team: list[str], species_types: dict) -> str | None:
    sets = []
    for sp in team:
        t = species_types.get(sp)
        if not t:
            t = species_types.get(sp.split("-")[0])
        if not t:
            return None
        sets.append(set(x.lower() for x in t))
    common = set.intersection(*sets)
    if not common:
        return None
    return sorted(common)[0] if len(common) > 1 else next(iter(common))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--replays-dir", type=Path,
                    default=Path("showdown/replays/gen9monotype"))
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    print(f"=== loading canonical sets ===")
    cs = build_canonical_sets()
    canonical_moves: dict[str, dict[str, list[str]]] = {
        t: {sp: extract_moves_from_block(b) for sp, b in sps.items()}
        for t, sps in cs.items()
    }
    species_types = json.load(open(SPECIES_TYPES_PATH))

    files = sorted(args.replays_dir.glob("*.json"))
    if args.limit:
        files = files[:args.limit]
    print(f"=== processing {len(files)} replays ===")

    actor_feats_list, opp_feats_list = [], []
    state_feats_list = []
    cand_moves_list = []
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

        p1_type = detect_type(rec["p1_team"], species_types)
        p2_type = detect_type(rec["p2_team"], species_types)
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

            ct = canonical_moves.get(actor_type, {})
            moves = ct.get(ev["actor_species"])
            if not moves:
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
            state_feats_list.append(ev["state_features"])
            cand_moves_list.append(moves)
            hp_actor_list.append(ev["hp_actor"])
            hp_opp_list.append(ev["hp_opp"])
            tta_list.append(TYPE_NAME_TO_IDX.get(actor_type, -1))
            tto_list.append(TYPE_NAME_TO_IDX.get(opp_type, -1))
            y_list.append(move_idx)

        if (k + 1) % 500 == 0:
            print(f"  [{k+1}/{len(files)}] events kept {len(y_list)}, "
                  f"skipped {sum(skipped.values())}", flush=True)

    print(f"\n=== extracted {len(y_list)} events from {len(files)} replays ===")
    print(f"  skipped: {skipped}")

    all_moves: set[str] = set()
    for ms in cand_moves_list:
        all_moves.update(ms)
    vocab = sorted(all_moves)
    move_to_idx = {m: i for i, m in enumerate(vocab)}

    cand_idx_arr = np.zeros((len(cand_moves_list), 4), dtype=np.int32)
    for i, ms in enumerate(cand_moves_list):
        for j in range(4):
            cand_idx_arr[i, j] = move_to_idx[ms[j]] if j < len(ms) else -1

    np.savez_compressed(
        args.out,
        actor_features=np.stack(actor_feats_list).astype(np.float32),
        opp_features=np.stack(opp_feats_list).astype(np.float32),
        state_features=np.stack(state_feats_list).astype(np.float32),
        candidate_moves=cand_idx_arr,
        hp_actor=np.array(hp_actor_list, dtype=np.float32),
        hp_opp=np.array(hp_opp_list, dtype=np.float32),
        team_type_actor=np.array(tta_list, dtype=np.int64),
        team_type_opp=np.array(tto_list, dtype=np.int64),
        y=np.array(y_list, dtype=np.int64),
        move_vocab=np.array(vocab, dtype=object),
    )
    print(f"  vocab size: {len(vocab)}")
    print(f"  state_features shape: ({len(state_feats_list)}, {STATE_DIM})")
    print(f"=== saved to {args.out} ===")


if __name__ == "__main__":
    main()
