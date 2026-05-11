#!/usr/bin/env python3
"""
Parse a Showdown gen9 replay JSON into a structured trajectory:
  - winner: "p1" | "p2" | "tie" | None
  - per turn: {"p1_action": ..., "p2_action": ...} where action is
    {"type": "move"|"switch", "name": <move-or-species>}
  - revealed team specs per side (best-effort, for state reconstruction later)

This MVP focuses on extracting the move/switch *choices* per turn. State
reconstruction (building a poke_engine State string) will plug in on top.

Usage:
  .venv/bin/python showdown/replay_parse_gen9.py <replay.json>
"""

from __future__ import annotations

import json
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class PokemonReveal:
    species: str
    nickname: str | None = None
    moves: set[str] = field(default_factory=set)
    item: str | None = None
    ability: str | None = None
    tera_type: str | None = None
    terastallized: bool = False


@dataclass
class ReplayTrajectory:
    replay_id: str
    format: str
    p1_name: str
    p2_name: str
    p1_rating: int | None
    p2_rating: int | None
    winner: str | None  # "p1" | "p2" | "tie" | None
    p1_team: dict[str, PokemonReveal]  # nickname -> reveal
    p2_team: dict[str, PokemonReveal]
    turns: list[dict]  # [{"turn": 1, "p1_action": ..., "p2_action": ...}, ...]
    aborted: bool = False
    p1_lead: str | None = None  # species each side led with at battle start
    p2_lead: str | None = None


def _norm(name: str) -> str:
    return re.sub(r"[^a-z0-9]", "", name.lower())


def _parse_active_id(s: str) -> tuple[str, str] | None:
    """'p1a: Kingambit' -> ('p1', 'Kingambit'). 'p1a' / 'p2a' indicates active slot."""
    m = re.match(r"(p[12])a:\s*(.+)", s.strip())
    if not m:
        return None
    return m.group(1), m.group(2)


def parse_replay(data: dict) -> ReplayTrajectory:
    """Parse a Showdown replay JSON into a structured trajectory."""
    log = data.get("log", "")
    lines = log.split("\n")

    p1_name = p2_name = ""
    p1_rating = p2_rating = None
    p1_team: dict[str, PokemonReveal] = {}
    p2_team: dict[str, PokemonReveal] = {}
    # active nickname -> side ('p1' or 'p2'); used to attribute moves
    active_p1: str | None = None
    active_p2: str | None = None
    winner = None
    turns: list[dict] = []
    cur_turn: dict | None = None
    # Track whether next switch on each side is "forced" (post-faint).
    p1_force_switch = False
    p2_force_switch = False
    lead_p1: str | None = None
    lead_p2: str | None = None

    def begin_turn(n: int):
        nonlocal cur_turn
        if cur_turn is not None:
            turns.append(cur_turn)
        cur_turn = {"turn": n, "p1_action": None, "p2_action": None,
                    "p1_terad": False, "p2_terad": False,
                    "p1_pivot": None, "p2_pivot": None}

    def attribute_action(side: str, action: dict):
        if cur_turn is None:
            return
        key = "p1_action" if side == "p1" else "p2_action"
        # Only fill if not already set — first voluntary action of the turn wins.
        if cur_turn[key] is None:
            cur_turn[key] = action

    def get_or_create(side: str, nickname: str, species: str) -> PokemonReveal:
        team = p1_team if side == "p1" else p2_team
        # Replays often reveal a mon as "Species, M" or "Species" — normalize.
        species_clean = species.split(",")[0].strip()
        rev = team.get(nickname)
        if rev is None:
            rev = PokemonReveal(species=species_clean, nickname=nickname)
            team[nickname] = rev
        elif not rev.species:
            rev.species = species_clean
        return rev

    for raw in lines:
        line = raw.strip()
        if not line or not line.startswith("|"):
            continue
        parts = line.split("|")
        # parts[0] is empty, parts[1] is the event name
        if len(parts) < 2:
            continue
        event = parts[1]

        if event == "player" and len(parts) >= 5:
            slot = parts[2]
            name = parts[3]
            try:
                rating = int(parts[5]) if len(parts) > 5 and parts[5] else None
            except ValueError:
                rating = None
            if slot == "p1":
                p1_name, p1_rating = name, rating
            elif slot == "p2":
                p2_name, p2_rating = name, rating

        elif event == "poke" and len(parts) >= 4:
            # Team preview: |poke|p1|Species, M|<details>
            slot = parts[2]
            species_part = parts[3]
            species_clean = species_part.split(",")[0].strip()
            # Use species as placeholder nickname until we see the real one
            # via |switch|. The real nickname links in then.
            rev = PokemonReveal(species=species_clean, nickname=species_clean)
            team = p1_team if slot == "p1" else p2_team
            # Avoid duplicate species (e.g. multi-form). Use unique key.
            key = species_clean
            i = 0
            while key in team:
                i += 1
                key = f"{species_clean}_{i}"
            team[key] = rev

        elif event == "turn" and len(parts) >= 3:
            try:
                begin_turn(int(parts[2]))
            except ValueError:
                pass

        elif event == "switch" and len(parts) >= 4:
            ident = _parse_active_id(parts[2])
            if not ident:
                continue
            side, nickname = ident
            species = parts[3]
            rev = get_or_create(side, nickname, species)
            if side == "p1":
                if not p1_force_switch and cur_turn is not None and cur_turn["p1_action"] is None:
                    # First voluntary action of the turn.
                    cur_turn["p1_action"] = {"type": "switch", "name": rev.species}
                elif cur_turn is not None:
                    # Already moved this turn (pivot like U-turn) or post-faint.
                    cur_turn["p1_pivot"] = rev.species
                p1_force_switch = False
                active_p1 = nickname
                if lead_p1 is None:
                    lead_p1 = rev.species
            else:
                if not p2_force_switch and cur_turn is not None and cur_turn["p2_action"] is None:
                    cur_turn["p2_action"] = {"type": "switch", "name": rev.species}
                elif cur_turn is not None:
                    cur_turn["p2_pivot"] = rev.species
                p2_force_switch = False
                active_p2 = nickname
                if lead_p2 is None:
                    lead_p2 = rev.species

        elif event == "drag" and len(parts) >= 4:
            # Whirlwind / Dragon Tail / Roar — opponent forced our switch.
            # Not a voluntary action; just track active.
            ident = _parse_active_id(parts[2])
            if not ident:
                continue
            side, nickname = ident
            species = parts[3]
            get_or_create(side, nickname, species)
            if side == "p1":
                active_p1 = nickname
            else:
                active_p2 = nickname

        elif event == "move" and len(parts) >= 4:
            # |move|p1a: Kingambit|Sucker Punch|p2a: ...
            ident = _parse_active_id(parts[2])
            if not ident:
                continue
            side, nickname = ident
            move_name = parts[3]
            # Record move in the user's revealed moveset
            team = p1_team if side == "p1" else p2_team
            rev = team.get(nickname)
            if rev is None:
                # First time seeing this mon — sometimes happens before |switch|
                # in unusual logs. Create a placeholder.
                rev = PokemonReveal(species=nickname, nickname=nickname)
                team[nickname] = rev
            rev.moves.add(_norm(move_name))
            attribute_action(side, {"type": "move", "name": move_name})

        elif event == "faint" and len(parts) >= 3:
            ident = _parse_active_id(parts[2])
            if not ident:
                continue
            side = ident[0]
            # Next switch on this side will be a forced (post-faint) switch.
            if side == "p1":
                p1_force_switch = True
            else:
                p2_force_switch = True

        elif event == "-item" and len(parts) >= 4:
            ident = _parse_active_id(parts[2])
            if ident:
                side, nickname = ident
                team = p1_team if side == "p1" else p2_team
                rev = team.get(nickname)
                if rev:
                    rev.item = parts[3]

        elif event == "-enditem" and len(parts) >= 4:
            # Fires when an item is *consumed* (Berry, Air Balloon, Booster
            # Energy, Focus Sash) or knocked off — reveals the item name even
            # when no |-item| event ever set it.
            ident = _parse_active_id(parts[2])
            if ident:
                side, nickname = ident
                team = p1_team if side == "p1" else p2_team
                rev = team.get(nickname)
                if rev and not rev.item:
                    rev.item = parts[3]

        elif event == "-ability" and len(parts) >= 4:
            ident = _parse_active_id(parts[2])
            if ident:
                side, nickname = ident
                team = p1_team if side == "p1" else p2_team
                rev = team.get(nickname)
                if rev and not rev.ability:
                    rev.ability = parts[3]

        elif event == "-terastallize" and len(parts) >= 4:
            ident = _parse_active_id(parts[2])
            if ident:
                side, nickname = ident
                team = p1_team if side == "p1" else p2_team
                rev = team.get(nickname)
                if rev:
                    rev.terastallized = True
                    rev.tera_type = parts[3]
                if cur_turn is not None:
                    cur_turn[f"{side}_terad"] = True

        elif event == "win" and len(parts) >= 3:
            wname = parts[2]
            if wname == p1_name:
                winner = "p1"
            elif wname == p2_name:
                winner = "p2"

        elif event == "tie":
            winner = "tie"

    if cur_turn is not None:
        turns.append(cur_turn)

    aborted = winner is None or len(turns) < 2

    return ReplayTrajectory(
        replay_id=data.get("id", ""),
        format=data.get("format", ""),
        p1_name=p1_name,
        p2_name=p2_name,
        p1_rating=p1_rating,
        p2_rating=p2_rating,
        winner=winner,
        p1_team=p1_team,
        p2_team=p2_team,
        turns=turns,
        aborted=aborted,
        p1_lead=lead_p1,
        p2_lead=lead_p2,
    )


def _summarize(traj: ReplayTrajectory) -> None:
    print(f"replay {traj.replay_id} ({traj.format})")
    print(f"  p1: {traj.p1_name} ({traj.p1_rating}) vs p2: {traj.p2_name} ({traj.p2_rating})")
    print(f"  winner: {traj.winner}, aborted: {traj.aborted}")
    print(f"  turns: {len(traj.turns)}")
    print(f"  p1 team revealed: {[(k, len(v.moves), v.item, v.ability) for k, v in traj.p1_team.items()]}")
    print(f"  p2 team revealed: {[(k, len(v.moves), v.item, v.ability) for k, v in traj.p2_team.items()]}")
    # Show first 3 turns
    for t in traj.turns[:3]:
        print(f"    T{t['turn']}: p1={t['p1_action']} p2={t['p2_action']}")
    if len(traj.turns) > 3:
        print(f"    ... ({len(traj.turns) - 3} more turns) ...")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(f"usage: {sys.argv[0]} <replay.json> [<replay.json> ...]")
        sys.exit(1)
    for p in sys.argv[1:]:
        with open(p) as f:
            data = json.load(f)
        traj = parse_replay(data)
        _summarize(traj)
        print()
