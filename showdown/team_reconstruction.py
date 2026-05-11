#!/usr/bin/env python3
"""Reconstruct a Showdown-format team paste from a replay's revealed info.

Replay logs reveal partial info (species, some moves, sometimes item/ability/tera).
We fill the gaps using Smogon chaos statistics — top-frequency moves, items,
ability, tera type, and EV spread per species — to produce a team string suitable
for `build_pe_state_gen9`.

Used by replay_to_trajectory (Step 4 of the gen9 value-net pipeline).
"""

from __future__ import annotations

import re

from showdown.chaos_stats import ChaosStats, _normalize_name
from showdown.replay_parse_gen9 import PokemonReveal


_DEFAULT_SPREAD = ("Jolly", {"hp": 0, "atk": 252, "def": 4, "spa": 0, "spd": 0, "spe": 252})
_EV_ORDER = ("hp", "atk", "def", "spa", "spd", "spe")
_EV_LABEL = {"hp": "HP", "atk": "Atk", "def": "Def", "spa": "SpA", "spd": "SpD", "spe": "Spe"}


def _info_score(rev: PokemonReveal) -> int:
    """Rank reveals by how much info they carry (so we keep the richer of duplicates)."""
    return (
        len(rev.moves)
        + (1 if rev.item else 0)
        + (1 if rev.ability else 0)
        + (1 if rev.tera_type else 0)
    )


def _dedup_by_species(reveals: dict[str, PokemonReveal]) -> dict[str, PokemonReveal]:
    """Collapse preview-keyed + nickname-keyed entries for the same species.

    Two passes:
      1. Exact-species dedup keeps the richest reveal per species id.
      2. Drop form-masked previews ('Zamazenta-*') when a concrete reveal of
         the same base species exists ('Zamazenta', 'Zamazenta-Crowned').
    """
    by_species: dict[str, PokemonReveal] = {}
    for rev in reveals.values():
        existing = by_species.get(rev.species)
        if existing is None or _info_score(rev) > _info_score(existing):
            by_species[rev.species] = rev

    for masked in [k for k in by_species if k.endswith("-*")]:
        prefix = masked[:-2]
        for other in by_species:
            if other != masked and (other == prefix or other.startswith(prefix + "-")):
                del by_species[masked]
                break
    return by_species


def _format_evs(evs: dict[str, int]) -> str:
    parts = [f"{evs[k]} {_EV_LABEL[k]}" for k in _EV_ORDER if evs.get(k, 0) > 0]
    return " / ".join(parts) if parts else "0 HP"


def _build_mon_block(rev: PokemonReveal, chaos: ChaosStats) -> str:
    """Render one mon as a Showdown-paste block."""
    norm = _normalize_name(rev.species)
    stats = chaos.pokemon.get(norm)

    # Moves: revealed first, fill from chaos top-moves.
    moves: list[str] = list(rev.moves)
    if stats is not None:
        for m in stats.top_moves(8):
            if len(moves) >= 4:
                break
            if m not in moves:
                moves.append(m)
    while len(moves) < 4:
        moves.append("splash")
    moves = moves[:4]

    # Item.
    if rev.item:
        item = rev.item
    elif stats is not None and stats.top_item():
        item = stats.top_item()
    else:
        item = "leftovers"

    # Ability.
    if rev.ability:
        ability = rev.ability
    elif stats is not None and stats.top_ability():
        ability = stats.top_ability()
    else:
        ability = "noability"

    # Tera.
    if rev.tera_type:
        tera = rev.tera_type
    elif stats is not None and stats.top_tera_type():
        tera = stats.top_tera_type()
    else:
        tera = "Stellar"

    # Spread.
    spread = stats.top_spread() if stats is not None else None
    if spread is None or all(v == 0 for v in spread[1].values()):
        nature, evs = _DEFAULT_SPREAD
    else:
        nature, evs = spread

    lines = [
        f"{rev.species} @ {item}",
        f"Ability: {ability}",
        f"Tera Type: {tera}",
        f"EVs: {_format_evs(evs)}",
        f"{nature} Nature",
    ]
    lines.extend(f"- {m}" for m in moves)
    return "\n".join(lines)


def reconstruct_team(reveals: dict[str, PokemonReveal], chaos: ChaosStats,
                     lead_species: str | None = None) -> str:
    """Return a Showdown-format team paste suitable for `build_pe_state_gen9`.

    If `lead_species` is provided, the matching mon is placed at slot 0 so the
    engine starts with the same active mon as the replay.
    """
    deduped = _dedup_by_species(reveals)
    ordered = list(deduped.values())
    if lead_species is not None:
        lead_norm = _normalize_name(lead_species)
        for i, rev in enumerate(ordered):
            if _normalize_name(rev.species) == lead_norm:
                ordered.insert(0, ordered.pop(i))
                break
    blocks = [_build_mon_block(rev, chaos) for rev in ordered]
    return "\n\n".join(blocks)
