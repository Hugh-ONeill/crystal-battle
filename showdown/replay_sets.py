# Tier-2 opponent set inference: sets observed in real ladder replays.
#
# Built by build_replay_sets.py from the scraped corpus. Two indexes:
#   species: joint MOVESET FRAGMENTS with counts (moves seen together in one
#            game) plus item/ability/tera counters
#   teams:   archetype index keyed by the sorted 6-species roster; a preview
#            match predicts every mon's moves/tera from games of that team
#
# Structural caveat that shapes the whole tier: choice items (Scarf, Specs,
# Band) never emit a protocol event, so replay item counts only cover
# self-revealing items (Leftovers, Life Orb, boots, orbs, berries...). Moves
# and tera types are the reliable replay signal; items and spreads must come
# from the curated/chaos tiers. This is also why foul-play's replay tier is
# moves-only.

from __future__ import annotations

import json
import re
from pathlib import Path

HERE = Path(__file__).parent


def _normalize(name: str) -> str:
    return re.sub(r"[^a-z0-9]", "", name.lower())


class ReplaySetsIndex:
    def __init__(self, format: str = "gen9ou", path: str | Path | None = None):
        raw = json.loads(Path(path or HERE / f"{format}_replay_sets.json")
                         .read_text())
        self.replays = raw.get("replays", 0)
        self.species = raw.get("species", {})
        # movesets arrive as [[moves...], count]; normalize to tuples
        for entry in self.species.values():
            entry["movesets"] = [(tuple(ms), c) for ms, c in entry["movesets"]]
        self.teams = raw.get("teams", {})
        for team in self.teams.values():
            for mon in team["mons"].values():
                mon["movesets"] = [(tuple(ms), c) for ms, c in mon["movesets"]]

    def team_match(self, species_iterable) -> dict | None:
        """Archetype entry for this exact 6-species roster, or None."""
        species = sorted(_normalize(s) for s in species_iterable)
        if len(species) != 6:
            return None
        return self.teams.get("|".join(species))

    @staticmethod
    def _consistent(fragments, known_moves: tuple[str, ...]):
        known = {_normalize(m) for m in known_moves}
        return [(ms, c) for ms, c in fragments if known <= set(ms)]

    def movesets(self, species: str, known_moves: tuple[str, ...] = (),
                 team: dict | None = None) -> list[tuple[tuple, int]]:
        """Fragments consistent with revealed moves, archetype-first."""
        if team is not None:
            entry = team["mons"].get(_normalize(species))
            if entry:
                out = self._consistent(entry["movesets"], known_moves)
                if out:
                    return out
        entry = self.species.get(_normalize(species))
        if entry is None:
            return []
        return self._consistent(entry["movesets"], known_moves)

    def pick_moves(self, species: str, known_moves: tuple[str, ...] = (),
                   team: dict | None = None, rng=None) -> list[str] | None:
        """One observed moveset fragment: count-weighted sample with rng,
        most-common without. None when nothing consistent was ever seen."""
        frags = self.movesets(species, known_moves, team)
        if not frags:
            return None
        if rng is None:
            return list(max(frags, key=lambda f: f[1])[0])
        return list(rng.choices([f[0] for f in frags],
                                weights=[f[1] for f in frags])[0])

    def pick_tera(self, species: str, team: dict | None = None,
                  rng=None) -> str | None:
        for source in ((team or {}).get("mons", {}).get(_normalize(species)),
                       self.species.get(_normalize(species))):
            teras = (source or {}).get("teras") or []
            if teras:
                if rng is None:
                    return teras[0][0]
                return rng.choices([t for t, _ in teras],
                                   weights=[c for _, c in teras])[0]
        return None


_index_cache: dict[str, ReplaySetsIndex | None] = {}


def get_index(format: str = "gen9ou") -> ReplaySetsIndex | None:
    if format not in _index_cache:
        try:
            _index_cache[format] = ReplaySetsIndex(format=format)
        except Exception:
            _index_cache[format] = None
    return _index_cache[format]
