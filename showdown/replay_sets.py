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
        self._partial_cache: dict = {}
        self.teams = raw.get("teams", {})
        for team in self.teams.values():
            for mon in team["mons"].values():
                mon["movesets"] = [(tuple(ms), c) for ms, c in mon["movesets"]]

    def partial_team_match(self, species_iterable, min_overlap: int = 5,
                           min_count: int = 3) -> dict | None:
        """Merge every archetype core sharing >= min_overlap species with this
        roster into one synthetic entry (same shape as team_match).

        WHY: team_match needs all SIX species to line up, which fires on 70.5%
        of the rosters we actually face (measured 2026-08-02 over 730 booked
        games). Relaxing to >=5 reaches 87.1% and >=4 reaches 98.4% — and the
        conditioning is worth having because a mon's build genuinely shifts
        with its neighbours: median total-variation from the pooled move
        distribution is 0.23-0.39 across Gholdengo/Great Tusk/Kingambit, versus
        a resample-null of 0.09-0.12.

        Each contributing core is weighted by its own observed count, so a
        core seen 70 times counts for more than one seen 3 times, and cores
        below min_count are ignored as possible one-player home-brews (the
        same gate team_match uses). Returns None when nothing clears the bar,
        so every caller falls through to its existing behaviour.
        """
        want = frozenset(_normalize(s) for s in species_iterable)
        if len(want) < min_overlap:
            return None
        # scanning 67k cores costs ~400ms at min_overlap=4, and the caller
        # (_resolve_archetype) runs EVERY TURN, once per world — so the
        # result is memoised on the roster. Rosters are fixed at preview, so
        # one battle pays this once.
        ck = (want, min_overlap, min_count)
        if ck in self._partial_cache:
            return self._partial_cache[ck]
        merged: dict[str, dict] = {}
        total = 0
        for key, team in self.teams.items():
            if team.get("count", 0) < min_count:
                continue
            if len(want & set(key.split("|"))) < min_overlap:
                continue
            total += team["count"]
            for sp, mon in team["mons"].items():
                acc = merged.setdefault(
                    sp, {"movesets": {}, "items": {}, "teras": {}})
                for ms, c in mon["movesets"]:
                    acc["movesets"][ms] = acc["movesets"].get(ms, 0) + c
                for field in ("items", "teras"):
                    for name, c in mon.get(field) or []:
                        acc[field][name] = acc[field].get(name, 0) + c
        if not merged:
            self._partial_cache[ck] = None
            return None
        out = {"count": total, "partial": True, "mons": {
            sp: {"movesets": sorted(a["movesets"].items(),
                                    key=lambda kv: -kv[1]),
                 "items": sorted(a["items"].items(), key=lambda kv: -kv[1]),
                 "teras": sorted(a["teras"].items(), key=lambda kv: -kv[1])}
            for sp, a in merged.items()}}
        self._partial_cache[ck] = out
        return out

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
                   team: dict | None = None, rng=None,
                   min_unanchored_count: int = 3) -> list[str] | None:
        """One observed moveset fragment: count-weighted sample with rng,
        most-common without. None when nothing consistent was ever seen.

        Tiered confidence: archetype fragments (from a count>=3 TEAM match)
        are trusted as-is — the team match IS the confidence, and per-game
        reveals make full archetype fragments legitimately low-count. Only
        species-level fragments face the unanchored count gate, since a
        species set seen once out of context is noise (the suite A/B lesson;
        the earlier flat gate wrongly nuked archetype fragments -> dnite
        A44%->C28% regression)."""
        # tier 1: archetype fragments, no count gate
        if team is not None:
            entry = team["mons"].get(_normalize(species))
            if entry:
                frags = self._consistent(entry["movesets"], known_moves)
                pick = self._weighted_pick(frags, rng)
                if pick is not None:
                    return pick
        # tier 2: species-level fragments, count-gated when unanchored
        entry = self.species.get(_normalize(species))
        if entry is None:
            return None
        frags = self._consistent(entry["movesets"], known_moves)
        if not known_moves:
            frags = [f for f in frags if f[1] >= min_unanchored_count]
        return self._weighted_pick(frags, rng)

    @staticmethod
    def _weighted_pick(frags, rng) -> list[str] | None:
        if not frags:
            return None
        if rng is None:
            return list(max(frags, key=lambda f: f[1])[0])
        return list(rng.choices([f[0] for f in frags],
                                weights=[f[1] for f in frags])[0])

    def corroborates(self, species: str, moves, min_count: int = 2,
                     min_moves: int = 3) -> bool:
        """True when a moveset resembles observed ladder play. Fragments are
        mostly partial (1-2 revealed moves per game), so corroboration is
        MOVE-level: at least min_moves of the candidate's moves must each
        appear in the corpus >= min_count times for this species. Vets
        curated (editorial) sets against reality."""
        entry = self.species.get(_normalize(species))
        if entry is None:
            return False
        move_counts: dict[str, int] = {}
        for ms, c in entry["movesets"]:
            for m in ms:
                move_counts[m] = move_counts.get(m, 0) + c
        seen = sum(1 for m in moves
                   if move_counts.get(_normalize(m), 0) >= min_count)
        return seen >= min(min_moves, len(list(moves)))

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
