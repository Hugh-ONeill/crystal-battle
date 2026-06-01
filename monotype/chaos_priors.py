"""
Smogon-derived priors for opponent move/item/ability prediction in monotype.

Loads the same Smogon moveset .txt files used by `canonical_sets.py` but
keeps the full probability distributions (instead of collapsing to top-1).

  priors = MonotypeChaosPriors()
  priors.move_probs("Heatran", "fire")
  # -> {"earthpower": 0.90, "magmastorm": 0.66, "stealthrock": 0.63, ...}

Per-type indexing matters for monotype: Heatran on a Fire team runs
different sets than Heatran on a Steel team, so the combined chaos JSON
that other Smogon formats publish is the wrong granularity here.
"""

from __future__ import annotations

import re
from functools import lru_cache
from pathlib import Path

from monotype.canonical_sets import (
    MONOTYPE_TYPES,
    SMOGON_STATS_DIR,
    parse_moveset_file,
)


def _normalize_id(name: str) -> str:
    """lowercase + strip non-alnum (matches poke-engine move/item/ability ids)."""
    return re.sub(r"[^a-z0-9]", "", name.lower())


class _SpeciesPriors:
    """Move/item/ability/spread distributions for one (species, type) cell."""

    def __init__(self, raw: dict):
        # raw moves come as [(name, pct%)] sorted desc. % is per-slot
        # (= P(species runs this move) since each species has 4 move slots).
        # The pct already accounts for total mon usage, so it's directly
        # interpretable as the probability the mon has this move.
        self._moves = {_normalize_id(n): p / 100.0 for n, p in raw["moves"]}
        self._items = {_normalize_id(n): p / 100.0 for n, p in raw["items"]}
        self._abilities = {_normalize_id(n): p / 100.0 for n, p in raw["abilities"]}
        self._spreads = list(raw["spreads"])  # leave as [(spread_str, pct)]

    def move_probs(self) -> dict[str, float]:
        return dict(self._moves)

    def top_moves(self, n: int = 4) -> list[tuple[str, float]]:
        return sorted(self._moves.items(), key=lambda kv: -kv[1])[:n]

    def item_probs(self) -> dict[str, float]:
        return dict(self._items)

    def top_item(self) -> tuple[str, float] | None:
        if not self._items:
            return None
        return max(self._items.items(), key=lambda kv: kv[1])

    def ability_probs(self) -> dict[str, float]:
        return dict(self._abilities)

    def top_ability(self) -> tuple[str, float] | None:
        if not self._abilities:
            return None
        return max(self._abilities.items(), key=lambda kv: kv[1])


class MonotypeChaosPriors:
    """Per-(type, species) Smogon priors loaded from monotype moveset files."""

    def __init__(self, elo_bucket: int = 1500, stats_dir: Path | None = None):
        self.elo_bucket = elo_bucket
        self.stats_dir = stats_dir or SMOGON_STATS_DIR
        self._cache: dict[str, dict[str, _SpeciesPriors]] = {}
        for t in MONOTYPE_TYPES:
            path = self.stats_dir / f"gen9monotype-mono{t}-{elo_bucket}.txt"
            if not path.exists():
                self._cache[t] = {}
                continue
            parsed = parse_moveset_file(path)
            self._cache[t] = {sp: _SpeciesPriors(info) for sp, info in parsed.items()
                              if info["moves"]}

    def get(self, species: str, team_type: str) -> _SpeciesPriors | None:
        """Look up (species, type). team_type case-insensitive. Falls back to
        a base-form species if Hyphenated forms aren't in the table."""
        t = team_type.lower()
        per_type = self._cache.get(t, {})
        if species in per_type:
            return per_type[species]
        # Try base form: "Ogerpon-Hearthflame" -> "Ogerpon"
        base = species.split("-")[0]
        if base in per_type:
            return per_type[base]
        # Last resort: case-insensitive lookup
        for k in per_type:
            if k.lower() == species.lower():
                return per_type[k]
        return None

    def move_probs(self, species: str, team_type: str) -> dict[str, float]:
        sp = self.get(species, team_type)
        return sp.move_probs() if sp else {}

    def top_moves(self, species: str, team_type: str, n: int = 4) -> list[tuple[str, float]]:
        sp = self.get(species, team_type)
        return sp.top_moves(n) if sp else []


_DEFAULT_PRIORS: MonotypeChaosPriors | None = None


def default_priors(elo_bucket: int = 1500) -> MonotypeChaosPriors:
    """Lazy-singleton accessor so callers don't reload the moveset files."""
    global _DEFAULT_PRIORS
    if _DEFAULT_PRIORS is None or _DEFAULT_PRIORS.elo_bucket != elo_bucket:
        _DEFAULT_PRIORS = MonotypeChaosPriors(elo_bucket=elo_bucket)
    return _DEFAULT_PRIORS


# ---------------------------------------------------------------------
# Side-aware lookup: given an engine Side object, identify the active mon
# and the team's mono-type, then return move probs for that active.
# ---------------------------------------------------------------------

def _norm_species(s: str) -> str:
    """Normalize a species name for cross-source matching."""
    return re.sub(r"[^a-z0-9]", "", s.lower())


_SPECIES_TYPES_NORM: dict[str, list[str]] | None = None


def _species_types_norm() -> dict[str, list[str]]:
    """Lazy-load species_types.json into normalized-key form."""
    global _SPECIES_TYPES_NORM
    if _SPECIES_TYPES_NORM is None:
        import json
        p = Path(__file__).parent.parent / "showdown" / "species_types.json"
        raw = json.load(open(p))
        _SPECIES_TYPES_NORM = {_norm_species(k): v for k, v in raw.items()}
    return _SPECIES_TYPES_NORM


@lru_cache(maxsize=2048)
def _detect_side_type(species_tuple: tuple[str, ...]) -> str | None:
    """Infer the shared monotype from a tuple of species ids.

    poke-engine ids are already lowercase no-space; species_types.json uses
    title-case with hyphens/spaces. Normalize both ends for a robust match.
    """
    if not species_tuple:
        return None
    table = _species_types_norm()
    type_sets = []
    for sp_id in species_tuple:
        key = _norm_species(sp_id)
        types = table.get(key)
        if not types:
            # Try the base form (Ogerpon for Ogerpon-Hearthflame, etc.)
            for cand_key, cand_types in table.items():
                if cand_key.startswith(key) or key.startswith(cand_key):
                    types = cand_types
                    break
        if not types:
            return None
        type_sets.append(set(t.lower() for t in types))
    common = set.intersection(*type_sets)
    if not common:
        return None
    if len(common) == 1:
        return next(iter(common))
    return sorted(common)[0]


def active_move_probs(side, priors: MonotypeChaosPriors | None = None) -> dict[str, float]:
    """Return move-id -> probability for `side`'s currently active mon."""
    if priors is None:
        priors = default_priors()
    species_tuple = tuple(p.id for p in side.pokemon)
    team_type = _detect_side_type(species_tuple)
    if not team_type:
        return {}
    active = side.pokemon[int(side.active_index)]
    return priors.move_probs(active.id.title(), team_type)


def reweight_by_priors(move_results, priors_for_active: dict[str, float],
                       alpha: float = 1.0) -> dict[str, float]:
    """Combine MCTS visit counts with chaos priors at the active.

    Final score = visits * (1 + alpha * prior_prob).  alpha = 0 disables the
    bias (pure MCTS); alpha = 1 gives ~equal weight to a 100% prior and the
    full visit count.

    Returns {move_choice: combined_score}; caller picks argmax.
    """
    out: dict[str, float] = {}
    for r in move_results:
        if r.move_choice.endswith("-tera"):
            continue
        # poke-engine move ids are already lowercase no-space; chaos priors
        # use the same normalization (via _normalize_id), so the move_choice
        # for an attacking move matches directly. Switch moves ("switch X")
        # have no chaos entry and just get the visit-count fallback.
        mv_key = r.move_choice.split()[-1] if r.move_choice.startswith("switch") else r.move_choice
        prior = priors_for_active.get(_normalize_id(mv_key), 0.0)
        out[r.move_choice] = r.visits * (1.0 + alpha * prior)
    return out
