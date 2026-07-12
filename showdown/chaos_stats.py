# GSC OU usage statistics from Smogon chaos JSON
# parses the raw weighted counts into usable probabilities
#
# source: https://www.smogon.com/stats/2025-10/chaos/gen2ou-1760.json

from __future__ import annotations

import json
from pathlib import Path


def _normalize_name(name: str) -> str:
    """Normalize Pokemon/move names to lowercase no-space form."""
    import re
    return re.sub(r"[^a-z0-9]", "", name.lower())


class ChaosStats:
    """Parsed Smogon chaos statistics for opponent prediction."""

    def __init__(self, path: str | Path | None = None, format: str = "gen2ou"):
        if path is None:
            path = Path(__file__).parent / f"{format}_chaos.json"
        with open(path) as f:
            raw = json.load(f)

        self.pokemon: dict[str, PokemonStats] = {}
        for name, stats in raw["data"].items():
            norm = _normalize_name(name)
            self.pokemon[norm] = PokemonStats(name, stats)

        # sorted by usage
        self.ranking = sorted(self.pokemon.keys(),
                              key=lambda k: -self.pokemon[k].usage)

    def predict_team(self, revealed: dict[str, RevealedMon],
                     n_fill: int = 5) -> list[PredictedMon]:
        """Predict unrevealed opponent Pokemon based on what's been seen.

        Args:
            revealed: dict of normalized_name -> RevealedMon with known info
            n_fill: number of unrevealed slots to fill

        Returns:
            list of PredictedMon, most likely first
        """
        candidates = self._team_candidate_scores(set(revealed.keys()))
        ranked = sorted(candidates.keys(), key=lambda k: -candidates[k])

        result = []
        for species in ranked[:n_fill]:
            stats = self.pokemon[species]
            # predict moveset: narrow by any observed moves if we had them
            moves = stats.top_moves(4)
            item = stats.top_item()
            result.append(PredictedMon(species, stats.display_name, moves, item))

        return result

    def _team_candidate_scores(self, revealed_names: set[str]) -> dict[str, float]:
        """usage x teammate-correlation score for every unrevealed species."""
        candidates = {}
        for species, stats in self.pokemon.items():
            if species in revealed_names:
                continue
            score = stats.usage
            for rev_name in revealed_names:
                if rev_name in self.pokemon:
                    teammate_prob = self.pokemon[rev_name].teammate_prob(species)
                    if teammate_prob > 0:
                        score *= (1.0 + teammate_prob * 2.0)
            candidates[species] = score
        return candidates

    def sample_team(self, revealed: dict[str, "RevealedMon"], n_fill: int,
                    rng) -> list[str]:
        """Sampled variant of predict_team: species drawn without replacement
        proportional to the same usage x teammate score, instead of taking
        the deterministic top-N. Returns normalized species names."""
        candidates = self._team_candidate_scores(set(revealed.keys()))
        out: list[str] = []
        while candidates and len(out) < n_fill:
            keys = list(candidates.keys())
            species = rng.choices(
                keys, weights=[candidates[k] for k in keys])[0]
            out.append(species)
            del candidates[species]
        return out

    def narrow_moveset(self, species: str, known_moves: list[str]) -> list[str]:
        """Given some revealed moves, predict the remaining ones.

        Uses conditional probability: if we know move A is in the set,
        which other moves most commonly appear alongside A?
        """
        if species not in self.pokemon:
            return ["doubleedge", "earthquake", "rest", "sleeptalk"]

        stats = self.pokemon[species]
        all_moves = stats.move_probs()

        # filter out known moves, sort remaining by probability
        remaining = {m: p for m, p in all_moves.items()
                     if m not in {_normalize_name(k) for k in known_moves}}

        # return enough to fill 4 slots
        n_need = 4 - len(known_moves)
        ranked = sorted(remaining.keys(), key=lambda k: -remaining[k])
        return list(known_moves) + ranked[:n_need]


class PokemonStats:
    """Parsed stats for one Pokemon species."""

    def __init__(self, display_name: str, raw: dict):
        self.display_name = display_name
        self.raw_count = raw["Raw count"]
        self.usage = raw.get("usage", 0)

        # normalize move weights to probabilities (P(mon has this move))
        move_total = sum(raw.get("Moves", {}).values())
        slots = move_total / 4 if move_total > 0 else 1
        self._moves = {_normalize_name(m): w / slots
                       for m, w in raw.get("Moves", {}).items() if w > 0}

        # items: normalize to probabilities
        item_total = sum(raw.get("Items", {}).values())
        self._items = {_normalize_name(i): w / item_total if item_total > 0 else 0
                       for i, w in raw.get("Items", {}).items() if w > 0}

        # teammates: normalize to probabilities
        self._teammates = {}
        for name, weight in raw.get("Teammates", {}).items():
            norm = _normalize_name(name)
            # teammate weight / raw_count = P(teammate | this mon)
            self._teammates[norm] = weight / self.raw_count if self.raw_count > 0 else 0

        # checks and counters
        self._counters = {}
        for name, vals in raw.get("Checks and Counters", {}).items():
            norm = _normalize_name(name)
            if isinstance(vals, list) and len(vals) >= 2:
                self._counters[norm] = vals[1]  # KO/switch rate

        # abilities (gen3+) — names already lowercase ids in chaos JSON
        ab_total = sum(raw.get("Abilities", {}).values())
        self._abilities = {a: w / ab_total if ab_total > 0 else 0
                           for a, w in raw.get("Abilities", {}).items() if w > 0}

        # tera types (gen9) — chaos keys are lowercase type names
        tt_total = sum(raw.get("Tera Types", {}).values())
        self._tera_types = {t: w / tt_total if tt_total > 0 else 0
                            for t, w in raw.get("Tera Types", {}).items() if w > 0}

        # spreads — "Nature:hp/atk/def/spa/spd/spe" -> weight
        # store as sorted list of (nature, evs_dict, prob) so callers can pick top.
        sp_total = sum(raw.get("Spreads", {}).values())
        self._spreads: list[tuple[str, dict[str, int], float]] = []
        for spread, weight in raw.get("Spreads", {}).items():
            if weight <= 0 or ":" not in spread:
                continue
            nature, ev_str = spread.split(":", 1)
            parts = ev_str.split("/")
            if len(parts) != 6:
                continue
            try:
                ev_vals = [int(x) for x in parts]
            except ValueError:
                continue
            evs = dict(zip(("hp", "atk", "def", "spa", "spd", "spe"), ev_vals))
            prob = weight / sp_total if sp_total > 0 else 0
            self._spreads.append((nature, evs, prob))
        self._spreads.sort(key=lambda x: -x[2])

    def move_probs(self) -> dict[str, float]:
        """Get move probabilities (P(mon has this move))."""
        return dict(self._moves)

    def top_moves(self, n: int = 4) -> list[str]:
        """Get the N most likely moves."""
        ranked = sorted(self._moves.keys(), key=lambda k: -self._moves[k])
        return ranked[:n]

    def top_item(self) -> str:
        """Get the most likely item."""
        if not self._items:
            return "leftovers"
        return max(self._items.keys(), key=lambda k: self._items[k])

    def item_prob(self, item: str) -> float:
        return self._items.get(_normalize_name(item), 0)

    def teammate_prob(self, species: str) -> float:
        return self._teammates.get(_normalize_name(species), 0)

    def counter_score(self, species: str) -> float:
        return self._counters.get(_normalize_name(species), 0)

    def top_ability(self) -> str | None:
        if not self._abilities:
            return None
        return max(self._abilities.keys(), key=lambda k: self._abilities[k])

    def top_tera_type(self) -> str | None:
        if not self._tera_types:
            return None
        return max(self._tera_types.keys(), key=lambda k: self._tera_types[k])

    def top_spread(self) -> tuple[str, dict[str, int]] | None:
        """Return (nature, evs_dict) of the most-frequent spread, or None."""
        if not self._spreads:
            return None
        nature, evs, _ = self._spreads[0]
        return nature, evs

    # natures that boost speed, for the pessimistic-spread tiebreak
    _PLUS_SPE = frozenset({"Timid", "Jolly", "Hasty", "Naive"})

    def sample_set(self, rng, known_moves: tuple[str, ...] = (),
                   speed_pessimistic: bool = False) -> dict:
        """Sample one plausible full set from the chaos distributions.

        Committing to the single top set is systematically wrong whenever
        the opponent runs anything else; searching over sampled sets (like
        foul-play does) is robust to that. `known_moves` are already
        revealed — they occupy slots, so only the remainder is sampled
        (without replacement, probability-weighted).

        `speed_pessimistic` builds the worst-plausible-speed variant: the
        fastest listed spread, and Choice Scarf whenever it has >=2% usage.
        Motivation: speed-floor inference only fires AFTER a scarfer outspeeds
        something — one body too late to stop a sweep. Searching one
        pessimistic world alongside the plain sample hedges pre-emptively.

        Returns the same dict shape as the top-set path: nature/evs/item/
        ability/moves/tera_type.
        """
        def pick(dist: dict[str, float]) -> str | None:
            if not dist:
                return None
            keys = list(dist.keys())
            return rng.choices(keys, weights=[dist[k] for k in keys])[0]

        if self._spreads:
            if speed_pessimistic:
                nature, evs, _ = max(
                    self._spreads,
                    key=lambda s: (s[1]["spe"], s[0] in self._PLUS_SPE, s[2]))
            else:
                nature, evs, _ = rng.choices(
                    self._spreads, weights=[p for _, _, p in self._spreads])[0]
        else:
            nature, evs = "Serious", dict.fromkeys(
                ("hp", "atk", "def", "spa", "spd", "spe"), 85)

        known = {_normalize_name(m) for m in known_moves}
        pool = {m: p for m, p in self._moves.items() if m not in known}
        moves: list[str] = []
        while pool and len(moves) < 4 - len(known):
            m = pick(pool)
            moves.append(m)
            del pool[m]

        if speed_pessimistic and self._items.get("choicescarf", 0) >= 0.02:
            item = "choicescarf"
        else:
            item = pick(self._items) or "none"
        return {
            "nature": nature,
            "evs": evs,
            "item": item,
            "ability": pick(self._abilities),
            "moves": moves,
            "tera_type": pick(self._tera_types),
        }


class RevealedMon:
    """Info we've observed about an opponent's Pokemon."""
    def __init__(self, species: str, known_moves: list[str] | None = None,
                 known_item: str | None = None, hp_frac: float = 1.0):
        self.species = species
        self.known_moves = known_moves or []
        self.known_item = known_item
        self.hp_frac = hp_frac


class PredictedMon:
    """A predicted opponent Pokemon with likely moveset and item."""
    def __init__(self, species: str, display_name: str,
                 moves: list[str], item: str):
        self.species = species
        self.display_name = display_name
        self.moves = moves
        self.item = item
