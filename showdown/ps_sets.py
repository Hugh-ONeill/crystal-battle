# Tier-1 opponent set inference: Pokemon Showdown's curated set database.
#
# ps_sets_gen9.json (vendored by fetch_ps_sets.py) carries complete sets per
# species per format: the Smogon dex analysis sets plus a "Showdown Usage"
# composite. Unlike chaos-stat marginals, these are JOINT sets — the spread
# that goes with the item that goes with the moves — so sampling them can't
# assemble combinations nobody runs. This mirrors the top tier of foul-play's
# TeamDatasets (its biggest architectural edge over pure chaos inference).
#
# The index normalizes everything into the same dict shape the translator's
# _opp_set produces, precomputes each candidate's speed stat, and answers
# "which candidates are consistent with what we've observed".

from __future__ import annotations

import json
import re
from pathlib import Path

DATA_PATH = Path(__file__).parent / "ps_sets_gen9.json"
# Curated meta-tech overlay (tracked): PS's export flattens Smogon's slashed
# move options to the FIRST choice only (verified 2026-07-30: 0/278 gen9ou
# sets carry an alternative), so tech moves like Kingambit's slashed Low Kick
# are invisible to set sampling. The overlay re-adds them; a slot may be a
# LIST of alternatives, expanded into per-combination candidates below.
OVERRIDES_PATH = Path(__file__).parent / "ps_sets_overrides.json"

_STATS = ("hp", "atk", "def", "spa", "spd", "spe")


def _normalize(name: str) -> str:
    return re.sub(r"[^a-z0-9]", "", name.lower())


def _calc_stat(base: int, iv: int, ev: int, level: int, mult: float,
               is_hp: bool = False) -> int:
    inner = (2 * base + iv + ev // 4) * level // 100
    if is_hp:
        return inner + level + 10
    return int((inner + 5) * mult)


# Gen 3+ nature table (boosted, nerfed); neutral natures absent
_NATURE_TABLE = {
    "adamant": ("atk", "spa"), "modest": ("spa", "atk"),
    "jolly": ("spe", "spa"), "timid": ("spe", "atk"),
    "bold": ("def", "atk"), "calm": ("spd", "atk"),
    "impish": ("def", "spa"), "careful": ("spd", "spa"),
    "relaxed": ("def", "spe"), "quiet": ("spa", "spe"),
    "brave": ("atk", "spe"), "sassy": ("spd", "spe"),
    "naughty": ("atk", "spd"), "lonely": ("atk", "def"),
    "hasty": ("spe", "def"), "naive": ("spe", "spd"),
    "rash": ("spa", "spd"), "mild": ("spa", "def"),
    "gentle": ("spd", "def"), "lax": ("def", "spd"),
}


class PSSetsIndex:
    """Per-format index of curated full sets, in _opp_set dict shape."""

    def __init__(self, format: str = "gen9ou", path: str | Path | None = None,
                 base_stats: dict[str, dict] | None = None,
                 overrides_path: str | Path | None = OVERRIDES_PATH):
        raw = json.loads(Path(path or DATA_PATH).read_text())
        fmt = raw.get(format, {})
        # merge the meta-tech overlay: same-name sets REPLACE the vendored
        # entry (e.g. re-slashing a flattened slot), new names ADD candidates
        if overrides_path and Path(overrides_path).exists():
            ov = json.loads(Path(overrides_path).read_text()).get(format, {})
            for section in ("dex", "stats"):
                for species, sets in (ov.get(section) or {}).items():
                    fmt.setdefault(section, {}).setdefault(species, {}).update(sets)
        if base_stats is None:
            from poke_env.data.gen_data import GenData
            pokedex = GenData.from_gen(9).pokedex
            base_stats = {k: v.get("baseStats", {}) for k, v in pokedex.items()}

        self.candidates: dict[str, list[dict]] = {}
        for section, weight in (("stats", 1.5), ("dex", 1.0)):
            for species, sets in fmt.get(section, {}).items():
                norm = _normalize(species)
                bs = base_stats.get(norm, {})
                for set_name, s in sets.items():
                    for cand in self._parse_set(set_name, s, bs, weight):
                        self.candidates.setdefault(norm, []).append(cand)

    @staticmethod
    def _parse_set(set_name: str, s: dict, bs: dict, weight: float) -> list[dict]:
        # A slot may be a LIST of slashed alternatives (the overlay restores
        # what PS's export flattens to the first option). Expand every
        # combination into its own candidate with the sampling weight split
        # evenly, so consistent() filtering on revealed moves keeps exactly
        # the variants the observations allow.
        slots: list[list[str]] = []
        for m in (s.get("moves") or []):
            if isinstance(m, str):
                slots.append([m])
            elif isinstance(m, list):
                opts = [o for o in m if isinstance(o, str)]
                if opts:
                    slots.append(opts)
        if not slots:
            return []
        from itertools import product
        combos = list(product(*slots))[:8]   # slash-blowup guard
        evs = dict.fromkeys(_STATS, 0)
        evs.update({k: v for k, v in (s.get("evs") or {}).items() if k in evs})
        ivs = dict.fromkeys(_STATS, 31)
        ivs.update({k: v for k, v in (s.get("ivs") or {}).items() if k in ivs})
        nature = (s.get("nature") or "Serious").capitalize()
        pair = _NATURE_TABLE.get(nature.lower())
        spe_mult = 1.1 if pair and pair[0] == "spe" else \
            (0.9 if pair and pair[1] == "spe" else 1.0)
        shared = {
            "weight": weight / len(combos),
            "nature": nature,
            "evs": evs,
            "ivs": ivs,
            "item": _normalize(s.get("item") or "") or "none",
            "ability": _normalize(s.get("ability") or "") or None,
            "tera_type": (s.get("teraType") or "").lower() or None,
            "spe_stat": _calc_stat(bs.get("spe", 80), ivs["spe"], evs["spe"],
                                   100, spe_mult),
        }
        out = []
        for combo in combos:
            chosen = [m for slot, m in zip(slots, combo) if len(slot) > 1]
            name = f"{set_name} ({'/'.join(chosen)})" if chosen else set_name
            out.append({"name": name,
                        "moves": [_normalize(m) for m in combo][:4],
                        **shared})
        return out

    def consistent(self, species: str, known_moves: tuple[str, ...] = (),
                   known_item: str | None = None,
                   known_ability: str | None = None,
                   speed_floor: float | None = None) -> list[dict]:
        """Candidates compatible with everything observed so far. A revealed
        move outside a candidate's moveset, a mismatched revealed item or
        ability, or a spread too slow for an observed speed floor (scarf
        credited when the candidate holds one) all eliminate it."""
        out = []
        for cand in self.candidates.get(species, []):
            cmoves = set(cand["moves"])
            if any(_normalize(m) not in cmoves for m in known_moves):
                continue
            if known_item and cand["item"] != _normalize(known_item):
                continue
            if known_ability and cand["ability"] and \
                    cand["ability"] != _normalize(known_ability):
                continue
            if speed_floor is not None:
                mult = 1.5 if cand["item"] == "choicescarf" else 1.0
                if cand["spe_stat"] * mult < speed_floor:
                    continue
            out.append(cand)
        return out


_index_cache: dict[str, PSSetsIndex] = {}


def get_index(format: str = "gen9ou") -> PSSetsIndex | None:
    """Cached per-format index; None when no vendored data covers the format."""
    if format not in _index_cache:
        try:
            _index_cache[format] = PSSetsIndex(format=format)
        except Exception:
            _index_cache[format] = None
    return _index_cache[format]
