"""Full-set team archive for opponent-world sampling.

WHY (measured 2026-07-23): foul-play's opponent beliefs come 93% from
correlated whole-team archives, giving it 57.6% move accuracy at team
preview — before a single reveal — because two revealed mons identify the
archetype and pin the UNREVEALED members' full sets. Our chaos-prior
marginals structurally cannot do that (the scarf-IV-modeled-as-Booster-80%
failure is the signature error), and the existing replay archetype tier
only patches moves/tera onto chaos spreads (replay logs can't see items or
EVs). This module supplies the missing tier: full sets (item/EVs/nature/
tera/moves) for the entire roster, from the 208k-team metamon paste corpus.

Index build (one-time, ~1 min):
  python showdown/team_archive.py build --teams ~/Developer/grimoire/metamon-data/teams \
      --out showdown/teams/team_archive_gen9ou.json
The index maps roster-key (sorted 6 species ids) -> team file paths, so
runtime memory stays small: matching candidates are lazy-parsed on demand.
Uniform sampling over file entries is popularity-weighted for free —
duplicated teams appear once per copy.

Runtime: exact-roster match only (v1). No match, or every candidate
eliminated by revealed-info filtering, returns None and the caller falls
through to the existing curated/chaos tiers.
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path


def _norm(s: str) -> str:
    return re.sub(r"[^a-z0-9]", "", s.lower())


def roster_key(species: list[str]) -> str:
    return ",".join(sorted(_norm(s) for s in species))


def _scan_species(path: Path) -> list[str]:
    """Cheap 6-species extraction: block-header lines only, no full parse."""
    species, expect_header = [], True
    for line in path.read_text(errors="replace").split("\n"):
        line = line.strip()
        if not line:
            expect_header = True
            continue
        if expect_header:
            species.append(line.split("@")[0].split("(")[0].strip())
            expect_header = False
    return species


def build_index(teams_root: Path, out: Path, fmt: str = "gen9ou",
                team_size: int = 6) -> dict:
    teams, by_roster = [], {}
    files = sorted(teams_root.rglob(f"team_*.{fmt}_team"))
    for f in files:
        sp = _scan_species(f)
        if len(sp) != team_size:
            continue
        idx = len(teams)
        teams.append(str(f.relative_to(teams_root)))
        by_roster.setdefault(roster_key(sp), []).append(idx)
    index = {"root": str(teams_root), "teams": teams, "by_roster": by_roster}
    out.write_text(json.dumps(index))
    return index


class TeamArchive:
    """Lazy-loading roster matcher over the built index."""

    def __init__(self, index_path: str):
        idx = json.loads(Path(index_path).read_text())
        self._root = Path(idx["root"])
        self._teams = idx["teams"]
        self._by_roster = idx["by_roster"]
        self._parsed: dict[int, list[dict] | None] = {}

    def candidates(self, species: list[str]) -> list[int]:
        return self._by_roster.get(roster_key(species), [])

    def _team(self, idx: int) -> list[dict] | None:
        if idx not in self._parsed:
            from showdown.local_battle import parse_showdown_team
            try:
                self._parsed[idx] = parse_showdown_team(
                    (self._root / self._teams[idx]).read_text(errors="replace"))
            except Exception:
                self._parsed[idx] = None
        return self._parsed[idx]

    @staticmethod
    def _consistent(team: list[dict], revealed: dict) -> bool:
        """revealed: normalized species -> dict(moves=set, item=str|None,
        ability=str|None). A candidate survives only if every observation
        fits it. Unknown fields constrain nothing."""
        by_sp = {m["species"]: m for m in team}
        for sp, obs in revealed.items():
            mon = by_sp.get(sp)
            if mon is None:
                return False
            if not set(obs.get("moves") or ()) <= set(mon["moves"]):
                return False
            item = obs.get("item")
            if item and mon["item"] and item != mon["item"]:
                return False
            ability = obs.get("ability")
            if ability and mon["ability"] and ability != mon["ability"]:
                return False
        return True

    def sample(self, species: list[str], revealed: dict, rng=None):
        """One CORRELATED full team consistent with everything revealed, as
        {normalized species: set-dict}, or None. Deterministic (first
        consistent candidate) without rng; popularity-weighted draw with."""
        cand = self.candidates(species)
        if not cand:
            return None
        if rng is not None:
            cand = list(cand)
            rng.shuffle(cand)
        for idx in cand:
            team = self._team(idx)
            if team and self._consistent(team, revealed):
                return {m["species"]: m for m in team}
        return None


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build")
    b.add_argument("--teams", required=True)
    b.add_argument("--out", required=True)
    b.add_argument("--fmt", default="gen9ou")
    args = ap.parse_args()
    if args.cmd == "build":
        idx = build_index(Path(args.teams).expanduser(), Path(args.out), args.fmt)
        rosters = len(idx["by_roster"])
        print(f"{len(idx['teams'])} teams, {rosters} distinct rosters -> {args.out}")


if __name__ == "__main__":
    main()
