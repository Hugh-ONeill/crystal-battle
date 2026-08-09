#!/usr/bin/env python3
"""Build a PROXY TEAM SUITE for a scouted opponent: their rosters, their
observed behavior, playable team files.

WHY (2026-08-04): richwoman is foul-play, her 12 rosters are known, and 11
of 12 are exact metamon-archive matches at 95%+ set fidelity — so
`par_series --opponent foulplay --fp-suite <proxy>` is a high-fidelity home
game against HER, hundreds of games a day instead of ~25 ladder games a
night. Teambuilding iteration needs exactly that loop.

Per roster: archive candidates for the exact roster, scored by overlap
with the opponent's OBSERVED sets (scouting book, per-roster first, then
species-level: moves + item + tera weighted like the archive+book tier);
best candidate becomes the proxy. Rosters without archive coverage are
synthesized: top-4 observed moves per species, best observed item/tera,
spread from the curated-PS set (chaos top item when the book is silent).

Usage:
  build_opponent_proxy.py --opponent richwoman --out showdown/teams/rw_proxy
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from pathlib import Path

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE.parent))


def norm(s: str) -> str:
    return re.sub(r"[^a-z0-9]", "", (s or "").lower())


def book_sets(prof: dict, roster: list[str]) -> dict:
    """species(norm) -> {'moves': Counter, 'items': Counter, 'teras': Counter}
    per-roster record preferred, species-level blended underneath."""
    from showdown.team_archive import roster_key
    out: dict = {}
    layers = []
    per = (prof.get("sets_by_roster") or {}).get(roster_key(roster))
    if per:
        layers.append(per)
    if prof.get("sets"):
        layers.append(prof["sets"])
    for layer in layers:
        for sp, e in layer.items():
            d = out.setdefault(norm(sp), {"moves": Counter(),
                                          "items": Counter(),
                                          "teras": Counter()})
            for k, tgt in (("moves", "moves"), ("items", "items"),
                           ("teras", "teras")):
                for v, c in (e.get(k) or {}).items():
                    d[tgt][norm(v)] += c
    return out


def score_candidate(team: dict, obs: dict) -> float:
    """Overlap of a parsed candidate with observed behavior; item and tera
    weigh like a move slot each (the axes where archives are strongest)."""
    hit = tot = 0.0
    for sp, (moves, item, tera) in team.items():
        o = obs.get(sp)
        if not o:
            continue
        seen_moves = set(o["moves"])
        hit += len(moves & seen_moves)
        tot += len(moves)
        if o["items"]:
            tot += 1
            hit += item in o["items"]
        if o["teras"]:
            tot += 1
            hit += tera in o["teras"]
    return hit / tot if tot else 0.0


def synthesize(species: str, obs: dict) -> str:
    """A paste block for one mon from observations + curated-PS scaffolding."""
    from showdown.ps_sets import get_index
    o = obs.get(norm(species)) or {"moves": Counter(), "items": Counter(),
                                   "teras": Counter()}
    ps = None
    try:
        idx = get_index("gen9ou")
        cands = idx.candidates.get(norm(species)) or []
        if cands:
            # candidate whose item matches the observed item, else first
            want = o["items"].most_common(1)
            ps = next((c for c in cands
                       if want and norm(c.get("item")) == norm(want[0][0])),
                      cands[0])
    except Exception:
        pass
    moves = [m for m, _ in o["moves"].most_common(4)]
    if len(moves) < 4 and ps:
        for m in ps.get("moves", []):
            if norm(m) not in {norm(x) for x in moves}:
                moves.append(norm(m))
            if len(moves) == 4:
                break
    item = (o["items"].most_common(1) or [(ps.get("item") if ps else
                                           "leftovers", 0)])[0][0]
    tera = (o["teras"].most_common(1) or [((ps.get("tera_type") or "normal")
                                           if ps else "normal", 0)])[0][0]
    from showdown.belief_accuracy_truth import parse_paste  # noqa: F401
    evs = (ps.get("evs") if ps else None) or {}
    ev_s = " / ".join(f"{v} {k.upper()}" for k, v in evs.items() if v) \
        or "252 HP / 4 Def / 252 SpD"
    nature = (ps.get("nature") if ps else None) or "Serious"
    lines = [f"{species} @ {_display(item)}",
             f"Ability: {(ps.get('ability') if ps else None) or ''}".rstrip(': '),
             f"Tera Type: {_display(tera)}",
             f"EVs: {ev_s}", f"{nature} Nature"]
    lines = [l for l in lines if l and not l.endswith("Ability")]
    lines += [f"- {_display(m)}" for m in moves[:4]]
    return "\n".join(lines)


_DISPLAY_CACHE: dict | None = None


def _display(token: str) -> str:
    """normalized id -> display name via the moves/items/dex data."""
    global _DISPLAY_CACHE
    if _DISPLAY_CACHE is None:
        _DISPLAY_CACHE = {}
        try:
            from showdown.set_inference import _moves_data
            for mid, e in _moves_data().items():
                if e.get("name"):
                    _DISPLAY_CACHE[mid] = e["name"]
        except Exception:
            pass
    t = norm(token)
    if t in _DISPLAY_CACHE:
        return _DISPLAY_CACHE[t]
    # fall back to title-casing the token (items mostly)
    return re.sub(r"(^|(?<= ))([a-z])", lambda m: m.group(2).upper(),
                  re.sub(r"([a-z])([A-Z0-9])", r"\1 \2", token))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--opponent", default="richwoman")
    ap.add_argument("--out", default=str(HERE / "teams" / "rw_proxy"))
    ap.add_argument("--max-cands", type=int, default=120)
    # An archive team that merely SHARES A ROSTER is not the opponent. Built
    # for richwoman, where 11 of 12 rosters matched the archive at 95%+, the
    # old code took ANY candidate that existed and only synthesized when none
    # did. MuratiBot-1 exposed that: its best candidate scored 67% and got 4
    # of 6 ITEMS wrong (Choice Band Scizor for the real Leftovers Swords
    # Dance one, Booster Energy Iron Valiant for a Choice Scarf), which is a
    # different opponent to practise against. Below this threshold, 121 games
    # of direct observation beat a roster-shaped guess.
    ap.add_argument("--min-obs-match", type=float, default=0.85)
    args = ap.parse_args()

    from showdown.belief_accuracy import ArchiveTier
    from showdown.team_archive import roster_key

    book = json.loads((HERE / "scouting_book.json").read_text())
    prof = book.get(args.opponent)
    if not prof:
        sys.exit(f"no scouting profile for {args.opponent}")
    arch = ArchiveTier(max_cands=args.max_cands)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    rosters = sorted(prof["rosters"], key=lambda x: -x[1])
    for i, (roster, n_games) in enumerate(rosters, 1):
        obs = book_sets(prof, roster)
        cands = arch.candidates(roster)
        best, best_score = None, -1.0
        for team, _w in cands:
            s = score_candidate(team, obs)
            if s > best_score:
                best, best_score = team, s
        anchor = norm(roster[0])
        name = f"{i:02d}_{anchor}.txt"
        if best is not None and best_score >= args.min_obs_match:
            # write the archive candidate's ORIGINAL paste (full spreads),
            # found by re-matching the roster into the index files
            src = _best_source_paste(arch, roster, obs)
            (out_dir / name).write_text(src)
            print(f"{name}: archive candidate, obs-match {best_score:.0%} "
                  f"({n_games} book games)")
        else:
            blocks = [synthesize(sp, obs) for sp in roster]
            (out_dir / name).write_text("\n\n".join(blocks) + "\n")
            why = ("no archive candidate" if best is None
                   else f"best archive match only {best_score:.0%} "
                        f"< --min-obs-match {args.min_obs_match:.0%}")
            print(f"{name}: SYNTHESIZED from book+PS ({n_games} book games) "
                  f"— {why}")


def _best_source_paste(arch, roster, obs) -> str:
    """The raw paste text of the best-scoring archive file for a roster."""
    from showdown.team_archive import roster_key
    k = arch.roster_key(roster)
    best_path, best_s = None, -1.0
    for ci in (arch.by.get(k) or [])[: arch.max_cands]:
        entry = arch.teams[ci]
        path = entry if isinstance(entry, str) else entry[0]
        team = arch._parse(path)
        if not team:
            continue
        s = score_candidate(
            {sp: (mv, it, tr) for sp, (mv, it, tr) in team.items()}, obs)
        if s > best_s:
            best_path, best_s = path, s
    return (arch.root / best_path).read_text(errors="replace")


if __name__ == "__main__":
    main()
