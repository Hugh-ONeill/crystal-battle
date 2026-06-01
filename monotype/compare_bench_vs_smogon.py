#!/usr/bin/env python3
"""
Compare bench_monotype.py round-robin standings against Smogon's type-vs-type
matchup-chart baseline.

For each team we know its type (detected from the paste). The Smogon chart
gives expected winrate for (team_type vs opponent_type) at a given ELO bucket.
The team's "expected" bench winrate is a games-decided-weighted average of those
per-opponent baselines. Delta = engine_winrate - expected.

Big positive delta -> team punches above its type's reputation in our engine.
Big negative delta -> piloting confound or genuinely weak team for its type.

Inputs are parsed from existing artifacts so no re-running of the bench is
needed:
  - bench raw text (per-pair lines + per-team standings)
  - team paste file (so we can map team name -> type)
  - Smogon matchup chart .txt
  - species_types.json (showdown/ by default)

Usage:
  .venv/bin/python monotype/compare_bench_vs_smogon.py \\
      --bench-raw monotype/bench/teams_v6_bench_raw.txt \\
      --teams-file monotype/teams/teams_v6.txt \\
      --matchup-chart monotype/smogon_stats/2026-04/matchup/gen9monotype-matchup_chart-1500.txt
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).parent
ROOT = HERE.parent

# Matchup chart row/col canonical order (= type-effectiveness chart order).
CHART_TYPES = [
    "normal", "fighting", "flying", "poison", "ground", "rock", "bug",
    "ghost", "steel", "fire", "water", "grass", "electric", "psychic",
    "ice", "dragon", "dark", "fairy",
]


# ---------- Team paste -> per-team type detection ----------

_TEAM_HEADER_RE = re.compile(r"^=== \[gen9monotype\] (.+?) ===\s*$", re.M)


def parse_team_paste(path: Path) -> list[tuple[str, list[str]]]:
    """Return [(team_name, [species, ...]), ...] in file order."""
    text = path.read_text()
    parts = _TEAM_HEADER_RE.split(text)
    teams: list[tuple[str, list[str]]] = []
    for i in range(1, len(parts), 2):
        name = parts[i].strip()
        body = parts[i + 1]
        species: list[str] = []
        for line in body.splitlines():
            line = line.strip()
            if not line:
                continue
            # First non-empty line of each block looks like "Species @ Item"
            # or "Species (Gender) @ Item" or "Species (form-name) @ Item".
            if "@" in line and not line.lower().startswith(("ability:", "evs:",
                    "ivs:", "tera type:", "level:", "shiny:", "happiness:",
                    "- ", "nature")):
                head = line.split("@")[0].strip()
                # strip parenthesized gender/nick if present: "Mon (M)" -> "Mon"
                head = re.sub(r"\s*\([^)]*\)\s*$", "", head).strip()
                if head:
                    species.append(head)
        teams.append((name, species))
    return teams


def detect_team_type(species: list[str], species_types: dict[str, list[str]]) -> str | None:
    if not species:
        return None
    type_sets = []
    for sp in species:
        types = species_types.get(sp)
        if not types:
            base = sp.split("-")[0]
            types = species_types.get(base)
        if not types:
            return None
        type_sets.append(set(types))
    common = set.intersection(*type_sets)
    if len(common) == 1:
        return next(iter(common))
    if len(common) > 1:
        return sorted(common)[0]
    return None


# ---------- Smogon matchup chart parser ----------

_CELL_RE = re.compile(r"\|\s*(-?\d+\.\d+)%?")


def parse_matchup_chart(path: Path) -> dict[tuple[str, str], float]:
    """Return {(row_type, col_type): winrate_fraction_0to1}, mirrors omitted."""
    lines = path.read_text().splitlines()
    chart: dict[tuple[str, str], float] = {}

    # Find the header row to get col order.  Some months can re-order;
    # parse it dynamically rather than relying on CHART_TYPES.
    col_types: list[str] = []
    for ln in lines:
        if ln.startswith("|") and "normal" in ln and "fighting" in ln:
            tokens = [t.strip() for t in ln.strip().strip("|").split("|")]
            # first cell of header row is blank corner
            col_types = [t.lower() for t in tokens[1:] if t]
            break
    if not col_types:
        raise ValueError(f"no header row found in {path}")

    cur_row: str | None = None
    for ln in lines:
        if not ln.startswith("|"):
            continue
        tokens = [t.strip() for t in ln.strip().strip("|").split("|")]
        if not tokens:
            continue
        first = tokens[0]
        if first and first.lower() in CHART_TYPES:
            cur_row = first.lower()
            continue  # this line is the games-weight line, skip
        if cur_row is None:
            continue
        # this is a percentage line for cur_row
        pct_tokens = [t for t in tokens if t.endswith("%")]
        if len(pct_tokens) != len(col_types):
            continue
        for col_type, pct_str in zip(col_types, pct_tokens):
            try:
                pct = float(pct_str.rstrip("%")) / 100.0
            except ValueError:
                continue
            if col_type == cur_row:
                continue  # diagonal is a placeholder, not real data
            chart[(cur_row, col_type)] = pct
        cur_row = None  # next row header line will reset
    return chart


# ---------- bench raw text parser ----------

_PAIR_LINE_RE = re.compile(
    r"^\s*(?P<a>.+?)\s+vs\s+(?P<b>.+?)\s*:\s*"
    r"(?P<aw>\d+)W\s+(?P<bw>\d+)L\s+(?P<dr>\d+)D\s+"
    r"\(\s*[\d.]+%\s+for\s+.+?,\s*\d+/\d+\s+decided\)\s*$"
)

_TEAM_INDEX_RE = re.compile(r"^\s*\[\s*(\d+)\s*\]\s+(.+?)\s*$")
_STANDINGS_RE = re.compile(
    r"^\s*(?P<name>.+?)\s+(?P<pct>\d+\.\d+)%\s+\[(?P<lo>[\d. ]+),\s*(?P<hi>[\d. ]+)\]\s+"
    r"(?P<W>\d+)/\s*(?P<L>\d+)/\s*(?P<D>\d+)\s+(?P<dec>\d+\.\d+)%\s*$"
)


def parse_bench_raw(path: Path, teams_in_file_order: list[str]) -> dict:
    """Extract team-name list, per-pair counts, and per-team standings."""
    text = path.read_text()
    lines = text.splitlines()

    # Team index list (from header) — preferred name source since it matches
    # how the bench printed them (truncated names appear later in standings).
    team_names: list[str] = []
    for ln in lines:
        m = _TEAM_INDEX_RE.match(ln)
        if m:
            team_names.append(m.group(2).strip())
        elif team_names and not ln.startswith("  ["):
            # past the index block
            break
    if not team_names:
        # Fall back to file order
        team_names = list(teams_in_file_order)

    name_to_idx = {n: i for i, n in enumerate(team_names)}

    # truncated-name matcher: "Grounded Grounded Gro…" -> longest prefix match
    def resolve(name: str) -> int | None:
        if name in name_to_idx:
            return name_to_idx[name]
        if name.endswith("…"):
            prefix = name[:-1].rstrip()
            cands = [n for n in team_names if n.startswith(prefix)]
            if len(cands) == 1:
                return name_to_idx[cands[0]]
        return None

    # Per-pair counts (indexed by sorted (a_idx, b_idx))
    pairs: dict[tuple[int, int], dict[str, int]] = defaultdict(
        lambda: {"a_wins": 0, "b_wins": 0, "draws": 0, "a_idx": -1, "b_idx": -1}
    )
    for ln in lines:
        m = _PAIR_LINE_RE.match(ln)
        if not m:
            continue
        ai = resolve(m.group("a").strip())
        bi = resolve(m.group("b").strip())
        if ai is None or bi is None or ai == bi:
            continue
        key = (min(ai, bi), max(ai, bi))
        s = pairs[key]
        s["a_idx"], s["b_idx"] = ai, bi
        # bench prints with `a` as the first team; if file order matches, ai<bi
        if ai == key[0]:
            s["a_wins"] += int(m.group("aw"))
            s["b_wins"] += int(m.group("bw"))
        else:
            s["a_wins"] += int(m.group("bw"))
            s["b_wins"] += int(m.group("aw"))
        s["draws"] += int(m.group("dr"))

    # Per-team standings (truncated names possible)
    standings: dict[int, dict] = {}
    in_standings = False
    for ln in lines:
        if "Per-team standings" in ln:
            in_standings = True
            continue
        if not in_standings:
            continue
        m = _STANDINGS_RE.match(ln)
        if not m:
            continue
        idx = resolve(m.group("name").strip())
        if idx is None:
            continue
        standings[idx] = {
            "winrate": float(m.group("pct")) / 100.0,
            "wilson_lo": float(m.group("lo").strip()) / 100.0,
            "wilson_hi": float(m.group("hi").strip()) / 100.0,
            "wins": int(m.group("W")),
            "losses": int(m.group("L")),
            "draws": int(m.group("D")),
            "decided_pct": float(m.group("dec")) / 100.0,
        }

    return {"team_names": team_names, "pairs": dict(pairs), "standings": standings}


# ---------- baseline computation ----------

def expected_winrate_vs_mix(
    own_type: str,
    opponent_mix: list[tuple[str, int]],
    chart: dict[tuple[str, str], float],
) -> tuple[float, int]:
    """Games-decided-weighted average of chart[(own, opp)] over opponent mix.

    Returns (expected_fraction, total_weight). Mirrors and missing entries are
    skipped from both numerator and denominator.
    """
    num = 0.0
    den = 0
    for opp_type, n_decided in opponent_mix:
        if opp_type == own_type:
            continue
        wr = chart.get((own_type.lower(), opp_type.lower()))
        if wr is None or n_decided <= 0:
            continue
        num += wr * n_decided
        den += n_decided
    if den == 0:
        return (0.5, 0)
    return (num / den, den)


# ---------- main ----------

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--bench-raw", required=True, type=Path)
    p.add_argument("--teams-file", required=True, type=Path)
    p.add_argument("--matchup-chart", required=True, type=Path)
    p.add_argument("--species-types", type=Path,
                   default=ROOT / "showdown" / "species_types.json")
    p.add_argument("--verbose", action="store_true",
                   help="also print biggest per-pair deltas")
    p.add_argument("--top-pairs", type=int, default=10)
    args = p.parse_args()

    species_types = json.load(open(args.species_types))
    teams = parse_team_paste(args.teams_file)
    chart = parse_matchup_chart(args.matchup_chart)
    bench = parse_bench_raw(args.bench_raw, [t[0] for t in teams])

    n = len(bench["team_names"])
    team_types: list[str | None] = [None] * n
    # Map by name (bench-order may differ from file-order in theory)
    name_to_species = {name: spec for name, spec in teams}
    for i, name in enumerate(bench["team_names"]):
        spec = name_to_species.get(name)
        if spec is None:
            # truncated-name handling
            for nm, sp in teams:
                if name.endswith("…") and nm.startswith(name[:-1].rstrip()):
                    spec = sp
                    break
        if spec:
            team_types[i] = detect_team_type(spec, species_types)

    # Build per-team opponent mix from per-pair counts
    opp_mix: list[list[tuple[str, int]]] = [[] for _ in range(n)]
    for (lo, hi), s in bench["pairs"].items():
        decided = s["a_wins"] + s["b_wins"]
        if decided == 0:
            continue
        # `a_idx`/`b_idx` carry the original first-listed team for that pair
        # in the raw output; bench treats either direction as the same pair.
        # decided games are the only ones that contribute baseline weight.
        ai, bi = s["a_idx"], s["b_idx"]
        # symmetric: each side faces the other for `decided` games
        if team_types[ai] and team_types[bi]:
            opp_mix[ai].append((team_types[bi], decided))
            opp_mix[bi].append((team_types[ai], decided))

    # Compute expected baseline per team
    rows = []
    for i, name in enumerate(bench["team_names"]):
        tp = team_types[i]
        st = bench["standings"].get(i)
        if not st or not tp:
            continue
        exp, weight = expected_winrate_vs_mix(tp, opp_mix[i], chart)
        eng = st["winrate"]
        delta = eng - exp
        # very rough z: ((eng - exp) / sd), sd from Wilson half-width
        ci_half = (st["wilson_hi"] - st["wilson_lo"]) / 2 or 1e-9
        z = delta / (ci_half / 1.96)
        rows.append({
            "i": i, "name": name, "type": tp,
            "engine": eng, "expected": exp, "delta": delta,
            "weight": weight, "wins": st["wins"], "losses": st["losses"],
            "draws": st["draws"], "z": z,
            "ci": (st["wilson_lo"], st["wilson_hi"]),
        })

    rows.sort(key=lambda r: r["delta"], reverse=True)

    print(f"\n=== bench-vs-Smogon comparison ===")
    print(f"  bench:   {args.bench_raw}")
    print(f"  teams:   {args.teams_file}")
    print(f"  chart:   {args.matchup_chart}")
    print()
    print(f"  {'team':<24} {'type':<8} {'engine':>8} {'expected':>9} "
          f"{'delta':>7} {'z':>6}  {'95% CI':>15}")
    for r in rows:
        ci_str = f"[{r['ci'][0]*100:4.1f},{r['ci'][1]*100:5.1f}]"
        name = (r["name"][:23] + "…") if len(r["name"]) > 24 else r["name"]
        print(f"  {name:<24} {r['type']:<8} "
              f"{r['engine']*100:7.1f}% {r['expected']*100:8.1f}% "
              f"{r['delta']*100:+6.1f}% {r['z']:+5.1f}σ  {ci_str:>15}")

    if args.verbose:
        # Per-pair deltas, biggest in magnitude
        print(f"\n=== top {args.top_pairs} per-pair deltas (engine winrate for team A) ===")
        pair_rows = []
        names = bench["team_names"]
        for (lo, hi), s in bench["pairs"].items():
            decided = s["a_wins"] + s["b_wins"]
            if decided == 0:
                continue
            ai, bi = s["a_idx"], s["b_idx"]
            ta, tb = team_types[ai], team_types[bi]
            if not ta or not tb or ta == tb:
                continue
            eng = s["a_wins"] / decided
            exp = chart.get((ta.lower(), tb.lower()))
            if exp is None:
                continue
            pair_rows.append({
                "a": names[ai], "b": names[bi], "ta": ta, "tb": tb,
                "eng": eng, "exp": exp, "delta": eng - exp,
                "decided": decided,
            })
        pair_rows.sort(key=lambda r: abs(r["delta"]), reverse=True)
        for r in pair_rows[:args.top_pairs]:
            print(f"  {r['a'][:18]:<18}({r['ta']:<6}) vs {r['b'][:18]:<18}({r['tb']:<6})  "
                  f"engine {r['eng']*100:5.1f}%  smogon {r['exp']*100:5.1f}%  "
                  f"Δ{(r['delta'])*100:+6.1f}%  ({r['decided']} dec)")


if __name__ == "__main__":
    main()
