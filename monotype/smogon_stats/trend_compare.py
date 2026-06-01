#!/usr/bin/env python3
"""Compare gen9monotype meta across months: type-strength shifts + per-type usage movers.

Reads the vendored smogon_stats/<month>/{matchup,usage} files and prints:
  1. Per-type mean matchup win% (a proxy for type strength) with month-over-month delta.
  2. Biggest per-type Pokemon usage movers (risers/fallers) over the window.

Usage:
  .venv/bin/python monotype/smogon_stats/trend_compare.py --months 2026-02 2026-03 2026-04 --elo 1760
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path

HERE = Path(__file__).parent

TYPES = ["normal", "fighting", "flying", "poison", "ground", "rock", "bug",
         "ghost", "steel", "fire", "water", "grass", "electric", "psychic",
         "ice", "dragon", "dark", "fairy"]


def parse_matchup(path: Path) -> dict[str, float]:
    """Return {type: mean win% across non-mirror opponents}."""
    if not path.exists():
        return {}
    lines = path.read_text().splitlines()
    # column header order is the row order: find the header line of type names
    col_types = None
    rows: dict[str, list[float]] = {}
    cur = None
    for ln in lines:
        ln = ln.strip()
        if not ln.startswith("|"):
            continue
        cells = [c.strip() for c in ln.strip("|").split("|")]
        if col_types is None and cells[0] == "" and cells[1] in TYPES:
            col_types = cells[1:]
            continue
        if col_types is None:
            continue
        # a type row starts with the type name in cell 0; its win% line has empty cell 0
        if cells[0] in TYPES:
            cur = cells[0]
            continue
        if cur and cells[0] == "" and "%" in ln:
            wins = []
            for c, t in zip(cells[1:], col_types):
                m = re.match(r"([\d.]+)%", c)
                if m and t != cur:  # skip mirror
                    wins.append(float(m.group(1)))
            if wins:
                rows[cur] = sum(wins) / len(wins)
            cur = None
    return rows


def parse_usage(path: Path) -> dict[str, float]:
    """Return {pokemon: weighted usage %} from a per-type usage file."""
    if not path.exists():
        return {}
    out = {}
    for ln in path.read_text().splitlines():
        ln = ln.strip()
        if not ln.startswith("|"):
            continue
        cells = [c.strip() for c in ln.strip("|").split("|")]
        if len(cells) < 3 or not cells[0].isdigit():
            continue
        m = re.match(r"([\d.]+)%", cells[2])
        if m:
            out[cells[1]] = float(m.group(1))
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--months", nargs="+", required=True)
    p.add_argument("--elo", type=int, default=1760)
    p.add_argument("--top", type=int, default=3, help="movers to show per type")
    args = p.parse_args()
    first, last = args.months[0], args.months[-1]

    # --- type strength ---
    strength = {m: parse_matchup(HERE / m / "matchup" / f"gen9monotype-matchup_chart-{args.elo}.txt")
                for m in args.months}
    print(f"=== TYPE STRENGTH (mean matchup win%, elo {args.elo}) ===")
    print(f"{'type':9} " + " ".join(f"{m[-2:]:>7}" for m in args.months) + f"   Δ({first[-2:]}->{last[-2:]})")
    rank = sorted(TYPES, key=lambda t: strength[last].get(t, 0), reverse=True)
    for t in rank:
        vals = [strength[m].get(t) for m in args.months]
        cells = " ".join(f"{v:6.1f}%" if v is not None else "   -  " for v in vals)
        d = (vals[-1] - vals[0]) if (vals[0] is not None and vals[-1] is not None) else None
        ds = f"{d:+5.1f}" if d is not None else "  -"
        flag = "  <<" if d is not None and d >= 2.0 else ("  >>" if d is not None and d <= -2.0 else "")
        print(f"{t:9} {cells}   {ds}{flag}")

    # --- per-type usage movers ---
    print(f"\n=== PER-TYPE USAGE MOVERS ({first} -> {last}, elo {args.elo}) ===")
    for t in TYPES:
        u0 = parse_usage(HERE / first / "usage" / f"gen9monotype-mono{t}-{args.elo}.txt")
        u1 = parse_usage(HERE / last / "usage" / f"gen9monotype-mono{t}-{args.elo}.txt")
        if not u0 or not u1:
            continue
        deltas = sorted(((u1.get(k, 0) - u0.get(k, 0), k) for k in set(u0) | set(u1)),
                        key=lambda x: x[0])
        risers = [d for d in reversed(deltas) if d[0] >= 3.0][:args.top]
        fallers = [d for d in deltas if d[0] <= -3.0][:args.top]
        if not risers and not fallers:
            continue
        r = ", ".join(f"{k} {d:+.0f}" for d, k in risers) or "-"
        f = ", ".join(f"{k} {d:+.0f}" for d, k in fallers) or "-"
        print(f"  mono{t:9} up: {r:50}  down: {f}")


if __name__ == "__main__":
    main()
