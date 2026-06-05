"""Re-weight a round-robin bench by opponent-type ladder popularity.

Raw bench win% averages all 17 opponents equally. On the real 1500 ladder you
face types in proportion to their usage, so this weights each matchup's win%
by the OPPONENT type's presence share — giving expected win% vs the real field.

  .venv/bin/python -m monotype.weighted_standings monotype/bench/<file>.txt
"""
from __future__ import annotations
import re
import sys
from collections import defaultdict

# team name -> monotype type
TYPE_OF = {
    "Dork Team": "dark", "Totally Normal Team": "normal", "Spooky Team": "ghost",
    "Supa Fitingu": "fighting", "I Believe In Fairies": "fairy",
    "Grounded Grounded Grounded": "ground", "Ice Ice Baby": "ice",
    "Heavy Metal": "steel", "Stop Bugging Me": "bug", "Electric Boogaloo": "electric",
    "Fly Away Now": "flying", "Toxic Team": "poison", "Rock Hard": "rock",
    "RuPauls Dragon Race": "dragon", "Soft and Wet": "water",
    "Your Ass is Grass": "grass", "Brain Blast": "psychic", "Disco Inferno": "fire",
}
# 1500 type presence share (May 2026, matchup-chart row-sum %) — ladder frequency
WEIGHT = {
    "dragon": 11.0, "ghost": 9.6, "water": 9.3, "dark": 8.0, "fighting": 7.0,
    "steel": 6.5, "flying": 6.4, "bug": 6.4, "ground": 5.9, "poison": 5.7,
    "fairy": 4.9, "fire": 4.8, "normal": 3.6, "rock": 2.4, "psychic": 2.3,
    "grass": 2.2, "ice": 2.1, "electric": 2.1,
}
NAMES = list(TYPE_OF)


def resolve(disp):
    """Map a (possibly truncated '…') display name back to the full team name."""
    d = disp.strip().rstrip("…").strip()
    for n in NAMES:
        if n.startswith(d) or n == disp.strip():
            return n
    return None


def main():
    path = sys.argv[1]
    text = open(path).read()
    sect = text.split("Per-pair results")[1].split("Per-team standings")[0]
    # per-team {opponent_type: [wins, losses]}
    vs = defaultdict(lambda: defaultdict(lambda: [0, 0]))
    for line in sect.splitlines():
        m = re.match(r"\s*(.+?)\s+vs\s+(.+?)\s*:\s*(\d+)W\s+(\d+)L\s+(\d+)D", line)
        if not m:
            continue
        a, b = resolve(m.group(1)), resolve(m.group(2))
        if not a or not b or a == b:
            continue
        aw, bw = int(m.group(3)), int(m.group(4))
        ta, tb = TYPE_OF[a], TYPE_OF[b]
        vs[a][tb][0] += aw; vs[a][tb][1] += bw
        vs[b][ta][0] += bw; vs[b][ta][1] += aw

    rows = []
    for team, opps in vs.items():
        raw_w = raw_l = 0
        num = den = 0.0
        for ot, (w, l) in opps.items():
            raw_w += w; raw_l += l
            dec = w + l
            if dec:
                num += WEIGHT[ot] * (w / dec)
                den += WEIGHT[ot]
        raw = 100 * raw_w / (raw_w + raw_l) if (raw_w + raw_l) else 0
        wtd = 100 * num / den if den else 0
        rows.append((wtd, raw, team))
    rows.sort(reverse=True)
    print(f"  {'team':26} {'type':8} {'raw%':>6} {'ladder-wtd%':>12}   Δ")
    print("  " + "-" * 60)
    for wtd, raw, team in rows:
        print(f"  {team:26} {TYPE_OF[team]:8} {raw:5.1f}% {wtd:11.1f}%   {wtd-raw:+5.1f}")


if __name__ == "__main__":
    main()
