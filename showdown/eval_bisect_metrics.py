"""Behavioral metrics for one eval-bisect arm: Gliscor's move mix (the
flagship pressure-suppression signal) plus side-wide housekeeping-vs-pressure
click rates and the (noise-level, recorded-anyway) winrate.

Usage: eval_bisect_metrics.py <series_name>   # reads showdown/bench/<name>_L*
"""
import glob
import sys
from collections import Counter, defaultdict

BENCH = "/home/wiz/Developer/grimoire/crystal-battle/showdown/bench"
KEY = ("Substitute", "Toxic", "Defog", "Rest", "Protect", "Spikes",
       "Stealth Rock", "Swords Dance")


def run(name):
    gliscor = defaultdict(Counter)
    keymoves = defaultdict(Counter)
    winners = Counter()
    for f in glob.glob(f"{BENCH}/{name}_L*_ours.log"):
        role = {}
        for line in open(f, errors="replace"):
            if line.find("|") < 0:
                continue
            p = line.rstrip("\n").split("|")[1:]
            if not p:
                continue
            if p[0] == "player" and len(p) >= 3 and p[2]:
                role[p[1]] = "cb" if p[2].startswith("CBGen9") else "fp"
            elif p[0] == "move" and len(p) >= 3:
                side = role.get(p[1][:2])
                if not side:
                    continue
                if "Gliscor" in p[1]:
                    gliscor[side][p[2]] += 1
                if p[2] in KEY:
                    keymoves[side][p[2]] += 1
            elif p[0] == "win":
                winners["cb" if p[1].startswith("CBGen9") else "fp"] += 1

    n = sum(winners.values()) or 1
    print(f"[{name}] {winners['cb']}W-{winners['fp']}L of {n} "
          f"({100 * winners['cb'] / n:.0f}% — noise-level, do not interpret)")
    for side in ("cb", "fp"):
        tot = sum(gliscor[side].values()) or 1
        sub = 100 * gliscor[side]["Substitute"] / tot
        tox = 100 * gliscor[side]["Toxic"] / tot
        pro = 100 * gliscor[side]["Protect"] / tot
        print(f"[{name}] {side} gliscor: {tot} moves, Sub {sub:.1f}% "
              f"Toxic {tox:.1f}% Protect {pro:.1f}%")
    row = "  ".join(f"{m}:{keymoves['cb'][m]}/{keymoves['fp'][m]}" for m in KEY)
    print(f"[{name}] key clicks cb/fp: {row}")


if __name__ == "__main__":
    run(sys.argv[1])
