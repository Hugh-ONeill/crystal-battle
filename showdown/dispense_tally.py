"""Dispense-order W/L tally for the par_series SPRT gate.

Completion-order tallies are biased: short games decide first and short
games skew losses (loss-trace: we lose short), so verdicts fired early in a
run lean accept-h0 — observed live on the 2026-07-22 certification gate,
which concluded at n=30 while every game longer than the elapsed four
minutes was still in flight.

Fix: consume outcomes in DISPENSE order. Each lane's foul-play log carries
the game headers (with the global dispense index) and winner lines in
sequence, so the j-th winner in a lane belongs to the j-th header. The SPRT
tally is then the longest CONTIGUOUS decided prefix of dispense indices
1..m: dispense order is outcome-independent, so the statistic follows the
exact path a purely sequential run would produce, observed with lag. A
decided game beyond an in-flight index waits until the prefix reaches it.

A dispensed game with no winner that its lane has already moved past
(timeout or crash) is a no-decision: skipped, the prefix continues. A
lane's LAST header with no winner is treated as in flight and blocks the
prefix — the conservative reading while a series runs.

Usage: dispense_tally.py <bench_dir> <name> [us_prefix] [them_prefix]
Prints four numbers: "<W> <L> <prefix_end> <decided_total>"
"""
import re
import sys
from pathlib import Path

HDR = re.compile(r"^=== lane \d+ game (\d+)/\d+ team: ")


def lane_outcomes(path, us, them):
    """[(global_index, 'W'|'L'|None), ...] in this lane's play order."""
    games = []
    for line in open(path, errors="replace"):
        m = HDR.match(line)
        if m:
            games.append([int(m.group(1)), None])
        elif line.startswith("INFO     Winner: ") and games:
            name = line.split("Winner:", 1)[1].strip()
            if games[-1][1] is None:
                if name.startswith(us):
                    games[-1][1] = "W"
                elif name.startswith(them):
                    games[-1][1] = "L"
    return games


def prefix_tally(bench_dir, name, us="CBGen9", them="FPSpar1"):
    outcome = {}  # global index -> 'W' | 'L' | 'skip' | 'inflight'
    for path in Path(bench_dir).glob(f"{name}_L*_foulplay.log"):
        games = lane_outcomes(path, us, them)
        for i, (g, res) in enumerate(games):
            outcome[g] = res or ("inflight" if i == len(games) - 1 else "skip")
    wins = losses = prefix_end = 0
    g = 1
    while outcome.get(g) not in (None, "inflight"):
        if outcome[g] == "W":
            wins += 1
        elif outcome[g] == "L":
            losses += 1
        prefix_end = g
        g += 1
    decided = sum(1 for v in outcome.values() if v in ("W", "L"))
    return wins, losses, prefix_end, decided


if __name__ == "__main__":
    us = sys.argv[3] if len(sys.argv) > 3 else "CBGen9"
    them = sys.argv[4] if len(sys.argv) > 4 else "FPSpar1"
    print(*prefix_tally(sys.argv[1], sys.argv[2], us, them))
