"""Paired tally for interleaved A/B series (par_series --ab).

Games are dealt in PAIRS: pair k = games (2k-1, 2k), both on the same suite
team; odd global index = arm A (baseline env), even = arm B (baseline + the
--ab env spec). Because the two arms alternate through the same wall-clock
run, any run-scoped confound (the Jul-23 level step, server state, thermal,
fp-side drift) hits both arms equally and cancels in the difference — the
lesson of 2026-07-24, when every gate-vs-historical-level verdict of the era
died and only the same-night control pair survived.

The SPRT statistic is the share of arm-A wins among DISCORDANT pairs
(exactly one arm won its game). Concordant pairs carry no information about
the difference and are not scored. Under "no difference" the discordant
share is 0.5; sprt.py's existing Bernoulli machinery applies unchanged with
W = discordant pairs A won, L = discordant pairs B won.

Prefix discipline mirrors dispense_tally.py: only the contiguous prefix of
COMPLETE pairs (both games decided) counts, because completion order is
outcome-biased while dispense order is not. A draw/error game never
completes its pair, which stalls the prefix at that pair — the lag is
bounded and outcome-independent.

Output (single line):
    nA nB prefix_pairs complete_pairs wA lA wB lB
where nA/nB are discordant counts inside the prefix and wA/lA/wB/lB are
whole-run per-arm tallies (reporting, not gating).
"""

import glob
import re
import sys

GAME_RE = re.compile(r"=== lane \d+ game (\d+)/\d+ team: ")
WIN_RE = re.compile(r"Winner: (\S+)")


def outcomes_from_logs(bench_dir: str, name: str) -> dict[int, bool]:
    """Map global game index -> True if our side (CBGen9*) won."""
    out: dict[int, bool] = {}
    for path in glob.glob(f"{bench_dir}/{name}_L*_foulplay.log"):
        game = None
        for line in open(path, errors="replace"):
            m = GAME_RE.search(line)
            if m:
                game = int(m.group(1))
                continue
            w = WIN_RE.search(line)
            if w and game is not None:
                out[game] = w.group(1).startswith("CBGen9")
                game = None
    return out


def paired_counts(outcomes: dict[int, bool]):
    pairs = {}
    for g, won in outcomes.items():
        pairs.setdefault((g + 1) // 2, {})[g % 2] = won  # 1 = arm A, 0 = arm B
    complete = {k: v for k, v in pairs.items() if len(v) == 2}
    prefix = 0
    while (prefix + 1) in complete:
        prefix += 1
    n_a = sum(1 for k in range(1, prefix + 1)
              if complete[k][1] and not complete[k][0])
    n_b = sum(1 for k in range(1, prefix + 1)
              if complete[k][0] and not complete[k][1])
    w_a = sum(1 for g, won in outcomes.items() if g % 2 == 1 and won)
    l_a = sum(1 for g, won in outcomes.items() if g % 2 == 1 and not won)
    w_b = sum(1 for g, won in outcomes.items() if g % 2 == 0 and won)
    l_b = sum(1 for g, won in outcomes.items() if g % 2 == 0 and not won)
    return n_a, n_b, prefix, len(complete), w_a, l_a, w_b, l_b


if __name__ == "__main__":
    bench_dir, name = sys.argv[1], sys.argv[2]
    print(*paired_counts(outcomes_from_logs(bench_dir, name)))
