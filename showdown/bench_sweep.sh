#!/bin/bash
# Canonical multi-matchup bench sweep across the gen9 OU sample teams.
# Defaults to bench_engines.py (dev vs ref). Override BENCH=bench_algos.py
# (and pass extra flags via EXTRA="--algo-a mcts --algo-b emm") for algo bench.
#
# Usage:
#   showdown/bench_sweep.sh                                    # default 300ms, ~3 seeds/matchup
#   SEARCH_MS=1000 showdown/bench_sweep.sh                     # longer budget
#   BENCH=bench_algos.py EXTRA="--algo-a mcts --algo-b emm" showdown/bench_sweep.sh
#   OUT=/tmp/sweep.txt showdown/bench_sweep.sh                 # redirect output
#
# Matchups (team_a vs team_b, label):
#   0 vs 0  Mirror Sun (60g)
#   0 vs 3  Sun vs Stall
#   1 vs 2  BO vs Balance
#   4 vs 5  Rain vs TR
#   0 vs 6  Sun vs Screens HO       (new)
#   6 vs 3  Screens HO vs Stall     (new)
#   7 vs 4  Sand vs Rain            (new, weather conflict)
#   8 vs 5  Webs vs TR              (new, speed-control conflict)
#   9 vs 2  Kingambit HO vs Balance (new, priority pressure)

set -e
cd "$(dirname "$0")/.."

PY=${PY:-/home/wiz/Developer/grimoire/crystal-battle/.venv/bin/python}
BENCH=${BENCH:-bench_engines.py}
SEARCH_MS=${SEARCH_MS:-300}
WORKERS=${WORKERS:-22}
EXTRA=${EXTRA:-}
OUT=${OUT:-/tmp/bench_sweep.txt}

run() {
  local label="$1" team1="$2" team2="$3" games="$4"; shift 4
  local seeds=("$@")
  echo "" >> "$OUT"
  echo "=== $label (teams $team1 vs $team2) ===" >> "$OUT"
  for s in "${seeds[@]}"; do
    echo "--- seed $s ---" >> "$OUT"
    $PY showdown/$BENCH --gen 9 --team1 $team1 --team2 $team2 \
      --games $games --search-ms $SEARCH_MS --workers $WORKERS \
      --seed $s $EXTRA >> "$OUT" 2>&1
  done
}

> "$OUT"
echo "bench: $BENCH search-ms: $SEARCH_MS workers: $WORKERS extra: $EXTRA" >> "$OUT"

# Existing 4 matchups
run "MIRROR SUN"          0 0 30  1000 5000 9000
run "SUN vs STALL"        0 3 20  2000 6000 8000
run "BO vs BALANCE"       1 2 20  3000 7000 10000
run "RAIN vs TR"          4 5 20  4000 7000 11000

# New 5 matchups
run "SUN vs SCREENS HO"   0 6 20  1500 5500 9500
run "SCREENS HO vs STALL" 6 3 20  2500 6500 8500
run "SAND vs RAIN"        7 4 20  3500 7500 10500
run "WEBS vs TR"          8 5 20  4500 7700 11500
run "KINGAMBIT HO vs BAL" 9 2 20  3300 7300 10300

echo "" >> "$OUT"
echo "done" >> "$OUT"
