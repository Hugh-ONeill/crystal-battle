#!/bin/sh
# Parallel bench series vs stock foul-play: N concurrent game LANES.
#
# WHY. The bench noise floor is ~15pp (the same config produced 22.2% and
# 37.5% in two separate arms), so resolving anything below ~20pp needs
# hundreds of games per arm — and we were running exactly ONE game at a time
# on a 24-core box, at ~60 games/hour. A game pair costs ~2-3 cores (load
# averaged ~1.5-2 during the sequential sweeps), so 6-8 lanes fit comfortably
# and turn "1300 games per arm = 22 hours" into roughly three.
#
# Throughput is the binding constraint on what is answerable here; a smarter
# stopping rule (SPRT) only makes each answer cheaper at the margin. Do this
# first, layer SPRT on top second.
#
# LANE ISOLATION. Each lane needs its own username pair — a same-name login
# KICKS the running bot, which would silently corrupt every other lane. Lane k
# runs as CBGen9L<k> vs FPSpar1L<k>; both keep the CBGen9 / FPSpar1 PREFIX so
# the existing tallies (grep "Winner: CBGen9") still match unchanged. Team
# files are copied per-lane too, or concurrent lanes would race on one path.
#
# Usage: par_series.sh <name> <games_per_lane> <lanes> [--suite DIR] [our args...]
# Tally: grep -c "^INFO     Winner: CBGen9" showdown/bench/<name>_L*_foulplay.log
set -u
CB_ABS=/home/wiz/Developer/grimoire/crystal-battle
NAME="$1"; PER_LANE="$2"; LANES="$3"; shift 3
SUITE_DIR=""
if [ "${1:-}" = "--suite" ]; then SUITE_DIR="$2"; shift 2; fi
# MUST be absolute: each game cd's into $FP to run foul-play, so a relative
# suite path stops resolving from game 2 onward — which silently produced an
# empty team name and killed every game after the first in each lane.
case "$SUITE_DIR" in
  ""|/*) ;;
  *) SUITE_DIR="$CB_ABS/$SUITE_DIR" ;;
esac

export PYTHONUNBUFFERED=1
CB=/home/wiz/Developer/grimoire/crystal-battle
FP=/home/wiz/Developer/grimoire/foul-play
PER_GAME_TIMEOUT="${PER_GAME_TIMEOUT:-1200}"
cd "$CB" || exit 1

if ! "$CB/.venv/bin/python" -c \
    "import socket; socket.create_connection(('127.0.0.1', 8000), 2).close()" \
    2>/dev/null; then
  echo "FATAL: local Showdown server not up on :8000" >&2; exit 1
fi
if pgrep -f "gen9_player[.]py .*--mode accept" >/dev/null 2>&1; then
  echo "FATAL: a bench series is already running" >&2; exit 1
fi

if [ -n "$SUITE_DIR" ]; then
  N_TEAMS=$(ls "$SUITE_DIR"/*.txt | wc -l)
  mkdir -p "$FP/teams/teams/gen9/ou/suite"
else
  N_TEAMS=0
fi

echo "=== parallel series '$NAME': $LANES lanes x $PER_LANE games "
echo "    (= $((LANES * PER_LANE)) games) at $(date +%H:%M:%S) ==="

lane=1
while [ "$lane" -le "$LANES" ]; do
  (
    OURS_LOG="$CB/showdown/bench/${NAME}_L${lane}_ours.log"
    FP_LOG="$CB/showdown/bench/${NAME}_L${lane}_foulplay.log"
    : > "$OURS_LOG"; : > "$FP_LOG"
    US="CBGen9L${lane}"; THEM="FPSpar1L${lane}"
    g=1
    while [ "$g" -le "$PER_LANE" ]; do
      cd "$CB"   # each iteration starts from a known cwd (we cd to $FP below)
      if [ "$N_TEAMS" -gt 0 ]; then
        # stagger lanes across the suite so coverage stays balanced
        idx=$(( ((lane - 1) + (g - 1) * LANES) % N_TEAMS + 1 ))
        OUR_TEAM=$(ls "$SUITE_DIR"/*.txt | sort | sed -n "${idx}p")
        BASE="L${lane}_$(basename "$OUR_TEAM" .txt)"
        cp "$OUR_TEAM" "$FP/teams/teams/gen9/ou/suite/$BASE"
        FP_TEAM="gen9/ou/suite/$BASE"
      else
        OUR_TEAM="$CB/showdown/teams/gen9ou_sample.txt"
        BASE="legacy_default"; FP_TEAM="gen9/ou/sample_legal"
      fi
      echo "=== lane $lane game $g/$PER_LANE team: $BASE ($(date +%H:%M:%S)) ===" >> "$OURS_LOG"
      echo "=== lane $lane game $g/$PER_LANE team: $BASE ($(date +%H:%M:%S)) ===" >> "$FP_LOG"
      if [ "${CB_PIN_CAPS:-1}" = "1" ]; then
        CB_CAPS="--base-max-ms ${CB_SEARCH_MS:-300} --grind-max-ms ${CB_SEARCH_MS:-300}"
      else
        CB_CAPS=""
      fi
      cd "$CB"
      .venv/bin/python showdown/gen9_player.py --local --username "$US" \
          --mode accept --format gen9ou --team "$OUR_TEAM" \
          --search-ms "${CB_SEARCH_MS:-300}" --set-samples 2 \
          $CB_CAPS --n-games 1 --log-level 20 \
          "$@" >> "$OURS_LOG" 2>&1 &
      OURS_PID=$!
      sleep 5
      cd "$FP"
      timeout "$PER_GAME_TIMEOUT" .venv/bin/python run.py \
          --websocket-uri ws://localhost:8000/showdown/websocket \
          --ps-username "$THEM" --bot-mode challenge_user \
          --user-to-challenge "$US" --pokemon-format gen9ou \
          --team-name "$FP_TEAM" --search-time-ms "${FP_SEARCH_MS:-300}" \
          --run-count 1 --log-level INFO >> "$FP_LOG" 2>&1
      kill "$OURS_PID" 2>/dev/null
      wait "$OURS_PID" 2>/dev/null
      g=$((g + 1))
    done
  ) &
  lane=$((lane + 1))
done
wait

W=$(cat "$CB"/showdown/bench/${NAME}_L*_foulplay.log 2>/dev/null | grep -c "^INFO     Winner: CBGen9")
L=$(cat "$CB"/showdown/bench/${NAME}_L*_foulplay.log 2>/dev/null | grep -c "^INFO     Winner: FPSpar1")
N=$((W + L))
echo "=== '$NAME' complete at $(date +%H:%M:%S): ${W}W-${L}L of ${N} decided ==="
[ "$N" -gt 0 ] && echo "    $(echo "$W $N" | awk '{printf "%.1f%%", 100*$1/$2}')"
