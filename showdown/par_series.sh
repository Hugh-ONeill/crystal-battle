#!/bin/sh
# Parallel bench series vs stock foul-play: N lanes pulling games off a QUEUE.
#
# WHY A QUEUE (measured 2026-07-22, parthru2). The static split (games_per_lane
# fixed per lane) delivered only 3.2x effective parallelism from 6 lanes: game
# length varies 6x (32..192 turns), so the run's wall clock was the unluckiest
# lane's 427s while four lanes idled for minutes. Per-TURN cost under 6-way
# concurrency is identical to sequential (0.89 vs 0.91 s/turn; server wait
# 439 vs 436 ms/turn) — the local Showdown server is NOT a bottleneck at this
# scale, and CPU sits under 3 of 24 cores. The losses are (1) lane imbalance,
# fixed here by the queue, and (2) per-game process startup, fixed by ONE
# persistent worker per lane (--team-reload rotates the team through a fixed
# per-lane file, so the worker no longer restarts per game; foul-play still
# boots per game — its side is 32MB/3.5s, not worth touching). The queue is
# also where SPRT early-stop belongs: stop handing out games once the series
# can conclude.
#
# LANE ISOLATION. Each lane needs its own username pair — a same-name login
# KICKS the running bot, which would silently corrupt every other lane. Lane k
# runs as CBGen9L<k> vs FPSpar1L<k>; both keep the CBGen9 / FPSpar1 PREFIX so
# the existing tallies (grep "Winner: CBGen9") still match unchanged. Team
# files are copied per-game too, or concurrent lanes would race on one path.
#
# Usage: par_series.sh <name> <total_games> <lanes> [--suite DIR] [our args...]
#        (arg 2 is the TOTAL game count now, not games per lane)
# Tally: grep -c "^INFO     Winner: CBGen9" showdown/bench/<name>_L*_foulplay.log
set -u
CB_ABS=/home/wiz/Developer/grimoire/crystal-battle
NAME="$1"; TOTAL="$2"; LANES="$3"; shift 3
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

QUEUE="$CB/showdown/bench/${NAME}.queue"
echo 0 > "$QUEUE"
# Hand out the next 1-based global game index, or fail when the run is done.
# flock serializes the read-increment-write; lanes calling this concurrently
# each get a distinct index exactly once.
next_game() {
  (
    flock -x 9
    n=$(($(cat "$QUEUE") + 1))
    [ "$n" -gt "$TOTAL" ] && exit 1
    echo "$n" > "$QUEUE"
    echo "$n"
  ) 9>>"$QUEUE.lock"
}

echo "=== parallel series '$NAME': $TOTAL games over $LANES lanes at $(date +%H:%M:%S) ==="

if [ "${CB_PIN_CAPS:-1}" = "1" ]; then
  CB_CAPS="--base-max-ms ${CB_SEARCH_MS:-300} --grind-max-ms ${CB_SEARCH_MS:-300}"
else
  CB_CAPS=""
fi

lane=1
while [ "$lane" -le "$LANES" ]; do
  (
    OURS_LOG="$CB/showdown/bench/${NAME}_L${lane}_ours.log"
    FP_LOG="$CB/showdown/bench/${NAME}_L${lane}_foulplay.log"
    : > "$OURS_LOG"; : > "$FP_LOG"
    US="CBGen9L${lane}"; THEM="FPSpar1L${lane}"
    # ONE persistent worker per lane (851MB of replay sets/priors/nets and
    # ~4s of startup per process — load once, play the whole lane). The team
    # rotates per game by swapping this fixed file; --team-reload makes the
    # worker re-read it at every challenge accept (/utm) and team preview.
    LANE_TEAM="$CB/showdown/bench/${NAME}_L${lane}.team"
    OURS_PID=""
    starts=0
    start_worker() {
      cd "$CB"
      .venv/bin/python showdown/gen9_player.py --local --username "$US" \
          --mode accept --format gen9ou --team "$LANE_TEAM" --team-reload on \
          --search-ms "${CB_SEARCH_MS:-300}" --set-samples 2 \
          $CB_CAPS --n-games 999 --log-level 20 \
          "$@" >> "$OURS_LOG" 2>&1 &
      OURS_PID=$!
      # Readiness probe instead of a blind sleep 5: the worker logs
      # "Starting listening" ~1s in and is logged in ms later; foul-play
      # then takes ~3s to boot before it challenges, which is grace enough.
      # The log persists across restarts, so compare the COUNT to launches.
      starts=$((starts + 1))
      i=0
      while [ "$(grep -c "Starting listening" "$OURS_LOG")" -lt "$starts" ] \
            && [ "$i" -lt 60 ]; do
        sleep 0.25; i=$((i + 1))
      done
      sleep 0.5
    }
    while g=$(next_game); do
      cd "$CB"   # each iteration starts from a known cwd (we cd to $FP below)
      if [ "$N_TEAMS" -gt 0 ]; then
        # rotate the suite by global index so coverage stays balanced no
        # matter which lane happens to pull the game
        idx=$(( (g - 1) % N_TEAMS + 1 ))
        OUR_TEAM=$(ls "$SUITE_DIR"/*.txt | sort | sed -n "${idx}p")
        BASE="G${g}_$(basename "$OUR_TEAM" .txt)"
        cp "$OUR_TEAM" "$FP/teams/teams/gen9/ou/suite/$BASE"
        FP_TEAM="gen9/ou/suite/$BASE"
      else
        OUR_TEAM="$CB/showdown/teams/gen9ou_sample.txt"
        BASE="legacy_default"; FP_TEAM="gen9/ou/sample_legal"
      fi
      cp "$OUR_TEAM" "$LANE_TEAM"
      echo "=== lane $lane game $g/$TOTAL team: $BASE ($(date +%H:%M:%S)) ===" >> "$OURS_LOG"
      echo "=== lane $lane game $g/$TOTAL team: $BASE ($(date +%H:%M:%S)) ===" >> "$FP_LOG"
      [ -z "$OURS_PID" ] && start_worker "$@"
      cd "$FP"
      timeout "$PER_GAME_TIMEOUT" .venv/bin/python run.py \
          --websocket-uri ws://localhost:8000/showdown/websocket \
          --ps-username "$THEM" --bot-mode challenge_user \
          --user-to-challenge "$US" --pokemon-format gen9ou \
          --team-name "$FP_TEAM" --search-time-ms "${FP_SEARCH_MS:-300}" \
          --run-count 1 --log-level INFO >> "$FP_LOG" 2>&1
      if [ "$?" -eq 124 ]; then
        # hung game: the worker is stuck in a battle that will never finish
        # (max_concurrent_battles=1 would wedge every later game in the
        # lane) — restart it for a clean slate. Also self-heals a crashed
        # worker: foul-play's unanswered challenge times out and lands here.
        echo "=== lane $lane game $g TIMED OUT; restarting worker ===" >> "$OURS_LOG"
        kill "$OURS_PID" 2>/dev/null
        wait "$OURS_PID" 2>/dev/null
        OURS_PID=""
      fi
    done
    if [ -n "$OURS_PID" ]; then
      kill "$OURS_PID" 2>/dev/null
      wait "$OURS_PID" 2>/dev/null
    fi
    rm -f "$LANE_TEAM"
  ) &
  lane=$((lane + 1))
done
wait
rm -f "$QUEUE" "$QUEUE.lock"

W=$(cat "$CB"/showdown/bench/${NAME}_L*_foulplay.log 2>/dev/null | grep -c "^INFO     Winner: CBGen9")
L=$(cat "$CB"/showdown/bench/${NAME}_L*_foulplay.log 2>/dev/null | grep -c "^INFO     Winner: FPSpar1")
N=$((W + L))
echo "=== '$NAME' complete at $(date +%H:%M:%S): ${W}W-${L}L of ${N} decided ==="
[ "$N" -gt 0 ] && echo "    $(echo "$W $N" | awk '{printf "%.1f%%", 100*$1/$2}')"
