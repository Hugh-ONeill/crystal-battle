#!/bin/sh
# Batched A/B series vs stock foul-play on the local Showdown server.
#
# Runs BATCHES sequential batches of GAMES games, with FRESH bot processes
# per batch and a hard per-batch timeout. Motivation: a 200-game series in
# one process pair wedged silently (series 11: poke-env stopped dispatching
# battle requests after a timer-lossed game, idle main thread, no traceback).
# Fresh processes bound the blast radius of any hang to one batch, and the
# timeout skips a stuck batch instead of stalling the series.
#
# Usage: ab_series.sh <name> <batches> <games_per_batch> [--suite DIR] [our extra args...]
# Logs:  showdown/bench/<name>_ours.log / <name>_foulplay.log (appended)
# Tally: grep -c "^INFO     Winner: CBGen9" <name>_foulplay.log
#
# --suite DIR: frozen-suite mode. Batch i is a MIRROR of the i-th team file
# (sorted, cycling) from DIR — both bots play the same team, so team strength
# cancels and the pooled result measures piloting across the whole suite.
# Per-batch team markers land in both logs for per-matchup breakdowns.

set -u
NAME="$1"; BATCHES="$2"; GAMES="$3"; shift 3
SUITE_DIR=""
if [ "${1:-}" = "--suite" ]; then
  SUITE_DIR="$2"; shift 2
fi
# unbuffered, or decision prints hide in stdout buffers and diagnostics lie
export PYTHONUNBUFFERED=1
CB=/home/wiz/Developer/grimoire/crystal-battle
FP=/home/wiz/Developer/grimoire/foul-play
OURS_LOG="$CB/showdown/bench/${NAME}_ours.log"
FP_LOG="$CB/showdown/bench/${NAME}_foulplay.log"
# per-GAME hard timeout (one game per process now). Default covers a 300ms
# stall game (~6 min) with margin; 5s runs set PER_GAME_TIMEOUT higher
# (stall at 5s/turn ~1 hr). A wedged game is killed and skipped here.
PER_GAME_TIMEOUT="${PER_GAME_TIMEOUT:-1200}"

# fail loudly if the local Showdown server is down, instead of burning every
# batch on connection-refused (series 13 first attempt)
if ! "$CB/.venv/bin/python" -c \
    "import socket; socket.create_connection(('127.0.0.1', 8000), 2).close()" \
    2>/dev/null; then
  echo "FATAL: no Showdown server listening on :8000" >&2
  exit 1
fi

if [ -n "$SUITE_DIR" ]; then
  # absolute always: the loop cd's between repos, so a relative path that
  # resolves at parse time silently stops resolving from batch 2 on
  case "$SUITE_DIR" in
    /*) ;;
    *) SUITE_DIR="$CB/$SUITE_DIR" ;;
  esac
  N_TEAMS=$(ls "$SUITE_DIR"/*.txt | wc -l)
  mkdir -p "$FP/teams/teams/gen9/ou/suite"
fi

i=1
while [ "$i" -le "$BATCHES" ]; do
  if [ -n "$SUITE_DIR" ]; then
    idx=$(( (i - 1) % N_TEAMS + 1 ))
    OUR_TEAM=$(ls "$SUITE_DIR"/*.txt | sort | sed -n "${idx}p")
    TEAM_BASE=$(basename "$OUR_TEAM" .txt)
    cp "$OUR_TEAM" "$FP/teams/teams/gen9/ou/suite/$TEAM_BASE"
    FP_TEAM="gen9/ou/suite/$TEAM_BASE"
  else
    OUR_TEAM="$CB/showdown/teams/gen9ou_sample.txt"
    TEAM_BASE="legacy_default"
    FP_TEAM="gen9/ou/sample_legal"
  fi
  echo "=== batch $i/$BATCHES team: $TEAM_BASE ($(date +%H:%M:%S)) ===" >> "$OURS_LOG"
  echo "=== batch $i/$BATCHES team: $TEAM_BASE ($(date +%H:%M:%S)) ===" >> "$FP_LOG"
  # ONE GAME PER PROCESS: poke-env's accept_challenges(N) loop wedges between
  # games (challenge N+1 received but never accepted — root cause of every
  # dispatch freeze this project). Playing one game per fresh process removes
  # the between-games step entirely; a wedge can only cost the current game,
  # which the per-game timeout kills and skips. ~10s/game process overhead,
  # negligible vs 30s-5min games.
  # CB_PIN_CAPS=1 (default): pin base/grind budget caps to CB_SEARCH_MS so the
  # A/B is equal-budget (measures search QUALITY at fixed compute). Set 0 to
  # let the full budget-by-clock + grind package engage on the local challenge
  # timer's bank — the deployment config (unequal wall-time, but the historical
  # control says uniform extra time didn't help vs foul-play, so a win here
  # implicates the grind tools, not raw clock).
  if [ "${CB_PIN_CAPS:-1}" = "1" ]; then
    CB_CAPS="--base-max-ms ${CB_SEARCH_MS:-300} --grind-max-ms ${CB_SEARCH_MS:-300}"
  else
    CB_CAPS=""
  fi
  g=1
  while [ "$g" -le "$GAMES" ]; do
    cd "$CB"
    .venv/bin/python showdown/gen9_player.py --local --username CBGen9 \
        --mode accept --format gen9ou --team "$OUR_TEAM" \
        --search-ms "${CB_SEARCH_MS:-300}" --set-samples 2 \
        $CB_CAPS \
        --n-games 1 --log-level 20 \
        "$@" >> "$OURS_LOG" 2>&1 &
    OURS_PID=$!
    sleep 5
    cd "$FP"
    timeout "$PER_GAME_TIMEOUT" .venv/bin/python run.py \
        --websocket-uri ws://localhost:8000/showdown/websocket \
        --ps-username FPSpar1 --bot-mode challenge_user \
        --user-to-challenge CBGen9 --pokemon-format gen9ou \
        --team-name "$FP_TEAM" --search-time-ms "${FP_SEARCH_MS:-300}" \
        --run-count 1 --log-level INFO >> "$FP_LOG" 2>&1
    if [ "$?" -eq 124 ]; then
      echo "=== batch $i game $g TIMED OUT (skipping) ===" >> "$FP_LOG"
    fi
    kill "$OURS_PID" 2>/dev/null
    wait "$OURS_PID" 2>/dev/null
    g=$((g + 1))
  done
  i=$((i + 1))
done
echo "=== series $NAME complete ===" >> "$FP_LOG"
