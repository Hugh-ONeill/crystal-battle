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
# Usage: ab_series.sh <name> <batches> <games_per_batch> [our extra args...]
# Logs:  showdown/bench/<name>_ours.log / <name>_foulplay.log (appended)
# Tally: grep -c "^INFO     Winner: CBGen9" <name>_foulplay.log

set -u
NAME="$1"; BATCHES="$2"; GAMES="$3"; shift 3
# unbuffered, or decision prints hide in stdout buffers and diagnostics lie
export PYTHONUNBUFFERED=1

# fail loudly if the local Showdown server is down, instead of burning every
# batch on connection-refused (series 13 first attempt)
if ! nc -z localhost 8000 2>/dev/null; then
  echo "FATAL: no Showdown server listening on :8000" >&2
  exit 1
fi
CB=/home/wiz/Developer/grimoire/crystal-battle
FP=/home/wiz/Developer/grimoire/foul-play
OURS_LOG="$CB/showdown/bench/${NAME}_ours.log"
FP_LOG="$CB/showdown/bench/${NAME}_foulplay.log"
# generous: games run ~30s; allow 120s/game + 5 min slack per batch
BATCH_TIMEOUT=$((GAMES * 120 + 300))

i=1
while [ "$i" -le "$BATCHES" ]; do
  echo "=== batch $i/$BATCHES ($(date +%H:%M:%S)) ===" >> "$OURS_LOG"
  echo "=== batch $i/$BATCHES ($(date +%H:%M:%S)) ===" >> "$FP_LOG"
  cd "$CB"
  .venv/bin/python showdown/gen9_player.py --local --username CBGen9 \
      --mode accept --format gen9ou --team showdown/teams/gen9ou_sample.txt \
      --search-ms 300 --set-samples 2 --n-games "$GAMES" --log-level 20 \
      "$@" >> "$OURS_LOG" 2>&1 &
  OURS_PID=$!
  sleep 8
  cd "$FP"
  timeout "$BATCH_TIMEOUT" .venv/bin/python run.py \
      --websocket-uri ws://localhost:8000/showdown/websocket \
      --ps-username FPSpar1 --bot-mode challenge_user \
      --user-to-challenge CBGen9 --pokemon-format gen9ou \
      --team-name gen9/ou/sample_legal --search-time-ms 300 \
      --run-count "$GAMES" --log-level INFO >> "$FP_LOG" 2>&1
  FP_STATUS=$?
  kill "$OURS_PID" 2>/dev/null
  wait "$OURS_PID" 2>/dev/null
  if [ "$FP_STATUS" -eq 124 ]; then
    echo "=== batch $i TIMED OUT (skipping ahead) ===" >> "$FP_LOG"
  fi
  i=$((i + 1))
done
echo "=== series $NAME complete ===" >> "$FP_LOG"
