#!/bin/bash
# Record until crystal-battle wins (or the takes run out). Each try relaunches
# FPAiri fresh, runs one recorded take via attempt.sh, keeps the video on WIN.
# usage: retry_take.sh [take-numbers...]   (default: 1 2 3)
set -u
HERE="$(cd "$(dirname "$0")" && pwd)"
FP=${FOULPLAY_DIR:-$HOME/Developer/grimoire/foul-play}
LOGDIR=${PRISM_LOG_DIR:-${TMPDIR:-/tmp}/prism-recording}
mkdir -p "$LOGDIR"
FPLOG=$LOGDIR/fp_retry.log
export PRISM_DATE=${PRISM_DATE:-$(date +%Y%m%d)}
export PRISM_LOG_DIR=$LOGDIR

TAKES=("$@")
[ ${#TAKES[@]} -eq 0 ] && TAKES=(1 2 3)

for N in "${TAKES[@]}"; do
  echo "=== TAKE $N ==="
  pkill -f "run.py.*FPAiri" 2>/dev/null
  sleep 3
  ( cd "$FP" && .venv/bin/python -u run.py \
      --websocket-uri ws://localhost:8000/showdown/websocket \
      --ps-username FPAiri --bot-mode accept_challenge \
      --pokemon-format gen9ou --team-name gen9/ou/suite \
      --search-time-ms 300 --run-count 1 --log-level INFO > "$FPLOG" 2>&1 & )
  ready=""
  for i in $(seq 1 40); do
    grep -q "Waiting for a gen9ou challenge" "$FPLOG" 2>/dev/null && { ready=1; break; }
    sleep 2
  done
  if [ -z "$ready" ]; then echo "FPAiri did not come up"; continue; fi

  RESULT=$("$HERE/attempt.sh" "$N" | tail -1)
  echo "$RESULT"
  if echo "$RESULT" | grep -q "TAKE-WIN"; then
    echo "WON on take $N -> airi-prism-$PRISM_DATE-take$N.mp4"
    pkill -f "run.py.*FPAiri" 2>/dev/null
    exit 0
  fi
  # discard the losing video to save space; keep transcript for reference
  rm -f "$HOME/Videos/airi-prism-$PRISM_DATE-take$N.mp4"
done
echo "no win in ${#TAKES[@]} tries (transcripts kept)"
pkill -f "run.py.*FPAiri" 2>/dev/null
