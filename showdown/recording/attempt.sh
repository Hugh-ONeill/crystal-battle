#!/bin/bash
# One recording attempt, v3 (overlay era): Prism commentary renders on the
# tiled kitty panel (showdown/overlay_start.sh) and wf-recorder captures the
# whole workspace. Replaces the v2 relay flow: no Showdown login, no server
# patch, and NO fullscreen — arrange the battle-browser + prism-panel split
# once before recording; that split IS the broadcast frame.
# usage: attempt.sh <take-number>   prints TAKE-WIN or TAKE-LOSS (or HEALTH-FAIL)
set -u
N=$1
HERE="$(cd "$(dirname "$0")" && pwd)"
CB="$(cd "$HERE/../.." && pwd)"
PY=$CB/.venv/bin/python
DATE=${PRISM_DATE:-$(date +%Y%m%d)}
LOGDIR=${PRISM_LOG_DIR:-${TMPDIR:-/tmp}/prism-recording}
VID=$HOME/Videos/airi-prism-$DATE-take$N.mp4
TRANS=$HOME/Videos/airi-prism-$DATE-take$N-transcript.txt
AUDIO=${PRISM_AUDIO:-alsa_output.pci-0000_15_00.6.iec958-stereo.monitor}
OUTPUT=${PRISM_OUTPUT:-DP-4}
mkdir -p "$LOGDIR" "$HOME/Videos"

log() { echo "[attempt$N] $*"; }

# ---- clear leftovers from any previous attempt
pkill -f "caster_bridge.py --watch" 2>/dev/null
pkill -f "gen9_player.py.*CBAiri" 2>/dev/null
sleep 1

# ---- health gate: a test beat must come back as a generated line.
# Replaces the AIRI-era gate, which cleared AIRI's chat session (it resent the
# whole history per request and overflowed the context) and then watched
# AIRI's own log for the "no consumer" registration race. Neither applies to
# the caster: it holds no durable session (speak() clears the duo transcript
# on MATCH START) and it has no consumer handshake. The only thing worth
# gating on is the thing we actually need — does a beat produce a line.
gate() {
  local REPLY_LOG="$LOGDIR/gate$N.log"
  rm -f "$REPLY_LOG"
  (cd "$CB" && exec $PY -u showdown/caster_bridge.py --watch \
      --url "${PRISM_CASTER_URL:-ws://127.0.0.1:8131/ws}" \
      > "$REPLY_LOG" 2>&1) &
  local WATCH=$!
  sleep 2
  (cd "$CB" && $PY showdown/caster_bridge.py \
      --url "${PRISM_CASTER_URL:-ws://127.0.0.1:8131/ws}" \
      "[bridge-test] pre-take health gate $N" >/dev/null 2>&1)
  local i
  for i in $(seq 1 40); do          # cold gemma load can take 15-60s
    sleep 2
    if [ -s "$REPLY_LOG" ] && grep -q "[[:alpha:]]" "$REPLY_LOG"; then
      kill $WATCH 2>/dev/null; echo OK; return
    fi
  done
  kill $WATCH 2>/dev/null
  echo NOGEN
}
G=$(gate)
case "$G" in
  OK) log health-ok ;;
  *)  log HEALTH-FAIL-NOGEN; exit 1 ;;
esac

# ---- commentary panel stack (feed server + tiled kitty panel; idempotent)
bash "$CB/showdown/overlay_start.sh"

# ---- transcript tap: keeps a text record AND is the [RESULT] detector
(cd "$CB" && exec $PY -u showdown/caster_bridge.py --watch 2>&1 | tee "$TRANS") &
sleep 2

# ---- find (or open) the Showdown Firefox window and raise it in its tile
ADDR=$(hyprctl clients -j | $PY -c "
import json, sys
for c in json.load(sys.stdin):
    if 'firefox' in c.get('class', '').lower():
        t = c.get('title', '')
        if 'Showdown' in t or ' vs. ' in t:
            print(c['address']); break")
if [ -z "$ADDR" ]; then
  hyprctl dispatch exec -- "firefox --new-window http://localhost:8000" >/dev/null
  sleep 5
  ADDR=$(hyprctl clients -j | $PY -c "
import json, sys
for c in json.load(sys.stdin):
    if 'firefox' in c.get('class', '').lower() and 'Showdown' in c.get('title', ''):
        print(c['address']); break")
fi
if [ -z "$ADDR" ]; then log NO-SHOWDOWN-WINDOW; exit 1; fi
hyprctl dispatch focuswindow "address:$ADDR" >/dev/null
sleep 1
log "showdown window raised ($ADDR)"

# ---- recorder (whole output: browser tile + prism-panel tile)
wf-recorder -o "$OUTPUT" --audio="$AUDIO" -f "$VID" --overwrite >/dev/null 2>&1 &
REC_PID=$!
sleep 2
log recorder-rolling

# ---- match (headless runner) + spectator navigation
"$HERE/run_match.sh" > "$LOGDIR/runner_take$N.log" 2>&1 &
ROOM=$($PY "$HERE/room_id.py")
log "room: $ROOM"
if [ "$ROOM" = "NO-BATTLE-FOUND" ]; then
  kill -INT $REC_PID 2>/dev/null
  log NO-BATTLE; exit 1
fi
hyprctl dispatch focuswindow "address:$ADDR" >/dev/null
hyprctl dispatch exec -- "firefox http://localhost:8000/$ROOM" >/dev/null

# ---- wait for the result beat (cap 25 min); leave time for the wrap-up
# line to land on the panel on camera, then stop
SECONDS=0
until grep -q "\[RESULT\]" "$TRANS" 2>/dev/null || [ $SECONDS -gt 1500 ]; do sleep 5; done
sleep 30
kill -INT $REC_PID 2>/dev/null
sleep 3
pkill -f "caster_bridge.py --watch" 2>/dev/null

if grep -q "\[RESULT\] WIN" "$TRANS" 2>/dev/null; then
  log TAKE-WIN
else
  log TAKE-LOSS
fi
