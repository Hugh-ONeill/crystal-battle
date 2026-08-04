#!/bin/bash
# Bug-hunt runner: N games over VARIED team pairings, text only.
#
# Not attempt.sh. That script exists to produce a watchable take, and most of
# it is in service of the video — kiosk browser, headless wlroots output,
# ffmpeg, the presentation clock, the null sink. All of that is fragile
# (three separate hunts died on it) and none of it tells us whether the
# commentary is CORRECT. This runner keeps the parts that generate lines and
# drops the parts that photograph them.
#
# What is deliberately different from a take:
#
#   * VARIED TEAMS. attempt.sh pins one pairing so the demo looks the same
#     every time; a bug hunt wants the opposite. Both sides rotate, including
#     one mirror — species appearing on BOTH teams is what produced the
#     preview-lead guard's false positive (2d4cfeb), so it is a case worth
#     running on purpose rather than waiting to meet.
#   * OWN PORT (8141). A demo caster lives on 8131. Sharing it is how takes
#     72-75 "validated" six guards against a caster booted hours earlier.
#   * THE CASTER IS ALWAYS FRESH, never reused, and dies with the hunt. The
#     whole point is to exercise code that was written today.
#   * NO TTS. Barge-in needs a clip in flight, so its rate CANNOT be measured
#     here — that is a speech-mode question and needs a speech-mode run.
#
# usage: hunt.sh [n-games]     (default: all pairings below)
set -u
HERE="$(cd "$(dirname "$0")" && pwd)"
CB="$(cd "$HERE/../.." && pwd)"
BC="${CRYSTAL_BROADCAST:-$HOME/Developer/grimoire/crystal-broadcast}"
PY=$CB/.venv/bin/python
FP=${FOULPLAY_DIR:-$HOME/Developer/grimoire/foul-play}
DATE=$(date +%Y%m%d-%H%M%S)
OUT=${HUNT_OUT:-$HOME/Documents/Videos/hunt-$DATE}
PORT=${HUNT_PORT:-8141}
URL="ws://127.0.0.1:$PORT/ws"
SEARCH_MS=${HUNT_SEARCH_MS:-2000}
OPP_MS=${HUNT_OPP_MS:-300}
PACE=${HUNT_TURN_PACE:-8}
GAME_TIMEOUT=${HUNT_GAME_TIMEOUT:-900}

# ours|theirs. Rotated on both sides; the last is a deliberate mirror.
PAIRINGS=(
  "07_kommoo_offense|03_dnite_tinglu_balance"
  "02_zarude_sun|08_gargtreads"
  "04_cinderace_kyurem_bo|05_samu_bliss_fat"
  "08_gargtreads|07_kommoo_offense"
  "03_dnite_tinglu_balance|04_cinderace_kyurem_bo"
  "05_samu_bliss_fat|02_zarude_sun"
  "07_kommoo_offense|06_clodsire_gliscor_stall"
  "01_legacy_mirror|01_legacy_mirror"
)
[ $# -ge 1 ] && PAIRINGS=("${PAIRINGS[@]:0:$1}")

mkdir -p "$OUT"
log() { echo "[hunt] $*"; }
up()  { ss -tln 2>/dev/null | grep -q ":$1 "; }

CASTER_PID=""; FP_PID=""; CB_PID=""; TAP_PID=""
# KILL BY PID, NEVER BY PATTERN. pkill -f gen9_player.py matches the user's
# bench lanes and the nightly ladder bot; it has destroyed two A/B series.
cleanup() {
  for p in $TAP_PID $CB_PID $FP_PID $CASTER_PID; do
    [ -n "$p" ] && kill "$p" 2>/dev/null
  done
  sleep 1
  for p in $TAP_PID $CB_PID $FP_PID $CASTER_PID; do
    [ -n "$p" ] && kill -9 "$p" 2>/dev/null
  done
}
trap cleanup EXIT INT TERM

# ---- preflight
up 8000 || { log "NO-SHOWDOWN — local server is not on :8000"; exit 1; }
up "$PORT" && { log "PORT-BUSY — something already holds :$PORT"; exit 1; }
BC_SHA=$(cd "$BC" && git rev-parse --short HEAD 2>/dev/null || echo unknown)
CB_SHA=$(cd "$CB" && git rev-parse --short HEAD 2>/dev/null || echo unknown)
log "crystal-broadcast $BC_SHA | crystal-battle $CB_SHA | out $OUT"
{ echo "broadcast=$BC_SHA"; echo "battle=$CB_SHA"; echo "date=$DATE"; } \
  > "$OUT/shas.txt"

# ---- caster, fresh, ours, and short-lived.
# CB_SHEET_LOG dumps the board each line was written FROM. The transcript
# holds the beat and the line and nothing in between, so without this an
# audit can only ask "does this claim look wrong" instead of "does this
# claim contradict the board the model was actually given" — which is the
# only question that tells us whether the sheet is doing its job.
( cd "$BC" && CB_SHEET_LOG="$OUT/sheets.txt" exec $PY \
    crystal_broadcast/caster.py --port "$PORT" \
    </dev/null >"$OUT/caster.log" 2>&1 ) &
CASTER_PID=$!
for i in $(seq 1 60); do up "$PORT" && break; sleep 1; done
up "$PORT" || { log CASTER-DOWN; exit 1; }
log "caster up on :$PORT (pid $CASTER_PID)"

# ---- health gate: a persona line, not merely a connection. The tap's own
# banner satisfied the old [[:alpha:]] check about two seconds in, which is
# how a caster that could not generate at all passed as healthy.
( cd "$BC" && exec $PY -m crystal_broadcast.caster_bridge --watch --url "$URL" \
    >"$OUT/gate.log" 2>&1 ) &
GATE_TAP=$!
sleep 2
( cd "$BC" && $PY -m crystal_broadcast.caster_bridge --url "$URL" \
    "[bridge-test] hunt health gate" >/dev/null 2>&1 )
GATE=NOGEN
for i in $(seq 1 45); do
  sleep 2
  if grep -qE "^[[:space:]]*(Prism|Fracture):" "$OUT/gate.log" 2>/dev/null; then
    GATE=OK; break
  fi
done
kill $GATE_TAP 2>/dev/null
[ "$GATE" = OK ] || { log HEALTH-FAIL-NOGEN; exit 1; }
log "health ok"

# ---- the games
i=0
for pair in "${PAIRINGS[@]}"; do
  i=$((i + 1))
  OURS="${pair%%|*}"; THEIRS="${pair##*|}"
  CBNAME="CBHunt$i"; FPNAME="FPHunt$i"     # unique per game, and not a
  TRANS="$OUT/game$i-$OURS-vs-$THEIRS.txt" # PREFIX of any reserved name
  log "game $i/${#PAIRINGS[@]}: $OURS vs $THEIRS"

  ( cd "$BC" && exec $PY -m crystal_broadcast.caster_bridge --watch \
      --url "$URL" >"$TRANS" 2>&1 ) &
  TAP_PID=$!
  sleep 2

  ( cd "$FP" && exec .venv/bin/python -u run.py \
      --websocket-uri ws://localhost:8000/showdown/websocket \
      --ps-username "$FPNAME" --bot-mode accept_challenge \
      --pokemon-format gen9ou --team-name "gen9/ou/suite/$THEIRS" \
      --search-time-ms "$OPP_MS" --run-count 1 --log-level INFO \
      ) >"$OUT/fp$i.log" 2>&1 &
  FP_PID=$!
  READY=0
  for _ in $(seq 1 45); do
    grep -q "Waiting for a gen9ou challenge" "$OUT/fp$i.log" 2>/dev/null \
      && { READY=1; break; }
    sleep 2
  done
  [ "$READY" = 1 ] || { log "  OPPONENT-DID-NOT-START"; kill $TAP_PID $FP_PID 2>/dev/null; continue; }

  ( cd "$CB" && exec $PY -u showdown/gen9_player.py --local \
      --username "$CBNAME" --mode challenge --user-to-challenge "$FPNAME" \
      --format gen9ou --team "showdown/teams/suite_v1/$OURS.txt" \
      --n-games 1 --search-ms "$SEARCH_MS" \
      --airi --caster-url "$URL" --airi-turn-pace "$PACE" \
      ) >"$OUT/runner$i.log" 2>&1 &
  CB_PID=$!

  # the engine is the authority here: no viewer, so [RESULT] in the
  # transcript really does mean the game is over
  DONE=0
  for _ in $(seq 1 $((GAME_TIMEOUT / 5))); do
    grep -q "\[RESULT\]" "$TRANS" 2>/dev/null && { DONE=1; break; }
    kill -0 $CB_PID 2>/dev/null || { sleep 8; \
      grep -q "\[RESULT\]" "$TRANS" 2>/dev/null && DONE=1; break; }
    sleep 5
  done
  [ "$DONE" = 1 ] && log "  done" || log "  TIMEOUT-OR-DIED"

  sleep 3
  for p in $TAP_PID $CB_PID $FP_PID; do kill "$p" 2>/dev/null; done
  sleep 2
  for p in $TAP_PID $CB_PID $FP_PID; do kill -9 "$p" 2>/dev/null; done
  TAP_PID=""; CB_PID=""; FP_PID=""
  BEATS=$(grep -c "^\[" "$TRANS" 2>/dev/null || echo 0)
  LINES=$(grep -cE "^[[:space:]]*(Prism|Fracture):" "$TRANS" 2>/dev/null || echo 0)
  log "  $BEATS beats, $LINES spoken lines -> $(basename "$TRANS")"
done

log "hunt complete: $OUT"
grep -c "REPLY (" "$OUT/caster.log" 2>/dev/null | xargs -I{} echo "[hunt] replies fired: {}"
