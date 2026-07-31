#!/bin/bash
# One recording attempt, v4 (composite era).
#
# The frame is now a SINGLE kiosk browser window showing broadcast.html: the
# self-hosted Showdown client on top, the commentary strip beneath it, black
# on black. That replaces v3's two-tile split (battle browser + kitty panel),
# which is why there is no overlay_start.sh here and no window arranging to do
# beforehand.
#
# Four things in here are load-bearing and were each learned the hard way on
# 2026-07-27:
#
#   * UNIQUE USERNAMES PER TAKE. Killing a player mid-battle leaves the battle
#     alive on the server; when the same name logs back in, poke-env resumes it
#     and the "new" take opens on turn 66 of the old game.
#   * THE OPPONENT TEAM IS PINNED. `--team-name gen9/ou/suite` picks at random
#     across the whole directory — the 8 base teams plus hundreds of generated
#     pairing files — so stall comes up often and the game never ends.
#   * STOP ON THE VIEWER, NOT THE ENGINE. The client replays from turn 1 when
#     it joins and animates minutes behind, so [RESULT] in the transcript means
#     the engine is done, not the video. We wait for a PRESENTED |win| from the
#     presentation clock instead.
#   * KILL BY PID. Never pattern-kill: `pkill -f "gen9_player.py"` matches the
#     user's bench lanes and has destroyed two A/B series.
#
# usage: attempt.sh <take-number>   prints TAKE-WIN / TAKE-LOSS / a failure tag
set -u
N=$1
HERE="$(cd "$(dirname "$0")" && pwd)"
CB="$(cd "$HERE/../.." && pwd)"
BC="${CRYSTAL_BROADCAST:-$HOME/Developer/grimoire/crystal-broadcast}"
PY=$CB/.venv/bin/python
FP=${FOULPLAY_DIR:-$HOME/Developer/grimoire/foul-play}
DATE=${PRISM_DATE:-$(date +%Y%m%d)}
LOGDIR=${PRISM_LOG_DIR:-${TMPDIR:-/tmp}/prism-recording}
# Ask XDG where videos live rather than assuming ~/Videos: this box maps
# XDG_VIDEOS_DIR to ~/Documents/Videos, so writing to ~/Videos created a stray
# directory that xdg-user-dirs-update later relocated wholesale — takes appeared
# to vanish mid-session, and retry_take's keep/discard pointed at a dead path.
VIDDIR=${PRISM_VIDEO_DIR:-$(xdg-user-dir VIDEOS 2>/dev/null || echo "$HOME/Videos")}
VID=$VIDDIR/crystal-broadcast-$DATE-take$N.mp4
TRANS=$VIDDIR/crystal-broadcast-$DATE-take$N-transcript.txt
CLOCK=$LOGDIR/clock_take$N.jsonl
OUTPUT=${PRISM_OUTPUT:-DP-4}
# our team leads offense so the game reaches a decision; the opponent is
# pinned to bulky offense for the same reason (see the note above)
OUR_TEAM=${PRISM_TEAM:-showdown/teams/suite_v1/07_kommoo_offense.txt}
OPP_TEAM=${PRISM_OPP_TEAM:-gen9/ou/suite/04_cinderace_kyurem_bo}
# unique per take: nothing to rejoin, and not a prefix of any reserved name
CBNAME="CBTake${N}"
FPNAME="FPTake${N}"
mkdir -p "$LOGDIR" "$VIDDIR"

log() { echo "[attempt$N] $*"; }
up()  { ss -tln 2>/dev/null | grep -q ":$1 "; }

CB_PID=""; FP_PID=""; REC_PID=""; TAP_PID=""; CLOCK_PID=""
HEADLESS=""; WIN_ADDR=""
cleanup() {
  for p in $REC_PID $CB_PID $FP_PID $TAP_PID $CLOCK_PID; do
    kill "$p" 2>/dev/null
  done
  [ -n "$WIN_ADDR" ] && hyprctl dispatch closewindow "address:$WIN_ADDR" >/dev/null 2>&1
  # the headless output is PERSISTENT across takes (see its creation below) —
  # removing it here was the per-take dice roll that burned takes 34-39
}
trap cleanup EXIT

# ---- broadcast stack (idempotent; only the clock is restarted, so each take
# gets its own presentation log)
cd "$BC" || { log NO-BROADCAST-REPO; exit 1; }
# PRISM_SPEECH=1 gives the duo a voice (needs caster-avatars/tts_server.py
# on :8133). Off by default: audio is opt-in, and a take without it is
# exactly the take we have been recording all along.
SPEECH_ARGS=""
SPEECH_SINK=""
TURN_PACE=${PRISM_TURN_PACE:-8}
if [ -n "${PRISM_SPEECH:-}" ]; then
  # A recorder can only capture audio that reached a SINK, so the take
  # has to play — but playing to the default sink makes a seven-minute
  # take audible in the room, and WHICH device that is changes when
  # headphones come and go. A null sink keeps the capture and drops the
  # noise; it is torn down in cleanup.
  # ONE stable sink for every take, NOT one per take. The caster is
  # deliberately persistent across takes (setsid + disown, and cleanup does
  # not kill it), so a per-take sink meant take 2 onward played into a device
  # that had just been unloaded while the recorder listened to a fresh empty
  # one: take 17 on 2026-07-29 won and was completely silent. Create it only
  # if absent, and never unload it — a null sink costs nothing and outliving
  # the batch is the point.
  SPEECH_SINK=prism_speech
  if ! pactl list short sinks 2>/dev/null | grep -q "$SPEECH_SINK"; then
    pactl load-module module-null-sink sink_name="$SPEECH_SINK" \
          sink_properties=device.description=PrismSpeech >/dev/null 2>&1
  fi
  # The budget is what stops a busy beat queueing more speech than the
  # floor can hold: the first voice always finishes, the second is
  # dropped when it will not fit. Inert unless durations exist, which
  # is why it was never set before the TTS layer landed.
  SPEECH_BUDGET=${PRISM_SPEECH_BUDGET:-8}
  # Spoken commentary needs more room than written: a line that reads in
  # a glance takes four seconds to say, so the same beat density that is
  # comfortable as text arrives faster than the voices can clear it.
  # Widening the turn gate means fewer, better-spaced moments rather
  # than a queue of dropped ones.
  TURN_PACE=${PRISM_TURN_PACE:-12}
  # Stable like the sink, and for the same reason: the caster outlives a
  # take, so a per-take directory means take 2 onward writes its wavs
  # under take 1's name. Sequence numbers keep rising and the transcript
  # timestamps say which take a clip belongs to.
  SPEECH_ARGS="--speech --speech-out $LOGDIR/speech --speech-budget $SPEECH_BUDGET"
  SPEECH_ARGS="$SPEECH_ARGS --speech-sink $SPEECH_SINK"
fi
# --pts-url: discovered ABSENT 2026-07-30 — the v4 rewrite dropped it, so
# the caster ran wall-clock all era and TURN_PACE alone carried sync. The
# clock service restarts per take while the caster persists; the pts client
# reconnects and resets its camera gate on feed loss by design.
# The caster persisting across takes means caster.py EDITS DO NOT APPLY until
# it is bounced — which silently voided a whole evening's guard validation on
# 2026-07-30 (takes 72-75 ran a caster booted hours earlier; the new guards
# were never loaded, and their clean hunts proved nothing). Stamp the SHA the
# running caster was started from so staleness is one visible line instead of
# something inferred from a missing log suffix. Bouncing is left MANUAL on
# purpose: killing a caster mid-hunt costs the take in flight, and only the
# operator knows whether that trade is wanted right now.
CASTER_STAMP=$LOGDIR/caster.sha
HEAD_SHA=$(cd "$BC" && git rev-parse --short HEAD 2>/dev/null || echo unknown)
if up 8131; then
  RUNNING_SHA=$(cat "$CASTER_STAMP" 2>/dev/null || echo unknown)
  if [ "$RUNNING_SHA" != "$HEAD_SHA" ]; then
    log "WARNING: caster is running $RUNNING_SHA but crystal-broadcast HEAD is $HEAD_SHA — caster.py changes are NOT live; bounce it between takes to pick them up"
  else
    log "caster reused (crystal-broadcast $RUNNING_SHA, matches HEAD)"
  fi
else
  setsid $PY crystal_broadcast/caster.py $SPEECH_ARGS \
         --pts-url ws://127.0.0.1:8132/ </dev/null \
         >"$LOGDIR/caster.log" 2>&1 & disown
  echo "$HEAD_SHA" > "$CASTER_STAMP"
  log "caster started fresh (crystal-broadcast $HEAD_SHA)"
fi
up 8130 || { setsid $PY crystal_broadcast/commentary_overlay.py </dev/null \
             >"$LOGDIR/feed.log" 2>&1 & disown; }
up 8127 || { setsid $PY crystal_broadcast/serve_client.py </dev/null \
             >"$LOGDIR/client.log" 2>&1 & disown; }
for i in $(seq 1 40); do up 8131 && up 8130 && up 8127 && break; sleep 1; done
up 8131 && up 8130 && up 8127 || { log STACK-DOWN; exit 1; }

pkill -f -- "--log $CLOCK" 2>/dev/null
rm -f "$CLOCK"
setsid $PY crystal_broadcast/presentation_clock.py --log "$CLOCK" </dev/null \
    >"$LOGDIR/clock_take$N.log" 2>&1 &
CLOCK_PID=$!
disown
sleep 2
log "stack up (caster 8131, feed 8130, client 8127, clock 8132)"

# ---- health gate: a test beat must come back as a generated line
gate() {
  local REPLY_LOG="$LOGDIR/gate$N.log" WATCH i
  rm -f "$REPLY_LOG"
  (cd "$CB" && exec $PY -m crystal_broadcast.caster_bridge --watch \
      --url "${PRISM_CASTER_URL:-ws://127.0.0.1:8131/ws}" \
      > "$REPLY_LOG" 2>&1) &
  WATCH=$!
  sleep 2
  (cd "$CB" && $PY -m crystal_broadcast.caster_bridge \
      --url "${PRISM_CASTER_URL:-ws://127.0.0.1:8131/ws}" \
      "[bridge-test] pre-take health gate $N" >/dev/null 2>&1)
  for i in $(seq 1 40); do          # cold gemma load can take 15-60s
    sleep 2
    if [ -s "$REPLY_LOG" ] && grep -q "[[:alpha:]]" "$REPLY_LOG"; then
      kill $WATCH 2>/dev/null; echo OK; return
    fi
  done
  kill $WATCH 2>/dev/null
  echo NOGEN
}
[ "$(gate)" = OK ] || { log HEALTH-FAIL-NOGEN; exit 1; }
log health-ok

# ---- transcript tap: the text record, and how we read WIN vs LOSS.
# Straight to the file, NO tee: with a pipeline, $! is the subshell, and
# killing it ORPHANS python+tee — which reconnect after a caster restart and
# write the NEXT take's game into THIS take's transcript (caught 2026-07-30:
# take 49's file carried take 50's beats). exec makes $! the actual tap.
(cd "$CB" && exec $PY -m crystal_broadcast.caster_bridge --watch \
    > "$TRANS" 2>&1) &
TAP_PID=$!
sleep 2

# ---- opponent, then us
( cd "$FP" && exec .venv/bin/python -u run.py \
    --websocket-uri ws://localhost:8000/showdown/websocket \
    --ps-username "$FPNAME" --bot-mode accept_challenge \
    --pokemon-format gen9ou --team-name "$OPP_TEAM" \
    --search-time-ms "${PRISM_OPP_MS:-300}" --run-count 1 \
    --log-level INFO ) >"$LOGDIR/fp_take$N.log" 2>&1 &
FP_PID=$!
for i in $(seq 1 45); do
  grep -q "Waiting for a gen9ou challenge" "$LOGDIR/fp_take$N.log" 2>/dev/null \
    && break
  sleep 2
done
grep -q "Waiting for a gen9ou challenge" "$LOGDIR/fp_take$N.log" 2>/dev/null \
  || { log OPPONENT-DID-NOT-START; exit 1; }

( cd "$CB" && exec $PY -u showdown/gen9_player.py --local \
    --username "$CBNAME" --mode challenge --user-to-challenge "$FPNAME" \
    --format gen9ou --team "$OUR_TEAM" --n-games 1 \
    --search-ms "${PRISM_SEARCH_MS:-2000}" \
    --airi --airi-turn-pace "$TURN_PACE" ) \
    >"$LOGDIR/runner_take$N.log" 2>&1 &
CB_PID=$!
sleep 12
ROOM=$(cd "$CB" && timeout 30 $PY "$HERE/room_id.py" --player "$CBNAME")
log "room: $ROOM"
[ "$ROOM" = "NO-BATTLE-FOUND" ] && { log NO-BATTLE; exit 1; }

# ---- the frame renders on a VIRTUAL output, so a take never takes over the
# screen. Recording captures a wlroots OUTPUT, so the frame has to be visible
# SOMEWHERE; putting it on the real monitor makes the machine unusable for the
# length of a take, and closing the window to get your desktop back leaves the
# recorder running against whatever was behind it. A headless output is visible
# to the compositor and to wf-recorder, and invisible to you.
# Sized so the battle region is exactly 16:9 (width x (width*9/16 + panel));
# any other ratio pillarboxes the scene inside the frame.
FRAME_W=${PRISM_FRAME_W:-1600}
PANEL=${PRISM_PANEL:-300}
FRAME_H=$(( FRAME_W * 9 / 16 + PANEL ))
# ONE persistent headless output, NOT one per take — the prism_speech sink
# lesson again: a per-take create/remove of a resource the whole batch needs
# is a dice roll per take, and on the night of 2026-07-29 the create+mode
# race went chronic (5 of 8 takes burned on VIRTUAL-OUTPUT-MODE-FAILED, one
# success in six tries, GPU ~23GB loaded). Reuse any existing headless
# output; create only when absent; cleanup no longer removes it. A wrong-res
# leftover is fine — the mode loop below re-applies to it.
# NAME the output instead of discovering it. `hyprctl output create
# headless` (no name) stopped yielding a findable HEADLESS-N, and because
# discovery worked by DIFFING the monitor list around the create, it found
# nothing and failed silently — the whole 88-97 hunt died on
# NO-VIRTUAL-OUTPUT, ten takes, before this was spotted. Asking for a name
# means we already know what to look for, so there is no race and no diff.
# The HEADLESS-* fallback is only for leftovers from the old scheme.
VOUT=${PRISM_VOUT:-demo}
HEADLESS=$(hyprctl monitors -j | $PY -c "
import json, sys
want = sys.argv[1]
names = [m['name'] for m in json.load(sys.stdin)]
print(want if want in names
      else next((n for n in names if n.startswith('HEADLESS')), ''))" "$VOUT")
if [ -z "$HEADLESS" ]; then
  hyprctl output create headless "$VOUT" >/dev/null 2>&1
  for i in $(seq 1 10); do          # poll: creation is not instant
    sleep 1
    HEADLESS=$(hyprctl monitors -j | $PY -c "
import json, sys
names = [m['name'] for m in json.load(sys.stdin)]
print(sys.argv[1] if sys.argv[1] in names else '')" "$VOUT")
    [ -n "$HEADLESS" ] && break
  done
fi
[ -z "$HEADLESS" ] && { log NO-VIRTUAL-OUTPUT; exit 1; }
# Apply-and-verify, don't fire-and-forget: the first `keyword monitor` after
# `output create` reports "ok" but silently no-ops, leaving the output at its
# 1920x1080 default — which pillarboxes the scene and makes the panel the wrong
# height. A second application takes, so retry until the size is actually real.
mode_ok() {
  hyprctl monitors -j | $PY -c "
import json, sys
name, w, h = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
for m in json.load(sys.stdin):
    if m['name'] == name and m['width'] == w and m['height'] == h:
        sys.exit(0)
sys.exit(1)" "$HEADLESS" "$FRAME_W" "$FRAME_H"
}
# 8 tries was too tight: take 12 on 2026-07-28 burned a whole attempt on
# VIRTUAL-OUTPUT-MODE-FAILED, and the race is worse now that TTS shares the GPU
# and the compositor is busier at exactly this moment. Retries are ~1s each and
# a lost take is ~7 minutes, so buy the headroom.
i=0
until mode_ok; do
  i=$((i + 1))
  # 40 x 3s: the 20 x 1.5s window lost five takes in one night when the
  # race went chronic — a 2-minute window is still cheap against a lost
  # take, and with the persistent output this loop usually no-ops anyway
  [ "$i" -gt 40 ] && { log "VIRTUAL-OUTPUT-MODE-FAILED (wanted ${FRAME_W}x${FRAME_H})"; exit 1; }
  hyprctl keyword monitor "$HEADLESS,${FRAME_W}x${FRAME_H}@60,auto,1" >/dev/null
  sleep 3
done
HWS=$(hyprctl monitors -j | $PY -c "
import json, sys
for m in json.load(sys.stdin):
    if m['name'] == sys.argv[1]:
        print(m.get('activeWorkspace', {}).get('id'))" "$HEADLESS")
log "virtual output $HEADLESS ${FRAME_W}x${FRAME_H} (workspace $HWS)"

for a in $(hyprctl clients -j | $PY -c "
import json, sys
for c in json.load(sys.stdin):
    t = c.get('title', '')
    if 'Prism Broadcast' in t or 'broadcast.html' in t: print(c['address'])"); do
  hyprctl dispatch closewindow "address:$a" >/dev/null
done
sleep 1

# Dedicated, extension-free profile. The default profile runs Dark Reader,
# Stylus, Tampermonkey, Violentmonkey and uBlock, all of which rewrite the
# page: Dark Reader inverted the overlay's dark palette and left the battle
# sprites alone, which is what painted a white band under the scene. --no-remote
# matters as much as -profile — without it a already-running Firefox swallows
# the --new-window and renders the frame in the default profile anyway.
FF_PROFILE=${PRISM_FF_PROFILE:-$HOME/.local/share/prism-broadcast-profile}
if [ ! -d "$FF_PROFILE" ]; then
  mkdir -p "$FF_PROFILE"
  cat > "$FF_PROFILE/user.js" <<'EOF'
user_pref("browser.shell.checkDefaultBrowser", false);
user_pref("browser.startup.homepage_override.mstone", "ignore");
user_pref("browser.aboutwelcome.enabled", false);
user_pref("datareporting.policy.dataSubmissionEnabled", false);
user_pref("toolkit.telemetry.reportingpolicy.firstRun", false);
user_pref("browser.tabs.warnOnClose", false);
user_pref("browser.sessionstore.resume_from_crash", false);
user_pref("layout.css.prefers-color-scheme.content-override", 0);
EOF
  log "created clean firefox profile $FF_PROFILE"
fi

hyprctl dispatch exec -- \
  "firefox --no-remote -profile \"$FF_PROFILE\" --kiosk --new-window \"http://127.0.0.1:8129/broadcast.html?battle=$ROOM&panel=$PANEL\"" \
  >/dev/null

# ---- recorder — started right AFTER the frame spawn, before the client
# animates. The old post-window-wait start sliced the first turns off every
# video (a REAL first-turn crit callout read as imagined, take 54); starting
# BEFORE the spawn put the capture on the output during firefox's WebRender
# init and coincided with a spawn crash (take 56). The client needs ~10s to
# load and join, so this window keeps the head fix without the contention.
REC_AUDIO=""
if [ -n "${PRISM_SPEECH:-}" ]; then
  SINK="${SPEECH_SINK:-$(pactl get-default-sink 2>/dev/null)}"
  [ -n "$SINK" ] && REC_AUDIO="--audio=${SINK}.monitor"
  log "recording audio from ${SINK:-unknown}.monitor"
fi
wf-recorder -o "$HEADLESS" $REC_AUDIO -f "$VID" --overwrite >"$LOGDIR/rec_take$N.log" 2>&1 &
REC_PID=$!
sleep 2
kill -0 $REC_PID 2>/dev/null || { log RECORDER-FAILED; exit 1; }
log "recording -> $VID"

# poll for the window instead of one-shot sleeping: a firefox crash-restart
# (take 56) then costs seconds, not the take
WIN_ADDR=""
for i in $(seq 1 22); do
  sleep 2
  WIN_ADDR=$(hyprctl clients -j | $PY -c "
import json, sys
for c in json.load(sys.stdin):
    t = c.get('title', '')
    if 'Prism Broadcast' in t or 'broadcast.html' in t:
        print(c['address']); break")
  [ -n "$WIN_ADDR" ] && break
done
[ -z "$WIN_ADDR" ] && { log NO-BROADCAST-WINDOW; exit 1; }
hyprctl dispatch movetoworkspacesilent "$HWS,address:$WIN_ADDR" >/dev/null
sleep 4

# ---- wait for the VIEWER to reach the end, not the engine
viewer_done() {
  $PY - "$CLOCK" <<'PYEOF'
import json, sys
try:
    for line in open(sys.argv[1]):
        try:
            e = json.loads(line)
        except ValueError:
            continue
        # "|tie|" needs the trailing pipe: "|tie" also prefix-matches
        # "|tier|[Gen 9] OU", the format banner every battle opens with, so the
        # take "finished" one second in and recorded a 10s clip.
        if e.get("kind") == "presented" and \
                (e.get("line") or "").startswith(("|win|", "|tie|")):
            sys.exit(0)
except OSError:
    pass
sys.exit(1)
PYEOF
}
frame_alive() {
  hyprctl clients -j | $PY -c "
import json, sys
print(any(c['address'] == sys.argv[1] for c in json.load(sys.stdin)))" \
    "$WIN_ADDR" | grep -q True
}
SECONDS=0
while ! viewer_done; do
  # if the frame dies (closed by hand, crash) stop at once — otherwise the
  # recorder keeps rolling against whatever is behind it
  frame_alive || { log FRAME-CLOSED; break; }
  [ $SECONDS -gt "${PRISM_CAP:-1800}" ] && { log VIEWER-TIMEOUT; break; }
  sleep 8
done
viewer_done && log "viewer reached the end (${SECONDS}s)"
sleep 8                      # let the wrap-up line sit on camera
kill -INT $REC_PID 2>/dev/null; REC_PID=""
sleep 3

if grep -q "\[RESULT\] WIN" "$TRANS" 2>/dev/null; then
  log TAKE-WIN
else
  log TAKE-LOSS
fi
