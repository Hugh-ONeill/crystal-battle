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
  [ -n "$HEADLESS" ] && hyprctl output remove "$HEADLESS" >/dev/null 2>&1
}
trap cleanup EXIT

# ---- broadcast stack (idempotent; only the clock is restarted, so each take
# gets its own presentation log)
cd "$BC" || { log NO-BROADCAST-REPO; exit 1; }
up 8131 || { setsid $PY crystal_broadcast/caster.py </dev/null \
             >"$LOGDIR/caster.log" 2>&1 & disown; }
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

# ---- transcript tap: the text record, and how we read WIN vs LOSS
(cd "$CB" && exec $PY -m crystal_broadcast.caster_bridge --watch 2>&1 \
    | tee "$TRANS") &
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
    --airi --airi-turn-pace "${PRISM_TURN_PACE:-8}" ) \
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
BEFORE=$(hyprctl monitors -j | $PY -c "
import json, sys
print(' '.join(m['name'] for m in json.load(sys.stdin)))")
hyprctl output create headless >/dev/null 2>&1
sleep 2
HEADLESS=$(hyprctl monitors -j | $PY -c "
import json, sys
before = set(sys.argv[1].split())
new = [m['name'] for m in json.load(sys.stdin) if m['name'] not in before]
print(new[0] if new else '')" "$BEFORE")
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
i=0
until mode_ok; do
  i=$((i + 1))
  [ "$i" -gt 8 ] && { log "VIRTUAL-OUTPUT-MODE-FAILED (wanted ${FRAME_W}x${FRAME_H})"; exit 1; }
  hyprctl keyword monitor "$HEADLESS,${FRAME_W}x${FRAME_H}@60,auto,1" >/dev/null
  sleep 1
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
sleep 15
WIN_ADDR=$(hyprctl clients -j | $PY -c "
import json, sys
for c in json.load(sys.stdin):
    t = c.get('title', '')
    if 'Prism Broadcast' in t or 'broadcast.html' in t:
        print(c['address']); break")
[ -z "$WIN_ADDR" ] && { log NO-BROADCAST-WINDOW; exit 1; }
hyprctl dispatch movetoworkspacesilent "$HWS,address:$WIN_ADDR" >/dev/null
sleep 4

# ---- recorder
wf-recorder -o "$HEADLESS" -f "$VID" --overwrite >"$LOGDIR/rec_take$N.log" 2>&1 &
REC_PID=$!
sleep 3
kill -0 $REC_PID 2>/dev/null || { log RECORDER-FAILED; exit 1; }
log "recording -> $VID"

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
