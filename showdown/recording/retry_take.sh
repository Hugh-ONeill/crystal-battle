#!/bin/bash
# Record until crystal-battle wins (or the takes run out), keeping the video
# only on a WIN.
#
# v4: attempt.sh now owns the whole take — the broadcast stack, BOTH bots
# (with per-take usernames and a pinned opponent team), the kiosk frame, the
# recorder, and the stop condition. This script is just the retry loop and the
# keep/discard decision. It used to launch the opponent itself under a shared
# name, which is exactly how an abandoned take poisoned the next one: killing
# a player mid-battle leaves the battle alive, and the same name logging back
# in resumes it.
#
# usage: retry_take.sh [take-numbers...]     (default: 1 2 3)
#   PRISM_TEAM      our team      (default suite_v1/07_kommoo_offense.txt)
#   PRISM_OPP_TEAM  their team    (default gen9/ou/suite/04_cinderace_kyurem_bo)
#   PRISM_KEEP_LOSSES=1  keep losing videos instead of deleting them
set -u
HERE="$(cd "$(dirname "$0")" && pwd)"
LOGDIR=${PRISM_LOG_DIR:-${TMPDIR:-/tmp}/prism-recording}
mkdir -p "$LOGDIR"
export PRISM_DATE=${PRISM_DATE:-$(date +%Y%m%d)}
export PRISM_LOG_DIR=$LOGDIR
VIDDIR=${PRISM_VIDEO_DIR:-$(xdg-user-dir VIDEOS 2>/dev/null || echo "$HOME/Videos")}
export PRISM_VIDEO_DIR=$VIDDIR

TAKES=("$@")
[ ${#TAKES[@]} -eq 0 ] && TAKES=(1 2 3)

# A live bench series shares the Showdown server with us, and today's takes
# would add load and room churn on top of it. Warn rather than refuse: the
# user may well know and not care.
if pgrep -f "par_series" >/dev/null 2>&1; then
  echo "WARNING: a par_series bench looks live — takes will share :8000 with it"
fi

for N in "${TAKES[@]}"; do
  echo "=== TAKE $N ==="
  # must match attempt.sh's XDG-resolved dir, or the keep/discard below acts on
  # a path nothing was ever written to
  VID="$VIDDIR/crystal-broadcast-$PRISM_DATE-take$N.mp4"
  # Via a file, NOT $(attempt.sh | tail -1). Command substitution waits for EOF
  # on the pipe, and attempt.sh leaves the broadcast stack running as background
  # children that inherit the write end — so the substitution blocks forever
  # even after attempt.sh has exited. That wedged two harness runs.
  OUT="$LOGDIR/attempt_take$N.out"
  "$HERE/attempt.sh" "$N" >"$OUT" 2>&1
  RESULT=$(tail -1 "$OUT")
  echo "$RESULT"
  case "$RESULT" in
    *TAKE-WIN*)
      echo "WON on take $N -> $VID"
      exit 0
      ;;
    *TAKE-LOSS*)
      if [ -n "${PRISM_KEEP_LOSSES:-}" ]; then
        echo "kept (loss): $VID"
      else
        rm -f "$VID"          # transcripts are kept either way
      fi
      ;;
    *)
      echo "take $N failed before recording ($RESULT)"
      rm -f "$VID"
      ;;
  esac
done
echo "no win in ${#TAKES[@]} tries (transcripts kept in ~/Videos)"
