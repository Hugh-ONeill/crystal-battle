#!/bin/sh
# Overnight free-play ladder runner for crystal-battle (PokeAgent Season 2).
#
# Wraps ladder_session.sh in a suspend-inhibit block so the laptop stays awake
# for the whole run (a mid-run suspend bit us before), plays N sequential games
# (ladder_session does one-game-per-process + per-game timeout + pool rotation),
# then runs the loss-trace + per-opponent standings pass over this session's log.
#
# Sequential by necessity: the whole team shares one PokeAgent credential and a
# same-name login kicks the running bot, so only ONE PAC-Crystal ladder stream
# can exist. The preflight below refuses to start if another is already up.
#
# Usage:
#   showdown/overnight_ladder.sh [n_games] [pool_dir] [extra gen9_player args...]
# e.g.
#   showdown/overnight_ladder.sh 120 showdown/teams/pool_hl
#   showdown/overnight_ladder.sh                       # 120 games, pool_hl
#
# Env:
#   LADDER_FORMAT     queue (default gen9oulongtimer — the active baseline queue)
#   SERVER            gen9_player --server target (default pokeagent)
#   PER_GAME_TIMEOUT  per-game kill in seconds (default 1800)
# Credentials come from ~/.config/crystal-battle/pokeagent.env (PS_USERNAME /
# PS_PASSWORD), sourced by ladder_session.sh.

set -u
CB=/home/wiz/Developer/grimoire/crystal-battle
cd "$CB" || exit 1

N_GAMES="${1:-120}"
POOL="${2:-$CB/showdown/teams/pool_hl}"
if [ $# -ge 2 ]; then shift 2; else shift $#; fi   # rest -> gen9_player

SERVER="${SERVER:-pokeagent}"
export LADDER_FORMAT="${LADDER_FORMAT:-gen9oulongtimer}"
# per-game timeout doubles as the queue-wait: a thin baseline pool means we
# often sit queued with no match, so keep it long enough for a real long-timer
# game AND to catch a baseline that queues sporadically, but short enough that
# a dead slot recycles (re-queues) rather than blocking for ages. 900s = 15min.
export PER_GAME_TIMEOUT="${PER_GAME_TIMEOUT:-900}"
# RUN_DEADLINE bounds the WHOLE session by wall-clock (default 8h) so an empty
# queue can't turn a 120-slot run into a multi-day churn of search-timeouts.
# The run stops at whichever comes first: N_GAMES slots or this deadline.
RUN_DEADLINE="${RUN_DEADLINE:-8h}"

CRED_FILE="$HOME/.config/crystal-battle/pokeagent.env"
[ -f "$CRED_FILE" ] || { echo "FATAL: cred file $CRED_FILE missing" >&2; exit 1; }
[ -d "$POOL" ] || { echo "FATAL: team pool $POOL missing" >&2; exit 1; }
# games are logged under this name; the loss-trace pass needs it to pick our side
USERNAME=$(. "$CRED_FILE"; echo "${PS_USERNAME:-PAC-Crystal}")

# preflight: another same-name ladder stream would kick this one off the server
if pgrep -f "ladder_session.sh" >/dev/null 2>&1 || \
   pgrep -f "gen9_player.py .*--mode ladder" >/dev/null 2>&1; then
  echo "FATAL: a ladder session / watcher is already running as $USERNAME." >&2
  echo "  Stop it first (e.g. pkill -f ladder_session.sh); a same-name login" >&2
  echo "  kicks the running bot off the ladder." >&2
  exit 1
fi

STAMP=$(date +%Y%m%d_%H%M%S)
TAG="overnight_$STAMP"
LOG="$CB/showdown/bench/${TAG}_ladder.log"
DESK="$CB/showdown/desk_reads_${STAMP}.jsonl"
ANALYSIS="$CB/showdown/bench/${TAG}_analysis.txt"

echo "=== overnight ladder: up to $N_GAMES games (or ${RUN_DEADLINE}) on $LADDER_FORMAT as $USERNAME ==="
echo "  pool:     $POOL"
echo "  log:      $LOG"
echo "  desk-log: $DESK   (Brier calibration accrual)"
echo "  analysis: $ANALYSIS"
START=$(date +%s)

# one suspend-inhibit block around the whole run. --adaptive on turns on the
# budget-by-clock escalation path; a dated desk-log accrues calibration data.
# `timeout $RUN_DEADLINE` caps the whole session by wall-clock so a thin/empty
# queue can't churn search-timeouts past the night.
systemd-inhibit --mode=block --what=sleep:idle \
    --why="crystal-battle overnight ladder ($N_GAMES games)" \
    timeout "$RUN_DEADLINE" \
    sh showdown/ladder_session.sh "$TAG" "$N_GAMES" "$SERVER" "$POOL" \
        --adaptive on --desk-log "$DESK" "$@"

END=$(date +%s)
MINS=$(( (END - START) / 60 ))
echo "=== session wall time: ${MINS} min ==="

# end-of-run analysis: winrate + CI + per-opponent standings + loss collapses
{
  echo "=== overnight session $TAG ($MINS min) ==="
  .venv/bin/python showdown/loss_trace.py --name "$USERNAME" \
      --collapse-examples 3 "$LOG" 2>&1
} | tee "$ANALYSIS"

echo "done. analysis saved to $ANALYSIS"
