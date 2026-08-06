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
# DECOUPLED 2026-08-02: the player's own queue watchdog (--queue-timeout,
# default 150s / CB_QUEUE_TIMEOUT) bails an UNMATCHED slot with exit 3, so
# this cap now only bounds MATCHED live games — keep it generous (900s was
# killing richwoman grinds at T85+ as disconnect forfeits) with no dead-slot
# cost: an empty queue recycles in ~2.5 min instead of 45.
#
# RAISED 900 -> 2700 on 2026-07-30. At 900s we were KILLING OUR OWN LIVE GAMES:
# three in one 16-slot session, two of them richwoman grinds already at T85 and
# T90. That is worse than the wasted time on three counts — the game is a loss
# we forfeit by disconnect, it never reaches our log so the tally silently
# EXCLUDES it, and it is biased: the games that hit the cap are the long
# attrition grinds, which is exactly what the anti-hazard subpool was built to
# win, so the reweight experiment was being measured with its best cases cut
# out. Games have also lengthened as the pool shifted (avg 25 -> 40 turns).
# 2700s covers roughly a 250-turn game at the observed ~10.6s/turn; the server
# clock (1500s bank/side) is the real terminator, so the wrapper should only
# ever catch a genuinely wedged process. Cost: a dead-queue slot now blocks 45
# min instead of 15 — acceptable, since an empty queue means no games either
# way. TODO: decouple the two by bailing early when no battle room appears.
export PER_GAME_TIMEOUT="${PER_GAME_TIMEOUT:-2700}"
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

# After a fired-on-resume start (box woke seconds before us) the Wi-Fi lags
# the service: on 2026-08-06 game 1 died on "Temporary failure in name
# resolution". Wait for NetworkManager, then prove DNS end-to-end against the
# actual ladder host; a still-dead network after the timeout starts anyway —
# a late-joining network costs the first slot, same as before, and the
# one-game-per-process loop recovers on slot 2.
nm-online -q -t 60 2>/dev/null || true
waited=0
until getent hosts battling.pokeagentchallenge.com >/dev/null 2>&1; do
  if [ "$waited" -ge 90 ]; then
    echo "WARNING: ladder host still unresolvable after ${waited}s; starting anyway"
    break
  fi
  sleep 5
  waited=$((waited + 5))
done

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
#
# Probe the inhibitor FIRST and RETRY before giving up: when the 01:00 timer
# fires on resume (box was asleep at 01:00) the service starts mid-suspend-
# cycle and the first probe is guaranteed to fail. The old probe-once
# fallback (added after 2026-08-02, when that failure killed the session in
# 1s) ran the whole 2026-08-06 session uninhibited and hypridle re-suspended
# the box MID-GAME at 03:41. The race window is seconds wide; a minute of 5s
# probes covers it with room to spare. (Still probe-then-run: wrapping the
# session in `|| retry` instead would re-run the whole session whenever the
# INNER command exits nonzero — do not do that.)
INHIBIT="systemd-inhibit --mode=block --what=sleep:idle --why=crystal-battle-overnight-ladder"
probes=0
until $INHIBIT true 2>/dev/null; do
  probes=$((probes + 1))
  if [ "$probes" -ge 12 ]; then
    echo "WARNING: suspend inhibitor still unavailable after $probes probes over ~60s; running uninhibited"
    INHIBIT=""
    break
  fi
  sleep 5
done
$INHIBIT \
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

# refresh the per-opponent scouting book with everything through this session.
# The book had gone stale once (richwoman ABSENT through 205 games, played
# bookless until 2026-07-30); rebuilding from all logs each run keeps the
# set/lead priors current and is cheap. Failure is non-fatal by design.
if .venv/bin/python showdown/scouting_book.py --name "$USERNAME" \
      "$CB"/showdown/bench/overnight_*_ladder.log >/dev/null 2>&1; then
  echo "scouting book refreshed (all sessions through $TAG)"
else
  echo "WARNING: scouting book refresh failed; book may be stale" >&2
fi

echo "done. analysis saved to $ANALYSIS"
