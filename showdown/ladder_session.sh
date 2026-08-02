#!/bin/sh
# Ladder session driver: N ladder games, ONE GAME PER PROCESS.
#
# poke-env's between-games loop (accept_challenges AND player.ladder alike)
# wedges after a completed game — challenge/search never re-issued, process
# alive but idle (diagnosed 2026-07-15). Same structural fix as ab_series.sh:
# each game runs in a fresh process that searches the ladder once, plays,
# and exits. A wedge or hang costs one game (killed by the per-game timeout).
#
# Also rotates the team per game from a pool directory: broadcasting one team
# all session is a free read for opponents (and for foul-play-derived bots
# whose databases know the sample teams verbatim).
#
# Usage:
#   PS_PASSWORD=... showdown/ladder_session.sh <name> <n_games> <server> \
#       <team_pool_dir> [extra gen9_player args...]
# e.g.
#   PS_PASSWORD=secret showdown/ladder_session.sh pa1 20 pokeagent \
#       showdown/teams/pool_hl --search-ms 300 --adaptive on
#
# Team pools (built by showdown/curate_team_pool.py, gitignored):
#   pool_hl           40 most-popular legal High-Ladder cores (current meta)
#   pool_competitive  11 legal human Smogon sample teams (coherent-set default)
#
# Env: PS_USERNAME (default CBGen9), PS_PASSWORD (registered accounts),
#      PER_GAME_TIMEOUT (default 1800s — extended-timer games run long)

set -u
NAME="$1"; N_GAMES="$2"; SERVER="$3"; POOL="$4"; shift 4
CB=/home/wiz/Developer/grimoire/crystal-battle

# credentials: explicit env wins; else fall back to the locked config file
# (~/.config/crystal-battle/pokeagent.env — OUTSIDE the public repo, 600)
CRED_FILE="$HOME/.config/crystal-battle/pokeagent.env"
if [ -z "${PS_PASSWORD:-}" ] && [ -f "$CRED_FILE" ]; then
  set -a; . "$CRED_FILE"; set +a
fi
LOG="$CB/showdown/bench/${NAME}_ladder.log"
USERNAME="${PS_USERNAME:-CBGen9}"
FORMAT="${LADDER_FORMAT:-gen9ou}"
PER_GAME_TIMEOUT="${PER_GAME_TIMEOUT:-1800}"

cd "$CB"
[ -d "$POOL" ] || { echo "FATAL: team pool dir $POOL missing" >&2; exit 1; }

# Config banner: sessions are compared against each other weeks apart (see
# `ladder tally all`), and a winrate is meaningless without the argv and code
# version behind it — a run with the tera search off once looked like a 30pp
# regression. Record it in the log itself; nothing else knows what we ran with.
COMMIT=$(git -C "$CB" rev-parse --short HEAD 2>/dev/null || echo "?")
[ -n "$(git -C "$CB" status --porcelain --untracked-files=no 2>/dev/null)" ] && COMMIT="$COMMIT+dirty"
echo "=== session config | commit=$COMMIT | format=$FORMAT | server=$SERVER | pool=$(basename "$POOL") | timeout=${PER_GAME_TIMEOUT} | argv= $*" >> "$LOG"

# Interleaved A/B on the ladder. Set AB_FLAG to an extra argument string and
# every OTHER game carries it, so both arms see the same opponent mix, the same
# team rotation and the same code — the only difference is the flag. Ladder
# opponents cannot be paired the way par_series pairs suite teams (we do not
# choose who we are matched with), so this is a randomised trial rather than a
# paired one: unbiased, just needing more games. Arms are stamped into the
# per-game banner so `ladder tally` can split them.
#   AB_FLAG='--team-archive showdown/teams/team_archive_gen9ou.json' ladder start
g=1
wins=0
while [ "$g" -le "$N_GAMES" ]; do
  # rotate team: pseudo-random pick from the pool
  TEAM=$(ls "$POOL"/*.txt | shuf -n 1)
  ARM=""; ARM_ARGS=""
  if [ -n "${AB_FLAG:-}" ]; then
    if [ $(( g % 2 )) -eq 0 ]; then ARM=" arm=B"; ARM_ARGS="$AB_FLAG"; else ARM=" arm=A"; fi
  fi
  echo "=== game $g/$N_GAMES team: $(basename "$TEAM" .txt)$ARM ($(date +%H:%M:%S)) ===" >> "$LOG"
  # shellcheck disable=SC2086  # ARM_ARGS is an intentional word-split arg list
  timeout "$PER_GAME_TIMEOUT" .venv/bin/python showdown/gen9_player.py \
      --server "$SERVER" --username "$USERNAME" \
      --mode ladder --format "$FORMAT" --team "$TEAM" \
      --n-games 1 --log-level 20 $ARM_ARGS \
      "$@" >> "$LOG" 2>&1
  status=$?
  if [ "$status" -eq 124 ]; then
    # PER_GAME_TIMEOUT killed a MATCHED live game — with the queue watchdog
    # bailing unmatched slots at exit 3, this is always self-inflicted now
    echo "=== game $g TIMED OUT (skipped) ===" >> "$LOG"
  elif [ "$status" -eq 3 ]; then
    echo "=== game $g NO MATCH (queue empty; slot freed early) ===" >> "$LOG"
    g=$((g + 1))
    sleep 3
    continue  # nothing played — skip the book refresh
  fi
  grep -q "finished: 1W" "$LOG" && :  # tally computed at the end
  # per-game book refresh so a BRAND-NEW opponent is scouted from our second
  # game against them, not from the next session (LoblollyFreeplayv1 appeared
  # mid-session 2026-07-31 and would have stayed bookless until teardown).
  # Each game is its own process, so the next game re-reads the fresh book.
  # Cheap (~1-2s over all logs) and non-fatal by design, like the teardown's.
  .venv/bin/python showdown/scouting_book.py --name "$USERNAME" \
      showdown/bench/overnight_*_ladder.log >/dev/null 2>&1 || \
      echo "=== book refresh failed after game $g (non-fatal) ===" >> "$LOG"
  g=$((g + 1))
  sleep 3  # courtesy gap between ladder searches
done
wins=$(grep -c "finished: 1W" "$LOG")
losses=$(grep -c "finished: 0W / 1L" "$LOG")
echo "=== session $NAME complete: ${wins}W - ${losses}L of $N_GAMES ===" | tee -a "$LOG"
