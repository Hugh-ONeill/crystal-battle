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

g=1
wins=0
while [ "$g" -le "$N_GAMES" ]; do
  # rotate team: pseudo-random pick from the pool
  TEAM=$(ls "$POOL"/*.txt | shuf -n 1)
  echo "=== game $g/$N_GAMES team: $(basename "$TEAM" .txt) ($(date +%H:%M:%S)) ===" >> "$LOG"
  timeout "$PER_GAME_TIMEOUT" .venv/bin/python showdown/gen9_player.py \
      --server "$SERVER" --username "$USERNAME" \
      --mode ladder --format "$FORMAT" --team "$TEAM" \
      --n-games 1 --log-level 20 \
      "$@" >> "$LOG" 2>&1
  status=$?
  if [ "$status" -eq 124 ]; then
    echo "=== game $g TIMED OUT (skipped) ===" >> "$LOG"
  fi
  grep -q "finished: 1W" "$LOG" && :  # tally computed at the end
  g=$((g + 1))
  sleep 3  # courtesy gap between ladder searches
done
wins=$(grep -c "finished: 1W" "$LOG")
losses=$(grep -c "finished: 0W / 1L" "$LOG")
echo "=== session $NAME complete: ${wins}W - ${losses}L of $N_GAMES ===" | tee -a "$LOG"
