#!/bin/sh
# Paired ladder A/B: alternates one gen9_player flag setting between games so
# both arms face the SAME opponent pool in the SAME time window. The ladder's
# opponent mix drifts (baselines cycle in and out), which is exactly what a
# historical before/after comparison cannot control for — alternating does.
#
# Each PAIR uses the same team: game 2k = control (arm B), 2k+1 = treatment
# (arm A), same roster, so team matchup variance cancels within the pair.
#
# Usage:
#   showdown/ladder_ab.sh <name> <pairs> <pool_dir> "<A args>" "<B args>" [common args...]
# e.g. priors on vs off:
#   showdown/ladder_ab.sh priors 15 showdown/teams/pool_hl \
#       "--opp-priors on" "--opp-priors off" --adaptive on
set -u
NAME="$1"; PAIRS="$2"; POOL="$3"; A_ARGS="$4"; B_ARGS="$5"; shift 5
CB=/home/wiz/Developer/grimoire/crystal-battle
cd "$CB" || exit 1

CRED_FILE="$HOME/.config/crystal-battle/pokeagent.env"
if [ -z "${PS_PASSWORD:-}" ] && [ -f "$CRED_FILE" ]; then
  set -a; . "$CRED_FILE"; set +a
fi
USERNAME="${PS_USERNAME:-PAC-Crystal}"
FORMAT="${LADDER_FORMAT:-gen9oulongtimer}"
PER_GAME_TIMEOUT="${PER_GAME_TIMEOUT:-900}"
A_LOG="$CB/showdown/bench/${NAME}_A_ladder.log"
B_LOG="$CB/showdown/bench/${NAME}_B_ladder.log"
rm -f "$A_LOG" "$B_LOG"

# preflight: a second same-name stream would kick this one off the server
if pgrep -f "ladder_session[.]sh" >/dev/null 2>&1 || \
   pgrep -f "gen9_player[.]py .*--mode ladder" >/dev/null 2>&1; then
  echo "FATAL: another ladder stream is live as $USERNAME" >&2; exit 1
fi

echo "=== paired ladder A/B '$NAME': $PAIRS pairs on $FORMAT as $USERNAME ==="
echo "  A: $A_ARGS"
echo "  B: $B_ARGS"

p=1
while [ "$p" -le "$PAIRS" ]; do
  TEAM=$(ls "$POOL"/*.txt | shuf -n 1)      # one team for BOTH games of the pair
  for arm in A B; do
    if [ "$arm" = "A" ]; then EXTRA="$A_ARGS"; LOG="$A_LOG"; else EXTRA="$B_ARGS"; LOG="$B_LOG"; fi
    echo "=== pair $p/$PAIRS arm $arm team: $(basename "$TEAM" .txt) ($(date +%H:%M:%S)) ===" >> "$LOG"
    timeout "$PER_GAME_TIMEOUT" .venv/bin/python showdown/gen9_player.py \
        --server pokeagent --username "$USERNAME" \
        --mode ladder --format "$FORMAT" --team "$TEAM" \
        --n-games 1 --log-level 20 $EXTRA "$@" >> "$LOG" 2>&1
    sleep 3
  done
  a_w=$(grep -c "finished: 1W" "$A_LOG"); a_l=$(grep -c "finished: 0W" "$A_LOG")
  b_w=$(grep -c "finished: 1W" "$B_LOG"); b_l=$(grep -c "finished: 0W" "$B_LOG")
  echo "  after pair $p:  A ${a_w}W-${a_l}L   B ${b_w}W-${b_l}L"
  p=$((p + 1))
done
echo "=== A/B '$NAME' complete ==="
