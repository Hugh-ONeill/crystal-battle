#!/bin/sh
# Paired tree-reuse A/B vs stock foul-play on the local server.
#
# Runs the SAME suite twice — once with --tree-reuse on, once off — so batch i
# of each arm plays the identical team (ab_series cycles the sorted suite by
# batch index). Everything else is held fixed, including CB_PIN_CAPS=1, which
# pins the budget caps to CB_SEARCH_MS: both arms get the same milliseconds per
# decision, so a difference measures search QUALITY at equal compute, which is
# exactly what reuse is supposed to buy (retained visits add depth for free).
#
# Reuse rides the K==1 path, so --set-samples 1 keeps it engaged from turn 1
# instead of only after the late-game world collapse. Both arms use it.
#
# Usage: showdown/reuse_ab.sh <name> <batches> <games_per_batch>
set -u
NAME="$1"; BATCHES="$2"; GAMES="$3"
CB=/home/wiz/Developer/grimoire/crystal-battle
cd "$CB" || exit 1
SUITE=showdown/teams/suite_v1
COMMON="--suite $SUITE --set-samples 1"

for arm in on off; do
  echo "=== arm --tree-reuse $arm  ($(date +%H:%M:%S)) ==="
  sh showdown/ab_series.sh "${NAME}_$arm" "$BATCHES" "$GAMES" \
      $COMMON --tree-reuse "$arm" >/dev/null 2>&1
done

echo
echo "=== RESULT (wins are ours; foul-play log records the winner) ==="
for arm in on off; do
  W=$(grep -c "^INFO     Winner: CBGen9" "$CB/showdown/bench/${NAME}_${arm}_foulplay.log" 2>/dev/null || echo 0)
  L=$(grep -c "^INFO     Winner: FPSpar1" "$CB/showdown/bench/${NAME}_${arm}_foulplay.log" 2>/dev/null || echo 0)
  N=$((W + L))
  K=$(grep -c "tree reuse: kept" "$CB/showdown/bench/${NAME}_${arm}_ours.log" 2>/dev/null || echo 0)
  F=$(grep -c "tree reuse: fresh" "$CB/showdown/bench/${NAME}_${arm}_ours.log" 2>/dev/null || echo 0)
  printf "  reuse %-3s : %2dW-%2dL of %2d" "$arm" "$W" "$L" "$N"
  [ "$N" -gt 0 ] && printf "  (%.1f%%)" "$(echo "$W $N" | awk '{print 100*$1/$2}')"
  printf "   reuse events kept=%s fresh=%s\n" "$K" "$F"
done
