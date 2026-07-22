#!/bin/sh
# Paired A/B of the multi-world MERGE RULE, vs stock foul-play.
#
# _merge_mcts_results used to sum RAW visits across sampled worlds, which
# silently weighted each world by how fast it happened to simulate — a cheaper
# position accumulates more iterations in the same milliseconds and so got a
# louder vote. Commit 9d503ab changed it to contribute VISIT SHARE instead.
#
# That is principled, but it is unmeasured, and it confounds every multi-world
# result taken since: pooled K=2 arms went 24W-48L (33.3%) BEFORE the change
# and 8W-28L (22.2%) after, Fisher p=0.270 — unresolved, but the wrong
# direction. Principle has lost to measurement twice already (tree reuse, tera
# suppression), so this settles it.
#
# K IS PINNED AT 2: with a single world the merge is provably the identity, so
# the arms would be bit-identical and the A/B would measure nothing.
# CB_MERGE_RAW=1 selects the legacy rule per process; one build, both arms.
#
# Usage: showdown/merge_ab.sh <name> <batches> <games_per_batch>
set -u
NAME="$1"; BATCHES="$2"; GAMES="$3"
CB=/home/wiz/Developer/grimoire/crystal-battle
cd "$CB" || exit 1
SUITE=showdown/teams/suite_v1

echo "=== arm NORMALIZED (visit share, current default)  ($(date +%H:%M:%S)) ==="
unset CB_MERGE_RAW
sh showdown/ab_series.sh "${NAME}_norm" "$BATCHES" "$GAMES" \
    --suite "$SUITE" --set-samples 2 >/dev/null 2>&1

echo "=== arm RAW (legacy visit summing)  ($(date +%H:%M:%S)) ==="
CB_MERGE_RAW=1 export CB_MERGE_RAW
sh showdown/ab_series.sh "${NAME}_raw" "$BATCHES" "$GAMES" \
    --suite "$SUITE" --set-samples 2 >/dev/null 2>&1

echo
echo "=== RESULT (same teams, same budget, K=2 both arms) ==="
for arm in norm raw; do
  FP="$CB/showdown/bench/${NAME}_${arm}_foulplay.log"
  W=$(grep -c "^INFO     Winner: CBGen9" "$FP" 2>/dev/null); W=${W:-0}
  L=$(grep -c "^INFO     Winner: FPSpar1" "$FP" 2>/dev/null); L=${L:-0}
  N=$((W + L))
  printf "  %-4s : %2dW-%2dL of %2d" "$arm" "$W" "$L" "$N"
  [ "$N" -gt 0 ] && printf "  (%s%%)" "$(echo "$W $N" | awk '{printf "%.1f", 100*$1/$2}')"
  printf "\n"
done
