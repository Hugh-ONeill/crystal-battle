#!/bin/sh
# Paired A/B of the fork's EXTENDED evaluate() against the UPSTREAM-equivalent
# one (the eval foul-play actually runs), vs stock foul-play on the local server.
#
# Our src/genx/evaluate.rs is +456/-54 lines over v0.0.47: retuned hazards plus
# whole new term families (weather, terrain, Wish/Future Sight, Perish, Encore/
# Taunt/Yawn/Salt Cure, item table, threat-scaled boosts, hopeless-matchup,
# speed tier). Every constant is hand-picked, never validated against outcomes,
# and tuned for MONOTYPE while this is OU. This asks the obvious question that
# was never asked: do they help?
#
# CB_EVAL_BASELINE=1 reverts evaluate() to upstream's feature set + constants at
# RUNTIME, so both arms run one identical build and no rebuild happens mid-series.
# Verified the switch is real: eval outputs differ per position, and it costs
# only ~1% throughput (847k -> 856k it/s), so this measures eval CONTENT, not speed.
#
# Both arms play the same suite teams at the same pinned budget (CB_PIN_CAPS=1).
#
# Usage: showdown/eval_ab.sh <name> <batches> <games_per_batch>
set -u
NAME="$1"; BATCHES="$2"; GAMES="$3"
CB=/home/wiz/Developer/grimoire/crystal-battle
cd "$CB" || exit 1
SUITE=showdown/teams/suite_v1

echo "=== arm EXTENDED (our evaluate)  ($(date +%H:%M:%S)) ==="
unset CB_EVAL_BASELINE
sh showdown/ab_series.sh "${NAME}_ext" "$BATCHES" "$GAMES" \
    --suite "$SUITE" >/dev/null 2>&1

echo "=== arm BASELINE (upstream-equivalent evaluate)  ($(date +%H:%M:%S)) ==="
CB_EVAL_BASELINE=1 export CB_EVAL_BASELINE
sh showdown/ab_series.sh "${NAME}_base" "$BATCHES" "$GAMES" \
    --suite "$SUITE" >/dev/null 2>&1

echo
echo "=== RESULT ==="
for arm in ext base; do
  FP="$CB/showdown/bench/${NAME}_${arm}_foulplay.log"
  W=$(grep -c "^INFO     Winner: CBGen9" "$FP" 2>/dev/null); W=${W:-0}
  L=$(grep -c "^INFO     Winner: FPSpar1" "$FP" 2>/dev/null); L=${L:-0}
  N=$((W + L))
  printf "  %-5s : %2dW-%2dL of %2d\n" "$arm" "$W" "$L" "$N"
done
