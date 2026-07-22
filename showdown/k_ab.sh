#!/bin/sh
# Fixed-K sweep: how many sampled opponent worlds should we search?
#
# K is currently pinned at --set-samples 2 (collapsing to 1 late). Two signals
# say more worlds may be worth real winrate: foul-play samples 2x worlds while
# the opponent's active has <3 revealed moves, and our own loss-trace says we
# lose EARLY vs foul-play (wins avg 127 turns, losses 68; they draw first blood
# in 34/52 losses) — exactly the phase where set uncertainty is highest and
# extra worlds hedge it.
#
# FAIRNESS NOTE: CB_PIN_CAPS=1 pins the per-decision millisecond budget, so
# every arm gets the SAME WALL CLOCK per turn while higher K uses proportionally
# more CPU across parallel threads. That is deliberate — a real game is bounded
# by the server clock, not by our core count, and we have 24 idle cores while
# foul-play's ProcessPool runs its extra worlds sequentially. Equal-wall-clock
# is the deployment-relevant comparison; it is NOT equal-CPU.
#
# Usage: showdown/k_ab.sh <name> <batches> <games_per_batch> [K values...]
set -u
NAME="$1"; BATCHES="$2"; GAMES="$3"; shift 3
KS="${*:-1 2 4}"
CB=/home/wiz/Developer/grimoire/crystal-battle
cd "$CB" || exit 1
SUITE=showdown/teams/suite_v1

for k in $KS; do
  echo "=== arm K=$k  ($(date +%H:%M:%S)) ==="
  sh showdown/ab_series.sh "${NAME}_k${k}" "$BATCHES" "$GAMES" \
      --suite "$SUITE" --set-samples "$k" >/dev/null 2>&1
done

echo
echo "=== RESULT (same teams, same wall-clock budget per decision) ==="
for k in $KS; do
  FP="$CB/showdown/bench/${NAME}_k${k}_foulplay.log"
  W=$(grep -c "^INFO     Winner: CBGen9" "$FP" 2>/dev/null); W=${W:-0}
  L=$(grep -c "^INFO     Winner: FPSpar1" "$FP" 2>/dev/null); L=${L:-0}
  N=$((W + L))
  printf "  K=%-2s : %2dW-%2dL of %2d" "$k" "$W" "$L" "$N"
  [ "$N" -gt 0 ] && printf "  (%s%%)" "$(echo "$W $N" | awk '{printf "%.1f", 100*$1/$2}')"
  printf "\n"
done
