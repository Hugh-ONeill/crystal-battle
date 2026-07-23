#!/bin/sh
# Eval-bisect round 2: CONFIRMATION on a Gliscor-only suite. Round 1 (48
# games/arm, full suite_v1) split bimodally — Sub restored under hopeless /
# threat / tera / weather+terrain / pending, suppressed under the rest — but
# restored arms also had 1.5-2x the Gliscor moves, so the metric was
# entangled with game mix (only 3 of 8 teams carry Gliscor). Here EVERY game
# is a Gliscor mirror, so per-arm Sub% rests on ~32 Gliscor games instead of
# ~18, and the restored-cluster arms rerun against both controls.
#
# Results append to showdown/bench/eval_bisect2_summary.txt
set -u
CB=/home/wiz/Developer/grimoire/crystal-battle
cd "$CB" || exit 1
GAMES="${BISECT_GAMES:-32}"
LANES="${BISECT_LANES:-8}"
SUMMARY="$CB/showdown/bench/eval_bisect2_summary.txt"
: > "$SUMMARY"
echo "eval bisect round 2 (gliscor suite): $GAMES games x $LANES lanes per arm," \
     "started $(date '+%F %T')" | tee -a "$SUMMARY"

run_arm() {  # $1 label, $2 CB_EVAL_BASELINE, $3 CB_EVAL_OFF
  echo "--- arm $1 (BASELINE='$2' OFF='$3') at $(date +%H:%M:%S)" >> "$SUMMARY"
  if CB_EVAL_BASELINE="$2" CB_EVAL_OFF="$3" \
      sh "$CB/showdown/par_series.sh" "evc_$1" "$GAMES" "$LANES" \
      --suite showdown/teams/suite_gliscor >> "$SUMMARY" 2>&1; then
    "$CB/.venv/bin/python" "$CB/showdown/eval_bisect_metrics.py" "evc_$1" \
        >> "$SUMMARY" 2>&1
  else
    echo "ARM $1 FAILED — continuing" >> "$SUMMARY"
  fi
}

run_arm ext      ""  ""
run_arm base     "1" ""
run_arm hopeless ""  "hopeless"
run_arm threat   ""  "threat"
run_arm tera     ""  "tera"
run_arm wt       ""  "weather,terrain"
run_arm pending  ""  "pending"

echo "=== bisect round 2 complete at $(date '+%F %T') ===" | tee -a "$SUMMARY"
