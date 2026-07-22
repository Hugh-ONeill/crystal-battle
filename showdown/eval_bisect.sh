#!/bin/sh
# Overnight eval-term bisect: which fork-added eval term suppresses pressure
# play (the SubTox-Gliscor signal from the 2026-07-22 behavioral autopsy).
#
# Sequential arms of par_series, each with a different CB_EVAL_OFF value.
# The metric is BEHAVIOR (Gliscor Sub/Toxic rate via eval_bisect_metrics.py),
# which resolves in ~40 games — winrate at this n is noise and is recorded
# only so nobody has to ask. Two control arms anchor the chain: `ext` (no
# flags) must reproduce ~Sub 7%, `base` (everything off) ~Sub 15%; if the
# controls don't separate, the night's data is void — check them first.
#
# Usage: eval_bisect.sh            (BISECT_GAMES=48 BISECT_LANES=8 to override)
# Results: showdown/bench/eval_bisect_summary.txt (appended live, arm by arm)
set -u
CB=/home/wiz/Developer/grimoire/crystal-battle
cd "$CB" || exit 1
GAMES="${BISECT_GAMES:-48}"
LANES="${BISECT_LANES:-8}"
SUMMARY="$CB/showdown/bench/eval_bisect_summary.txt"
: > "$SUMMARY"
echo "eval bisect: $GAMES games x $LANES lanes per arm, started $(date '+%F %T')" \
    | tee -a "$SUMMARY"

run_arm() {  # $1 label, $2 CB_EVAL_BASELINE value, $3 CB_EVAL_OFF value
  echo "--- arm $1 (BASELINE='$2' OFF='$3') at $(date +%H:%M:%S)" >> "$SUMMARY"
  if CB_EVAL_BASELINE="$2" CB_EVAL_OFF="$3" \
      sh "$CB/showdown/par_series.sh" "evb_$1" "$GAMES" "$LANES" \
      --suite showdown/teams/suite_v1 >> "$SUMMARY" 2>&1; then
    "$CB/.venv/bin/python" "$CB/showdown/eval_bisect_metrics.py" "evb_$1" \
        >> "$SUMMARY" 2>&1
  else
    echo "ARM $1 FAILED — continuing with the rest" >> "$SUMMARY"
  fi
}

run_arm ext       ""  ""
run_arm base      "1" ""
run_arm hazards   ""  "hazards"
run_arm hopeless  ""  "hopeless"
run_arm threat    ""  "threat"
run_arm speedtier ""  "speedtier"
run_arm tera      ""  "tera"
run_arm volatiles ""  "volatiles"
run_arm items     ""  "items"
run_arm wt        ""  "weather,terrain"
run_arm pending   ""  "pending"
run_arm hazhope   ""  "hazards,hopeless"

echo "=== bisect chain complete at $(date '+%F %T') ===" | tee -a "$SUMMARY"
