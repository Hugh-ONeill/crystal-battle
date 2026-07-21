#!/bin/sh
# Run the exploit-probe suite against our live player on the LOCAL server.
# Each probe is one degenerate scripted policy; a probe WIN (or a crash,
# wedge, or turn-limit drag) is the finding — that is the whole point.
#
# Usage: showdown/run_probes.sh [games_per_probe]   (default 1)
# Probe results land in showdown/bench/probe_<policy>_ours.log
set -u
CB=/home/wiz/Developer/grimoire/crystal-battle
N="${1:-1}"
OUR_TEAM="${OUR_TEAM:-$CB/showdown/teams/suite_v1/03_dnite_tinglu_balance.txt}"
cd "$CB"

for policy in protect ppstall boom switchspam; do
  case "$policy" in
    boom) PT="$CB/showdown/teams/probes/probe_boom.txt" ;;
    *)    PT="$CB/showdown/teams/probes/probe_stall.txt" ;;
  esac
  LOG="$CB/showdown/bench/probe_${policy}_ours.log"
  rm -f "$LOG"
  echo "=== probe: $policy (n=$N) ==="
  timeout 1200 .venv/bin/python showdown/gen9_player.py --local --username CBProbe \
      --mode accept --format gen9ou --team "$OUR_TEAM" \
      --search-ms 300 --set-samples 2 --adaptive on \
      --n-games "$N" --log-level 20 > "$LOG" 2>&1 &
  sleep 8
  timeout 1000 .venv/bin/python showdown/exploit_probes.py --policy "$policy" \
      --n "$N" --username "PROBE$policy" --opponent CBProbe --team "$PT" 2>&1 | tail -1
  wait
  echo "  ours: $(grep -o 'finished:.*' "$LOG" | tail -1)"
  echo "  search failures: $(grep -c 'translate/search failed' "$LOG")" \
       "| collapses: $(grep -c 'world collapse' "$LOG")" \
       "| endgame solves: $(grep -c 'ENDGAME SOLVED' "$LOG")"
  sleep 3
done
echo "=== probe suite done ==="
