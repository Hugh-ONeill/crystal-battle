#!/bin/bash
# Headless CBDemo runner for one recorded exhibition match vs FPDemo.
#
# CBDemo/FPDemo ONLY. Never CBGen9/FPSpar1/PAC-Crystal (same-name login kicks
# a running series or ladder bot offline). The demo names are also chosen so
# they are not a PREFIX of any reserved name: this harness pattern-kills by
# name, and 'FPSpar' would have matched the reserved 'FPSpar1'. Two A/B
# series were destroyed that way on 2026-07-27.
#
# TURN PACE STAYS ON (8s) — dropping it was TRIED AND FAILED 2026-07-27.
#
# The plan was that PTS scheduling (caster.py --pts-url) would replace it, so
# the engine could run full speed and stop burning our own clock, which is
# what makes pacing unusable on a real ladder under the PokeAgent Standard
# timer. PTS does keep individual LINES in sync. What it does not fix is the
# BEAT PRODUCTION RATE: Director.min_interval is 20s of WALL CLOCK, and with
# the engine unpaced a whole 40-turn game resolves in about three minutes, so
# the gate let through FIVE beats total (MATCH START, T2, T5, T8, T23) for a
# broadcast the viewer watches for 10+ minutes. Fifteen consecutive turns with
# no commentary at all, and a T8 -> T23 jump while the viewer was on turn 14.
#
# The fix is to gate beats on TURNS rather than seconds when PTS is on: the
# viewer watches every turn, so "am I talking too much" is a question about
# viewer time, and viewer time is proportional to turns, not to engine
# wall-clock. Until that lands, keep the pace hold. See TODO.
HERE="$(cd "$(dirname "$0")" && pwd)"
cd "$HERE/../.." || exit 1
PACE="${PRISM_TURN_PACE:-8}"
exec .venv/bin/python -u showdown/gen9_player.py \
  --local --username CBDemo --mode challenge --user-to-challenge FPDemo \
  --format gen9ou \
  --team "${PRISM_TEAM:-showdown/teams/suite_v1/03_dnite_tinglu_balance.txt}" \
  --n-games 1 --search-ms "${PRISM_SEARCH_MS:-2000}" \
  --airi --airi-turn-pace "$PACE"
