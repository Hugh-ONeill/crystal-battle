#!/bin/bash
# Headless CBAiri runner for one recorded exhibition match vs FPAiri.
# CBAiri/FPAiri only — never CBGen9/FPSpar1 (same-name login kicks a running
# series bot offline).
HERE="$(cd "$(dirname "$0")" && pwd)"
cd "$HERE/../.." || exit 1
exec .venv/bin/python -u showdown/gen9_player.py \
  --local --username CBAiri --mode challenge --user-to-challenge FPAiri \
  --format gen9ou \
  --team "${PRISM_TEAM:-showdown/teams/suite_v1/03_dnite_tinglu_balance.txt}" \
  --n-games 1 --search-ms "${PRISM_SEARCH_MS:-2000}" \
  --airi --airi-turn-pace "${PRISM_TURN_PACE:-8}"
