#!/bin/sh
# Self-contained overnight value-net escalation A/B. Runs two matched arms
# back to back under one unit, no human interaction required:
#   arm A (control):   adaptive escalation with plain deep MCTS
#   arm B (treatment): adaptive escalation with the value net in the deep-think
# Everything else identical (rationed bank, no-op filters, slow mirrors), so
# the A/B - B difference isolates the value net's contribution.
#
# Robustness for unattended running:
#   - ensures the local Showdown server is up before EACH arm (restarts a
#     dead server so a mid-night crash can't strand the second arm)
#   - ab_series.sh already bounds each batch with a hard timeout and skips a
#     wedged batch, so a poke-env dispatch freeze costs one batch, not the run
#
# Launch (suspend-inhibited, memory-capped):
#   systemd-run --user --unit=cb-overnight -p MemoryMax=8G \
#     -p WorkingDirectory=$PWD --quiet /bin/sh -c \
#     'exec systemd-inhibit --mode=block --who=crystal-battle \
#        --why="overnight value-net A/B" showdown/overnight_valuenet_ab.sh'

set -u
CB=/home/wiz/Developer/grimoire/crystal-battle
PS=/home/wiz/Developer/grimoire/pokemon-showdown
SERVER_LOG=/tmp/claude-1000/-home-wiz/2832b258-9bba-45cb-b93a-55ebf27a6b56/scratchpad/showdown-server.log

ensure_server() {
  "$CB/.venv/bin/python" -c \
    "import socket; socket.create_connection(('127.0.0.1', 8000), 2).close()" \
    2>/dev/null && return 0
  echo "[overnight] server down; restarting" >&2
  cd "$PS"
  setsid nohup node pokemon-showdown start 8000 --no-security \
    > "$SERVER_LOG" 2>&1 &
  i=0
  while [ "$i" -lt 40 ]; do
    "$CB/.venv/bin/python" -c \
      "import socket; socket.create_connection(('127.0.0.1', 8000), 2).close()" \
      2>/dev/null && return 0
    sleep 1; i=$((i + 1))
  done
  echo "[overnight] server failed to come up" >&2
  return 1
}

BATCHES=2
GAMES=20
SUITE=showdown/teams/suite_slow
COMMON="--suite $SUITE --data-tiers on --stochastic on --adaptive on --escalate-ms 2000"

cd "$CB"

echo "[overnight] === ARM A: control (no value net) $(date +%H:%M) ==="
ensure_server || exit 1
showdown/ab_series.sh vnab_ctrl "$BATCHES" "$GAMES" $COMMON

echo "[overnight] === ARM B: value net in escalation $(date +%H:%M) ==="
ensure_server || exit 1
showdown/ab_series.sh vnab_val "$BATCHES" "$GAMES" $COMMON \
  --value-net showdown/value_net_gen9_v3.onnx --value-alpha 0.5 --value-batch 32

echo "[overnight] === value-net A/B complete $(date +%H:%M) ==="
