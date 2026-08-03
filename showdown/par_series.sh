#!/bin/sh
# Parallel bench series vs stock foul-play: N lanes pulling games off a QUEUE.
#
# WHY A QUEUE (measured 2026-07-22, parthru2). The static split (games_per_lane
# fixed per lane) delivered only 3.2x effective parallelism from 6 lanes: game
# length varies 6x (32..192 turns), so the run's wall clock was the unluckiest
# lane's 427s while four lanes idled for minutes. Per-TURN cost under 6-way
# concurrency is identical to sequential (0.89 vs 0.91 s/turn; server wait
# 439 vs 436 ms/turn) — the local Showdown server is NOT a bottleneck at this
# scale, and CPU sits under 3 of 24 cores. The losses are (1) lane imbalance,
# fixed here by the queue, and (2) per-game process startup, fixed by ONE
# persistent worker per lane (--team-reload rotates the team through a fixed
# per-lane file, so the worker no longer restarts per game; foul-play still
# boots per game — its side is 32MB/3.5s, not worth touching). The queue is
# also where SPRT early-stop belongs: stop handing out games once the series
# can conclude.
#
# LANE ISOLATION. Each lane needs its own username pair — a same-name login
# KICKS the running bot, which would silently corrupt every other lane. Lane k
# runs as CBGen9L<k> vs FPSpar1L<k>; both keep the CBGen9 / FPSpar1 PREFIX so
# the existing tallies (grep "Winner: CBGen9") still match unchanged. Team
# files are copied per-game too, or concurrent lanes would race on one path.
#
# SPRT EARLY-STOP (--sprt P0 P1). Bernoulli SPRT gate in the queue dispenser:
# H0 winrate<=P0 vs H1 winrate>=P1 on decided games (showdown/sprt.py). The
# dispenser checks the running W-L before handing out each game and stops the
# series on a verdict — in-flight games still finish and count. TOTAL becomes
# the inconclusive cap, and the launch gate REFUSES a series whose cap is
# below the Wald expected n (the noise-floor rule: don't start what can't
# conclude; SPRT_FORCE=1 overrides). alpha/beta via SPRT_ALPHA/SPRT_BETA,
# default 0.05.
#
# The tally is DISPENSE-ORDERED (showdown/dispense_tally.py): only the
# contiguous decided prefix of dispense indices counts, because completion
# order is biased — short games decide first and short games skew losses, so
# a completion-order tally leans accept-h0 early in a run (bit the 2026-07-22
# certification gate at n=30). Dispense order is outcome-independent, so the
# LLR follows the path a purely sequential run would produce, with lag: a
# marathon at a low index delays the verdict but never distorts it.
#
# Usage: par_series.sh <name> <total_games> <lanes> [--suite DIR]
#            [--sprt P0 P1] [--ab "VAR=v[ VAR2=v2]"] [--ab-sprt P0 P1]
#            [our args...]
#        (arg 2 is the TOTAL game count now, not games per lane)
#
# INTERLEAVED A/B (--ab). Games are dealt in PAIRS on the same suite team:
# odd global index = arm A (baseline env), even = arm B (baseline + the --ab
# spec). Each lane keeps TWO persistent workers (CBGen9L<k>A / CBGen9L<k>B)
# because CB_* engine knobs are OnceLock-read once per process — process
# isolation is the config isolation; no runtime setters needed. Any
# run-scoped confound hits both arms alternately and cancels in the paired
# difference: single-arm gates vs historical levels died three times in one
# week (2026-07-22 x2, 2026-07-24 level step); this design is why. --ab-sprt
# gates on the A-share of DISCORDANT pairs (paired_tally.py, dispense-order
# pair prefix): H0 share<=P0 vs H1 share>=P1, 0.5 = no difference; accept-h1
# reads "A better / B worse", accept-h0 the reverse. RAM: two workers per
# lane at ~851MB each — size lanes to the box.
# Tally: grep -c "^INFO     Winner: CBGen9" showdown/bench/<name>_L*_foulplay.log
set -u
CB_ABS=/home/wiz/Developer/grimoire/crystal-battle
NAME="$1"; TOTAL="$2"; LANES="$3"; shift 3
SUITE_DIR=""; FP_SUITE_DIR=""; SPRT_P0=""; SPRT_P1=""; AB_ENV=""; AB_P0=""; AB_P1=""
OPPONENT="foulplay"
while :; do
  case "${1:-}" in
    --suite) SUITE_DIR="$2"; shift 2 ;;
    --fp-suite) FP_SUITE_DIR="$2"; shift 2 ;;
    --sprt)  SPRT_P0="$2"; SPRT_P1="$3"; shift 3 ;;
    --ab)      AB_ENV="$2"; shift 2 ;;
    --ab-sprt) AB_P0="$2"; AB_P1="$3"; shift 3 ;;
    # --opponent spar: challenge with the predictable human-policy sparring bot
    # (showdown/spar_bot.py) instead of foul-play. It plays the SAME human
    # distribution our opp-net predicts, so the opponent-policy prior has a real
    # target to exploit (neutral vs near-optimal fp), AND it never chokes on our
    # slow escalation turns (fp's KeyError 'battle') — unblocking grind-depth.
    # Everything else (--ab pairing, SPRT gates, tally) is opponent-agnostic:
    # spar_bot emits fp's exact "INFO     Winner:" line into the same log.
    --opponent) OPPONENT="$2"; shift 2 ;;
    *) break ;;
  esac
done
case "$OPPONENT" in foulplay|spar) ;; *)
  echo "FATAL: --opponent must be 'foulplay' or 'spar'" >&2; exit 1 ;;
esac
if [ -n "$AB_ENV" ]; then
  if [ -n "$SPRT_P0" ]; then
    echo "FATAL: --sprt and --ab are mutually exclusive; the A/B gate is --ab-sprt" >&2
    exit 1
  fi
  if [ $((TOTAL % 2)) -ne 0 ]; then
    echo "FATAL: --ab deals games in pairs; TOTAL must be even" >&2
    exit 1
  fi
fi
if [ -n "$AB_P0" ] && [ -z "$AB_ENV" ]; then
  echo "FATAL: --ab-sprt without --ab" >&2
  exit 1
fi
# A missing --suite silently ran EVERY game on the single legacy sample team,
# which contaminated the 2026-07-23 phgate/recert/syngate series (single-team
# level presented as suite level; game-length claims confounded). Make that
# an explicit choice, never a default.
if [ -z "$SUITE_DIR" ] && [ "${CB_ALLOW_LEGACY_TEAM:-0}" != "1" ]; then
  echo "FATAL: no --suite given — this runs every game on the ONE legacy" \
       "sample team (suite dirs live in showdown/teams/). Set" \
       "CB_ALLOW_LEGACY_TEAM=1 if that is really what you want." >&2
  exit 1
fi
# MUST be absolute: each game cd's into $FP to run foul-play, so a relative
# suite path stops resolving from game 2 onward — which silently produced an
# empty team name and killed every game after the first in each lane.
case "$SUITE_DIR" in
  ""|/*) ;;
  *) SUITE_DIR="$CB_ABS/$SUITE_DIR" ;;
esac
case "$FP_SUITE_DIR" in
  ""|/*) ;;
  *) FP_SUITE_DIR="$CB_ABS/$FP_SUITE_DIR" ;;
esac

# Username collision guard (2026-08-03). Worker names are fixed strings, so
# a run that is killed mid-flight leaves the SERVER holding that name's battle
# and the next run's worker re-joins the stale game on login instead of
# starting a fresh one — observed live: two smoke runs in a row silently
# joined battle-gen9ou-17908 and never reached turn 1. Setting
# CB_BENCH_TAG=<short> gives a run its own namespace without touching anyone
# else's server or processes. Default empty = names unchanged.
UTAG="${CB_BENCH_TAG:-}"
export PYTHONUNBUFFERED=1
CB=/home/wiz/Developer/grimoire/crystal-battle
FP=/home/wiz/Developer/grimoire/foul-play
PER_GAME_TIMEOUT="${PER_GAME_TIMEOUT:-1200}"
cd "$CB" || exit 1

if ! "$CB/.venv/bin/python" -c \
    "import socket; socket.create_connection(('127.0.0.1', 8000), 2).close()" \
    2>/dev/null; then
  echo "FATAL: local Showdown server not up on :8000" >&2; exit 1
fi

# tenancy preflight (2026-08-01): a 9G bench sharing a 30G box with resident
# Gemma and a Blender session took the whole SYSTEM down (reboot required).
# Non-fatal — the operator may know better — but say it loudly.
AVAIL_G=$(awk '/MemAvailable/{print int($2/1048576)}' /proc/meminfo)
if [ "$AVAIL_G" -lt 12 ]; then
  echo "WARNING: only ${AVAIL_G}G MemAvailable — a 9G bench on a busy box" >&2
  echo "         risks a SYSTEM oom (2026-08-01 reboot). Consider waiting." >&2
fi
if pgrep -f "gen9_player[.]py .*--mode accept" >/dev/null 2>&1; then
  echo "FATAL: a bench series is already running" >&2; exit 1
fi

if [ -n "$SUITE_DIR" ]; then
  N_TEAMS=$(ls "$SUITE_DIR"/*.txt | wc -l)
  mkdir -p "$FP/teams/teams/gen9/ou/suite"
else
  N_TEAMS=0
fi
# --fp-suite: foul-play draws from a SECOND dir (same global-index rotation)
# instead of mirroring our team — for cross-matchup arms (e.g. our stall vs
# their offense). Without it, mirror behavior is unchanged.
if [ -n "$FP_SUITE_DIR" ]; then
  FP_N_TEAMS=$(ls "$FP_SUITE_DIR"/*.txt | wc -l)
else
  FP_N_TEAMS=0
fi

QUEUE="$CB/showdown/bench/${NAME}.queue"
echo 0 > "$QUEUE"
rm -f "$QUEUE.verdict"
tally() {  # $1 = winner-name prefix; logs may not exist yet on game 1
  cat "$CB"/showdown/bench/${NAME}_L*_foulplay.log 2>/dev/null \
    | grep -c "^INFO     Winner: $1"
}
# Hand out the next 1-based global game index, or fail when the run is done —
# either the cap is reached or the SPRT gate has concluded. flock serializes
# the read-increment-write; lanes calling this concurrently each get a
# distinct index exactly once.
next_game() {
  (
    flock -x 9
    [ -f "$QUEUE.verdict" ] && exit 1
    if [ -n "$SPRT_P0" ]; then
      T=$("$CB/.venv/bin/python" "$CB/showdown/dispense_tally.py" \
          "$CB/showdown/bench" "$NAME") || T="0 0 0 0"
      W=$(echo "$T" | awk '{print $1}'); L=$(echo "$T" | awk '{print $2}')
      V=$("$CB/.venv/bin/python" "$CB/showdown/sprt.py" "$W" "$L" \
          "$SPRT_P0" "$SPRT_P1" "${SPRT_ALPHA:-0.05}" "${SPRT_BETA:-0.05}")
      case "$V" in
        accept*)
          P=$(echo "$T" | awk '{print $3}'); D=$(echo "$T" | awk '{print $4}')
          echo "$V after ${W}W-${L}L (dispense prefix $P, $D decided overall)" \
              > "$QUEUE.verdict"
          exit 1 ;;
      esac
    fi
    if [ -n "$AB_P0" ]; then
      T=$("$CB/.venv/bin/python" "$CB/showdown/paired_tally.py" \
          "$CB/showdown/bench" "$NAME") || T="0 0 0 0 0 0 0 0"
      nA=$(echo "$T" | awk '{print $1}'); nB=$(echo "$T" | awk '{print $2}')
      V=$("$CB/.venv/bin/python" "$CB/showdown/sprt.py" "$nA" "$nB" \
          "$AB_P0" "$AB_P1" "${SPRT_ALPHA:-0.05}" "${SPRT_BETA:-0.05}")
      case "$V" in
        accept*)
          P=$(echo "$T" | awk '{print $3}')
          echo "$V on discordant pairs: A won $nA, B won $nB (pair prefix $P)" \
              > "$QUEUE.verdict"
          exit 1 ;;
      esac
    fi
    n=$(($(cat "$QUEUE") + 1))
    [ "$n" -gt "$TOTAL" ] && exit 1
    echo "$n" > "$QUEUE"
    echo "$n"
  ) 9>>"$QUEUE.lock"
}

echo "=== parallel series '$NAME': $TOTAL games over $LANES lanes at $(date +%H:%M:%S) ==="
if [ -n "$SPRT_P0" ]; then
  EN=$("$CB/.venv/bin/python" "$CB/showdown/sprt.py" --expected-n \
       "$SPRT_P0" "$SPRT_P1" "${SPRT_ALPHA:-0.05}" "${SPRT_BETA:-0.05}") || exit 1
  echo "    SPRT gate: H0 p<=$SPRT_P0 vs H1 p>=$SPRT_P1" \
       "(alpha=${SPRT_ALPHA:-0.05} beta=${SPRT_BETA:-0.05}), expected n ~ $EN"
  if [ "$TOTAL" -lt "$EN" ] && [ "${SPRT_FORCE:-0}" != "1" ]; then
    echo "FATAL: cap $TOTAL < expected n $EN — this series likely cannot" \
         "conclude; raise the cap or set SPRT_FORCE=1" >&2
    exit 1
  fi
fi
if [ -n "$AB_ENV" ]; then
  echo "    interleaved A/B: arm A = baseline, arm B = env '$AB_ENV'" \
       "(pairs share a team; odd game = A, even = B)"
  if [ -n "$AB_P0" ]; then
    EN=$("$CB/.venv/bin/python" "$CB/showdown/sprt.py" --expected-n \
         "$AB_P0" "$AB_P1" "${SPRT_ALPHA:-0.05}" "${SPRT_BETA:-0.05}") || exit 1
    # discordant rate assumed 2*L*(1-L) at ambient level L~0.25 -> 0.375
    NEED=$(awk -v e="$EN" 'BEGIN{printf "%d", e / 0.375 + 1}')
    echo "    paired SPRT gate: H0 A-share<=$AB_P0 vs H1 >=$AB_P1 on" \
         "discordant pairs (0.5 = no difference); expected ~$EN discordant" \
         "= ~$NEED pairs = ~$((NEED * 2)) games at 0.375 discordance"
    if [ $((TOTAL / 2)) -lt "$NEED" ] && [ "${SPRT_FORCE:-0}" != "1" ]; then
      echo "FATAL: cap $((TOTAL / 2)) pairs < ~$NEED needed — raise the cap" \
           "or set SPRT_FORCE=1" >&2
      exit 1
    fi
  fi
fi

if [ "${CB_PIN_CAPS:-1}" = "1" ]; then
  CB_CAPS="--base-max-ms ${CB_SEARCH_MS:-300} --grind-max-ms ${CB_SEARCH_MS:-300}"
else
  CB_CAPS=""
fi

# CB_ZYGOTE=1: fork workers from one warm data image (worker_zygote.py)
# instead of cold-loading ~1.1GB per worker (replay index 768M + chaos +
# PS sets). Measured 2026-07-25: 16 cold workers = 17GB; zygote fleet
# shares the read-only data COW. Env-flag A/B arms share one zygote (env
# applied post-fork, before any engine OnceLock read); PYTHONPATH
# build-shadow arms CANNOT (module already loaded pre-fork) and are refused.
ZY_FIFO=""; ZY_PID=""; ZY_READY=""; LANE_PIDS=""

# Tear down the zygote and the run's scratch files. Runs on normal exit AND on
# INT/TERM, so an interrupted series doesn't leak the warm-image process —
# which, under the usual `systemd-inhibit ... sh par_series.sh` launch, also
# means the sleep block gets dropped. Idempotent: the EXIT trap re-runs it
# after an INT/TERM path already did.
cleanup() {
  for p in $LANE_PIDS; do kill "$p" 2>/dev/null; done
  LANE_PIDS=""
  if [ -n "$ZY_PID" ]; then
    timeout 5 sh -c "echo shutdown > '$ZY_FIFO'" 2>/dev/null
    sleep 0.5
    kill "$ZY_PID" 2>/dev/null
    wait "$ZY_PID" 2>/dev/null
    ZY_PID=""
  fi
  [ -n "$ZY_FIFO" ] && rm -f "$ZY_FIFO" "$ZY_READY" \
      "$CB"/showdown/bench/"${NAME}"_L*.zyreq \
      "$CB"/showdown/bench/"${NAME}"_L*.zyresp
  rm -f "$QUEUE" "$QUEUE.lock" "$QUEUE.verdict"
}
trap 'cleanup' EXIT
trap 'cleanup; exit 130' INT
trap 'cleanup; exit 143' TERM

# DEFAULT-ON 2026-07-27 (validated end-to-end at 16 lanes: ~5.6GB fleet vs ~17GB
# cold, private bounded to ~440MB even on a 294-turn stall game). Set CB_ZYGOTE=0
# to force cold workers. Build-shadow (PYTHONPATH) arms can't fork-share the
# pre-loaded engine module, so they GRACEFULLY fall back to cold workers here.
if [ "${CB_ZYGOTE:-1}" = "1" ]; then
  case "$AB_ENV" in
    *PYTHONPATH*)
      echo "    note: CB_ZYGOTE auto-disabled (PYTHONPATH build-shadow arm can't" \
           "fork-share the pre-loaded engine; using cold workers)" >&2 ;;
    *)
      ZY_FIFO="$CB/showdown/bench/${NAME}.zyfifo"
      ZY_READY="$CB/showdown/bench/${NAME}.zyready"
      rm -f "$ZY_FIFO" "$ZY_READY"
      "$CB/.venv/bin/python" "$CB/showdown/worker_zygote.py" --fifo "$ZY_FIFO" \
          --ready-file "$ZY_READY" \
          > "$CB/showdown/bench/${NAME}_zygote.log" 2>&1 &
      ZY_PID=$!
      i=0
      while [ ! -s "$ZY_READY" ] && [ "$i" -lt 240 ]; do
        sleep 0.5; i=$((i + 1))
      done
      if [ ! -s "$ZY_READY" ]; then
        echo "FATAL: zygote not ready after 120s (see ${NAME}_zygote.log)" >&2
        kill "$ZY_PID" 2>/dev/null
        exit 1
      fi
      echo "    zygote up (pid $ZY_PID): workers fork from one warm data image" ;;
  esac
fi

lane=1
while [ "$lane" -le "$LANES" ]; do
  (
    # Stagger lane starts: mirror arms launched in lockstep reach the same
    # game state at the same wall-clock moment across all lanes — at
    # 2026-07-23 17:50:08 all TEN foul-play lanes froze on (plausibly) the
    # same pathological position simultaneously and timer-forfeited at once,
    # fabricating 10 wins in one series. A few seconds of skew decorrelates.
    sleep $(( (lane - 1) * 3 ))
    OURS_LOG="$CB/showdown/bench/${NAME}_L${lane}_ours.log"
    FP_LOG="$CB/showdown/bench/${NAME}_L${lane}_foulplay.log"
    : > "$FP_LOG"
    THEM="FPSpar1${UTAG}L${lane}"
    if [ -n "$AB_ENV" ]; then
      : > "$CB/showdown/bench/${NAME}_L${lane}A_ours.log"
      : > "$CB/showdown/bench/${NAME}_L${lane}B_ours.log"
    else
      : > "$OURS_LOG"
    fi
    # ONE persistent worker per lane (851MB of replay sets/priors/nets and
    # ~4s of startup per process — load once, play the whole lane). The team
    # rotates per game by swapping a fixed file; --team-reload makes the
    # worker re-read it at every challenge accept (/utm) and team preview.
    # Under --ab there are TWO workers (arm suffix A/B in username, team
    # file, and log); arm B's process carries $AB_ENV baked into its env
    # since CB_* knobs are OnceLock-read once per process. Single-arm state
    # lives in the A slots (empty suffix, ${arm:-A}).
    OURS_PID_A=""; OURS_PID_B=""; starts_A=0; starts_B=0
    start_worker() {  # $1 = arm suffix: "" (single), "A", or "B"
      arm="$1"; shift
      cd "$CB"
      W_LOG="$CB/showdown/bench/${NAME}_L${lane}${arm}_ours.log"
      W_TEAM="$CB/showdown/bench/${NAME}_L${lane}${arm}.team"
      W_ENV=""
      [ "$arm" = "B" ] && W_ENV="$AB_ENV"
      if [ -n "$ZY_FIFO" ]; then
        # zygote spawn: request file in, child pid out
        REQ="$CB/showdown/bench/${NAME}_L${lane}${arm}.zyreq"
        RESP="$CB/showdown/bench/${NAME}_L${lane}${arm}.zyresp"
        rm -f "$RESP"
        {
          echo "resp=$RESP"
          echo "log=$W_LOG"
          for kv in $W_ENV; do echo "env $kv"; done
          for tok in --local --username "CBGen9${UTAG}L${lane}${arm}" --mode accept \
              --format gen9ou --team "$W_TEAM" --team-reload on \
              --search-ms "${CB_SEARCH_MS:-300}" \
              $CB_CAPS --n-games 999 --log-level 20; do
            echo "arg $tok"
          done
          for tok in "$@"; do echo "arg $tok"; done
        } > "$REQ"
        echo "$REQ" > "$ZY_FIFO"
        i=0
        while [ ! -s "$RESP" ] && [ "$i" -lt 60 ]; do
          sleep 0.25; i=$((i + 1))
        done
        eval "OURS_PID_${arm:-A}=\$(cat \"$RESP\" 2>/dev/null)"
      else
        env $W_ENV .venv/bin/python showdown/gen9_player.py --local \
            --username "CBGen9${UTAG}L${lane}${arm}" \
            --mode accept --format gen9ou --team "$W_TEAM" --team-reload on \
            --search-ms "${CB_SEARCH_MS:-300}" \
            $CB_CAPS --n-games 999 --log-level 20 \
            "$@" >> "$W_LOG" 2>&1 &
        eval "OURS_PID_${arm:-A}=$!"
      fi
      # Readiness probe instead of a blind sleep 5: the worker logs
      # "Starting listening" ~1s in and is logged in ms later; foul-play
      # then takes ~3s to boot before it challenges, which is grace enough.
      # The log persists across restarts, so compare the COUNT to launches.
      eval "starts_${arm:-A}=\$((starts_${arm:-A} + 1))"
      eval "want=\$starts_${arm:-A}"
      i=0
      while [ "$(grep -c "Starting listening" "$W_LOG")" -lt "$want" ] \
            && [ "$i" -lt 60 ]; do
        sleep 0.25; i=$((i + 1))
      done
      sleep 0.5
    }
    LANE_DRY=0   # consecutive games this lane finished with no decision
    while g=$(next_game); do
      cd "$CB"   # each iteration starts from a known cwd (we cd to $FP below)
      DEC_BEFORE=$(grep -c '^INFO     Winner:' "$FP_LOG" 2>/dev/null)
      DEC_BEFORE=${DEC_BEFORE:-0}
      if [ -n "$AB_ENV" ]; then
        # pairs share a team; odd = arm A, even = arm B
        ridx=$(( (g + 1) / 2 ))
        if [ $((g % 2)) -eq 1 ]; then ARM="A"; else ARM="B"; fi
      else
        ridx=$g; ARM=""
      fi
      US="CBGen9${UTAG}L${lane}${ARM}"
      LANE_TEAM="$CB/showdown/bench/${NAME}_L${lane}${ARM}.team"
      CUR_LOG="$CB/showdown/bench/${NAME}_L${lane}${ARM}_ours.log"
      if [ "$N_TEAMS" -gt 0 ]; then
        # rotate the suite by rotation index (game, or pair under --ab) so
        # coverage stays balanced no matter which lane pulls the game
        idx=$(( (ridx - 1) % N_TEAMS + 1 ))
        OUR_TEAM=$(ls "$SUITE_DIR"/*.txt | sort | sed -n "${idx}p")
        if [ "$FP_N_TEAMS" -gt 0 ]; then
          # BOTH indices used to advance on ridx, so the (ours, theirs) pair
          # cycled with period lcm(N, FP_N) and only lcm-many DISTINCT
          # matchups were ever played — not N x FP_N. Whenever the two counts
          # share a factor the suite silently collapses: the 2026-08-03 webs
          # screen was 4 x 8, so each of our teams met just TWO of foul-play's
          # eight and its 320 games could not be compared with round 3, where
          # 15 x 8 is coprime and every pair occurred. Advance theirs only
          # after a full cycle of ours, which enumerates the cross product.
          pair=$(( (ridx - 1) % (N_TEAMS * FP_N_TEAMS) ))
          idx=$(( pair % N_TEAMS + 1 ))
          OUR_TEAM=$(ls "$SUITE_DIR"/*.txt | sort | sed -n "${idx}p")
          fidx=$(( pair / N_TEAMS % FP_N_TEAMS + 1 ))
          FP_SRC=$(ls "$FP_SUITE_DIR"/*.txt | sort | sed -n "${fidx}p")
          BASE="G${g}_$(basename "$OUR_TEAM" .txt)_vs_$(basename "$FP_SRC" .txt)"
          cp "$FP_SRC" "$FP/teams/teams/gen9/ou/suite/$BASE"
        else
          BASE="G${g}_$(basename "$OUR_TEAM" .txt)"
          cp "$OUR_TEAM" "$FP/teams/teams/gen9/ou/suite/$BASE"
        fi
        FP_TEAM="gen9/ou/suite/$BASE"
      else
        OUR_TEAM="$CB/showdown/teams/gen9ou_sample.txt"
        BASE="legacy_default"; FP_TEAM="gen9/ou/sample_legal"
      fi
      cp "$OUR_TEAM" "$LANE_TEAM"
      echo "=== lane $lane game $g/$TOTAL team: $BASE ($(date +%H:%M:%S)) ===" >> "$CUR_LOG"
      echo "=== lane $lane game $g/$TOTAL team: $BASE ($(date +%H:%M:%S)) ===" >> "$FP_LOG"
      eval "curpid=\$OURS_PID_${ARM:-A}"
      [ -z "$curpid" ] && start_worker "$ARM" "$@"
      if [ "$OPPONENT" = "spar" ]; then
        # spar bot reads a plaintext paste and connects to :8000 itself; stays
        # at $CB (no cd $FP). Mirror team by default, or the fp-suite src under
        # --fp-suite, so the matchup matches what foul-play would have played.
        # SPAR_NET picks the policy spar samples (default human); SPAR_TEMPERATURE
        # tilts it (lower = sharper/stronger/more predictable). fp net = a
        # stronger, still-predictable opponent when the human net is too weak to
        # give a full-strength CB any close games to swing.
        timeout "$PER_GAME_TIMEOUT" .venv/bin/python showdown/spar_bot.py \
            --username "$THEM" --mode challenge --user-to-challenge "$US" \
            --format gen9ou --team "${FP_SRC:-$OUR_TEAM}" --n-games 1 \
            --net "${SPAR_NET:-showdown/opp_policy_human_v1.pt}" \
            --temperature "${SPAR_TEMPERATURE:-1.0}" >> "$FP_LOG" 2>&1
      else
        cd "$FP"
        timeout "$PER_GAME_TIMEOUT" .venv/bin/python run.py \
            --websocket-uri ws://localhost:8000/showdown/websocket \
            --ps-username "$THEM" --bot-mode challenge_user \
            --user-to-challenge "$US" --pokemon-format gen9ou \
            --team-name "$FP_TEAM" --search-time-ms "${FP_SEARCH_MS:-300}" \
            --run-count 1 --log-level INFO >> "$FP_LOG" 2>&1
      fi
      if [ "$?" -eq 124 ]; then
        # hung game: the worker is stuck in a battle that will never finish
        # (max_concurrent_battles=1 would wedge every later game in the
        # lane) — restart it for a clean slate. Also self-heals a crashed
        # worker: foul-play's unanswered challenge times out and lands here.
        echo "=== lane $lane game $g TIMED OUT; restarting worker ===" >> "$CUR_LOG"
        eval "curpid=\$OURS_PID_${ARM:-A}"
        kill "$curpid" 2>/dev/null
        wait "$curpid" 2>/dev/null
        eval "OURS_PID_${ARM:-A}=''"
      fi
      # LANE QUARANTINE (2026-08-03). A lane whose opponent dies fails every
      # game in seconds while the dispenser keeps feeding it: bookopen lane 2
      # ate 231 of 400 games to produce 58 decisions. Stop pulling work once a
      # lane has gone this dry — the other lanes finish the series instead of
      # the queue draining into a black hole.
      DEC_AFTER=$(grep -c '^INFO     Winner:' "$FP_LOG" 2>/dev/null)
      DEC_AFTER=${DEC_AFTER:-0}
      if [ "${DEC_AFTER:-0}" -gt "${DEC_BEFORE:-0}" ]; then
        LANE_DRY=0
      else
        LANE_DRY=$((LANE_DRY + 1))
        if [ "$LANE_DRY" -ge "${CB_LANE_DRY_MAX:-8}" ]; then
          echo "=== lane $lane QUARANTINED after $LANE_DRY consecutive" \
               "no-decision games; not pulling more work ===" | tee -a "$CUR_LOG" >&2
          break
        fi
      fi
    done
    for p in "$OURS_PID_A" "$OURS_PID_B"; do
      if [ -n "$p" ]; then
        kill "$p" 2>/dev/null
        wait "$p" 2>/dev/null
      fi
    done
    rm -f "$CB/showdown/bench/${NAME}_L${lane}.team" \
          "$CB/showdown/bench/${NAME}_L${lane}A.team" \
          "$CB/showdown/bench/${NAME}_L${lane}B.team"
  ) &
  LANE_PIDS="$LANE_PIDS $!"
  lane=$((lane + 1))
done
# Wait on the LANES ONLY, never a bare `wait`. The zygote is also a background
# child of this script, and it never exits on its own — it waits for the
# shutdown message that `cleanup` sends AFTER this point. A bare `wait` is
# therefore a deadlock against our own zygote: every lane finishes, the series
# is complete, and the script hangs forever holding whatever inhibitor it was
# launched under (3 for 3 on 2026-07-27 — cbest dispensed and decided all 200
# games at 15:23 and the wrapper was still blocking suspend at 17:14).
for p in $LANE_PIDS; do
  wait "$p" 2>/dev/null
done
LANE_PIDS=""

W=$(tally CBGen9)
L=$(tally FPSpar1)
N=$((W + L))
DEALT=$(cat "$CB/showdown/bench/${NAME}"_L*_foulplay.log 2>/dev/null | grep -c "^=== lane ")
echo "=== '$NAME' complete at $(date +%H:%M:%S): ${W}W-${L}L of ${N} decided ==="
[ "$N" -gt 0 ] && echo "    $(echo "$W $N" | awk '{printf "%.1f%%", 100*$1/$2}')"
# DEALT vs DECIDED (2026-08-03). A lane whose opponent dies keeps taking games
# off the queue and failing them in seconds, so a half-dead run reports a
# plausible-looking "N decided" and reads as complete: the bookopen A/B dealt
# 400 and decided 221 because lane 2's foul-play crashed at battle start
# (231 dealt / 58 decided / 342 tracebacks on that lane alone). Say both
# numbers, and say loudly when they disagree.
if [ "${DEALT:-0}" -gt 0 ]; then
  echo "    dealt ${DEALT}, decided ${N}"
  LOST=$((DEALT - N))
  if [ "$LOST" -gt 0 ] && [ $((LOST * 10)) -gt "$DEALT" ]; then
    echo "    WARNING: ${LOST} of ${DEALT} dealt games produced NO decision (>10%)." >&2
    echo "             Check per-lane health before trusting this result:" >&2
    for lg in "$CB/showdown/bench/${NAME}"_L*_foulplay.log; do
      [ -f "$lg" ] || continue
      echo "             $(basename "$lg"): dealt=$(grep -c '^=== lane ' "$lg")" \
           "decided=$(grep -c '^INFO     Winner:' "$lg")" \
           "tracebacks=$(grep -c 'Traceback' "$lg")" >&2
    done
  fi
fi
if [ -n "$SPRT_P0" ]; then
  if [ -f "$QUEUE.verdict" ]; then
    echo "    SPRT: $(cat "$QUEUE.verdict") (final tally ${W}W-${L}L incl." \
         "games in flight at the verdict)"
  else
    echo "    SPRT: inconclusive at the $TOTAL-game cap"
  fi
fi
if [ -n "$AB_ENV" ]; then
  T=$("$CB/.venv/bin/python" "$CB/showdown/paired_tally.py" \
      "$CB/showdown/bench" "$NAME") || T="0 0 0 0 0 0 0 0"
  set -- $T
  nA=$1; nB=$2; PFX=$3; CPL=$4; wA=$5; lA=$6; wB=$7; lB=$8
  pct() { [ $(($1 + $2)) -gt 0 ] && echo "$1 $2" \
      | awk '{printf " (%.1f%%)", 100 * $1 / ($1 + $2)}'; }
  echo "    arm A baseline:      ${wA}W-${lA}L$(pct "$wA" "$lA")"
  echo "    arm B '$AB_ENV': ${wB}W-${lB}L$(pct "$wB" "$lB")"
  echo "    discordant pairs (prefix $PFX of $CPL complete): A won $nA, B won $nB"
  if [ -n "$AB_P0" ]; then
    if [ -f "$QUEUE.verdict" ]; then
      echo "    paired SPRT: $(cat "$QUEUE.verdict")"
    else
      echo "    paired SPRT: no detectable difference at the" \
           "$((TOTAL / 2))-pair cap"
    fi
  fi
fi
# queue/lock/verdict and the zygote are torn down by the EXIT trap (cleanup),
# which runs after this summary has read the verdict file.
