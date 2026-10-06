#!/usr/bin/env bash
# GPU queue for the Dobot X-Trainer try-out (branch dobot-hw), run in tmux overnight.
#
# 1. single-task red: the hardware proof policy, first so it is ready earliest;
# 2. seq_ft and adaptive GPM (gpm_ne90) on red -> green -> yellow (no rollouts: no simulator);
# 3. single-task green and yellow references.
#
# Pause during robot rollouts so GPU contention does not distort inference latency:
#   kill -STOP <pid of the running python>  ...  kill -CONT <pid>
set -u
cd "$(dirname "$0")/.."
LOG=results/dobot_queue
mkdir -p "$LOG"
COMMON=(--embodiment dobot_xtrainer --no-eval --amp)

run() {
  local name=$1; shift
  echo "[queue] $(date -Is) start $name: $*" | tee -a "$LOG/queue.log"
  .venv/bin/python "$@" > "$LOG/$name.log" 2>&1
  echo "[queue] $(date -Is) end $name rc=$?" | tee -a "$LOG/queue.log"
}

run single_red    scripts/gate0.py --tasks dobot/puzzle_red_retrieve --n-demos 110 "${COMMON[@]}"
run seq_ft        scripts/run_continual.py --curriculum dobot_puzzle --method seq_ft "${COMMON[@]}"
run gpm_ne90      scripts/run_continual.py --curriculum dobot_puzzle --method gpm_ne90 "${COMMON[@]}"
run single_green  scripts/gate0.py --tasks dobot/puzzle_green_retrieve --n-demos 114 "${COMMON[@]}"
run single_yellow scripts/gate0.py --tasks dobot/puzzle_yellow_retrieve --n-demos 132 "${COMMON[@]}"
echo "[queue] $(date -Is) all done" | tee -a "$LOG/queue.log"
