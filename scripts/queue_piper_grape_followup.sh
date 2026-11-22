#!/usr/bin/env bash
# Piper grape follow-up: three variants of the best policy of the grape v2 queue, grape_all at 336 px
# with daylight 0.3 (held-out joint MAE 0.0250 rad; repeating the last command 0.0267). The
# checkpoint sweep of 8 Nov found held-out error still falling at 30k steps in all eight policies,
# with the cosine schedule's lr near 0 by then. About 11h20 in all:
#
#   1. 60k steps, the schedule stretched to match (about 5h40)  results/piper_daylight_60k
#   2. daylight 0.5 instead of 0.3 (about 2h50)                   results/piper_daylight05
#   3. seed 1, otherwise the 30k recipe (about 2h50)              results/piper_daylight (…__seed1)
#
# Each then gets the offline check on final.pt (the same 5 held-out demos from pick_grape_v2).
# Run it only when no other GPU job is running, from a clean commit.
# Launch:  tmux new-session -d -s piper_next "bash scripts/queue_piper_grape_followup.sh; exec bash"
# Status:  the last line of results/piper_queue/grape_followup/queue.log is "QUEUE DONE ok" or
#          "QUEUE DONE FAILED: <steps>". A failed policy does not stop the ones after it.
set -u
cd "$(dirname "$0")/.."
LOG=results/piper_queue/grape_followup
mkdir -p "$LOG"
TASK=grape_all_crop336
FAILED=()

run() {
  local name=$1; shift
  echo "[queue] $(date -Is) start $name: $*" | tee -a "$LOG/queue.log"
  .venv/bin/python "$@" > "$LOG/$name.log" 2>&1
  local rc=$?  # captured before $(date) below can overwrite $?
  echo "[queue] $(date -Is) end $name rc=$rc" | tee -a "$LOG/queue.log"
  [ "$rc" -eq 0 ] || FAILED+=("$name")
  return "$rc"
}

# One policy and its offline check: NAME RESULTS_ROOT SEED STEPS [gate0 flags...].
policy() {
  local name=$1 root=$2 seed=$3 steps=$4
  run "train_$name" scripts/gate0.py --tasks "piper/$TASK" --embodiment agilex_piper_crop336 \
    --policy flowpolicy_base_t1300 --n-demos 85 --no-eval --train-steps "$steps" --amp --seed "$seed" \
    --checkpoint-every 5000 --results-root "$root" "${@:5}" || return 0
  run "check_$name" scripts/hw/piper_offline_check.py --task "$TASK" \
    --checkpoint "$root/single__piper__${TASK}__seed${seed}/checkpoints/final.pt"
}

policy all336_daylight_60k results/piper_daylight_60k 0 60000 --augment-daylight 0.3
policy all336_daylight05   results/piper_daylight05   0 30000 --augment-daylight 0.5
policy all336_daylight_s1  results/piper_daylight     1 30000 --augment-daylight 0.3
grep -h "held_out\|training" "$LOG"/check_*.log | tee -a "$LOG/queue.log"
if [ "${#FAILED[@]}" -gt 0 ]; then
  echo "QUEUE DONE FAILED: ${FAILED[*]}" | tee -a "$LOG/queue.log"
else
  echo "QUEUE DONE ok" | tee -a "$LOG/queue.log"
fi
