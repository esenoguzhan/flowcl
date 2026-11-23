#!/usr/bin/env bash
# Piper, the second table (~/piper_datasets/new_grape, 90 demos of 23 Nov 2026, front camera in a new pose):
# three policies at 336 px with daylight 0.3, every demo trained on (the user's choice: nothing held out, so the
# offline check measures fit only, and the three are compared on the arm):
#   new_grape_336 seed 0, new_grape_trim_336 seed 0 (the wait at home trimmed), new_grape_336 seed 1.
# Two caches (CPU, minutes each), then 30k steps each: 0.53 s a step on ThinkStation-2 (timed 23 Nov), about
# 4.4 h per policy, about 13.5 h in all. The trainings are independent: one failing (a GPU lockup, as the env-B
# seed-2 run's Xid 8 at 19:40 on 23 Nov) does not stop the next. gpu.csv logs the GPU every 60 s.
#
# Launch:  tmux new-session -d -s piper_new_grape "bash scripts/queue_piper_new_grape.sh; exec bash"
# Status:  the last line of results/piper_queue/new_grape/queue.log is "QUEUE DONE ok" or "QUEUE DONE FAILED: <steps>".
set -u
cd "$(dirname "$0")/.."
LOG=results/piper_queue/new_grape
mkdir -p "$LOG"
ROOT=results/piper_daylight
FAILED=()

nvidia-smi --query-gpu=timestamp,temperature.gpu,power.draw,clocks.sm,clocks.mem,utilization.gpu,memory.used,pstate,clocks_throttle_reasons.active \
  --format=csv -l 60 >> "$LOG/gpu.csv" 2>&1 &
GPU_LOG_PID=$!
trap 'kill "$GPU_LOG_PID" 2>/dev/null' EXIT

run() {
  local name=$1; shift
  echo "[queue] $(date -Is) start $name: $*" | tee -a "$LOG/queue.log"
  .venv/bin/python "$@" > "$LOG/$name.log" 2>&1
  local rc=$?  # captured before $(date) below can overwrite $?
  echo "[queue] $(date -Is) end $name rc=$rc" | tee -a "$LOG/queue.log"
  [ "$rc" -eq 0 ] || FAILED+=("$name")
  return "$rc"
}

policy() {  # name task seed
  local name=$1 task=$2 seed=$3
  run "train_$name" scripts/gate0.py --tasks "piper/$task" --embodiment agilex_piper_table2_336 \
       --policy flowpolicy_base_t1300 --n-demos 90 --no-eval --train-steps 30000 --amp --seed "$seed" \
       --checkpoint-every 5000 --results-root "$ROOT" --augment-daylight 0.3 \
    && run "check_$name" scripts/hw/piper_offline_check.py --task "$task" \
       --checkpoint "$ROOT/single__piper__${task}__seed${seed}/checkpoints/final.pt"
}

if run cache_plain scripts/prepare_piper.py --build --task new_grape_336 \
   && run cache_trim scripts/prepare_piper.py --build --task new_grape_trim_336; then
  policy plain_s0 new_grape_336 0
  policy trim_s0 new_grape_trim_336 0
  policy plain_s1 new_grape_336 1
fi
grep -h "training\|held_out" "$LOG"/check_*.log 2>/dev/null | tee -a "$LOG/queue.log"
if [ "${#FAILED[@]}" -gt 0 ]; then
  echo "QUEUE DONE FAILED: ${FAILED[*]}" | tee -a "$LOG/queue.log"
else
  echo "QUEUE DONE ok" | tee -a "$LOG/queue.log"
fi
