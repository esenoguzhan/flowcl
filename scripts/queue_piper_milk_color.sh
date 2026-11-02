#!/usr/bin/env bash
# Piper: two more policies on the milk session (task milk_plate, 40 demos, nothing held out), the
# companions of scripts/queue_piper_milk.sh's daylight pair with the colour jitter alone
# (brightness, contrast, saturation; no colour temperature, no illumination ramp, no shifts) at
# the strength daylight uses inside, so the two pairs differ only in what daylight adds. Each is
# Gate 0's recipe (30k steps, batch 64, lr 1e-4, AMP, seed 0) with a checkpoint every 5k, then the
# offline open-loop check on final.pt (training episodes only). About 1.5 h on the 4090. Run it only
# when no other GPU job is running, from a clean commit (the runs record git_sha).
#
#   5 raw + colour      results/piper_color/single__piper__milk_plate_unmasked__seed0
#   6 masked + colour   results/piper_color/single__piper__milk_plate__seed0
#
# Launch:  tmux new-session -d -s piper_milk_color "bash scripts/queue_piper_milk_color.sh; exec bash"
# Status:  the last line of results/piper_queue/milk_color/queue.log is "QUEUE DONE ok" or
#          "QUEUE DONE FAILED: <step>".
set -u
cd "$(dirname "$0")/.."
LOG=results/piper_queue/milk_color
mkdir -p "$LOG"
COLOR=0.3
ROOT=results/piper_color

run() {
  local name=$1; shift
  echo "[queue] $(date -Is) start $name: $*" | tee -a "$LOG/queue.log"
  .venv/bin/python "$@" > "$LOG/$name.log" 2>&1
  local rc=$?  # captured before $(date) below can overwrite $?
  echo "[queue] $(date -Is) end $name rc=$rc" | tee -a "$LOG/queue.log"
  if [ "$rc" -ne 0 ]; then
    echo "QUEUE DONE FAILED: $name" | tee -a "$LOG/queue.log"
    exit 1
  fi
}

# One policy and its offline check: NAME TASK EMBODIMENT. The embodiment is the task's in
# flowcl/data/piper_tasks.py; the cache refuses any other.
policy() {
  local name=$1 task=$2 embodiment=$3
  run "train_$name" scripts/gate0.py --tasks "piper/$task" --embodiment "$embodiment" \
    --n-demos 40 --no-eval --train-steps 30000 --amp --seed 0 --checkpoint-every 5000 \
    --results-root "$ROOT" --augment-color "$COLOR"
  run "check_$name" scripts/hw/piper_offline_check.py --task "$task" \
    --checkpoint "$ROOT/single__piper__${task}__seed0/checkpoints/final.pt"
}

policy raw_color milk_plate_unmasked agilex_piper_unmasked
policy masked_color milk_plate agilex_piper
grep -h "training" "$LOG"/check_*.log | tee -a "$LOG/queue.log"
echo "QUEUE DONE ok" | tee -a "$LOG/queue.log"
