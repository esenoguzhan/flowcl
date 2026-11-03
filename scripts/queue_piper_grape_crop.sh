#!/usr/bin/env bash
# Piper: the grape policy with the front camera cut to the table (task grape_plate_crop, 40 demos,
# nothing held out; embodiment agilex_piper_crop, the operator mask plus the working area the
# user drew), to compare with grape_plate on the robot the same day. Gate 0's recipe (30k
# steps, batch 64, lr 1e-4, AMP, seed 0) with a checkpoint every 5k, so an early checkpoint can
# go to the robot while training continues; then the offline open-loop check on final.pt
# (training episodes only). About 40 min on the 4090.
# Run it from a clean commit (the run records git_sha).
#
#   cropped  results/single__piper__grape_plate_crop__seed0
#
# Needs the cache: uv run python scripts/prepare_piper.py --build --task grape_plate_crop
# Launch:  tmux new-session -d -s piper_grape_crop "bash scripts/queue_piper_grape_crop.sh; exec bash"
# Status:  the last line of results/piper_queue/grape_crop/queue.log is "QUEUE DONE ok" or
#          "QUEUE DONE FAILED: <step>".
set -u
cd "$(dirname "$0")/.."
LOG=results/piper_queue/grape_crop
mkdir -p "$LOG"

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

run train_cropped scripts/gate0.py --tasks piper/grape_plate_crop --embodiment agilex_piper_crop \
  --n-demos 40 --no-eval --train-steps 30000 --amp --seed 0 --checkpoint-every 5000
run check_cropped scripts/hw/piper_offline_check.py --task grape_plate_crop \
  --checkpoint results/single__piper__grape_plate_crop__seed0/checkpoints/final.pt
grep -h "training" "$LOG"/check_*.log | tee -a "$LOG/queue.log"
echo "QUEUE DONE ok" | tee -a "$LOG/queue.log"
