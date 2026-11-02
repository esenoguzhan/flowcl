#!/usr/bin/env bash
# Piper: four policies on the milk session (task milk_plate, 40 demos, nothing held out), the
# front-camera mask crossed with daylight augmentation. Each is Gate 0's recipe (30k steps, batch
# 64, lr 1e-4, AMP, seed 0) with a checkpoint every 5k, then the offline open-loop check on
# final.pt (training episodes only). About 3 h on the 4090. Run it only when no other GPU job is
# running, from a clean commit (the runs record git_sha).
#
#   1 raw               results/single__piper__milk_plate_unmasked__seed0
#   2 masked            results/single__piper__milk_plate__seed0
#   3 raw + daylight    results/piper_daylight/single__piper__milk_plate_unmasked__seed0
#   4 masked + daylight results/piper_daylight/single__piper__milk_plate__seed0
#
# Needs both caches: uv run python scripts/prepare_piper.py --build --task milk_plate (and
# milk_plate_unmasked).
# Launch:  tmux new-session -d -s piper_milk "bash scripts/queue_piper_milk.sh; exec bash"
# Status:  the last line of results/piper_queue/milk/queue.log is "QUEUE DONE ok" or
#          "QUEUE DONE FAILED: <step>".
set -u
cd "$(dirname "$0")/.."
LOG=results/piper_queue/milk
mkdir -p "$LOG"
DAYLIGHT=0.3

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

# One policy and its offline check: NAME TASK EMBODIMENT RESULTS_ROOT [gate0 flags...]. The
# embodiment is the task's in flowcl/data/piper_tasks.py; the cache refuses any other.
policy() {
  local name=$1 task=$2 embodiment=$3 root=$4
  local where=()
  [ "$root" != results ] && where=(--results-root "$root")
  run "train_$name" scripts/gate0.py --tasks "piper/$task" --embodiment "$embodiment" \
    --n-demos 40 --no-eval --train-steps 30000 --amp --seed 0 --checkpoint-every 5000 \
    "${where[@]}" "${@:5}"
  run "check_$name" scripts/hw/piper_offline_check.py --task "$task" \
    --checkpoint "$root/single__piper__${task}__seed0/checkpoints/final.pt"
}

policy raw milk_plate_unmasked agilex_piper_unmasked results
policy masked milk_plate agilex_piper results
policy raw_daylight milk_plate_unmasked agilex_piper_unmasked results/piper_daylight --augment-daylight "$DAYLIGHT"
policy masked_daylight milk_plate agilex_piper results/piper_daylight --augment-daylight "$DAYLIGHT"
grep -h "training" "$LOG"/check_*.log | tee -a "$LOG/queue.log"
echo "QUEUE DONE ok" | tee -a "$LOG/queue.log"
