#!/usr/bin/env bash
# Piper: the new grape demos, alone and with the old ones, each without and then with daylight
# 0.3, at 336 and then 448 px: eight policies. pick_grape_v2 gives 50 episodes, 5 of them held out
# from every zone of the map (38, 41, 24, 27, 31), so 45 train alone and 85 with pick_grape's 40.
# Gate 0's recipe (30k steps, lr 1e-4, AMP, seed 0) with a checkpoint every 5k, then the offline
# check on final.pt, now with real held-out demos. Batch 64 does not fit at 448 px, so there it is
# 32 with 2 batches per step: the same effective batch, about 1.9x the time. 336 px runs first, so
# the queue can be stopped there. Run it only when no other GPU job is running, from a clean commit.
#
#   results/single__piper__<task>__seed0                 no augmentation
#   results/piper_daylight/single__piper__<task>__seed0  daylight 0.3
#
# Needs the caches: uv run python scripts/prepare_piper.py --build --task <task>
# Launch:  tmux new-session -d -s piper_v2 "bash scripts/queue_piper_grape_v2.sh; exec bash"
# Status:  the last line of results/piper_queue/grape_v2/queue.log is "QUEUE DONE ok" or
#          "QUEUE DONE FAILED: <steps>". A failed policy does not stop the ones after it.
set -u
cd "$(dirname "$0")/.."
LOG=results/piper_queue/grape_v2
mkdir -p "$LOG"
DAYLIGHT=0.3
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

# One policy and its offline check: NAME TASK EMBODIMENT POLICY N_DEMOS RESULTS_ROOT [gate0 flags...].
policy() {
  local name=$1 task=$2 embodiment=$3 config=$4 demos=$5 root=$6
  local where=()
  [ "$root" != results ] && where=(--results-root "$root")
  run "train_$name" scripts/gate0.py --tasks "piper/$task" --embodiment "$embodiment" --policy "$config" \
    --n-demos "$demos" --no-eval --train-steps 30000 --amp --seed 0 --checkpoint-every 5000 \
    "${where[@]}" "${@:7}" || return 0
  run "check_$name" scripts/hw/piper_offline_check.py --task "$task" \
    --checkpoint "$root/single__piper__${task}__seed0/checkpoints/final.pt"
}

# Both trainings of one set of demos at one size, without and then with augmentation.
pair() {
  local name=$1 size=$2 config=$3 demos=$4; shift 4
  policy "${name}${size}" "${name}_crop${size}" "agilex_piper_crop${size}" "$config" "$demos" results "$@"
  policy "${name}${size}_daylight" "${name}_crop${size}" "agilex_piper_crop${size}" "$config" "$demos" \
    results/piper_daylight --augment-daylight "$DAYLIGHT" "$@"
}

pair grape_v2 336 flowpolicy_base_t1300 45
pair grape_all 336 flowpolicy_base_t1300 85
pair grape_v2 448 flowpolicy_base_t2200 45 --batch-size 32 --accumulation-steps 2
pair grape_all 448 flowpolicy_base_t2200 85 --batch-size 32 --accumulation-steps 2
grep -h "held_out\|training" "$LOG"/check_*.log | tee -a "$LOG/queue.log"
if [ "${#FAILED[@]}" -gt 0 ]; then
  echo "QUEUE DONE FAILED: ${FAILED[*]}" | tee -a "$LOG/queue.log"
else
  echo "QUEUE DONE ok" | tee -a "$LOG/queue.log"
fi
