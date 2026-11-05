#!/usr/bin/env bash
# Piper: does a finer input fix the aim? The table crop at 224 and 336 px on the grape and milk
# sessions (40 demos each), each without and then with daylight augmentation. At 128 px a patch
# of the frozen encoder covers about 100 x 70 mm of table, more than the grape. Gate 0's recipe (30k steps,
# batch 64, lr 1e-4, AMP, seed 0) with a checkpoint every 5k, then the offline open-loop check on
# final.pt (training episodes only). No random shifts: they would move the image by about the
# precision this queue is after. Run it only when no other GPU job is running, from a clean
# commit (the runs record git_sha).
#
#   results/single__piper__<task>__seed0                 no augmentation
#   results/piper_daylight/single__piper__<task>__seed0  daylight 0.3
#
# Needs the caches: uv run python scripts/prepare_piper.py --build --task <task>
# Launch:  tmux new-session -d -s piper_res "bash scripts/queue_piper_resolution.sh; exec bash"
# Status:  the last line of results/piper_queue/resolution/queue.log is "QUEUE DONE ok" or
#          "QUEUE DONE FAILED: <steps>". A failed policy does not stop the ones after it.
set -u
cd "$(dirname "$0")/.."
LOG=results/piper_queue/resolution
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

# Both trainings of one task at one size, without and then with augmentation.
pair() {
  local name=$1 task=$2 size=$3 config=$4 demos=$5
  policy "${name}${size}" "${task}_crop${size}" "agilex_piper_crop${size}" "$config" "$demos" results
  policy "${name}${size}_daylight" "${task}_crop${size}" "agilex_piper_crop${size}" "$config" "$demos" \
    results/piper_daylight --augment-daylight "$DAYLIGHT"
}

pair grape grape_plate 224 flowpolicy_base 40
pair grape grape_plate 336 flowpolicy_base_t1300 40
pair milk milk_plate 224 flowpolicy_base 40
pair milk milk_plate 336 flowpolicy_base_t1300 40
grep -h "training" "$LOG"/check_*.log | tee -a "$LOG/queue.log"
if [ "${#FAILED[@]}" -gt 0 ]; then
  echo "QUEUE DONE FAILED: ${FAILED[*]}" | tee -a "$LOG/queue.log"
else
  echo "QUEUE DONE ok" | tee -a "$LOG/queue.log"
fi
