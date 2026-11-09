#!/usr/bin/env bash
# Piper: grape_all at 336 px with daylight 0.3, as results/piper_daylight/single__piper__grape_all_crop336__seed0,
# but with the wait at home before the reach trimmed from every demo (task grape_all_trim_crop336). On 9 Nov 14 of
# 26 rollouts never left home: at home the policy kept predicting the reach 10-14 ticks ahead, past the 8 executed.
# Builds the cache (about 10 min), trains 30k steps (about 2h50), then the offline check on the 5 held-out demos,
# which are trimmed too, so their numbers are not comparable to the untrimmed policies'.
#
# Launch:  tmux new-session -d -s piper_trim "bash scripts/queue_piper_grape_trim.sh; exec bash"
# Status:  the last line of results/piper_queue/grape_trim/queue.log is "QUEUE DONE ok" or "QUEUE DONE FAILED: <steps>".
set -u
cd "$(dirname "$0")/.."
LOG=results/piper_queue/grape_trim
mkdir -p "$LOG"
TASK=grape_all_trim_crop336
ROOT=results/piper_daylight
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

run cache scripts/prepare_piper.py --build --task "$TASK" \
  && run train scripts/gate0.py --tasks "piper/$TASK" --embodiment agilex_piper_crop336 \
       --policy flowpolicy_base_t1300 --n-demos 85 --no-eval --train-steps 30000 --amp --seed 0 \
       --checkpoint-every 5000 --results-root "$ROOT" --augment-daylight 0.3 \
  && run check scripts/hw/piper_offline_check.py --task "$TASK" \
       --checkpoint "$ROOT/single__piper__${TASK}__seed0/checkpoints/final.pt"
grep -h "held_out\|training" "$LOG"/check.log 2>/dev/null | tee -a "$LOG/queue.log"
if [ "${#FAILED[@]}" -gt 0 ]; then
  echo "QUEUE DONE FAILED: ${FAILED[*]}" | tee -a "$LOG/queue.log"
else
  echo "QUEUE DONE ok" | tee -a "$LOG/queue.log"
fi
