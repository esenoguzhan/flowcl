#!/usr/bin/env bash
# Piper, second table (new_grape), 24-27 Nov 2026: trimming the still stretches at home and shadow augmentation,
# ranked offline before Friday's arm session (plan of 24 Nov; first session's record
# docs/runs/2026-11-24_piper_new_grape_rollouts.md).
#
#   0. verify the four held-out caches; rebuild the shadow-map banks (sha256 logged); reference checks of the
#      existing all-90 daylight policies (seed 0 and 1).
#   A. on 80 demos (10 held out), seed 0: v0 plain, b0 trimmed at both ends, e0 end only, s0 start only, b3_mix
#      both ends + shadows (half real maps, half synthetic on the front; synthetic on the wrist). Each with the
#      offline, relight and start checks.
#   B. scripts/hw/piper_choose_recipe.py applies the fixed rule to block A and prints the recipe; it is trained on
#      all 90 demos, seeds 0 and 1, with the same checks. Skipped when nothing qualifies.
#   C. ablations on 80 demos: v3_mix (shadows, no trim), b1_synth (synthetic only), b3_front (front only),
#      s3_mix (start trim + shadows).
#
# 30k steps at 0.53 s a step on ThinkStation-2 (plus ~5 % for shadows): about 4.5 h per policy, 11 policies, about
# 50 h. Trainings are independent: one failing does not stop the next. A training is skipped when less than 25 GB
# is free. Stop the queue before a robot session (it holds the GPU).
#
# Launch:  tmux new-session -d -s piper_shadow "bash scripts/queue_piper_shadow.sh; exec bash"
# Status:  the last line of results/piper_queue/shadow/queue.log is "QUEUE DONE ok" or "QUEUE DONE FAILED: <steps>".
set -u
cd "$(dirname "$0")/.."
LOG=results/piper_queue/shadow
mkdir -p "$LOG"
ROOT=results/piper_shadow
MAPS=datasets/piper/shadow_maps/table2_train.npz
MIX=(--augment-shadow 0.5 --augment-shadow-maps "$MAPS" --augment-shadow-map-share 0.5)
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
checks() {  # name task run_dir
  local ckpt="$3/checkpoints/final.pt"
  run "offline_$1" scripts/hw/piper_offline_check.py --task "$2" --checkpoint "$ckpt"
  run "relight_$1" scripts/hw/piper_relight_check.py --checkpoint "$ckpt"
  run "start_$1" scripts/hw/piper_start_check.py --checkpoint "$ckpt"
}
policy() {  # name task n_demos seed [gate0 flags...]
  local name=$1 task=$2 n=$3 seed=$4; shift 4
  local free_gb
  free_gb=$(df -BG --output=avail . | tail -1 | tr -dc '0-9')
  if [ "$free_gb" -lt 25 ]; then
    echo "[queue] $(date -Is) skip train_$name: only ${free_gb} GB free" | tee -a "$LOG/queue.log"
    FAILED+=("train_$name")
    return 1
  fi
  run "train_$name" scripts/gate0.py --tasks "piper/$task" --embodiment agilex_piper_table2_336 \
       --policy flowpolicy_base_t1300 --n-demos "$n" --no-eval --train-steps 30000 --amp --seed "$seed" \
       --checkpoint-every 5000 --results-root "$ROOT/$name" --augment-daylight 0.3 "$@" \
    && checks "$name" "$task" "$ROOT/$name/single__piper__${task}__seed${seed}"
}

# 0. inputs
run verify_caches -c "from flowcl.data.piper_tasks import get_task, cache_path
from flowcl.data.piper_adapter import verify_cache
for t in ('new_grape_ho10_336', 'new_grape_trim2_ho10_336', 'new_grape_trimend_ho10_336', 'new_grape_trim_ho10_336'):
    verify_cache(cache_path(t), get_task(t)); print(t, 'ok')" || exit 1
run maps scripts/hw/piper_shadow_maps.py --task new_grape_ho10_336 || exit 1
grep -h "sha256" "$LOG/maps.log" | tee -a "$LOG/queue.log"
for s in 0 1; do
  ref=results/piper_daylight/single__piper__new_grape_336__seed$s
  run "relight_ref_s$s" scripts/hw/piper_relight_check.py --checkpoint "$ref/checkpoints/final.pt" --out "$LOG/ref_s${s}_relight_check.json"
  run "start_ref_s$s" scripts/hw/piper_start_check.py --checkpoint "$ref/checkpoints/final.pt" --out "$LOG/ref_s${s}_start_check.json"
done

# A. trims and shadows on 80 demos
policy v0 new_grape_ho10_336 80 0
policy b0 new_grape_trim2_ho10_336 80 0
policy e0 new_grape_trimend_ho10_336 80 0
policy s0 new_grape_trim_ho10_336 80 0
policy b3_mix new_grape_trim2_ho10_336 80 0 "${MIX[@]}"

# B. the chosen recipe on all 90 demos
run choose scripts/hw/piper_choose_recipe.py --out "$LOG/choice.json"
cat "$LOG/choose.log" | tee -a "$LOG/queue.log"
recipe=$(grep "^RECIPE " "$LOG/choose.log" | head -1)
if [ -n "$recipe" ]; then
  read -r _ name task flags <<< "$recipe"
  if ! [ -f "datasets/piper/cache/$task.h5" ]; then
    run "cache_$task" scripts/prepare_piper.py --build --task "$task"
  fi
  # shellcheck disable=SC2086  # the recipe's flags are words
  policy "${name}_s0" "$task" 90 0 $flags
  # shellcheck disable=SC2086
  policy "${name}_s1" "$task" 90 1 $flags
else
  echo "[queue] $(date -Is) block B skipped: no recipe qualified (see choose.log)" | tee -a "$LOG/queue.log"
fi

# C. ablations
policy v3_mix new_grape_ho10_336 80 0 "${MIX[@]}"
policy b1_synth new_grape_trim2_ho10_336 80 0 --augment-shadow 0.5
policy b3_front new_grape_trim2_ho10_336 80 0 "${MIX[@]}" --augment-shadow-cameras front
policy s3_mix new_grape_trim_ho10_336 80 0 "${MIX[@]}"

grep -h "\[start_check\]\|\[relight\] \(clean\|front\)" "$LOG"/start_*.log "$LOG"/relight_*.log 2>/dev/null | tee -a "$LOG/queue.log"
if [ "${#FAILED[@]}" -gt 0 ]; then
  echo "QUEUE DONE FAILED: ${FAILED[*]}" | tee -a "$LOG/queue.log"
else
  echo "QUEUE DONE ok" | tee -a "$LOG/queue.log"
fi
