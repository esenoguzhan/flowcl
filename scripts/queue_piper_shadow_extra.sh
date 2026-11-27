#!/usr/bin/env bash
# Piper new_grape, night of Thu 26 to Fri 27 Nov 2026, before the arm session at about 12:00. Follows
# scripts/queue_piper_shadow.sh (done 26 Nov 15:53, "QUEUE DONE FAILED: train_v3_mix", the GPU stall, Xid 8).
#
# Held out, the recipe the rule chose (s3_mix: start trim + shadows) scored below b3_mix (both trims + shadows):
# relit approach MAE 0.032 against 0.027, premature opening 0.12-0.20 against 0.00, plausible reach 0.78-0.80
# against 0.87-0.90, from one seed on 10 episodes. The rule judged the shadows on b3_mix against b0 and added them
# to s0's trim. Tonight, in order of use for Friday:
#   1. b3_mix_all90_s0: both trims + shadows on all 90 demos, seed 0. A second deployable policy next to the
#      rule's s3_mix_all90_s0/_s1; exploratory, the rule did not pick it. Its cache is built first.
#   2. b3_mix_s1 and 3. s3_mix_s1: the two held-out recipes again with seed 1 (80 demos), to see whether
#      b3_mix's lead holds on a second seed.
# Each with the offline, relight and start checks, as in the first queue. v3_mix is not rerun (after Friday).
#
# About 4 h 40 min per policy with its checks on ThinkStation-2 (0.55 s a step with shadows). A deadline keeps the
# GPU free for the arm: a training is not started unless it can end, checks included, by DEADLINE
# (default Fri 27 Nov 11:30), and every step is killed at DEADLINE. Trainings are independent.
#
# Launch:  tmux new-session -d -s piper_extra "bash scripts/queue_piper_shadow_extra.sh; exec bash"
# Dry run: DRY=1 bash scripts/queue_piper_shadow_extra.sh   (prints the steps, runs nothing)
# Status:  the last line of results/piper_queue/shadow_extra/queue.log is "QUEUE DONE ok" or
#          "QUEUE DONE FAILED: <steps>".
set -u
cd "$(dirname "$0")/.."
DRY=${DRY:-0}
LOG=results/piper_queue/shadow_extra
[ "$DRY" = 1 ] && LOG=$(mktemp -d)
mkdir -p "$LOG"
ROOT=results/piper_shadow
MAPS=datasets/piper/shadow_maps/table2_train.npz
MIX=(--augment-shadow 0.5 --augment-shadow-maps "$MAPS" --augment-shadow-map-share 0.5)
DEADLINE=${DEADLINE:-2026-11-27T11:30:00+01:00}
DEADLINE_S=$(date -d "$DEADLINE" +%s) || exit 1
POLICY_S=$((4 * 3600 + 45 * 60))  # one training + its checks, with margin
FAILED=()
if [ "$DRY" != 1 ]; then
  nvidia-smi --query-gpu=timestamp,temperature.gpu,power.draw,clocks.sm,clocks.mem,utilization.gpu,memory.used,pstate,clocks_throttle_reasons.active \
    --format=csv -l 60 >> "$LOG/gpu.csv" 2>&1 &
  GPU_LOG_PID=$!
  trap 'kill "$GPU_LOG_PID" 2>/dev/null' EXIT
fi
say() { echo "[queue] $(date -Is) $*" | tee -a "$LOG/queue.log"; }
run() {
  local name=$1; shift
  local left=$((DEADLINE_S - $(date +%s)))
  if [ "$left" -le 0 ]; then
    say "skip $name: past the deadline $DEADLINE"
    FAILED+=("$name"); return 1
  fi
  say "start $name: $*"
  if [ "$DRY" = 1 ]; then return 0; fi
  timeout --signal=INT --kill-after=120 "$left" .venv/bin/python "$@" > "$LOG/$name.log" 2>&1
  local rc=$?  # captured before $(date) below can overwrite $?
  [ "$rc" -eq 124 ] && say "$name stopped at the deadline $DEADLINE"
  say "end $name rc=$rc"
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
  local free_gb now
  now=$(date +%s)
  if [ $((now + POLICY_S)) -gt "$DEADLINE_S" ]; then
    say "skip train_$name: would end after the deadline $DEADLINE"
    FAILED+=("train_$name"); return 1
  fi
  free_gb=$(df -BG --output=avail . | tail -1 | tr -dc '0-9')
  if [ "$free_gb" -lt 25 ]; then
    say "skip train_$name: only ${free_gb} GB free"
    FAILED+=("train_$name"); return 1
  fi
  run "train_$name" scripts/gate0.py --tasks "piper/$task" --embodiment agilex_piper_table2_336 \
       --policy flowpolicy_base_t1300 --n-demos "$n" --no-eval --train-steps 30000 --amp --seed "$seed" \
       --checkpoint-every 5000 --results-root "$ROOT/$name" --augment-daylight 0.3 "$@" \
    && checks "$name" "$task" "$ROOT/$name/single__piper__${task}__seed${seed}"
}

# 0. inputs: the all-90 both-trims cache (new), the caches of the first queue, the shadow-map bank it built
say "git $(git rev-parse --short HEAD)$( [ -n "$(git status --porcelain)" ] && echo ' DIRTY')"
[ -s "$MAPS" ] || { say "missing $MAPS"; exit 1; }
say "maps file sha256 $(sha256sum "$MAPS" | cut -c1-16)"
if ! [ -f datasets/piper/cache/new_grape_trim2_336.h5 ]; then
  run cache_new_grape_trim2_336 scripts/prepare_piper.py --build --task new_grape_trim2_336 || exit 1
fi
run verify_caches -c "from flowcl.data.piper_tasks import get_task, cache_path
from flowcl.data.piper_adapter import verify_cache
for t in ('new_grape_trim2_336', 'new_grape_trim2_ho10_336', 'new_grape_trim_ho10_336'):
    verify_cache(cache_path(t), get_task(t)); print(t, 'ok')" || exit 1

# 1. a second deployable policy; 2-3. seed 1 of the two held-out recipes
policy b3_mix_all90_s0 new_grape_trim2_336 90 0 "${MIX[@]}"
policy b3_mix_s1 new_grape_trim2_ho10_336 80 1 "${MIX[@]}"
policy s3_mix_s1 new_grape_trim_ho10_336 80 1 "${MIX[@]}"

for n in b3_mix_all90_s0 b3_mix_s1 s3_mix_s1; do
  grep -h "\[offline_check\] held_out\|\[relight\] \(clean\|front\)\|\[start_check\] \(sun\|transition\|demo_light\)" \
    "$LOG/offline_$n.log" "$LOG/relight_$n.log" "$LOG/start_$n.log" 2>/dev/null | sed "s/^/$n /" | tee -a "$LOG/queue.log"
done
if [ "${#FAILED[@]}" -gt 0 ]; then
  echo "QUEUE DONE FAILED: ${FAILED[*]}" | tee -a "$LOG/queue.log"
else
  echo "QUEUE DONE ok" | tee -a "$LOG/queue.log"
fi
