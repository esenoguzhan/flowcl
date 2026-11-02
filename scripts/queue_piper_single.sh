#!/usr/bin/env bash
# Piper F1, one single-task policy (thesis plan Workstream F): train on a registry task's first
# N_DEMOS demos (no rollouts: no simulator), then the offline open-loop check on its held-out and
# training episodes. About 40 min on the 4090. Run it only when no other GPU job is running, from
# a clean commit (the run records git_sha).
#
# Usage:   bash scripts/queue_piper_single.sh [TASK] [N_DEMOS] [gate0 flags...]   (default: sausage_pot_lid 30)
#          e.g. ... sausage_pot_lid_all 53 --augment-shift-px 8 --augment-color 0.2 --checkpoint-every 5000
# Needs the cache: uv run python scripts/prepare_piper.py --build --task TASK
# Launch:  tmux new-session -d -s piper "bash scripts/queue_piper_single.sh sausage_pot_lid_v3 12; exec bash"
# Status:  the last line of results/piper_queue/<TASK>/queue.log is "QUEUE DONE ok" or
#          "QUEUE DONE FAILED: <step>".
set -u
cd "$(dirname "$0")/.."
TASK=${1:-sausage_pot_lid}
N_DEMOS=${2:-30}
EXTRA=("${@:3}")
LOG=results/piper_queue/$TASK
mkdir -p "$LOG"
RUN=results/single__piper__${TASK}__seed0

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

run single scripts/gate0.py --tasks "piper/$TASK" --embodiment agilex_piper \
  --n-demos "$N_DEMOS" --no-eval --train-steps 30000 --amp --seed 0 "${EXTRA[@]}"
run offline_check scripts/hw/piper_offline_check.py --task "$TASK" --checkpoint "$RUN/checkpoints/final.pt"
echo "QUEUE DONE ok" | tee -a "$LOG/queue.log"
