#!/usr/bin/env bash
# Piper F1, first single-task policy (thesis plan Workstream F): train on the 30 training demos of
# sausage_pot_lid (no rollouts: no simulator), then the offline open-loop check on the 6 held-out
# and 6 training episodes. About 45 min on the 4090. Run it only when no other GPU job is running,
# from a clean commit (the run records git_sha).
#
# Needs the cache: uv run python scripts/prepare_piper.py --build
# Launch:  tmux new-session -d -s piper "bash scripts/queue_piper_single.sh; exec bash"
# Status:  the last line of results/piper_queue/queue.log is "QUEUE DONE ok" or "QUEUE DONE FAILED: <step>".
set -u
cd "$(dirname "$0")/.."
LOG=results/piper_queue
mkdir -p "$LOG"
RUN=results/single__piper__sausage_pot_lid__seed0

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

run single_sausage scripts/gate0.py --tasks piper/sausage_pot_lid --embodiment agilex_piper \
  --n-demos 30 --no-eval --train-steps 30000 --amp --seed 0
run offline_check scripts/hw/piper_offline_check.py --checkpoint "$RUN/checkpoints/final.pt"
echo "QUEUE DONE ok" | tee -a "$LOG/queue.log"
