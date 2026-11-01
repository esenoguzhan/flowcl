#!/usr/bin/env bash
# Calibrate the offline open-loop check on LIBERO (record
# docs/runs/2026-11-02_libero_offline_calibration.md). About 2.5 h on the 4090. Run it only when no
# other GPU job is running, from a clean commit (the runs record git_sha), from the main tree (the
# LIBERO config is global).
#
# The moka-pot task (libero_10, two stages, 100 % in Gate 0 with 50 demos) with 20, 30 and 40 demos
# (demos 0..n-1), each with Gate 0's recipe and its 50 closed-loop rollouts, then the offline check
# on demos 40-49 (unseen by every run) and 0-9 (trained on by every run):
#
#   results/libero_calib/n<N>/<run>/            the run, its eval.json and offline_check.json
#   results/libero_calib/n<N>/gate0/gate0.json  the Gate 0 verdict for the run
#   results/libero_calib/curve.txt              success and the offline check, one line per run
#
# Launch:  tmux new-session -d -s libero_calib "bash scripts/queue_libero_calibration.sh; exec bash"
# Status:  the last line of results/libero_calib/queue/queue.log is "QUEUE DONE ok" or
#          "QUEUE DONE FAILED: <step>".
set -u
cd "$(dirname "$0")/.."
ROOT=results/libero_calib
LOG=$ROOT/queue
mkdir -p "$LOG"
TASK=libero_10/KITCHEN_SCENE3_turn_on_the_stove_and_put_the_moka_pot_on_it
RUN_ID=single__libero_10__KITCHEN_SCENE3_turn_on_the_stove_and_put_the_moka_pot_on_it__seed0

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

for N in 20 30 40; do
  DIR=$ROOT/n$N
  run "train_n$N" scripts/gate0.py --tasks "$TASK" --n-demos "$N" --train-steps 30000 --amp --seed 0 \
    --results-root "$DIR" --out-dir "$DIR/gate0"
  run "check_n$N" scripts/libero_offline_check.py --checkpoint "$DIR/$RUN_ID/checkpoints/final.pt"
  .venv/bin/python - "$N" "$DIR/$RUN_ID" >> "$ROOT/curve.txt" <<'EOF'
import json, sys
n, run = sys.argv[1], sys.argv[2]
task = json.load(open(f"{run}/eval.json"))["tasks"][0]
check = json.load(open(f"{run}/offline_check.json"))
parts = [f"n={n}: success {task['success_rate']:.2f} [{task['ci_low']:.2f}, {task['ci_high']:.2f}]"]
for group in ("held_out", "training"):
    s = check[group]["summary"]
    parts.append(f"{group} integrated {s['integrated_policy_mae_executed']:.4f} vs standing still "
                 f"{s['integrated_standing_still_mae_executed']:.4f} "
                 f"(ratio {s['integrated_ratio_to_standing_still']:.2f})")
print("; ".join(parts))
EOF
  tail -1 "$ROOT/curve.txt" | tee -a "$LOG/queue.log"
done
echo "QUEUE DONE ok" | tee -a "$LOG/queue.log"
