#!/usr/bin/env bash
# Piper: the two cheap levers against memorisation on the 30-demo sausage_pot_lid split (record
# docs/runs/2026-11-01_piper_f1_prep.md). About 1.5 h on the 4090. Run it only when no other GPU job
# is running, from a clean commit (the runs record git_sha).
#
#   A  results/piper_ckpt/  the reference recipe with a checkpoint every 5k steps (earlier stopping);
#                           its final.pt must equal the reference run's tensor for tensor.
#   B  results/piper_aug/   the same with image augmentation (shifts up to 8 px, colour jitter 0.2).
#
# Every checkpoint of both runs gets the offline open-loop check (held out v3 10-15, training 0-5,
# both hold-still baselines): <run>/offline_check_<checkpoint>.json, the held-out lines collected in
# results/piper_queue/levers/curve.txt.
#
# Launch:  tmux new-session -d -s piper "bash scripts/queue_piper_levers.sh; exec bash"
# Status:  the last line of results/piper_queue/levers/queue.log is "QUEUE DONE ok" or
#          "QUEUE DONE FAILED: <step>".
set -u
cd "$(dirname "$0")/.."
LOG=results/piper_queue/levers
mkdir -p "$LOG"
RUN_ID=single__piper__sausage_pot_lid__seed0
REFERENCE=results/$RUN_ID/checkpoints/final.pt
TRAIN=(scripts/gate0.py --tasks piper/sausage_pot_lid --embodiment agilex_piper --n-demos 30 --no-eval
       --train-steps 30000 --amp --seed 0 --checkpoint-every 5000)

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

check_all() {
  local label=$1 dir=$2
  for ckpt in "$dir"/checkpoints/step_*.pt "$dir"/checkpoints/final.pt; do
    local name
    name=$(basename "$ckpt" .pt)
    run "check_${label}_$name" scripts/hw/piper_offline_check.py --checkpoint "$ckpt" \
      --out "$dir/offline_check_$name.json"
    grep "held_out" "$LOG/check_${label}_$name.log" | sed "s/^/$label $name: /" >> "$LOG/curve.txt"
  done
}

run train_ckpt "${TRAIN[@]}" --results-root results/piper_ckpt
run identity -c "
import sys, torch
a = torch.load('$REFERENCE', weights_only=False)['state_dict']
b = torch.load('results/piper_ckpt/$RUN_ID/checkpoints/final.pt', weights_only=False)['state_dict']
diff = [k for k in a if k not in b or not torch.equal(a[k], b[k])]
print(('IDENTICAL' if not diff else 'DIFFERS') + f': {len(diff)} of {len(a)} tensors differ from the reference final.pt')
"
cat "$LOG/identity.log" | tee -a "$LOG/queue.log"
check_all ckpt "results/piper_ckpt/$RUN_ID"

run train_aug "${TRAIN[@]}" --augment-shift-px 8 --augment-color 0.2 --results-root results/piper_aug
check_all aug "results/piper_aug/$RUN_ID"
echo "QUEUE DONE ok" | tee -a "$LOG/queue.log"
