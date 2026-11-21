#!/usr/bin/env bash
# The 4-task reference runs retrained on the new machine (ThinkStation-2, RTX PRO 4000 Blackwell,
# torch 2.9.1+cu128): seq_ft, plain GPM and the flat-0.99 GPM (gpm_e99) on seq_hetero for one seed,
# into results/envB/. The archived runs in results/ were made on gamma (RTX 4090, torch 2.4.1+cu121)
# and cannot be reproduced or identity-checked bit for bit here, so every reference below is a run
# of this queue, never an archived one. (run_continual's default T1 reference is the archived
# results/<namespace> run, so it is always passed explicitly.) Same commands as the archived runs:
# seq_ft plain; GPM with the T1 pairing check against seq_ft; e99 with T1 pairing against seq_ft
# and identity against plain GPM at stages 0-1 (as scripts/queue_high_protection.sh). Same seed
# namespace (seq_hetero__seq_ft__seed<S>) as the archived runs, so data order, noise and rollout
# seeds are the same streams; only the machine differs. Orchestration only.
# Run it only when no other GPU job is running.
#
# Usage:    bash scripts/queue_envb_refs.sh [--seed S] [--from-step K]
#   --from-step K   resume: steps < K are not rerun (status from their outputs on disk; a missing
#                   output blocks its dependents). A run directory left by an interrupted training
#                   step must be moved aside first (to results/interrupted/).
#
#   0  preflight       (every launch: clean tree, idle GPU, CUDA kernels run, the run directories
#                       of the steps to run absent, disk)
#   1  seq_ft run      (~5 h on gamma)
#   2  GPM run         (~5 h; T1 pairing against step 1)
#   3  e99 run         (~5 h; T1 pairing against step 1, identity against step 2 at stages 0-1)
#   4  sequence report, GPM vs seq_ft  -> results/envB/gpm_seq_seed<S>/report.json
#   5  sequence report, e99 vs seq_ft  -> results/envB/gpm_seq_e99_seed<S>/report.json
#   6  stage-0 identity of GPM against seq_ft (bitwise on gamma; descriptive here)
#                      -> results/envB/identity_gpm_vs_seqft_seed<S>_stage0.json
#
# Status: every non-zero rc and missing prior output is collected; the last line is
# "QUEUE DONE ok" (exit 0) or "QUEUE DONE FAILED: <steps>" (exit 1).
#
# Launch:   tmux new-session -d -s envb_refs "bash scripts/queue_envb_refs.sh; exec bash"
# Testing:  QUEUE_DRY_RUN=1, QUEUE_FAIL_STEP=<step name>, QUEUE_LOG_ROOT.
set -u
USAGE="usage: queue_envb_refs.sh [--seed S] [--from-step K]"
SEED=0
FROM=0
while [ $# -gt 0 ]; do
    case "$1" in
        --seed) SEED="${2:-}"; shift 2 || { echo "$USAGE" >&2; exit 2; } ;;
        --from-step) FROM="${2:-}"; shift 2 || { echo "$USAGE" >&2; exit 2; } ;;
        *) echo "unknown argument '$1'; $USAGE" >&2; exit 2 ;;
    esac
done
case "$SEED" in 0|1|2) ;; *) echo "--seed must be 0, 1 or 2, got '$SEED'" >&2; exit 2 ;; esac
case "$FROM" in ''|*[!0-9]*) echo "--from-step must be an integer 0..6, got '$FROM'" >&2; exit 2 ;; esac
if [ "$FROM" -gt 6 ]; then echo "--from-step must be an integer 0..6, got '$FROM'" >&2; exit 2; fi

cd "$(dirname "$0")/.." || { echo "cannot cd to repo root" >&2; exit 1; }
export MUJOCO_GL=egl

CUR=seq_hetero
RES=results/envB
REF=$RES/${CUR}__seq_ft__seed$SEED
GPM=$RES/${CUR}__gpm_projected_adam__seed$SEED
E99=$RES/${CUR}__gpm_projected_adam_e99__seed$SEED
if [ "$SEED" = 0 ]; then SUF=""; else SUF="_seed$SEED"; fi
REP_GPM=$RES/gpm_seq$SUF/report.json
REP_E99=$RES/gpm_seq_e99_seed$SEED/report.json
ID_GPM=$RES/identity_gpm_vs_seqft_seed${SEED}_stage0.json
RUN_ARGS=(--curriculum "$CUR" --amp --seed "$SEED" --results-root "$RES")

LOGDIR="${QUEUE_LOG_ROOT:-results/logs}/queue_$(date +%Y%m%d_%H%M%S)_envb_refs_seed$SEED"
mkdir -p "$LOGDIR"
QUEUE_LOG="$LOGDIR/queue.log"
FAILED=""

note() { echo "$(date '+%F %T') $*" | tee -a "$QUEUE_LOG"; }

step() {
    local name=$1; shift
    note "START $name: $*"
    local rc
    if [ "${QUEUE_DRY_RUN:-0}" = "1" ]; then
        echo "$*" > "$LOGDIR/$name.log"
        if [ "${QUEUE_FAIL_STEP:-}" = "$name" ]; then rc=1; else rc=0; fi
    else
        "$@" > "$LOGDIR/$name.log" 2>&1
        rc=$?
    fi
    note "END $name rc=$rc"
    if [ $rc -ne 0 ]; then FAILED="$FAILED $name"; fi
    return $rc
}

want() { [ "$1" -ge "$FROM" ]; }

prior() {
    local name=$1 path=$2
    if [ -e "$path" ]; then note "PRIOR $name ok: $path"; return 0; fi
    note "PRIOR $name missing: $path"
    FAILED="$FAILED $name(missing)"
    return 1
}

# run_or_prior K NAME OUTPUT CMD...: run step K, or (before --from-step) check its output.
run_or_prior() {
    local k=$1 name=$2 path=$3; shift 3
    if want "$k"; then step "${k}_$name" "$@"
    else prior "${k}_$name" "$path"; fi
}

finish() {
    if [ -z "$FAILED" ]; then note "QUEUE DONE ok"; exit 0; fi
    note "QUEUE DONE FAILED:$FAILED"; exit 1
}

# The preflight: refuses the launch on a dirty tree, a busy GPU, CUDA kernels that do not run, a
# target that already exists, or less than 60 GB free on the results disk.
preflight() {
    local ok=1 t
    if [ -n "$(git status --porcelain)" ]; then echo "dirty tree:"; git status --porcelain; ok=0; fi
    local pids
    pids=$(timeout 30 nvidia-smi --query-compute-apps=pid --format=csv,noheader) || { echo "nvidia-smi failed"; ok=0; }
    if [ -n "$pids" ]; then echo "GPU busy: $pids"; ok=0; fi
    uv run python -m pytest -q -m gpu tests/test_environment.py || { echo "CUDA kernels do not run"; ok=0; }
    # a training step that will run must not find its run directory (move a partial one aside)
    want 1 && [ -e "$REF" ] && { echo "target exists: $REF"; ok=0; }
    want 2 && [ -e "$GPM" ] && { echo "target exists: $GPM"; ok=0; }
    want 3 && [ -e "$E99" ] && { echo "target exists: $E99"; ok=0; }
    local free_gb
    free_gb=$(df -BG --output=avail "$(pwd)/results" | tail -1 | tr -dc '0-9')
    if [ "$free_gb" -lt 60 ]; then echo "only ${free_gb} GB free"; ok=0; fi
    echo "git $(git rev-parse HEAD), $(uv run python -c 'import torch; print(torch.__version__, torch.cuda.get_device_name(0))')"
    [ $ok = 1 ]
}

PY="uv run python"
note "QUEUE START envb_refs seed=$SEED from_step=$FROM git $(git rev-parse HEAD) status '$(git status --porcelain | wc -l) changes'"

# The preflight runs on every launch, resumes included.
step 0_preflight preflight || { note "STOP: the preflight refused the launch"; finish; }

ref_ok=0; gpm_ok=0; e99_ok=0
run_or_prior 1 seqft "$REF/result.json" \
    $PY scripts/run_continual.py "${RUN_ARGS[@]}" --method seq_ft && ref_ok=1
if [ $ref_ok = 1 ]; then
    run_or_prior 2 gpm "$GPM/result.json" \
        $PY scripts/run_continual.py "${RUN_ARGS[@]}" --method gpm --t1-reference-run "$REF" && gpm_ok=1
else
    note "SKIP 2-6: 1_seqft failed or is missing"; FAILED="$FAILED 2-6(skipped)"; finish
fi
if [ $gpm_ok = 1 ]; then
    run_or_prior 3 e99 "$E99/result.json" \
        $PY scripts/run_continual.py "${RUN_ARGS[@]}" --method gpm_e99 --t1-reference-run "$REF" \
        --identity-reference-run "$GPM" --identity-stages 0 1 && e99_ok=1
    run_or_prior 4 seqrep_gpm "$REP_GPM" \
        $PY scripts/sequence_report.py --method-run "$GPM" --reference-run "$REF" --out "$REP_GPM"
    run_or_prior 6 identity_gpm "$ID_GPM" \
        $PY -c "import json, sys; from pathlib import Path; from flowcl.train.continual import stage_identity_check as c; r = c(Path(sys.argv[1]) / 'checkpoints' / 'stage0.pt', Path(sys.argv[2]), 0); Path(sys.argv[3]).write_text(json.dumps(r, indent=2) + '\n'); print(r['n_different'], 'of', r['n_tensors'], 'tensors differ')" "$GPM" "$REF" "$ID_GPM"
else
    note "SKIP 3-6: 2_gpm failed or is missing"; FAILED="$FAILED 3-6(skipped)"
fi
if [ $e99_ok = 1 ]; then
    run_or_prior 5 seqrep_e99 "$REP_E99" \
        $PY scripts/sequence_report.py --method-run "$E99" --reference-run "$REF" --out "$REP_E99"
elif [ $gpm_ok = 1 ]; then
    note "SKIP 5: 3_e99 failed or is missing"; FAILED="$FAILED 5(skipped)"
fi
finish
