#!/usr/bin/env bash
# C2, the reverse-order control for one seed: seq_ft, plain GPM and adaptive GPM on
# seq_hetero_reverse, their sequence reports, and the pre-registered summary. Orchestration only;
# the rule is configs/analysis/reverse_order.yaml. scripts/queue_seed_pair.sh (the forward runs'
# executed record) is not used: every output here is tagged _reverse, and every reference is passed
# explicitly. Run it only when no other GPU job is running.
#
# Usage:    bash scripts/queue_reverse_order.sh <seed> [--from-step K] [--smoke DIR]
#   --from-step K  resume: steps < K are not rerun (status from their outputs on disk; a missing
#                  output blocks its dependents). A run directory left by an interrupted training
#                  step must be moved aside first.
#   --smoke DIR    the whole pipeline at 200 steps and 1 episode per cell into DIR (outside
#                  results/), on a dirty tree, without --single-task-baseline; the summary runs with
#                  --smoke (scratch only, discarded).
#
#   0  seq_ft            (~5.6 h)
#   1  plain GPM         (~5.0 h; T1 pairing against step 0's run)
#   2  adaptive GPM      (~4.7 h; T1 pairing, identity against step 1's run at stages 0-1)
#   3  sequence report, plain GPM      -> gpm_seq_reverse_seed<S>/report.json
#   4  sequence report, adaptive GPM   -> gpm_seq_ne90_reverse_seed<S>/report.json
#   5  summary (ALWAYS runs)           -> reverse_order/seed<S>/summary.json
#
# Status: every non-zero rc and every missing prior output is collected; the last line is
# "QUEUE DONE ok" (exit 0) or "QUEUE DONE FAILED: <steps>" (exit 1). A valid scientific verdict,
# including "inconclusive", is summary rc 0.
#
# Launch:   tmux new-session -d -s reverse0 "bash scripts/queue_reverse_order.sh 0; exec bash"
# Testing:  QUEUE_DRY_RUN=1, QUEUE_FAIL_STEP=<step name>, QUEUE_LOG_ROOT, QUEUE_RESULTS_ROOT.

set -u
USAGE="usage: queue_reverse_order.sh <seed> [--from-step K] [--smoke DIR]"
SEED="${1:?$USAGE}"
shift
FROM=0
SMOKE=""
while [ $# -gt 0 ]; do
    case "$1" in
        --from-step) FROM="${2:-}"; shift 2 || { echo "$USAGE" >&2; exit 2; } ;;
        --smoke) SMOKE="${2:-}"; shift 2 || { echo "$USAGE" >&2; exit 2; } ;;
        *) echo "unknown argument '$1'; $USAGE" >&2; exit 2 ;;
    esac
done
case "$SEED" in ''|*[!0-9]*) echo "seed must be a non-negative integer, got '$SEED'" >&2; exit 2;; esac
case "$FROM" in ''|*[!0-9]*) echo "--from-step must be an integer 0..5, got '$FROM'" >&2; exit 2;; esac
if [ "$FROM" -gt 5 ]; then echo "--from-step must be an integer 0..5, got '$FROM'" >&2; exit 2; fi

cd "$(dirname "$0")/.." || { echo "cannot cd to repo root" >&2; exit 1; }
export MUJOCO_GL=egl

S=$SEED
CUR=seq_hetero_reverse
REF=${CUR}__seq_ft__seed$S
GPM=${CUR}__gpm_projected_adam__seed$S
NE90=${CUR}__gpm_projected_adam_ne90__seed$S
RUN_ARGS=(--curriculum "$CUR" --seed "$S" --amp)
SUMMARY_ARGS=(--seed "$S")
if [ -n "$SMOKE" ]; then
    case "$SMOKE" in /*) ;; *) echo "--smoke needs an absolute directory" >&2; exit 2 ;; esac
    case "$SMOKE/" in "$(pwd)/results/"*) echo "--smoke must be outside results/" >&2; exit 2 ;; esac
    RES="$SMOKE"
    mkdir -p "$RES"
    RUN_ARGS+=(--results-root "$RES" --steps-per-task 200 --n-episodes 1 --allow-dirty)
    SUMMARY_ARGS+=(--results-root "$RES" --out "$RES/reverse_order/seed$S" --smoke)
    TAG="_smoke"
else
    RES="${QUEUE_RESULTS_ROOT:-results}"
    if [ "$S" = 0 ]; then RUN_ARGS+=(--single-task-baseline); fi   # Gate 0 refs exist for seed 0 only
    TAG=""
fi

LOGDIR="${QUEUE_LOG_ROOT:-results/logs}/queue_$(date +%Y%m%d_%H%M%S)_reverse_order_seed${S}${TAG}"
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

PY="uv run python"
note "QUEUE START reverse_order seed $S smoke='$SMOKE' from_step=$FROM git $(git rev-parse HEAD) status '$(git status --porcelain | wc -l) changes'"

ref_ok=0; gpm_ok=0; ne90_ok=0
run_or_prior 0 seqft "$RES/$REF/result.json" \
    $PY scripts/run_continual.py "${RUN_ARGS[@]}" --method seq_ft && ref_ok=1

if [ $ref_ok = 1 ]; then
    run_or_prior 1 gpm "$RES/$GPM/result.json" \
        $PY scripts/run_continual.py "${RUN_ARGS[@]}" --method gpm \
        --t1-reference-run "$RES/$REF" && gpm_ok=1
else
    note "SKIP 1-4: 0_seqft failed or is missing (the summary still runs)"
fi

if [ $gpm_ok = 1 ]; then
    run_or_prior 2 ne90 "$RES/$NE90/result.json" \
        $PY scripts/run_continual.py "${RUN_ARGS[@]}" --method gpm_ne90 \
        --t1-reference-run "$RES/$REF" --identity-reference-run "$RES/$GPM" --identity-stages 0 1 \
        && ne90_ok=1
    run_or_prior 3 seqrep_gpm "$RES/gpm_seq_reverse_seed$S/report.json" \
        $PY scripts/sequence_report.py --method-run "$RES/$GPM" --reference-run "$RES/$REF" \
        --out "$RES/gpm_seq_reverse_seed$S/report.json"
elif [ $ref_ok = 1 ]; then
    note "SKIP 2-4: 1_gpm failed or is missing (the summary still runs)"
fi

if [ $ne90_ok = 1 ]; then
    run_or_prior 4 seqrep_ne90 "$RES/gpm_seq_ne90_reverse_seed$S/report.json" \
        $PY scripts/sequence_report.py --method-run "$RES/$NE90" --reference-run "$RES/$REF" \
        --out "$RES/gpm_seq_ne90_reverse_seed$S/report.json"
elif [ $gpm_ok = 1 ]; then
    note "SKIP 4: 2_ne90 failed or is missing (the summary still runs)"
fi

step 5_summary $PY scripts/reverse_order.py "${SUMMARY_ARGS[@]}"

if [ -z "$FAILED" ]; then
    note "QUEUE DONE ok"
    exit 0
fi
note "QUEUE DONE FAILED:$FAILED"
exit 1
