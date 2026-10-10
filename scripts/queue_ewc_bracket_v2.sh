#!/usr/bin/env bash
# EWC λ bracket v2 (Workstream B): the T1 Fisher diagnostics, the seven pre-registered arms (sampled
# Fisher, mean-normalized), the selection (with the single edge extension if r = 10000 is selected), and a
# final verification. Orchestration only; the rule is configs/analysis/baseline_selection_ewc_v2.yaml,
# which must already be calibrated and committed. v1's queue (queue_ewc_bracket.sh) is its executed record
# and is not used. Run it only when no other GPU job is running.
#
# Usage:    bash scripts/queue_ewc_bracket_v2.sh [--from-step K] [--smoke DIR]
#   --from-step K  resume: arm steps < K are not rerun (status from their published rows); the diagnostics
#                  step (0) runs only when K = 0; selection, extension and verification always run. K in 0..8.
#   --smoke DIR    the smoke pipeline into DIR (outside results/), on a dirty tree: 4-task seq_ft and EWC
#                  (v2's estimator and normalization) at 200 steps and 1 episode, EWC identity-checked
#                  against seq_ft at stage 0, EWC resumed at stage 3 and compared with compare_resume
#                  (every artifact, tensors, per-episode evaluations), the diagnostics at 64 samples, and
#                  one bracket arm at 200 steps whose T1 Fisher must equal the calibration's.
#
#   0     diagnostics                          -> results/baseline_selection_v2/ewc_fisher_diagnostics.json
#   1-7   arm r = 0.01 ... 10000               -> results/baseline_selection_v2/ewc/seed0/ewcb__r<r>__seed0
#   8     select (rc 10 = the edge extension is owed: not a failure)
#   9     extension arm (only after rc 10)
#   10    select again (only after the extension)
#   11    verify: every required arm and the bundle under this rule and calibration
#
# Status: the last line is "QUEUE DONE ok" (exit 0) or "QUEUE DONE FAILED: <steps>" (exit 1).
# Launch:   tmux new-session -d -s ewcb2 "bash scripts/queue_ewc_bracket_v2.sh; exec bash"
# Testing:  QUEUE_DRY_RUN=1, QUEUE_FAIL_STEP=<step name>, QUEUE_STEP_RC=<step>=<rc>, QUEUE_LOG_ROOT,
#           QUEUE_RESULTS_ROOT.

set -u
USAGE="usage: queue_ewc_bracket_v2.sh [--from-step K] [--smoke DIR]"
FROM=0
SMOKE=""
while [ $# -gt 0 ]; do
    case "$1" in
        --from-step) FROM="${2:-}"; shift 2 || { echo "$USAGE" >&2; exit 2; } ;;
        --smoke) SMOKE="${2:-}"; shift 2 || { echo "$USAGE" >&2; exit 2; } ;;
        *) echo "unknown argument '$1'; $USAGE" >&2; exit 2 ;;
    esac
done
case "$FROM" in ''|*[!0-9]*) echo "--from-step must be an integer 0..8, got '$FROM'" >&2; exit 2;; esac
if [ "$FROM" -gt 8 ]; then echo "--from-step must be an integer 0..8, got '$FROM'" >&2; exit 2; fi

cd "$(dirname "$0")/.." || { echo "cannot cd to repo root" >&2; exit 1; }
export MUJOCO_GL=egl

RES="${QUEUE_RESULTS_ROOT:-results}"
RULE=configs/analysis/baseline_selection_ewc_v2.yaml
RATIOS="0.01 0.1 1 10 100 1000 10000"     # the rule's ewc.ratios (tested)
if [ -n "$SMOKE" ]; then
    case "$SMOKE" in /*) ;; *) echo "--smoke needs an absolute directory" >&2; exit 2 ;; esac
    case "$SMOKE/" in "$(pwd)/results/"*) echo "--smoke must be outside results/" >&2; exit 2 ;; esac
    mkdir -p "$SMOKE"
    TAG="_smoke"
else
    TAG=""
fi

LOGDIR="${QUEUE_LOG_ROOT:-results/logs}/queue_$(date +%Y%m%d_%H%M%S)_ewc_bracket_v2${TAG}"
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
        rc=0
        if [ "${QUEUE_FAIL_STEP:-}" = "$name" ]; then rc=1; fi
        case "${QUEUE_STEP_RC:-}" in "$name="*) rc="${QUEUE_STEP_RC#*=}" ;; esac
    else
        "$@" > "$LOGDIR/$name.log" 2>&1
        rc=$?
    fi
    note "END $name rc=$rc"
    return $rc
}

fail() { FAILED="$FAILED $1"; }

PY="uv run python"
BS=(scripts/baseline_selection.py --config "$RULE")
note "QUEUE START ewc_bracket_v2 smoke='$SMOKE' from_step=$FROM git $(git rev-parse HEAD) status '$(git status --porcelain | wc -l) changes'"

if [ -n "$SMOKE" ]; then
    CL=(--curriculum seq_hetero --seed 0 --amp --results-root "$SMOKE" --steps-per-task 200 --n-episodes 1 --allow-dirty)
    REF="$SMOKE/seq_hetero__seq_ft__seed0"
    step s0_smoke_method $PY "${BS[@]}" --write-smoke-method "$SMOKE/ewc_smoke.yaml" || fail s0_smoke_method
    step s1_seqft $PY scripts/run_continual.py "${CL[@]}" --method seq_ft || fail s1_seqft
    step s2_ewc $PY scripts/run_continual.py "${CL[@]}" --method "$SMOKE/ewc_smoke.yaml" \
        --t1-reference-run "$REF" --identity-reference-run "$REF" --identity-stages 0 || fail s2_ewc
    RCL=(--curriculum seq_hetero --seed 0 --amp --results-root "$SMOKE/resumed" --steps-per-task 200 --n-episodes 1 --allow-dirty)
    step s3_ewc_resume $PY scripts/run_continual.py "${RCL[@]}" --method "$SMOKE/ewc_smoke.yaml" \
        --resume-run "$SMOKE/seq_hetero__ewc__seed0" --start-stage 3 || fail s3_ewc_resume
    step s4_ewc_compare $PY scripts/compare_resume.py --source "$SMOKE/seq_hetero__ewc__seed0" \
        --resumed "$SMOKE/resumed/seq_hetero__ewc__seed0" --stage 3 --all-artifacts \
        --out "$SMOKE/resume_ewc.json" || fail s4_ewc_compare
    step s5_diagnostics $PY "${BS[@]}" --diagnostics --diag-samples 64 --out-root "$SMOKE/bracket" || fail s5_diagnostics
    step s6_bracket_arm $PY "${BS[@]}" --smoke-arm 1 --out-root "$SMOKE/bracket" --allow-dirty || fail s6_bracket_arm
else
    if [ "$FROM" = 0 ]; then
        step 0_diagnostics $PY "${BS[@]}" --diagnostics || fail 0_diagnostics
    else
        note "SKIP 0_diagnostics: resuming at step $FROM"
    fi
    K=1
    for R in $RATIOS; do
        ROW="$RES/baseline_selection_v2/ewc/seed0/ewcb__r${R}__seed0/row.json"
        if [ "$K" -ge "$FROM" ]; then
            step "${K}_arm_r$R" $PY "${BS[@]}" --arm "$R" || fail "${K}_arm_r$R"
        elif [ -e "$ROW" ]; then
            note "PRIOR ${K}_arm_r$R ok: $ROW"
        else
            note "PRIOR ${K}_arm_r$R missing: $ROW"; fail "${K}_arm_r$R(missing)"
        fi
        K=$((K + 1))
    done
    ARM_FAILED=0
    case "$FAILED" in *_arm_*) ARM_FAILED=1 ;; esac
    if [ $ARM_FAILED = 0 ]; then
        step 8_select $PY "${BS[@]}" --select
        rc=$?
        if [ $rc = 10 ]; then
            note "edge extension owed (rc 10)"
            if step 9_extension $PY "${BS[@]}" --extension; then
                step 10_select $PY "${BS[@]}" --select || fail 10_select
            else
                fail 9_extension
            fi
        elif [ $rc != 0 ]; then
            fail 8_select
        fi
        step 11_verify $PY "${BS[@]}" --verify || fail 11_verify
    else
        note "SKIP 8-11: an arm failed or is missing"
    fi
fi

if [ -z "$FAILED" ]; then
    note "QUEUE DONE ok"
    exit 0
fi
note "QUEUE DONE FAILED:$FAILED"
exit 1
