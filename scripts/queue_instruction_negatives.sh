#!/usr/bin/env bash
# The instruction-negatives pilot: the hold check, gpm_neg on similar_ab_stove, the stage-A panel and
# its discrimination gate, allowlist_ft_neg (identity at stage 0 against gpm_neg), the stage-1 swaps,
# the probes and the pre-registered summary. Orchestration only; the rule is
# configs/analysis/instruction_negatives.yaml. Run it only when no other GPU job is running.
#
# Usage:    bash scripts/queue_instruction_negatives.sh [--from-step K] [--smoke DIR]
#   --from-step K  resume: steps < K are not rerun (status from their outputs on disk; a missing
#                  output blocks its dependents). A run directory left by an interrupted training
#                  step must be moved aside first.
#   --smoke DIR    the whole pipeline at 200 steps, 1 episode per cell and 3 probe batches into DIR
#                  (outside results/), on a dirty tree; both runs whatever the gate says; the
#                  historical runs are read from results/; the summary runs with --smoke.
#
#    0  hold check (no policy)                        (~2 min)
#    1  gpm_neg run, --no-t1-check (needs 0)          (~2.6 h)
#    2  panel and gate on its stage0.pt (needs 1)     (~1 h)
#    3  allowlist_ft_neg run (needs 2; skipped when the gate failed, a valid outcome)  (~2.5 h)
#    4  stage-1 swaps: gpm (needs 1), allow (needs 3) (~40 min)
#    5  probes (needs 1)                              (~25 min)
#    6  summary (ALWAYS runs)                         -> instruction_negatives/summary.json
#
# Status: every non-zero rc and every missing prior output is collected; the last line is
# "QUEUE DONE ok" (exit 0) or "QUEUE DONE FAILED: <steps>" (exit 1). A gate failure is not a failure.
#
# Launch:   tmux new-session -d -s instrneg "bash scripts/queue_instruction_negatives.sh; exec bash"
# Testing:  QUEUE_DRY_RUN=1, QUEUE_FAIL_STEP=<step name>, QUEUE_GATE=pass|fail (dry run only),
#           QUEUE_LOG_ROOT, QUEUE_RESULTS_ROOT.

set -u
USAGE="usage: queue_instruction_negatives.sh [--from-step K] [--smoke DIR]"
FROM=0
SMOKE=""
while [ $# -gt 0 ]; do
    case "$1" in
        --from-step) FROM="${2:-}"; shift 2 || { echo "$USAGE" >&2; exit 2; } ;;
        --smoke) SMOKE="${2:-}"; shift 2 || { echo "$USAGE" >&2; exit 2; } ;;
        *) echo "unknown argument '$1'; $USAGE" >&2; exit 2 ;;
    esac
done
case "$FROM" in ''|*[!0-9]*) echo "--from-step must be an integer 0..6, got '$FROM'" >&2; exit 2;; esac
if [ "$FROM" -gt 6 ]; then echo "--from-step must be an integer 0..6, got '$FROM'" >&2; exit 2; fi

cd "$(dirname "$0")/.." || { echo "cannot cd to repo root" >&2; exit 1; }
export MUJOCO_GL=egl

CUR=similar_ab_stove
GPM_RUN="${CUR}__gpm_projected_adam_neg__seed0"
ALLOW_RUN="${CUR}__allowlist_ft_neg__seed0"
OUT=instruction_negatives

RUN_ARGS=(--curriculum "$CUR" --seed 0 --amp)
CL_ARGS=()
PANEL_ARGS=()
PROBE_ARGS=()
SUMMARY_ARGS=()
if [ -n "$SMOKE" ]; then
    case "$SMOKE" in /*) ;; *) echo "--smoke needs an absolute directory" >&2; exit 2 ;; esac
    case "$SMOKE/" in "$(pwd)/results/"*) echo "--smoke must be outside results/" >&2; exit 2 ;; esac
    RES="$SMOKE"
    mkdir -p "$RES"
    RUN_ARGS+=(--results-root "$RES" --steps-per-task 200 --n-episodes 1 --allow-dirty)
    CL_ARGS+=(--results-root "$RES" --allow-dirty)
    PANEL_ARGS+=(--smoke)
    PROBE_ARGS+=(--smoke-batches 3)
    SUMMARY_ARGS+=(--results-root "$RES" --historical-root "$(pwd)/results" --smoke)
    TAG="_smoke"
else
    RES="${QUEUE_RESULTS_ROOT:-results}"
    CL_ARGS+=(--results-root "$RES")
    SUMMARY_ARGS+=(--results-root "$RES")
    TAG=""
fi

LOGDIR="${QUEUE_LOG_ROOT:-results/logs}/queue_$(date +%Y%m%d_%H%M%S)_instruction_negatives${TAG}"
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

run_or_prior() {
    local k=$1 name=$2 path=$3; shift 3
    if want "$k"; then step "${k}_$name" "$@"
    else prior "${k}_$name" "$path"; fi
}

gate_ok() {
    if [ "${QUEUE_DRY_RUN:-0}" = "1" ]; then [ "${QUEUE_GATE:-pass}" = "pass" ]; return; fi
    $PY scripts/instruction_negatives.py --part gate "${CL_ARGS[@]}" > /dev/null 2>&1
}

PY="uv run python"
note "QUEUE START instruction_negatives smoke='$SMOKE' from_step=$FROM git $(git rev-parse HEAD) status '$(git status --porcelain | wc -l) changes' rule $(sha256sum configs/analysis/instruction_negatives.yaml | cut -c1-12) bank $(sha256sum configs/analysis/instruction_negatives_bank.yaml | cut -c1-12)"

HOLD_OK=0; GPM_OK=0; PANEL_OK=0; ALLOW_OK=0
run_or_prior 0 hold "$RES/$OUT/hold/hold.json" \
    $PY scripts/instruction_negatives.py --part hold "${CL_ARGS[@]}" && HOLD_OK=1

if [ "$HOLD_OK" = 1 ]; then
    run_or_prior 1 gpm "$RES/$GPM_RUN/result.json" \
        $PY scripts/run_continual.py "${RUN_ARGS[@]}" --method gpm_neg --no-t1-check && GPM_OK=1
else
    note "SKIP 1-5: 0_hold failed or is missing (the summary still runs)"
fi

if [ "$GPM_OK" = 1 ]; then
    REF="$RES/$GPM_RUN"
    run_or_prior 2 panel "$RES/$OUT/panel/panel.json" \
        $PY scripts/instruction_negatives.py --part panel "${CL_ARGS[@]}" "${PANEL_ARGS[@]}" && PANEL_OK=1
    if [ "$PANEL_OK" = 1 ] && { [ -n "$SMOKE" ] || gate_ok; }; then
        run_or_prior 3 allow "$RES/$ALLOW_RUN/result.json" \
            $PY scripts/run_continual.py "${RUN_ARGS[@]}" --method allowlist_ft_neg \
            --t1-reference-run "$REF" --identity-reference-run "$REF" --identity-stages 0 && ALLOW_OK=1
    elif [ "$PANEL_OK" = 1 ]; then
        note "SKIP 3: the discrimination gate failed (gate_failed, a valid outcome; allow-dependent steps skipped)"
    else
        note "SKIP 3: 2_panel failed or is missing"
    fi
    run_or_prior 4 swaps_gpm "$RES/$OUT/swaps/gpm/swaps.json" \
        $PY scripts/instruction_negatives.py --part swaps --arm gpm "${CL_ARGS[@]}"
    if [ "$ALLOW_OK" = 1 ]; then
        run_or_prior 4 swaps_allow "$RES/$OUT/swaps/allow/swaps.json" \
            $PY scripts/instruction_negatives.py --part swaps --arm allow "${CL_ARGS[@]}"
    fi
    run_or_prior 5 probes "$RES/$OUT/probes" \
        $PY scripts/instruction_negatives.py --part probe "${CL_ARGS[@]}" "${PROBE_ARGS[@]}"
elif [ "$HOLD_OK" = 1 ]; then
    note "SKIP 2-5: 1_gpm failed or is missing (the summary still runs)"
fi

step 6_summary $PY scripts/instruction_negatives.py --part summary "${SUMMARY_ARGS[@]}"

if [ -z "$FAILED" ]; then
    note "QUEUE DONE ok"
    exit 0
fi
note "QUEUE DONE FAILED:$FAILED"
exit 1
