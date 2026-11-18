#!/usr/bin/env bash
# Instruction negatives on the other two B tasks of the 5 Oct study (cabinet top, milk), seed 0: four
# stage-1 runs RESUMED from the stove pilot's stage-A checkpoints (gpm_neg from gpm_neg,
# allowlist_ft_neg from allowlist_ft_neg), the cabinet stage-1 swaps, the probes and the
# pre-registered summary. Orchestration only; the rule is
# configs/analysis/instruction_negatives_curricula.yaml. Run it only when no other GPU job is running.
#
# Usage:    bash scripts/queue_instruction_negatives_curricula.sh [--from-step K] [--smoke DIR --smoke-sources SRC]
#   --from-step K   resume: steps < K are not rerun (status from their outputs on disk; a missing
#                   output blocks its dependents). A run directory left by an interrupted training
#                   step must be moved aside first.
#   --smoke DIR     200 steps, 1 episode per cell, 3 probe batches, into DIR (outside results/), on a
#                   dirty tree, resuming from the stove smoke in SRC (its two runs and its
#                   instruction_negatives/ folder); pins computed from SRC; historical runs from
#                   results/. Adds step 1e: gpm_neg resumed into similar_ab_stove from SRC must equal
#                   SRC's stage1.pt bit for bit (the runner's identity check and compare_resume).
#
#    0  preflight: pins, the sources' equal weights, clean tree, idle GPU, disk, targets absent
#    1  cabinet  gpm_neg           resumed from the stove gpm_neg source         (~50 min)
#    2  cabinet  allowlist_ft_neg  resumed from the stove allowlist_ft_neg source (~45 min)
#    3  milk     gpm_neg                                                         (~50 min)
#    4  milk     allowlist_ft_neg                                                (~45 min)
#    5  cabinet stage-1 swaps: gpm (needs 1), allow (needs 2)                     (~40 min)
#    6  probes on every stage-1 checkpoint                                        (~5 min)
#    7  summary (ALWAYS runs)                -> instruction_negatives_curricula/summary.json
#
# Status: every non-zero rc and every missing prior output is collected; the last line is
# "QUEUE DONE ok" (exit 0) or "QUEUE DONE FAILED: <steps>" (exit 1). A failed preflight skips 1-6.
#
# Launch:   tmux new-session -d -s instrnegcur "bash scripts/queue_instruction_negatives_curricula.sh; exec bash"
# Testing:  QUEUE_DRY_RUN=1, QUEUE_FAIL_STEP=<step name>, QUEUE_LOG_ROOT, QUEUE_RESULTS_ROOT.

set -u
USAGE="usage: queue_instruction_negatives_curricula.sh [--from-step K] [--smoke DIR --smoke-sources SRC]"
FROM=0
SMOKE=""
SMOKE_SRC=""
while [ $# -gt 0 ]; do
    case "$1" in
        --from-step) FROM="${2:-}"; shift 2 || { echo "$USAGE" >&2; exit 2; } ;;
        --smoke) SMOKE="${2:-}"; shift 2 || { echo "$USAGE" >&2; exit 2; } ;;
        --smoke-sources) SMOKE_SRC="${2:-}"; shift 2 || { echo "$USAGE" >&2; exit 2; } ;;
        *) echo "unknown argument '$1'; $USAGE" >&2; exit 2 ;;
    esac
done
case "$FROM" in ''|*[!0-9]*) echo "--from-step must be an integer 0..7, got '$FROM'" >&2; exit 2;; esac
if [ "$FROM" -gt 7 ]; then echo "--from-step must be an integer 0..7, got '$FROM'" >&2; exit 2; fi
if [ -n "$SMOKE" ] && [ -z "$SMOKE_SRC" ]; then echo "--smoke needs --smoke-sources" >&2; exit 2; fi
if [ -z "$SMOKE" ] && [ -n "$SMOKE_SRC" ]; then echo "--smoke-sources needs --smoke" >&2; exit 2; fi

cd "$(dirname "$0")/.." || { echo "cannot cd to repo root" >&2; exit 1; }
export MUJOCO_GL=egl

GPM_SRC=similar_ab_stove__gpm_projected_adam_neg__seed0
ALLOW_SRC=similar_ab_stove__allowlist_ft_neg__seed0
OUT=instruction_negatives_curricula

RUN_ARGS=(--seed 0 --amp)
CL_ARGS=()
PRE_ARGS=()
PROBE_ARGS=()
SUMMARY_ARGS=()
if [ -n "$SMOKE" ]; then
    for d in "$SMOKE" "$SMOKE_SRC"; do
        case "$d" in /*) ;; *) echo "--smoke and --smoke-sources need absolute directories" >&2; exit 2 ;; esac
    done
    case "$SMOKE/" in "$(pwd)/results/"*) echo "--smoke must be outside results/" >&2; exit 2 ;; esac
    RES="$SMOKE"
    SRC="$SMOKE_SRC"
    mkdir -p "$RES"
    RUN_ARGS+=(--results-root "$RES" --steps-per-task 200 --n-episodes 1 --allow-dirty)
    CL_ARGS+=(--results-root "$RES" --sources-root "$SRC" --allow-dirty)
    PRE_ARGS+=(--results-root "$RES" --sources-root "$SRC" --smoke)
    PROBE_ARGS+=(--smoke --smoke-batches 3)
    SUMMARY_ARGS+=(--results-root "$RES" --sources-root "$SRC" --smoke)
    TAG="_smoke"
else
    RES="${QUEUE_RESULTS_ROOT:-results}"
    SRC="$RES"
    CL_ARGS+=(--results-root "$RES")
    PRE_ARGS+=(--results-root "$RES")
    SUMMARY_ARGS+=(--results-root "$RES")
    TAG=""
fi

LOGDIR="${QUEUE_LOG_ROOT:-results/logs}/queue_$(date +%Y%m%d_%H%M%S)_instruction_negatives_curricula${TAG}"
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

PY="uv run python"
note "QUEUE START instruction_negatives_curricula smoke='$SMOKE' sources='$SRC' from_step=$FROM git $(git rev-parse HEAD) status '$(git status --porcelain | wc -l) changes' rule $(sha256sum configs/analysis/instruction_negatives_curricula.yaml | cut -c1-12) free '$(df -h --output=avail / | tail -1 | tr -d ' ')'"

# the stage-0 SHA-256 each resumed run must find (the rule's pins; in smoke, the smoke sources')
PIN_ARGS=(--sources-root "$SRC")
if [ -n "$SMOKE" ]; then PIN_ARGS+=(--results-root "$RES" --smoke); fi
PINS=$($PY scripts/instruction_negatives_curricula.py --part pins "${PIN_ARGS[@]}" 2>/dev/null)
PIN_GPM=$(echo "$PINS" | awk '$1=="gpm"{print $2}')
PIN_ALLOW=$(echo "$PINS" | awk '$1=="allow"{print $2}')
note "PINS gpm ${PIN_GPM:-MISSING} allow ${PIN_ALLOW:-MISSING}"
if [ -z "$PIN_GPM" ] || [ -z "$PIN_ALLOW" ] || [ "$PIN_GPM" = "None" ] || [ "$PIN_ALLOW" = "None" ]; then
    if [ "${QUEUE_DRY_RUN:-0}" = "1" ]; then PIN_GPM=DRYRUN; PIN_ALLOW=DRYRUN; else PIN_GPM=MISSING; PIN_ALLOW=MISSING; fi
fi

PRE_OK=0
run_or_prior 0 preflight "$RES/$OUT/preflight.json" \
    $PY scripts/instruction_negatives_curricula.py --part preflight "${PRE_ARGS[@]}" && PRE_OK=1

declare -A OK=([cabinet_gpm]=0 [cabinet_allow]=0 [milk_gpm]=0 [milk_allow]=0)
# resumed K CURRICULUM SHORT ARM: one stage-1 run from the stove source of the same arm
resumed() {
    local k=$1 cur=$2 short=$3 arm=$4 method src pin target
    case $arm in
        gpm) method=gpm_neg; src=$GPM_SRC; pin=$PIN_GPM; target="${cur}__gpm_projected_adam_neg__seed0" ;;
        allow) method=allowlist_ft_neg; src=$ALLOW_SRC; pin=$PIN_ALLOW; target="${cur}__allowlist_ft_neg__seed0" ;;
    esac
    if want "$k"; then note "RESUME ${k}_${short}_$arm from $SRC/$src/checkpoints/stage0.pt sha256 $pin"; fi
    run_or_prior "$k" "${short}_$arm" "$RES/$target/result.json" \
        $PY scripts/run_continual.py --curriculum "$cur" "${RUN_ARGS[@]}" --method "$method" \
        --resume-run "$SRC/$src" --start-stage 1 --resume-sha256 "$pin" && OK[${short}_$arm]=1
}

if [ "$PRE_OK" = 1 ]; then
    resumed 1 similar_ab_cabinet cabinet gpm
    if [ -n "$SMOKE" ] && want 1; then
        # smoke only: resuming reproduces a full run for the new method (bit for bit at stage 1)
        step 1e_equality_run $PY scripts/run_continual.py --curriculum similar_ab_stove --seed 0 --amp \
            --results-root "$RES/equality" --steps-per-task 200 --n-episodes 1 --allow-dirty --method gpm_neg \
            --resume-run "$SRC/$GPM_SRC" --start-stage 1 --resume-sha256 "$PIN_GPM" \
            --identity-reference-run "$SRC/$GPM_SRC" --identity-stages 1 \
            && step 1e_equality_check $PY scripts/instruction_negatives_curricula.py --part resume-equality \
                --source-run "$SRC/$GPM_SRC" --resumed-run "$RES/equality/$GPM_SRC" --results-root "$RES"
    fi
    resumed 2 similar_ab_cabinet cabinet allow
    resumed 3 similar_ab_milk milk gpm
    resumed 4 similar_ab_milk milk allow
    if [ "${OK[cabinet_gpm]}" = 1 ]; then
        run_or_prior 5 swaps_cabinet_gpm "$RES/$OUT/cabinet/swaps/gpm/swaps.json" \
            $PY scripts/instruction_negatives_curricula.py --part swaps --curriculum cabinet --arm gpm "${CL_ARGS[@]}"
    else
        note "SKIP 5 swaps_cabinet_gpm: 1_cabinet_gpm failed or is missing"
    fi
    if [ "${OK[cabinet_allow]}" = 1 ]; then
        run_or_prior 5 swaps_cabinet_allow "$RES/$OUT/cabinet/swaps/allow/swaps.json" \
            $PY scripts/instruction_negatives_curricula.py --part swaps --curriculum cabinet --arm allow "${CL_ARGS[@]}"
    else
        note "SKIP 5 swaps_cabinet_allow: 2_cabinet_allow failed or is missing"
    fi
    run_or_prior 6 probes "$RES/$OUT/cabinet/probes" \
        $PY scripts/instruction_negatives_curricula.py --part probe "${CL_ARGS[@]}" "${PROBE_ARGS[@]}"
else
    note "SKIP 1-6: 0_preflight failed or is missing (the summary still runs)"
fi

step 7_summary $PY scripts/instruction_negatives_curricula.py --part summary "${SUMMARY_ARGS[@]}"

if [ -z "$FAILED" ]; then
    note "QUEUE DONE ok"
    exit 0
fi
note "QUEUE DONE FAILED:$FAILED"
exit 1
