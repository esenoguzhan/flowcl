#!/usr/bin/env bash
# Instruction negatives at seeds 1-2: per seed a new negatives base on stove (gpm_neg), its panel and
# gate, cabinet top resumed from it, and plain-GPM controls on both (stove full, cabinet top resumed);
# then the probes, the pre-registered summary, the GPM + neg swaps and the summary again.
# Orchestration only; the rule is configs/analysis/instruction_negatives_seeds.yaml. Run it only when
# no other GPU job is running.
#
# Usage:    bash scripts/queue_instruction_negatives_seeds.sh [--from-step K] [--smoke DIR]
#   --from-step K  resume: steps < K are not rerun (status from their outputs on disk; a missing
#                  output blocks its dependents). A run directory left by an interrupted training
#                  step must be moved aside first.
#   --smoke DIR    seed 1 only, 200 steps, 1 episode per cell, 3 probe batches, into DIR (outside
#                  results/), on a dirty tree; the gate is not applied; the summary runs with --smoke.
#
#    0      preflight: pins, hold record, clean tree, idle GPU, disk, targets absent
#    per seed (seed 1: steps 1-5, seed 2: steps 6-10):
#      +0   neg  stove    gpm_neg full run, --no-t1-check                 (~2 h 10)
#      +1   panel and gate on its stage0.pt (needs +0)                    (~40 min)
#      +2   neg  cabinet  resumed at stage 1 from +0's stage0.pt (needs +0) (~1 h)
#      +3   ctrl stove    plain GPM full run, --no-t1-check                (~1 h 45)
#      +4   ctrl cabinet  resumed at stage 1 from +3's stage0.pt (needs +3) (~1 h)
#    11     probes (12)                                                    (~25 min)
#    12     summary (ALWAYS runs)            -> instruction_negatives_seeds/summary.json
#    13     swaps, GPM + neg: stove (needs +0), cabinet (needs +2), per seed (~1.5 h)
#    14     summary again (ALWAYS runs)
#
# Status: every non-zero rc and every missing prior output is collected; the last line is
# "QUEUE DONE ok" (exit 0) or "QUEUE DONE FAILED: <steps>" (exit 1). A failed preflight skips 1-11, 13.
#
# Launch:   tmux new-session -d -s instrnegseeds "bash scripts/queue_instruction_negatives_seeds.sh; exec bash"
# Testing:  QUEUE_DRY_RUN=1, QUEUE_FAIL_STEP=<step name>, QUEUE_LOG_ROOT, QUEUE_RESULTS_ROOT.

set -u
USAGE="usage: queue_instruction_negatives_seeds.sh [--from-step K] [--smoke DIR]"
FROM=0
SMOKE=""
while [ $# -gt 0 ]; do
    case "$1" in
        --from-step) FROM="${2:-}"; shift 2 || { echo "$USAGE" >&2; exit 2; } ;;
        --smoke) SMOKE="${2:-}"; shift 2 || { echo "$USAGE" >&2; exit 2; } ;;
        *) echo "unknown argument '$1'; $USAGE" >&2; exit 2 ;;
    esac
done
case "$FROM" in ''|*[!0-9]*) echo "--from-step must be an integer 0..14, got '$FROM'" >&2; exit 2;; esac
if [ "$FROM" -gt 14 ]; then echo "--from-step must be an integer 0..14, got '$FROM'" >&2; exit 2; fi

cd "$(dirname "$0")/.." || { echo "cannot cd to repo root" >&2; exit 1; }
export MUJOCO_GL=egl

OUT=instruction_negatives_seeds
SEEDS=(1 2)
RUN_EXTRA=()
CL_ARGS=()
SMOKE_ARGS=()
PROBE_ARGS=()
if [ -n "$SMOKE" ]; then
    case "$SMOKE" in /*) ;; *) echo "--smoke needs an absolute directory" >&2; exit 2 ;; esac
    case "$SMOKE/" in "$(pwd)/results/"*) echo "--smoke must be outside results/" >&2; exit 2 ;; esac
    RES="$SMOKE"
    mkdir -p "$RES"
    SEEDS=(1)
    RUN_EXTRA+=(--results-root "$RES" --steps-per-task 200 --n-episodes 1 --allow-dirty)
    CL_ARGS+=(--results-root "$RES" --allow-dirty)
    SMOKE_ARGS+=(--smoke --seeds 1)
    PROBE_ARGS+=(--smoke-batches 3)
    TAG="_smoke"
else
    RES="${QUEUE_RESULTS_ROOT:-results}"
    CL_ARGS+=(--results-root "$RES")
    TAG=""
fi

LOGDIR="${QUEUE_LOG_ROOT:-results/logs}/queue_$(date +%Y%m%d_%H%M%S)_instruction_negatives_seeds${TAG}"
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

stage0_sha() {
    if [ "${QUEUE_DRY_RUN:-0}" = "1" ]; then echo DRYRUN; return; fi
    sha256sum "$1/checkpoints/stage0.pt" 2>/dev/null | cut -d' ' -f1
}

PY="uv run python"
note "QUEUE START instruction_negatives_seeds smoke='$SMOKE' seeds='${SEEDS[*]}' from_step=$FROM git $(git rev-parse HEAD) status '$(git status --porcelain | wc -l) changes' rule $(sha256sum configs/analysis/instruction_negatives_seeds.yaml | cut -c1-12) free '$(df -h --output=avail / | tail -1 | tr -d ' ')'"

PRE_OK=0
run_or_prior 0 preflight "$RES/$OUT/preflight.json" \
    $PY scripts/instruction_negatives_seeds.py --part preflight "${CL_ARGS[@]}" "${SMOKE_ARGS[@]}" && PRE_OK=1

declare -A OK=()
if [ "$PRE_OK" = 1 ]; then
    i=0
    for N in "${SEEDS[@]}"; do
        k=$((1 + 5 * i))
        RUN_ARGS=(--seed "$N" --amp "${RUN_EXTRA[@]}")
        NEG_STOVE="similar_ab_stove__gpm_projected_adam_neg__seed$N"
        NEG_CAB="similar_ab_cabinet__gpm_projected_adam_neg__seed$N"
        CTRL_STOVE="similar_ab_stove__gpm_projected_adam__seed$N"
        CTRL_CAB="similar_ab_cabinet__gpm_projected_adam__seed$N"
        OK[neg_stove_$N]=0; OK[neg_cab_$N]=0; OK[ctrl_stove_$N]=0; OK[ctrl_cab_$N]=0
        run_or_prior "$k" "neg_stove_s$N" "$RES/$NEG_STOVE/result.json" \
            $PY scripts/run_continual.py --curriculum similar_ab_stove "${RUN_ARGS[@]}" --method gpm_neg --no-t1-check \
            && OK[neg_stove_$N]=1
        if [ "${OK[neg_stove_$N]}" = 1 ]; then
            run_or_prior $((k + 1)) "panel_s$N" "$RES/$OUT/seed$N/stove/panel/panel.json" \
                $PY scripts/instruction_negatives_seeds.py --part panel --seed "$N" "${CL_ARGS[@]}" "${SMOKE_ARGS[@]}"
            PIN=$(stage0_sha "$RES/$NEG_STOVE")
            if want $((k + 2)); then note "RESUME $((k + 2))_neg_cab_s$N from $RES/$NEG_STOVE/checkpoints/stage0.pt sha256 ${PIN:-MISSING}"; fi
            run_or_prior $((k + 2)) "neg_cab_s$N" "$RES/$NEG_CAB/result.json" \
                $PY scripts/run_continual.py --curriculum similar_ab_cabinet "${RUN_ARGS[@]}" --method gpm_neg \
                --resume-run "$RES/$NEG_STOVE" --start-stage 1 --resume-sha256 "${PIN:-MISSING}" && OK[neg_cab_$N]=1
        else
            note "SKIP $((k + 1))-$((k + 2)) (seed $N): ${k}_neg_stove_s$N failed or is missing"
        fi
        run_or_prior $((k + 3)) "ctrl_stove_s$N" "$RES/$CTRL_STOVE/result.json" \
            $PY scripts/run_continual.py --curriculum similar_ab_stove "${RUN_ARGS[@]}" --method gpm --no-t1-check \
            && OK[ctrl_stove_$N]=1
        if [ "${OK[ctrl_stove_$N]}" = 1 ]; then
            PIN=$(stage0_sha "$RES/$CTRL_STOVE")
            if want $((k + 4)); then note "RESUME $((k + 4))_ctrl_cab_s$N from $RES/$CTRL_STOVE/checkpoints/stage0.pt sha256 ${PIN:-MISSING}"; fi
            run_or_prior $((k + 4)) "ctrl_cab_s$N" "$RES/$CTRL_CAB/result.json" \
                $PY scripts/run_continual.py --curriculum similar_ab_cabinet "${RUN_ARGS[@]}" --method gpm \
                --resume-run "$RES/$CTRL_STOVE" --start-stage 1 --resume-sha256 "${PIN:-MISSING}" && OK[ctrl_cab_$N]=1
        else
            note "SKIP $((k + 4)) (seed $N): $((k + 3))_ctrl_stove_s$N failed or is missing"
        fi
        i=$((i + 1))
    done
    run_or_prior 11 probes "$RES/$OUT/seed${SEEDS[0]}/stove/probes" \
        $PY scripts/instruction_negatives_seeds.py --part probe "${CL_ARGS[@]}" "${SMOKE_ARGS[@]}" "${PROBE_ARGS[@]}"
else
    note "SKIP 1-11: 0_preflight failed or is missing (the summary still runs)"
fi

step 12_summary $PY scripts/instruction_negatives_seeds.py --part summary "${CL_ARGS[@]}" "${SMOKE_ARGS[@]}"

if [ "$PRE_OK" = 1 ]; then
    for N in "${SEEDS[@]}"; do
        if [ "${OK[neg_stove_$N]}" = 1 ]; then
            run_or_prior 13 "swaps_stove_s$N" "$RES/$OUT/seed$N/stove/swaps/gpm/swaps.json" \
                $PY scripts/instruction_negatives_seeds.py --part swaps --seed "$N" --b stove "${CL_ARGS[@]}"
        else
            note "SKIP 13 swaps_stove_s$N: the neg stove run failed or is missing"
        fi
        if [ "${OK[neg_cab_$N]}" = 1 ]; then
            run_or_prior 13 "swaps_cabinet_s$N" "$RES/$OUT/seed$N/cabinet/swaps/gpm/swaps.json" \
                $PY scripts/instruction_negatives_seeds.py --part swaps --seed "$N" --b cabinet "${CL_ARGS[@]}"
        else
            note "SKIP 13 swaps_cabinet_s$N: the neg cabinet run failed or is missing"
        fi
    done
fi

step 14_summary $PY scripts/instruction_negatives_seeds.py --part summary "${CL_ARGS[@]}" "${SMOKE_ARGS[@]}"

if [ -z "$FAILED" ]; then
    note "QUEUE DONE ok"
    exit 0
fi
note "QUEUE DONE FAILED:$FAILED"
exit 1
