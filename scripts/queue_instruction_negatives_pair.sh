#!/usr/bin/env bash
# Instruction negatives on one A -> B pair (rule configs/analysis/instruction_negatives_<rule>.yaml): the
# hold check, then per seed a negatives base and its stage B (one full run), its panel and gate, and a
# plain-GPM control (full run); then the probes, a preliminary summary, the negatives runs' swaps and the
# final summary. Orchestration only. Run it only when no other GPU job is running.
#
# Usage:    bash scripts/queue_instruction_negatives_pair.sh [--rule NAME] [--from-step K] [--smoke DIR]
#   --rule NAME    the rule (default: wine)
#   --from-step K  resume: steps < K are not rerun; a step is reused when its expected output is complete
#                  (result.json, written last by the runner; panel.json; probe.json; swaps.json). A run
#                  directory left by an interrupted step must be moved aside first. K > 0 runs the
#                  resume preflight (pins, clean tree, idle GPU, disk); the summary checks every reused
#                  output.
#   --smoke DIR    the first seed only, 200 steps, 1 episode per cell, 3 probe batches, into DIR (outside
#                  results/), on a dirty tree; the gate is not applied; the summaries run with --smoke.
#
#    0      preflight (K = 0) / resume preflight (K > 0)
#    1      hold check (no policy); a failure skips 2-11 and 13
#    per seed i = 0, 1, 2 (steps 2+3i .. 4+3i):
#      +0   neg   full run (the rule's negatives method), --no-t1-check   (~2 h 10)
#      +1   panel and gate on its stage0.pt (needs +0)                  (~40 min)
#      +2   ctrl  full run (plain GPM), --no-t1-check                    (~1 h 45)
#    11     probes (base, neg stage 1, ctrl stage 1 per seed; a missing checkpoint is skipped)
#    12     summary, preliminary (ALWAYS runs)   -> <out>/summary_preliminary.json
#    13     swaps of each neg run (needs its run)
#    14     summary, final (ALWAYS runs)         -> <out>/summary.json
#
# Status: every non-zero rc and every missing prior output is collected; the last line is
# "QUEUE DONE ok" (exit 0) or "QUEUE DONE FAILED: <steps>" (exit 1).
#
# Launch:   tmux new-session -d -s instrnegwine "bash scripts/queue_instruction_negatives_pair.sh --rule wine; exec bash"
# Testing:  QUEUE_DRY_RUN=1, QUEUE_FAIL_STEP=<step name>, QUEUE_LOG_ROOT, QUEUE_RESULTS_ROOT.

set -u
USAGE="usage: queue_instruction_negatives_pair.sh [--rule NAME] [--from-step K] [--smoke DIR]"
RULE=wine
FROM=0
SMOKE=""
while [ $# -gt 0 ]; do
    case "$1" in
        --rule) RULE="${2:-}"; shift 2 || { echo "$USAGE" >&2; exit 2; } ;;
        --from-step) FROM="${2:-}"; shift 2 || { echo "$USAGE" >&2; exit 2; } ;;
        --smoke) SMOKE="${2:-}"; shift 2 || { echo "$USAGE" >&2; exit 2; } ;;
        *) echo "unknown argument '$1'; $USAGE" >&2; exit 2 ;;
    esac
done
case "$FROM" in ''|*[!0-9]*) echo "--from-step must be an integer 0..14, got '$FROM'" >&2; exit 2;; esac
if [ "$FROM" -gt 14 ]; then echo "--from-step must be an integer 0..14, got '$FROM'" >&2; exit 2; fi
case "$RULE" in ''|*[!a-z0-9_]*) echo "--rule must be a lower-case name, got '$RULE'" >&2; exit 2;; esac

cd "$(dirname "$0")/.." || { echo "cannot cd to repo root" >&2; exit 1; }
export MUJOCO_GL=egl
PY="uv run python"
CLI=(scripts/instruction_negatives_pair.py --rule "$RULE")
RULE_FILE="configs/analysis/instruction_negatives_${RULE}.yaml"
[ -f "$RULE_FILE" ] || { echo "no rule $RULE_FILE" >&2; exit 2; }

INFO=$($PY "${CLI[@]}" --part info 2>/dev/null) || { echo "cannot read the rule $RULE_FILE" >&2; exit 2; }
eval "$INFO"
read -r -a SEED_LIST <<< "$SEEDS"
if [ "${#SEED_LIST[@]}" -gt 3 ]; then echo "at most 3 seeds (steps 2-10)" >&2; exit 2; fi

RUN_EXTRA=()
CL_ARGS=()
SMOKE_ARGS=()
PROBE_ARGS=()
if [ -n "$SMOKE" ]; then
    case "$SMOKE" in /*) ;; *) echo "--smoke needs an absolute directory" >&2; exit 2 ;; esac
    case "$SMOKE/" in "$(pwd)/results/"*) echo "--smoke must be outside results/" >&2; exit 2 ;; esac
    RES="$SMOKE"
    mkdir -p "$RES"
    SEED_LIST=("${SEED_LIST[0]}")
    RUN_EXTRA+=(--results-root "$RES" --steps-per-task 200 --n-episodes 1 --allow-dirty)
    CL_ARGS+=(--results-root "$RES" --allow-dirty)
    SMOKE_ARGS+=(--smoke --seeds "${SEED_LIST[0]}")
    PROBE_ARGS+=(--smoke-batches 3)
    TAG="_smoke"
else
    RES="${QUEUE_RESULTS_ROOT:-results}"
    CL_ARGS+=(--results-root "$RES")
    TAG=""
fi

LOGDIR="${QUEUE_LOG_ROOT:-results/logs}/queue_$(date +%Y%m%d_%H%M%S)_instruction_negatives_${RULE}${TAG}"
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

note "QUEUE START instruction_negatives_${RULE} smoke='$SMOKE' seeds='${SEED_LIST[*]}' from_step=$FROM git $(git rev-parse HEAD) status '$(git status --porcelain | wc -l) changes' rule $(sha256sum "$RULE_FILE" | cut -c1-12) free '$(df -h --output=avail / | tail -1 | tr -d ' ')'"

PRE_OK=0
if [ "$FROM" -gt 0 ]; then
    step 0_preflight_resume $PY "${CLI[@]}" --part preflight --resume "${CL_ARGS[@]}" "${SMOKE_ARGS[@]}" && PRE_OK=1
else
    step 0_preflight $PY "${CLI[@]}" --part preflight "${CL_ARGS[@]}" "${SMOKE_ARGS[@]}" && PRE_OK=1
fi

HOLD_OK=0
declare -A OK=()
if [ "$PRE_OK" = 1 ]; then
    run_or_prior 1 hold "$RES/$OUT/hold/hold.json" $PY "${CLI[@]}" --part hold "${CL_ARGS[@]}" && HOLD_OK=1
    if [ "$HOLD_OK" = 1 ] && [ "${QUEUE_DRY_RUN:-0}" != "1" ] && [ "$FROM" -gt 1 ]; then
        # a reused hold record must have passed
        grep -q '"passed": true' "$RES/$OUT/hold/hold.json" 2>/dev/null || { HOLD_OK=0; FAILED="$FAILED 1_hold(failed)"; }
    fi
else
    note "SKIP 1-11, 13: the preflight failed (the summaries still run)"
fi

if [ "$HOLD_OK" = 1 ]; then
    i=0
    for N in "${SEED_LIST[@]}"; do
        k=$((2 + 3 * i))
        RUN_ARGS=(--curriculum "$CURRICULUM" --seed "$N" --amp "${RUN_EXTRA[@]}")
        NEG_RUN="${CURRICULUM}__${NEG_DISPLAY}__seed$N"
        CTRL_RUN="${CURRICULUM}__${CTRL_DISPLAY}__seed$N"
        OK[neg_$N]=0; OK[ctrl_$N]=0
        run_or_prior "$k" "neg_s$N" "$RES/$NEG_RUN/result.json" \
            $PY scripts/run_continual.py "${RUN_ARGS[@]}" --method "$NEG_METHOD" --no-t1-check && OK[neg_$N]=1
        if [ "${OK[neg_$N]}" = 1 ]; then
            run_or_prior $((k + 1)) "panel_s$N" "$RES/$OUT/seed$N/panel/panel.json" \
                $PY "${CLI[@]}" --part panel --seed "$N" "${CL_ARGS[@]}" "${SMOKE_ARGS[@]}"
        else
            note "SKIP $((k + 1)) panel_s$N: ${k}_neg_s$N failed or is missing (its probes and swaps are skipped too)"
        fi
        run_or_prior $((k + 2)) "ctrl_s$N" "$RES/$CTRL_RUN/result.json" \
            $PY scripts/run_continual.py "${RUN_ARGS[@]}" --method "$CTRL_METHOD" --no-t1-check && OK[ctrl_$N]=1
        i=$((i + 1))
    done
    run_or_prior 11 probes "$RES/$OUT/seed${SEED_LIST[0]}/probes" \
        $PY "${CLI[@]}" --part probe "${CL_ARGS[@]}" "${SMOKE_ARGS[@]}" "${PROBE_ARGS[@]}"
elif [ "$PRE_OK" = 1 ]; then
    note "SKIP 2-11, 13: the hold check failed or is missing (the summaries still run)"
fi

step 12_summary_preliminary $PY "${CLI[@]}" --part summary "${CL_ARGS[@]}" "${SMOKE_ARGS[@]}"

if [ "$HOLD_OK" = 1 ]; then
    for N in "${SEED_LIST[@]}"; do
        if [ "${OK[neg_$N]}" = 1 ]; then
            run_or_prior 13 "swaps_s$N" "$RES/$OUT/seed$N/swaps/gpm/swaps.json" \
                $PY "${CLI[@]}" --part swaps --seed "$N" "${CL_ARGS[@]}"
        else
            note "SKIP 13 swaps_s$N: the neg run failed or is missing"
        fi
    done
fi

step 14_summary_final $PY "${CLI[@]}" --part summary --final "${CL_ARGS[@]}" "${SMOKE_ARGS[@]}"

if [ -z "$FAILED" ]; then
    note "QUEUE DONE ok"
    exit 0
fi
note "QUEUE DONE FAILED:$FAILED"
exit 1
