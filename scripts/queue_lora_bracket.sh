#!/usr/bin/env bash
# LoRA rank × learning-rate bracket (Workstream B): the nine pre-registered arms, the selection (with up
# to two edge extensions when the grid's selection lies on an edge), and a final verification.
# Orchestration only; the rule is configs/analysis/baseline_selection_lora.yaml, committed before any arm.
# Run it only when no other GPU job is running.
#
# Usage:    bash scripts/queue_lora_bracket.sh [--from-step K] [--deadline HH:MM|@EPOCH]
#                                              [--arm-estimate-min M] [--extension-estimate-min M]
#                                              [--smoke DIR]
#   --from-step K   resume: arm steps < K are not rerun (status from their published rows); K in 0..9
#                   (9: every arm is done, go straight to the selection).
#   --deadline T    the GPU window's end. Before each arm or extension, if now + its estimate (the longest
#                   arm this queue has run, or --arm-estimate-min / --extension-estimate-min if longer)
#                   passes T, the queue stops between arms: finished rows are kept, nothing is selected or
#                   published, and it ends "QUEUE DONE DEFERRED: <steps>" (exit 3). Resume with --from-step.
#   --smoke DIR     the smoke pipeline into DIR (outside results/), on a dirty tree: 4-task seq_ft and LoRA
#                   (rank 4, lr 1e-3) at 200 steps and 1 episode, LoRA identity-checked against seq_ft at
#                   stage 0; LoRA resumed at stage 3 and compared with compare_resume (tensors, per-episode
#                   evaluations, method history); the bracket path against the runner (an arm from the smoke
#                   seq_ft stage 0 must reproduce the smoke LoRA run's stage 1 under the same TrainConfig);
#                   and a step-time benchmark at ranks 0 (full fine-tuning), 4, 64 and 256.
#
#   0-8   arm (rank, lr), rank-major          -> results/baseline_selection_lora/lora/seed0/lorab__r<R>__lr<LR>__seed0
#   9     select (rc 10 = an edge extension is owed: not a failure)
#   10_k  extension k (k = 1, 2; only while select returns 10)
#   11_k  select again
#   12    verify: inputs, checkpoints, the bundle and the recomputed selection
#
# Status: the last line is "QUEUE DONE ok" (exit 0), "QUEUE DONE DEFERRED: <steps>" (exit 3) or
# "QUEUE DONE FAILED: <steps>" (exit 1).
# Launch:   tmux new-session -d -s lorab "bash scripts/queue_lora_bracket.sh --deadline 15:00 ...; exec bash"
# Testing:  QUEUE_DRY_RUN=1, QUEUE_FAIL_STEP=<step name>, QUEUE_STEP_RC=<step>=<rc> (several separated by
#           commas), QUEUE_LOG_ROOT, QUEUE_RESULTS_ROOT, QUEUE_FAKE_NOW=<epoch> with QUEUE_FAKE_STEP_S=<s>
#           (a fake clock that each step advances).

set -u
USAGE="usage: queue_lora_bracket.sh [--from-step K] [--deadline HH:MM|@EPOCH] [--arm-estimate-min M] [--extension-estimate-min M] [--smoke DIR]"
FROM=0
SMOKE=""
DEADLINE=""
ARM_EST_MIN=0
EXT_EST_MIN=0
while [ $# -gt 0 ]; do
    case "$1" in
        --from-step) FROM="${2:-}"; shift 2 || { echo "$USAGE" >&2; exit 2; } ;;
        --smoke) SMOKE="${2:-}"; shift 2 || { echo "$USAGE" >&2; exit 2; } ;;
        --deadline) DEADLINE="${2:-}"; shift 2 || { echo "$USAGE" >&2; exit 2; } ;;
        --arm-estimate-min) ARM_EST_MIN="${2:-}"; shift 2 || { echo "$USAGE" >&2; exit 2; } ;;
        --extension-estimate-min) EXT_EST_MIN="${2:-}"; shift 2 || { echo "$USAGE" >&2; exit 2; } ;;
        *) echo "unknown argument '$1'; $USAGE" >&2; exit 2 ;;
    esac
done
case "$FROM" in ''|*[!0-9]*) echo "--from-step must be an integer 0..9, got '$FROM'" >&2; exit 2;; esac
if [ "$FROM" -gt 9 ]; then echo "--from-step must be an integer 0..9, got '$FROM'" >&2; exit 2; fi
for v in "$ARM_EST_MIN" "$EXT_EST_MIN"; do
    case "$v" in ''|*[!0-9]*) echo "estimates must be whole minutes, got '$v'" >&2; exit 2;; esac
done

FAKE_CLOCK="${QUEUE_FAKE_NOW:-}"
clock() { if [ -n "$FAKE_CLOCK" ]; then echo "$FAKE_CLOCK"; else date +%s; fi; }

DEADLINE_TS=""
if [ -n "$DEADLINE" ]; then
    case "$DEADLINE" in
        @*) DEADLINE_TS="${DEADLINE#@}"
            case "$DEADLINE_TS" in ''|*[!0-9]*) echo "bad --deadline '$DEADLINE'" >&2; exit 2;; esac ;;
        [0-2][0-9]:[0-5][0-9])
            DEADLINE_TS=$(date -d "$DEADLINE" +%s 2>/dev/null) || { echo "bad --deadline '$DEADLINE'" >&2; exit 2; }
            if [ "$DEADLINE_TS" -le "$(clock)" ]; then DEADLINE_TS=$((DEADLINE_TS + 86400)); fi ;;
        *) echo "bad --deadline '$DEADLINE' (HH:MM or @EPOCH)" >&2; exit 2 ;;
    esac
fi

cd "$(dirname "$0")/.." || { echo "cannot cd to repo root" >&2; exit 1; }
export MUJOCO_GL=egl

RES="${QUEUE_RESULTS_ROOT:-results}"
RULE=configs/analysis/baseline_selection_lora.yaml
ARMS="4:0.0001 4:0.0003 4:0.001 16:0.0001 16:0.0003 16:0.001 64:0.0001 64:0.0003 64:0.001"   # the rule's grid
if [ -n "$SMOKE" ]; then
    case "$SMOKE" in /*) ;; *) echo "--smoke needs an absolute directory" >&2; exit 2 ;; esac
    case "$SMOKE/" in "$(pwd)/results/"*) echo "--smoke must be outside results/" >&2; exit 2 ;; esac
    mkdir -p "$SMOKE"
    TAG="_smoke"
else
    TAG=""
fi

LOGDIR="${QUEUE_LOG_ROOT:-results/logs}/queue_$(date +%Y%m%d_%H%M%S)_lora_bracket${TAG}"
mkdir -p "$LOGDIR"
QUEUE_LOG="$LOGDIR/queue.log"
FAILED=""
DEFERRED=""
LONGEST=0

note() { echo "$(date '+%F %T') $*" | tee -a "$QUEUE_LOG"; }

step_rc_override() {   # QUEUE_STEP_RC="a=10,b=1"
    local name=$1 item
    IFS=',' read -ra items <<< "${QUEUE_STEP_RC:-}"
    for item in "${items[@]}"; do
        case "$item" in "$name="*) echo "${item#*=}"; return 0 ;; esac
    done
    return 1
}

step() {
    local name=$1; shift
    note "START $name: $*"
    local rc started
    started=$(clock)
    if [ "${QUEUE_DRY_RUN:-0}" = "1" ]; then
        echo "$*" > "$LOGDIR/$name.log"
        rc=0
        if [ "${QUEUE_FAIL_STEP:-}" = "$name" ]; then rc=1; fi
        if override=$(step_rc_override "$name"); then rc=$override; fi
        if [ -n "$FAKE_CLOCK" ]; then FAKE_CLOCK=$((FAKE_CLOCK + ${QUEUE_FAKE_STEP_S:-0})); fi
    else
        "$@" > "$LOGDIR/$name.log" 2>&1
        rc=$?
    fi
    LAST_DURATION=$(( $(clock) - started ))
    note "END $name rc=$rc"
    return $rc
}

fail() { FAILED="$FAILED $1"; }

fits() {   # fits <name> <estimate minutes>: may an arm start before the deadline?
    [ -z "$DEADLINE_TS" ] && return 0
    local est=$(( $2 * 60 ))
    if [ "$LONGEST" -gt "$est" ]; then est=$LONGEST; fi
    local now; now=$(clock)
    if [ $((now + est)) -le "$DEADLINE_TS" ]; then return 0; fi
    note "DEFERRED $1: now $(date -d @"$now" '+%F %T') + estimate $((est / 60)) min passes the deadline $(date -d @"$DEADLINE_TS" '+%F %T')"
    DEFERRED="$DEFERRED $1"
    return 1
}

PY="uv run python"
LS=(scripts/lora_selection.py --config "$RULE")
note "QUEUE START lora_bracket smoke='$SMOKE' from_step=$FROM deadline='${DEADLINE}' arm_estimate=${ARM_EST_MIN}min extension_estimate=${EXT_EST_MIN}min git $(git rev-parse HEAD) status '$(git status --porcelain | wc -l) changes'"

if [ -n "$SMOKE" ]; then
    CL=(--curriculum seq_hetero --seed 0 --amp --results-root "$SMOKE" --steps-per-task 200 --n-episodes 1 --allow-dirty)
    REF="$SMOKE/seq_hetero__seq_ft__seed0"
    RUN="$SMOKE/seq_hetero__lora__seed0"
    step s0_smoke_method $PY "${LS[@]}" --write-smoke-method "$SMOKE/lora_smoke.yaml" --rank 4 --lr 0.001 || fail s0_smoke_method
    step s1_seqft $PY scripts/run_continual.py "${CL[@]}" --method seq_ft || fail s1_seqft
    step s2_lora $PY scripts/run_continual.py "${CL[@]}" --method "$SMOKE/lora_smoke.yaml" \
        --t1-reference-run "$REF" --identity-reference-run "$REF" --identity-stages 0 || fail s2_lora
    RCL=(--curriculum seq_hetero --seed 0 --amp --results-root "$SMOKE/resumed" --steps-per-task 200 --n-episodes 1 --allow-dirty)
    step s3_lora_resume $PY scripts/run_continual.py "${RCL[@]}" --method "$SMOKE/lora_smoke.yaml" \
        --resume-run "$RUN" --start-stage 3 || fail s3_lora_resume
    step s4_lora_compare $PY scripts/compare_resume.py --source "$RUN" \
        --resumed "$SMOKE/resumed/seq_hetero__lora__seed0" --stage 3 --all-artifacts --method-state \
        --out "$SMOKE/resume_lora.json" || fail s4_lora_compare
    step s5_bracket_identity $PY "${LS[@]}" --smoke-arm 4 0.001 --reference-run "$REF" --identity-run "$RUN" \
        --smoke-steps 200 --out-root "$SMOKE/bracket" --allow-dirty || fail s5_bracket_identity
    step s6_benchmark $PY "${LS[@]}" --benchmark --bench-ranks 0 4 64 256 --bench-steps 300 \
        --out-root "$SMOKE/benchmark" || fail s6_benchmark
else
    K=0
    STOP=0
    for CELL in $ARMS; do
        R="${CELL%%:*}"; LR="${CELL#*:}"
        NAME="${K}_arm_r${R}_lr${LR}"
        ROW="$RES/baseline_selection_lora/lora/seed0/lorab__r${R}__lr${LR}__seed0/row.json"   # LR is written as its %g tag
        if [ "$K" -lt "$FROM" ]; then
            if [ -e "$ROW" ]; then note "PRIOR $NAME ok: $ROW"; else note "PRIOR $NAME missing: $ROW"; fail "$NAME(missing)"; fi
        elif [ "$STOP" = 1 ]; then
            DEFERRED="$DEFERRED $NAME"
        elif fits "$NAME" "$ARM_EST_MIN"; then
            if step "$NAME" $PY "${LS[@]}" --arm "$R" "$LR"; then
                if [ "$LAST_DURATION" -gt "$LONGEST" ]; then LONGEST=$LAST_DURATION; fi
            else
                fail "$NAME"
            fi
        else
            STOP=1
        fi
        K=$((K + 1))
    done
    ARM_FAILED=0
    case "$FAILED" in *_arm_*) ARM_FAILED=1 ;; esac
    if [ -n "$DEFERRED" ]; then
        note "SKIP 9-12: arms deferred past the deadline"
    elif [ $ARM_FAILED = 1 ]; then
        note "SKIP 9-12: an arm failed or is missing"
    else
        step 9_select $PY "${LS[@]}" --select
        rc=$?
        E=1
        while [ $rc = 10 ] && [ $E -le 2 ]; do
            note "edge extension owed (rc 10)"
            if ! fits "10_extension_$E" "$EXT_EST_MIN"; then rc=deferred; break; fi
            if step "10_extension_$E" $PY "${LS[@]}" --extension; then
                if [ "$LAST_DURATION" -gt "$LONGEST" ]; then LONGEST=$LAST_DURATION; fi
                step "11_select_$E" $PY "${LS[@]}" --select
                rc=$?
            else
                fail "10_extension_$E"; rc=failed; break
            fi
            E=$((E + 1))
        done
        case "$rc" in
            0) step 12_verify $PY "${LS[@]}" --verify || fail 12_verify ;;
            deferred) note "SKIP 12: an extension was deferred past the deadline" ;;
            failed) note "SKIP 12: an extension failed" ;;
            10) fail "select(extensions still owed after 2)" ;;
            *) fail "select(rc $rc)" ;;
        esac
    fi
fi

if [ -n "$FAILED" ]; then
    note "QUEUE DONE FAILED:$FAILED"
    exit 1
fi
if [ -n "$DEFERRED" ]; then
    note "QUEUE DONE DEFERRED:$DEFERRED"
    exit 3
fi
note "QUEUE DONE ok"
exit 0
