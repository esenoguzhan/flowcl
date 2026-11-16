#!/usr/bin/env bash
# Adaptive GPM at f = 0.7 (gpm_ne70) on seq_hetero for seeds 0, 1, 2, before E3 (pre-E3 plan, A3).
# E3 resumes every arm from its 4-task run at stage 4, so all arms share seq_hetero's seed namespace;
# a from-scratch run on seq_hetero_t8 would not. So f = 0.7 runs here as a 4-task run first, paired
# with the existing runs of each seed. Orchestration only; the checks are flowcl/experiments/ne70_queue.py.
# Run it only when no other GPU job is running (a second process could perturb cuDNN and break the
# bitwise identity the checks rely on).
#
# Usage:    bash scripts/queue_ne70.sh [--from-step K] [--smoke DIR --smoke-refs REFS]
#   --from-step K   resume: steps < K are not rerun (status from their outputs on disk; a missing
#                   output blocks its dependents). A run directory left by an interrupted training step
#                   must be moved aside first.
#   --smoke DIR     seed 2 only, 200 steps and 1 episode per cell, into DIR (outside results/), on a
#                   dirty tree, against the smoke references in REFS (a queue_baselines.sh --smoke
#                   root: seq_ft and plain GPM, seed 2).
#
#   0      preflight       (clean tree, idle GPU, free disk, targets absent, config, references by SHA-256)
#   per seed S (k = 1 + 3 * index of S):
#   k+0    ne70 run        (~4.8 h; T1 pairing against seq_ft, identity against plain GPM at stages 0-1)
#   k+1    check           (identity {0, 1} passed, T1 pairing passed, Task-1 memory equal to plain GPM's)
#   k+2    strict report   -> gpm_seq_ne70_seed<S>/report.json
#
# IDENTITY STOP: when a run fails and its identity_stage0/1.json reads "passed": false, every later
# seed is skipped (logged STOP); the same failure would recur. Any other failure skips only that
# seed's dependents.
#
# Status: every non-zero rc, skipped seed and missing prior output is collected; the last line is
# "QUEUE DONE ok" (exit 0) or "QUEUE DONE FAILED: <steps>" (exit 1).
#
# Launch:   tmux new-session -d -s ne70 "bash scripts/queue_ne70.sh; exec bash"
# Testing:  QUEUE_DRY_RUN=1, QUEUE_FAIL_STEP=<step name>, QUEUE_IDENTITY_FAIL=<step name>,
#           QUEUE_LOG_ROOT, QUEUE_RESULTS_ROOT.

set -u
USAGE="usage: queue_ne70.sh [--from-step K] [--smoke DIR --smoke-refs REFS]"
FROM=0
SMOKE=""
REFS=""
while [ $# -gt 0 ]; do
    case "$1" in
        --from-step) FROM="${2:-}"; shift 2 || { echo "$USAGE" >&2; exit 2; } ;;
        --smoke) SMOKE="${2:-}"; shift 2 || { echo "$USAGE" >&2; exit 2; } ;;
        --smoke-refs) REFS="${2:-}"; shift 2 || { echo "$USAGE" >&2; exit 2; } ;;
        *) echo "unknown argument '$1'; $USAGE" >&2; exit 2 ;;
    esac
done
case "$FROM" in ''|*[!0-9]*) echo "--from-step must be an integer 0..9, got '$FROM'" >&2; exit 2;; esac
if [ "$FROM" -gt 9 ]; then echo "--from-step must be an integer 0..9, got '$FROM'" >&2; exit 2; fi

cd "$(dirname "$0")/.." || { echo "cannot cd to repo root" >&2; exit 1; }
export MUJOCO_GL=egl

CUR=seq_hetero
RUN_ARGS=(--curriculum "$CUR" --amp --method gpm_ne70)
REPORT_ARGS=(--strict-provenance)
PRE_ARGS=()
if [ -n "$SMOKE" ]; then
    case "$SMOKE" in /*) ;; *) echo "--smoke needs an absolute directory" >&2; exit 2 ;; esac
    case "$SMOKE/" in "$(pwd)/results/"*) echo "--smoke must be outside results/" >&2; exit 2 ;; esac
    case "$REFS" in /*) ;; *) echo "--smoke needs --smoke-refs (an absolute directory)" >&2; exit 2 ;; esac
    RES="$SMOKE"
    mkdir -p "$RES"
    RUN_ARGS+=(--results-root "$RES" --steps-per-task 200 --n-episodes 1 --allow-dirty)
    REPORT_ARGS+=(--allow-dirty)
    PRE_ARGS+=(--smoke --results-root "$RES" --refs "$REFS")
    SEEDS="2"
    TAG="_smoke"
else
    if [ -n "$REFS" ]; then echo "--smoke-refs goes with --smoke" >&2; exit 2; fi
    RES="${QUEUE_RESULTS_ROOT:-results}"
    REFS="$RES"
    SEEDS="0 1 2"
    TAG=""
fi

LOGDIR="${QUEUE_LOG_ROOT:-results/logs}/queue_$(date +%Y%m%d_%H%M%S)_ne70${TAG}"
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

# identity_failed STEP RUN_DIR: an identity file of the run reads "passed": false.
identity_failed() {
    local f
    if [ "${QUEUE_DRY_RUN:-0}" = "1" ]; then
        [ "${QUEUE_IDENTITY_FAIL:-}" = "$1" ] && { note "IDENTITY FAILED (dry run): $1"; return 0; }
        return 1
    fi
    for f in "$2/identity_stage0.json" "$2/identity_stage1.json"; do
        if [ -e "$f" ] && grep -q '"passed": false' "$f"; then
            note "IDENTITY FAILED: $f"
            return 0
        fi
    done
    return 1
}

PY="uv run python"
note "QUEUE START ne70 smoke='$SMOKE' refs='$REFS' from_step=$FROM git $(git rev-parse HEAD) status '$(git status --porcelain | wc -l) changes' free '$(df -h --output=avail / | tail -1 | tr -d ' ')'"

if want 0; then
    # shellcheck disable=SC2086
    step 0_preflight $PY scripts/ne70_queue.py --preflight --seeds $SEEDS "${PRE_ARGS[@]}" || {
        note "STOP: the preflight refused the launch"
        note "QUEUE DONE FAILED:$FAILED"
        exit 1
    }
fi

STOP=0
INDEX=0
for S in $SEEDS; do
    K=$((1 + 3 * INDEX)); INDEX=$((INDEX + 1))
    REF=$REFS/${CUR}__seq_ft__seed$S
    GPM=$REFS/${CUR}__gpm_projected_adam__seed$S
    NE70=$RES/${CUR}__gpm_projected_adam_ne70__seed$S
    OUT=$RES/gpm_seq_ne70_seed$S/report.json
    if [ $STOP = 1 ]; then
        note "SKIP seed $S (steps $K-$((K + 2))): identity stop"
        FAILED="$FAILED seed${S}(identity_stop)"
        continue
    fi
    run_ok=0; check_ok=0
    run_or_prior $K "run_s$S" "$NE70/result.json" \
        $PY scripts/run_continual.py "${RUN_ARGS[@]}" --seed "$S" \
        --t1-reference-run "$REF" --identity-reference-run "$GPM" --identity-stages 0 1 \
        && run_ok=1
    if [ $run_ok = 0 ]; then
        if want $K && identity_failed "${K}_run_s$S" "$NE70"; then
            STOP=1
            note "STOP identity failure at seed $S: every later seed is skipped"
        fi
        note "SKIP $((K + 1))-$((K + 2)): ${K}_run_s$S failed or is missing"
        continue
    fi
    if want $((K + 1)); then
        step "$((K + 1))_check_s$S" $PY scripts/ne70_queue.py --check-run "$NE70" --gpm-run "$GPM" && check_ok=1
    else
        check_ok=1; note "PRIOR $((K + 1))_check_s$S: not rerun (before --from-step)"
    fi
    if [ $check_ok = 0 ]; then
        note "SKIP $((K + 2)): $((K + 1))_check_s$S failed"
        continue
    fi
    run_or_prior $((K + 2)) "report_s$S" "$OUT" \
        $PY scripts/sequence_report.py --method-run "$NE70" --reference-run "$REF" --out "$OUT" \
        "${REPORT_ARGS[@]}"
done

if [ -z "$FAILED" ]; then
    note "QUEUE DONE ok"
    exit 0
fi
note "QUEUE DONE FAILED:$FAILED"
exit 1
