#!/usr/bin/env bash
# SGP update-rule coupling test at T4 (alpha 800, seed 0; exploratory): the reproduction gate,
# rate calibration and pilots, the frozen selection, five more arms, the summary.
# Orchestration only; the rule is configs/analysis/sgp_coupling.yaml. Run it only when no
# other GPU job is running. No step launches a four-task run.
#
# Usage:    bash scripts/queue_sgp_coupling.sh [--from-step K]
#   --from-step K  resume: steps < K are not rerun (status from their output on disk). Arms,
#                  pilots and the selection bundle publish atomically; published ones are
#                  re-verified and reused, abandoned attempt directories are ignored. The gate
#                  steps (1, 13, 15) always run.
#
#   0      arm aw_soft = A800's T4 reproduced (~70 min)
#   1      repro_gate: did aw_soft reproduce A800 exactly?     no -> 2-17 skipped
#   2-3    AdamW pilots aw_free, aw_hard (3k steps, step norms recorded; ~15 min)
#   4-9    SGD pilots sgd_free, sgd_hard at 0.5 / 1 / 2 x the matched rate (~35 min)
#   10     select: rates frozen in an atomic bundle
#   11-12  arms aw_hard, aw_free (~2.3 h)
#   13     rate_gate free                                        no -> 14 skipped
#   14     arm sgd_free (~70 min)
#   15     rate_gate proj                                        no -> 16-17 skipped
#   16-17  arms sgd_hard, sgd_soft (~2.3 h)
#   18     summary and verdict: ALWAYS runs
#
# Launch:   tmux new-session -d -s sgpcoup "bash scripts/queue_sgp_coupling.sh; exec bash"
# Testing:  QUEUE_DRY_RUN=1, QUEUE_FAIL_STEP=<step name>, QUEUE_LOG_ROOT, QUEUE_RESULTS_ROOT.

set -u
USAGE="usage: queue_sgp_coupling.sh [--from-step K]"
LAST=18
FROM=0
while [ $# -gt 0 ]; do
    case "$1" in
        --from-step) FROM="${2:-}"; shift 2 || { echo "$USAGE" >&2; exit 2; } ;;
        *) echo "unknown argument '$1'; $USAGE" >&2; exit 2 ;;
    esac
done
case "$FROM" in ''|*[!0-9]*) echo "--from-step must be an integer 0..$LAST, got '$FROM'" >&2; exit 2;; esac
if [ "$FROM" -gt $LAST ]; then echo "--from-step must be an integer 0..$LAST, got '$FROM'" >&2; exit 2; fi

cd "$(dirname "$0")/.." || { echo "cannot cd to repo root" >&2; exit 1; }
export MUJOCO_GL=egl

LOGDIR="${QUEUE_LOG_ROOT:-results/logs}/queue_$(date +%Y%m%d_%H%M%S)_sgp_coupling"
mkdir -p "$LOGDIR"
QUEUE_LOG="$LOGDIR/queue.log"
RES="${QUEUE_RESULTS_ROOT:-results}"
OUT="$RES/sgp_coupling"

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
    return $rc
}

want() { [ "$1" -ge "$FROM" ]; }

prior() {
    local name=$1 path=$2
    if [ -e "$path" ]; then note "PRIOR $name ok: $path"; return 0; fi
    note "PRIOR $name missing: $path"
    return 1
}

# run_or_prior K NAME OUTPUT COMMAND...: run step K unless resuming past it.
run_or_prior() {
    local k=$1 name=$2 path=$3; shift 3
    if want "$k"; then step "${k}_$name" "$@"
    else prior "${k}_$name" "$path"; fi
}

PY="uv run python scripts/sgp_coupling.py"
arm() { echo "$OUT/arms/coup__$1__seed0/row.json"; }
pilot() { echo "$OUT/pilots/pilot__$1__seed0/row.json"; }
note "QUEUE START sgp coupling from_step=$FROM git $(git rev-parse HEAD) status '$(git status --porcelain | wc -l) changes'"

# shellcheck disable=SC2086
{
run_or_prior 0 arm_aw_soft "$(arm aw_soft)" $PY --part arm --arm aw_soft

gate_ok=0
step 1_repro_gate $PY --part repro_gate && gate_ok=1
if [ $gate_ok = 1 ]; then
    run_or_prior 2 pilot_aw_free "$(pilot aw_free)" $PY --part pilot --arm aw_free
    run_or_prior 3 pilot_aw_hard "$(pilot aw_hard)" $PY --part pilot --arm aw_hard
    i=4
    for a in sgd_free sgd_hard; do
        for s in 0.5 1 2; do
            run_or_prior $i "pilot_${a}_x$s" "$(pilot "${a}_x$s")" $PY --part pilot --arm $a --scale $s
            i=$((i + 1))
        done
    done
    run_or_prior 10 select "$OUT/selection/selection.done.json" $PY --part select
    run_or_prior 11 arm_aw_hard "$(arm aw_hard)" $PY --part arm --arm aw_hard
    run_or_prior 12 arm_aw_free "$(arm aw_free)" $PY --part arm --arm aw_free
    if step 13_rate_gate_free $PY --part rate_gate --rate free; then
        run_or_prior 14 arm_sgd_free "$(arm sgd_free)" $PY --part arm --arm sgd_free
    else
        note "SKIP 14: rate free not selected (or no selection bundle)"
    fi
    if step 15_rate_gate_proj $PY --part rate_gate --rate proj; then
        run_or_prior 16 arm_sgd_hard "$(arm sgd_hard)" $PY --part arm --arm sgd_hard
        run_or_prior 17 arm_sgd_soft "$(arm sgd_soft)" $PY --part arm --arm sgd_soft
    else
        note "SKIP 16-17: rate proj not selected (or no selection bundle)"
    fi
else
    note "SKIP 2-17: the reproduction gate is closed (verdict invalid_reproduction); the summary still runs"
fi
step 18_summary $PY --part summary
}

note "QUEUE DONE"
