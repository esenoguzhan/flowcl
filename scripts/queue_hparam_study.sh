#!/usr/bin/env bash
# Exploratory training-budget and learning-rate study (single seed), one part per step.
# Orchestration only; the study is flowcl/experiments/hparam_study.py and everything it
# reports is fixed in configs/analysis/hparam_study.yaml. Run it only when no other GPU job
# is running (a second process can perturb cuDNN and break other runs' identity checks).
#
# Usage:    bash scripts/queue_hparam_study.sh [--from-step K]
#   --from-step K  resume after an interruption: steps < K are NOT rerun; their status is
#                  taken from their report on disk ("PRIOR <step> ok/missing"). A run
#                  directory left by an interrupted step must be moved aside first.
#
#   0. trajectory, Object      (~1.7 h)
#   1. trajectory, LIBERO-10   (~1.7 h)
#   2. cl_budget: Object after Spatial for 2.5k-20k steps (~2.5 h)
#   3. cl_lr: learning rates at B* (needs step 2's report; ~0.5-2 h)
#
# Never stops early: no `set -e`; each step's exit code is in
# results/logs/queue_<stamp>_hparam/queue.log, one log per step next to it.
#
# Launch:   tmux new-session -d -s hparam "bash scripts/queue_hparam_study.sh; exec bash"
# Testing:  QUEUE_DRY_RUN=1, QUEUE_FAIL_STEP=<step name>, QUEUE_LOG_ROOT, QUEUE_RESULTS_ROOT
#           (as in queue_seed_pair.sh).

set -u
USAGE="usage: queue_hparam_study.sh [--from-step K]"
FROM=0
while [ $# -gt 0 ]; do
    case "$1" in
        --from-step) FROM="${2:-}"; shift 2 || { echo "$USAGE" >&2; exit 2; } ;;
        *) echo "unknown argument '$1'; $USAGE" >&2; exit 2 ;;
    esac
done
case "$FROM" in ''|*[!0-9]*) echo "--from-step must be an integer 0..3, got '$FROM'" >&2; exit 2;; esac
if [ "$FROM" -gt 3 ]; then echo "--from-step must be an integer 0..3, got '$FROM'" >&2; exit 2; fi

cd "$(dirname "$0")/.." || { echo "cannot cd to repo root" >&2; exit 1; }
export MUJOCO_GL=egl

LOGDIR="${QUEUE_LOG_ROOT:-results/logs}/queue_$(date +%Y%m%d_%H%M%S)_hparam"
mkdir -p "$LOGDIR"
QUEUE_LOG="$LOGDIR/queue.log"
RES="${QUEUE_RESULTS_ROOT:-results}"
OUT="$RES/hparam_study"

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

STUDY="uv run python scripts/hparam_study.py"
# Mirrors configs/analysis/hparam_study.yaml trajectory.tasks (checked by the tests).
TRAJ_SUITES="libero_object libero_10"
note "QUEUE START hparam study from_step=$FROM git $(git rev-parse HEAD) status '$(git status --porcelain | wc -l) changes'"

i=0
for s in $TRAJ_SUITES; do
    if want $i; then
        # shellcheck disable=SC2086
        step "${i}_trajectory_$s" $STUDY --part trajectory --task-index $i
    else
        prior "${i}_trajectory_$s" "$OUT/trajectory/traj__${s}__seed0/report.json"
    fi
    i=$((i + 1))
done

budget_ok=0
if want 2; then
    # shellcheck disable=SC2086
    step 2_cl_budget $STUDY --part cl_budget && budget_ok=1
else
    prior 2_cl_budget "$OUT/cl_budget/report.json" && budget_ok=1
fi

if [ $budget_ok = 1 ]; then
    # shellcheck disable=SC2086
    want 3 && step 3_cl_lr $STUDY --part cl_lr
else
    note "SKIP 3_cl_lr: the cl_budget report is missing (B* is read from it)"
fi

note "QUEUE DONE"
