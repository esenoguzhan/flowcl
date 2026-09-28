#!/usr/bin/env bash
# Broader optimizer study: the fine-tuning stability-plasticity map on T1 -> T2 (seeds 0-2).
# Orchestration only; the rule is configs/analysis/optimizer_study.yaml. Run it only when no
# other GPU job is running. No gates: this is a map, not a gated follow-up.
#
# Usage:    bash scripts/queue_optimizer_study.sh [--from-step K]
#   --from-step K  resume: steps < K are not rerun (status from their done_* marker).
#                  Runs publish atomically, so a rerun step reuses its published runs and
#                  retrains only what is missing.
#
#   0-2  the core grid, seeds 1, 2, 0 (13 new runs, ~7.4 h)
#   3-4  the one-factor variants, seeds 0, 1 (10 runs, ~4.3 h)
#   5    the summary: per-batch probe units for every run, frontiers, existence, effects
#
# Launch:   tmux new-session -d -s optstudy "bash scripts/queue_optimizer_study.sh; exec bash"
# Testing:  QUEUE_DRY_RUN=1, QUEUE_FAIL_STEP=<step name>, QUEUE_LOG_ROOT, QUEUE_RESULTS_ROOT.

set -u
USAGE="usage: queue_optimizer_study.sh [--from-step K]"
FROM=0
while [ $# -gt 0 ]; do
    case "$1" in
        --from-step) FROM="${2:-}"; shift 2 || { echo "$USAGE" >&2; exit 2; } ;;
        *) echo "unknown argument '$1'; $USAGE" >&2; exit 2 ;;
    esac
done
case "$FROM" in ''|*[!0-9]*) echo "--from-step must be an integer 0..5, got '$FROM'" >&2; exit 2;; esac
if [ "$FROM" -gt 5 ]; then echo "--from-step must be an integer 0..5, got '$FROM'" >&2; exit 2; fi

cd "$(dirname "$0")/.." || { echo "cannot cd to repo root" >&2; exit 1; }
export MUJOCO_GL=egl

LOGDIR="${QUEUE_LOG_ROOT:-results/logs}/queue_$(date +%Y%m%d_%H%M%S)_optimizer_study"
mkdir -p "$LOGDIR"
QUEUE_LOG="$LOGDIR/queue.log"
RES="${QUEUE_RESULTS_ROOT:-results}"
OUT="$RES/optimizer_study"

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

PY="uv run python scripts/optimizer_study.py"
note "QUEUE START optimizer study from_step=$FROM git $(git rev-parse HEAD) status '$(git status --porcelain | wc -l) changes'"

all_ok=1
i=0
for spec in "grid 1" "grid 2" "grid 0" "variants 0" "variants 1"; do
    set -- $spec
    part=$1; seed=$2
    if want $i; then
        # shellcheck disable=SC2086
        step "${i}_${part}_s$seed" $PY --part "$part" --seed "$seed" || all_ok=0
    else
        prior "${i}_${part}_s$seed" "$OUT/done_${part}_seed$seed.json" || all_ok=0
    fi
    i=$((i + 1))
done

if [ $all_ok = 1 ]; then
    # shellcheck disable=SC2086
    step 5_summary $PY --summary
else
    note "SKIP 5_summary: a grid or variant step failed or is missing (rerun with --from-step)"
fi

note "QUEUE DONE"
