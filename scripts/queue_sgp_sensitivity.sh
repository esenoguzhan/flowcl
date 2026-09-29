#!/usr/bin/env bash
# SGP sensitivity on the standard recipe: bracket on seed 0, select and freeze, validate on
# seeds 1-2, summarise. Orchestration only; the rule is configs/analysis/sgp_sensitivity.yaml.
# Run it only when no other GPU job is running. No step launches four-task SGP.
#
# Usage:    bash scripts/queue_sgp_sensitivity.sh [--from-step K]
#   --from-step K  resume: steps < K are not rerun (status from their output on disk). Runs and
#                  the selection bundle publish atomically; published ones are re-verified and
#                  reused, abandoned attempt directories are ignored.
#
#   0  bracket, seed 0: SGP alpha 100 / 400 / 1600 and freeze-only (~4.2 h)
#   1  select and freeze (atomic selection bundle)
#   2  select_gate: was an alpha selected?          no -> 3-4 are skipped
#   3-4 validation, seeds 1 and 2 (~2.1 h)
#   5  summary and verdict: ALWAYS runs once a selection bundle exists
#
# Launch:   tmux new-session -d -s sgpsens "bash scripts/queue_sgp_sensitivity.sh; exec bash"
# Testing:  QUEUE_DRY_RUN=1, QUEUE_FAIL_STEP=<step name>, QUEUE_LOG_ROOT, QUEUE_RESULTS_ROOT.

set -u
USAGE="usage: queue_sgp_sensitivity.sh [--from-step K]"
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

LOGDIR="${QUEUE_LOG_ROOT:-results/logs}/queue_$(date +%Y%m%d_%H%M%S)_sgp_sensitivity"
mkdir -p "$LOGDIR"
QUEUE_LOG="$LOGDIR/queue.log"
RES="${QUEUE_RESULTS_ROOT:-results}"
OUT="$RES/sgp_sensitivity"

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

PY="uv run python scripts/sgp_sensitivity.py"
note "QUEUE START sgp sensitivity from_step=$FROM git $(git rev-parse HEAD) status '$(git status --porcelain | wc -l) changes'"

# shellcheck disable=SC2086
{
bracket_ok=0
if want 0; then step 0_bracket $PY --part bracket && bracket_ok=1
else prior 0_bracket "$OUT/done_bracket_seed0.json" && bracket_ok=1; fi

bundle_ok=0
if [ $bracket_ok = 1 ]; then
    if want 1; then step 1_select $PY --part select && bundle_ok=1
    else prior 1_select "$OUT/selection/selection.done.json" && bundle_ok=1; fi
else
    note "SKIP 1-5: the bracket failed or is missing"
fi

if [ $bundle_ok = 1 ]; then
    gate_ok=0
    step 2_select_gate $PY --part select_gate && gate_ok=1
    if [ $gate_ok = 1 ]; then
        i=3
        for s in 1 2; do
            if want $i; then step "${i}_validate_s$s" $PY --part validate --seed $s
            else prior "${i}_validate_s$s" "$OUT/validation_seed$s.json"; fi
            i=$((i + 1))
        done
    else
        note "SKIP 3-4: no qualifying alpha (verdict no_qualifying_alpha); the summary still runs"
    fi
    step 5_summary $PY --part summary
elif [ $bracket_ok = 1 ]; then
    note "SKIP 2-5: no selection bundle was published"
fi
}

note "QUEUE DONE"
