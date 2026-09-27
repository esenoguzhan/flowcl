#!/usr/bin/env bash
# Low-update continual fine-tuning: held-out validation, the gated four-task chain, and the
# gated projection pilot. Orchestration only; the rules are configs/analysis/low_update.yaml
# and configs/analysis/projection_pilot.yaml. Run it only when no other GPU job is running.
#
# Usage:    bash scripts/queue_low_update.sh [--from-step K]
#   --from-step K  resume: steps < K are not rerun; their status comes from their output on
#                  disk ("PRIOR <step> ok/missing"). Stages publish atomically, so an
#                  interrupted stage leaves only an ignored .attempt_* directory; a published
#                  stage is re-validated and reused.
#
#   0-1  validation, seeds 1 and 2 (~20 min each)
#   2    the validation verdict (seed 0 reported, not counted)
#   3    gate: validated on both held-out seeds?          no -> everything below is skipped
#   4-6  the chain, seeds 0, 1, 2 (stages 2-3, ~55 min each)
#   7    the chain summary and classification
#   8    pilot gate: validated and chain not underlearns?  no -> 9-11 are skipped
#   9-10 the projection pilot at lr 3e-5 (7 runs) and 1e-4 (6 runs)
#   11   the pilot summary and selection
#
# No step launches SGP seeds 1-2 or four-task SGP. Never stops early (no `set -e`).
#
# Launch:   tmux new-session -d -s lowupd "bash scripts/queue_low_update.sh; exec bash"
# Testing:  QUEUE_DRY_RUN=1, QUEUE_FAIL_STEP=<step name>, QUEUE_LOG_ROOT, QUEUE_RESULTS_ROOT.

set -u
USAGE="usage: queue_low_update.sh [--from-step K]"
FROM=0
while [ $# -gt 0 ]; do
    case "$1" in
        --from-step) FROM="${2:-}"; shift 2 || { echo "$USAGE" >&2; exit 2; } ;;
        *) echo "unknown argument '$1'; $USAGE" >&2; exit 2 ;;
    esac
done
case "$FROM" in ''|*[!0-9]*) echo "--from-step must be an integer 0..11, got '$FROM'" >&2; exit 2;; esac
if [ "$FROM" -gt 11 ]; then echo "--from-step must be an integer 0..11, got '$FROM'" >&2; exit 2; fi

cd "$(dirname "$0")/.." || { echo "cannot cd to repo root" >&2; exit 1; }
export MUJOCO_GL=egl

LOGDIR="${QUEUE_LOG_ROOT:-results/logs}/queue_$(date +%Y%m%d_%H%M%S)_low_update"
mkdir -p "$LOGDIR"
QUEUE_LOG="$LOGDIR/queue.log"
RES="${QUEUE_RESULTS_ROOT:-results}"
LU="$RES/low_update"
PP="$RES/projection_pilot"

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

LUPY="uv run python scripts/low_update.py"
PPPY="uv run python scripts/projection_pilot.py"
note "QUEUE START low update from_step=$FROM git $(git rev-parse HEAD) status '$(git status --porcelain | wc -l) changes'"

# shellcheck disable=SC2086
{
i=0
for s in 1 2; do
    if want $i; then step "${i}_validation_s$s" $LUPY --part validation_seed --seed $s
    else prior "${i}_validation_s$s" "$LU/validation_seed$s.json"; fi
    i=$((i + 1))
done

want 2 && step 2_validation $LUPY --part validation

# The gates are always re-evaluated (they only read written verdicts).
gate_ok=0
step 3_gate $LUPY --part gate && gate_ok=1

chain_ok=0
if [ $gate_ok = 1 ]; then
    i=4
    for s in 0 1 2; do
        if want $i; then step "${i}_chain_s$s" $LUPY --part chain --seed $s
        else prior "${i}_chain_s$s" "$LU/seed$s/stage3/row.json"; fi
        i=$((i + 1))
    done
    if want 7; then step 7_chain_summary $LUPY --part chain_summary && chain_ok=1
    else prior 7_chain_summary "$LU/chain_summary.json" && chain_ok=1; fi
else
    note "SKIP 4-11: the validation gate is closed (not validated on both held-out seeds)"
fi

pilot_ok=0
if [ $chain_ok = 1 ]; then
    step 8_pilot_gate $LUPY --part pilot_gate && pilot_ok=1
fi
if [ $pilot_ok = 1 ]; then
    want 9 && step 9_pilot_lr3e-5 $PPPY --lr 3e-5
    want 10 && step 10_pilot_lr1e-4 $PPPY --lr 1e-4
    step 11_pilot_summary $PPPY --summary
elif [ $gate_ok = 1 ]; then
    note "SKIP 9-11: the pilot gate is closed (the chain under-learns, or no chain summary)"
fi
}

note "QUEUE DONE"
