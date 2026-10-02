#!/usr/bin/env bash
# A1, the stage-boundary resume test: resume the config's source run at its stage into
# results/<out>, then compare it with the source. Orchestration only; the rule is the config
# (default configs/analysis/resume_check.yaml: plain GPM seed 0 at stage 3).
# Run it only when no other GPU job is running.
#
# Usage:    bash scripts/queue_resume_check.sh [--config C]
#
#   0  resume: scripts/run_continual.py --resume-run ... --start-stage 3 (about 1 h: train T4,
#      update the memory, evaluate four tasks); stops before evaluation if stage 3's checkpoint is
#      not bitwise the source's
#   1  check: scripts/resume_check.py (exit 0 pass, 1 fail); ALWAYS runs
#
# Status: the last line is "QUEUE DONE ok" (exit 0) or "QUEUE DONE FAILED: <steps>" (exit 1).
# Launch:   tmux new-session -d -s resume "bash scripts/queue_resume_check.sh; exec bash"
# Testing:  QUEUE_DRY_RUN=1, QUEUE_FAIL_STEP=<step name>, QUEUE_LOG_ROOT.

set -u
CONFIG=configs/analysis/resume_check.yaml
while [ $# -gt 0 ]; do
    case "$1" in
        --config) CONFIG="${2:-}"; shift 2 || { echo "usage: queue_resume_check.sh [--config C]" >&2; exit 2; } ;;
        *) echo "unknown argument '$1'" >&2; exit 2 ;;
    esac
done
cd "$(dirname "$0")/.." || { echo "cannot cd to repo root" >&2; exit 1; }
export MUJOCO_GL=egl

read -r OUT RUN STAGE METHOD STAGE_SHA < <(uv run python -c "
from omegaconf import OmegaConf
c = OmegaConf.load('$CONFIG')
s = c.source
print(c.out, s.run, s.stage, s.method, s.sha256.stage2)") || { echo "cannot read $CONFIG" >&2; exit 2; }
[ -n "${STAGE_SHA:-}" ] || { echo "cannot read $CONFIG" >&2; exit 2; }

LOGDIR="${QUEUE_LOG_ROOT:-results/logs}/queue_$(date +%Y%m%d_%H%M%S)_$OUT"
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

SRC=results/$RUN
note "QUEUE START $OUT config=$CONFIG git $(git rev-parse HEAD) status '$(git status --porcelain | wc -l) changes'"

step 0_resume uv run python scripts/run_continual.py --curriculum seq_hetero --method "$METHOD" --seed 0 --amp \
    --resume-run "$SRC" --start-stage "$STAGE" --resume-sha256 "$STAGE_SHA" \
    --identity-reference-run "$SRC" --identity-stages "$STAGE" --results-root "results/$OUT"
step 1_check uv run python scripts/resume_check.py --config "$CONFIG"

if [ -z "$FAILED" ]; then
    note "QUEUE DONE ok"
    exit 0
fi
note "QUEUE DONE FAILED:$FAILED"
exit 1
