#!/usr/bin/env bash
# A1, the stage-boundary resume test: resume plain GPM seed 0 at stage 3 into results/resume_check,
# then compare it with the source. Orchestration only; the rule is configs/analysis/resume_check.yaml.
# Run it only when no other GPU job is running.
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
cd "$(dirname "$0")/.." || { echo "cannot cd to repo root" >&2; exit 1; }
export MUJOCO_GL=egl

LOGDIR="${QUEUE_LOG_ROOT:-results/logs}/queue_$(date +%Y%m%d_%H%M%S)_resume_check"
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

SRC=results/seq_hetero__gpm_projected_adam__seed0
STAGE2_SHA=$(uv run python -c "from omegaconf import OmegaConf; print(OmegaConf.load('configs/analysis/resume_check.yaml').source.sha256.stage2)")
note "QUEUE START resume_check git $(git rev-parse HEAD) status '$(git status --porcelain | wc -l) changes'"

step 0_resume uv run python scripts/run_continual.py --curriculum seq_hetero --method gpm --seed 0 --amp \
    --resume-run "$SRC" --start-stage 3 --resume-sha256 "$STAGE2_SHA" \
    --identity-reference-run "$SRC" --identity-stages 3 --results-root results/resume_check
step 1_check uv run python scripts/resume_check.py

if [ -z "$FAILED" ]; then
    note "QUEUE DONE ok"
    exit 0
fi
note "QUEUE DONE FAILED:$FAILED"
exit 1
