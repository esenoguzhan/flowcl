#!/usr/bin/env bash
# Overnight GPU queue, 24-25 Sep 2026. Orchestration only: sequential commands, no
# experiment logic (that lives in flowcl/ and the committed configs).
#
#   1. gpm_projected_adam_ne90, seq_hetero, seed 0 — the adaptive-GPM causal test, with the
#      fail-fast identity check against the plain GPM run at stages 0-1 (~5 h).
#   2. Only if 1 succeeded: its sequence report, forgetting diagnostics and the
#      pre-registered adaptive outcome report (~30 min).
#   3. Always: seq_ft, seq_hetero, seed 1 — the paired reference for later seeds (~5.6 h).
#
# Never stops early: no `set -e`, no `exit`; each step's exit code is recorded in
# results/logs/queue_<stamp>/queue.log and one log per step sits next to it.
#
# Launch:   tmux new-session -d -s overnight "bash scripts/queue_2026-09-24_overnight.sh; exec bash"
# Testing:  QUEUE_DRY_RUN=1 echoes each command instead of running it;
#           QUEUE_FAIL_STEP=<step name> makes that step return 1;
#           QUEUE_LOG_ROOT overrides results/logs.

set -u
cd "$(dirname "$0")/.." || { echo "cannot cd to repo root" >&2; exit 1; }
export MUJOCO_GL=egl

LOGDIR="${QUEUE_LOG_ROOT:-results/logs}/queue_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$LOGDIR"
QUEUE_LOG="$LOGDIR/queue.log"

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

VARIANT=seq_hetero__gpm_projected_adam_ne90__seed0
note "QUEUE START git $(git rev-parse HEAD) status '$(git status --porcelain | wc -l) changes'"

if step 1_gpm_ne90 uv run python scripts/run_continual.py --curriculum seq_hetero \
        --method gpm_ne90 --seed 0 --amp --single-task-baseline \
        --identity-reference-run results/seq_hetero__gpm_projected_adam__seed0 \
        --identity-stages 0 1; then
    step 2_sequence_report uv run python scripts/sequence_report.py \
        --method-run "results/$VARIANT" --out results/gpm_seq_ne90/report.json
    if step 3_diagnostics uv run python scripts/forgetting_diagnostics.py \
            --method-run "$VARIANT" --out results/forgetting_diag_ne90/report.json; then
        step 4_adaptive_report uv run python scripts/adaptive_report.py
    else
        note "SKIP 4_adaptive_report: diagnostics failed"
    fi
else
    note "SKIP 2_sequence_report 3_diagnostics 4_adaptive_report: 1_gpm_ne90 failed"
fi

step 5_seq_ft_seed1 uv run python scripts/run_continual.py --curriculum seq_hetero \
    --method seq_ft --seed 1 --amp

note "QUEUE DONE"
