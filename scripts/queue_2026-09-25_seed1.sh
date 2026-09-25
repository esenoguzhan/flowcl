#!/usr/bin/env bash
# Seed-1 replication queue, 25 Sep 2026. Orchestration only: sequential commands, no
# experiment logic (that lives in flowcl/ and the committed configs).
#
#   1. gpm_projected_adam, seq_hetero, seed 1 — the plain baseline (~4.7 h). Its T1
#      pairing against seq_ft seed 1 runs automatically.
#   2-3. Its sequence report and forgetting diagnostics against seq_ft seed 1 (~20 min).
#   4. gpm_projected_adam_ne90, seed 1 — identity-checked against run 1 at stages 0-1.
#      Needs run 1's checkpoints, so it runs only if 1 succeeded (~4.7 h).
#   5-6. The variant's sequence report and forgetting diagnostics.
#   7. The pre-registered adaptive report for seed 1; 8. the replication summary (0, 1).
#
# Runs are sequential: two processes sharing the GPU could perturb cuDNN's algorithm choice
# and break the bitwise identity the comparison relies on.
#
# Never stops early: no `set -e`, no `exit`; each step's exit code is recorded in
# results/logs/queue_<stamp>/queue.log and one log per step sits next to it.
#
# Launch:   tmux new-session -d -s seed1 "bash scripts/queue_2026-09-25_seed1.sh; exec bash"
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

REF=seq_hetero__seq_ft__seed1
GPM=seq_hetero__gpm_projected_adam__seed1
NE90=seq_hetero__gpm_projected_adam_ne90__seed1
note "QUEUE START git $(git rev-parse HEAD) status '$(git status --porcelain | wc -l) changes'"

gpm_ok=0; ne90_ok=0; diag_gpm_ok=0; diag_ne90_ok=0; adaptive_ok=0

if step 1_gpm_s1 uv run python scripts/run_continual.py --curriculum seq_hetero \
        --method gpm --seed 1 --amp; then
    gpm_ok=1
fi

if [ $gpm_ok = 1 ]; then
    # The baseline's own analyses first (~20 min), so its results are readable early.
    step 2_seqrep_gpm_s1 uv run python scripts/sequence_report.py \
        --method-run "results/$GPM" --reference-run "results/$REF" \
        --out results/gpm_seq_seed1/report.json
    if step 3_diag_gpm_s1 uv run python scripts/forgetting_diagnostics.py \
            --method-run "$GPM" --reference-run "$REF" \
            --out results/forgetting_diag_seed1/report.json; then
        diag_gpm_ok=1
    fi
    if step 4_gpm_ne90_s1 uv run python scripts/run_continual.py --curriculum seq_hetero \
            --method gpm_ne90 --seed 1 --amp \
            --identity-reference-run "results/$GPM" --identity-stages 0 1; then
        ne90_ok=1
    fi
else
    note "SKIP 2_seqrep_gpm_s1 3_diag_gpm_s1 4_gpm_ne90_s1: 1_gpm_s1 failed"
fi

if [ $ne90_ok = 1 ]; then
    step 5_seqrep_ne90_s1 uv run python scripts/sequence_report.py \
        --method-run "results/$NE90" --reference-run "results/$REF" \
        --out results/gpm_seq_ne90_seed1/report.json
    if step 6_diag_ne90_s1 uv run python scripts/forgetting_diagnostics.py \
            --method-run "$NE90" --reference-run "$REF" \
            --out results/forgetting_diag_ne90_seed1/report.json; then
        diag_ne90_ok=1
    fi
else
    note "SKIP 5_seqrep_ne90_s1 6_diag_ne90_s1: no adaptive seed-1 run"
fi

if [ $diag_gpm_ok = 1 ] && [ $diag_ne90_ok = 1 ]; then
    if step 7_adaptive_s1 uv run python scripts/adaptive_report.py --seed 1; then
        adaptive_ok=1
    fi
else
    note "SKIP 7_adaptive_s1: a seed-1 diagnostics report is missing"
fi

if [ $adaptive_ok = 1 ]; then
    step 8_replication uv run python scripts/adaptive_report.py --replication 0 1
else
    note "SKIP 8_replication: no seed-1 adaptive report"
fi

note "QUEUE DONE"
