#!/usr/bin/env bash
# Seed-pair queue: plain GPM and adaptive GPM (ne90) for one seed, their analyses, the
# pre-registered adaptive report, and the replication summary over seeds 0..N.
# Orchestration only: sequential commands, no experiment logic (that lives in flowcl/ and
# the committed configs). The dated queues (queue_2026-09-2*_*.sh) are kept as records.
#
# Usage:    bash scripts/queue_seed_pair.sh <seed> [--with-seq-ft]
#   --with-seq-ft  first trains the paired seq_ft reference for this seed (step 0); every
#                  GPM step needs it and is skipped if it fails.
#
#   0. seq_ft seed N (optional, ~5.3 h)
#   1. gpm_projected_adam seed N (~4.7 h; T1 pairing against seq_ft seed N, automatic)
#   2-3. its sequence report and forgetting diagnostics (~20 min)
#   4. gpm_projected_adam_ne90 seed N, identity-checked against run 1 at stages 0-1
#   5-6. its sequence report and forgetting diagnostics
#   7. the adaptive report for seed N (consumes both sequence reports and diagnostics)
#   8. the replication summary over seeds 0..N
#
# Runs are sequential: two processes sharing the GPU could perturb cuDNN's algorithm choice
# and break the bitwise identity the comparison relies on.
#
# Never stops early: no `set -e`, no `exit` after start-up; each step's exit code is
# recorded in results/logs/queue_<stamp>/queue.log, one log per step next to it.
#
# Launch:   tmux new-session -d -s seed2 "bash scripts/queue_seed_pair.sh 2 --with-seq-ft; exec bash"
# Testing:  QUEUE_DRY_RUN=1 echoes each command instead of running it;
#           QUEUE_FAIL_STEP=<step name> makes that step return 1;
#           QUEUE_LOG_ROOT overrides results/logs.

set -u
SEED="${1:?usage: queue_seed_pair.sh <seed> [--with-seq-ft]}"
WITH_SEQFT=0
[ "${2:-}" = "--with-seq-ft" ] && WITH_SEQFT=1
case "$SEED" in ''|*[!0-9]*) echo "seed must be a non-negative integer, got '$SEED'" >&2; exit 2;; esac

cd "$(dirname "$0")/.." || { echo "cannot cd to repo root" >&2; exit 1; }
export MUJOCO_GL=egl

LOGDIR="${QUEUE_LOG_ROOT:-results/logs}/queue_$(date +%Y%m%d_%H%M%S)_seed${SEED}"
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

S=$SEED
REF=seq_hetero__seq_ft__seed$S
GPM=seq_hetero__gpm_projected_adam__seed$S
NE90=seq_hetero__gpm_projected_adam_ne90__seed$S
REPLICATION_SEEDS=$(seq -s ' ' 0 "$S")
note "QUEUE START seed $S with_seq_ft=$WITH_SEQFT git $(git rev-parse HEAD) status '$(git status --porcelain | wc -l) changes'"

ref_ok=1; gpm_ok=0; seqrep_gpm_ok=0; diag_gpm_ok=0; ne90_ok=0; seqrep_ne90_ok=0
diag_ne90_ok=0; adaptive_ok=0

if [ $WITH_SEQFT = 1 ]; then
    ref_ok=0
    if step "0_seqft_s$S" uv run python scripts/run_continual.py --curriculum seq_hetero \
            --method seq_ft --seed "$S" --amp; then
        ref_ok=1
    fi
fi

if [ $ref_ok = 1 ]; then
    if step "1_gpm_s$S" uv run python scripts/run_continual.py --curriculum seq_hetero \
            --method gpm --seed "$S" --amp; then
        gpm_ok=1
    fi
else
    note "SKIP 1_gpm_s$S and every later step: 0_seqft_s$S failed"
fi

if [ $gpm_ok = 1 ]; then
    # The baseline's own analyses first (~20 min), so its results are readable early.
    if step "2_seqrep_gpm_s$S" uv run python scripts/sequence_report.py \
            --method-run "results/$GPM" --reference-run "results/$REF" \
            --out "results/gpm_seq_seed$S/report.json"; then
        seqrep_gpm_ok=1
    fi
    if step "3_diag_gpm_s$S" uv run python scripts/forgetting_diagnostics.py \
            --method-run "$GPM" --reference-run "$REF" \
            --out "results/forgetting_diag_seed$S/report.json"; then
        diag_gpm_ok=1
    fi
    if step "4_gpm_ne90_s$S" uv run python scripts/run_continual.py --curriculum seq_hetero \
            --method gpm_ne90 --seed "$S" --amp \
            --identity-reference-run "results/$GPM" --identity-stages 0 1; then
        ne90_ok=1
    fi
elif [ $ref_ok = 1 ]; then
    note "SKIP 2_seqrep_gpm_s$S 3_diag_gpm_s$S 4_gpm_ne90_s$S: 1_gpm_s$S failed"
fi

if [ $ne90_ok = 1 ]; then
    if step "5_seqrep_ne90_s$S" uv run python scripts/sequence_report.py \
            --method-run "results/$NE90" --reference-run "results/$REF" \
            --out "results/gpm_seq_ne90_seed$S/report.json"; then
        seqrep_ne90_ok=1
    fi
    if step "6_diag_ne90_s$S" uv run python scripts/forgetting_diagnostics.py \
            --method-run "$NE90" --reference-run "$REF" \
            --out "results/forgetting_diag_ne90_seed$S/report.json"; then
        diag_ne90_ok=1
    fi
else
    note "SKIP 5_seqrep_ne90_s$S 6_diag_ne90_s$S: no adaptive seed-$S run"
fi

if [ $seqrep_gpm_ok = 1 ] && [ $diag_gpm_ok = 1 ] && [ $seqrep_ne90_ok = 1 ] && [ $diag_ne90_ok = 1 ]; then
    if step "7_adaptive_s$S" uv run python scripts/adaptive_report.py --seed "$S"; then
        adaptive_ok=1
    fi
else
    note "SKIP 7_adaptive_s$S: a seed-$S sequence report or diagnostics report is missing"
fi

if [ $adaptive_ok = 1 ] && [ "$S" -ge 1 ]; then
    # shellcheck disable=SC2086
    step 8_replication uv run python scripts/adaptive_report.py --replication $REPLICATION_SEEDS
elif [ "$S" -lt 1 ]; then
    note "SKIP 8_replication: replication needs at least two seeds"
else
    note "SKIP 8_replication: no seed-$S adaptive report"
fi

note "QUEUE DONE"
