#!/usr/bin/env bash
# Seed-pair queue: plain GPM and adaptive GPM (ne90) for one seed, their analyses, the
# pre-registered adaptive report, and the replication summary over seeds 0..N.
# Orchestration only: sequential commands, no experiment logic (that lives in flowcl/ and
# the committed configs). The dated queues (queue_2026-09-2*_*.sh) are kept as records.
#
# Usage:    bash scripts/queue_seed_pair.sh <seed> [--with-seq-ft] [--from-step K]
#   --with-seq-ft  first trains the paired seq_ft reference for this seed (step 0); every
#                  GPM step needs it and is skipped if it fails.
#   --from-step K  resume after an interruption: steps < K are NOT rerun. Their status is
#                  taken from their outputs on disk, logged as "PRIOR <step> ok/missing";
#                  a missing output blocks every step that depends on it, exactly as a
#                  failure would. Training runs cannot resume mid-run: a run directory left
#                  by an interrupted step K must be moved aside first (run_continual refuses
#                  to overwrite a run).
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
# Resume:   tmux new-session -d -s seed2 "bash scripts/queue_seed_pair.sh 2 --with-seq-ft --from-step 4; exec bash"
# Testing:  QUEUE_DRY_RUN=1 echoes each command instead of running it;
#           QUEUE_FAIL_STEP=<step name> makes that step return 1;
#           QUEUE_LOG_ROOT overrides results/logs;
#           QUEUE_RESULTS_ROOT overrides results/ for the --from-step output checks only.

set -u
USAGE="usage: queue_seed_pair.sh <seed> [--with-seq-ft] [--from-step K]"
SEED="${1:?$USAGE}"
shift
WITH_SEQFT=0
FROM=0
while [ $# -gt 0 ]; do
    case "$1" in
        --with-seq-ft) WITH_SEQFT=1; shift ;;
        --from-step) FROM="${2:-}"; shift 2 || { echo "$USAGE" >&2; exit 2; } ;;
        *) echo "unknown argument '$1'; $USAGE" >&2; exit 2 ;;
    esac
done
case "$SEED" in ''|*[!0-9]*) echo "seed must be a non-negative integer, got '$SEED'" >&2; exit 2;; esac
case "$FROM" in ''|*[!0-9]*) echo "--from-step must be an integer 0..8, got '$FROM'" >&2; exit 2;; esac
if [ "$FROM" -gt 8 ]; then echo "--from-step must be an integer 0..8, got '$FROM'" >&2; exit 2; fi

cd "$(dirname "$0")/.." || { echo "cannot cd to repo root" >&2; exit 1; }
export MUJOCO_GL=egl

LOGDIR="${QUEUE_LOG_ROOT:-results/logs}/queue_$(date +%Y%m%d_%H%M%S)_seed${SEED}"
mkdir -p "$LOGDIR"
QUEUE_LOG="$LOGDIR/queue.log"
RES="${QUEUE_RESULTS_ROOT:-results}"

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

# Step K runs unless we resume after it.
want() { [ "$1" -ge "$FROM" ]; }

# A step before --from-step: done iff its output exists on disk.
prior() {
    local name=$1 path=$2
    if [ -e "$path" ]; then
        note "PRIOR $name ok: $path"
        return 0
    fi
    note "PRIOR $name missing: $path"
    return 1
}

S=$SEED
REF=seq_hetero__seq_ft__seed$S
GPM=seq_hetero__gpm_projected_adam__seed$S
NE90=seq_hetero__gpm_projected_adam_ne90__seed$S
if [ "$S" = 0 ]; then ADAPTIVE_OUT=adaptive_gpm/report.json; else ADAPTIVE_OUT=adaptive_gpm_seed$S/report.json; fi
REPLICATION_SEEDS=$(seq -s ' ' 0 "$S")
note "QUEUE START seed $S with_seq_ft=$WITH_SEQFT from_step=$FROM git $(git rev-parse HEAD) status '$(git status --porcelain | wc -l) changes'"

ref_ok=1; gpm_ok=0; seqrep_gpm_ok=0; diag_gpm_ok=0; ne90_ok=0; seqrep_ne90_ok=0
diag_ne90_ok=0; adaptive_ok=0

if [ $WITH_SEQFT = 1 ]; then
    ref_ok=0
    if want 0; then
        step "0_seqft_s$S" uv run python scripts/run_continual.py --curriculum seq_hetero \
            --method seq_ft --seed "$S" --amp && ref_ok=1
    else
        prior "0_seqft_s$S" "$RES/$REF/result.json" && ref_ok=1
    fi
fi

if [ $ref_ok = 1 ]; then
    if want 1; then
        step "1_gpm_s$S" uv run python scripts/run_continual.py --curriculum seq_hetero \
            --method gpm --seed "$S" --amp && gpm_ok=1
    else
        prior "1_gpm_s$S" "$RES/$GPM/result.json" && gpm_ok=1
    fi
else
    note "SKIP 1_gpm_s$S and every later step: 0_seqft_s$S failed or is missing"
fi

if [ $gpm_ok = 1 ]; then
    # The baseline's own analyses first (~20 min), so its results are readable early.
    if want 2; then
        step "2_seqrep_gpm_s$S" uv run python scripts/sequence_report.py \
            --method-run "results/$GPM" --reference-run "results/$REF" \
            --out "results/gpm_seq_seed$S/report.json" && seqrep_gpm_ok=1
    else
        prior "2_seqrep_gpm_s$S" "$RES/gpm_seq_seed$S/report.json" && seqrep_gpm_ok=1
    fi
    if want 3; then
        step "3_diag_gpm_s$S" uv run python scripts/forgetting_diagnostics.py \
            --method-run "$GPM" --reference-run "$REF" \
            --out "results/forgetting_diag_seed$S/report.json" && diag_gpm_ok=1
    else
        prior "3_diag_gpm_s$S" "$RES/forgetting_diag_seed$S/report.json" && diag_gpm_ok=1
    fi
    if want 4; then
        step "4_gpm_ne90_s$S" uv run python scripts/run_continual.py --curriculum seq_hetero \
            --method gpm_ne90 --seed "$S" --amp \
            --identity-reference-run "results/$GPM" --identity-stages 0 1 && ne90_ok=1
    else
        prior "4_gpm_ne90_s$S" "$RES/$NE90/result.json" && ne90_ok=1
    fi
elif [ $ref_ok = 1 ]; then
    note "SKIP 2_seqrep_gpm_s$S 3_diag_gpm_s$S 4_gpm_ne90_s$S: 1_gpm_s$S failed or is missing"
fi

if [ $ne90_ok = 1 ]; then
    if want 5; then
        step "5_seqrep_ne90_s$S" uv run python scripts/sequence_report.py \
            --method-run "results/$NE90" --reference-run "results/$REF" \
            --out "results/gpm_seq_ne90_seed$S/report.json" && seqrep_ne90_ok=1
    else
        prior "5_seqrep_ne90_s$S" "$RES/gpm_seq_ne90_seed$S/report.json" && seqrep_ne90_ok=1
    fi
    if want 6; then
        step "6_diag_ne90_s$S" uv run python scripts/forgetting_diagnostics.py \
            --method-run "$NE90" --reference-run "$REF" \
            --out "results/forgetting_diag_ne90_seed$S/report.json" && diag_ne90_ok=1
    else
        prior "6_diag_ne90_s$S" "$RES/forgetting_diag_ne90_seed$S/report.json" && diag_ne90_ok=1
    fi
else
    note "SKIP 5_seqrep_ne90_s$S 6_diag_ne90_s$S: no adaptive seed-$S run"
fi

if [ $seqrep_gpm_ok = 1 ] && [ $diag_gpm_ok = 1 ] && [ $seqrep_ne90_ok = 1 ] && [ $diag_ne90_ok = 1 ]; then
    if want 7; then
        step "7_adaptive_s$S" uv run python scripts/adaptive_report.py --seed "$S" && adaptive_ok=1
    else
        prior "7_adaptive_s$S" "$RES/$ADAPTIVE_OUT" && adaptive_ok=1
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
