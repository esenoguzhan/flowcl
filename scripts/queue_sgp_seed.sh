#!/usr/bin/env bash
# SGP queue for one seed: the SGP baseline (sgp_projected_adam) and adaptive SGP
# (sgp_projected_adam_ne90), their analyses, the per-batch probe cells, the pre-registered
# SGP report, and the replication once every registered seed has a report.
# Orchestration only: sequential commands, no experiment logic (that lives in flowcl/ and
# the committed configs; the rule is configs/analysis/sgp.yaml). queue_seed_pair.sh is
# left unchanged as the executed record of the GPM runs.
#
# Usage:    bash scripts/queue_sgp_seed.sh <seed> [--from-step K]
#   --from-step K  resume after an interruption: steps < K are NOT rerun. Their status is
#                  taken from their outputs on disk, logged as "PRIOR <step> ok/missing";
#                  a missing output blocks every step that depends on it, exactly as a
#                  failure would. A run directory left by an interrupted training step must
#                  be moved aside first (run_continual refuses to overwrite a run).
#
# Needs, for the seed: seq_ft, plain GPM and adaptive GPM runs with their sequence reports
# and diagnostics (checked and logged as INPUT lines).
#
#   0. sgp seed N (~5 h), identity-checked against plain GPM at stage 0
#   1-2. its sequence report and forgetting diagnostics (~20 min)
#   3. sgp_ne90 seed N (~5 h), identity-checked against run 0 at stages 0-1
#   4-5. its sequence report and forgetting diagnostics
#   6. the per-batch probe cells for P, A, G and N (~1 min)
#   7. the SGP report for seed N
#   8. the replication over the registered seeds (skipped until all have reports)
#
# Runs are sequential: two processes sharing the GPU could perturb cuDNN's algorithm choice
# and break the bitwise identity the comparison relies on.
#
# Never stops early: no `set -e`, no `exit` after start-up; each step's exit code is
# recorded in results/logs/queue_<stamp>_sgp_seed<N>/queue.log, one log per step next to it.
#
# Launch:   tmux new-session -d -s sgp0 "bash scripts/queue_sgp_seed.sh 0; exec bash"
# Resume:   tmux new-session -d -s sgp0 "bash scripts/queue_sgp_seed.sh 0 --from-step 3; exec bash"
# Testing:  QUEUE_DRY_RUN=1 echoes each command instead of running it;
#           QUEUE_FAIL_STEP=<step name> makes that step return 1;
#           QUEUE_LOG_ROOT overrides results/logs;
#           QUEUE_RESULTS_ROOT overrides results/ for the input and --from-step checks only.

set -u
USAGE="usage: queue_sgp_seed.sh <seed> [--from-step K]"
SEED="${1:?$USAGE}"
shift
FROM=0
while [ $# -gt 0 ]; do
    case "$1" in
        --from-step) FROM="${2:-}"; shift 2 || { echo "$USAGE" >&2; exit 2; } ;;
        *) echo "unknown argument '$1'; $USAGE" >&2; exit 2 ;;
    esac
done
case "$SEED" in ''|*[!0-9]*) echo "seed must be a non-negative integer, got '$SEED'" >&2; exit 2;; esac
case "$FROM" in ''|*[!0-9]*) echo "--from-step must be an integer 0..8, got '$FROM'" >&2; exit 2;; esac
if [ "$FROM" -gt 8 ]; then echo "--from-step must be an integer 0..8, got '$FROM'" >&2; exit 2; fi
REGISTERED_SEEDS="0 1 2"  # configs/analysis/sgp.yaml replication.seeds
case " $REGISTERED_SEEDS " in *" $SEED "*) ;; *) echo "seed $SEED is not registered in sgp.yaml" >&2; exit 2;; esac

cd "$(dirname "$0")/.." || { echo "cannot cd to repo root" >&2; exit 1; }
export MUJOCO_GL=egl

LOGDIR="${QUEUE_LOG_ROOT:-results/logs}/queue_$(date +%Y%m%d_%H%M%S)_sgp_seed${SEED}"
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

# An existing input this queue does not produce.
input() {
    local name=$1 path=$2
    if [ -e "$path" ]; then
        note "INPUT $name ok: $path"
        return 0
    fi
    note "INPUT $name missing: $path"
    return 1
}

S=$SEED
REF=seq_hetero__seq_ft__seed$S
GPM=seq_hetero__gpm_projected_adam__seed$S
NE90=seq_hetero__gpm_projected_adam_ne90__seed$S
SGP=seq_hetero__sgp_projected_adam__seed$S
SGPNE=seq_hetero__sgp_projected_adam_ne90__seed$S
if [ "$S" = 0 ]; then SUF=""; else SUF="_seed$S"; fi
note "QUEUE START sgp seed $S from_step=$FROM git $(git rev-parse HEAD) status '$(git status --porcelain | wc -l) changes'"

train_inputs=1
input reference "$RES/$REF/result.json" || train_inputs=0
input plain_gpm "$RES/$GPM/result.json" || train_inputs=0
report_inputs=1
input adaptive_gpm "$RES/$NE90/result.json" || report_inputs=0
input seqrep_gpm "$RES/gpm_seq$SUF/report.json" || report_inputs=0
input seqrep_gpm_ne90 "$RES/gpm_seq_ne90$SUF/report.json" || report_inputs=0
input diag_gpm "$RES/forgetting_diag$SUF/report.json" || report_inputs=0
input diag_gpm_ne90 "$RES/forgetting_diag_ne90$SUF/report.json" || report_inputs=0

sgp_ok=0; seqrep_sgp_ok=0; diag_sgp_ok=0; ne90_ok=0; seqrep_ne90_ok=0; diag_ne90_ok=0
probe_ok=0; report_ok=0

if [ $train_inputs = 1 ]; then
    if want 0; then
        step "0_sgp_s$S" uv run python scripts/run_continual.py --curriculum seq_hetero \
            --method sgp --seed "$S" --amp \
            --identity-reference-run "results/$GPM" --identity-stages 0 && sgp_ok=1
    else
        prior "0_sgp_s$S" "$RES/$SGP/result.json" && sgp_ok=1
    fi
else
    note "SKIP every step: the seq_ft or plain GPM seed-$S run is missing"
fi

if [ $sgp_ok = 1 ]; then
    # The baseline's own analyses first (~20 min), so its results are readable early.
    if want 1; then
        step "1_seqrep_sgp_s$S" uv run python scripts/sequence_report.py \
            --method-run "results/$SGP" --reference-run "results/$REF" \
            --out "results/sgp_seq_seed$S/report.json" && seqrep_sgp_ok=1
    else
        prior "1_seqrep_sgp_s$S" "$RES/sgp_seq_seed$S/report.json" && seqrep_sgp_ok=1
    fi
    if want 2; then
        step "2_diag_sgp_s$S" uv run python scripts/forgetting_diagnostics.py \
            --method-run "$SGP" --reference-run "$REF" \
            --out "results/forgetting_diag_sgp_seed$S/report.json" && diag_sgp_ok=1
    else
        prior "2_diag_sgp_s$S" "$RES/forgetting_diag_sgp_seed$S/report.json" && diag_sgp_ok=1
    fi
    if want 3; then
        step "3_sgp_ne90_s$S" uv run python scripts/run_continual.py --curriculum seq_hetero \
            --method sgp_ne90 --seed "$S" --amp \
            --identity-reference-run "results/$SGP" --identity-stages 0 1 && ne90_ok=1
    else
        prior "3_sgp_ne90_s$S" "$RES/$SGPNE/result.json" && ne90_ok=1
    fi
elif [ $train_inputs = 1 ]; then
    note "SKIP 1_seqrep_sgp_s$S 2_diag_sgp_s$S 3_sgp_ne90_s$S: 0_sgp_s$S failed or is missing"
fi

if [ $ne90_ok = 1 ]; then
    if want 4; then
        step "4_seqrep_sgp_ne90_s$S" uv run python scripts/sequence_report.py \
            --method-run "results/$SGPNE" --reference-run "results/$REF" \
            --out "results/sgp_seq_ne90_seed$S/report.json" && seqrep_ne90_ok=1
    else
        prior "4_seqrep_sgp_ne90_s$S" "$RES/sgp_seq_ne90_seed$S/report.json" && seqrep_ne90_ok=1
    fi
    if want 5; then
        step "5_diag_sgp_ne90_s$S" uv run python scripts/forgetting_diagnostics.py \
            --method-run "$SGPNE" --reference-run "$REF" \
            --out "results/forgetting_diag_sgp_ne90_seed$S/report.json" && diag_ne90_ok=1
    else
        prior "5_diag_sgp_ne90_s$S" "$RES/forgetting_diag_sgp_ne90_seed$S/report.json" && diag_ne90_ok=1
    fi
else
    note "SKIP 4_seqrep_sgp_ne90_s$S 5_diag_sgp_ne90_s$S: no adaptive SGP seed-$S run"
fi

if [ $diag_sgp_ok = 1 ] && [ $diag_ne90_ok = 1 ] && [ $report_inputs = 1 ]; then
    if want 6; then
        step "6_probe_cells_s$S" uv run python scripts/probe_cells.py --seed "$S" && probe_ok=1
    else
        prior "6_probe_cells_s$S" "$RES/sgp_probe_seed$S/report.json" && probe_ok=1
    fi
else
    note "SKIP 6_probe_cells_s$S: a seed-$S diagnostics report or GPM input is missing"
fi

if [ $probe_ok = 1 ] && [ $seqrep_sgp_ok = 1 ] && [ $seqrep_ne90_ok = 1 ]; then
    if want 7; then
        step "7_sgp_report_s$S" uv run python scripts/sgp_report.py --seed "$S" && report_ok=1
    else
        prior "7_sgp_report_s$S" "$RES/sgp_seed$S/report.json" && report_ok=1
    fi
else
    note "SKIP 7_sgp_report_s$S: the probe cells or a seed-$S sequence report is missing"
fi

others_ok=1
missing_seeds=""
for other in $REGISTERED_SEEDS; do
    [ "$other" = "$S" ] && continue
    if [ ! -e "$RES/sgp_seed$other/report.json" ]; then others_ok=0; missing_seeds="$missing_seeds $other"; fi
done
if [ $report_ok = 1 ] && [ $others_ok = 1 ]; then
    step 8_sgp_replication uv run python scripts/sgp_report.py --replication
elif [ $report_ok = 1 ]; then
    note "SKIP 8_sgp_replication: no SGP report yet for seed(s)$missing_seeds"
else
    note "SKIP 8_sgp_replication: no seed-$S SGP report"
fi

note "QUEUE DONE"
