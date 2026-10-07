#!/usr/bin/env bash
# C1, the high-protection control: gpm_projected_adam_e99 on seq_hetero for seeds 0, 1, 2, each
# with its sequence report and pre-registered seed report, then the replication. Orchestration
# only; the rule is configs/analysis/high_protection.yaml. Run it only when no other GPU job is
# running (two processes could perturb cuDNN and break the bitwise identity the rule relies on).
#
# Usage:    bash scripts/queue_high_protection.sh [--from-step K] [--smoke DIR]
#   --from-step K  resume: steps < K are not rerun (status from their outputs on disk; a missing
#                  output blocks its dependents). A run directory left by an interrupted training
#                  step must be moved aside first.
#   --smoke DIR    seed 2 only (derived thresholds), 200 steps and 1 episode per cell, into DIR
#                  (outside results/), on a dirty tree: first seq_ft, plain and adaptive GPM and
#                  their sequence reports (steps s0-s4), then steps 6-8 with --smoke. No replication.
#
#   per seed S (k = 3 * index of S in 0 1 2):
#   k+0  e99 run         (~5.1 h; T1 pairing against seq_ft, identity against plain at stages 0-1)
#   k+1  sequence report -> gpm_seq_e99_seed<S>/report.json
#   k+2  seed report     -> high_protection/seed<S>/report.json
#   9    replication     -> high_protection/replication.json (once all three seed reports exist)
#
# IDENTITY STOP: when an e99 run fails and its identity_stage0/1.json reads "passed": false, every
# later seed is skipped (logged STOP); the same failure would recur. Any other failure skips only
# that seed's dependents.
#
# Status: every non-zero rc, skipped seed and missing prior output is collected; the last line is
# "QUEUE DONE ok" (exit 0) or "QUEUE DONE FAILED: <steps>" (exit 1). A seed report exits 3 for an
# invalid_* verdict, so it is listed as failed; every other verdict is rc 0.
#
# Launch:   tmux new-session -d -s hiprot "bash scripts/queue_high_protection.sh; exec bash"
# Testing:  QUEUE_DRY_RUN=1, QUEUE_FAIL_STEP=<step name>, QUEUE_LOG_ROOT, QUEUE_RESULTS_ROOT.

set -u
USAGE="usage: queue_high_protection.sh [--from-step K] [--smoke DIR]"
FROM=0
SMOKE=""
while [ $# -gt 0 ]; do
    case "$1" in
        --from-step) FROM="${2:-}"; shift 2 || { echo "$USAGE" >&2; exit 2; } ;;
        --smoke) SMOKE="${2:-}"; shift 2 || { echo "$USAGE" >&2; exit 2; } ;;
        *) echo "unknown argument '$1'; $USAGE" >&2; exit 2 ;;
    esac
done
case "$FROM" in ''|*[!0-9]*) echo "--from-step must be an integer 0..9, got '$FROM'" >&2; exit 2;; esac
if [ "$FROM" -gt 9 ]; then echo "--from-step must be an integer 0..9, got '$FROM'" >&2; exit 2; fi

cd "$(dirname "$0")/.." || { echo "cannot cd to repo root" >&2; exit 1; }
export MUJOCO_GL=egl

CUR=seq_hetero
RUN_ARGS=(--curriculum "$CUR" --amp)
REPORT_ARGS=()
if [ -n "$SMOKE" ]; then
    case "$SMOKE" in /*) ;; *) echo "--smoke needs an absolute directory" >&2; exit 2 ;; esac
    case "$SMOKE/" in "$(pwd)/results/"*) echo "--smoke must be outside results/" >&2; exit 2 ;; esac
    RES="$SMOKE"
    mkdir -p "$RES"
    RUN_ARGS+=(--results-root "$RES" --steps-per-task 200 --n-episodes 1 --allow-dirty)
    REPORT_ARGS+=(--results-root "$RES" --smoke)
    SEEDS="2"
    TAG="_smoke"
else
    RES="${QUEUE_RESULTS_ROOT:-results}"
    SEEDS="0 1 2"
    TAG=""
fi

LOGDIR="${QUEUE_LOG_ROOT:-results/logs}/queue_$(date +%Y%m%d_%H%M%S)_high_protection${TAG}"
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

want() { [ "$1" -ge "$FROM" ]; }

prior() {
    local name=$1 path=$2
    if [ -e "$path" ]; then note "PRIOR $name ok: $path"; return 0; fi
    note "PRIOR $name missing: $path"
    FAILED="$FAILED $name(missing)"
    return 1
}

# run_or_prior K NAME OUTPUT CMD...: run step K, or (before --from-step) check its output.
run_or_prior() {
    local k=$1 name=$2 path=$3; shift 3
    if want "$k"; then step "${k}_$name" "$@"
    else prior "${k}_$name" "$path"; fi
}

input() {
    local name=$1 path=$2
    if [ -e "$path" ]; then note "INPUT $name ok: $path"; return 0; fi
    note "INPUT $name missing: $path"
    return 1
}

# identity_failed RUN_DIR: an identity file of the run reads "passed": false.
identity_failed() {
    local f
    for f in "$1/identity_stage0.json" "$1/identity_stage1.json"; do
        if [ -e "$f" ] && grep -q '"passed": false' "$f"; then
            note "IDENTITY FAILED: $f"
            return 0
        fi
    done
    return 1
}

PY="uv run python"
note "QUEUE START high_protection smoke='$SMOKE' from_step=$FROM git $(git rev-parse HEAD) status '$(git status --porcelain | wc -l) changes'"

if [ -n "$SMOKE" ]; then
    # The comparators of the smoke seed, at smoke settings (production uses the existing runs).
    S=2
    REF=${CUR}__seq_ft__seed$S
    GPM=${CUR}__gpm_projected_adam__seed$S
    NE90=${CUR}__gpm_projected_adam_ne90__seed$S
    step s0_seqft $PY scripts/run_continual.py "${RUN_ARGS[@]}" --seed $S --method seq_ft
    step s1_gpm $PY scripts/run_continual.py "${RUN_ARGS[@]}" --seed $S --method gpm \
        --t1-reference-run "$RES/$REF"
    step s2_ne90 $PY scripts/run_continual.py "${RUN_ARGS[@]}" --seed $S --method gpm_ne90 \
        --t1-reference-run "$RES/$REF" --identity-reference-run "$RES/$GPM" --identity-stages 0 1
    step s3_seqrep_gpm $PY scripts/sequence_report.py --method-run "$RES/$GPM" \
        --reference-run "$RES/$REF" --out "$RES/gpm_seq_seed$S/report.json"
    step s4_seqrep_ne90 $PY scripts/sequence_report.py --method-run "$RES/$NE90" \
        --reference-run "$RES/$REF" --out "$RES/gpm_seq_ne90_seed$S/report.json"
fi

STOP=0
for S in $SEEDS; do
    case "$S" in 0) K=0 ;; 1) K=3 ;; 2) K=6 ;; esac
    REF=${CUR}__seq_ft__seed$S
    GPM=${CUR}__gpm_projected_adam__seed$S
    NE90=${CUR}__gpm_projected_adam_ne90__seed$S
    E99=${CUR}__gpm_projected_adam_e99__seed$S
    if [ "$S" = 0 ]; then SUF=""; else SUF="_seed$S"; fi
    if [ $STOP = 1 ]; then
        note "SKIP seed $S (steps $K-$((K + 2))): identity stop"
        FAILED="$FAILED seed${S}(identity_stop)"
        continue
    fi
    inputs=1
    input "seq_ft_s$S" "$RES/$REF/result.json" || inputs=0
    input "plain_s$S" "$RES/$GPM/result.json" || inputs=0
    input "adaptive_s$S" "$RES/$NE90/result.json" || inputs=0
    input "seqrep_plain_s$S" "$RES/gpm_seq$SUF/report.json" || inputs=0
    input "seqrep_adaptive_s$S" "$RES/gpm_seq_ne90$SUF/report.json" || inputs=0
    if [ $inputs = 0 ]; then
        note "SKIP seed $S (steps $K-$((K + 2))): an input is missing"
        FAILED="$FAILED seed${S}(inputs)"
        continue
    fi

    e99_ok=0; seqrep_ok=0
    run_or_prior $K "e99_s$S" "$RES/$E99/result.json" \
        $PY scripts/run_continual.py "${RUN_ARGS[@]}" --seed "$S" --method gpm_e99 \
        --t1-reference-run "$RES/$REF" --identity-reference-run "$RES/$GPM" --identity-stages 0 1 \
        && e99_ok=1
    if [ $e99_ok = 1 ]; then
        run_or_prior $((K + 1)) "seqrep_s$S" "$RES/gpm_seq_e99_seed$S/report.json" \
            $PY scripts/sequence_report.py --method-run "$RES/$E99" --reference-run "$RES/$REF" \
            --out "$RES/gpm_seq_e99_seed$S/report.json" && seqrep_ok=1
    else
        if want $K && identity_failed "$RES/$E99"; then
            STOP=1
            note "STOP identity failure at seed $S: every later seed is skipped"
        fi
        note "SKIP $((K + 1))-$((K + 2)): ${K}_e99_s$S failed or is missing"
        continue
    fi
    if [ $seqrep_ok = 1 ]; then
        run_or_prior $((K + 2)) "report_s$S" "$RES/high_protection/seed$S/report.json" \
            $PY scripts/high_protection.py --seed "$S" "${REPORT_ARGS[@]}"
    else
        note "SKIP $((K + 2)): $((K + 1))_seqrep_s$S failed or is missing"
    fi
done

if [ -z "$SMOKE" ]; then
    reports=1
    for S in 0 1 2; do [ -e "$RES/high_protection/seed$S/report.json" ] || reports=0; done
    if [ $reports = 1 ]; then
        step 9_replication $PY scripts/high_protection.py --replication
    else
        note "SKIP 9_replication: not every seed has a report"
        FAILED="$FAILED 9_replication(skipped)"
    fi
fi

if [ -z "$FAILED" ]; then
    note "QUEUE DONE ok"
    exit 0
fi
note "QUEUE DONE FAILED:$FAILED"
exit 1
