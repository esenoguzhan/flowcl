#!/usr/bin/env bash
# C1 follow-up: the stage-2 checkpoints of e99 and adaptive GPM, seeds 0-2, evaluated on Object's 50 fixed
# initial states in 3 repetitions (rep 0 = the C1 namespace, i.e. C1's own episodes; rep r = namespace
# __rep<r>, fresh sampling noise), then the pre-registered seed reports and the replication.
# Orchestration only; the rule is configs/analysis/high_protection_followup.yaml. Run it only when no
# other GPU job is running.
#
# Usage:    bash scripts/queue_high_protection_followup.sh [--from-step K] [--smoke DIR]
#   --from-step K  resume: steps < K are not rerun (status from their outputs on disk).
#   --smoke DIR    seed 0 only, repetitions 0-1, 3 episodes, outputs into DIR (outside results/); runs and
#                  C1 reports are read from results/. No replication.
#
#   per seed S (k = 7 * index of S in 0 1 2), repetition r in 0 1 2:
#   k+2r    e99 evaluation       -> high_protection_followup/seed<S>/e99.rep<r>.eval.json      (~6 min)
#   k+2r+1  adaptive evaluation  -> high_protection_followup/seed<S>/adaptive.rep<r>.eval.json (~6 min)
#   k+6     seed report          -> high_protection_followup/seed<S>/report.json
#   21      replication          -> high_protection_followup/replication.json (all three reports exist)
#
# Status: the last line is "QUEUE DONE ok" (exit 0) or "QUEUE DONE FAILED: <steps>" (exit 1). A seed
# report exits 3 for an invalid verdict, so it is listed as failed.
#
# Launch:   tmux new-session -d -s hpfollow "bash scripts/queue_high_protection_followup.sh; exec bash"
# Testing:  QUEUE_DRY_RUN=1, QUEUE_FAIL_STEP=<step name>, QUEUE_LOG_ROOT, QUEUE_RESULTS_ROOT.

set -u
USAGE="usage: queue_high_protection_followup.sh [--from-step K] [--smoke DIR]"
FROM=0
SMOKE=""
while [ $# -gt 0 ]; do
    case "$1" in
        --from-step) FROM="${2:-}"; shift 2 || { echo "$USAGE" >&2; exit 2; } ;;
        --smoke) SMOKE="${2:-}"; shift 2 || { echo "$USAGE" >&2; exit 2; } ;;
        *) echo "unknown argument '$1'; $USAGE" >&2; exit 2 ;;
    esac
done
case "$FROM" in ''|*[!0-9]*) echo "--from-step must be an integer 0..21, got '$FROM'" >&2; exit 2;; esac
if [ "$FROM" -gt 21 ]; then echo "--from-step must be an integer 0..21, got '$FROM'" >&2; exit 2; fi

cd "$(dirname "$0")/.." || { echo "cannot cd to repo root" >&2; exit 1; }
export MUJOCO_GL=egl

CUR=seq_hetero
TASK=libero_object/pick_up_the_milk_and_place_it_in_the_basket   # the rule's task_key (tested)
RUNS="${QUEUE_RESULTS_ROOT:-results}"
REPORT_ARGS=()
if [ -n "$SMOKE" ]; then
    case "$SMOKE" in /*) ;; *) echo "--smoke needs an absolute directory" >&2; exit 2 ;; esac
    case "$SMOKE/" in "$(pwd)/results/"*) echo "--smoke must be outside results/" >&2; exit 2 ;; esac
    OUT="$SMOKE"
    mkdir -p "$OUT"
    EP_ARGS=(--n-episodes 3)
    REPS="0 1"
    SEEDS="0"
    REPORT_ARGS+=(--results-root "$OUT" --smoke)
    TAG="_smoke"
else
    OUT="$RUNS"
    EP_ARGS=()       # the protocol's 50 episodes = the task's 50 fixed initial states
    REPS="0 1 2"
    SEEDS="0 1 2"
    TAG=""
fi

LOGDIR="${QUEUE_LOG_ROOT:-results/logs}/queue_$(date +%Y%m%d_%H%M%S)_high_protection_followup${TAG}"
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

run_or_prior() {
    local k=$1 name=$2 path=$3; shift 3
    if want "$k"; then step "${k}_$name" "$@"
    else prior "${k}_$name" "$path"; fi
}

PY="uv run python"
note "QUEUE START high_protection_followup smoke='$SMOKE' from_step=$FROM git $(git rev-parse HEAD) status '$(git status --porcelain | wc -l) changes'"

for S in $SEEDS; do
    case "$S" in 0) K=0 ;; 1) K=7 ;; 2) K=14 ;; esac
    D="$OUT/high_protection_followup/seed$S"   # evaluate.py creates it
    ok=1
    for R in $REPS; do
        if [ "$R" = 0 ]; then NS=${CUR}__seq_ft__seed$S; else NS=${CUR}__seq_ft__seed${S}__rep$R; fi
        for pair in "0 e99 ${CUR}__gpm_projected_adam_e99__seed$S" "1 adaptive ${CUR}__gpm_projected_adam_ne90__seed$S"; do
            set -- $pair
            run_or_prior $((K + 2 * R + $1)) "eval_${2}_rep${R}_s$S" "$D/$2.rep$R.eval.json" \
                $PY scripts/evaluate.py --checkpoint "$RUNS/$3/checkpoints/stage2.pt" --tasks "$TASK" \
                --run-id "$NS" "${EP_ARGS[@]}" --out "$D/$2.rep$R.eval.json" || ok=0
        done
    done
    if [ $ok = 1 ]; then
        run_or_prior $((K + 6)) "report_s$S" "$D/report.json" \
            $PY scripts/high_protection_followup.py --seed "$S" "${REPORT_ARGS[@]}"
    else
        note "SKIP $((K + 6))_report_s$S: an evaluation failed or is missing"
    fi
done

if [ -z "$SMOKE" ]; then
    reports=1
    for S in 0 1 2; do [ -e "$OUT/high_protection_followup/seed$S/report.json" ] || reports=0; done
    if [ $reports = 1 ]; then
        step 21_replication $PY scripts/high_protection_followup.py --replication
    else
        note "SKIP 21_replication: not every seed has a report"
        FAILED="$FAILED 21_replication(skipped)"
    fi
fi

if [ -z "$FAILED" ]; then
    note "QUEUE DONE ok"
    exit 0
fi
note "QUEUE DONE FAILED:$FAILED"
exit 1
