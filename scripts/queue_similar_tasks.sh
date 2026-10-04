#!/usr/bin/env bash
# The similar-task diagnostic: train task A (Gate 0 recipe), build A's memory, measure every
# registered task B at A's checkpoint, then the summary. Orchestration only; the rule is
# configs/analysis/similar_tasks.yaml. No continual training. Run it only when no other GPU job is
# running.
#
# Usage:    bash scripts/queue_similar_tasks.sh [--from-step K] [--smoke DIR]
#   --from-step K  resume: steps < K are not rerun (status from their outputs on disk).
#   --smoke DIR    A at 200 steps and 1 episode, every measurement at 3 batches, into DIR (outside
#                  results/), on a dirty tree (scratch only, discarded).
#
#   0  train A: scripts/gate0.py --tasks <A> --amp --out-dir <res>/gate0_similar   (~45 min)
#   1  A's memory, self-control, recapture, Gate 3 on A (A's gradients)
#   2+ one step per task B (input overlap, Gate 3, alignment, instruction swaps)
#   last  summary (ALWAYS runs; exit 4 if incomplete)
#
# Status: the last line is "QUEUE DONE ok" (exit 0) or "QUEUE DONE FAILED: <steps>" (exit 1).
# Launch:   tmux new-session -d -s similar "bash scripts/queue_similar_tasks.sh; exec bash"
# Testing:  QUEUE_DRY_RUN=1, QUEUE_FAIL_STEP=<step name>, QUEUE_LOG_ROOT, QUEUE_RESULTS_ROOT.

set -u
USAGE="usage: queue_similar_tasks.sh [--from-step K] [--smoke DIR]"
FROM=0
SMOKE=""
while [ $# -gt 0 ]; do
    case "$1" in
        --from-step) FROM="${2:-}"; shift 2 || { echo "$USAGE" >&2; exit 2; } ;;
        --smoke) SMOKE="${2:-}"; shift 2 || { echo "$USAGE" >&2; exit 2; } ;;
        *) echo "unknown argument '$1'; $USAGE" >&2; exit 2 ;;
    esac
done
case "$FROM" in ''|*[!0-9]*) echo "--from-step must be a non-negative integer, got '$FROM'" >&2; exit 2;; esac

cd "$(dirname "$0")/.." || { echo "cannot cd to repo root" >&2; exit 1; }
export MUJOCO_GL=egl

PY="uv run python"
TASK_A=$($PY -c "from flowcl.experiments.similar_tasks import load_config; print(load_config()['task_a'])") \
    || { echo "cannot read the config" >&2; exit 2; }
mapfile -t TASKS_B < <($PY scripts/similar_tasks.py --part tasks)
[ "${#TASKS_B[@]}" -gt 0 ] || { echo "no B tasks in the config" >&2; exit 2; }
LAST=$(( ${#TASKS_B[@]} + 2 ))
if [ "$FROM" -gt $LAST ]; then echo "--from-step must be 0..$LAST, got '$FROM'" >&2; exit 2; fi

GATE0_ARGS=(--seed 0 --batch-size 64 --lr 1e-4 --amp --tasks "$TASK_A")
ST_ARGS=()
if [ -n "$SMOKE" ]; then
    case "$SMOKE" in /*) ;; *) echo "--smoke needs an absolute directory" >&2; exit 2 ;; esac
    case "$SMOKE/" in "$(pwd)/results/"*) echo "--smoke must be outside results/" >&2; exit 2 ;; esac
    RES="$SMOKE"; mkdir -p "$RES"
    GATE0_ARGS+=(--train-steps 200 --n-episodes 1 --results-root "$RES" --out-dir "$RES/gate0_similar")
    ST_ARGS+=(--results-root "$RES" --smoke-batches 3 --allow-dirty)
    TAG="_smoke"
else
    RES="${QUEUE_RESULTS_ROOT:-results}"
    GATE0_ARGS+=(--train-steps 30000 --out-dir "$RES/gate0_similar")
    TAG=""
fi
A_RUN="$RES/single__${TASK_A//\//__}__seed0"

LOGDIR="${QUEUE_LOG_ROOT:-results/logs}/queue_$(date +%Y%m%d_%H%M%S)_similar_tasks${TAG}"
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

note "QUEUE START similar_tasks smoke='$SMOKE' from_step=$FROM git $(git rev-parse HEAD) status '$(git status --porcelain | wc -l) changes'"

a_ok=0; mem_ok=0
run_or_prior 0 train_a "$A_RUN/checkpoints/final.pt" $PY scripts/gate0.py "${GATE0_ARGS[@]}" && a_ok=1
if [ $a_ok = 1 ]; then
    run_or_prior 1 memory "$RES/similar_tasks/memory/memory.json" \
        $PY scripts/similar_tasks.py --part memory "${ST_ARGS[@]}" && mem_ok=1
else
    note "SKIP 1-$((LAST - 1)): 0_train_a failed or is missing (the summary still runs)"
fi
if [ $mem_ok = 1 ]; then
    k=2
    for b in "${TASKS_B[@]}"; do
        run_or_prior $k "measure_$(echo "$b" | cut -d/ -f2 | cut -c1-40)" "$RES/similar_tasks/tasks/${b//\//__}/result.json" \
            $PY scripts/similar_tasks.py --part measure --task "$b" "${ST_ARGS[@]}"
        k=$((k + 1))
    done
elif [ $a_ok = 1 ]; then
    note "SKIP 2-$((LAST - 1)): 1_memory failed or is missing (the summary still runs)"
fi
step "${LAST}_summary" $PY scripts/similar_tasks.py --part summary "${ST_ARGS[@]:0:2}"

if [ -z "$FAILED" ]; then
    note "QUEUE DONE ok"
    exit 0
fi
note "QUEUE DONE FAILED:$FAILED"
exit 1
