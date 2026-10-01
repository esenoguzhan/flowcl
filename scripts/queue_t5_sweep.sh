#!/usr/bin/env bash
# T5 saturation stress test (seed 0; exploratory): the replay of adaptive GPM's memory under
# SGP's rule (a gate), seven T5 arms from adaptive GPM's post-T4 state, the summary.
# Orchestration only; the rule is configs/analysis/t5_sweep.yaml. Run it only when no other GPU
# job is running. No step launches a T6-T8 run.
#
# Usage:    bash scripts/queue_t5_sweep.sh [--from-step K]
#   --from-step K  resume: steps < K are not rerun (status from their output on disk). The replay
#                  and the arms publish atomically; published ones are re-verified and reused,
#                  abandoned attempt directories are ignored. The gate step (1) always runs.
#
#   0     replay: adaptive GPM's memory rebuilt with SGP's rule, checked, sigma' recorded,
#         T5's start Gram (~1 h)
#   1     replay_gate: did the replay reproduce the memory?      no -> 2-8 skipped
#   2-8   arms hard, free, sgp_a25, sgp_a100, sgp_a400, sgp_a800, sgp_a1600 (~75 min each)
#   9     summary and verdict: ALWAYS runs
#
# Launch:   tmux new-session -d -s t5sweep "bash scripts/queue_t5_sweep.sh; exec bash"
# Testing:  QUEUE_DRY_RUN=1, QUEUE_FAIL_STEP=<step name>, QUEUE_LOG_ROOT, QUEUE_RESULTS_ROOT.

set -u
USAGE="usage: queue_t5_sweep.sh [--from-step K]"
LAST=9
FROM=0
while [ $# -gt 0 ]; do
    case "$1" in
        --from-step) FROM="${2:-}"; shift 2 || { echo "$USAGE" >&2; exit 2; } ;;
        *) echo "unknown argument '$1'; $USAGE" >&2; exit 2 ;;
    esac
done
case "$FROM" in ''|*[!0-9]*) echo "--from-step must be an integer 0..$LAST, got '$FROM'" >&2; exit 2;; esac
if [ "$FROM" -gt $LAST ]; then echo "--from-step must be an integer 0..$LAST, got '$FROM'" >&2; exit 2; fi

cd "$(dirname "$0")/.." || { echo "cannot cd to repo root" >&2; exit 1; }
export MUJOCO_GL=egl

LOGDIR="${QUEUE_LOG_ROOT:-results/logs}/queue_$(date +%Y%m%d_%H%M%S)_t5_sweep"
mkdir -p "$LOGDIR"
QUEUE_LOG="$LOGDIR/queue.log"
RES="${QUEUE_RESULTS_ROOT:-results}"
OUT="$RES/t5_sweep"

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

want() { [ "$1" -ge "$FROM" ]; }

prior() {
    local name=$1 path=$2
    if [ -e "$path" ]; then note "PRIOR $name ok: $path"; return 0; fi
    note "PRIOR $name missing: $path"
    return 1
}

run_or_prior() {
    local k=$1 name=$2 path=$3; shift 3
    if want "$k"; then step "${k}_$name" "$@"
    else prior "${k}_$name" "$path"; fi
}

PY="uv run python scripts/t5_sweep.py"
note "QUEUE START t5 sweep from_step=$FROM git $(git rev-parse HEAD) status '$(git status --porcelain | wc -l) changes'"

# shellcheck disable=SC2086
{
run_or_prior 0 replay "$OUT/replay/replay.json" $PY --part replay

gate_ok=0
step 1_replay_gate $PY --part replay_gate && gate_ok=1
if [ $gate_ok = 1 ]; then
    i=2
    for a in hard free sgp_a25 sgp_a100 sgp_a400 sgp_a800 sgp_a1600; do
        run_or_prior $i "arm_$a" "$OUT/arms/t5__${a}__seed0/row.json" $PY --part arm --arm $a
        i=$((i + 1))
    done
else
    note "SKIP 2-8: the replay gate is closed (verdict replay_invalid); the summary still runs"
fi
step 9_summary $PY --part summary
}

note "QUEUE DONE"
