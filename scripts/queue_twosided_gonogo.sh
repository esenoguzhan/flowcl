#!/usr/bin/env bash
# E1, the two-sided protection go/no-go measurement (seed 0; no training): four captures from
# adaptive GPM's stage checkpoints, then the summary. Orchestration only; the rule is
# configs/analysis/twosided_gonogo.yaml. Run it only when no other GPU job is running.
#
# Usage:    bash scripts/queue_twosided_gonogo.sh [--from-step K]
#   --from-step K  resume: steps < K are not rerun (status from their output on disk). Captures
#                  publish atomically; published ones are re-verified and reused. The summary
#                  always runs.
#
#   0-3   capture task index i (T1-T4) at stage{i}.pt: production input capture, memory replay,
#         grad-enabled Hutchinson output Grams, forward identity
#   4     summary and verdict: ALWAYS runs (exit 0 pass/fail, 3 invalid, 4 incomplete)
#
# Status: every non-zero rc (and every missing prior output) is collected. The last line is
# "QUEUE DONE ok" (exit 0) or "QUEUE DONE FAILED: <steps>" (exit 1).
#
# Launch:   tmux new-session -d -s overnight "bash scripts/queue_twosided_gonogo.sh; exec bash"
# Testing:  QUEUE_DRY_RUN=1, QUEUE_FAIL_STEP=<step name>, QUEUE_LOG_ROOT, QUEUE_RESULTS_ROOT.

set -u
USAGE="usage: queue_twosided_gonogo.sh [--from-step K]"
LAST=4
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

LOGDIR="${QUEUE_LOG_ROOT:-results/logs}/queue_$(date +%Y%m%d_%H%M%S)_twosided_gonogo"
mkdir -p "$LOGDIR"
QUEUE_LOG="$LOGDIR/queue.log"
RES="${QUEUE_RESULTS_ROOT:-results}"
OUT="$RES/twosided_gonogo"
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

PY="uv run python scripts/twosided_gonogo.py"
note "QUEUE START twosided_gonogo from_step=$FROM git $(git rev-parse HEAD) status '$(git status --porcelain | wc -l) changes'"

# shellcheck disable=SC2086
{
for i in 0 1 2 3; do
    run_or_prior $i "capture_t$i" "$OUT/capture/task$i/capture.json" $PY --part capture --task $i
done
step 4_summary $PY --part summary
}

if [ -z "$FAILED" ]; then
    note "QUEUE DONE ok"
    exit 0
fi
note "QUEUE DONE FAILED:$FAILED"
exit 1
