#!/usr/bin/env bash
# E1-B, the angle-B capacity gate (seed 0; no training): four captures of the integrated executed
# prefix from adaptive GPM's stage checkpoints, then the summary. Orchestration only; the rule is
# configs/analysis/twosided_angleb.yaml. Run it only when no other GPU job is running.
#
# Usage:    bash scripts/queue_twosided_angleb.sh [--smoke DIR]
#   --smoke DIR  into DIR (absolute, outside results/): task 0's first 3 batches (time per batch,
#                peak GPU memory, the identity check), then the summary on it (exit 4, incomplete,
#                is the expected status).
#   production   captures c0-c3 (each independent; a published capture is re-verified and reused,
#                so a rerun resumes), then the summary, which ALWAYS runs (exit 0 pass/fail,
#                3 invalid, 4 incomplete).
#
# Status: every unexpected rc is collected. The last line is "QUEUE DONE ok" (exit 0) or
# "QUEUE DONE FAILED: <steps>" (exit 1).
#
# Launch:   tmux new-session -d -s angleb "bash scripts/queue_twosided_angleb.sh; exec bash"
# Testing:  QUEUE_DRY_RUN=1, QUEUE_FAIL_STEP=<step name>, QUEUE_LOG_ROOT.

set -u
USAGE="usage: queue_twosided_angleb.sh [--smoke DIR]"
SMOKE=""
while [ $# -gt 0 ]; do
    case "$1" in
        --smoke) SMOKE="${2:-}"; shift 2 || { echo "$USAGE" >&2; exit 2; } ;;
        *) echo "unknown argument '$1'; $USAGE" >&2; exit 2 ;;
    esac
done
cd "$(dirname "$0")/.." || { echo "cannot cd to repo root" >&2; exit 1; }
if [ -n "$SMOKE" ]; then
    case "$SMOKE" in /*) ;; *) echo "--smoke needs an absolute directory" >&2; exit 2 ;; esac
    case "$SMOKE/" in "$(pwd)/results/"*) echo "--smoke must be outside results/" >&2; exit 2 ;; esac
    TAG="_smoke"
else
    TAG=""
fi

LOGDIR="${QUEUE_LOG_ROOT:-results/logs}/queue_$(date +%Y%m%d_%H%M%S)_twosided_angleb${TAG}"
mkdir -p "$LOGDIR"
QUEUE_LOG="$LOGDIR/queue.log"
FAILED=""

note() { echo "$(date '+%F %T') $*" | tee -a "$QUEUE_LOG"; }

step() {   # step <name> <expected rc> <cmd...>
    local name=$1 want=$2; shift 2
    note "START $name: $*"
    local rc
    if [ "${QUEUE_DRY_RUN:-0}" = "1" ]; then
        echo "$*" > "$LOGDIR/$name.log"
        if [ "${QUEUE_FAIL_STEP:-}" = "$name" ]; then rc=1; else rc=$want; fi
    else
        "$@" > "$LOGDIR/$name.log" 2>&1
        rc=$?
    fi
    note "END $name rc=$rc"
    if [ "$rc" -ne "$want" ]; then FAILED="$FAILED $name"; fi
}

PY="uv run python scripts/twosided_angleb.py"
note "QUEUE START twosided_angleb smoke='$SMOKE' git $(git rev-parse HEAD) status '$(git status --porcelain | wc -l) changes'"

# shellcheck disable=SC2086
if [ -n "$SMOKE" ]; then
    mkdir -p "$SMOKE"
    step s0 0 $PY --part capture --task 0 --smoke-batches 3 --results-root "$SMOKE" --allow-dirty
    step s1_summary 4 $PY --part summary --results-root "$SMOKE"
else
    for i in 0 1 2 3; do
        step "c$i" 0 $PY --part capture --task $i
    done
    step summary 0 $PY --part summary
fi

if [ -z "$FAILED" ]; then
    note "QUEUE DONE ok"
    exit 0
fi
note "QUEUE DONE FAILED:$FAILED"
exit 1
