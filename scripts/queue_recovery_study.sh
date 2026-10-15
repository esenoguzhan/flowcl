#!/usr/bin/env bash
# The recovery measurement study (configs/analysis/recovery_study.yaml). Orchestration only.
#
# Usage:  bash scripts/queue_recovery_study.sh --smoke DIR
#         bash scripts/queue_recovery_study.sh --workers N [--fallback] --suite-result FILE
#   --smoke DIR     into DIR (outside results/): the sim tests (run_episode equals rollout(); the interior
#                   suffix reproduced from the saved generator state; order invariance), per-phase timing at
#                   an early and a late query (benchmark, task 0), the throughput of 1, 2 and 4 concurrent
#                   benchmark workers, and a smoke worker on task 0 (2 nominal rollouts, an early and a late
#                   query, 2 seeds; the determinism gate) with a dry-run analysis.
#   production      requires results/recovery_study/validation/validation.json to allow production at the
#                   rule's decision level; N task workers in parallel over the 4 tasks (--fallback: the
#                   rule's frozen 4-query fallback, chosen from the smoke's estimate and logged), then the
#                   analysis, published only if FILE reports the CPU suite green at this commit.
# Status: the last line is "QUEUE DONE ok" (exit 0) or "QUEUE DONE FAILED: <steps>" (exit 1).
# Testing: QUEUE_DRY_RUN=1, QUEUE_FAIL_STEP, QUEUE_LOG_ROOT, QUEUE_RESULTS_ROOT.

set -u
USAGE="usage: queue_recovery_study.sh --smoke DIR | --workers N [--fallback] --suite-result FILE"
SMOKE=""; WORKERS=""; FALLBACK=""; SUITE=""
while [ $# -gt 0 ]; do
    case "$1" in
        --smoke) SMOKE="${2:-}"; shift 2 || { echo "$USAGE" >&2; exit 2; } ;;
        --workers) WORKERS="${2:-}"; shift 2 || { echo "$USAGE" >&2; exit 2; } ;;
        --fallback) FALLBACK="--fallback"; shift ;;
        --suite-result) SUITE="${2:-}"; shift 2 || { echo "$USAGE" >&2; exit 2; } ;;
        *) echo "unknown argument '$1'; $USAGE" >&2; exit 2 ;;
    esac
done
cd "$(dirname "$0")/.." || exit 1
export MUJOCO_GL=egl
if [ -n "$SMOKE" ]; then
    case "$SMOKE" in /*) ;; *) echo "--smoke needs an absolute directory" >&2; exit 2 ;; esac
    case "$SMOKE/" in "$(pwd)/results/"*) echo "--smoke must be outside results/" >&2; exit 2 ;; esac
    mkdir -p "$SMOKE"; TAG="_smoke"
else
    case "$WORKERS" in ''|*[!0-9]*) echo "--workers N (1-4) is required; $USAGE" >&2; exit 2 ;; esac
    if [ "$WORKERS" -lt 1 ] || [ "$WORKERS" -gt 4 ]; then echo "--workers must be 1-4" >&2; exit 2; fi
    case "$SUITE" in /*) ;; *) echo "--suite-result needs an absolute path; $USAGE" >&2; exit 2 ;; esac
    TAG=""
fi
RES="${QUEUE_RESULTS_ROOT:-results}"
LOGDIR="${QUEUE_LOG_ROOT:-results/logs}/queue_$(date +%Y%m%d_%H%M%S)_recovery${TAG}"
mkdir -p "$LOGDIR"; QUEUE_LOG="$LOGDIR/queue.log"; FAILED=""
note() { echo "$(date '+%F %T') $*" | tee -a "$QUEUE_LOG"; }
step() {
    local name=$1; shift
    note "START $name: $*"
    local rc
    if [ "${QUEUE_DRY_RUN:-0}" = "1" ]; then
        echo "$*" > "$LOGDIR/$name.log"; rc=0
        if [ "${QUEUE_FAIL_STEP:-}" = "$name" ]; then rc=1; fi
    else
        "$@" > "$LOGDIR/$name.log" 2>&1; rc=$?
    fi
    note "END $name rc=$rc"
    return $rc
}
fail() { FAILED="$FAILED $1"; }
parallel() {   # parallel <name> <cmd...> per task: one process per given task index, wait for all
    local name=$1; shift
    local pids="" k
    for k in $TASKS; do
        step "${name}_t$k" "$@" "$k" &
        pids="$pids $!"
        while [ "$(jobs -rp | wc -l)" -ge "$SLOTS" ]; do sleep 5; done
    done
    local bad=0 p
    for p in $pids; do wait "$p" || bad=1; done
    return $bad
}
PY="uv run python"
RS=(scripts/recovery_study.py)
note "QUEUE START recovery smoke='$SMOKE' workers='$WORKERS' $FALLBACK git $(git rev-parse HEAD) status '$(git status --porcelain | wc -l) changes'"

if [ -n "$SMOKE" ]; then
    step s0_sim_tests uv run pytest -q -m sim tests/test_recovery_sim.py -p no:cacheprovider || fail s0_sim_tests
    step s1_benchmark $PY "${RS[@]}" --benchmark 0 --out "$SMOKE/benchmark_t0.json" || fail s1_benchmark
    for N in 1 2 4; do
        TASKS="$(seq 0 $((N - 1)) | tr '\n' ' ')"; SLOTS=$N
        T0=$(date +%s)
        parallel "s2_concurrency${N}" $PY "${RS[@]}" --benchmark || fail "s2_concurrency$N"   # JSON in each step log
        note "THROUGHPUT workers=$N wall=$(( $(date +%s) - T0 ))s"
    done
    step s3_smoke_worker $PY "${RS[@]}" --task 0 --smoke --out-root "$SMOKE" || fail s3_smoke_worker
    step s4_dry_analysis $PY "${RS[@]}" --analyze --out-root "$SMOKE" || fail s4_dry_analysis
else
    step p0_validation $PY - <<'EOF' || fail p0_validation
import json, sys
from flowcl.experiments import recovery_study as rs
rule = rs.load_rule()
v = json.load(open(rs._root() / rule["out"] / "validation" / "validation.json"))
d = v["decision"]
ok = (d["production_may_start"] and float(d["level"]) == float(rule["decision"]["level"])
      and v["rule_sha256"] == rs.file_sha256(rs.rule_path()) and not v["git_sha"].endswith("-dirty"))
print(json.dumps(d), "rule level", rule["decision"]["level"])
sys.exit(0 if ok else 1)
EOF
    if [ -z "$FAILED" ]; then
        TASKS="0 1 2 3"; SLOTS=$WORKERS
        parallel p1_worker $PY "${RS[@]}" $FALLBACK --task || fail p1_workers
        step p2_analyze $PY "${RS[@]}" --analyze --suite-result "$SUITE" || fail p2_analyze
    else
        note "SKIP workers: the validation does not allow production at the rule's level"
    fi
fi
if [ -n "$FAILED" ]; then note "QUEUE DONE FAILED:$FAILED"; exit 1; fi
note "QUEUE DONE ok"; exit 0
