#!/usr/bin/env bash
# The 4-task baseline runs (Workstream B; Track 2's test runs): ewc, lora, replay, consft, inflora and
# projected_lora on seq_hetero for one seed, each paired with seq_ft of that seed, then its strict
# sequence report. Orchestration only; the checks are flowcl/experiments/baseline_queue.py. Settings come
# from the frozen selection bundles (configs/method/*.yaml). Run it only when no other GPU job is running
# (the preflight refuses otherwise).
#
# Usage:    bash scripts/queue_baselines.sh [--seed S] [--from-step K] [--smoke DIR]
#   --seed S       0 (default), 1 or 2. On seed 0 the five bracketed methods' stage 1 is also checked
#                  bitwise against the selected arm's checkpoint; seeds 1-2 have the preflight's config
#                  check instead.
#   --from-step K  resume: steps < K are not rerun; a prior run counts only if check-run passes on it, a
#                  prior report only if it passes strict provenance again (a failure blocks dependents).
#                  A run directory left by an interrupted run must be moved aside first.
#   --smoke DIR    seed 2 (derived thresholds), 200 steps and 1 episode, into DIR (outside results/), on a
#                  dirty tree: a smoke seq_ft and gpm (s0, s1), then every step below against them. The
#                  smoke cannot check stage 1 against the arms.
#
#   0          preflight (clean tree, idle GPU, targets absent, reference/bundle/arm hashes, configs)
#   per method i = 0..5 (k = 1 + 3 i):
#   k+0  run     results/seq_hetero__<M>__seed<S>  (T1 pairing and stage-0 identity against seq_ft)
#   k+1  check   identity stages as expected, passed; projected: Task-1 memory equals GPM's
#   k+2  report  results/baselines_seq/seed<S>/<M>.json (--strict-provenance)
#
# IDENTITY STOP: a run whose identity_stage0.json reads "passed": false means the environment drifted;
# every later method is skipped (logged STOP). A failed stage-1 identity skips only that method's check
# and report. Any other failure skips only that method's dependents.
#
# Status: every non-zero rc, skipped step and missing prior output is collected; the last line is
# "QUEUE DONE ok" (exit 0) or "QUEUE DONE FAILED: <steps>" (exit 1).
#
# Launch:   tmux new-session -d -s baselines "bash scripts/queue_baselines.sh --seed 0; exec bash"
# Testing:  QUEUE_DRY_RUN=1, QUEUE_FAIL_STEP=<step name>, QUEUE_IDENTITY_FAIL=<step name>:<stage>,
#           QUEUE_LOG_ROOT, QUEUE_RESULTS_ROOT.

set -u
USAGE="usage: queue_baselines.sh [--seed S] [--from-step K] [--smoke DIR]"
SEED=0
FROM=0
SMOKE=""
while [ $# -gt 0 ]; do
    case "$1" in
        --seed) SEED="${2:-}"; shift 2 || { echo "$USAGE" >&2; exit 2; } ;;
        --from-step) FROM="${2:-}"; shift 2 || { echo "$USAGE" >&2; exit 2; } ;;
        --smoke) SMOKE="${2:-}"; shift 2 || { echo "$USAGE" >&2; exit 2; } ;;
        *) echo "unknown argument '$1'; $USAGE" >&2; exit 2 ;;
    esac
done
case "$SEED" in 0|1|2) ;; *) echo "--seed must be 0, 1 or 2, got '$SEED'" >&2; exit 2 ;; esac
case "$FROM" in ''|*[!0-9]*) echo "--from-step must be an integer 0..18, got '$FROM'" >&2; exit 2;; esac
if [ "$FROM" -gt 18 ]; then echo "--from-step must be an integer 0..18, got '$FROM'" >&2; exit 2; fi

cd "$(dirname "$0")/.." || { echo "cannot cd to repo root" >&2; exit 1; }
export MUJOCO_GL=egl

CUR=seq_hetero
METHODS="ewc lora replay consft inflora projected_lora"
RUN_ARGS=(--curriculum "$CUR" --amp)
REPORT_ARGS=(--strict-provenance)
CHECK_ARGS=()
PRE_ARGS=()
if [ -n "$SMOKE" ]; then
    case "$SMOKE" in /*) ;; *) echo "--smoke needs an absolute directory" >&2; exit 2 ;; esac
    case "$SMOKE/" in "$(pwd)/results/"*) echo "--smoke must be outside results/" >&2; exit 2 ;; esac
    SEED=2
    RES="$SMOKE"
    mkdir -p "$RES"
    RUN_ARGS+=(--results-root "$RES" --steps-per-task 200 --n-episodes 1 --allow-dirty)
    REPORT_ARGS+=(--allow-dirty)
    CHECK_ARGS+=(--allow-dirty)
    PRE_ARGS+=(--smoke --results-root "$RES")
    TAG="_smoke"
else
    RES="${QUEUE_RESULTS_ROOT:-results}"
    if [ -n "${QUEUE_RESULTS_ROOT:-}" ]; then PRE_ARGS+=(--results-root "$RES"); fi
    TAG=""
fi
REF="$RES/${CUR}__seq_ft__seed$SEED"
GPM="$RES/${CUR}__gpm_projected_adam__seed$SEED"
OUT="$RES/baselines_seq/seed$SEED"

LOGDIR="${QUEUE_LOG_ROOT:-results/logs}/queue_$(date +%Y%m%d_%H%M%S)_baselines_seed${SEED}${TAG}"
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

# identity_failed RUN_DIR STAGE NAME: the run's identity file for STAGE reads "passed": false (in a dry
# run: QUEUE_IDENTITY_FAIL=NAME:STAGE).
identity_failed() {
    local f="$1/identity_stage$2.json"
    if [ "${QUEUE_DRY_RUN:-0}" = "1" ]; then
        [ "${QUEUE_IDENTITY_FAIL:-}" = "$3:$2" ] && { note "IDENTITY FAILED (dry run): $f"; return 0; }
        return 1
    fi
    if [ -e "$f" ] && grep -q '"passed": false' "$f"; then
        note "IDENTITY FAILED: $f"
        return 0
    fi
    return 1
}

PY="uv run python"
note "QUEUE START baselines seed=$SEED smoke='$SMOKE' from_step=$FROM git $(git rev-parse HEAD) status '$(git status --porcelain | wc -l) changes'"

if [ -n "$SMOKE" ]; then
    # The comparators at smoke settings (production uses the existing runs).
    step s0_seqft $PY scripts/run_continual.py "${RUN_ARGS[@]}" --seed $SEED --method seq_ft || true
    step s1_gpm $PY scripts/run_continual.py "${RUN_ARGS[@]}" --seed $SEED --method gpm \
        --t1-reference-run "$REF" || true
fi

STOP=0
if want 0; then
    step 0_preflight $PY scripts/baseline_queue.py preflight --seed $SEED --methods $METHODS "${PRE_ARGS[@]}" \
        --out "$LOGDIR/preflight.json" || STOP=1
    if [ $STOP -eq 1 ]; then note "STOP: the preflight refused the launch"; fi
fi

i=0
for M in $METHODS; do
    k=$((1 + 3 * i))
    i=$((i + 1))
    RUN="$RES/${CUR}__${M}__seed$SEED"
    if [ $STOP -eq 1 ]; then
        note "SKIP ${k}_run_$M ${k}+1, ${k}+2 (stopped)"
        FAILED="$FAILED ${k}_run_$M(stopped)"
        continue
    fi
    ARM_ARGS=()
    if [ "$SEED" = "0" ] && [ "$M" != "replay" ]; then
        ARM=$($PY -c "from flowcl.experiments.baseline_queue import arm_checkpoint; print(arm_checkpoint('$M'))") \
            || { note "cannot resolve the arm of $M"; FAILED="$FAILED ${k}_arm_$M"; continue; }
        ARM_ARGS=(--identity-reference-checkpoint "1=$ARM")
    fi
    MEM_ARGS=()
    case "$M" in inflora|projected_lora) MEM_ARGS=(--gpm-memory "$GPM/method/memory_task0.pt") ;; esac

    # k+0: the run (or, before --from-step, its check on disk)
    ok=1
    if want $k; then
        step "${k}_run_$M" $PY scripts/run_continual.py "${RUN_ARGS[@]}" --seed $SEED --method "$M" \
            --t1-reference-run "$REF" --identity-reference-run "$REF" --identity-stages 0 "${ARM_ARGS[@]}" || ok=0
        if [ $ok -eq 0 ]; then
            if identity_failed "$RUN" 0 "${k}_run_$M"; then
                note "STOP: stage-0 identity failed for $M; every later method is skipped"
                STOP=1
            elif identity_failed "$RUN" 1 "${k}_run_$M"; then
                note "stage-1 identity failed for $M; its check and report are skipped"
            fi
            note "SKIP $((k + 1))_check_$M $((k + 2))_report_$M (run failed)"
            continue
        fi
    fi

    # k+1: the queue's own check of the run (also of a prior run before --from-step)
    step "$((k + 1))_check_$M" $PY scripts/baseline_queue.py check-run --run "$RUN" --method "$M" \
        --seed $SEED "${MEM_ARGS[@]}" --out "$LOGDIR/check_$M.json" \
        || { note "SKIP $((k + 2))_report_$M (check failed)"; continue; }

    # k+2: the strict report (or, before --from-step, the prior report checked again)
    if want $((k + 2)); then
        step "$((k + 2))_report_$M" $PY scripts/sequence_report.py --method-run "$RUN" --reference-run "$REF" \
            "${REPORT_ARGS[@]}" --out "$OUT/$M.json" || true
    else
        step "$((k + 2))_prior_report_$M" $PY scripts/baseline_queue.py check-report --report "$OUT/$M.json" \
            "${CHECK_ARGS[@]}" || true
    fi
done

if [ -z "$FAILED" ]; then
    note "QUEUE DONE ok"
    exit 0
fi
note "QUEUE DONE FAILED:$FAILED"
exit 1
