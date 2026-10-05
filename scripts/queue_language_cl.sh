#!/usr/bin/env bash
# The language-conditioned base study: the langbase_cabinet_stove curriculum (stage 0: A co-trained
# with S2 for 2 x steps; stage 1: S) under seq_ft and plain GPM, the swap rollouts, the probes and the
# pre-registered summary. Orchestration only; the rule is configs/analysis/language_cl.yaml. GPM must
# equal seq_ft at stage 0 bit for bit (identity check) and passes its T1 pairing. Run it only when no
# other GPU job is running.
#
# Usage:    bash scripts/queue_language_cl.sh [--from-step K] [--smoke DIR]
#   --from-step K  resume: steps < K are not rerun (status from their outputs on disk; a missing
#                  output blocks its dependents). A run directory left by an interrupted training
#                  step must be moved aside first.
#   --smoke DIR    the whole pipeline at 200 steps (400 at the joint stage), 1 episode per cell and 3
#                  probe batches into DIR (outside results/), on a dirty tree; the A-only base is read
#                  from results/; the summary runs with --smoke (scratch only).
#
#    0  seq_ft                                 (~2.7 h)
#    1  swaps, stage 0, seq_ft (needs 0)       (<= 35 min)
#    2  gpm (identity at stage 0 against 0)    (~2.8 h)
#    3  swaps, stage 1, seq_ft (needs 0)       (<= 20 min)
#    4  swaps, stage 1, gpm (needs 2)          (<= 20 min)
#    5  probes (four checkpoints)              (~30 min)
#    6  summary (ALWAYS runs)                  -> language_cl/summary.json
#
# Status: every non-zero rc and every missing prior output is collected; the last line is
# "QUEUE DONE ok" (exit 0) or "QUEUE DONE FAILED: <steps>" (exit 1). The summary's own status
# (valid 0 / invalid 3 / incomplete 4) is its rc.
#
# Launch:   tmux new-session -d -s langcl "bash scripts/queue_language_cl.sh; exec bash"
# Testing:  QUEUE_DRY_RUN=1, QUEUE_FAIL_STEP=<step name>, QUEUE_LOG_ROOT, QUEUE_RESULTS_ROOT.

set -u
USAGE="usage: queue_language_cl.sh [--from-step K] [--smoke DIR]"
FROM=0
SMOKE=""
while [ $# -gt 0 ]; do
    case "$1" in
        --from-step) FROM="${2:-}"; shift 2 || { echo "$USAGE" >&2; exit 2; } ;;
        --smoke) SMOKE="${2:-}"; shift 2 || { echo "$USAGE" >&2; exit 2; } ;;
        *) echo "unknown argument '$1'; $USAGE" >&2; exit 2 ;;
    esac
done
case "$FROM" in ''|*[!0-9]*) echo "--from-step must be an integer 0..6, got '$FROM'" >&2; exit 2;; esac
if [ "$FROM" -gt 6 ]; then echo "--from-step must be an integer 0..6, got '$FROM'" >&2; exit 2; fi

cd "$(dirname "$0")/.." || { echo "cannot cd to repo root" >&2; exit 1; }
export MUJOCO_GL=egl

CUR=langbase_cabinet_stove
declare -A LABEL=([seq_ft]=seq_ft [gpm]=gpm_projected_adam)
run() { echo "${CUR}__${LABEL[$1]}__seed0"; }   # run directory name: arm

RUN_ARGS=(--curriculum "$CUR" --seed 0 --amp)
CL_ARGS=()
SUMMARY_ARGS=()
if [ -n "$SMOKE" ]; then
    case "$SMOKE" in /*) ;; *) echo "--smoke needs an absolute directory" >&2; exit 2 ;; esac
    case "$SMOKE/" in "$(pwd)/results/"*) echo "--smoke must be outside results/" >&2; exit 2 ;; esac
    RES="$SMOKE"
    mkdir -p "$RES"
    RUN_ARGS+=(--results-root "$RES" --steps-per-task 200 --n-episodes 1 --allow-dirty)
    CL_ARGS+=(--results-root "$RES" --allow-dirty)
    PROBE_ARGS=(--a-only-root "$(pwd)/results" --smoke-batches 3)
    SUMMARY_ARGS+=(--results-root "$RES" --a-only-root "$(pwd)/results" --smoke)
    TAG="_smoke"
else
    RES="${QUEUE_RESULTS_ROOT:-results}"
    CL_ARGS+=(--results-root "$RES")
    PROBE_ARGS=()
    SUMMARY_ARGS+=(--results-root "$RES")
    TAG=""
fi

LOGDIR="${QUEUE_LOG_ROOT:-results/logs}/queue_$(date +%Y%m%d_%H%M%S)_language_cl${TAG}"
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

PY="uv run python"
note "QUEUE START language_cl smoke='$SMOKE' from_step=$FROM git $(git rev-parse HEAD) status '$(git status --porcelain | wc -l) changes' rule $(sha256sum configs/analysis/language_cl.yaml | cut -c1-12) curriculum $(sha256sum configs/curriculum/$CUR.yaml | cut -c1-12)"

SEQ_OK=0; GPM_OK=0
run_or_prior 0 seqft "$RES/$(run seq_ft)/result.json" \
    $PY scripts/run_continual.py "${RUN_ARGS[@]}" --method seq_ft && SEQ_OK=1

if [ "$SEQ_OK" = 1 ]; then
    REF="$RES/$(run seq_ft)"
    run_or_prior 1 swaps0 "$RES/language_cl/swaps/stage0_seq_ft/swaps.json" \
        $PY scripts/language_cl.py --part swaps --stage 0 --arm seq_ft "${CL_ARGS[@]}"
    run_or_prior 2 gpm "$RES/$(run gpm)/result.json" \
        $PY scripts/run_continual.py "${RUN_ARGS[@]}" --method gpm \
        --t1-reference-run "$REF" --identity-reference-run "$REF" --identity-stages 0 && GPM_OK=1
    run_or_prior 3 swaps1_seqft "$RES/language_cl/swaps/stage1_seq_ft/swaps.json" \
        $PY scripts/language_cl.py --part swaps --stage 1 --arm seq_ft "${CL_ARGS[@]}"
    if [ "$GPM_OK" = 1 ]; then
        run_or_prior 4 swaps1_gpm "$RES/language_cl/swaps/stage1_gpm/swaps.json" \
            $PY scripts/language_cl.py --part swaps --stage 1 --arm gpm "${CL_ARGS[@]}"
    else
        note "SKIP 4: 2_gpm failed or is missing"
    fi
    run_or_prior 5 probes "$RES/language_cl/probes" \
        $PY scripts/language_cl.py --part probe "${CL_ARGS[@]}" "${PROBE_ARGS[@]}"
else
    note "SKIP 1-5: 0_seqft failed or is missing (the summary still runs)"
fi

step 6_summary $PY scripts/language_cl.py --part summary "${SUMMARY_ARGS[@]}"

if [ -z "$FAILED" ]; then
    note "QUEUE DONE ok"
    exit 0
fi
note "QUEUE DONE FAILED:$FAILED"
exit 1
