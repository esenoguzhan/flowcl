#!/usr/bin/env bash
# The similar-vs-diverse two-task continual study: three curricula A -> B (S similar, D diverse, S2
# similar) x three arms (seq_ft, allowlist_ft, plain GPM), the stage-0 extras, the stage-1 probes, and
# the pre-registered summary. Orchestration only; the rule is configs/analysis/similar_cl.yaml. S's
# runs train stage 0 (allowlist_ft and gpm must equal seq_ft there bit for bit); D's and S2's resume at
# stage 1 from S's stage0.pt of the same arm, with its SHA-256 expected. S2 runs last, so it can be
# stopped if the GPU is needed (the summary then reports it incomplete). Run it only when no other GPU
# job is running.
#
# Usage:    bash scripts/queue_similar_cl.sh [--from-step K] [--smoke DIR]
#   --from-step K  resume: steps < K are not rerun (status from their outputs on disk; a missing
#                  output blocks its dependents). A run directory left by an interrupted training
#                  step must be moved aside first.
#   --smoke DIR    the whole pipeline at 200 steps, 1 episode per cell and 3 probe batches into DIR
#                  (outside results/), on a dirty tree; the summary runs with --smoke (scratch only).
#
#    0  S seq_ft                         (~1.6 h)
#    1  stage-0 extras (needs 0)         (zero-shot milk and cabinet, probe losses, instruction probes)
#    2  S allowlist_ft                   (~1.6 h; identity at stage 0 against 0)
#    3  S gpm                            (~1.7 h; T1 pairing and identity at stage 0 against 0)
#    4  D seq_ft        resumed from 0   (~0.85 h each)
#    5  D allowlist_ft  resumed from 2
#    6  D gpm           resumed from 3
#    7  stage-1 probes, S and D
#    8  S2 seq_ft       resumed from 0
#    9  S2 allowlist_ft resumed from 2
#   10  S2 gpm          resumed from 3
#   11  stage-1 probes, S2
#   12  summary (ALWAYS runs)            -> similar_cl/summary.json
#
# Status: every non-zero rc and every missing prior output is collected; the last line is
# "QUEUE DONE ok" (exit 0) or "QUEUE DONE FAILED: <steps>" (exit 1). The summary's own status
# (valid 0 / invalid 3 / incomplete 4) is its rc.
#
# Launch:   tmux new-session -d -s similarcl "bash scripts/queue_similar_cl.sh; exec bash"
# Testing:  QUEUE_DRY_RUN=1, QUEUE_FAIL_STEP=<step name>, QUEUE_LOG_ROOT, QUEUE_RESULTS_ROOT.

set -u
USAGE="usage: queue_similar_cl.sh [--from-step K] [--smoke DIR]"
FROM=0
SMOKE=""
while [ $# -gt 0 ]; do
    case "$1" in
        --from-step) FROM="${2:-}"; shift 2 || { echo "$USAGE" >&2; exit 2; } ;;
        --smoke) SMOKE="${2:-}"; shift 2 || { echo "$USAGE" >&2; exit 2; } ;;
        *) echo "unknown argument '$1'; $USAGE" >&2; exit 2 ;;
    esac
done
case "$FROM" in ''|*[!0-9]*) echo "--from-step must be an integer 0..12, got '$FROM'" >&2; exit 2;; esac
if [ "$FROM" -gt 12 ]; then echo "--from-step must be an integer 0..12, got '$FROM'" >&2; exit 2; fi

cd "$(dirname "$0")/.." || { echo "cannot cd to repo root" >&2; exit 1; }
export MUJOCO_GL=egl

S=similar_ab_stove; D=similar_ab_milk; S2=similar_ab_cabinet
declare -A LABEL=([seq_ft]=seq_ft [allowlist_ft]=allowlist_ft [gpm]=gpm_projected_adam)
run() { echo "${1}__${LABEL[$2]}__seed0"; }   # run directory name: curriculum, arm

RUN_ARGS=(--seed 0 --amp)
CL_ARGS=()
SUMMARY_ARGS=()
if [ -n "$SMOKE" ]; then
    case "$SMOKE" in /*) ;; *) echo "--smoke needs an absolute directory" >&2; exit 2 ;; esac
    case "$SMOKE/" in "$(pwd)/results/"*) echo "--smoke must be outside results/" >&2; exit 2 ;; esac
    RES="$SMOKE"
    mkdir -p "$RES"
    RUN_ARGS+=(--results-root "$RES" --steps-per-task 200 --n-episodes 1 --allow-dirty)
    CL_ARGS+=(--results-root "$RES" --smoke-batches 3 --allow-dirty)
    SUMMARY_ARGS+=(--results-root "$RES" --smoke)
    TAG="_smoke"
else
    RES="${QUEUE_RESULTS_ROOT:-results}"
    CL_ARGS+=(--results-root "$RES")
    SUMMARY_ARGS+=(--results-root "$RES")
    TAG=""
fi

LOGDIR="${QUEUE_LOG_ROOT:-results/logs}/queue_$(date +%Y%m%d_%H%M%S)_similar_cl${TAG}"
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

# the expected SHA-256 of a source's stage0.pt, computed when the resumed step starts
stage0_sha() {
    local f="$RES/$1/checkpoints/stage0.pt"
    if [ -f "$f" ]; then sha256sum "$f" | cut -d' ' -f1
    elif [ "${QUEUE_DRY_RUN:-0}" = "1" ]; then echo "DRYRUN"
    else echo "MISSING"; fi
}

PY="uv run python"
note "QUEUE START similar_cl smoke='$SMOKE' from_step=$FROM git $(git rev-parse HEAD) status '$(git status --porcelain | wc -l) changes' rule $(sha256sum configs/analysis/similar_cl.yaml | cut -c1-12)"

declare -A OK=([seq_ft]=0 [allowlist_ft]=0 [gpm]=0)
run_or_prior 0 S_seqft "$RES/$(run "$S" seq_ft)/result.json" \
    $PY scripts/run_continual.py --curriculum "$S" "${RUN_ARGS[@]}" --method seq_ft && OK[seq_ft]=1

if [ "${OK[seq_ft]}" = 1 ]; then
    REF="$RES/$(run "$S" seq_ft)"
    run_or_prior 1 stage0 "$RES/similar_cl/stage0/stage0.json" \
        $PY scripts/similar_cl.py --part stage0 "${CL_ARGS[@]}"
    run_or_prior 2 S_allow "$RES/$(run "$S" allowlist_ft)/result.json" \
        $PY scripts/run_continual.py --curriculum "$S" "${RUN_ARGS[@]}" --method allowlist_ft \
        --t1-reference-run "$REF" --identity-reference-run "$REF" --identity-stages 0 && OK[allowlist_ft]=1
    run_or_prior 3 S_gpm "$RES/$(run "$S" gpm)/result.json" \
        $PY scripts/run_continual.py --curriculum "$S" "${RUN_ARGS[@]}" --method gpm \
        --t1-reference-run "$REF" --identity-reference-run "$REF" --identity-stages 0 && OK[gpm]=1
else
    note "SKIP 1-11: 0_S_seqft failed or is missing (the summary still runs)"
fi

# resumed CURRICULUM FIRST_STEP SHORT: the three arms of one curriculum, each from S's same arm
resumed() {
    local cur=$1 k=$2 short=$3 arm src
    for arm in seq_ft allowlist_ft gpm; do
        if [ "${OK[$arm]}" = 1 ]; then
            src="$RES/$(run "$S" "$arm")"
            local name
            case $arm in seq_ft) name=seqft ;; allowlist_ft) name=allow ;; gpm) name=gpm ;; esac
            if want "$k"; then
                local sha
                sha=$(stage0_sha "$(run "$S" "$arm")")
                note "RESUME ${k}_${short}_$name from $src/checkpoints/stage0.pt sha256 $sha"
            fi
            run_or_prior "$k" "${short}_$name" "$RES/$(run "$cur" "$arm")/result.json" \
                $PY scripts/run_continual.py --curriculum "$cur" "${RUN_ARGS[@]}" --method "$arm" \
                --resume-run "$src" --start-stage 1 --resume-sha256 "${sha:-UNUSED}"
        else
            note "SKIP $k: S's $arm run failed or is missing"
        fi
        k=$((k + 1))
    done
}

if [ "${OK[seq_ft]}" = 1 ]; then
    resumed "$D" 4 D
    run_or_prior 7 probes_SD "$RES/similar_cl/probes" \
        $PY scripts/similar_cl.py --part probe --curricula "$S" "$D" "${CL_ARGS[@]}"
    resumed "$S2" 8 S2
    run_or_prior 11 probes_S2 "$RES/similar_cl/probes" \
        $PY scripts/similar_cl.py --part probe --curricula "$S2" "${CL_ARGS[@]}"
fi

step 12_summary $PY scripts/similar_cl.py --part summary "${SUMMARY_ARGS[@]}"

if [ -z "$FAILED" ]; then
    note "QUEUE DONE ok"
    exit 0
fi
note "QUEUE DONE FAILED:$FAILED"
exit 1
