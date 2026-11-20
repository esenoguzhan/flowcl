#!/usr/bin/env bash
# The allocation-rules pilot, seed 0 (rule configs/analysis/allocation_rules.yaml): three allocation rules
# (gpm_alloc_novelty, gpm_alloc_size, gpm_alloc_room) at e99's capacity after every task, on seq_hetero,
# against the existing flat-0.99 run (gpm_projected_adam_e99). Orchestration only; the checks are
# flowcl/experiments/alloc_queue.py (preflight) and flowcl/experiments/allocation_rules.py (check, summary).
# Run it only when no other GPU job is running (a second process could perturb cuDNN and break the bitwise
# identity the checks rely on).
#
# Usage:    bash scripts/queue_alloc.sh [--from-step K] [--smoke DIR --smoke-refs REFS]
#   --from-step K   resume: steps < K are not rerun (status from their outputs on disk; a missing output
#                   blocks its dependents). A run directory left by an interrupted training step must be
#                   moved aside first.
#   --smoke DIR     seed 2 only, 200 steps and 1 episode per cell, into DIR (outside results/), on a dirty
#                   tree, against the smoke references in REFS (seq_ft and plain GPM, seed 2). A smoke first
#                   trains its own e99 (the comparator) and writes method configs with that run's counts.
#
#    0      preflight       (clean tree, rule committed, idle GPU, disk, targets absent, configs, pins,
#                            archived e99 checks, the offline prediction against the registered ranks)
#    1-3    e99 evaluations on stage 3: rep 0 (all 4 tasks, the reproduction gate), rep 1, rep 2 (~1.2 h)
#    per arm a = novelty, size, room (K = 4 + 6 * index):
#      K+0  run              (~4.8 h; T1 pairing against seq_ft, identity against plain GPM at stages 0-1)
#      K+1  check            (identity, T1 memory, comparator config, matching, post-T2 prediction)
#      K+2  strict report    -> gpm_seq_alloc_<a>_seed<S>/report.json
#      K+3  rep 0 on T4      (the reproduction gate)
#      K+4  rep 1, K+5 rep 2 (all 4 tasks)
#    22     summary          (ALWAYS runs) -> results/allocation_rules/seed<S>/summary.json
#
# IDENTITY STOP: when a run fails and its identity_stage0/1.json reads "passed": false, every later arm is
# skipped (logged STOP); the same failure would recur. Any other failure skips only that arm's dependents.
#
# Status: every non-zero rc, skipped arm and missing prior output is collected; the last line is
# "QUEUE DONE ok" (exit 0) or "QUEUE DONE FAILED: <steps>" (exit 1).
#
# Launch:   tmux new-session -d -s alloc "bash scripts/queue_alloc.sh; exec bash"
# Testing:  QUEUE_DRY_RUN=1, QUEUE_FAIL_STEP=<step name>, QUEUE_IDENTITY_FAIL=<step name>,
#           QUEUE_LOG_ROOT, QUEUE_RESULTS_ROOT.
set -u
USAGE="usage: queue_alloc.sh [--from-step K] [--smoke DIR --smoke-refs REFS]"
FROM=0
SMOKE=""
REFS=""
while [ $# -gt 0 ]; do
    case "$1" in
        --from-step) FROM="${2:-}"; shift 2 || { echo "$USAGE" >&2; exit 2; } ;;
        --smoke) SMOKE="${2:-}"; shift 2 || { echo "$USAGE" >&2; exit 2; } ;;
        --smoke-refs) REFS="${2:-}"; shift 2 || { echo "$USAGE" >&2; exit 2; } ;;
        *) echo "unknown argument '$1'; $USAGE" >&2; exit 2 ;;
    esac
done
case "$FROM" in ''|*[!0-9]*) echo "--from-step must be an integer 0..22, got '$FROM'" >&2; exit 2;; esac
if [ "$FROM" -gt 22 ]; then echo "--from-step must be an integer 0..22, got '$FROM'" >&2; exit 2; fi

cd "$(dirname "$0")/.." || { echo "cannot cd to repo root" >&2; exit 1; }
export MUJOCO_GL=egl

CUR=seq_hetero
ARMS="novelty size room"
TASKS=(libero_spatial/pick_up_the_black_bowl_from_table_center_and_place_it_on_the_plate
       libero_object/pick_up_the_milk_and_place_it_in_the_basket
       libero_goal/open_the_middle_drawer_of_the_cabinet
       libero_10/KITCHEN_SCENE3_turn_on_the_stove_and_put_the_moka_pot_on_it)
T4="${TASKS[3]}"
RUN_ARGS=(--curriculum "$CUR" --amp)
REPORT_ARGS=(--strict-provenance)
EVAL_ARGS=()
PRE_ARGS=()
CHK_ARGS=()
if [ -n "$SMOKE" ]; then
    case "$SMOKE" in /*) ;; *) echo "--smoke needs an absolute directory" >&2; exit 2 ;; esac
    case "$SMOKE/" in "$(pwd)/results/"*) echo "--smoke must be outside results/" >&2; exit 2 ;; esac
    case "$REFS" in /*) ;; *) echo "--smoke needs --smoke-refs (an absolute directory)" >&2; exit 2 ;; esac
    if [ "$FROM" != 0 ]; then echo "--from-step does not go with --smoke" >&2; exit 2; fi
    RES="$SMOKE"
    mkdir -p "$RES"
    RUN_ARGS+=(--results-root "$RES" --steps-per-task 200 --n-episodes 1 --allow-dirty)
    REPORT_ARGS+=(--allow-dirty)
    EVAL_ARGS+=(--n-episodes 1)
    PRE_ARGS+=(--smoke --results-root "$RES" --refs "$REFS")
    S=2
    TAG="_smoke"
else
    if [ -n "$REFS" ]; then echo "--smoke-refs goes with --smoke" >&2; exit 2; fi
    RES="${QUEUE_RESULTS_ROOT:-results}"
    REFS="$RES"
    S=0
    TAG=""
fi
NS=${CUR}__seq_ft__seed$S
SEQFT=$REFS/${CUR}__seq_ft__seed$S
GPM=$REFS/${CUR}__gpm_projected_adam__seed$S
E99=$RES/${CUR}__gpm_projected_adam_e99__seed$S
OUTD=$RES/allocation_rules/seed$S
if [ -n "$SMOKE" ]; then
    CHK_ARGS+=(--smoke --results-root "$RES" --comparator "$E99" --plain "$GPM")
fi

LOGDIR="${QUEUE_LOG_ROOT:-results/logs}/queue_$(date +%Y%m%d_%H%M%S)_alloc${TAG}"
mkdir -p "$LOGDIR" "$OUTD"
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

# identity_failed STEP RUN_DIR: an identity file of the run reads "passed": false.
identity_failed() {
    local f
    if [ "${QUEUE_DRY_RUN:-0}" = "1" ]; then
        [ "${QUEUE_IDENTITY_FAIL:-}" = "$1" ] && { note "IDENTITY FAILED (dry run): $1"; return 0; }
        return 1
    fi
    for f in "$2/identity_stage0.json" "$2/identity_stage1.json"; do
        if [ -e "$f" ] && grep -q '"passed": false' "$f"; then
            note "IDENTITY FAILED: $f"
            return 0
        fi
    done
    return 1
}

PY="uv run python"
note "QUEUE START alloc smoke='$SMOKE' refs='$REFS' seed=$S from_step=$FROM git $(git rev-parse HEAD) status '$(git status --porcelain | wc -l) changes' free '$(df -h --output=avail / | tail -1 | tr -d ' ')'"

if want 0; then
    step 0_preflight $PY scripts/alloc_queue.py --preflight --seeds $S "${PRE_ARGS[@]}" || {
        note "STOP: the preflight refused the launch"
        note "QUEUE DONE FAILED:$FAILED"
        exit 1
    }
fi

METHOD_OF() { echo "gpm_alloc_$1"; }
if [ -n "$SMOKE" ]; then
    # The smoke's comparator: an e99 trained like the arms, then configs with its own counts.
    step 0a_e99_smoke $PY scripts/run_continual.py "${RUN_ARGS[@]}" --method gpm_e99 --seed "$S" \
        --t1-reference-run "$SEQFT" --identity-reference-run "$GPM" --identity-stages 0 1 || {
        note "STOP: the smoke e99 failed"; note "QUEUE DONE FAILED:$FAILED"; exit 1; }
    step 0b_smoke_configs $PY scripts/allocation_rules.py smoke-configs --e99-run "$E99" --out "$RES/configs" || {
        note "STOP: no smoke configs"; note "QUEUE DONE FAILED:$FAILED"; exit 1; }
    METHOD_OF() { echo "$RES/configs/gpm_alloc_$1_smoke.yaml"; }
fi

# e99's evaluations on stage 3 (it exists; its rep 0 is re-run on every task as the reproduction gate).
run_or_prior 1 "eval_e99_rep0" "$OUTD/e99.rep0.eval.json" \
    $PY scripts/evaluate.py --checkpoint "$E99/checkpoints/stage3.pt" --tasks "${TASKS[@]}" \
    --run-id "$NS" "${EVAL_ARGS[@]}" --out "$OUTD/e99.rep0.eval.json"
for R in 1 2; do
    run_or_prior $((1 + R)) "eval_e99_rep$R" "$OUTD/e99.rep$R.eval.json" \
        $PY scripts/evaluate.py --checkpoint "$E99/checkpoints/stage3.pt" --tasks "${TASKS[@]}" \
        --run-id "${NS}__rep$R" "${EVAL_ARGS[@]}" --out "$OUTD/e99.rep$R.eval.json"
done

STOP=0
INDEX=0
for A in $ARMS; do
    K=$((4 + 6 * INDEX)); INDEX=$((INDEX + 1))
    RUN=$RES/${CUR}__gpm_projected_adam_alloc_${A}__seed$S
    OUT=$RES/gpm_seq_alloc_${A}_seed$S/report.json
    if [ $STOP = 1 ]; then
        note "SKIP arm $A (steps $K-$((K + 5))): identity stop"
        FAILED="$FAILED ${A}(identity_stop)"
        continue
    fi
    run_ok=0; check_ok=0
    run_or_prior $K "run_$A" "$RUN/result.json" \
        $PY scripts/run_continual.py "${RUN_ARGS[@]}" --method "$(METHOD_OF "$A")" --seed "$S" \
        --t1-reference-run "$SEQFT" --identity-reference-run "$GPM" --identity-stages 0 1 \
        && run_ok=1
    if [ $run_ok = 0 ]; then
        if want $K && identity_failed "${K}_run_$A" "$RUN"; then
            STOP=1
            note "STOP identity failure at arm $A: every later arm is skipped"
        fi
        note "SKIP $((K + 1))-$((K + 5)): ${K}_run_$A failed or is missing"
        continue
    fi
    if want $((K + 1)); then
        step "$((K + 1))_check_$A" $PY scripts/allocation_rules.py check --arm "$A" --seed "$S" --run-only \
            "${CHK_ARGS[@]}" && check_ok=1
    else
        check_ok=1; note "PRIOR $((K + 1))_check_$A: not rerun (before --from-step)"
    fi
    if [ $check_ok = 0 ]; then
        note "SKIP $((K + 2))-$((K + 5)): $((K + 1))_check_$A failed"
        continue
    fi
    run_or_prior $((K + 2)) "report_$A" "$OUT" \
        $PY scripts/sequence_report.py --method-run "$RUN" --reference-run "$SEQFT" --out "$OUT" \
        "${REPORT_ARGS[@]}"
    run_or_prior $((K + 3)) "eval_${A}_rep0" "$OUTD/$A.rep0.eval.json" \
        $PY scripts/evaluate.py --checkpoint "$RUN/checkpoints/stage3.pt" --tasks "$T4" \
        --run-id "$NS" "${EVAL_ARGS[@]}" --out "$OUTD/$A.rep0.eval.json"
    for R in 1 2; do
        run_or_prior $((K + 3 + R)) "eval_${A}_rep$R" "$OUTD/$A.rep$R.eval.json" \
            $PY scripts/evaluate.py --checkpoint "$RUN/checkpoints/stage3.pt" --tasks "${TASKS[@]}" \
            --run-id "${NS}__rep$R" "${EVAL_ARGS[@]}" --out "$OUTD/$A.rep$R.eval.json"
    done
done

# The summary always runs: it reports whatever is complete and labels the rest invalid.
SUM_ARGS=()
if [ -n "$SMOKE" ]; then SUM_ARGS+=(--smoke --results-root "$RES" --comparator "$E99" --plain "$GPM"); fi
step 22_summary $PY scripts/allocation_rules.py summary --seed "$S" "${SUM_ARGS[@]}"

if [ -z "$FAILED" ]; then
    note "QUEUE DONE ok"
    exit 0
fi
note "QUEUE DONE FAILED:$FAILED"
exit 1
