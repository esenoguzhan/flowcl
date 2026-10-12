#!/usr/bin/env bash
# One projected-LoRA family's rank × learning-rate bracket, T1 -> T2 on the development seed (the
# projected-LoRA comparison): the rule's grid arms, the selection (with up to two edge extensions when the
# grid's selection lies on an edge), and a final verification. Orchestration only; the rule
# (configs/analysis/baseline_selection_{projected_lora,inflora,pca_trained}.yaml, committed before any
# arm) names the family, the grid and the output directory. K_old must already be captured
# (scripts/lora_selection.py --capture-kold). Run it only when no other GPU job is running.
#
# Usage:    bash scripts/queue_projected_lora_bracket.sh --config RULE [--from-step K]
#                                                        [--smoke DIR [--smoke-shared DIR]]
#   --from-step K     resume: arm steps < K are not rerun (status from their published rows); K in 0..N
#                     (N, the grid's size: every arm is done, go straight to the selection).
#   --smoke DIR       the smoke pipeline into DIR (outside results/), on a dirty tree, at the rule's first
#                     rank and largest lr: 4-task seq_ft and gpm_projected_adam (into --smoke-shared, default
#                     DIR; reused when already there) and the family's run at 200 steps and 1 episode, the
#                     family's stage 0 identity-checked against seq_ft's; the family's Task-1 memory equal to
#                     the smoke GPM run's, basis for basis; the family resumed at stage 3 and compared with
#                     compare_resume (tensors, per-episode evaluations, method state without timings); the
#                     bracket path against the runner (an arm from the smoke seq_ft stage 0 with the smoke
#                     run's own memory must reproduce its stage 1 and history); and a step-time and cost
#                     benchmark at rank 0 (full fine-tuning) and the rule's ranks on the production world.
#
#   0..N-1  arm (rank, lr), rank-major   -> results/<out>/lora/seed0/<prefix>__r<R>__lr<LR>__seed0
#   s       select (rc 10 = an edge extension is owed: not a failure)
#   e_k     extension k (k = 1, 2; only while select returns 10), then select again (s_k)
#   v       verify: inputs (memory and K_old included), checkpoints, the bundle, the recomputed selection
#
# Status: the last line is "QUEUE DONE ok" (exit 0) or "QUEUE DONE FAILED: <steps>" (exit 1).
# Launch:   tmux new-session -d -s plorab "bash scripts/queue_projected_lora_bracket.sh --config ...; exec bash"
# Testing:  QUEUE_DRY_RUN=1, QUEUE_FAIL_STEP=<step name>, QUEUE_STEP_RC=<step>=<rc> (several separated by
#           commas), QUEUE_LOG_ROOT, QUEUE_RESULTS_ROOT.

set -u
USAGE="usage: queue_projected_lora_bracket.sh --config RULE [--from-step K] [--smoke DIR [--smoke-shared DIR]]"
RULE=""
FROM=0
SMOKE=""
SHARED=""
while [ $# -gt 0 ]; do
    case "$1" in
        --config) RULE="${2:-}"; shift 2 || { echo "$USAGE" >&2; exit 2; } ;;
        --from-step) FROM="${2:-}"; shift 2 || { echo "$USAGE" >&2; exit 2; } ;;
        --smoke) SMOKE="${2:-}"; shift 2 || { echo "$USAGE" >&2; exit 2; } ;;
        --smoke-shared) SHARED="${2:-}"; shift 2 || { echo "$USAGE" >&2; exit 2; } ;;
        *) echo "unknown argument '$1'; $USAGE" >&2; exit 2 ;;
    esac
done
case "$FROM" in ''|*[!0-9]*) echo "--from-step must be a non-negative integer, got '$FROM'" >&2; exit 2;; esac
if [ -n "$SHARED" ] && [ -z "$SMOKE" ]; then echo "--smoke-shared needs --smoke" >&2; exit 2; fi

cd "$(dirname "$0")/.." || { echo "cannot cd to repo root" >&2; exit 1; }
export MUJOCO_GL=egl
if [ -z "$RULE" ] || [ ! -f "$RULE" ]; then echo "--config must name a rule file; $USAGE" >&2; exit 2; fi

PY="uv run python"
# The rule's family, output directory, arm prefix, grid (rank:lr-tag, rank-major), the smoke cell and the
# benchmark ranks. Read from the rule itself, so the queue cannot drift from it.
INFO=$($PY - "$RULE" <<'EOF'
import sys
from flowcl.experiments import lora_selection as ls
cfg = ls.load_config(sys.argv[1])
if not cfg.get("projection"):
    sys.exit("not a projection rule")
g = ls.grid(cfg)
print(ls.family(cfg))
print(cfg["out"])
print(ls.arm_prefix(cfg))
print(" ".join(f"{r}:{ls.lr_tag(x)}" for r, x in g))
print(f"{min(r for r, _ in g)}:{ls.lr_tag(max(x for _, x in g))}")
print(" ".join(str(r) for r in [0] + sorted({r for r, _ in g})))
EOF
) || { echo "cannot read the rule $RULE" >&2; exit 2; }
FAMILY=$(sed -n 1p <<< "$INFO"); OUT=$(sed -n 2p <<< "$INFO"); PREFIX=$(sed -n 3p <<< "$INFO")
ARMS=$(sed -n 4p <<< "$INFO"); SMOKE_CELL=$(sed -n 5p <<< "$INFO"); BENCH_RANKS=$(sed -n 6p <<< "$INFO")
N_ARMS=$(wc -w <<< "$ARMS")
if [ "$FROM" -gt "$N_ARMS" ]; then echo "--from-step must be an integer 0..$N_ARMS, got '$FROM'" >&2; exit 2; fi

RES="${QUEUE_RESULTS_ROOT:-results}"
if [ -n "$SMOKE" ]; then
    for d in "$SMOKE" "${SHARED:-$SMOKE}"; do
        case "$d" in /*) ;; *) echo "--smoke and --smoke-shared need absolute directories" >&2; exit 2 ;; esac
        case "$d/" in "$(pwd)/results/"*) echo "--smoke and --smoke-shared must be outside results/" >&2; exit 2 ;; esac
    done
    SHARED="${SHARED:-$SMOKE}"
    mkdir -p "$SMOKE" "$SHARED"
    TAG="_smoke"
else
    TAG=""
fi

LOGDIR="${QUEUE_LOG_ROOT:-results/logs}/queue_$(date +%Y%m%d_%H%M%S)_${FAMILY}_bracket${TAG}"
mkdir -p "$LOGDIR"
QUEUE_LOG="$LOGDIR/queue.log"
FAILED=""

note() { echo "$(date '+%F %T') $*" | tee -a "$QUEUE_LOG"; }

step_rc_override() {   # QUEUE_STEP_RC="a=10,b=1"
    local name=$1 item
    IFS=',' read -ra items <<< "${QUEUE_STEP_RC:-}"
    for item in "${items[@]}"; do
        case "$item" in "$name="*) echo "${item#*=}"; return 0 ;; esac
    done
    return 1
}

step() {
    local name=$1; shift
    note "START $name: $*"
    local rc
    if [ "${QUEUE_DRY_RUN:-0}" = "1" ]; then
        echo "$*" > "$LOGDIR/$name.log"
        rc=0
        if [ "${QUEUE_FAIL_STEP:-}" = "$name" ]; then rc=1; fi
        if override=$(step_rc_override "$name"); then rc=$override; fi
    else
        "$@" > "$LOGDIR/$name.log" 2>&1
        rc=$?
    fi
    note "END $name rc=$rc"
    return $rc
}

fail() { FAILED="$FAILED $1"; }

LS=(scripts/lora_selection.py --config "$RULE")
note "QUEUE START ${FAMILY}_bracket rule=$RULE smoke='$SMOKE' shared='$SHARED' from_step=$FROM arms='$ARMS' git $(git rev-parse HEAD) status '$(git status --porcelain | wc -l) changes'"

if [ -n "$SMOKE" ]; then
    R="${SMOKE_CELL%%:*}"; LR="${SMOKE_CELL#*:}"
    CL=(--curriculum seq_hetero --seed 0 --amp --steps-per-task 200 --n-episodes 1 --allow-dirty)
    REF="$SHARED/seq_hetero__seq_ft__seed0"
    GPM="$SHARED/seq_hetero__gpm_projected_adam__seed0"
    RUN="$SMOKE/seq_hetero__${FAMILY}__seed0"
    step s0_smoke_method $PY "${LS[@]}" --write-smoke-method "$SMOKE/method_smoke.yaml" --rank "$R" --lr "$LR" \
        || fail s0_smoke_method
    if [ -e "$REF/result.json" ]; then note "REUSE s1_seqft: $REF"; else
        step s1_seqft $PY scripts/run_continual.py "${CL[@]}" --results-root "$SHARED" --method seq_ft || fail s1_seqft
    fi
    if [ -e "$GPM/result.json" ]; then note "REUSE s2_gpm: $GPM"; else
        step s2_gpm $PY scripts/run_continual.py "${CL[@]}" --results-root "$SHARED" --method gpm \
            --t1-reference-run "$REF" --identity-reference-run "$REF" --identity-stages 0 || fail s2_gpm
    fi
    step s3_run $PY scripts/run_continual.py "${CL[@]}" --results-root "$SMOKE" --method "$SMOKE/method_smoke.yaml" \
        --t1-reference-run "$REF" --identity-reference-run "$REF" --identity-stages 0 || fail s3_run
    step s4_memory_equal $PY "${LS[@]}" --compare-memory "$GPM/method/memory_task0.pt" "$RUN/method/memory_task0.pt" \
        --out "$SMOKE/memory_equal.json" || fail s4_memory_equal
    step s5_resume $PY scripts/run_continual.py "${CL[@]}" --results-root "$SMOKE/resumed" \
        --method "$SMOKE/method_smoke.yaml" --resume-run "$RUN" --start-stage 3 || fail s5_resume
    step s6_compare $PY scripts/compare_resume.py --source "$RUN" --resumed "$SMOKE/resumed/seq_hetero__${FAMILY}__seed0" \
        --stage 3 --all-artifacts --method-state --out "$SMOKE/resume_compare.json" || fail s6_compare
    step s7_bracket_identity $PY "${LS[@]}" --smoke-arm "$R" "$LR" --reference-run "$REF" --identity-run "$RUN" \
        --smoke-steps 200 --out-root "$SMOKE/bracket" --allow-dirty || fail s7_bracket_identity
    # shellcheck disable=SC2086
    step s8_benchmark $PY "${LS[@]}" --benchmark --bench-ranks $BENCH_RANKS --bench-steps 300 \
        --out-root "$SMOKE/benchmark" || fail s8_benchmark
else
    K=0
    for CELL in $ARMS; do
        R="${CELL%%:*}"; LR="${CELL#*:}"
        NAME="${K}_arm_r${R}_lr${LR}"
        ROW="$RES/$OUT/lora/seed0/${PREFIX}__r${R}__lr${LR}__seed0/row.json"
        if [ "$K" -lt "$FROM" ]; then
            if [ -e "$ROW" ]; then note "PRIOR $NAME ok: $ROW"; else note "PRIOR $NAME missing: $ROW"; fail "$NAME(missing)"; fi
        else
            step "$NAME" $PY "${LS[@]}" --arm "$R" "$LR" || fail "$NAME"
        fi
        K=$((K + 1))
    done
    case "$FAILED" in
        *_arm_*) note "SKIP select and verify: an arm failed or is missing" ;;
        *)
            step s_select $PY "${LS[@]}" --select
            rc=$?
            E=1
            while [ $rc = 10 ] && [ $E -le 2 ]; do
                note "edge extension owed (rc 10)"
                if step "e_extension_$E" $PY "${LS[@]}" --extension; then
                    step "s_select_$E" $PY "${LS[@]}" --select
                    rc=$?
                else
                    fail "e_extension_$E"; rc=failed; break
                fi
                E=$((E + 1))
            done
            case "$rc" in
                0) step v_verify $PY "${LS[@]}" --verify || fail v_verify ;;
                failed) note "SKIP verify: an extension failed" ;;
                10) fail "select(extensions still owed after 2)" ;;
                *) fail "select(rc $rc)" ;;
            esac ;;
    esac
fi

if [ -n "$FAILED" ]; then
    note "QUEUE DONE FAILED:$FAILED"
    exit 1
fi
note "QUEUE DONE ok"
exit 0
