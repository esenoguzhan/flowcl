#!/usr/bin/env bash
# Mon 12 Oct's chained GPU queue: Track 2's measures (step 2), then the projected-LoRA comparison's three
# brackets automatically. Orchestration only; every rule is committed before this runs (the measures
# section of configs/analysis/behavior_predictors.yaml; configs/analysis/baseline_selection_{projected_lora,
# inflora,pca_trained}.yaml). Run it on a clean tree, with the full CPU suite started at the same commit
# and writing SUITE (its last lines "exit <rc>", the date and the short SHA).
#
# Usage:    bash scripts/queue_tonight.sh --suite-result SUITE --smoke DIR [--suite-wait-min M]
#
#   t0_kold                  K_old: the one-time Task-1 input Gram at seq_ft seed 0's stage 0 (immutable)
#   t1_energy_gpm            the layer-output-change reference for gpm_projected_adam seed 0's stage 1
#   t2_energy_lora           ... and for LoRA v2's selected arm (rank 16, lr 3e-5)
#   t3_measures_smoke        the measures on 3 cells into DIR/measures (instrument checks; nothing published)
#   t4_smoke_<family>        each family's smoke queue into DIR/<family> (seq_ft and gpm_projected_adam
#                            shared in DIR/shared), for projected_lora, inflora, pca_trained
#   t5_suite                 wait (up to M minutes, default 240) for SUITE: "exit 0" at this commit, tree clean
#   t6_measures              every manifest cell; published only if the instrument checks pass and SUITE is
#                            green (skipped if t3 or t5 failed)
#   t7_bracket_<family>      each family's bracket queue, only if its smoke passed and the suite is green;
#                            the families are otherwise independent (projected_lora, inflora, pca_trained)
#
# Status: the last line is "QUEUE DONE ok" (exit 0) or "QUEUE DONE FAILED: <steps>" (exit 1).
# Launch:   tmux new-session -d -s tonight "bash scripts/queue_tonight.sh --suite-result ... --smoke ...; exec bash"
# Testing:  QUEUE_DRY_RUN=1 (the sub-queues are called with QUEUE_DRY_RUN inherited), QUEUE_FAIL_STEP,
#           QUEUE_STEP_RC, QUEUE_LOG_ROOT, QUEUE_RESULTS_ROOT.

set -u
USAGE="usage: queue_tonight.sh --suite-result SUITE --smoke DIR [--suite-wait-min M]"
SUITE=""
SMOKE=""
WAIT_MIN=240
while [ $# -gt 0 ]; do
    case "$1" in
        --suite-result) SUITE="${2:-}"; shift 2 || { echo "$USAGE" >&2; exit 2; } ;;
        --smoke) SMOKE="${2:-}"; shift 2 || { echo "$USAGE" >&2; exit 2; } ;;
        --suite-wait-min) WAIT_MIN="${2:-}"; shift 2 || { echo "$USAGE" >&2; exit 2; } ;;
        *) echo "unknown argument '$1'; $USAGE" >&2; exit 2 ;;
    esac
done
if [ -z "$SUITE" ] || [ -z "$SMOKE" ]; then echo "$USAGE" >&2; exit 2; fi
case "$WAIT_MIN" in ''|*[!0-9]*) echo "--suite-wait-min must be whole minutes, got '$WAIT_MIN'" >&2; exit 2;; esac
case "$SMOKE" in /*) ;; *) echo "--smoke needs an absolute directory" >&2; exit 2 ;; esac
case "$SUITE" in /*) ;; *) echo "--suite-result needs an absolute path" >&2; exit 2 ;; esac

cd "$(dirname "$0")/.." || { echo "cannot cd to repo root" >&2; exit 1; }
case "$SMOKE/" in "$(pwd)/results/"*) echo "--smoke must be outside results/" >&2; exit 2 ;; esac
export MUJOCO_GL=egl

RES="${QUEUE_RESULTS_ROOT:-results}"
LOGDIR="${QUEUE_LOG_ROOT:-results/logs}/queue_$(date +%Y%m%d_%H%M%S)_tonight"
mkdir -p "$LOGDIR" "$SMOKE"
QUEUE_LOG="$LOGDIR/queue.log"
FAILED=""

note() { echo "$(date '+%F %T') $*" | tee -a "$QUEUE_LOG"; }

step_rc_override() {
    local name=$1 item
    IFS=',' read -ra items <<< "${QUEUE_STEP_RC:-}"
    for item in "${items[@]}"; do
        case "$item" in "$name="*) echo "${item#*=}"; return 0 ;; esac
    done
    return 1
}

step() {   # sub-queues run for real in a dry run too (they dry-run their own steps)
    local name=$1; shift
    note "START $name: $*"
    local rc
    if [ "${QUEUE_DRY_RUN:-0}" = "1" ] && [ "$1" != "bash" ]; then
        echo "$*" > "$LOGDIR/$name.log"
        rc=0
    else
        "$@" > "$LOGDIR/$name.log" 2>&1
        rc=$?
    fi
    if [ "${QUEUE_DRY_RUN:-0}" = "1" ]; then
        if [ "${QUEUE_FAIL_STEP:-}" = "$name" ]; then rc=1; fi
        if override=$(step_rc_override "$name"); then rc=$override; fi
    fi
    note "END $name rc=$rc"
    return $rc
}

fail() { FAILED="$FAILED $1"; }

suite_green() {   # SUITE reports success at this commit, and the tree is clean
    [ -f "$SUITE" ] && grep -qx "exit 0" "$SUITE" && grep -qx "$(git rev-parse --short HEAD)" "$SUITE" \
        && [ -z "$(git status --porcelain)" ]
}

wait_suite() {
    local waited=0
    while ! { [ -f "$SUITE" ] && grep -q "^exit " "$SUITE"; }; do
        if [ "$waited" -ge $((WAIT_MIN * 60)) ]; then echo "no suite result after $WAIT_MIN min"; return 1; fi
        sleep 60; waited=$((waited + 60))
    done
    if suite_green; then echo "suite green: $(tail -3 "$SUITE" | tr '\n' ' ')"; return 0; fi
    echo "suite not green at $(git rev-parse --short HEAD): $(tail -4 "$SUITE" | tr '\n' ' ')"; return 1
}

PY="uv run python"
FAMILIES="projected_lora inflora pca_trained"
rule() { echo "configs/analysis/baseline_selection_$1.yaml"; }
LS=(scripts/lora_selection.py --config "$(rule projected_lora)")
OUTP="$RES/baseline_selection_projected"
note "QUEUE START tonight suite='$SUITE' smoke='$SMOKE' git $(git rev-parse HEAD) status '$(git status --porcelain | wc -l) changes'"

step t0_kold $PY "${LS[@]}" --capture-kold || fail t0_kold
step t1_energy_gpm $PY "${LS[@]}" --energy "$RES/seq_hetero__gpm_projected_adam__seed0/checkpoints/stage0.pt" \
    "$RES/seq_hetero__gpm_projected_adam__seed0/checkpoints/stage1.pt" \
    --out "$OUTP/energy_gpm_projected_adam_seed0_stage1.json" || fail t1_energy_gpm
step t2_energy_lora $PY "${LS[@]}" --energy "$RES/seq_hetero__seq_ft__seed0/checkpoints/stage0.pt" \
    "$RES/baseline_selection_lora_v2/lora/seed0/lorab__r16__lr3e-05__seed0/checkpoints/final.pt" \
    --out "$OUTP/energy_lora_v2_selected.json" || fail t2_energy_lora
MEASURES_SMOKE=ok
step t3_measures_smoke $PY scripts/behavior_predictors.py --measures --smoke 3 --out-root "$SMOKE/measures" \
    || { fail t3_measures_smoke; MEASURES_SMOKE=failed; }
SMOKED=""
for F in $FAMILIES; do
    if step "t4_smoke_$F" bash scripts/queue_projected_lora_bracket.sh --config "$(rule "$F")" \
            --smoke "$SMOKE/$F" --smoke-shared "$SMOKE/shared"; then
        SMOKED="$SMOKED $F"
    else
        fail "t4_smoke_$F"
    fi
done
SUITE_OK=0
if step t5_suite wait_suite; then SUITE_OK=1; else fail t5_suite; fi

if [ $SUITE_OK = 1 ] && [ $MEASURES_SMOKE = ok ]; then
    step t6_measures $PY scripts/behavior_predictors.py --measures --suite-result "$SUITE" || fail t6_measures
else
    note "SKIP t6_measures: the suite is not green or the measures smoke failed"
fi
for F in $FAMILIES; do
    if [ $SUITE_OK != 1 ]; then
        note "SKIP t7_bracket_$F: the suite is not green"
    elif [ -n "$(echo "$FAILED" | tr ' ' '\n' | grep -x 't0_kold')" ]; then
        note "SKIP t7_bracket_$F: K_old was not captured"
    elif case " $SMOKED " in *" $F "*) false ;; *) true ;; esac; then
        note "SKIP t7_bracket_$F: its smoke failed"
    else
        step "t7_bracket_$F" bash scripts/queue_projected_lora_bracket.sh --config "$(rule "$F")" || fail "t7_bracket_$F"
    fi
done

if [ -n "$FAILED" ]; then
    note "QUEUE DONE FAILED:$FAILED"
    exit 1
fi
note "QUEUE DONE ok"
exit 0
