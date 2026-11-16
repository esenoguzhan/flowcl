#!/usr/bin/env bash
# The 8-task resume smoke (pre-E3 plan, B): E3's path, resuming 4-task runs into seq_hetero_t8 at
# stage 4 with seen-only evaluation, at 200 steps and 1 episode, seed 2, outside results/. Checks
# are flowcl/experiments/t8_smoke.py (step 12). Run from a clean tree (the strict reports require a
# clean git SHA) and when no other GPU job is running.
#
# Usage:    bash scripts/queue_t8_smoke.sh --root DIR --new NEW --ref REF
#   DIR  output root (absolute, outside results/); NEW, REF: queue_baselines.sh --smoke roots made
#        with the current and with the pre-§5 runner code (seq_ft, plain GPM, replay; seed 2).
#
#   0      the f = 0.7 4-task smoke: queue_ne70.sh --smoke DIR/ne70 --smoke-refs NEW
#   1-4    from_new/: seq_ft, gpm, replay (sources NEW), ne70 (source DIR/ne70)
#   5-7    from_ref/: seq_ft, gpm, replay (sources REF)
#   8-9    probe/: gpm, replay from NEW with --probe-every 50
#   10     cut/: gpm from NEW, killed by its own PID during stage 6 once train/stage5_record.json
#          exists; then: no result.json, records 0-5, no record 6
#   11     again/: the cut run resumed at stage 6 from its own folder
#   12     checks: scripts/check_t8_smoke.py --root DIR
# Every resume first runs the same command with --check-resume-only (step <k>_check).
#
# Status: the last line is "QUEUE DONE ok" (exit 0) or "QUEUE DONE FAILED: <steps>" (exit 1).
# Launch:   tmux new-session -d -s t8smoke "bash scripts/queue_t8_smoke.sh --root ... ; exec bash"
# Testing:  QUEUE_DRY_RUN=1, QUEUE_FAIL_STEP=<step name>, QUEUE_LOG_ROOT.

set -u
USAGE="usage: queue_t8_smoke.sh --root DIR --new NEW --ref REF"
ROOT=""; NEW=""; REF=""
while [ $# -gt 0 ]; do
    case "$1" in
        --root) ROOT="${2:-}"; shift 2 || { echo "$USAGE" >&2; exit 2; } ;;
        --new) NEW="${2:-}"; shift 2 || { echo "$USAGE" >&2; exit 2; } ;;
        --ref) REF="${2:-}"; shift 2 || { echo "$USAGE" >&2; exit 2; } ;;
        *) echo "unknown argument '$1'; $USAGE" >&2; exit 2 ;;
    esac
done
for d in "$ROOT" "$NEW" "$REF"; do
    case "$d" in /*) ;; *) echo "$USAGE (absolute directories)" >&2; exit 2 ;; esac
done

cd "$(dirname "$0")/.." || { echo "cannot cd to repo root" >&2; exit 1; }
case "$ROOT/" in "$(pwd)/results/"*) echo "--root must be outside results/" >&2; exit 2 ;; esac
export MUJOCO_GL=egl
mkdir -p "$ROOT"

LOGDIR="${QUEUE_LOG_ROOT:-$ROOT/logs}/queue_$(date +%Y%m%d_%H%M%S)_t8_smoke"
mkdir -p "$LOGDIR"
QUEUE_LOG="$LOGDIR/queue.log"
FAILED=""
PY="uv run python"
# The interrupted run is started without uv, so $! is the training process itself.
DIRECT_PY="$(pwd)/.venv/bin/python"
S=2
T8=seq_hetero_t8
COMMON=(--curriculum "$T8" --seed "$S" --amp --steps-per-task 200 --n-episodes 1 --no-eval-unseen)

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

stage_sha() {
    if [ "${QUEUE_DRY_RUN:-0}" = "1" ]; then echo DRYRUN; return; fi
    if [ -f "$1" ]; then sha256sum "$1" | cut -d' ' -f1; else echo MISSING; fi
}

# resume K NAME METHOD SOURCE OUT_ROOT START [EXTRA...]: check, then run.
resume() {
    local k=$1 name=$2 method=$3 src=$4 out=$5 start=$6; shift 6
    local sha
    sha=$(stage_sha "$src/checkpoints/stage$((start - 1)).pt")
    local args=("${COMMON[@]}" --method "$method" --resume-run "$src" --start-stage "$start"
                --resume-sha256 "$sha" --results-root "$out" "$@")
    step "${k}_check_$name" $PY scripts/run_continual.py "${args[@]}" --check-resume-only || return 1
    step "${k}_$name" $PY scripts/run_continual.py "${args[@]}"
}

note "QUEUE START t8_smoke root=$ROOT new=$NEW ref=$REF git $(git rev-parse HEAD) status '$(git status --porcelain | wc -l) changes' free '$(df -h --output=avail / | tail -1 | tr -d ' ')'"

step 0_ne70_smoke env QUEUE_LOG_ROOT="$LOGDIR/ne70_queue" bash scripts/queue_ne70.sh --smoke "$ROOT/ne70" --smoke-refs "$NEW"

resume 1 new_seq_ft seq_ft "$NEW/seq_hetero__seq_ft__seed$S" "$ROOT/from_new" 4
resume 2 new_gpm gpm "$NEW/seq_hetero__gpm_projected_adam__seed$S" "$ROOT/from_new" 4
resume 3 new_replay replay "$NEW/seq_hetero__replay__seed$S" "$ROOT/from_new" 4
resume 4 new_ne70 gpm_ne70 "$ROOT/ne70/seq_hetero__gpm_projected_adam_ne70__seed$S" "$ROOT/from_new" 4
resume 5 ref_seq_ft seq_ft "$REF/seq_hetero__seq_ft__seed$S" "$ROOT/from_ref" 4
resume 6 ref_gpm gpm "$REF/seq_hetero__gpm_projected_adam__seed$S" "$ROOT/from_ref" 4
resume 7 ref_replay replay "$REF/seq_hetero__replay__seed$S" "$ROOT/from_ref" 4
resume 8 probe_gpm gpm "$NEW/seq_hetero__gpm_projected_adam__seed$S" "$ROOT/probe" 4 --probe-every 50
resume 9 probe_replay replay "$NEW/seq_hetero__replay__seed$S" "$ROOT/probe" 4 --probe-every 50

# 10: an interrupted run. Started directly (no uv) in the background, so $! is its own PID.
CUT="$ROOT/cut/${T8}__gpm_projected_adam__seed$S"
SRC="$NEW/seq_hetero__gpm_projected_adam__seed$S"
SHA=$(stage_sha "$SRC/checkpoints/stage3.pt")
CUT_ARGS=("${COMMON[@]}" --method gpm --resume-run "$SRC" --start-stage 4 --resume-sha256 "$SHA"
          --results-root "$ROOT/cut")
cut_ok=0
if step 10_check_cut $PY scripts/run_continual.py "${CUT_ARGS[@]}" --check-resume-only; then
    note "START 10_cut: $DIRECT_PY scripts/run_continual.py ${CUT_ARGS[*]} (killed during stage 6)"
    if [ "${QUEUE_DRY_RUN:-0}" = "1" ]; then
        echo "${CUT_ARGS[*]}" > "$LOGDIR/10_cut.log"
        [ "${QUEUE_FAIL_STEP:-}" = "10_cut" ] || cut_ok=1
    else
        "$DIRECT_PY" scripts/run_continual.py "${CUT_ARGS[@]}" > "$LOGDIR/10_cut.log" 2>&1 &
        PID=$!
        note "10_cut running as PID $PID; waiting for $CUT/train/stage5_record.json"
        while [ ! -f "$CUT/train/stage5_record.json" ] && kill -0 "$PID" 2>/dev/null; do sleep 5; done
        if [ -f "$CUT/train/stage5_record.json" ] && kill -0 "$PID" 2>/dev/null; then
            sleep 20   # into stage 6
            kill "$PID"
            wait "$PID"
            note "10_cut: PID $PID killed during stage 6 (exit $?)"
            if [ ! -e "$CUT/result.json" ] && [ -f "$CUT/train/stage5_record.json" ] \
                    && [ ! -e "$CUT/train/stage6_record.json" ]; then
                cut_ok=1
            else
                note "10_cut: unexpected state (result.json or record 6 present)"
            fi
        else
            note "10_cut: the run ended before stage 5's record (see 10_cut.log)"
        fi
    fi
    note "END 10_cut ok=$cut_ok"
fi
[ $cut_ok = 1 ] || FAILED="$FAILED 10_cut"

if [ $cut_ok = 1 ]; then
    resume 11 again gpm "$CUT" "$ROOT/again" 6
else
    note "SKIP 11_again: 10_cut failed"
fi

step 12_checks $PY scripts/check_t8_smoke.py --root "$ROOT"

if [ -z "$FAILED" ]; then
    note "QUEUE DONE ok"
    exit 0
fi
note "QUEUE DONE FAILED:$FAILED"
exit 1
