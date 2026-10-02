#!/usr/bin/env bash
# Gate 0 for T6-T8 (docs/thesis_plan.md, A2): one independent single-task policy per slot,
# T5's Gate 0 recipe (30k steps, lr 1e-4, batch 64, AMP, seed 0). Orchestration only; the slots
# and backups are follow_up in configs/analysis/t5_sweep.yaml, the rule is Gate 0's
# (flowcl/analysis/gates.py), the backup rule is flowcl/experiments/gate0_t8.py. Run it only when
# no other GPU job is running.
#
# Usage:    bash scripts/queue_gate0_t8.sh [--config C] [--from-step K]
#   --config C     a later round's slots and output name (e.g. configs/analysis/gate0_t8_r2.yaml);
#                  default: round 1 (t5_sweep.yaml follow_up -> results/gate0_t8)
#   --from-step K  resume at slot K (in the config's slot order); earlier slots are only checked.
#
# Per slot: refuse (a failure) if its run directory or report already exists; run the primary;
# run the backup ONLY if this primary step returned 0 and its fresh report covers exactly the
# primary task with passed: false. A crashed primary never triggers a backup.
#
# Status: the last line is "QUEUE DONE ok" (exit 0) or "QUEUE DONE FAILED: <steps>" (exit 1).
#
# Launch:   tmux new-session -d -s gate0t8 "bash scripts/queue_gate0_t8.sh; exec bash"
# Testing:  QUEUE_DRY_RUN=1, QUEUE_FAIL_STEP=<step name>, QUEUE_LOG_ROOT, QUEUE_RESULTS_ROOT,
#           QUEUE_DRY_FAILED_SLOTS="T7 ..." (a dry primary of that slot writes a failing report).

set -u
USAGE="usage: queue_gate0_t8.sh [--config C] [--from-step K]"
LAST=2
FROM=0
CONFIG=""
while [ $# -gt 0 ]; do
    case "$1" in
        --from-step) FROM="${2:-}"; shift 2 || { echo "$USAGE" >&2; exit 2; } ;;
        --config) CONFIG="${2:-}"; shift 2 || { echo "$USAGE" >&2; exit 2; } ;;
        *) echo "unknown argument '$1'; $USAGE" >&2; exit 2 ;;
    esac
done
case "$FROM" in ''|*[!0-9]*) echo "--from-step must be an integer 0..$LAST, got '$FROM'" >&2; exit 2;; esac
if [ "$FROM" -gt $LAST ]; then echo "--from-step must be an integer 0..$LAST, got '$FROM'" >&2; exit 2; fi

cd "$(dirname "$0")/.." || { echo "cannot cd to repo root" >&2; exit 1; }
export MUJOCO_GL=egl

HELP="uv run python scripts/gate0_t8.py"
CFG_ARGS=()
if [ -n "$CONFIG" ]; then CFG_ARGS=(--config "$CONFIG"); fi
NAME=$($HELP out "${CFG_ARGS[@]}") || { echo "cannot read the config '$CONFIG'" >&2; exit 2; }

LOGDIR="${QUEUE_LOG_ROOT:-results/logs}/queue_$(date +%Y%m%d_%H%M%S)_$NAME"
mkdir -p "$LOGDIR"
QUEUE_LOG="$LOGDIR/queue.log"
RES="${QUEUE_RESULTS_ROOT:-results}"
OUT="$RES/$NAME"
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

dry_report() {  # testing only: a dry primary of a slot in QUEUE_DRY_FAILED_SLOTS fails Gate 0
    local slot=$1 task=$2 dir=$3
    case " ${QUEUE_DRY_FAILED_SLOTS:-} " in
        *" $slot "*)
            mkdir -p "$dir"
            printf '{"passed": false, "evidence": {"per_task": {"%s": {"success_rate": 0.5}}}}\n' "$task" \
                > "$dir/gate0.json" ;;
    esac
}

GATE0="uv run python scripts/gate0.py --seed 0 --train-steps 30000 --batch-size 64 --lr 1e-4 --amp"
note "QUEUE START $NAME config='$CONFIG' from_step=$FROM git $(git rev-parse HEAD) status '$(git status --porcelain | wc -l) changes'"

SLOTS=$($HELP tasks "${CFG_ARGS[@]}") || { note "cannot read the slots"; note "QUEUE DONE FAILED: slots"; exit 1; }

k=0
# shellcheck disable=SC2086
while read -r slot primary backup <&3; do
    if [ "$k" -lt "$FROM" ]; then
        if [ -e "$OUT/$slot/gate0.json" ]; then note "PRIOR ${k}_$slot ok: $OUT/$slot/gate0.json"
        else note "PRIOR ${k}_$slot missing: $OUT/$slot/gate0.json"; FAILED="$FAILED ${k}_$slot(missing)"; fi
        k=$((k + 1)); continue
    fi
    if ! stale=$($HELP fresh --results-root "$RES" --slot-dir "$OUT/$slot" --task "$primary"); then
        note "REFUSE ${k}_$slot: $stale"
        FAILED="$FAILED ${k}_$slot(stale)"
        k=$((k + 1)); continue
    fi
    since=$(date +%s)
    sleep 1  # the report's mtime must be strictly after the step started
    if step "${k}_$slot" $GATE0 --tasks "$primary" --out-dir "$OUT/$slot"; then
        if [ "${QUEUE_DRY_RUN:-0}" = "1" ]; then dry_report "$slot" "$primary" "$OUT/$slot"; fi
        decision=$($HELP backup --report "$OUT/$slot/gate0.json" --task "$primary" --since "$since")
        drc=$?
        if [ $drc -eq 0 ]; then
            note "BACKUP ${k}_$slot: the primary failed Gate 0; running $backup"
            if ! stale=$($HELP fresh --results-root "$RES" --slot-dir "$OUT/${slot}_backup" --task "$backup"); then
                note "REFUSE ${k}_${slot}_backup: $stale"
                FAILED="$FAILED ${k}_${slot}_backup(stale)"
            else
                step "${k}_${slot}_backup" $GATE0 --tasks "$backup" --out-dir "$OUT/${slot}_backup"
            fi
        elif [ $drc -eq 1 ]; then
            note "PASS ${k}_$slot: the primary passed Gate 0; no backup"
        else
            note "NO BACKUP ${k}_$slot: $decision"
            if [ "${QUEUE_DRY_RUN:-0}" != "1" ]; then FAILED="$FAILED ${k}_$slot($decision)"; fi
        fi
    else
        note "CRASH ${k}_$slot: the backup is not run"
    fi
    k=$((k + 1))
done 3<<< "$SLOTS"

if [ -z "$FAILED" ]; then
    note "QUEUE DONE ok"
    exit 0
fi
note "QUEUE DONE FAILED:$FAILED"
exit 1
