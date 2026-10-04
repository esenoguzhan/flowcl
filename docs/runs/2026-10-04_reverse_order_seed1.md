# C2 reverse order, seed 1: `task` (Object forgotten across Spatial; Goal retained at position 2, borderline)

**Status: the pre-registered control, the first replication seed.** The registered verdict is
**`task`**, with the label `borderline_position_retention`. Every validity check passed, and the
adaptive comparison is `valid`.

| Cell (plain GPM) | Acquisition `R[j][j]` | θ_j | Retention | State |
|---|---|---|---|---|
| **position:** Goal after Object, `R[2][1]` | 100 [100, 100] | 0.85 | **88 [78, 96]** (borderline) | `retained` |
| **task:** Object after Spatial, `R[3][2]` | 92 [84, 98] | 0.67 | **0 [0, 0]** | `forgotten` |

- **The task cell.** Object was learned to 92% and lost completely when Spatial was trained: drop
  0.92 [0.84, 0.98].
- **The position cell.** Goal stayed above θ after Object (88 against 0.85). The drop of 0.12
  [0.04, 0.22] is real, but the point estimate stays above θ. The cell is `retained`, flagged
  borderline.
- **The last transition again took most of what was left** (descriptive):
  - Goal 88 → 0%: drop 0.88 [0.78, 0.96];
  - Object 92 → 0%: drop 0.92 [0.84, 0.98];
  - LIBERO-10 96 → 76%: drop 0.20 [0.06, 0.34], which is partial this time, unlike seed 0 (0.74).
- **Adaptive GPM kept the old tasks, but its cost moved to T3.**
  - Final retention was 92 / 90 / 50%, against plain's 76 / 0 / 0%. The paired gains are
    +16 [+4, +30], +90 [+82, +98] and +50 [+36, +64] pp.
  - Spatial's plasticity is unchanged (94 against 94%, 0 [−10, +10] pp).
  - But adaptive GPM learned **Object to only 64% [50, 76]**, against plain's 92%. On seed 0 the
    two were equal, at 82%.

**Date (local, CEST):** Sat 3 Oct 14:32:12 → Sun 4 Oct 05:29:41.
- seq_ft 5.1 h;
- plain GPM 5.0 h;
- adaptive GPM 4.9 h;
- reports and summary 30 s.

**Code:** `6fc3616`, clean at queue start. Every run's `git_sha` is `6fc36164…`.
- The rule (`configs/analysis/reverse_order.yaml`, `d0bf349`), the summary and the queue are
  unchanged since `d0bf349`.
- A dry run before launch confirmed every command: seed 1, the reverse curriculum, and no
  `--single-task-baseline`.

**Cite:** `results/reverse_order/seed1/summary.json` (SHA-256 `2238203a…`).

---

## 1. Design

The same as seed 0 (`docs/runs/2026-10-03_reverse_order_seed0.md` §1).
- The runs share the namespace `seq_hetero_reverse__seq_ft__seed1`, and statistics are fitted on
  LIBERO-10.
- There is no `--single-task-baseline`, because Gate 0 references exist for seed 0 only.
- θ comes from this seed's reverse seq_ft diagonal, 96 / 100 / 82 / 94%, so θ = 0.81 / 0.85 /
  0.67 / 0.79.
- Object's θ (0.67) is lower than seed 0's (0.83), because seq_ft learned Object to only 82% here.
- `weak_reference` (diagonal < 0.70) is not set.

## 2. Validity (all passed)

| Check | Result |
|---|---|
| Completeness | three 4×4 matrices, 50 rollouts per cell |
| Production settings | curriculum, seed 1, namespace, recipe, 50 episodes, method configs, clean `git_sha`: all three runs |
| Normalization statistics | one fingerprint (`7c823817…`, the same as seed 0's: the statistics depend only on the LIBERO-10 demos); fitted on LIBERO-10; every stage checkpoint matches |
| T1 pairing | relative weight difference 0.0000, plain and adaptive |
| Identity, adaptive vs plain | 0 of 656 tensors differ at stage 0, and 0 of 656 at stage 1 |
| Episode pairing | identical per-episode seeds in every (stage, task), seq_ft vs plain and seq_ft vs adaptive |
| Sequence reports | every provenance check passed, plain and adaptive |
| Thresholds | the report blocks equal the summary's θ and the reverse seq_ft's `result.json` SHA-256 |

## 3. Results

**The retention matrices** (rows: after stage; columns: LIBERO-10 / Goal / Object / Spatial; %):

| After | seq_ft | plain GPM | adaptive GPM |
|---|---|---|---|
| LIBERO-10 | 96 / 0 / 0 / 0 | 96 / 0 / 0 / 0 | 96 / 0 / 0 / 0 |
| Goal | 0 / 100 / 0 / 0 | 98 / 100 / 0 / 0 | 98 / 100 / 0 / 0 |
| Object | 0 / 0 / 82 / 0 | 96 / **88** / 92 / 0 | 94 / 100 / **64** / 0 |
| Spatial | 0 / 0 / 0 / 94 | **76 / 0 / 0** / 94 | **92 / 90 / 50** / 94 |

**Overall performance (ACC over T1–T4):** seq_ft 0.235, plain GPM 0.425, adaptive GPM 0.815.

**The registered drops** (rollout-paired):

| Drop | plain GPM | adaptive GPM | adaptive − plain (joint bootstrap) |
|---|---|---|---|
| Goal, `R[1][1] − R[2][1]` | 0.12 [0.04, 0.22] | 0.00 [0.00, 0.00] | −0.12 [−0.22, −0.04] |
| Object, `R[2][2] − R[3][2]` | **0.92 [0.84, 0.98]** | 0.14 [−0.04, 0.32] | **−0.78 [−0.96, −0.58]** |

**Final retention, adaptive − plain:**
- LIBERO-10 +0.16 [+0.04, +0.30];
- Goal +0.90 [+0.82, +0.98];
- Object +0.50 [+0.36, +0.64].

Spatial's plasticity is 0.00 [−0.10, +0.10]. Object's acquisition was 64 against 92%, a T3
plasticity cost; it is not a registered quantity.

**The protection mechanism** (trunk medians):

| Memory update after | plain: energy in memory | plain: target | plain: occupancy after | adaptive: target | adaptive: occupancy after |
|---|---:|---:|---:|---:|---:|
| Goal (T2) | 0.905 | 0.95 | 0.56 | 0.990 | 0.83 |
| Object (T3) | 0.929 | 0.95 | 0.58 | 0.998 | 0.96 |
| Spatial (T4) | 0.934 | 0.95 | 0.60 | ≈0.999 | 0.99 |

- The plain/adaptive pattern is seed 0's.
- Decoder occupancy after T4 is plain 0.06 against adaptive 0.57.
- Adaptive GPM's trunk was 0.96 full when it learned Object, against plain's 0.58. That matches
  where its T3 cost appeared on this seed, but the cause is not tested.

**The forward order, seed 1, unpaired** (final rows, Spatial / Object / Goal / LIBERO-10):
- plain 92 / 0 / 20 / 96;
- adaptive 94 / 82 / 72 / 74.

Forward plain GPM lost Object at position 2 (78 → 2%).

## 4. The pre-registered judgement

| Quantity | Value | Rule | Result |
|---|---|---|---|
| Position: Goal acquired / after Object | 1.00 / **0.88** [0.78, 0.96] | forgotten if < 0.85 | **retained** (borderline) |
| Task: Object acquired / after Spatial | 0.92 / **0.00** | learned if ≥ 0.67; forgotten if < 0.67 | **forgotten** |

**Verdict: `task`** (summary exit 0), with labels `[borderline_position_retention]`. Adaptive
comparison: `valid`.

## 5. What this does and does not show

**It shows,** for seed 1 in the reverse order:
- Object was forgotten when Spatial followed it;
- Goal at position 2 lost 12 pp across Object but stayed above θ;
- under plain GPM the last transition again erased the later tasks (Goal and Object), with a
  partial LIBERO-10 loss;
- adaptive GPM protected them (+50 to +90 pp). On this seed its plasticity cost fell on Object at
  T3 (−28 pp), not on Spatial at T4.

**It does not show:**
- that Object is intrinsically fragile. "task" is descriptive: Object's successor (Spatial) also
  erased Goal on this seed, and LIBERO-10 and Goal on seed 0;
- a cause for the shifted adaptive cost;
- anything across seeds yet (the three-seed section comes with seed 2's record).

## 6. Disclosures

1. **Rule timing.** The rule, summary and queue are unchanged since `d0bf349`, committed before
   seed 0. This run started from a clean `6fc3616`, whose only later changes are records.
2. **The position cell is borderline:** 88 [78, 96] against θ = 0.85. The check conditions on the
   estimated θ; seq_ft's Goal diagonal is 100 [100, 100].
3. **Object's θ is 0.67,** from seq_ft's 82% Object diagonal, which is lower than seed 0's 98%.
   Plain GPM's 92% clears it easily. Its retention of 0% would be `forgotten` under any θ in
   play.
4. **No single-task baseline** for seed 1, so FWT is not computed.
5. **The forward comparison is unpaired.**
6. **The record was written in a separate worktree** while seed 2 ran, so the running checkout
   stayed at `6fc3616`.

## 7. Artifact map

```
results/reverse_order/seed1/summary.json                      # (SHA-256 2238203a...)
results/seq_hetero_reverse__{seq_ft,gpm_projected_adam,gpm_projected_adam_ne90}__seed1/
results/gpm_seq_reverse_seed1/report.json, results/gpm_seq_ne90_reverse_seed1/report.json
results/logs/queue_20261003_143212_reverse_order_seed1/
```
