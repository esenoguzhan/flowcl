# C2 reverse order, seed 2: `both` (both cells one episode from θ); and the three seeds together

**Status: the pre-registered control, the second replication seed.** The registered verdict is
**`both`**, with the labels `borderline_position_retention` and `borderline_task_diagonal`. Every
validity check passed, and the adaptive comparison is `valid`.

| Cell (plain GPM) | Acquisition `R[j][j]` | θ_j | Retention | State |
|---|---|---|---|---|
| **position:** Goal after Object, `R[2][1]` | 100 [100, 100] | 0.85 | **84 [74, 94]** (borderline) | `forgotten` |
| **task:** Object after Spatial, `R[3][2]` | **80 [68, 90]** (borderline) | 0.79 | 0 [0, 0] | `forgotten` |

- **Both cells sit one episode from θ.**
  - Goal kept 42 of 50 episodes after Object, while 0.85 needs 43, so it is `forgotten`. Its drop
    is 0.16 [0.06, 0.26].
  - Object was learned to 40 of 50, while 0.79 needs 40, so it is learned. It then fell to 0%
    after Spatial: drop 0.80 [0.68, 0.90].
- **The last transition erased every earlier task again** (descriptive), as on seed 0:
  - LIBERO-10 98 → 28%: drop 0.70 [0.56, 0.84];
  - Goal 84 → 14%: drop 0.70 [0.56, 0.84];
  - Object 80 → 0%: drop 0.80 [0.68, 0.90].
- **Adaptive GPM kept them.**
  - Final retention was 68 / 98 / 76%, against plain's 28 / 14 / 0%: gains of +40 [+24, +56],
    +84 [+74, +94] and +76 [+64, +88] pp.
  - It paid in plasticity: Spatial −24 [−38, −12] pp (74 against 98%) and Object −10
    [−22, +2] pp (70 against 80%).

**Date (local, CEST):** Sun 4 Oct 05:29:41 → 20:29:35.
- seq_ft 5.1 h;
- plain GPM 5.1 h;
- adaptive GPM 4.9 h;
- reports and summary 30 s.

**Code:** `6fc3616`, clean at queue start; every run's `git_sha` is `6fc36164…`. The rule, summary
and queue are unchanged since `d0bf349`. The dry run before launch confirmed seed 2, the reverse
curriculum, and no `--single-task-baseline`.

**Cite:** `results/reverse_order/seed2/summary.json` (SHA-256 `11189cac…`).

---

## 1. Seed 2

**Design.** As seed 0 (`docs/runs/2026-10-03_reverse_order_seed0.md` §1):
- namespace `seq_hetero_reverse__seq_ft__seed2`;
- statistics fitted on LIBERO-10;
- no single-task baseline.

The reverse seq_ft diagonal was 98 / 100 / 94 / 96%, so θ = 0.83 / 0.85 / 0.79 / 0.81.
`weak_reference` is not set.

**Validity: all passed.**
- Completeness: 4×4 matrices, 50 rollouts per cell.
- Production settings: all three runs.
- Statistics: one fingerprint `7c823817…`, fitted on LIBERO-10, every checkpoint matches.
- T1 pairing: 0.0000, plain and adaptive.
- Identity, adaptive vs plain: 0 of 656 tensors differ at stage 0, and 0 of 656 at stage 1.
- Episode pairing: identical seeds, seq_ft vs plain and seq_ft vs adaptive.
- Sequence reports: every provenance check passed.
- Thresholds: the report blocks equal the summary's θ and the reference SHA.

**The retention matrices** (LIBERO-10 / Goal / Object / Spatial; %):

| After | seq_ft | plain GPM | adaptive GPM |
|---|---|---|---|
| LIBERO-10 | 98 / 0 / 0 / 0 | 98 / 0 / 0 / 0 | 98 / 0 / 0 / 0 |
| Goal | 0 / 100 / 0 / 0 | 88 / 100 / 0 / 0 | 88 / 100 / 0 / 0 |
| Object | 0 / 0 / 94 / 0 | 98 / **84** / **80** / 0 | 88 / 98 / 70 / 0 |
| Spatial | 0 / 0 / 0 / 96 | **28 / 14 / 0** / 98 | **68 / 98 / 76** / 74 |

**Overall performance (ACC over T1–T4):** seq_ft 0.24, plain GPM 0.35, adaptive GPM 0.79.

**The registered drops and contrasts:**

| Drop | plain GPM | adaptive GPM | adaptive − plain (joint bootstrap) |
|---|---|---|---|
| Goal, `R[1][1] − R[2][1]` | 0.16 [0.06, 0.26] | 0.02 [0.00, 0.06] | −0.14 [−0.26, −0.04] |
| Object, `R[2][2] − R[3][2]` | **0.80 [0.68, 0.90]** | −0.06 [−0.22, 0.10] | **−0.86 [−1.02, −0.70]** |

The joint statistic is a difference of two drops, so it ranges over [−2, 2]. Its percentile CI can
therefore extend past −1.

**The mechanism** (trunk medians) is as on seeds 0 and 1:
- **Plain GPM.**
  - The energy already in memory was 0.906 at the Goal update, 0.929 at Object and 0.934 at
    Spatial.
  - The target was 0.95.
  - Trunk occupancy grew 0.55 → 0.57 → 0.60.
- **Adaptive GPM.**
  - The target was 0.991 at Goal and 0.998 at Object.
  - Trunk occupancy grew 0.82 → 0.95 → 0.99.
- Decoder occupancy after T4 is plain 0.05 against adaptive 0.54.

**The forward order, seed 2, unpaired** (final rows, Spatial / Object / Goal / LIBERO-10):
- plain 96 / 24 / 96 / 94;
- adaptive 94 / 78 / 98 / 80.

Forward plain GPM lost Object at position 2 (86 → 22%).

## 2. The three seeds together

Per the rule, a matching valid verdict on all three seeds would show the pattern recurs across
those seeds. The table puts each seed's quantities side by side. Seed 0 is the development seed;
seeds 1–2 are the replication seeds.

| | Seed 0 | Seed 1 | Seed 2 |
|---|---|---|---|
| **Registered verdict** | `inconclusive` | `task` | `both` |
| Labels | borderline task diagonal | borderline position retention | borderline position retention, borderline task diagonal |
| θ Goal / θ Object | 0.85 / 0.83 | 0.85 / 0.67 | 0.85 / 0.79 |
| **Goal (position 2): acquired → after Object** | 100 → 98 | 100 → 88 | 100 → 84 |
| Goal drop, plain (paired) | 0.02 [0.00, 0.06] | 0.12 [0.04, 0.22] | 0.16 [0.06, 0.26] |
| Position cell | `retained` | `retained` (borderline) | `forgotten` (borderline, one episode) |
| **Object (position 3): acquired → after Spatial** | 82 → 0 | 92 → 0 | 80 → 0 |
| Object drop, plain (paired) | 0.82 [0.70, 0.92] | 0.92 [0.84, 0.98] | 0.80 [0.68, 0.90] |
| Task cell | `not_learned` (one episode) | `forgotten` | `forgotten` (acquisition borderline, one episode) |
| Plain drops across Spatial, L10 / Goal / Object | 0.74 / 0.96 / 0.82 | 0.20 / 0.88 / 0.92 | 0.70 / 0.70 / 0.80 |
| Plain final row, L10 / Goal / Object / Spatial | 20 / 2 / 0 / 98 | 76 / 0 / 0 / 94 | 28 / 14 / 0 / 98 |
| Adaptive final row | 96 / 98 / 72 / 84 | 92 / 90 / 50 / 94 | 68 / 98 / 76 / 74 |
| **Adaptive − plain, final retention** (L10 / Goal / Object) | +76 / +96 / +72 | +16 / +90 / +50 | +40 / +84 / +76 |
| Adaptive − plain, drop contrast Goal / Object | −0.02 / **−0.72** | **−0.12** / **−0.78** | **−0.14** / **−0.86** |
| **Adaptive plasticity cost,** Object (T3) | 0 [−14, +14] | **−28 [−44, −12]** | −10 [−22, +2] |
| **Adaptive plasticity cost,** Spatial (T4) | **−14 [−26, −4]** | 0 [−10, +10] | **−24 [−38, −12]** |
| Trunk occupancy after T4, plain / adaptive | 0.61 / 0.99 | 0.60 / 0.99 | 0.60 / 0.99 |
| ACC: seq_ft / plain / adaptive | 0.24 / 0.30 / 0.875 | 0.235 / 0.425 / 0.815 | 0.24 / 0.35 / 0.79 |

Bold contrast and cost entries have paired CIs that exclude 0.

**The registered statement.**
- The three verdicts differ (`inconclusive`, `task`, `both`), so **no three-seed recurrence is
  claimed**.
- The two replication seeds also give different verdicts (`task`, `both`). The difference is the
  position cell: Goal kept 88% on seed 1 and 84% on seed 2, against θ = 0.85. That is one or two
  episodes on either side of the line.
- **The replication seeds agree on the task cell:** both are `forgotten`. Seed 0's task cell is
  `not_learned` by one episode, and its Object retention is 0% as well.

**Descriptive patterns,** each holding on every seed:
- **Object is lost completely across Spatial:** 82 / 92 / 80 → 0%, with paired drops of
  0.80–0.92 and every CI far above 0.
- **Goal at position 2 loses little across Object:** 2 / 12 / 16 pp. It crosses θ only on seed 2,
  by one episode. The forward order's position-2 collapse (Object 78 / 78 / 86 → 2 / 2 / 22%)
  does not recur at position 2 in the reverse order.
- **The last transition (Spatial) erases Goal on every seed** (drops 0.70–0.96). It erases
  LIBERO-10 on seeds 0 and 2 (0.74 and 0.70), and only partly on seed 1 (0.20). **So Object is
  not the only task Spatial erases.** The `task` reading ("Object is fragile") is not separable
  from "the last transition's interference is large".
- **Plain GPM's memory protects little of each later task.** At every update, 90–94% of the new
  task's trunk input energy is already in memory, so the 0.95 target adds little: occupancy is
  0.60–0.61 after T4. Adaptive GPM reaches 0.99.

**The adaptive trade replicates on all three seeds.**
- Final retention of every earlier task is higher under adaptive GPM, with every paired CI above
  0. The gains range from +16 to +96 pp.
- Object's drop is cut by 0.72–0.86, with every CI below 0.
- There is a plasticity cost on every seed, but **its location varies:** Spatial on seed 0 (−14),
  Object on seed 1 (−28), and both on seed 2 (−10 and −24).
- Trunk occupancy reaches 0.99 after T4 on every seed.

## 3. What this does and does not show

**It shows,** across three seeds of the reverse order, with the rule's descriptive reading:
- the forward order's position-2 collapse does not recur at position 2: Goal loses 2–16 pp;
- Object, at position 3, is lost completely across Spatial on every seed;
- so is most of everything else that precedes Spatial;
- adaptive GPM's protection and its capacity/plasticity cost both replicate. The cost lands on
  whichever late task the full trunk constrains.

**It does not show:**
- **a cause** (position, task, pair or last-transition interference). Reversal also changes
  predecessors, successors and normalization, and the registered cells cannot separate them.
- **a stable registered verdict.** Every disagreement between seeds turns on one or two episodes
  at a threshold.

## 4. Disclosures

1. **Rule timing.** The rule, summary and queue are unchanged since `d0bf349`, committed before
   seed 0. Seeds 1–2 ran from a clean `6fc3616`, whose later changes are records only.
2. **Borderlines.**
   - Seed 2's position cell is 84 [74, 94] against 0.85, and its Object acquisition is 80
     [68, 90] against 0.79.
   - Seed 1's position cell is 88 [78, 96] against 0.85.
   - Seed 0's Object acquisition is 82 against 0.83.
   - The point-estimate rule decides each one. The borderline checks condition on the estimated
     θ, and θ's own uncertainty, from the seq_ft diagonals, is not included.
3. **The θ for Object differs by seed** (0.83 / 0.67 / 0.79), following seq_ft's Object diagonal
   (98 / 82 / 94%).
4. **No single-task baseline** for seeds 1–2.
5. **Forward comparisons are unpaired.**
6. **The records were written in a separate worktree** while seed 2 ran, and merged after its
   queue ended. Every run of seeds 1–2 records the clean `6fc3616`.

## 5. Consequences

**C2 is complete for three seeds.** The registered verdicts are `inconclusive` / `task` / `both`,
so no three-seed recurrence. The descriptive finding is stable:
- no position-2 collapse in the reverse order;
- a complete loss of Object, and most other tasks, across the last transition under plain GPM;
- adaptive GPM's protection and cost both replicate.

**For the thesis.**
- The forward order's Object loss is not a simple "position 2" effect.
- In both orders, plain GPM's fixed target leaves later tasks' new energy largely unprotected.
- Where the damage lands depends on the order: Object across Goal (forward), everything across
  Spatial (reverse).
- Adaptive allocation prevents both, and spends capacity and late-task plasticity to do it.
- Under `docs/thesis_plan.md` C2's outcome categories (position / task / both / neither), the
  registered evidence is mixed. The record reports it as such.

## 6. Artifact map

```
results/reverse_order/seed2/summary.json                      # (SHA-256 11189cac...)
results/seq_hetero_reverse__{seq_ft,gpm_projected_adam,gpm_projected_adam_ne90}__seed2/
results/gpm_seq_reverse_seed2/report.json, results/gpm_seq_ne90_reverse_seed2/report.json
results/logs/queue_20261004_052941_reverse_order_seed2/
seeds 0-1: docs/runs/2026-10-03_reverse_order_seed0.md, docs/runs/2026-10-04_reverse_order_seed1.md
```
