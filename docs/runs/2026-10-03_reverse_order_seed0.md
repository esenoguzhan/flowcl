# C2 reverse order, seed 0: `inconclusive` (Object learned 1 pp under θ); Goal retained at position 2; the Spatial stage erased every earlier task under plain GPM

**Status: a pre-registered control, one seed (the development seed).** The registered verdict is
**`inconclusive`**, with the label `borderline_task_diagonal`. Every validity check passed, and
the adaptive comparison is `valid`.

| Cell (plain GPM) | Acquisition `R[j][j]` | θ_j | Retention | State |
|---|---|---|---|---|
| **position:** Goal after Object, `R[2][1]` | 100 [100, 100] | 0.85 | **98 [94, 100]** | `retained` |
| **task:** Object after Spatial, `R[3][2]` | **82 [70, 92]** (borderline) | 0.83 | 0 [0, 0] | `not_learned` |

- **The verdict turns on one episode.** Plain GPM learned Object to 41 of 50 episodes, while θ
  needs 42 (0.83 = 98% − 15 pp). By the registered point-estimate rule the task cell is
  `not_learned`, so the verdict is `inconclusive`. The rule is not overridden.
- **Goal at position 2 was not forgotten.** Its drop across Object's training was 0.02
  [0.00, 0.06]. The forward order's position-2 loss (Object 78 → 2% after Goal) did not recur at
  position 2 in this curriculum.
- **The large loss came at the last transition, and it hit every earlier task.** These are
  descriptive numbers; no registered cell covers them. Rollout-paired drops across Spatial's
  training:
  - LIBERO-10 94 → 20%: drop 0.74 [0.62, 0.86];
  - Goal 98 → 2%: drop 0.96 [0.90, 1.00];
  - Object 82 → 0%: drop 0.82 [0.70, 0.92].
- **Adaptive GPM kept all three.**
  - Final retention was 96 / 98 / 72%, against plain's 20 / 2 / 0%. The paired gains are
    +76 [+64, +88], +96 [+90, +100] and +72 [+60, +84] pp.
  - The cost was Spatial's plasticity: 84 against 98%, −14 [−26, −4] pp.
  - Trunk occupancy after the last task was 0.99, against plain's 0.61.

**Date (local, CEST):** Fri 2 Oct 22:02:45 → Sat 3 Oct 12:53:51.
- seq_ft 5.0 h;
- plain GPM 5.0 h;
- adaptive GPM 4.8 h;
- reports and summary 30 s.

**Code:** `d0bf349`, clean at queue start (0 changes). Every run's `git_sha` is `d0bf3495…`.
The rule `configs/analysis/reverse_order.yaml` (SHA-256 `f0f118b8…`) and the derived threshold
entries in `sequence_report.yaml` (SHA-256 `e9a9ae27…`) were committed in `d0bf349`. Both SHAs
were identical before and after the smoke run.

**Cite:** `results/reverse_order/seed0/summary.json` (SHA-256 `1c3df813…`). It holds the verdict,
cells, labels, every check, the adaptive comparison with its contrasts, the drops, the mechanism,
the matrices, and the unpaired forward numbers.

---

## 1. Design (as pre-registered)

**The question.** In the forward order (Spatial → Object → Goal → LIBERO-10), plain GPM forgot
Object at position 2 when Goal was trained. In the reverse order (LIBERO-10 → Goal → Object →
Spatial), Goal is at position 2 and Object at position 3. Which one does plain GPM forget?

**The runs.**
- seq_ft, plain GPM (`gpm`) and adaptive GPM (`gpm_ne90`), all on the forward runs' recipe:
  30k steps, lr 1e-4, batch 64, AMP, 50 episodes per cell, and `--single-task-baseline`.
- They share the namespace `seq_hetero_reverse__seq_ft__seed0`, so plain and adaptive GPM are
  paired with reverse seq_ft episode for episode.
- **Normalization statistics are fitted on LIBERO-10,** not Spatial.

**The rule.**
- θ_j = `R_seqft[j][j] − 0.15`: 0.83 / 0.85 / 0.83 / 0.81.
- **Two cells.** Position is `R[2][1]`, Goal after Object. Task is `R[3][2]`, Object after
  Spatial.
- **Cell states.** A cell is `not_learned` if plain GPM's diagonal is below θ. Otherwise it is
  `forgotten` if its retention is below θ, else `retained`.
- **Verdict:** `incomplete`, `invalid`, `inconclusive`, then `both` / `position` / `task` /
  `neither`.
- **These verdicts are descriptive patterns, not causes.** Reversal also changes predecessors,
  successors and statistics, and "position" coincides with the Goal → Object pair.

## 2. Validity (all passed)

| Check | Result |
|---|---|
| Completeness | three 4×4 matrices, 50 rollouts in every cell |
| Production settings | curriculum, seed, namespace, recipe, 50 episodes, method configs, clean `git_sha`: all three runs |
| Normalization statistics | one fingerprint (`7c823817…`) across all three runs; `fitted_on_task_id` is the LIBERO-10 task; every stage checkpoint matches its run's `stats.json` |
| T1 pairing | passed: relative weight difference 0.0000 (plain and adaptive against reverse seq_ft) |
| Identity, adaptive vs plain | stage 0: 0 of 656 tensors differ; stage 1: 0 of 656 |
| Episode pairing | identical per-episode seeds in every (stage, task), seq_ft vs plain and seq_ft vs adaptive |
| Sequence reports (plain and adaptive) | every provenance check passed (clean tree, namespace, T1 pairing, occupancy non-decreasing, memory chained, residuals within bound, artifact hashes, frozen from stage 1) |
| Thresholds | the reports' threshold blocks equal the summary's θ and the SHA-256 of reverse seq_ft's `result.json` |
| Smoke run (before the commit; scratch only, discarded) | the whole pipeline at 200 steps and 1 episode in 31 min. `QUEUE DONE ok`, every enforced check passed, dirty tree excused as defined |

## 3. Results

**The retention matrices** (rows: after stage; columns: LIBERO-10 / Goal / Object / Spatial; %):

| After | seq_ft | plain GPM | adaptive GPM |
|---|---|---|---|
| LIBERO-10 | 98 / 0 / 0 / 0 | 98 / 0 / 0 / 0 | 98 / 0 / 0 / 0 |
| Goal | 0 / 100 / 0 / 0 | 98 / 100 / 0 / 0 | 98 / 100 / 0 / 0 |
| Object | 0 / 0 / 98 / 0 | 94 / 98 / **82** / 0 | 94 / 100 / 82 / 0 |
| Spatial | 0 / 0 / 0 / 96 | **20 / 2 / 0** / 98 | **96 / 98 / 72** / 84 |

- **Overall performance (ACC over T1–T4):** seq_ft 0.24, plain GPM 0.30, adaptive GPM 0.875.
- **seq_ft** loses every earlier task at every transition, as expected.

**The registered drops** (rollout-paired; episode seeds are shared across stages):

| Drop | plain GPM | adaptive GPM | adaptive − plain (joint bootstrap over matched episodes) |
|---|---|---|---|
| Goal, `R[1][1] − R[2][1]` | 0.02 [0.00, 0.06] | 0.00 [0.00, 0.00] | −0.02 [−0.06, 0.00] |
| Object, `R[2][2] − R[3][2]` | **0.82 [0.70, 0.92]** | 0.10 [−0.08, 0.28] | **−0.72 [−0.90, −0.54]** |

**Final retention, adaptive − plain** (paired):
- LIBERO-10 +0.76 [+0.64, +0.88];
- Goal +0.96 [+0.90, +1.00];
- Object +0.72 [+0.60, +0.84].

Spatial's plasticity is −0.14 [−0.26, −0.04].

**The protection mechanism.** Medians over the trunk's layers, from each run's memory history:

| Memory update after | plain: energy already in memory | plain: target / captured | plain: trunk occupancy after | adaptive: target / captured | adaptive: trunk occupancy after |
|---|---:|---|---:|---|---:|
| LIBERO-10 (T1) | n/a | 0.95 / 0.95 | 0.49 | 0.95 / 0.95 | 0.49 |
| Goal (T2) | 0.904 | 0.95 / 0.950 | 0.56 | 0.990 / 0.991 | 0.82 |
| Object (T3) | 0.925 | 0.95 / 0.950 | 0.59 | 0.998 / 0.998 | 0.95 |
| Spatial (T4) | 0.935 | 0.95 / 0.950 | 0.61 | 0.999 / 0.999 | 0.99 |

- **Plain GPM's fixed target protects little of each later task's new input energy.** At each
  update the memory already holds 90–94% of the new task's trunk input energy, so a 0.95 target
  adds only the rest. Its trunk occupancy grows only from 0.49 to 0.61. This is the same
  under-protection pattern as in the forward order.
- **Adaptive's target protects 90% of the new energy.** Its occupancy reaches 0.99 after T4,
  which is where its Spatial plasticity cost appears.
- The decoder shows the same pattern: plain 0.05 against adaptive 0.54 median occupancy after T4.

**The forward order, unpaired** (a different namespace, task order and normalization; reported
only). Forward seed 0's final rows:
- plain GPM, Spatial / Object / Goal / LIBERO-10: 88 / 0 / 56 / 90;
- adaptive GPM: 98 / 84 / 94 / 92.

In the forward order, plain GPM's large loss was Object at position 2 across Goal (78 → 2%). In
the reverse order it was every earlier task across Spatial.

## 4. The pre-registered judgement

| Quantity | Value | Rule | Result |
|---|---|---|---|
| Position cell: Goal acquired | 1.00 | ≥ θ_1 = 0.85 | learned |
| Position cell: Goal after Object | 0.98 | < 0.85 → forgotten | **retained** |
| Task cell: Object acquired | **0.82** [0.70, 0.92] | ≥ θ_2 = 0.83 | **not_learned** (borderline: the CI straddles θ) |
| Task cell: Object after Spatial | 0.00 | not judged | n/a |
| weak_reference (seq_ft diagonal < 0.70) | 98 / 100 / 98 / 96% | | not set |

**Verdict: `inconclusive`** (summary exit 0, a valid finding), with labels
`[borderline_task_diagonal]`. Adaptive comparison: `valid`.

## 5. What this does and does not show

**It shows,** for seed 0 in the reverse order under the registered rule:
- the task at position 2 (Goal) was not forgotten by its successor (Object). The forward
  position-2 loss did not recur at position 2 in this curriculum;
- the task cell (Object after Spatial) cannot be judged: Object's acquisition fell one episode
  short of θ;
- **descriptively,** under plain GPM the Spatial stage erased all three earlier tasks, including
  LIBERO-10, which had been retained through two transitions;
- adaptive GPM prevents that loss (+72 to +96 pp of final retention), at −14 pp of Spatial
  plasticity and 0.99 trunk occupancy. That is the forward order's trade, seen in the reverse
  order.

**It does not show:**
- **anything causal about position or task.** Reversal also changes predecessors, successors and
  normalization. "Retained at position 2" is one task in one curriculum.
- **whether the loss at the last transition is Spatial-specific.** No registered cell covers it.
  The loss is consistent with the fixed target's small protected share (§3, mechanism), but the
  cause is not tested.
- **other seeds.** This is seed 0, the development seed.

## 6. Disclosures

1. **Rule timing.** The rule, the derived threshold entries, the analysis code and the queue were
   committed in `d0bf349` before the registered run. The queue started at 22:02:45 from a clean
   tree. The rule was locked by SHA before the smoke run and was unchanged after it.
2. **The verdict turns on one episode** of Object's acquisition (41 of 50 against 42 needed).
   - The registered rule decides on the point estimate, and the label discloses the borderline.
   - The borderline check conditions on the estimated θ and does not include θ's uncertainty:
     seq_ft's Object diagonal is 98 [94, 100].
3. **Acquisition.** Plain and adaptive GPM both learned Object to 82%, against seq_ft's 98%. GPM
   freezes non-registry parameters from T2 on and projects the gradient, so a lower T3
   acquisition is a plasticity cost.
4. **The final-transition loss** of LIBERO-10 and Goal lies outside the registered cells. It is
   reported descriptively, with rollout-paired CIs.
5. **The forward comparison is unpaired:** different namespaces, task orders and normalization
   statistics.
6. **Development seed.** Seeds 1–2 (W2) use the same rule. A matching valid verdict on all three
   would show the pattern recurs across those seeds.

## 7. Consequences

- **C2 seed 0:** registered `inconclusive`.
  - The position reading of the forward Object loss is not supported at position 2 in this
    curriculum (Goal was retained).
  - The task reading is unjudged.
  - Seeds 1–2 run in W2 with the same rule and queue: `bash scripts/queue_reverse_order.sh 1`,
    then `2`, about 15 h each. They have no `--single-task-baseline`, because no Gate 0
    references exist for those seeds.
- **For the thesis story:**
  - plain GPM's fixed target leaves later tasks' new energy largely unprotected in both orders;
  - which transition does the damage depends on the order: Object across Goal in the forward
    order, everything across Spatial in the reverse order;
  - adaptive allocation prevents both, at a capacity and plasticity cost.
  - This is the forward finding's trade, seen in the reverse order on one seed. It is descriptive.
- **For D1,** the reverse order adds a second order in which adaptive GPM's protection holds,
  with its cost on the last task.

## 8. Artifact map

```
results/reverse_order/seed0/summary.json                      # verdict, cells, labels, checks, adaptive
                                                             # comparison, drops, contrasts, mechanism,
                                                             # matrices, forward (unpaired) (SHA-256 1c3df813...)
results/seq_hetero_reverse__seq_ft__seed0/                    # reverse seq_ft
results/seq_hetero_reverse__gpm_projected_adam__seed0/        # plain GPM (t1_pairing.json)
results/seq_hetero_reverse__gpm_projected_adam_ne90__seed0/   # adaptive GPM (identity_stage{0,1}.json)
results/gpm_seq_reverse_seed0/report.json, results/gpm_seq_ne90_reverse_seed0/report.json
results/logs/queue_20261002_220245_reverse_order_seed0/       # queue.log and one log per step
configs/analysis/reverse_order.yaml (d0bf349; SHA-256 f0f118b8...), configs/analysis/sequence_report.yaml
flowcl/experiments/reverse_order.py, scripts/reverse_order.py, scripts/queue_reverse_order.sh,
tests/test_reverse_order.py
```
