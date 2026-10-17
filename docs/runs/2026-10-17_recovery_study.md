# Recovery measurement study: GO. All four tasks pass both questions at 95%. Closed-loop harm from action perturbations is concentrated by action component and phase, and it is predictable beyond the perturbation's size

**Status: published** (Sat 17 Oct 04:34), outcome **GO** at the rule's 95% level.
- **What GO means here** (pre-written in the rule): "action vulnerability is structured within the tested
  perturbation family".
- It authorizes **an integrated-action Jacobian and capacity feasibility study**: measured criticality
  against uniform sensitivity, at matched storage. It does **not** authorize a method.
- **Scope:** one seed's policies, simulation only. Nothing here transfers to AgileX without its own
  validation.

**Provenance:**
- **Rule:** `configs/analysis/recovery_study.yaml`, SHA-256 `88a8a72b…`.
  - Committed in `2a31198`, before any perturbed rollout.
  - Amended in `45b0d08`, before any production continuation (the σ fallback, below).
- **Code:** `45b0d08`, clean tree. The full CPU suite was green at that commit (1113 passed).
- **Pre-run validation,** re-run under the amended rule at `45b0d08`, is identical to the first run:
  - false GO 0/200 under every null in both regimes (Clopper–Pearson upper bound 0.018), so production
    ran at the 95% level;
  - power at an effect of 2 logits, main regime: 0.91 / 1.00 / 0.98 (phase-only / component-only / mixed).
  - Copy: `docs/runs/2026-10-16_recovery_study_validation.json`.
- **Data:** 18,768 continuations under `results/recovery_study/.work/`, bound to the rule, checkpoint and
  git SHAs.
- **Analysis:** `results/recovery_study/analysis/analysis.json`, SHA-256 `33ac9f44…`.
  - A dry-run re-analysis of the same work files reproduces it exactly (provenance aside).
- **Queue:** `results/logs/queue_20261016_160727_recovery/`, 4 workers, `QUEUE DONE ok`.
  - The workers ran from 16:07 to 04:25 (12.3 h; the smoke had projected about 9 h). The analysis took
    10 min.

## The design, in brief (the rule is the specification)

**Policies:** for each task, the seq_ft seed-0 checkpoint just after training on it. Settings are the
production ones (Euler 10, `execute_k` 8, 600 steps).

**Nominal rollouts:** 12 per task, from the training demos' initial states (successes: 12 / 11 / 12 / 11).
Each successful rollout gets 6 queries, at the replans nearest 1/12, 3/12, …, 11/12 of its length.

**Perturbations,** applied to the first 8 executed actions of the chunk sampled at the query:
- 12 structured: position, rotation or gripper × ± × 1σ or 3σ, where σ is the per-dimension SD of the
  task's demos;
- 4 isotropic.

**Continuations:** 4 continuation seeds per query, shared with the unperturbed (nominal) continuation.
Each continuation rebuilds the query state exactly by reset and replay.

**Injury:** the nominal continuation succeeds and the perturbed one fails. The analysis uses the
**controller-relevant** change: the clipped position/rotation change plus the gripper's sign flips.

**Q1 (concentration):** cross-fitted S20 over (rollout × query × group) cells. Under uniform risk it is
0.2.

**Q2 (beyond magnitude):** the leave-one-rollout-out log-loss gain of R over U.
- U: nominal failure plus a size spline.
- R: U plus flips and phase × group.

**Intervals:** a full-procedure cluster bootstrap over rollouts, 2000 resamples.

## Results

| task | rollouts | queries | injuries (rollouts) | S20 [95%] | Q2 gain, nats [95%] | pass 95% | pass 99% |
|---|---|---|---|---|---|---|---|
| Spatial (bowl → plate) | 12 | 72 | 563 (12) | 0.65 [0.59, 0.70] | 0.098 [0.051, 0.122] | **yes** | yes |
| Object (milk → basket) | 11 | 66 | 564 (11) | 0.71 [0.66, 0.76] | 0.129 [0.086, 0.152] | **yes** | yes |
| Goal (middle drawer) | 12 | 72 | 328 (12) | 0.94 [0.85, 0.98] | 0.107 [0.074, 0.125] | **yes** | yes |
| libero_10 (stove + moka) | 11 | 66 | 382 (11) | 0.73 [0.56, 0.89] | 0.044 [0.004, 0.066] | **yes** | no (Q2 [−0.012, 0.074]) |

**The outcome rule** (GO needs at least 2 usable tasks passing both Q1 and Q2):
- 4 of 4 tasks pass at 95%.
- At 99%, 3 of 4 pass, so GO would also hold at 99%. That is descriptive only; the rule's level is
  95%.
- Every task is usable: injuries occur in every successful rollout.

**Determinism and the negative control:**
- **The determinism gate passed on all four tasks:** the interior suffix was reproduced from the saved
  generator state, and outcomes did not depend on order. `hard_reset` was on and cuDNN deterministic.
- **No-ops:** 3,246 continuations whose perturbation changed nothing the controller sees (gripper
  pushes that flip no sign).
  - 0 mismatches: actions, success and steps were all identical to the nominal continuation.
  - 0 injuries out of 3,185 at risk; 95% interval [0, 0.0012].
- **So the paired design has no noise floor:** every injury is caused by the perturbation.

## What the data show (descriptive)

**Injury rate by group** (effective, at-risk trials):

| task | position | rotation | gripper (flips) | isotropic | 1σ (all) | 3σ (all) | rescues |
|---|---|---|---|---|---|---|---|
| Spatial | 0.27 | 0.01 | 0.29 | 0.14 | 0.05 | 0.24 | 61 |
| Object | 0.26 | 0.03 | 0.41 | 0.14 | 0.06 | 0.26 | 33 |
| Goal | 0.22 | 0.01 | 0.00 | 0.06 | 0.01 | 0.15 | 0 |
| libero_10 | 0.19 | 0.04 | 0.03 | 0.13 | 0.05 | 0.17 | 99 |

**Injury rate by phase** (queries at 1/12 … 11/12 of the rollout):

| task | group | 1/12 | 3/12 | 5/12 | 7/12 | 9/12 | 11/12 |
|---|---|---|---|---|---|---|---|
| Spatial | position | 0.29 | 0.42 | 0.16 | 0.25 | 0.29 | 0.23 |
| | gripper | 0.04 | 0.31 | 0.10 | **0.79** | 0.48 | 0.05 |
| Object | position | 0.38 | 0.50 | 0.14 | 0.30 | 0.18 | 0.10 |
| | gripper | 0.00 | 0.33 | 0.73 | **0.91** | 0.50 | 0.00 |
| Goal | position | 0.38 | 0.34 | 0.37 | 0.11 | 0.04 | 0.06 |
| libero_10 | position | 0.34 | 0.06 | 0.23 | 0.29 | 0.18 | 0.05 |

Rotation stays at or below 0.09 at every phase. Goal's gripper is 0.00 throughout.

**Two kinds of structure:**
1. **By component:** position offsets harm; rotation offsets of the same size in σ units almost never do.
2. **By phase within a component:**
   - A gripper flip is harmless before the grasp and after the place, and fatal while the object is
     carried (Object 0.91, Spatial 0.79).
   - Goal's position offsets matter while the arm approaches the drawer, not once it is pulling.

**Temporal S20 alone** (cells = rollout × query; descriptive) is 0.37 / 0.43 / 0.46 / 0.50. So phase
concentration exists, but it is weaker than the concentration over phase × component cells.

## Caveats (read before using the result)

1. **The component contrast depends on the σ scaling, as the rule fixed it.**
   - Rotation's demo SD is small: 0.03–0.11 in action units, against 0.23–0.50 for position. So a 3σ
     rotation offset is a much smaller command than a 3σ position offset.
   - "Rotation is harmless" therefore means *harmless per demo SD*, not per radian.
   - The **phase structure within a component** (the gripper while carrying, Goal's approach) cannot come
     from scaling. It is the more robust part of the finding.
   - **The feasibility study must fix its output metric explicitly** (demo SD, action range or task
     space), and should check that the critical set survives the choice.
2. **Much of the structure is mechanically obvious:** dropping a carried object fails the task. GO says
   that harm is structured and predictable beyond size. It does not say that protecting that structure
   leaves useful capacity free. That question belongs to the feasibility study.
3. **Gripper trials are effectively 3σ against the current sign.**
   - The gripper action is ±1 and σ_grip ≈ 0.98, so a 1σ push, or a push in the current direction,
     flips nothing: those are the no-ops.
   - Almost every effective gripper trial flips all 8 steps.
   - Goal's drawer task never closes the gripper. Its σ_grip is the pooled fallback (the amendment), and
     its flips are harmless (0/288).
4. **libero_10 is the weakest task.**
   - Q2 0.044, whose interval nearly reaches 0, and it fails at 99%.
   - It has the most rescues (99): its nominal continuations fail sometimes (11 of 264), and perturbations
     can rescue them.
5. **One seed, single-task checkpoints** (each policy is evaluated on the task it was just trained on),
   simulation only, a constant 8-step offset family. A different family (e.g. noise, delays) is not
   covered.
6. **Q2's flip categories are in R, not U** (disclosed in the rule before any data). In U they would have
   absorbed the gripper-specific harm as "magnitude".

## Incidents and amendments (all before production; recorded in the rule and commits)

- **The σ fallback** (`45b0d08`). The smoke found that the Goal drawer task's gripper SD is exactly 0 (it
  never closes). A zero-SD dimension now uses the SD pooled over the four tasks' demos. The validation
  was re-run under the new rule SHA, with identical results.
- **BLAS oversubscription** (`079e844`). 18 validation workers × 24 BLAS threads stalled the machine.
  Simulation workers now use one BLAS thread each.
- **Runtime:** 12.3 h, against the smoke's 9 h projection.
  - The 10 h budget only chose the regime (main or fallback). The main regime was kept, as the smoke's
    projection required.
  - The projection came from task 0 at an early and a late query. Under 4 concurrent workers, Spatial
    took about 8.4 s per continuation.

## Relation to the plan

- **D2's angle B** (`thesis_plan.md`) needs critical outputs.
  - Track 2's freeze found none offline: position ranked slightly better than M0, and the gripper ranked
    worse.
  - This study finds them in closed loop: position, plus the gripper while carrying, with rotation nearly
    inert per σ. The offline and closed-loop views agree on position and disagree on the gripper. The
    gripper's harm is phase-specific, which an average loss misses. This is an exploratory note.
- **Per the pre-written interpretation,** GO authorizes the integrated-action Jacobian and capacity
  feasibility study (measured criticality against uniform sensitivity, at matched storage), and nothing
  more.
  - The method it could lead to is angle B written explicitly: Uᵀ ΔW M = 0, with U from the measured
    critical outputs, **replacing** GPM's ΔW M = 0.
  - Any such method includes the uniform-sensitivity control.
- **Timing:** D2 is at the end of W5. The feasibility study needs its own plan and rule, and it competes
  with the seed-0 baseline queue (Track 2's held-out test set) for the GPU.
