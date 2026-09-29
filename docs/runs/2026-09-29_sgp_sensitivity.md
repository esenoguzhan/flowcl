# SGP sensitivity on the standard recipe — seed-0 bracket, frozen α = 400, not validated

**Status:** the pre-registered verdict is **`not_validated`**.
- The frozen strength **α = 400** passed on held-out seed 1 but **failed the plasticity floor on seed 2**:
  Object 76%, against a floor of 82% (fine-tuning's 92% − 10 pp).
- Under the no-rescue rule, no other α is selected with seeds 1–2, and the four-task step
  (step 6) does not run.

**In short:**
- **Protection matches GPM.** At α ≥ 400, SGP protects Spatial about as well as hard GPM on every
  seed (90–98%), with lower loss-level forgetting than GPM.
- **Plasticity is inconsistent** against both fine-tuning and GPM (Object +8, −10 pp against GPM on
  seeds 1 and 2).
- **The decisive question was not testable here.** Does SGP recover plasticity where GPM saturates?
  That needs the four-task sequence; T1 → T2 does not stress capacity.

**Dates (local, CEST), 29 Sep:** bracket 14:49 → 18:22; selection 18:22; validation 18:22 → 20:09;
summary 20:10.

**Code:** `3ee48a5`, clean. The rule `configs/analysis/sgp_sensitivity.yaml` was committed with it,
before any run. The reused arms come from their own clean runs:
- SGP α = 25: `40cfdda`;
- plain GPM: `0184293`;
- seq_ft: `8132abc`.

**The selection bundle:** `results/sgp_sensitivity/selection/`, with `selection.json` SHA-256
`1ee50fec…`. It was published atomically, and both validation runs verified and recorded it.

**Cite:** `results/sgp_sensitivity/summary.json`: every arm, the probe units, the nominal
attenuation, the realized update split, the selection and the verdict.

---

## 1. Design (as pre-registered)
- **The transition:** Spatial → Object at the standard recipe (30k steps, lr 1e-4), from each
  seed's seq_ft `stage0.pt`, paired with seq_ft's stage 1.
- **The T1 memory** comes from plain GPM's `memory_task0.pt`. Every new run first asserted that its
  start weights equal plain GPM's stage 0 bitwise. SGP's importance is Eq. 2 at each α.
- **The bracket (seed 0):**
  - **eligible:** SGP α ∈ {25, 100, 400, 1600}; α = 25 is reused from the SGP-baseline run;
  - **controls:** freeze-only (new, since the pilot's was a dirty-tree run), hard GPM (reused) and
    fine-tuning (reused).
- **The selection:** among eligible α only, the **smallest** α with Object ≥ fine-tuning's − 10 pp
  and Spatial ≥ hard GPM's − 10 pp. It was then frozen in an immutable bundle.
- **Validation** on seeds 1–2, which are held out from α selection. The same floors apply per seed,
  both seeds must pass, and there is no rescue tuning.

## 2. The bracket (seed 0)

| Arm | Object | Spatial | Object vs GPM | Spatial vs GPM | Spatial probe | Object probe | Nominal unprotected (trunk) | Realized trunk update: parallel / perpendicular |
|---|---:|---:|---|---|---:|---:|---:|---|
| fine-tuning | 78 | 0 | 0 | −94 | 0.725 | 0.0064 | — | 36.0 / 22.1 |
| freeze-only | 78 | 0 | 0 | −94 | 0.575 | 0.0068 | — | 35.6 / 21.6 |
| SGP α = 25 | 80 | 50 | +2 [−10, +16] | −44 [−58, −30] | 0.089 | 0.0076 | 9.7% | 13.8 / 24.4 |
| SGP α = 100 | **90** | 82 | +12 [+2, +22] | −12 [−24, 0] | 0.041 | 0.0084 | 2.9% | 6.2 / 26.3 |
| **SGP α = 400** | **86** | **90** | +8 [−6, +22] | −4 [−14, +8] | 0.031 | 0.0089 | 0.8% | 2.0 / 27.7 |
| SGP α = 1600 | 92 | 98 | +14 [0, +28] | +4 [−4, +12] | 0.028 | 0.0091 | 0.2% | 0.6 / 28.3 |
| hard GPM | 78 | 94 | — | — | 0.042 | 0.0072 | 0 | 0 / 47.6 |

- Success is in %, with 50 paired episodes; differences are in pp, with paired 95% CIs.
- The probe losses are the diagnostics' fixed batches, recomputed from each checkpoint.
- "Parallel" and "perpendicular" are the trunk update's norms inside and outside the T1 memory
  (ε = 0.95). The decoder follows the same order, with parallel fractions 0.56 (fine-tuning) →
  0.17 (α 25) → 0.06 → 0.02 → 0.005 → 0 (GPM).

**Selection:**
- α = 25 fails Spatial by 34 pp.
- α = 100 misses the Spatial floor (84%) by 2 pp.
- α = 400 and 1600 qualify.
- The smallest qualifying α is **400**, which was frozen.

## 3. Validation (α = 400, held-out seeds)

| Seed | Object (floor) | Spatial (floor) | Object vs fine-tuning | Object vs GPM | Spatial vs GPM | Result |
|---|---|---|---|---|---|---|
| 1 | 86 (≥ 80) | 98 (≥ 88) | −4 [−16, +8] | +8 [−4, +20] | 0 | pass |
| 2 | **76** (≥ 82) | 90 (≥ 84) | **−16 [−30, −2]** | −10 [−26, +6] | −4 [−12, +4] | **fail**: plasticity |

**Verdict: `not_validated`.** At the loss level, α = 400 behaves the same way on all three seeds:
- Spatial probe loss 0.030–0.032, against GPM's 0.039–0.043;
- Object probe loss 0.0088–0.0089, against GPM's 0.0071–0.0072;
- a realized trunk update of 1.9–2.0 parallel / 27.4–27.8 perpendicular.

## 4. Reading
1. **Protection is well controlled by α.**
   - The realized parallel update falls monotonically with α: the trunk parallel fraction goes
     0.49 → 0.23 → 0.07 → 0.02. It tracks the nominal attenuation's order.
   - Spatial retention rises with it: 50 → 82 → 90 → 98%.
   - The nominal share (η(1 − λ)) is only a scale. The realized split is what reaches the weights.
2. **Hard GPM is not the α → ∞ limit of our SGP.**
   - At α = 1600 almost every basis has λ ≥ 0.99, yet the run differs from GPM:
     - trunk update: relative 0.23 against 0.39, and perpendicular 28 against 48;
     - Spatial probe loss: lower (0.028 against 0.042);
     - Object probe loss: higher (0.009 against 0.007).
   - **Why:** our SGP removes only the *exactly* hard bases (λ = 1) from the gradient before AdamW.
     Adam's moments therefore still include the protected components, which dominate: fine-tuning's
     realized trunk update is 85% parallel by norm fraction. So Adam's normalized steps are smaller.
     GPM removes every memory direction before Adam.
   - So the bracket varies both the scaling and what the optimizer sees, and "SGP versus GPM"
     differences at high α are optimizer-coupling effects, not softness.
   - This belongs in the thesis methods and discussion.
3. **Freeze-only does not protect.** Spatial is 0%, as with fine-tuning; its loss-level forgetting is
   lower (0.575 against 0.725). The projection, not the §7.4 freezing, does the protecting.
4. **Plasticity is where it fails.**
   - Against fine-tuning, α = 400's Object is −4 and −16 pp on the held-out seeds, and −16 is
     significant.
   - Against GPM it is +8 and −10, and neither is significant.
   - Its Object probe loss is consistently about 25% above GPM's. At T1 → T2, SGP buys better
     loss-level retention with slightly worse loss-level plasticity, and the rollouts vary by seed.
5. **What was not tested.** SGP's intended benefit is plasticity at T3 and T4, when GPM's memory is
   nearly full (trunk 94% occupied entering T4). T1 → T2 only calibrates protection.

## 5. Disclosures
1. **The worst projection residual is not in the summary.** The plan listed it, but it was not
   implemented in the summary step.
   - The scaled-update invariant (SGP) and the orthogonality invariant (GPM) are enforced at every
     step during training, and a violation raises. Every projected run completed, so the invariant
     held throughout; only the per-run maximum is missing from the report.
2. **Seed selection history.** Seed 0 is the selection seed here, as in the earlier studies. Seeds
   1–2 were used by the GPM, optimizer and low-update studies, but no SGP outcome at α = 400 had
   been observed on them before validation.
3. **A pre-registration design input.** The design table of nominal attenuation per α was computed
   from the T1 memory before the rule was committed. It involves no training and no outcomes.

## 6. Consequences
- **Fixed-α SGP is not validated at T1 → T2** under the registered rule. No α is re-picked with seeds
  1–2 (no rescue tuning), and the four-task α = 400 sequence does **not** run as a validated method.
- **For the thesis:**
  - SGP's strength α controls realized protection monotonically and predictably;
  - at high α, SGP's coupling (Adam seeing the protected components) gives lower loss-level
    forgetting than GPM, at a small loss-level plasticity cost;
  - behavioural plasticity at T1 → T2 is seed-dependent, and the rule's fine-tuning-referenced floor
    was not met on one of the two held-out seeds.
- **Open decisions** (not automatic):
  - an **exploratory** four-task α = 400 run, clearly labelled as not validated, since the
    saturation question can only be asked there;
  - or closing SGP as a characterization, with α = 25 as the main-table baseline;
  - either way: a methods note on the SGP–GPM coupling difference, and on the weight-decay no-op
    from the optimizer study.

## 7. Artifact map
```
results/sgp_sensitivity/summary.json          # selection, verdict, bracket + validation rows (cells, paired vs
                                              # GPM / fine-tuning, probe units, nominal attenuation, realised split,
                                              # update size), consistency reference, abandoned attempts (none)
results/sgp_sensitivity/selection/            # the frozen bundle: selection.json (SHA 1ee50fec...) + selection.done.json
results/sgp_sensitivity/seed0/sens__{sgp_a100,sgp_a400,sgp_a1600,freeze_only}__seed0/
results/sgp_sensitivity/seed{1,2}/sens__sgp_a400__seed{1,2}/   # validation runs (method records the selection SHA)
results/sgp_sensitivity/validation_seed{1,2}.json
results/logs/queue_20260929_144951_sgp_sensitivity/
configs/analysis/sgp_sensitivity.yaml (3ee48a5), flowcl/experiments/sgp_sensitivity.py
```
