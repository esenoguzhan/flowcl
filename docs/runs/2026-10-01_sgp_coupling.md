# SGP update-rule coupling test at T4 (α = 800, seed 0): `mixed`, because the retention guard failed

**Status: exploratory, one seed, no claim level.** The pre-registered verdict is **`mixed`**, with
the flag `gain_with_forgetting`.
- **The AdamW cost is real and local to T4.** From A800's own stage-2 state and memory, the scaled
  projection fits T4 worse than hard projection under AdamW: c_AW = **+0.23 [+0.18, +0.31]**, a
  loss ratio of ×1.26.
- **Under the exact SGD update rule that cost is gone.** c_SGD = **−0.035 [−0.039, −0.033]**, a
  ratio of ×0.965, and the interaction is Δ = **−0.27 [−0.35, −0.22]**.
- **The retention guard failed.** That is the one condition between this result and
  `coupling_supported`. SGD-soft kept Object at 22% against SGD-hard's 36% (−14 pp, more than the
  10 pp allowed; the paired CI [−30, +2] includes 0). At the loss level, SGD-soft kept Object
  slightly *better*. The label follows the rule.
- **Exact SGD is not a practical alternative here.** The projected SGD arms underfit T4 (78–80%;
  probe loss 3× AdamW's) and lost more Object in rollouts than the AdamW arms.

**In short:**
- A800's T4 deficit against adaptive GPM matches what this coupling cost accounts for. Hard
  projection with A800's own memory, from A800's own state, reaches a T4 probe loss of 0.0131,
  against adaptive GPM's 0.0135 (descriptive: these are different runs).
- Under AdamW, the scaled projection trades T4 plasticity for retention. Its realized update
  outside the memory is 35% smaller in the trunk than hard projection's, and it keeps Goal better
  (+14 pp [+2, +26]).
- Under vanilla SGD, the update outside the memory is about the same with and without scaling,
  and the scaled directions add a little T4 fit. This is the pattern the update-rule hypothesis
  predicts.
- The SGD arms also drop gradient clipping. The finding is therefore that **the exact SGD update
  rule removes the cost seen under our AdamW adaptation**. It is not that "AdamW alone causes it".
- The four-task SGD follow-up required `coupling_supported` together with a Q2 gain. That
  condition is not met, so the follow-up is not triggered.

**Dates (local, CEST):**
- **Attempt 1:** 30 Sep 14:24 → 15:28. The reproduction gate failed, and the queue skipped every
  step after it (§2).
- **Attempt 2:** 30 Sep 21:19 → 1 Oct 04:33:
  - `aw_soft` until 22:23, when the gate passed;
  - the pilots, 22:23 → 22:54, then the selection;
  - `aw_hard` until 23:58; `aw_free` until 01:10;
  - `sgd_free` until 02:20; `sgd_hard` until 03:25; `sgd_soft` until 04:32;
  - the summary, 04:33.

**Code:**
- **`7a73484`** (30 Sep 14:24:05) holds the rule (`configs/analysis/sgp_coupling.yaml`, SHA-256
  `f6916d1e…`) and the implementation. It was committed before any run.
- **`1e5ebf6`** (30 Sep 21:19:38) holds the text-cache fix (§2). It changes no rule: the rule
  file's SHA is unchanged.
- Every attempt-2 run, pilot, the selection and the summary record `1e5ebf6`, on a clean tree.

**Cite:** `results/sgp_coupling/summary.json`. It holds the verdict, the quantities with their
intervals, every arm, the gate and the selection.

---

## 1. Design (as pre-registered)

- **Every arm trains only T4 (LIBERO-10).** It starts from A800's `stage2.pt`, with A800's T3
  memory and importances (`memory_task2.pt`, SHA-checked).
  - It uses seq_ft's stream seed for stage 3, 30k steps (warmup plus cosine), a fresh optimizer,
    and the §7.4 freeze (registry weights only).
  - It evaluates all four tasks with 50 paired episodes, under the same episode seeds as every
    stage of A800, plus the diagnostics' 16 fixed probe batches.
- **The arms:**

  | | No projection (freeze only) | Hard (λ = 1, same memory) | Scaled (A800's λ) |
  |---|---|---|---|
  | AdamW, the recipe (lr 1e-4, clip 1.0) | `aw_free` | `aw_hard` | `aw_soft` = A800 |
  | Vanilla SGD (no momentum, no weight decay, no clipping) | `sgd_free` at η_free | `sgd_hard` at η_proj | `sgd_soft` at η_proj |

- **Why vanilla SGD:** under it, our adaptation ("remove the hard directions, then scale the
  realized step") is exactly ΔW = −ηG(I − MΛMᵀ). A unit test checks this, and also shows that
  clipping or AdamW breaks it.
- **The reproduction gate:** `aw_soft` must reproduce A800's T4 exactly. That means every tensor,
  the probes, the final and mean-last-50 losses, every logged per-layer `gradient_c` / `update_c`,
  and the 601 printed losses.
- **The SGD rates are matched mechanically:** η = median over t ∈ [100, 1000) of
  ‖ΔW_t‖ / (f_t ‖G_t‖), measured on AdamW pilots.
  - η_free comes from `aw_free`. η_proj comes from `aw_hard` and is **shared** by SGD-hard and
    SGD-soft.
  - Each rate is checked by 3,000-step pilots at 0.5×, 1× and 2×, using T4 losses only. 1× is
    chosen if it is viable; otherwise the nearest viable scale.
- **The quantities and the verdict order** are those of the config header: invalid reproduction,
  then no projection cost, then no T4 coupling cost, then inconclusive SGD, then coupling
  supported, otherwise mixed.

## 2. Attempt 1: the gate failed, on the text-encoder cache alone

- **The failure.** Every gate check failed: 90 of 656 tensors differed, as did 596 of 601 printed
  losses and 53,693 of 54,600 c-log values. The first loss was 1.588252, against A800's 1.588263.
  `gradient_c` already differed at step 0, so the two runs diverged in the first forward pass, not
  through training.
- **The cause.** The frozen text encoder caches each instruction's embedding the first time it
  sees it, in whatever precision is active at that moment.
  - The continual runner evaluates **every** task after every stage, so A800's T4 instruction was
    first encoded in fp32 by the stage-2 rollouts.
  - A fresh stage process instead first encoded it inside an AMP training step.
- **The fix** (`1e5ebf6`): the opt-in `warm_text_cache` encodes the stage's instruction in fp32
  before training, as the runner's evaluations did. Only this study turns it on. It is recorded
  in every row and refuses to reuse a run trained without it.
- **Verification before relaunch** (A800's own configuration, first 101 steps):
  - 364 of 364 logged c-values at steps 0 and 100 bitwise equal;
  - the printed losses at steps 1, 51 and 101 equal;
  - the start probe exactly A800's (1.4345726380863606).
- **Attempt 1's outputs are kept** at `results/sgp_coupling_attempt1_invalid_reproduction/`.

## 3. Provenance and validity (attempt 2): all passed

| Check | Result |
|---|---|
| Reproduction gate: tensors against A800's `stage3.pt` | **0 of 656** differ |
| Gate: probe losses (4 tasks) against A800's diagnostics `L[3][j]` | equal (e.g. T4 0.016619) |
| Gate: final and mean-last-50 training losses | exactly A800's |
| Gate: printed losses / logged c-values | **601 of 601** / **54,600 of 54,600** equal |
| Gate, descriptive: rollouts against A800's stage 3 | 2 Spatial episodes differ; Object, Goal and LIBERO-10 identical |
| Start checks (references' SHA-256, method, recipe, memory artifact, probe, T4 threshold) | pass |
| Pilots' T4 start probe against A800's `L[2][3]` | identical (1.43457) |
| Selection bundle | published atomically (SHA `9012dd31…`), recorded by every SGP arm |
| Projection residuals (worst residual / bound) | 0.027–0.029 (AdamW), 0.067–0.076 (SGD) |
| Frozen tensors unchanged | every arm |

The two differing Spatial episodes fit the same cache mechanism.
- In the runner, T1's instruction was cached during T1's AMP training.
- Here, it is encoded in fp32 at evaluation.
- The weights are identical, and the gate does not score rollouts.

## 4. Rate calibration and pilots

| Rate | Matched η (q10–q90 over steps) | 0.5× | 1× | 2× | Selected |
|---|---|---|---|---|---|
| free (from `aw_free`) | 0.264 (0.178–0.350) | viable: T4 probe 0.048 (AdamW 0.033) | **diverged** (3.8 × 10⁶) | diverged (4.6 × 10⁴) | **0.5× → lr 0.132** |
| proj (from `aw_hard`) | 0.590 (0.395–0.829) | viable: 0.080 (AdamW 0.045) | viable: 0.071 | viable: 0.062 | **1× → lr 0.590** |

- Unprojected SGD diverges at the norm-matched rate.
- The implied per-layer rates show why one global rate can't follow Adam's per-layer scaling:
  - for the free rate: trunk median 0.49, decoder median 0.66, maximum 19.5;
  - for the proj rate: decoder median 1.59, maximum 339.
- The registered rule stepped the free rate down to 0.5×. The projected rate stayed at 1×.

## 5. Results

**After T4** (success %, 50 paired episodes; probe losses on the 16 fixed batches):

| Arm | Spatial | Object | Goal | LIBERO-10 | Spatial probe | Object probe | Goal probe | **T4 probe** |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `aw_soft` (= A800) | 98 | 64 | 88 | 88 | 0.037 | 0.036 | 0.028 | 0.0166 |
| `aw_hard` | 96 | 56 | 74 | 96 | 0.040 | 0.048 | 0.043 | **0.0131** |
| `aw_free` | 0 | 0 | 0 | 90 | 0.763 | 1.105 | 1.250 | 0.0080 |
| `sgd_free` (lr 0.132) | 10 | 0 | 0 | 92 | 0.171 | 0.286 | 0.370 | 0.0166 |
| `sgd_hard` (lr 0.590) | 88 | 36 | 86 | 78 | 0.043 | 0.037 | 0.026 | 0.0407 |
| `sgd_soft` (lr 0.590) | 86 | **22** | 88 | 80 | 0.041 | 0.035 | 0.026 | 0.0393 |
| *for reference: stage 2 (start)* | 92 | 70 | 94 | 0 | | | | 1.435 |
| *adaptive GPM, stage 3 (other run)* | 98 | 84 | 94 | 92 | | | | 0.0135 |

**Scaled against hard, per update rule** (paired episodes; pp):

| | Spatial | Object | Goal | LIBERO-10 |
|---|---|---|---|---|
| AdamW | +2 [0, +6] | +8 [−6, +22] | **+14 [+2, +26]** | −8 [−18, +2] |
| SGD | −2 [−14, +10] | −14 [−30, +2] | +2 [−10, +14] | +2 [−14, +16] |

**Change from the common stage-2 scores** (paired; pp; Spatial / Object / Goal):
- `aw_soft`: +6 / −6 / −6;
- `aw_hard`: +4 / −14 / **−20**;
- `sgd_hard`: −4 / **−34** / −8;
- `sgd_soft`: −6 / **−48** / −6;
- both unprojected arms: −70 to −94 on every old task.

## 6. The pre-registered quantities and the verdict

| Quantity (T4 probe loss) | Value [95% CI, matched batches] | Ratio | Label |
|---|---|---|---|
| Premise: gap_AW = ln(L_aw_hard / L_aw_free) | +0.497 [+0.390, +0.658] | ×1.64 | hard projection costs T4 plasticity ✓ |
| Q1: c_AW = ln(L_aw_soft / L_aw_hard) | **+0.234 [+0.185, +0.308]** | ×1.26 | **cost** |
| Q2: c_SGD = ln(L_sgd_soft / L_sgd_hard) | **−0.035 [−0.039, −0.033]** | ×0.965 | **gain** |
| Δ = c_SGD − c_AW | **−0.270 [−0.346, −0.218]** | | below 0 ✓ |
| SGD viable: rates selected, `sgd_free` T4 ≥ 83% | 92% | | ✓ |
| Retention guard: `sgd_soft` ≥ `sgd_hard` − 10 pp (Spatial / Object / Goal) | 86 ≥ 78 ✓ / **22 < 26 ✗** / 88 ≥ 76 ✓ | | **fails** |
| Descriptive: ln(L_sgd_soft / L_aw_soft) | +0.86 [+0.74, +1.02] | ×2.36 | |

**Verdict: `mixed`** ("none of the pre-registered patterns"), flag `gain_with_forgetting`.
- Every condition of `coupling_supported` holds except the retention guard on Object.
- That guard compares rollout point estimates. The paired Object difference is −14 pp [−30, +2],
  and the Object probe loss favours `sgd_soft` (0.0348 against 0.0369).
- Behaviour and loss disagree, as in earlier studies. The label is not changed.

## 7. Mechanism: measured, not causal

**The realized T4 update**, split against A800's T3 memory (norms; trunk / decoder):

| Arm | Total | Inside the memory: hard | Inside the memory: scaled | Free (outside) | Relative update | `update_c` (trunk) |
|---|---|---|---|---|---|---|
| `aw_soft` | 37.4 / 38.4 | 0 / 0 | 2.3 / 3.8 | **37.3** / 38.2 | 0.31 | 0.87 |
| `aw_hard` | 57.3 / 43.7 | 0 / 0 | 0 / 0 | **57.3** / 43.7 | 0.42 | 0.19 |
| `sgd_soft` | 14.6 / 10.5 | 0 / 0 | 0.9 / 1.3 | **14.5** / 10.5 | 0.10 | 0.88 |
| `sgd_hard` | 15.3 / 11.5 | 0 / 0 | 0 / 0 | **15.3** / 11.5 | 0.11 | 0.01 |
| `aw_free` | 41.9 / 27.4 | 39.5 / 20.6 | 12.3 / 11.3 | 6.9 / 14.1 | 0.29 | — |

1. **Under AdamW, the scaled projection shrinks the useful step.**
   - 87% of AdamW's raw trunk step lies inside the memory, against 19% under hard projection.
   - The post-step scaling then removes most of it. What reaches the free directions is 35%
     smaller in the trunk (37.3 against 57.3) and 13% smaller in the decoder.
   - This coincides with the worse T4 fit (×1.26) and the better retention (Goal +14 pp).
2. **Under SGD, scaling does not touch the free component.**
   - SGD-soft's free update about matches SGD-hard's: −5% in the trunk, −9% in the decoder. The
     difference comes from diverging trajectories, since each step's free component is identical
     by construction.
   - The scaled directions add a small component (0.9 / 1.3) and a 3.5% better T4 fit.
3. **Matched initial step norms did not mean matched total movement.**
   - SGD steps shrink as the gradient shrinks, while Adam keeps normalizing. So the SGD arms moved
     about a third as far in total (relative update 0.10–0.11, against 0.31–0.42).
   - This is why only the within-rule contrasts (Q1, Q2, Δ) are interpreted. The SGD-against-AdamW
     contrast confounds update size and update rule.
4. **A note on the unprojected AdamW arm:** 99% of its trunk update norm lies inside the T3
   memory (94% along its hard directions). The memory captures where T4's gradient wants to go,
   which is what makes the projection costly (the premise).

## 8. What this does and does not show

**It shows,** for one seed, at α = 800, with A800's T3 memory and from A800's stage-2 state:
- Under our AdamW adaptation, the scaled projection costs T4 plasticity against hard projection
  with the same memory, and buys some retention.
- With the exact SGD update rule (no momentum, no weight decay, no clipping), that cost is absent.
  The scaled projection then fits T4 slightly better than hard, with a large interaction (Δ CI
  well below 0).
- The SGD-soft gain came with a rollout-level Object deficit that exceeds the pre-registered
  tolerance, though not significantly, and with a loss-level Object advantage.

**It does not show:**
- **That AdamW alone is responsible.** The SGD arms also drop clipping, and their step sizes
  evolve differently.
- **That exact SGP would work in the four-task setting.** The projected SGD arms underfit T4
  (78–80% success, a probe loss 3× AdamW's) and lose much more Object than the AdamW arms (−34
  and −48 pp from stage 2, against −14 and −6).
- **Anything beyond one seed, one α, one transition (T3 → T4) and one memory.**

## 9. Disclosures

1. **Attempt 1 and the fix** (§2). The fix (`1e5ebf6`) was written after attempt 1's gate failure
   and before attempt 2.
   - The rule file is unchanged.
   - Attempt 1 produced only a full `aw_soft` arm, a near-copy of A800's T4. No other arm's
     outcome existed before attempt 2.
   - The fix was verified on 101 steps of A800's own configuration.
2. **The smoke run before attempt 1.** It ran 200-step pilots into scratch, which showed T4 probe
   losses for five configurations, including SGD-soft and SGD-hard (0.188 against 0.195).
   - These were seen after the rule was locked, and its SHA was unchanged.
   - No old-task quantity was computed.
3. **The free rate's step-down** (1× diverged, so 0.5× was used) followed the registered rule.
4. **The guard's Object failure** is a point-estimate rule. The disagreement between rollouts and
   loss is reported, and the label is kept.
5. **α = 800 was chosen outside the registered bracket** (`docs/runs/2026-09-30_sgp_a800_seed0.md`).
   This is one seed, and exploratory.
6. **Earlier stage-path studies** (sensitivity, low-update, optimizer, projection pilot) trained
   with the same text-cache state as attempt 1. Their new runs therefore differ from
   runner-trained stages at the level of a different floating-point realization: a few episodes
   per task here. This is not a systematic bias. It matters only where those studies compared new
   runs with reused runner arms. This goes in the methods note.

## 10. Consequences and open decisions

- **For the thesis** (one seed, exploratory):
  - our SGP adaptation's T4 plasticity cost is an update-rule coupling effect: it is present
    under AdamW and absent under the exact SGD rule;
  - switching to exact SGD does not make scaled projection practical here.
  - Together with the sensitivity study, the α 800 run and the α = 25 seed-0 record, SGP's
    characterization is: α controls protection monotonically, and under AdamW the scaled
    directions take Adam's step budget, so near saturation SGP behaves like GPM with a smaller
    effective step.
- **The follow-up rule is not met:** the four-task SGD pair required `coupling_supported`
  together with a Q2 gain.
- **Open decisions** (not automatic):
  - close SGP as a characterization, with α = 25 as the registered main-table baseline. Seeds 1–2
    of the baseline would complete that row.
  - methods notes: the weight-decay no-op, the SGP–GPM optimizer coupling (this record), and the
    stage path's text-cache state;
  - not planned: a coupling-aware AdamW variant, for example one that removes the scaled
    components from what Adam's moments see. It would be a new method, not paper SGP.

## 11. Artifact map
```
results/sgp_coupling/summary.json                   # verdict, quantities + CIs, arms (cells, probes, update split,
                                                    # projection logs, paired contrasts), gate, selection
results/sgp_coupling/reproduction.json              # the gate: tensors, probes, losses, step log, c-logs, rollouts
results/sgp_coupling/selection/                     # frozen rates (selection.json SHA 9012dd31..., + done marker)
results/sgp_coupling/pilots/pilot__*__seed0/        # 2 AdamW calibration pilots (step norms) + 6 SGD pilots (T4 only)
results/sgp_coupling/arms/coup__{aw,sgd}_{soft,hard,free}__seed0/   # the six arms (row.json, final.pt, eval, losses)
results/sgp_coupling_attempt1_invalid_reproduction/ # attempt 1 (gate failed: text-cache state)
results/logs/queue_20260930_211938_sgp_coupling/    # attempt 2 queue (attempt 1: queue_20260930_142424_sgp_coupling/)
configs/analysis/sgp_coupling.yaml (7a73484), flowcl/experiments/sgp_coupling.py (1e5ebf6)
```
