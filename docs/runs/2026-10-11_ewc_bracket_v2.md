# EWC λ bracket v2 (seed 0, T1 → T2): with the sampled Fisher and per-task normalization, Spatial is still 0/50 at every λ that keeps Object above its floor (r ≤ 100). The rule selects r = 100 (λ = 0.001227) by the Spatial probe-loss tie-break. Only r ≥ 1000 reaches 2/50 Spatial, and there Object falls to 0.44 and 0.36

**Status: the pre-registered follow-up, complete.**
- The queue ended `QUEUE DONE ok`, the verification passed, and the bundle is published.
- **The registered selection is λ = 0.0012270405152822335 (r = 100), flagged
  `all_eligible_spatial_zero`.**
  - Every eligible arm (r ≤ 100) has 0 Spatial successes.
  - The tie-break on Spatial probe loss picks r = 100. Here that is also the strongest eligible λ.
- No edge extension was owed: r = 10000 was not selected, because it is ineligible (Object 0.36).
- This bracket is a follow-up decided after v1's outcome (`runs/2026-10-10_ewc_bracket.md`). Its
  tie-break was adopted after seeing v1, which the rule's header discloses. v1's rule, rows, bundle
  and record stand unchanged.

| r | λ | Spatial `R[1][0]` | Object `R[1][1]` (floor 0.68) | Object − seq_ft [95% CI] | Spatial probe loss | Object probe loss | mean loss, last 50 (with penalty) | weighted penalty, last log |
|---:|---:|---:|---:|---|---:|---:|---:|---:|
| 0.01 | 1.227e-7 | 0.00 | 0.84 | +0.06 [−0.04, +0.16] | 0.699 | 0.0065 | 0.0077 | 0.0001 |
| 0.1 | 1.227e-6 | 0.00 | 0.82 | +0.04 [−0.06, +0.16] | 0.664 | 0.0064 | 0.0079 | 0.0004 |
| 1 | 1.227e-5 | 0.00 | 0.78 | +0.00 [−0.12, +0.12] | 0.581 | 0.0065 | 0.0093 | 0.0017 |
| 10 | 1.227e-4 | 0.00 | 0.70 | −0.08 [−0.20, +0.04] | 0.381 | 0.0071 | 0.0126 | 0.0043 |
| **100** | **1.227e-3** | **0.00** | **0.72** | −0.06 [−0.20, +0.10] | **0.165** | 0.0096 | 0.0184 | 0.0074 |
| 1000 | 1.227e-2 | 0.04 | 0.44 ✗ | −0.34 [−0.48, −0.20] | 0.137 | 0.0175 | 0.0309 | 0.0119 |
| 10000 | 1.227e-1 | 0.04 | 0.36 ✗ | −0.42 [−0.58, −0.24] | 0.189 | 0.0383 | 0.0513 | 0.0109 |
| *seq_ft (reference)* | | *0.00* | *0.78* | | *0.725* | *0.0064* | | |
| *plain GPM (reference)* | | *0.94* | *0.78* | | *0.042* | *0.0072* | | |

(λ is on the mean-normalized Fisher, so its scale is not comparable to v1's λ. The ratio r is.)

**What it shows:**
- **While Object stays above its floor (r ≤ 100), the penalty reaches the old task only on the
  loss.** Spatial's probe loss falls monotonically with λ up to r = 1000, from 0.70 to 0.14.
  Spatial success stays at 0/50 until r = 1000.
- **The first Spatial successes cost most of Object.**
  - At r = 1000 and r = 10000, Spatial is 2/50. The paired difference from seq_ft is +0.04
    [0.00, +0.10], so the interval touches 0.
  - Object falls to 0.44 and 0.36, 34 and 42 points below seq_ft. Its CIs exclude 0.
- **Stronger λ stops helping even the loss.** At r = 10000, Spatial's probe loss rises again (0.189
  against 0.137 at r = 1000). The last-logged weighted penalty is no larger (0.0109 against 0.0119),
  while Object's probe loss doubles.
- **EWC never approaches GPM's regime at any tested λ.** GPM reaches 0.042 probe loss and 94%
  success. EWC's lowest probe loss is 0.137, 3.3 times GPM's.

**v1 (empirical Fisher) and v2 (sampled, normalized) at the same r.** Both are calibrated so that r
means the same penalty ratio at the seq_ft endpoint.

| r | Spatial probe loss v1 / v2 | Object v1 / v2 | Spatial v1 / v2 |
|---:|---|---|---|
| 0.01 | 0.732 / 0.699 | 0.84 / 0.84 | 0 / 0 |
| 0.1 | 0.682 / 0.664 | 0.82 / 0.82 | 0 / 0 |
| 1 | 0.559 / 0.581 | 0.86 / 0.78 | 0 / 0 |
| 10 | 0.351 / 0.381 | 0.80 / 0.70 | 0 / 0 |
| 100 | 0.170 / 0.165 | 0.70 / 0.72 | 0 / 0 |

- **At equal calibrated strength the two estimators trace nearly the same trade-off.** Probe losses
  are within 0.035 at every r. Object differs by at most 0.10, within the rollout noise of 50 episodes
  (CI half-widths are about 0.12).
- **v1's null was not an artifact of the empirical Fisher**, within r ≤ 100 on seed 0. Changing the
  estimator, with normalization, changed neither Spatial success nor the shape of the loss trade-off.
- Neither pre-written interpretation for a selected arm with Spatial > 0 applies (at r ≤ 100, or
  only at r ≥ 1000): the arms with Spatial > 0 are ineligible.

**Fisher diagnostics** (descriptive; never gate; `results/baseline_selection_v2/ewc_fisher_diagnostics.json`,
1024 samples, two seeds per estimator):
- **Reproduction checks pass.**
  - The empirical seed-A Fisher equals v1's fingerprint (`320120a7…`).
  - The sampled seed-A Fisher equals the calibration's (`e16b00c7…`).
- **Reliability between seeds** (log-correlation / top-1% Jaccard): sampled 0.994 / 0.82, empirical
  0.984 / 0.71. Both are stable; the sampled Fisher's important set is more reproducible.
- **Agreement between the estimators:** 0.971 / 0.64.
- **Concentration** (share of ΣF in the top 1% / top 0.1% of parameters):
  - sampled: 84% / 68%;
  - empirical: 72% / 45%;
  - near-zero importance fractions 1.3% / 1.5%.
- **Shares by module:**

  | Fisher | flow head | vision encoder | trunk | text encoder |
  |---|---:|---:|---:|---:|
  | sampled | 85% | 8% | 7% | ~0 |
  | empirical | 66% | 19% | 15% | ~0 |

  The sampled Fisher concentrates protection even more on the flow head.

**The null, stated with its ranges kept separate** (the rule's wording): "0/50 Spatial successes at
every tested eligible setting on seed 0".
- **v1 (empirical Fisher):** r ∈ [0.01, 100], all eligible.
- **v2 (sampled Fisher, mean-normalized):** r ∈ [0.01, 10000]. It is eligible up to r = 100; beyond
  that Object falls below the floor.

No wording implies that both estimators covered the wider range. No claim goes beyond the tested
grid and seed.

**What the rule now implies:**
- **The 4-task EWC runs use λ = 0.0012270405152822335** (r = 100, sampled Fisher, mean
  normalization), committed to `configs/method/ewc.yaml` in its own commit.
- At that strength EWC loses Spatial as seq_ft does (0.00). It costs Object 6 points (CI includes 0)
  and cuts Spatial's probe loss 4.4 times.
- **Production EWC seed 0's stage 1 is identity-checked against this arm**
  (`--identity-reference-checkpoint 1=results/baseline_selection_v2/ewc/seed0/ewcb__r100__seed0/checkpoints/final.pt`).
  If it reproduces the arm, the cell is excluded from Track 2.
- The 4-task runs wait for Track 2's freeze.

**Limits:**
- one transition on the development seed, which also enters the final evaluation;
- AdamW, not the papers' SGD/RMSProp;
- the diagonal Fisher's known blind spots (Kirkpatrick et al., Fig. 4C);
- Spatial's 2/50 at r ≥ 1000 has a CI touching 0.

**Date (local, CEST):** Sat 10 Oct 16:32:55 → Sun 11 Oct 00:39:20 (8 h 06 min).
- The diagnostics took 2 min 44 s.
- Each arm took 67.4–71.2 min, of which training was 48.8–49.1 min (v1's was 48.5–48.9).
- The selection took 4 s and the verification 4 s.

**Code:** `1fee1f3`, clean at queue start. Every row's `git_sha` is `1fee1f3…`.
- **The rule** `configs/analysis/baseline_selection_ewc_v2.yaml` has SHA-256 `464e1335…`.
  - It was committed in two steps before any arm: `5f1d124` uncalibrated, `2c381cb` calibrated.
- **The calibration** `results/baseline_selection_v2/ewc_calibration.json` has SHA-256 `cc1f370f…`.
  - It ran at 15:40 on `5f1d124` and used no EWC-arm outcome.
  - D = 1236.44 on the mean-normalized sampled F_T1 (divisor 0.0015514), and
    L = 0.0075858106359373775.
  - The T1 Fisher came from 1024 samples, seed 4030082706, eps seed 97512238. Its fingerprint is
    `e16b00c7…` raw and `be10c73d…` normalized.
- **Every arm's T1 Fisher fingerprint equals the calibration's**, checked before training.
- **The full CPU suite on `2c381cb`:** 1035 passed and 1 failed.
  - The failure was `test_v2_pipeline_on_a_fake_world`: once the committed rule was calibrated, the
    test loaded it instead of calibrating its own copy.
  - `1fee1f3` fixed the test (one line, test only). The bracket test file passes at `1fee1f3` (21
    tests, rerun Sun 11 Oct).

**The smoke** (scratch, Sat 10 Oct 15:40–15:58, on `2c381cb`, 200 steps, 1 episode; `QUEUE DONE ok`):
- EWC v2 is bitwise seq_ft at stage 0;
- resume equality at stage 3: tensors, the cumulative EWC state and every evaluation episode;
- the diagnostics ran at 64 samples;
- a bracket arm's T1 Fisher equals the calibration's.

**Cite:**
- `results/baseline_selection_v2/ewc/selection/selection.json` (SHA-256 `f09bc3ee…`, inside the
  immutable bundle);
- the arms `results/baseline_selection_v2/ewc/seed0/ewcb__r{0.01,0.1,1,10,100,1000,10000}__seed0/row.json`;
- the diagnostics file (SHA-256 `76a1c74a…`);
- the calibration file above;
- the queue log `results/logs/queue_20261010_163255_ewc_bracket_v2/queue.log`.

---

## Design (as pre-registered; `configs/analysis/baseline_selection_ewc_v2.yaml`)

- **What changes from v1** (checked against Kirkpatrick et al. 2017 and Schwarz et al. 2018):
  - **The Fisher estimator.** v2 uses the Fisher of a unit-variance Gaussian velocity model,
    diag E[JᵀJ]. Per sample, the summed NLL 0.5·Σ_valid (v − y)² is taken with pseudo-targets
    y = v.detach() + ε drawn from the model, with ε from its own generator. Every valid element is
    weighted equally.
  - **Per-task normalization.** Each task's Fisher is divided by its mean over all trainable
    parameters (Schwarz et al. §4).
  - **The grid** is r ∈ {0.01, …, 10000}, calibrated as in v1 (λ_r = 2rL/D).
  - **The tie-break** among eligible arms is the lower Spatial probe loss, then the smaller λ. It was
    adopted after v1's outcome and is disclosed as such. The flag `all_eligible_spatial_zero` marks a
    best eligible success of 0.
- **Unchanged from v1:**
  - seed 0, starting from seq_ft's `stage0.pt`;
  - T2 trained with the standard recipe, through `train_stage` with the runner's text cache;
  - each arm's T1 Fisher checked against the calibration's fingerprint;
  - cells paired with seq_ft's stage 1;
  - eligibility: Object ≥ seq_ft's Object − 0.10 (0.68);
  - none eligible → the smallest λ, flagged;
  - an edge extension (r = 1e5) owed only if r = 10000 is selected under the full ordering;
  - the bundle bound to the rule's and the calibration's SHA-256.
- **Track 2:** every bracket cell is excluded from the held-out set, and so are the T1 cells shared
  with seq_ft. So is any production cell that reproduces one: EWC seed 0's stage 1, if its identity
  check holds.
