# Track 2, step 3: THE FREEZE. On the development cells M0 already ranks well within tasks (C_a 0.82) and calibrates across tasks to 8 pp. The rule carries `m2_position` for ranking and `m3_mean` for calibration to the held-out test, but both development gains are small and their intervals include 0

**Status: frozen** (Thu 15 Oct 21:59), 13 days before the Wed 28 Oct target.
- The 4-task baseline runs, the held-out test set, may now be queued.
- The held-out claim is decided on them under the rule written here. Nothing below is a claim about the
  test set.
- **Development results only.** The cells share roots, so neither the 449 cells nor the 24 roots are
  independent training replications. Every interval is a two-level cluster bootstrap.

**Provenance:**
- **Rule:** the `selection` section of `configs/analysis/behavior_predictors.yaml`.
  - Committed in `1a6b6de`, **before any fitting.** No measure had been related to rollout success
    before it.
  - Rule-file SHA-256 `c2f3ed92…`.
- **Code:** `b941815`, with the full CPU suite green at that commit (1101 passed). Clean tree.
- **Inputs, bound by SHA:** the manifest `21c91ebc…` and the measures `5be958cc…`.
- **Bundle:** `results/track2/selection/`, with `selection.json` SHA-256 `b397bc0a…` and `summary.json`
  `4676a576…`.
- **Verification:** `--verify-selection` recomputes it bitwise (passed). The run took 13.6 s.
- **Before the freeze:** a dry run on these data printed only counts, convergence and runtime: 449
  cells, 21,660 within-task pairs, 24 roots, 114 branches, and every fit converged. No score was seen
  before publication.

## The scores (development; 2000 two-level cluster resamples, 95% intervals)

| measure | C_a (a) | C_a − M0 | K_b (b), lower is better | M0 − K_b | Brier |
|---|---|---|---|---|---|
| **m0** (reference) | 0.816 [0.737, 0.880] | — | 0.081 [0.053, 0.113] | — | 0.040 |
| m1 bin 0 (s < .25) | 0.808 | −0.009 [−0.029, 0.017] | 0.073 | +0.008 [−0.015, 0.030] | 0.043 |
| m1 bin 1 | 0.810 | −0.006 [−0.028, 0.016] | 0.081 | −0.000 [−0.024, 0.018] | 0.045 |
| m1 bin 2 | 0.810 | −0.006 [−0.035, 0.016] | 0.068 | +0.013 [−0.008, 0.032] | 0.040 |
| m1 bin 3 (s ≥ .75) | 0.810 | −0.007 [−0.019, 0.007] | 0.078 | +0.002 [−0.009, 0.006] | 0.043 |
| **m2 position** | **0.825** | **+0.008 [0.000, 0.023]** | 0.081 | −0.001 [−0.007, 0.016] | 0.038 |
| m2 rotation | 0.795 | −0.021 [−0.040, 0.030] | 0.090 | −0.009 [−0.059, 0.006] | 0.058 |
| m2 gripper | 0.771 | **−0.046 [−0.071, −0.024]** | 0.074 | +0.007 [−0.051, 0.025] | 0.063 |
| m2 executed (first 8) | 0.817 | +0.001 [−0.004, 0.007] | 0.080 | +0.001 [−0.003, 0.005] | 0.040 |
| m2 rest | 0.815 | −0.001 [−0.009, 0.005] | 0.082 | −0.001 [−0.005, 0.002] | 0.041 |
| **m3 mean** (sampled chunks) | 0.808 | −0.008 [−0.027, 0.025] | **0.065** | **+0.016 [−0.030, 0.031]** | 0.047 |
| m3 mean, executed | 0.807 | −0.009 [−0.030, 0.025] | 0.065 | +0.016 [−0.032, 0.031] | 0.050 |
| *m3 best-of-8 (oracle, not selectable)* | 0.811 | −0.006 | 0.074 | +0.007 | 0.044 |

**About the K_b and Brier intervals:** they are conditional on the cross-fitted predictions; they omit
the variation from refitting.

**Selection, applied by the code:**
- **Claim (a), within-task ranking:** `m2_position` is carried (+0.008).
- **Claim (b), cross-task calibration:** `m3_mean` is carried (+0.016). Its frozen pooled curve is
  logit p = −4.120 − 0.766 · log(m3_mean). M0's is −3.224 − 1.224 · log(m0); it is also frozen, for
  the test's reference.

## What the development data say

- **M0 is already a good within-task ranker.** C_a is 0.82 overall:
  - 0.86 on Spatial (7,612 pairs);
  - 0.78 on Object (10,042);
  - 0.89 on Goal's drawer task (2,093);
  - 0.77 on libero_10's stove task (1,817).

  No candidate improves on it by more than 0.01. `m2_position`'s gain has an interval touching 0, so
  it was carried only because the rule carries any positive point improvement. The only clear
  difference is negative: **the gripper-only loss ranks worse** (−0.046, CI excluding 0).
- **Cross-task calibration is the weaker side, and no measure fixes it clearly.**
  - Under one pooled curve, M0's task-level miscalibration is 8.1 pp (K_b).
  - Per-task intercepts would still gain 1605 units of deviance in-sample, so the task effect remains
    large.
  - `m3_mean`, the sampled-chunk error, cuts K_b to 6.5 pp and the deviance gain to 1190. That is the
    largest reduction, and its interval is wide (−0.030 to +0.031).
  - The motivating puzzle (equal M0, very different success across tasks) is **reduced, not removed.**
- **The flow-time and execution-window splits add nothing.** m1's bins, and executed against rest,
  are all within ±0.01 of M0 on both claims.
- **The oracle (best-of-8)** is not better than the mean draw on either claim.

## Sensitivities (they never change the selection)

| | cells | M0 C_a | `m2_position` − M0 (C_a) | M0 K_b | M0 − `m3_mean` (K_b) |
|---|---|---|---|---|---|
| main | 449 | 0.816 | +0.008 [0.000, 0.023] | 0.081 | +0.016 [−0.030, 0.031] |
| without seed 0's stage-0 root | 235 | 0.802 | +0.011 [−0.009, 0.028] | 0.077 | +0.008 [−0.055, 0.036] |
| first-divergence branches | 449 | 0.816 | +0.008 [−0.000, 0.025] | 0.081 | +0.016 [−0.026, 0.030] |
| pairs with \|Δcount\| ≥ 5 | 449 | 0.885 | +0.010 [0.001, 0.025] | 0.081 | +0.016 [−0.030, 0.031] |
| without bracket cells | 393 | 0.813 | +0.009 [−0.001, 0.023] | 0.089 | +0.010 [−0.032, 0.030] |
| without all-0 / all-50 cells | 342 | 0.723 | +0.011 [0.000, 0.032] | 0.068 | +0.005 [−0.030, 0.025] |
| **root-weighted** (point) | 10 / 16 roots | 0.851 | +0.011 | 0.076 | **−0.042** (m3_mean 0.118) |

- **`m2_position`'s small ranking gain has the same sign in every sensitivity.**
- **`m3_mean`'s calibration gain does not survive root weighting.** Averaged over roots, it is
  *worse* than M0 (0.118 against 0.076). The cell-weighted gain comes from the large roots. This
  makes (b)'s carried measure a weak candidate going into the test.

## What the freeze fixes for the held-out test (step 4)

- **The test set:** the reserved 4-task runs' cells at stages ≥ 1: replay, ewc, lora, consft,
  projected_lora and inflora, seeds 0–2. Cells whose (fingerprint, task) is in the development keys
  are excluded.
- **How it is measured:** the test manifest is built from those runs only, measured under step 2's
  section, and predicted with **the frozen coefficients**. The development cells are never refitted.
- **The claim:** "`m2_position` beats M0" on (a), or "`m3_mean` beats M0" on (b), needs both:
  1. an improvement on the test cells whose two-level cluster interval excludes 0;
  2. the same sign within every root (seed).

  Otherwise the result is "not shown".
- **The expectation, written now:** the development gains are about 0.01 and 0.016, with intervals
  through 0, so **"not shown" is the likely held-out result for both claims.** The minimum thesis
  tolerates that. The descriptive results (M0's within-task ranking of 0.82 and its 8 pp cross-task
  miscalibration) stand regardless.
- **Angle B at D2 (critical outputs from Track 2):** the bundle exists, so angle B is formally
  available. But Track 2 identified no clearly critical output. The best ranking signal is position;
  the gripper ranks worse. So Track 2 gives angle B little to work with. The recovery study (planned;
  next) measures critical outputs directly.
