# Track 2, step 2: the offline measures M0–M3 are published for all 449 development cells. Every recorded probe loss they can be checked against (281 values, 265 cells) is reproduced within 2.1e-5 relative

**Status: step 2 of Track 2, published.** It is not the freeze.
- Validation and the predictor selection (step 3) follow, under a rule written before any fitting.
- **The hard deadline Fri 30 Oct is unchanged.**
- **This record is descriptive only.** It reports the measures' distributions, the instrument checks
  and the runtime. **No measure is related to rollout success here**; that is step 3's rule.

**Provenance:**
- **Rule:** the `measures` section of `configs/analysis/behavior_predictors.yaml`, committed before any
  measure (`d213ddb`).
- **Bundle:** `results/track2/measures/` (`measures.json` SHA-256 `5be958cc…`, `summary.json`
  `547d7b15…`; `done.json` binds both).
- **Run:** queue `results/logs/queue_20261012_232356_tonight` at `99e8083`, clean tree.
  - Its full CPU suite passed at that commit (1091 passed, exit 0, 00:48).
  - Publication was gated on that suite and on the instrument checks.
  - The first launch (22:46, at `d213ddb`) was stopped before the measures. Its projected-LoRA smoke
    had found a bug in that method (see the comparison record), and the queue was relaunched at the
    fix's commit.
- **Measured at:** Tue 13 Oct 01:19–01:52 (33 min; 3.1 s per cell, about half of it checkpoint
  loading).

## What was measured

Each cell's kept-alias checkpoint was measured on its task's fixed bank:
- the task's 50 demos, with the checkpoint's frozen stats;
- the forgetting diagnostics' probe: 16 batches × 64 chunks, tags `gpm_pilot_probe::*`, with the same
  `s` and `A_0` for every checkpoint;
- fp32, eval mode.

The measures:

| | what | units stored |
|---|---|---|
| **M0** | masked flow-matching (velocity) loss | per batch and per demo, (sum, count) |
| **M1** | the same, per flow-time bin s ∈ [0, .25), [.25, .5), [.5, .75), [.75, 1] | same |
| **M2** | per action component (position 0–3, rotation 3–6, gripper 6–7); the first 8 executed steps against the rest | same |
| **M3** | sampled chunks: 8 draws per observation (bank `track2_m3::noise`), 10 Euler steps, on the first 8 batches; per-sample MSE against the demonstration chunk, mean over draws (a candidate proxy) and best-of-8 (an **oracle diagnostic**: a deployed policy draws once) | per batch and per demo |

All 449 cells use 16 batches and 50 demos; M3 uses 8 batches. The per-element errors reproduce each
batch's forward loss to 5.9e-8 relative at most.

## Instrument checks (passed)

281 recorded values, on 265 of the 449 cells, are reproduced by M0. The largest relative difference
is 2.1e-5, against a tolerance of 1e-3:
- 90 from the forgetting diagnostics' loss matrices (seq_ft, GPM, adaptive GPM; seeds 0–2);
- 177 from rows' per-cell `probe_loss`;
- 14 from legacy rows' retained/new `probe_loss`.

The other 184 cells have no recorded probe loss (rollout-only sources) and are measured without a
cross-check.

## Distributions (quartiles over cells: min / Q1 / median / Q3 / max)

| suite (cells) | M0 | M3 mean | M3 best / mean (median) |
|---|---|---|---|
| libero_spatial (130) | .0074 / .053 / .090 / .17 / 3.07 | .00014 / .0021 / .0080 / .016 / .41 | 0.72 |
| libero_object (146) | .0061 / .0090 / .025 / .047 / 2.23 | .00005 / .00012 / .0012 / .0057 / .34 | 0.66 |
| libero_goal (94) | .0039 / .0069 / .038 / .24 / 1.84 | .00006 / .00012 / .0012 / .0084 / .58 | 0.86 |
| libero_10 (76) | .0064 / .011 / .036 / .054 / 1.96 | .00009 / .00030 / .0016 / .0077 / .26 | 0.58 |
| libero_90 (3) | .0063 / – / .0086 / – / .013 | .00008 / – / .00016 / – / .00032 | 0.83 |

**Structure visible in every suite:**
- **M1:** the loss grows steeply with flow time. The last bin, s ∈ [.75, 1] (near the data), has a
  median 8–25× the first bin's and carries most of M0.
- **M2:** position carries most of the error (Spatial medians: position .157, rotation .033,
  gripper .045). The gripper has the heaviest tail (maxima 4–14).
- **Executed against rest:** the median ratio of the first 8 steps' loss to the rest's is 0.99–1.04
  per suite. The executed part of a chunk is not systematically easier offline.
- **M3:**
  - Best-of-8 is well below the mean (median ratio 0.58–0.86), so sampling variance is a large part
    of the sampled-chunk error on this bank.
  - M3 is on a different scale from M0: action error against velocity error.
- **The ranges are wide.** M0's maximum is about 20–35× each suite's median. These extreme cells are
  heavily forgotten checkpoints. No cell is excluded.

## What it means for step 3 (carried into its rule, written before any fitting)

- **Use the stored units, not the batches.** Chunks overlap within demos, so the per-demo (sum, count)
  pairs are the resampling unit inside a cell. Neither the 449 cells nor the 24 roots are
  independent training replications.
- **Decide before fitting:**
  - whether the primary score weights cells or roots (seed 0's stage-0 root holds 214 of 449);
  - a sensitivity analysis without that root;
  - **within-task ranking (a) and cross-task calibration (b)** as separate claims.
- **Scale differs by suite** (Spatial's median M0 is 3.5× Object's). That is the cross-task
  calibration problem (b) the manifest record anticipated.
- **M3's best-of-8 stays an oracle diagnostic.** Only M3's mean is a candidate deployable proxy.
- **The 184 unchecked cells** carry no instrument cross-check of their own. The same code path is
  checked on the other 265.
