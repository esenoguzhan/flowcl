# 4-task baselines, seeds 1–2 (`seq_hetero`): the seed-0 ranking holds on all three seeds. Replay keeps the most (mean final 95 / 69 / 95 / 95), and seed 0's Object (42) was its low seed. Both projected LoRA variants lose Object on every seed. EWC, LoRA and ConSFT keep no earlier task. Every identity check and strict report passed. The Track 2 test set is now complete; the test itself is not run

**Status: seeds 1 and 2 of the baseline queue, complete.** With seed 0
(`runs/2026-11-12_baselines_seed0.md`) this is the full 4-task baseline set: 6 methods × 3 seeds.
- **The queue:** `scripts/queue_baselines.sh --seed 1`, then `--seed 2`, at `39ce581` from a clean
  tree, in tmux `baselines_s12`.
  - Seed 1: Thu 12 Nov 20:47 → Sat 14 Nov 03:54 (31.1 h).
  - Seed 2: Sat 14 Nov 03:54 → Sun 15 Nov 11:05 (31.2 h).
  - Both ended `QUEUE DONE ok`. All 12 strict sequence reports passed. No reboot, no restart.
- **The six methods** and their settings are seed 0's: ewc, lora, replay, consft, inflora and
  projected_lora, from the frozen selection bundles (`configs/method/*.yaml`). Each run is paired with
  `seq_hetero__seq_ft__seed<S>`.
- **Seeds 1–2 are the first seeds outside the selection.** Each bracketed method's settings were
  chosen on seed 0's T1 → T2, so seed 0's stage 1 *is* the selected arm. Seeds 1–2 say how those
  choices carry over.
- **What it is not:** a Track 2 result.
  - The frozen test rule (`configs/analysis/behavior_predictors.yaml`) reads the cells of these 18
    runs, stages ≥ 1. Its claim needs a cluster-bootstrap interval excluding 0 *and* the same sign
    within every seed.
  - Step 4, which builds the test manifest and applies the post-freeze exclusion of bracket arms
    (`status.md` §4 item 1), has not run. Nothing here is a held-out verdict.
- **Logs:** `results/logs/queue_20261112_204717_baselines_seed1/`,
  `results/logs/queue_20261114_035431_baselines_seed2/`.
- **Reports:** `results/baselines_seq/seed{1,2}/<method>.json`.

## Across seeds 0–2

Final row (after T4), success over 50 rollouts. T1 Spatial (bowl → plate), T2 Object (milk →
basket), T3 Goal (middle drawer), T4 10 (stove + moka pot). Mean ± sd over 3 seeds is descriptive.

| method | seed 0 | seed 1 | seed 2 | mean ± sd | ACC mean ± sd | NBT mean ± sd |
|---|---|---|---|---|---:|---:|
| *seq_ft* | 0 0 0 98 | 0 0 0 98 | 0 0 0 90 | 0 / 0 / 0 / 95 | 0.24 ± 0.01 | 0.93 ± 0.03 |
| ewc | 0 0 0 78 | 0 0 0 74 | 2 0 0 58 | 1 / 0 / 0 / 70 ± 11 | 0.18 ± 0.02 | 0.84 ± 0.04 |
| lora | 0 0 0 56 | 0 0 0 58 | 0 0 0 66 | 0 / 0 / 0 / 60 ± 5 | 0.15 ± 0.01 | 0.73 ± 0.09 |
| replay | 96 42 90 96 | 94 82 98 94 | 96 82 98 96 | 95 ± 1 / 69 ± 23 / 95 ± 5 / 95 ± 1 | 0.89 ± 0.07 | 0.05 ± 0.05 |
| consft | 0 0 0 90 | 0 0 0 84 | 0 0 0 88 | 0 / 0 / 0 / 87 ± 3 | 0.22 ± 0.01 | 0.94 ± 0.01 |
| inflora | 92 28 78 70 | 100 26 60 78 | 96 10 78 72 | 96 ± 4 / 21 ± 10 / 72 ± 10 / 73 ± 4 | 0.66 ± 0.02 | 0.15 ± 0.08 |
| projected_lora | 90 4 88 96 | 84 6 66 84 | 88 10 60 80 | 87 ± 3 / 7 ± 3 / 71 ± 15 / 87 ± 8 | 0.63 ± 0.06 | 0.32 ± 0.06 |
| *GPM plain* | 88 0 56 90 | 92 0 20 96 | 96 24 96 94 | 92 ± 4 / 8 ± 14 / 57 ± 38 / 93 ± 3 | 0.63 ± 0.13 | 0.38 ± 0.16 |
| *GPM e99* | 92 70 78 92 | 98 66 94 76 | 100 56 98 90 | 97 ± 4 / 64 ± 7 / 90 ± 11 / 86 ± 9 | 0.84 ± 0.02 | 0.07 ± 0.02 |
| *GPM ne90* | 98 84 94 92 | 94 82 72 74 | 94 78 98 80 | 95 ± 2 / 81 ± 3 / 88 ± 14 / 82 ± 9 | 0.87 ± 0.06 | 0.03 ± 0.07 |

The GPM rows (italic) are earlier runs of the same seeds, not part of this queue. They are
unpaired with the baselines.

Object plasticity (the T2 diagonal) by seed, against seq_ft's:

| | seq_ft | ewc | lora | replay | consft | inflora | projected_lora |
|---|---:|---:|---:|---:|---:|---:|---:|
| seed 0 (selection) | 78 | 72 | 70 | 76 | 92 | 82 | 70 |
| seed 1 | 90 | 72 | 66 | 88 | 92 | 62 | 80 |
| seed 2 | 92 | 48 | 30 | 84 | 84 | 56 | 72 |

## Retention matrices, seed 1

Each row is "after training on task i", each column success on task j over 50 rollouts. Thresholds
(registered, seq_ft seed 1's diagonal minus 15 pp): 81 / 75 / 85 / 83.

| method | after T1 | after T2 | after T3 | after T4 (final) |
|---|---|---|---|---|
| *seq_ft* | 96 · · · | 0 90 · · | 0 0 100 · | 0 0 0 98 |
| ewc | 96 · · · | 0 72 · · | 0 0 92 · | 0 0 0 74 |
| lora | 96 · · · | 14 66 · · | 0 0 54 · | 0 0 0 58 |
| replay | 96 · · · | 96 88 · · | 98 78 98 · | 94 82 98 94 |
| consft | 96 · · · | 0 92 · · | 0 0 100 · | 0 0 0 84 |
| inflora | 96 · · · | 98 62 · · | 98 18 46 · | 100 26 60 78 |
| projected_lora | 96 · · · | 96 80 · · | 94 30 88 · | 84 6 66 84 |
| *GPM plain (`gpm_projected_adam`)* | 96 · · · | 98 78 · · | 86 2 98 · | 92 0 20 96 |
| *GPM e99* | 96 · · · | 98 78 · · | 98 64 100 · | 98 66 94 76 |
| *GPM ne90* | 96 · · · | 98 78 · · | 98 66 100 · | 94 82 72 74 |

Final row against seq_ft: paired difference with a 95 % bootstrap CI. Plasticity is the diagonal;
✗ marks a failure and ~ a borderline result.

| method | T1 final | T2 final | T3 final | T4 final | plasticity T1–T4 |
|---|---|---|---|---|---|
| ewc | +0.00 | +0.00 | +0.00 | −0.24 [−0.38, −0.12] | 96 72✗~ 92~ 74✗~ |
| lora | +0.00 | +0.00 | +0.00 | −0.40 [−0.54, −0.26] | 96 66✗~ 54✗ 58✗ |
| replay | +0.94 [+0.86, +1.00] | +0.82 [+0.70, +0.92] | +0.98 [+0.94, +1.00] | −0.04 [−0.12, +0.04] | 96 88 98 94 |
| consft | +0.00 | +0.00 | +0.00 | −0.14 [−0.24, −0.06] | 96 92 100 84~ |
| inflora | +1.00 [+1.00, +1.00] | +0.26 [+0.14, +0.38] | +0.60 [+0.46, +0.74] | −0.20 [−0.32, −0.08] | 96 62✗~ 46✗ 78✗~ |
| projected_lora | +0.84 [+0.74, +0.94] | +0.06 [+0.00, +0.14] | +0.66 [+0.52, +0.78] | −0.14 [−0.24, −0.06] | 96 80~ 88~ 84~ |

| method | ACC (final mean) | NBT | AUC | trainable params, stages 1–3 | method storage | exemplar-free | wall clock (train / total) |
|---|---:|---:|---:|---:|---:|---|---|
| *seq_ft* | 0.245 | 0.953 | 0.497 | 53.8 M | 0 | yes | 2.63 / 5.13 h |
| ewc | 0.185 | 0.867 | 0.453 | 53.8 M | 431 MB | yes | 3.14 / 5.67 h |
| lora | 0.145 | 0.720 | 0.421 | 2.06 M | 0 | yes | 3.02 / 5.53 h |
| replay | 0.920 | 0.027 | 0.928 | 53.8 M | 672 MB | **no** | 2.59 / 4.35 h |
| consft | 0.210 | 0.960 | 0.491 | 53.8 M | 0 | yes | 2.64 / 5.09 h |
| inflora | 0.660 | 0.060 | 0.740 | 1.02 M | 135 MB | yes | 3.15 / 5.22 h |
| projected_lora | 0.600 | 0.360 | 0.787 | 8.19 M | 139 MB | yes | 3.27 / 5.24 h |
| *GPM plain* | 0.520 | 0.533 | 0.745 | — | 143 MB | yes | 2.88 / 4.96 h |
| *GPM e99* | 0.835 | 0.053 | 0.887 | — | 199 MB | yes | 2.88 / 4.72 h |
| *GPM ne90* | 0.805 | 0.087 | 0.881 | — | 240 MB | yes | 2.89 / 4.80 h |

## Retention matrices, seed 2

Thresholds (derived: seq_ft seed 2 has no registered block, so its diagonal minus 15 pp):
81 / 77 / 83 / 75.

| method | after T1 | after T2 | after T3 | after T4 (final) |
|---|---|---|---|---|
| *seq_ft* | 96 · · · | 0 92 · · | 0 0 98 · | 0 0 0 90 |
| ewc | 96 · · · | 0 48 · · | 0 0 96 · | 2 0 0 58 |
| lora | 96 · · · | 0 30 · · | 0 0 70 · | 0 0 0 66 |
| replay | 96 · · · | 98 84 · · | 98 80 100 · | 96 82 98 96 |
| consft | 96 · · · | 0 84 · · | 0 0 100 · | 0 0 0 88 |
| inflora | 96 · · · | 96 56 · · | 98 22 88 · | 96 10 78 72 |
| projected_lora | 96 · · · | 96 72 · · | 96 22 94 · | 88 10 60 80 |
| *GPM plain (`gpm_projected_adam`)* | 96 · · · | 94 86 · · | 98 22 98 · | 96 24 96 94 |
| *GPM e99* | 96 · · · | 94 86 · · | 98 70 100 · | 100 56 98 90 |
| *GPM ne90* | 96 · · · | 94 86 · · | 96 78 100 · | 94 78 98 80 |

| method | T1 final | T2 final | T3 final | T4 final | plasticity T1–T4 |
|---|---|---|---|---|---|
| ewc | +0.02 [+0.00, +0.06] | +0.00 | +0.00 | −0.32 [−0.46, −0.18] | 96 48✗ 96 58✗ |
| lora | +0.00 | +0.00 | +0.00 | −0.24 [−0.38, −0.12] | 96 30✗ 70✗ 66✗~ |
| replay | +0.96 [+0.90, +1.00] | +0.82 [+0.70, +0.92] | +0.98 [+0.94, +1.00] | +0.06 [−0.04, +0.16] | 96 84~ 100 96 |
| consft | +0.00 | +0.00 | +0.00 | −0.02 [−0.10, +0.06] | 96 84~ 100 88 |
| inflora | +0.96 [+0.90, +1.00] | +0.10 [+0.02, +0.18] | +0.78 [+0.66, +0.88] | −0.18 [−0.32, −0.06] | 96 56✗ 88~ 72✗~ |
| projected_lora | +0.88 [+0.78, +0.96] | +0.10 [+0.02, +0.18] | +0.60 [+0.46, +0.74] | −0.10 [−0.22, +0.02] | 96 72✗~ 94 80~ |

| method | ACC (final mean) | NBT | AUC | trainable params, stages 1–3 | method storage | exemplar-free | wall clock (train / total) |
|---|---:|---:|---:|---:|---:|---|---|
| *seq_ft* | 0.225 | 0.953 | 0.493 | 53.8 M | 0 | yes | 2.64 / 5.17 h |
| ewc | 0.150 | 0.793 | 0.417 | 53.8 M | 431 MB | yes | 3.14 / 5.68 h |
| lora | 0.165 | 0.653 | 0.377 | 2.06 M | 0 | yes | 3.02 / 5.57 h |
| replay | 0.930 | 0.013 | 0.932 | 53.8 M | 672 MB | **no** | 2.59 / 4.34 h |
| consft | 0.220 | 0.933 | 0.483 | 53.8 M | 0 | yes | 2.63 / 5.06 h |
| inflora | 0.640 | 0.187 | 0.763 | 1.02 M | 134 MB | yes | 3.15 / 5.23 h |
| projected_lora | 0.595 | 0.347 | 0.775 | 8.19 M | 138 MB | yes | 3.29 / 5.28 h |
| *GPM plain* | 0.775 | 0.213 | 0.840 | — | 141 MB | yes | 2.89 / 4.85 h |
| *GPM e99* | 0.860 | 0.093 | 0.903 | — | 196 MB | yes | 2.95 / 4.79 h |
| *GPM ne90* | 0.875 | 0.040 | 0.912 | — | 234 MB | yes | 2.89 / 4.69 h |

Notes on the columns are as in the seed-0 record: ACC is `result.json`'s `F_1`; stage 0 is seq_ft's
shared stage 0, so 53.8 M parameters train there; "total" wall clock includes the rollouts.

## What it shows (descriptive)

- **The ranking holds on every seed:** replay > the two projected variants > ConSFT ≈ EWC ≈ LoRA
  ≈ seq_ft by ACC. The order within each pair moves between seeds.
- **EWC, LoRA and ConSFT forget every earlier task on every seed.** The one exception is EWC seed 2's
  final Spatial: 2 % (1 of 50 rollouts).
  - LoRA's Spatial after T2 falls from 52 (seed 0, its bracket's number) to 14 and 0.
  - ConSFT keeps the highest plasticity of the three: final T4 87 ± 3, against seq_ft's 95.
- **Replay keeps the most** (ACC 0.81 / 0.92 / 0.93), at 672 MB of stored demos. It is still the only
  method that is not exemplar-free.
  - **Seed 0 was its low seed.** Final Object is 42 there and 82 on seeds 1 and 2; seeds 1–2 have no
    retention failure, only Object borderline.
  - Its T4 plasticity matches seq_ft's on every seed (−0.02, −0.04, +0.06, every CI includes 0).
- **Both projected variants lose Object on every seed.**
  - projected_lora: final Object 4 / 6 / 10; inflora: 28 / 26 / 10.
  - Both keep Spatial (84–100). Goal varies: projected_lora 88 / 66 / 60, inflora 78 / 60 / 78.
  - **Seed 0's "both keep Goal better than plain GPM" does not hold on seed 2.** Plain GPM's Goal is
    56 / 20 / 96. Against it, projected_lora is +32 / +46 / −36 pp and inflora +22 / +40 / −18 pp
    (point estimates, unpaired). The sign flips on seed 2.
  - The e99 and ne90 GPM variants keep Object on every seed (56–84); plain GPM does not (0 / 0 / 24).
    The Object loss across Goal stays a GPM-family pattern, not one specific to LoRA.
- **Plasticity is lower off the selection seed for most bracketed methods.**
  - Object plasticity, seed 0 → seeds 1, 2: EWC 72 → 72, 48; LoRA 70 → 66, 30; inflora 82 → 62, 56.
    Over the same seeds, seq_ft's rises 78 → 90, 92.
  - projected_lora (70 → 80, 72) and ConSFT (92 → 92, 84) hold.
  - inflora fails T4 plasticity on all three seeds (70 / 78 / 72) and Goal on seed 1 (46).
  - **Reading, not a demonstration:** this fits selection on seed 0's stage 1, whose number is the
    selected arm's by construction. Three seeds cannot separate that from seed noise.
- **Three seeds, 50 rollouts per cell.** The sd values are over 3 runs; comparisons with GPM are
  unpaired point estimates.

## Provenance (all passed)

- **Stage-0 identity:** every method's `stage0.pt` equals its seed's seq_ft `stage0.pt`, 0 of 656
  tensors differing: seed 1 `d4b1c555…`, seed 2 `3e0eacd0…`.
- **Stage-1 identity:** not checked on seeds 1–2, by design (the brackets ran on seed 0 only). The
  preflight's check of each production config against its arm's recorded method block stood in.
- **T1 pairing:** relative weight difference 0.0 against seq_ft for all 12 runs (bound 0.05).
- **Projected variants:**
  - T1 memory equals the seed's GPM `memory_task0.pt` on all 91 layers (the queue's check-run), for
    both variants on both seeds.
  - Occupancy is non-decreasing and the memories chain.
  - The merge records exist at stages 1–3 with `constraint_ok`. The worst residual is 0.0013–0.0014
    of the tolerance.
- **Other checks:** clean git SHA (`39ce581`), seed namespace and artifact hashes pass for every run.
  `frozen_from_stage1` passes for lora and both projected variants. Methods outside the GPM family
  get explicit N/A records for the memory, residual and freezing checks.
- **Known debt (unchanged):** strict mode does not yet require non-empty residual logs for GPM and
  SGP runs. None was in this queue.

## Interruptions

None. Both seeds ran from step 0 without a restart, and `last -x` shows no reboot since Wed 11 Nov.

## Next

- **Track 2 step 4:** the post-freeze bracket-arm exclusion (`status.md` §4 item 1), then the
  held-out test on these 18 runs.
  - On seeds 1–2 no stage-1 cell reproduces a bracket arm, so the exclusion should only touch seed
    0's stage-1 cells. Step 4's acceptance check confirms that per method.
