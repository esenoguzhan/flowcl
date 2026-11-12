# 4-task baselines, seed 0 (`seq_hetero`): replay keeps the most (final 96 / 42 / 90 / 96). The two projected LoRA variants keep Spatial and Goal but lose Object. EWC, LoRA and ConSFT forget every earlier task. Every identity check passed bitwise. This is seed 0 only: Track 2's test needs seeds 1–2

**Status: the seed-0 slice of the baseline queue, complete.**
- **The queue:** `scripts/queue_baselines.sh --seed 0` at `27c7e88` from a clean tree. It ended
  `QUEUE DONE ok`, and all six strict sequence reports passed.
- **The six methods**, all on `seq_hetero` seed 0, each paired with `seq_hetero__seq_ft__seed0`:
  - ewc, lora, replay, consft;
  - inflora, projected_lora: the two LoRA variants with a GPM-style projection, both registry
    `projected_lora`.
- **Settings:** the frozen selection bundles (`configs/method/*.yaml`).
  - Each bracketed method's settings mean "the best among tested settings on the development seed",
    not a resolved optimum.
  - Replay has fixed settings: 10 demos per task, replay fraction 0.5.
- **What it is not:** a Track 2 result.
  - The frozen test rule (`configs/analysis/behavior_predictors.yaml`) needs seeds 0–2, with the
    same sign within every seed.
  - No held-out verdict is valid before seeds 1–2 have run.
  - The post-freeze exclusion of bracket arms (`status.md` §4) is also still open.
- **Logs:**
  - EWC: `results/logs/queue_20261109_171329_baselines_seed0/`.
  - Everything from LoRA on: `results/logs/queue_20261111_170949_baselines_seed0/`.
  - The interruptions are below.
- **Reports:** `results/baselines_seq/seed0/<method>.json`.

## Retention matrices

Each row is "after training on task i". Each column is success on task j, over 50 rollouts. The
tasks are T1 Spatial (bowl → plate), T2 Object (milk → basket), T3 Goal (middle drawer) and T4 10
(stove + moka pot). The thresholds are seq_ft's diagonal minus 15 pp: 75 / 63 / 85 / 83.

| method | after T1 | after T2 | after T3 | after T4 (final) |
|---|---|---|---|---|
| *seq_ft* | 90 · · · | 0 78 · · | 0 0 100 · | 0 0 0 98 |
| ewc | 90 · · · | 0 72 · · | 0 0 92 · | 0 0 0 78 |
| lora | 90 · · · | 52 70 · · | 0 0 88 · | 0 0 0 56 |
| replay | 90 · · · | 98 76 · · | 100 38 96 · | 96 42 90 96 |
| consft | 90 · · · | 0 92 · · | 0 0 100 · | 0 0 0 90 |
| inflora | 90 · · · | 90 82 · · | 96 32 88 · | 92 28 78 70 |
| projected_lora | 90 · · · | 94 70 · · | 92 8 96 · | 90 4 88 96 |
| *GPM plain (`gpm_projected_adam`)* | 90 · · · | 94 78 · · | 90 2 94 · | 88 0 56 90 |
| *GPM e99* | 90 · · · | 94 78 · · | 96 72 92 · | 92 70 78 92 |
| *GPM ne90* | 90 · · · | 94 78 · · | 96 68 94 · | 98 84 94 92 |

The final row against seq_ft is a paired difference with a 95 % bootstrap CI. Plasticity is the
diagonal; ✗ marks a failure and ~ a borderline result.

| method | T1 final | T2 final | T3 final | T4 final | plasticity T1–T4 |
|---|---|---|---|---|---|
| ewc | +0.00 | +0.00 | +0.00 | −0.20 [−0.32, −0.10] | 90 72~ 92~ 78✗~ |
| lora | +0.00 | +0.00 | +0.00 | −0.42 [−0.56, −0.28] | 90 70~ 88~ 56✗ |
| replay | +0.96 [+0.90, +1.00] | +0.42 [+0.28, +0.56] | +0.90 [+0.82, +0.98] | −0.02 [−0.08, +0.04] | 90 76 96 96 |
| consft | +0.00 | +0.00 | +0.00 | −0.08 [−0.18, +0.00] | 90 92 100 90~ |
| inflora | +0.92 [+0.84, +0.98] | +0.28 [+0.16, +0.40] | +0.78 [+0.66, +0.88] | −0.28 [−0.42, −0.14] | 90 82 88~ 70✗ |
| projected_lora | +0.90 [+0.80, +0.98] | +0.04 [+0.00, +0.10] | +0.88 [+0.78, +0.96] | −0.02 [−0.08, +0.04] | 90 70~ 96 96 |

| method | ACC (final mean) | NBT | AUC | trainable params, stages 1–3 | method storage | exemplar-free | wall clock (train / total) |
|---|---:|---:|---:|---:|---:|---|---|
| *seq_ft* | 0.245 | 0.893 | 0.467 | 53.8 M | 0 | yes | 2.84 / 5.56 h |
| ewc | 0.195 | 0.847 | 0.440 | 53.8 M | 431 MB | yes | 3.18 / 5.76 h |
| lora | 0.140 | 0.827 | 0.486 | 2.06 M | 0 | yes | 3.04 / 5.49 h |
| replay | 0.810 | 0.113 | 0.840 | 53.8 M | 672 MB | **no** | 2.61 / 4.47 h |
| consft | 0.225 | 0.940 | 0.480 | 53.8 M | 0 | yes | 2.64 / 5.06 h |
| inflora | 0.670 | 0.207 | 0.787 | 1.02 M | 133 MB | yes | 3.15 / 5.19 h |
| projected_lora | 0.695 | 0.247 | 0.767 | 8.19 M | 138 MB | yes | 3.28 / 5.26 h |
| *GPM plain* | 0.585 | 0.393 | 0.741 | — | 140 MB | yes | 2.89 / 4.96 h |
| *GPM e99* | 0.830 | 0.067 | 0.864 | — | 192 MB | yes | 2.94 / 4.80 h |
| *GPM ne90* | 0.920 | −0.047 | 0.885 | — | 231 MB | yes | 2.88 / 4.74 h |

Notes on the columns:
- ACC is `result.json`'s `F_1`, the mean of the final row. A6 renames it.
- Stage 0 is seq_ft's shared stage 0 for every method, so 53.8 M parameters train there; 85.2 M are
  frozen throughout.
- "Total" wall clock includes the rollouts.

## What it shows (seed 0, descriptive)

- **EWC, LoRA and ConSFT forget everything, as seq_ft does.** Every earlier task is 0/50 in the
  final row.
  - This matches their brackets, which were all flagged `all_eligible_spatial_zero`, except LoRA's.
  - LoRA keeps Spatial at 52 after T2, its bracket's number, then loses it at T3.
  - EWC and LoRA also lose T4 plasticity: 78 and 56 against 83.
  - ConSFT has the highest plasticity: 92 / 100 / 90.
- **Replay keeps the most** (ACC 0.81). It stores 10 demos per task (672 MB) and is the only method
  that is not exemplar-free.
  - It holds T1 and T3 near their peak.
  - Object falls to 38 after T3, then 42.
- **The projected variants keep Spatial and Goal and lose Object.**
  - projected_lora: final 90 / 4 / 88 / 96.
  - inflora: final 92 / 28 / 78 / 70; its T4 plasticity (70) fails.
  - Against plain GPM (88 / 0 / 56 / 90), both keep Goal better (+32 and +22 pp, point
    estimates, unpaired). Neither keeps Object.
  - The e99 and ne90 GPM variants keep Object (70 and 84).
  - So the Object loss across Goal is the same pattern as plain GPM's. It is not specific to the
    LoRA parameterisation.
- **One seed, 50 rollouts per cell.** The comparisons against GPM are point estimates from separate
  runs, not paired tests.

## Provenance (all passed)

- **Stage-0 identity:** every method's `stage0.pt` equals seq_ft seed 0's (`21a3e422…`), 0 of 656
  tensors differing.
- **Stage-1 identity:** for the five bracketed methods, each `stage1.pt` equals the selected arm's
  `final.pt`, 0 of 656 tensors differing:

  | method | arm `final.pt` |
  |---|---|
  | ewc | `8af2f1e3…` |
  | lora | `d9d1fd21…` |
  | consft | `226da77e…` |
  | inflora | `473fc338…` |
  | projected_lora | `756e4a89…` |

  Replay has no bracket.
- **Bracket stage-1 evaluation:** the arm's Spatial and Object rollouts against the run's.
  - All 10 task × method pairs have 0 of 50 success disagreements and 0 episode-length
    disagreements.
  - Spatial / Object successes:

    | method | Spatial | Object |
    |---|---:|---:|
    | ewc | 0 | 36 |
    | lora | 26 | 35 |
    | consft | 0 | 46 |
    | inflora | 45 | 41 |
    | projected_lora | 47 | 35 |

  - The 12 Oct manifest found one 2/50 disagreement between identical checkpoints across processes.
    None occurred here.
- **Projected variants:**
  - T1 memory equals GPM seed 0's `memory_task0.pt` on all 91 layers.
  - Occupancy is non-decreasing and the memories chain.
  - The merge records exist at stages 1–3 with `constraint_ok`. The worst residual is 0.0013
    (inflora) and 0.0014 (projected_lora) of the tolerance.
- **Other checks:**
  - The T1 pairing, seed namespace, clean git SHA and artifact hashes pass for every method.
  - `frozen_from_stage1` passes for lora and both projected variants.
  - Methods outside the GPM family get explicit N/A records for the memory, residual and freezing
    checks.
- **Known debt (unchanged):** strict mode does not yet require non-empty residual logs for GPM and
  SGP runs. No GPM or SGP run was in this queue, and their reports are unchanged.

## Interruptions

LoRA ran three times. The third attempt, a full rerun from scratch, is the reported run.
1. **Mon 9 Nov 22:59:** the first attempt started.
   - It was killed at **Tue 10 Nov 03:00** by `net-watchdog`, a local service that reboots the
     machine after 30 min offline.
   - The LRZ RADIUS certificate (`radius.lrz.de`) had expired, so eduroam could not authenticate.
   - The machine then rebooted every 2 h until the eduroam profile was changed on Wed 11 Nov 12:30.
   - Before the kill, the first attempt had passed both identity checks.
2. **Wed 11 Nov 12:42:** restarted with `--from-step 4`. It was killed by a power-off at about 17:05.
3. **Wed 11 Nov 17:09:** restarted with `--from-step 4`.
   - It re-checked EWC's run and its strict report (both passed), then ran LoRA through
     projected_lora without a break, ending Thu 12 Nov 18:39.

The two partial LoRA runs are kept in `results/interrupted/`. No reported number comes from them.

## Next

- **Seeds 1–2:** `bash scripts/queue_baselines.sh --seed 1`, then `--seed 2`, about 30 h each.
  Seeds 1–2 have no stage-1 arm identity; the preflight's full config check stands in for it.
- **Track 2 step 4:** the post-freeze bracket-arm exclusion (`status.md` §4 item 1), before the
  held-out test reads any of these cells.
