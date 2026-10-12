# Track 2, step 1: the development manifest has 449 cells (checkpoint × trained task, 50 rollouts) under 24 roots. Seed 0's shared stage 0 holds 214 of them. Every evaluation file under `results/` is reconciled, and 63 bitwise duplicates are collapsed (one disagrees by 2 of 50 rollouts)

**Status: step 1 of Track 2, published.** It is not the freeze.
- The measures (M0–M3), the validation and the predictor selection follow.
- **The hard deadline Fri 30 Oct is unchanged.**
- The manifest was built after LoRA v2's verification, so the bracket arms are included
  deterministically.

| | count |
|---|---:|
| evaluation files under `results/` | 266 |
| used (sources) | 227: runner 118, row 72, legacy row 7, trajectory 14, single 14, pilot 2 (by file) |
| excluded whole files | 39: repeated-state evaluations 18, copies inside resumed runs 12, Dobot 6, invalid or crashed directories 3 |
| task entries excluded | 155 unseen tasks (not yet trained at that checkpoint); no n ≠ 50 |
| **cells (after deduplication)** | **449** (512 aliases; 63 duplicates collapsed; 43 cells have more than one alias) |
| cells by kept source | runner 236, row 169, legacy row 12, trajectory 14, single 14, pilot 4 |
| bracket cells (EWC v1/v2, LoRA v1/v2; development data) | 56 |
| roots | 24: 22 checkpoints and 2 trajectories |
| branches, depth 1 (primary) | 114 |
| branches, first divergence (sensitivity) | 147 |

**By suite** (success rate per cell):

| suite | cells | all 0 | all 50 | median |
|---|---:|---:|---:|---:|
| libero_spatial | 130 | 33 | 1 | 0.48 |
| libero_object | 146 | 22 | 0 | 0.72 |
| libero_goal | 94 | 24 | 16 | 0.90 |
| libero_10 | 76 | 10 | 1 | 0.86 |
| libero_90 (T6–T8 single-task policies) | 3 | 0 | 0 | 0.92 |

In all, 89 cells are at 0, 18 at 50/50 and 342 in between.

**By root** (cells; depth-1 / first-divergence branches):
- **seq_hetero seed 0's stage 0** (bitwise shared by seq_ft, GPM and its variants, SGP, every
  bracket and study arm): **214 cells** (52 / 73).
- seq_hetero seeds 1 and 2: 55 (14 / 17) and 47 (10 / 13).
- seq_hetero_reverse seeds 0–2: 26 each (3 / 5).
- The similar-task base, shared by the three `similar_ab` curricula: 19 (10 / 10).
- The language base (`langbase`): 8 (3 / 3).
- The two hparam trajectories: 7 each (1 / 1).
- 14 Gate 0 single-task policies: 1 each.

**What it means for the later steps:**
- **The two-level cluster bootstrap will be dominated by one root.** 48% of the cells descend from
  seq_hetero seed 0's stage 0, and only 7 roots have more than 8 cells. The plan already accepts wide
  CIs. This makes the dominance concrete.
- **The two branch definitions differ by 33 branches**, mostly under seed 0's root (52 against 73).
  Trunk checkpoints with many descendants become their own branches under the literal definition.
  Both are stored per cell, the depth-1 one as primary (changed in `thesis_plan.md` §D, with the
  reason).
- **The motivating puzzle is in the data.** Medians differ by suite (Spatial 0.48, Object 0.72, Goal
  0.90). The cross-task calibration claim (b) is tested on exactly these cells.
- **Rollout determinism:** one of the 63 collapsed duplicates disagrees.
  - The cell: SGP a800-ne90's stage 3 (Spatial) and `sgp_coupling`'s `coup__aw_soft` arm.
  - Their checkpoints are bitwise identical and use the same 50 episode seeds, but scored 47 and 49
    successes.
  - So evaluations are not bitwise reproducible across processes in every case. The kept alias is the
    runner's (by precedence), and the disagreement is recorded on the cell.

**The held-out test** (stated in the rule, applied when the 4-task baseline runs exist):
- The future runs `seq_hetero__{replay,ewc,lora,consft}__seed*` are reserved. None existed at the
  build; one existing would have made it refuse.
- A future test cell whose (tensor fingerprint, task) is among the manifest's 449 development keys is
  excluded from the held-out analysis. That covers:
  - every method's stage 0 (seq_ft's);
  - production EWC or LoRA seed 0's stage 1 when it reproduces its bracket arm.

**Binding:** every alias records the SHA-256 of each file read to construct it:
- its evaluation, checkpoint, config, row and resume files;
- its lineage: each ancestor checkpoint's path, SHA-256 and tensor fingerprint.

The manifest and summary were published together by one rename, with a done-marker.

**Date:** Mon 12 Oct 21:12 (CEST).
- The dry run on Sun 11 Oct 22:00–22:25 fingerprinted 222 checkpoints into `results/track2_cache/`.
- The final build added LoRA v2's 5 arms.

**Code:** `2ec4c03` on branch `track2`, clean (`git_sha` in the manifest). It was merged into `main`
with a merge commit.
- **The rule:** the `manifest` section of `configs/analysis/behavior_predictors.yaml`, SHA-256
  `3d14961b…`, committed before the build.
- **Tests:** `tests/test_behavior_predictors.py` passed (4), covering:
  - every source layout;
  - exclusions and reconciliation;
  - duplicates with aliases and flags;
  - both branch definitions on a shared trunk;
  - a trajectory as one root;
  - the reserved runs;
  - file hashes;
  - atomic, immutable publication.
- **Isolation:** the build and tests ran from a separate worktree with `LIBERO_CONFIG_PATH` pointed at
  scratch, while the LoRA v2 queue used the GPU. `~/.libero/config.yaml` was checked before and after
  (MD5 `2362405a…`, unchanged).

**Cite:** `results/track2/manifest/manifest.json` (SHA-256 `21c91ebc…`), its `summary.json`
(`8a4ed0e9…`) and its `done.json`.
