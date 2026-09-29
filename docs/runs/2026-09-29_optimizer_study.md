# Optimizer study — the fine-tuning stability–plasticity map on T1 → T2 (seeds 0–2)

**Status:** exploratory map, with no selection. The pre-registered existence claim is **negative**:
none of the 7 core fine-tuning settings keeps Object within 10 pp of the standard recipe while
retaining Spatial, on all three seeds.

**In one sentence:** across learning rates of 1e-5 to 1e-4 and budgets of 5k to 30k steps,
fine-tuning trades Object learning for Spatial retention along a single frontier. Hard projection
(GPM) sits beyond that frontier on every seed.
- **On seeds 1 and 2:** every setting that learns Object within 10 pp of the standard recipe
  forgets Spatial completely, and every setting that keeps any Spatial learns Object 26–60 pp worse.
- **On seed 0:** both are reached. That is also the seed with the lowest standard Object score
  (§3).

**Dates (local, CEST):**
- the training queue ran 28 Sep 13:59 → 29 Sep 01:15 (23 runs);
- the summary failed that night on a tolerance check (§6.1);
- the official summary was produced on 29 Sep 12:38.

**Code:**
- **`77b1785`** (clean) for every new run. The rule `configs/analysis/optimizer_study.yaml` was
  committed with it, before any run.
- **`790ddc8`** for the summary. It amends only the probe-reproduction tolerance (§6.1).

**Cite:** `results/optimizer_study/summary.json`: every row, the per-batch probe units, update sizes,
frontiers, existence and one-factor labels.

---

## 1. Design (as pre-registered)
- **The transition:** Spatial → Object, from each seed's seq_ft `stage0.pt`, paired with seq_ft's
  stage 1 (its stream seed and rollout namespace).
- **Core grid:** 7 settings, on seeds 0–2, and the only settings eligible for the claim:
  - learning rate {1e-5, 3e-5, 1e-4} × steps {5k, 15k};
  - plus the standard recipe (30k, 1e-4), which is seq_ft's own stage 1;
  - all at batch 64, weight decay 1e-4 and a cosine schedule.
- **One-factor variants on seeds 0–1**, around the anchor (5k, 3e-5), descriptive only:
  - `constant_same_peak` (it also raises cumulative LR exposure);
  - weight decay 0 and 1e-2;
  - batch 32 × 10k and batch 128 × 2.5k, both at 320k examples seen, with the learning rate not
    rescaled.
- **Reused, verified and not rerun:**
  - seed 0: (5k, 3e-5), (5k, 1e-4) and (15k, 1e-4), from the hparam study at `8e6785a`;
  - seeds 1–2: (5k, 3e-5), from the low-update validation at `b6b644a`.

  All were verified against their recipes, with the schedule normalized to cosine because the field
  postdates them.
- **Measured:**
  - success on both tasks, paired against the standard recipe;
  - per-batch probe losses, recomputed from every checkpoint, which give matched-batch intervals;
  - the realized update against `stage0.pt`. The frozen encoders were verified unchanged.

## 2. The core grid

Success rates are per seed, 0 / 1 / 2. "Upd." is the relative trainable update,
`‖θ − θ_stage0‖ / ‖θ_stage0‖`, which is essentially the same on every seed.

| lr | steps | Object (%) | Spatial (%) | Object probe | Spatial probe | Upd. |
|---|---:|---|---|---:|---:|---:|
| 1e-5 | 5k | 62 / 34 / 32 | 50 / 46 / 10 | 0.038–0.039 | 0.075–0.084 | 0.032 |
| 1e-5 | 15k | 80 / 46 / 40 | 48 / 36 / 6 | 0.021 | 0.131–0.170 | 0.046 |
| 3e-5 | 5k | 76 / 60 / 50 | 50 / 28 / 2 | 0.023–0.024 | 0.090–0.099 | 0.053 |
| 3e-5 | 15k | 76 / 62 / 66 | 14 / 2 / 2 | 0.012–0.013 | 0.152–0.158 | 0.082 |
| 1e-4 | 5k | 76 / 84 / 72 | 0 / 0 / 0 | 0.016–0.017 | 0.159–0.169 | 0.112 |
| 1e-4 | 15k | 82 / 94 / 86 | 0 / 0 / 0 | 0.009 | 0.450–0.484 | 0.194 |
| **1e-4** | **30k (standard)** | **78 / 90 / 92** | **0 / 0 / 0** | 0.006 | 0.719–0.772 | 0.283 |
| *GPM (standard recipe)* | *30k* | *78 / 78 / 86* | ***94 / 98 / 94*** | *0.007* | *0.039–0.043* | — |

## 3. The existence claim: none

The rule, per seed: Object ≥ the standard's Object − 10 pp, **and** a paired Spatial gain whose CI
lower bound is above 0. A setting counts only if it passes on all three seeds. In the table, "o" is a
plasticity failure and "r" a retention failure.

| Setting | Seed 0 | Seed 1 | Seed 2 |
|---|---|---|---|
| 1e-5 × 5k | o | o | o |
| 1e-5 × 15k | pass | o | o, r |
| 3e-5 × 5k | pass | o | o, r |
| 3e-5 × 15k | pass | o, r | o, r |
| 1e-4 × 5k | r | r | o, r |
| 1e-4 × 15k | r | r | r |
| 1e-4 × 30k | r | r | r |

- **No setting passes on seeds 1 or 2.**
- **Every per-seed pass is on seed 0.** Its standard Object score (78%) is the lowest of the three
  seeds, which makes its floor the easiest to reach. This is the same seed that produced the
  low-update configuration.
- **Consequence:** the configurations that looked promising on seed 0 did not generalize.

## 4. Frontiers
- **The worst-seed success frontier:** (1e-5, 5k), (1e-5, 15k), (3e-5, 15k) and (1e-4, 15k).
  - Their worst-seed points, as (Object, Spatial), are (32, 10), (40, 6), (62, 2) and (82, 0).
  - The robust Spatial retention of **every** setting is at most 10%.
- **The worst-seed loss frontier:** (1e-5, 5k), (3e-5, 5k), (3e-5, 15k), (1e-4, 15k) and (1e-4, 30k).
  - This is a steep monotone trade: Object's probe loss goes 0.039 → 0.006 while Spatial's goes
    0.084 → 0.772.
- **GPM sits outside every fine-tuning frontier on every seed.**
  - Spatial success is 94–98%, against a best fine-tuning worst-seed value of 10%.
  - Its Spatial probe loss is 0.039–0.043, below any fine-tuning setting (at least 0.075).
  - It pays for this with Object success 0–12 pp below the standard fine-tuning recipe (0, −12 and −6 pp); its Object
    probe loss of 0.007 is close to the standard's 0.006.

## 5. Update size and forgetting
- **Loss-level forgetting rises with the size of the update** across the grid. Spatial's probe loss
  goes from 0.08 at a relative update of 0.03 to 0.75 at 0.28.
- **Plasticity rises with it:** Object's probe loss goes from 0.039 to 0.006.
- **Size alone does not determine forgetting.** At the same learning-rate × steps product,
  1e-5 × 15k moves the weights less than 3e-5 × 5k (0.046 against 0.053). Yet it forgets more at the
  loss level (0.13–0.17 against 0.09–0.10) and learns Object slightly better (0.021 against 0.024).
  Where the update goes matters, not only how large it is.
- **The update is spread evenly.** Registry trunk, registry decoder and non-registry parameters all
  move in similar proportion, with the trunk moving the most at high learning rates (0.34 at 30k).

## 6. One-factor effects

These are around the anchor (5k, 3e-5), on seeds 0 and 1. Differences are in percentage points for
success and in absolute probe loss. The probe intervals are matched-batch bootstrap intervals over
the 16 registered probe batches: they measure stability over those units, not population-level
uncertainty.

| Variant | Object success | Spatial success | Object probe | Spatial probe |
|---|---|---|---|---|
| constant LR, same peak | −18 / +26 · `inconclusive` | −42 / −28 · `supported_both_seeds` | +0.001 / +0.003 · `supported_both_seeds` | −0.007 / −0.011 · `supported_both_seeds` |
| batch 32 × 10k (more updates) | +10 / +18 · `supported_one_seed` | −18 / −26 · `supported_both_seeds` | −0.004 / −0.004 · `supported_both_seeds` | +0.005 / +0.005 · `supported_both_seeds` |
| batch 128 × 2.5k (fewer updates) | 0 / +4 · `inconclusive` | −4 / +2 · `inconclusive` | +0.006 / +0.006 · `supported_both_seeds` | −0.002 / −0.004 · `supported_both_seeds` |
| weight decay 0 | identical (bitwise) · `inconclusive` | identical · `inconclusive` | 0 · `inconclusive` | 0 · `inconclusive` |
| weight decay 1e-2 | +2 / −6 · `inconclusive` | 0 / −8 · `inconclusive` | +0.00002 · `directionally_consistent` | +0.0002 / +0.0003 · `supported_both_seeds` (negligible) |

**Reading:**
- **The number of updates matters.** At equal examples seen, more (smaller) updates buy plasticity
  and cost retention (batch 32), and fewer do the reverse at the loss level (batch 128). This is
  consistent with §5.
- **Behaviour and loss disagree for the constant schedule.** The constant LR forgets more in rollouts
  on both seeds, yet shows a slightly *lower* Spatial probe loss. Behavioural forgetting and
  probe-loss forgetting need not agree in magnitude.
- **Weight decay does nothing.** AdamW's decoupled decay multiplies weights by `1 − lr·wd`. At
  lr 3e-5 and wd 1e-4 that is `1 − 3e-9`, which rounds to exactly 1.0 in fp32, so wd 0 gives a
  bitwise-identical model (0 of 656 tensors differ).
  - The same holds for every registered run (lr 1e-4 × wd 1e-4 = 1e-8), so **the recipe's weight
    decay has effectively been off throughout the project.** No reported result changes, but the
    methods section should state it.
  - wd 1e-2 is resolvable (237 tensors change), but its effects are negligible. The "supported"
    Spatial-probe label reflects how tight the intervals over identical batches are, not a
    meaningful size: +0.0002 on a loss of 0.09.

## 7. Disclosures
1. **The probe tolerance was amended** (`790ddc8`).
   - The registered check (rtol 1e-5, atol 1e-8) failed at a relative difference of 1.4e-5.
   - A diagnostic over all 62 recomputations (31 checkpoints × 2 tasks) found 34 exact matches,
     a maximum of 2.1e-5 and none above 1e-4. The non-zero ones are values stored inside a training
     process and recomputed in a fresh one (GPU kernel noise).
   - The tolerance was amended to rtol 1e-4, atol 1e-7, after every training run and before the
     official summary. The check only confirms that the same probe was computed.
   - **The diagnostic ran the full summary logic in scratch**, so its headline (no setting achieves
     both) was seen before the amendment. The tolerance gates whether the summary runs, not what it
     concludes, and the grid's rows were already public by then.
2. **The weight-decay no-op** (§6) is a numerical property of fp32 AdamW at these learning rates, not
   a bug. The override demonstrably reaches the optimizer: the run config records 0.0, and 1e-2
   changes 237 tensors.

## 8. Consequences
- **Tuning the fine-tuning optimizer does not solve the T1 → T2 trade-off in this setting.**
  - The frontier is monotone, and it is governed mainly by how far, and for how many updates, the
    weights move.
  - This holds across two budgets and three learning rates, plus the schedule, weight-decay and
    batch-size variants.
- **The projection methods' advantage is not an artefact of an aggressive recipe.** GPM retains
  94–98% of Spatial at an Object cost of 0–12 pp, a point no fine-tuning setting approaches on any
  seed. This supports the thesis motivation directly.
- **Next steps, to plan:**
  - the SGP sensitivity study on the **standard** recipe, since the low-update backbone failed (the
    pilot's design carries over: controls, eligible α, the nominal and realized update split);
  - a methods note on the weight-decay no-op;
  - the remaining SGP-baseline seeds and the Stage A matrix.

## 9. Artifact map
```
results/optimizer_study/summary.json          # rows (all settings x seeds, provenance), probe units per batch,
                                              # update sizes, frontiers (per seed, robust), existence, one-factor
results/optimizer_study/seed{0,1,2}/opt__*/   # published runs: config, losses, eval.json, row.json, final.pt
results/optimizer_study/done_*.json           # per-part completion markers
results/logs/queue_20260928_135907_optimizer_study/, results/logs/optimizer_study_summary_*.log
configs/analysis/optimizer_study.yaml (77b1785; tolerance amended 790ddc8), flowcl/experiments/optimizer_study.py
```
