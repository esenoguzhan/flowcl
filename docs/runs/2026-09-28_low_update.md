# Low-update continual fine-tuning — held-out validation (not validated)

**Status:** the pre-registered verdict is **`not_validated`**. The configuration failed on **both**
held-out seeds, so the gated four-task chain and the projection pilot were skipped, as
pre-registered.

**In one sentence:** the configuration that looked like a free lunch on the seed that selected it
(5k steps at lr 3e-5 for T2) is, on held-out seeds, a **stability–plasticity trade-off**. It cuts
loss-level forgetting by about 8× on every seed, but it learns Object 30–42 pp worse than the
standard recipe, and its behavioural retention is inconsistent (28% and 2%).

**Date (local, CEST):** 28 Sep 2026, 00:32 → 01:25, in queue `results/logs/queue_20260928_003243_low_update`.

**Code:** `b6b644a`, clean. The rule `configs/analysis/low_update.yaml` was committed in the same
commit, before any run.

**The selection run.** Seed 0's stage 1 is the exploratory study's run
`results/hparam_study/cl_lr/cl__libero_object__b5000__lr3e-05__seed0`.
- It was produced at `8e6785a`, and its checkpoint SHA-256 is `a291819a…`.
- It was verified against the frozen configuration: 5 000 steps, lr 3e-5, seq_ft's stage-1 stream
  seed and namespace, starting from `stage0.pt`.
- It was not retrained.
- Its report was written after the queue, at 13:09, because the queue only ran the held-out seeds.
  That report is CPU-only and trains nothing.

**Cite these reports:**
- `results/low_update/validation.json` (the verdict);
- `results/low_update/validation_seed{0,1,2}.json` (the rows and classifications);
- `results/low_update/seed{1,2}/stage1/` (the published runs: `row.json`, `eval.json`,
  `final.pt`).

---

## 1. What was tested

**The frozen configuration**, from the exploratory study (`docs/runs/2026-09-27_hparam_study.md`),
selected on seed 0 and not tuned further:
- T1 is the standard recipe, the shared seq_ft `stage0.pt`;
- T2 (Object) is trained for **5 000 steps at lr 3e-5**, on its own warmup and cosine schedule;
- everything else follows the seq_ft recipe, with all parameters trained.

**Pairing:** the run uses seq_ft's own stage-1 stream seed and rolls out under seq_ft seed s's
namespace. Every cell is paired, episode for episode, with seq_ft, plain GPM and adaptive GPM of the
same seed.

**The pre-registered rule, per seed.** A seed passes only if both hold:
1. **Plasticity:** Object ≥ seq_ft's Object − 10 pp, on the point estimate.
2. **Retention:** the paired Spatial gain over seq_ft has a 95% CI lower bound above 0.

**The verdict** is `validated_on_held_out_seeds` only if **seeds 1 and 2** both pass. Seed 0 selected
the configuration: it is reported, not counted.

## 2. Results

Success rates, 50 paired episodes per cell; paired differences against seq_ft's standard stage 1
(30k steps, lr 1e-4):

| Seed | Object learned | seq_ft Object | Floor | Spatial retained | Paired Spatial gain | Plasticity | Retention | |
|---|---:|---:|---:|---:|---|---|---|---|
| 0 (selection) | 76% | 78% | 68% | 50% | +50 pp [+36, +64] | pass | pass | reported only |
| **1** | **60%** | 90% | 80% | 28% | +28 pp [+16, +40] | **fail** (−30 pp [−44, −16]) | pass | **fail** |
| **2** | **50%** | 92% | 82% | 2% | +2 pp [0, +6] | **fail** (−42 pp [−58, −28]) | **fail** | **fail** |

**Verdict: `not_validated`.** Seed 1 failed on plasticity; seed 2 failed on plasticity and
retention.

**Loss level**, from the diagnostics' fixed-batch probe, against the paired references' values at
the same stage:

| Seed | Spatial probe loss: low-update / seq_ft / GPM | Object probe loss: low-update / seq_ft / GPM |
|---|---|---|
| 0 | 0.090 / 0.725 / 0.042 | 0.024 / 0.0064 / 0.0072 |
| 1 | 0.090 / 0.719 / 0.043 | 0.023 / 0.0063 / 0.0072 |
| 2 | 0.099 / 0.772 / 0.039 | 0.024 / 0.0064 / 0.0071 |

**Against the projection methods.** At stage 1, plain and adaptive GPM are bitwise identical.
Low-update fine-tuning is behind them on both counts:
- **Spatial:** 94–98% under GPM; the paired differences are −44, −70 and −92 pp.
- **Object:** −2, −18 and −36 pp.

## 3. Reading
1. **The held-out validation did its job.** On the seed that selected it, the configuration looked
   like a free lunch: Object within 2 pp, half of Spatial kept. On both held-out seeds, Object falls
   30–42 pp short of the standard recipe.
   - Seed 0's standard Object was also the lowest of the three seeds (78%, against 90% and 92%), so
     the selection seed was an unusually easy comparison.
2. **Loss-level forgetting falls consistently.** Spatial's probe loss after training Object is about
   0.09–0.10 in every seed, against 0.72–0.77 under the standard recipe.
3. **Behavioural retention does not follow consistently.** It is 50%, 28% and 2% at almost the same
   loss level. Rollout success depends on crossing a task-specific threshold, so a similar
   loss-level reduction can leave a task fully forgotten or half kept.
4. **The plasticity cost is real and consistent.** Object's probe loss is about 3.7× the standard
   recipe's on every seed. 5k steps at lr 3e-5 simply move the weights less, which is exactly the
   trade-off a smaller update implies.
5. **Hard projection dominates on this transition.** At the standard recipe it keeps Spatial at
   94–98% with a similar or smaller Object cost. The projection methods' advantage is not an
   artefact of an overly aggressive fine-tuning recipe, at least at T1 → T2.

## 4. What this does and does not show
- **It shows** that lowering the update size (lr and budget together) reduces loss-level forgetting
  but does not give a free, validated improvement at this setting. The exploratory study's headline
  result (seed 0) does not generalize.
- **It does not show** that no optimizer setting improves the trade-off. Only one point, (5k, 3e-5),
  was validated, and the budget and learning rate were changed together. The selection used one
  seed, and its criteria allowed a 10-pp Object loss.
- **The registered comparisons are unaffected** (30k, lr 1e-4 for every method).

## 5. Consequences
- **Low-update continual fine-tuning is not adopted** as a validated baseline. It stays an
  exploratory observation: a real but costly trade-off.
- **The gated follow-ups did not run:** neither the four-task chain nor the seed-0 projection pilot,
  as pre-registered.
- **Next: a broader optimizer study, planned separately.** Its lessons from this failure:
  - select on more than one seed, or on a criterion that is less sensitive to one seed's baseline;
  - separate the learning rate from the budget;
  - measure plasticity on the same fixed scale as retention.

## 6. Artifact map
```
results/low_update/validation.json               # verdict (held-out seeds 1-2), seed 0 reported
results/low_update/validation_seed{0,1,2}.json   # per seed: row (cells, paired vs seq_ft/GPM/adaptive GPM,
                                                 # probe losses and references) + classification
results/low_update/seed{1,2}/stage1/             # published runs: config, losses, eval.json, row.json,
                                                 # checkpoints/final.pt (parent: seq_ft stage0.pt, SHA recorded)
results/logs/queue_20260928_003243_low_update/   # queue and step logs
configs/analysis/low_update.yaml (b6b644a), flowcl/experiments/low_update.py
```
