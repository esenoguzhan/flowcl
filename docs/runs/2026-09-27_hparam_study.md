# Training-budget and learning-rate study (exploratory, single seed)

**Status:** exploratory, seed 0 only. It changes no registered comparison. Every paired method run
stays at 30k steps and learning rate 1e-4.

**Headline findings:**
1. **Success saturates early.** Single-task success reaches 90% by 5k steps on both Object and
   LIBERO-10 and stays within noise of Gate 0's 30k result from then on.
2. **The held-out loss rises from 2.5k steps on.** It is lowest at about 2.5k steps and then climbs
   about 4× by 30k, while the training loss keeps falling. That is evidence consistent with
   overfitting the demos, but it does not hurt rollout success.
3. **Training longer forgets more, at the loss level.** When Object is fine-tuned after Spatial,
   5k steps learn Object as well as 30k, and every further step raises Spatial's probe loss (8.4×
   from 2.5k to 30k). In rollouts, Spatial is already fully lost at 5k.
4. **The learning rate matters most.** At 5k steps, lr 3e-5 learns Object as well as 1e-4 (76% in
   both) and **keeps 50% of Spatial**, against 0% at 1e-4: **+50 pp [+36, +64]**. That is plain
   fine-tuning, with no continual-learning method.

**Dates (local, CEST), 27 Sep:**
- the trajectories ran 10:15 → 12:56;
- the budget-vs-forgetting step crashed at 12:56 (§5) and was fixed and resumed 15:46 → 18:30;
- the learning-rate sweep ran 18:30 → 19:17.

**Code, clean:**
- `e00d025` for the trajectories;
- `8e6785a` for the budget and learning-rate parts. The fix touched only the CL reference lookup;
  the trajectory code is identical.

The config `configs/analysis/hparam_study.yaml` was committed in `79658c0` before any run. That
includes the B\* rule and every reported quantity.

**Cite these reports:**
- `results/hparam_study/trajectory/traj__{libero_object,libero_10}__seed0/report.json`;
- `results/hparam_study/cl_budget/report.json`;
- `results/hparam_study/cl_lr/report.json`.

---

## 1. Part 1 — training trajectories (single task)

**The setup:**
- **Recipe:** Gate 0's (policy, seed 0, 30k steps, lr 1e-4, batch 64, AMP), with the same seeding
  order as `train_on_tasks`.
- **Held-out demos:** the last 5 of the 50. Training used the other 45, and the normalization stats
  were fitted on those 45.
- **Checkpoints:** saved at 1k, 2.5k, 5k, 10k, 15k, 20k and 30k steps. These are mid-schedule
  checkpoints, so each is a lower bound on what a dedicated run of that length would reach.
- **Rollouts:** 50 at each checkpoint, under Gate 0's episode seeds, so every point is paired with
  Gate 0's 30k result.
- **Loss banks,** built once and reused unchanged at every checkpoint, each with K = 8 fixed
  `(s, A_0)` draws per chunk:
  - the **held-out bank** is every held-out chunk (691 for Object, 1 340 for LIBERO-10);
  - the **training bank** is an equal-size random subset of training chunks.

**Object** (Gate 0, 50 demos, 30k steps: 92% [84, 98]):

| Steps | Success [95% CI] | Paired vs Gate 0 | Train bank | Held-out bank | Held-out / train |
|---:|---|---|---:|---:|---:|
| 1 000 | 0% | −92 pp | 0.075 | 0.096 | 1.3 |
| 2 500 | 76% [64, 88] | −16 [−28, −4] | 0.046 | **0.095** | 2.0 |
| 5 000 | **90%** [82, 98] | −2 [−12, +8] | 0.032 | 0.114 | 3.6 |
| 10 000 | 84% [74, 94] | −8 [−20, +4] | 0.022 | 0.152 | 6.8 |
| 15 000 | 92% [84, 98] | 0 [−10, +10] | 0.016 | 0.204 | 12.7 |
| 20 000 | 98% [94, 100] | +6 [−2, +16] | 0.012 | 0.268 | 23.1 |
| 30 000 | 94% [86, 100] | +2 [−8, +12] | 0.007 | **0.373** | 51.6 |

**LIBERO-10** (Gate 0, 50 demos, 30k steps: 100%):

| Steps | Success [95% CI] | Paired vs Gate 0 | Train bank | Held-out bank | Held-out / train |
|---:|---|---|---:|---:|---:|
| 1 000 | 18% [8, 30] | −82 pp | 0.088 | 0.114 | 1.3 |
| 2 500 | 38% [24, 52] | −62 | 0.055 | **0.111** | 2.0 |
| 5 000 | **90%** [80, 98] | −10 [−20, −2] | 0.039 | 0.124 | 3.2 |
| 10 000 | 94% [86, 100] | −6 [−14, 0] | 0.028 | 0.165 | 5.8 |
| 15 000 | 94% [86, 100] | −6 [−14, 0] | 0.020 | 0.210 | 10.4 |
| 20 000 | 92% [84, 98] | −8 [−16, −2] | 0.015 | 0.272 | 18.1 |
| 30 000 | 92% [84, 98] | −8 [−16, −2] | 0.010 | **0.404** | 41.2 |

**Reading:**
- **The loss level.** The held-out loss is lowest at about 2.5k steps in both tasks, then rises
  steadily to about 4× that minimum, while the training loss falls about 10×.
  - The 8 fixed draws agree closely at every checkpoint (for example, 0.338–0.397 at Object's 30k),
    so the trend is not probe noise.
  - A sustained rise in held-out loss while the training loss falls is evidence consistent with
    overfitting the demos.
- **The behaviour.** Success plateaus at about 90% by 5k steps, or 5–10k for LIBERO-10.
  - On Object, every point from 5k on is within noise of Gate 0's 30k result.
  - LIBERO-10's plateau sits 6–8 pp under Gate 0's ceiling of 100%. It is flat from 10k on, so it
    points to this run's 45 demos (against Gate 0's 50) rather than to the number of steps.
  - Memorizing the demos does not hurt closed-loop success here.
- **Budget.** About 5–10k of the 30k steps are enough behaviourally, for both tasks.

## 2. Part 2 — budget vs forgetting (Object after Spatial)

**The setup:**
- The start is seq_ft seed 0's `stage0.pt`, the Spatial policy, which is bitwise the GPM runs'
  stage 0.
- Object is fine-tuned seq_ft-style (all parameters, 50 demos, lr 1e-4), with each budget on its
  own warmup and cosine schedule.
- Training uses seq_ft stage 1's data, `s` and `A_0` stream, and rollouts use seq_ft's seed
  namespace. Every variant is therefore paired, episode for episode, with seq_ft's own 30k stage 1.
- The probe losses use the forgetting diagnostics' fixed batches.

| Steps | Object learned | Paired vs 30k | Spatial retained | Paired vs 30k | Spatial probe loss | Object probe loss |
|---:|---|---|---|---|---:|---:|
| 2 500 | 58% | −20 [−32, −8] | **12%** | +12 [+4, +22] | 0.086 | 0.025 |
| 5 000 | 76% | −2 [−16, +12] | 0% | 0 | 0.169 | 0.016 |
| 10 000 | 74% | −4 [−18, +10] | 0% | 0 | 0.341 | 0.011 |
| 15 000 | 82% | +4 [−6, +16] | 0% | 0 | 0.473 | 0.009 |
| 20 000 | 82% | +4 [−8, +16] | 0% | 0 | 0.569 | 0.008 |
| 30 000 (seq_ft) | 78% | — | 0% | — | **0.725** | 0.006 |

**Reading:**
- **Plasticity saturates at about 5k steps.**
- **Forgetting keeps growing at the loss level.** Spatial's probe loss rises with every budget
  (8.4× from 2.5k to 30k).
- **In rollouts, forgetting is already total at 5k.** Rollout success is a threshold measure: once a
  task fails, further damage no longer shows.
- **Budget alone is not the lever.** Shortening the budget would not have prevented the rollout
  forgetting at this learning rate.

**The pre-registered budget rule gives B\* = 5k.** It is the smallest budget whose Object success
(76%) is within 10 pp of seq_ft's 78%. 5k, 10k, 15k and 20k all qualify.

## 3. Part 3 — learning rate at B\* = 5k

| Learning rate | Object learned | Paired vs 30k | Spatial retained | Paired vs 30k | Spatial probe loss | Object probe loss |
|---|---|---|---|---|---:|---:|
| 3e-5 | 76% | −2 [−18, +14] | **50%** | **+50 [+36, +64]** | **0.090** | 0.024 |
| 1e-4 (the recipe's) | 76% | −2 [−16, +12] | 0% | 0 | 0.169 | 0.016 |
| 3e-4 | 80% | +2 [−14, +18] | 0% | 0 | 0.486 | 0.015 |

**Reading:**
- **The trade-off is steep and favourable.** At lr 3e-5, Object is learned exactly as well (76%)
  and half of Spatial survives. Spatial's probe loss is about half that at 1e-4, and an eighth of
  the 30k recipe's.
- **The cost is only at the loss level.** Object is less tightly fitted: 0.024 against 0.016.
- **lr 3e-4 forgets more** (0.486) for no gain.

## 4. What this does and does not show

**It shows,** for one seed and one task pair (Spatial → Object):
- The 30k, lr 1e-4 recipe trains far past what learning a task needs. Behaviourally, 5–10k steps
  suffice.
- Extra steps and higher learning rates mostly buy forgetting, at the loss level and, once a task
  has been lost, in rollouts too.
- A plain fine-tuning recipe with a lower learning rate and a shorter budget keeps half of the
  previous task where the current recipe keeps none.

**It does not show:**
- that this holds on other seeds, task pairs, or the four-task sequence (50 episodes per cell,
  seed 0 only);
- that a low-forgetting recipe changes how methods compare. GPM's and SGP's advantages were all
  measured in the strong-forgetting regime of the current recipe.
- anything about the registered comparisons, which remain valid. Every method used the same recipe.

**Why it matters for the thesis:**
- How much seq_ft forgets depends heavily on the training recipe.
- Any claim of the form "method X reduces forgetting" is conditional on that recipe, and the
  write-up should say so.
- A study of the stability–plasticity trade-off as a function of the learning rate and the budget
  would itself be a legitimate result.

## 5. Disclosures
1. **The crash and the fix.** The budget-vs-forgetting step crashed at 12:56, one second after it
   started, before training.
   - **Cause:** the code read `seed_namespace_run_id` from seq_ft seed 0's `result.json`, which Gate
     1 wrote before that field existed.
   - **Fix (`8e6785a`):** the namespace now comes from the runner's rule (`seed_namespace_run_id`)
     and must agree with the recorded field (when present) and with the reference stage's rollout
     namespace. A test was added.
   - **Resumption:** the queue was resumed with `--from-step 2`, and the trajectory reports were
     reused. No result was produced before the fix. About 3 GPU-hours were lost while it sat idle.
2. **Where the code was written.** The study was implemented in a separate git worktree while the
   SGP seed-0 queue ran, because any file in the main tree would have made that queue's later steps
   refuse a dirty tree.
3. **A LIBERO config incident.**
   - **What happened.** Importing flowcl from that worktree rewrote the shared
     `~/.libero/config.yaml` to point at the worktree's empty LIBERO directory. It stayed that way
     for about 5 minutes on the evening of 26 Sep, while the SGP baseline was in the middle of T1
     training, which reads no LIBERO files.
   - **Response.** It was restored from the main tree, and every later worktree command used a
     private `LIBERO_CONFIG_PATH`.
   - **Effect: none.** The SGP baseline's stage-0 evaluation ran afterwards with the correct paths,
     and its stage 0 is bitwise identical to plain GPM's.

## 6. Next steps (to decide)
1. **Follow up the learning-rate finding** before relying on it:
   - lr ∈ {1e-5, 3e-5} × budget ∈ {5k, 10k}, on 2–3 seeds;
   - possibly a second task pair;
   - then, if it holds, the four-task sequence for seq_ft.
2. **The exploratory SGP sensitivity study** (the SGP seed-0 record, §7), starting with α. The
   learning rate interacts with SGP's soft directions, since every non-hard direction moves in
   proportion to the step, so the two studies should be designed together.
3. **SGP seeds 1–2 at α = 25**, still open.
4. **Any change to the recipe for future experiments needs its own plan:** re-running Gate 0 and
   Gate 1 and the references. Nothing is retrofitted to the existing comparisons.

## 7. Artifact map
```
results/hparam_study/trajectory/traj__<suite>__seed0/   # report.json (rows: success, paired vs Gate 0,
                                                        # train/held-out bank losses per draw), banks.pt
                                                        # (fixed chunks, s, A_0), losses.json, eval/,
                                                        # checkpoints/step{1000..30000}.pt, stats.json
results/hparam_study/cl_budget/report.json              # reference (seq_ft 30k) + one row per budget
results/hparam_study/cl_lr/report.json                  # B* choice (rule inputs) + one row per learning rate
results/hparam_study/cl_{budget,lr}/cl__libero_object__b<steps>__lr<lr>__seed0/  # config, losses, eval,
                                                        # final checkpoint, row.json
results/logs/queue_20260927_101540_hparam/, results/logs/queue_20260927_154619_hparam/
configs/analysis/hparam_study.yaml (79658c0), flowcl/experiments/hparam_study.py
```
