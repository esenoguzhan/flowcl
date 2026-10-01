# Thesis plan: 1 Oct 2026 → submission ~1 Mar 2027 (21 weeks), revised after external review

## Context

As of 1 Oct the project has a replicated causal result. Fixed total-energy GPM under-protects
later tasks, and the adaptive new-energy rule (`gpm_ne90`) fixes this: Object retention
+56/+64/+66 pp over 3 seeds. It also has a full forgetting/interference characterization (Gates 1–4,
the optimizer frontier) and an SGP characterization (α controls protection; under AdamW the scaled
directions consume Adam's step budget).

The adaptive rule's measured weakness is capacity:
- The trunk is 94% occupied before T4 and 99% after it.
- T4's loss is 1.6× plain GPM's on every seed.
- The T5 sweep's hard arm, from the post-T4 state, learns T5 like the free arm (72 vs 74%) but at
  2.5× the probe loss (0.016 vs 0.0064).
- Under that hard arm, old tasks still erode: T3 94→78% and T4 92→74%.

**Decisions taken (1 Oct):**
- Stage B hardware is **required**.
- All four baselines (replay, EWC, LoRA, ConSFT), 3 seeds.
- Second curriculum: **reverse order** (`seq_hetero_reverse`).

**Thesis story:** fixed protection under-protects later tasks → adaptive allocation fixes
retention → stronger protection consumes capacity → which directions actually need protection?

**Minimum acceptable thesis (to confirm with the supervisor at D1).** It must hold even if **both**
the two-sided method and the predictor study come out negative:
- characterization;
- adaptive protection, with its controls (e99, reverse order);
- the four baselines;
- the capacity limits over 8 tasks;
- hardware validation;
- each negative result reported with its pre-registered rule.

**Only two workstreams are optional additions to this floor:** Workstream D (Track 2) and the
two-sided method in Workstream E. Workstreams A, B, C and F, and E3's reference arms, *are* the
floor.

## Standing conventions

- **Pre-register.** A rule goes in `configs/analysis/<study>.yaml` and is committed before any run.
  Logic lives in `flowcl/experiments/<study>.py`, with a thin CLI in `scripts/`. Queues follow the
  `scripts/queue_seed_pair.sh` pattern: sequential, `--from-step`, `QUEUE_DRY_RUN`. Each study gets
  a record in `docs/runs/`. One process on the GPU at a time.
- **Pairing.**
  - Every run uses the seq_ft seed namespace (`flowcl/train/continual.py::seed_namespace_run_id`).
  - The **T1 pairing check applies only to methods whose T1 is plain fine-tuning by design**:
    replay, EWC (no penalty at T1), LoRA (T1 is full fine-tuning) and the GPM family. ConSFT
    declares its T1 behaviour in `docs/consft_notes.md`; if it changes the T1 objective, it is
    exempt and that is recorded.
  - **Identity checks** (`stage_identity_check`) run only where the equivalence is intentional:
    e99 against plain GPM at stages 0–1 (the T1 memory is 0.95); two-sided against plain GPM at
    **stage 0 only**.
- **Statistics.**
  - Rollout bootstraps measure evaluation uncertainty for one trained policy, not robustness over
    training seeds. Main claims show **per-seed paired effects explicitly**. With 3 seeds there is
    no seed-level CI; the claim requires a consistent sign.
  - **Seed 0 is the development seed** (brackets, pilots, continuation decisions). **Seeds 1–2 are
    the replication** and are reported separately.
  - "Within 10 pp" is a **screening rule** in pilots (point estimate).
  - **Final non-inferiority is evaluated separately on each replication seed (1 and 2).** The
    rollout-paired 95% CI of the difference must have its lower bound above −10 pp on each seed.
    That CI supports a claim about **those trained policies**, not about future training runs.
  - The same sign on 3 seeds is **directional replication**, not statistical certainty. Effect
    sizes are always shown per seed.
- **Metric wording.** ACC (final average success) is reported as **overall performance**, since it
  combines acquisition and retention. Forgetting is reported separately (NBT, per-task
  forgetting, F_first).
- **Testing.** Every new quantity gets a synthetic-case unit test plus a shape and orientation
  assertion (README §11). Before each queue, smoke-test at 200 steps into scratch.

## Workstream A: infrastructure and protocol fixes (W1)

1. **Stage-boundary resume** in `run_continual` (`flowcl/train/continual.py`), with
   `--resume-run DIR --start-stage K` in `scripts/run_continual.py`.
   - Load `checkpoints/stage{K-1}.pt` and verify its SHA. Restore the method state from the
     checkpoint's `method_artifacts` (`GPM.restore_memory` in `flowcl/methods/gpm.py`; a
     restore for each new method that stores state).
   - Assert that the curriculum's first K stages and the namespace match the source run's.
   - Rebuild the runner's text-cache state with `warm_runner_cache`
     (`flowcl/experiments/low_update.py`).
   - **Test:** resume the plain-GPM seed-0 run at stage 3 into scratch. Require:
     - the stage-3 checkpoint bitwise equal (0 of 656 tensors);
     - `memory_task3.pt` equal;
     - the per-episode successes in `eval/stage3.json` equal.
   - **Limit:** it recovers at stage boundaries. A crash inside a stage still loses that stage
     (about 45 min plus eval), not the whole run.
2. **`seq_hetero_t8`**: `seq_hetero_t5` plus T6–T8 from `t5_sweep.yaml` → `follow_up`.
   - It inherits seq_hetero's seed namespace, as `low_update.seed_context(extend_curriculum=…)`
     already does.
   - Gate 0 on T6–T8 (`scripts/gate0.py --tasks … --out-dir results/gate0_t8`), with the
     pre-registered backups.
3. **Seen-task-only evaluation** for 8-task and hardware runs (`eval_unseen: false`).
   - **Justification, worded correctly:** zero-shot success `R[j−1][j]` was 0 in every cell of
     every run. FWT therefore reduces to minus the mean single-task reference (−97.3 pp) and
     carries no information. Unseen tasks also cost 600-step timeouts.
   - Existing 4-task protocols are unchanged.
4. **ε schedule for GPM**: `eps_later` in `GPM.__init__` and the configs. Display name
   `gpm_projected_adam_e99`.
5. **Summary-metric uncertainty.**
   - A rollout-level paired bootstrap per run, for ACC, NBT and AUC (`flowcl/analysis/metrics.py`).
   - Per-seed tables in `flowcl/analysis/tables.py`, with the two sources of uncertainty labelled
     separately.
6. **Metric naming.** README §8.2 defines `F_1 = R[1][1] − R[2][1]` (first-task forgetting).
   `metrics.py:325` serializes `F_1` as final average success.
   - Tables and the thesis use **ACC** for final average success and **F_first** for README's F₁.
   - Update README §8.2. `tables.py` maps old `result.json` keys; existing artifacts are not
     rewritten.
7. **Close SGP.** When the T5 sweep finishes, write its record.
   `docs/methods_notes.md` collects the weight-decay no-op, the SGP–Adam coupling and the
   text-cache state. **No further SGP studies.**

## Workstream B: baselines (W1–W6; ConSFT runs in W7–W9)

| Method | Files | Design |
|---|---|---|
| **Replay** | `flowcl/methods/replay.py`, `configs/method/replay.yaml` | `on_task_end` buffers the first 10 demos of each finished task under the frozen T1 stats. `build_batch` returns 50% current-task and 50% buffer chunks, with generators from `derive_seed(ns, "replay::<task>", stage)`. At T1 it returns `None`, so stage 0 is bitwise seq_ft. `is_exemplar_free = False`; `stored_bytes` = buffer size. Fixed a priori (spec values). |
| **EWC** | `flowcl/methods/ewc.py`, `configs/method/ewc.yaml` | Online EWC. After each task: a diagonal empirical Fisher (per-sample gradients, fp32), summed across tasks, anchored at the post-task weights. The penalty is (λ/2)ΣF(θ−θ\*)² in `modify_loss`, in fp32. All parameters train. **λ is bracketed.** The full sequence (rather than spec §6's two-task probe) is a recorded deviation. |
| **Sequential merged LoRA** | `flowcl/models/lora.py`, `flowcl/methods/lora.py`, `configs/method/lora.yaml` | T1 is full fine-tuning. From T2 the base is frozen. Each task gets fresh rank-r adapters on every registry layer (a forward hook adding `scale·B A x`), **merged at task end**. No isolated per-task adapters are kept (there is no task ID at inference), and the thesis names it exactly this way. **r is bracketed** over {4, 16, 64}. |
| **ConSFT** | `docs/consft_notes.md` first, then `flowcl/methods/consft.py` | Reimplement from the paper. The review cites arXiv 2605.08879 (App. C.1): per-sample weighting, stop-gradient, and a temperature schedule whose initial value and annealing length need task-specific tuning. **Verify that against the paper.** If confirmed, a small pre-registered calibration bracket is allowed. The notes must address loss normalization relative to our masked loss. **The notes fix ConSFT's T1 policy before any calibration:** if it acts from T2, calibrate from seq_ft `stage0.pt`; if it acts during T1, train a ConSFT T1 on seed 0 and calibrate from that checkpoint. It uses `modify_loss(..., outputs=)`. |

**Selection** (`configs/analysis/baseline_selection.yaml`):
- **Setup:** EWC λ and LoRA r on seed 0 at T1→T2, from `seq_hetero__seq_ft__seed0/stage0.pt`,
  paired with seq_ft stage 1. ConSFT's schedule, if bracketed, starts from the checkpoint its T1
  policy dictates (see the table). This reuses the stage path
  in `flowcl/experiments/low_update.py` (`stage_setup` / `train_stage`) and `optimizer_study.py`.
- **Rule:** the value with the highest Spatial retention among those keeping Object ≥ seq_ft's
  Object − 10 pp. Ties go to the weaker setting. The choice is frozen in a bundle.
- **Disclosed:** the selection uses one transition on the development seed, and that seed is also
  in the final evaluation.

**Reports:**
- Make the memory-specific checks in `flowcl/experiments/sequence_report.py` conditional on the
  method.
- The main Stage A table (`flowcl/analysis/tables.py`, `scripts/make_tables.py`) has: success
  with CIs, ACC/NBT/AUC per seed, plasticity, stored MB, trainable parameters, wall-clock, and the
  exemplar-free flag.

## Workstream C: controls (W1–W4)

1. **High-protection control:** `gpm_projected_adam_e99`, 3 seeds, identity-checked at stages 0–1.
   - Its rule (`configs/analysis/high_protection.yaml`) reuses the cells in
     `flowcl/experiments/adaptive_report.py`, plus trunk ρ after each task.
   - **Question:** does the adaptive allocation matter, or only more protection?
2. **Reverse order** (`seq_hetero_reverse`): seq_ft, plain GPM and adaptive GPM × 3 seeds, through
   `queue_seed_pair.sh`, parameterized by curriculum. Stats are fitted on LIBERO-10, and per-seed
   thresholds are registered.
   - **Rule** (`configs/analysis/reverse_order.yaml`): does plain GPM forget position 2 (Goal) or
     Object (now position 3)?
   - **Outcomes:** "position", "task", "both" or "neither".

## Workstream D: Track 2, offline predictors of closed-loop forgetting (W2–W6; no training)

**Two separate claims:**
- **(a) Within-task ranking:** does measure M rank a task's checkpoints by success?
- **(b) Cross-task calibration:** at equal M, do Spatial, Goal and Object have similar success?
  This is the motivating puzzle: at a probe loss of 0.09–0.10, success is 88–90%, 56% and 2%.

**Data.** Every existing (checkpoint, task) cell with 50 rollouts in which the task was already
trained. Invalid attempts are excluded.

**Duplicates.** Cells are **deduplicated by (checkpoint SHA-256, task, episode seeds)**. Many runs
share bitwise stage checkpoints and paired episodes:
- GPM stage 0 equals seq_ft stage 0;
- adaptive stages 0–1 equal plain GPM's;
- SGP stage 0 equals GPM's.

**Lineage hierarchy (through shared ancestors):**
- **Level 1, root:** the T1 checkpoint a cell descends from: seq_ft seed s `stage0`, a Gate 0
  single-task policy, or an hparam trajectory.
- **Level 2, branch:** the first checkpoint where the cell's history diverges from every other run
  under the same root.
- **Level 3:** the cell.

**Measures** (fixed banks, common random numbers):
- **M0** probe loss (`flowcl/analysis/probes.py`).
- **M1** loss per s-bin, using Gate 4's bins (`flowcl/analysis/flowtime.py`).
- **M2** loss per action component (position, rotation, gripper), and on the executed first 8
  steps against the rest.
- **M3** sampled-chunk error via `policy.sample(batch, noise=A0_bank)` against the demo chunk,
  plus a best-of-K variant. One demo can penalize a different valid action, so this is a
  disclosed limitation of M3.

**Validation:**
- **Dependence.** A **two-level cluster bootstrap**: resample roots, then branches within each
  root. There are few roots (three seq_ft seeds plus the single-task policies), so these CIs will
  be wide, and the plan accepts that.
- **Leakage.** Optimizer, coupling and T5 arms descend from checkpoints in the fitting set, so they
  are **not** a held-out test.
- **Test set:** the **4-task baseline runs** (replay, EWC, LoRA, and ConSFT when available).
  Bracket cells are excluded, because they are inspected for baseline selection before the freeze.
  This test measures **transfer to new methods under shared tasks and initialization histories**.
  It says nothing about new tasks or embodiments.
- **The freeze, with a hard deadline so that optional work never blocks required work:**
  - Development analysis plus measure and model selection end with a committed selection bundle.
    The target is Wed 28 Oct; the **hard deadline is Fri 30 Oct**.
  - The 4-task baseline runs are queued after the commit or at the hard deadline, whichever comes
    first.
  - **If the deadline passes without a freeze,** the baseline queue is released, and Track 2
    **relinquishes the held-out predictor claim**. It is then reported as exploratory and
    descriptive only. Sealing results is not attempted: crash checks need the logs.
  - The held-out evaluation on baseline cells may finish later (W5–W6).
  - **Angle B at D2 is available only if the selection bundle exists by D2.**
- **Claim rule (descriptive):** "M beats M0" requires two things on the baseline test cells: an
  improvement with a two-level-cluster CI excluding 0, and the same sign within every root.
  Otherwise the record says "not shown", which the minimum thesis tolerates.

**Code:**
- `flowcl/analysis/behavior.py`;
- `flowcl/experiments/behavior_predictors.py`;
- `configs/analysis/behavior_predictors.yaml`;
- `scripts/behavior_predictors.py`.

**Hardware use:** debugging and ordering only. It **never selects which checkpoints enter the
fixed hardware evaluation set**; doing so would filter failures out of the evidence.

## Workstream E: main bet, two-sided protection

**Claim, worded narrowly.** GPM protects `d_out × k_in` weight directions per layer. Protecting
`U⊗M` (`k_out·k_in`), with `U` the energy-thresholded basis of the output-side Gauss-Newton
factor `E[JᵀJ]`, frees capacity. It **may** preserve retention, but only under assumptions E2
must test empirically:
- old inputs stay in M;
- old sensitivities stay in U as the network changes;
- small velocity changes mean small behavioural changes;
- rare decisive directions are not lost below the energy cut.

**The update:**

    G ← G − U Uᵀ G M Mᵀ      (and the same on the realized AdamW step, as in after_step)

**U estimation.** Hutchinson: backpropagate `u ~ N(0, I)` through the masked `outputs["velocity"]`
and capture `grad_output` per registry layer. The hook pattern is
`flowcl/analysis/hooks.py::gradient_reachability`, and `extend_basis`
(`flowcl/analysis/subspace.py`) is applied to the output Gram.

**Exemplar-free accumulation, as in GPM.** `U_t` is captured on task t's data at the **end of
task t** only. The protected set is the conservative union `(∪U_t) ⊗ (∪M_t)`. That is a superset
of `∪(U_t ⊗ M_t)` and may itself saturate over 8 tasks, which E1 projects.

**Novelty check (W2), before committing.** Two works are close:
- **NCL** (Kao et al. 2021): Kronecker-factored curvature;
- **OGD** (Farajtabar et al. 2020): protects model outputs through gradient constraints.

Write the precise difference before D2: hard thresholded per-layer Kronecker projection, basis
construction from the GGN output factor, task-end accumulation, and Adam-safe update projection.

**E1, measurement (W3, about 2 h, no training)** (`configs/analysis/twosided_gonogo.yaml`,
`flowcl/experiments/twosided_gonogo.py`):
- **Setup:** adaptive GPM seed 0. `U_1`, `U_2`, `U_3` are captured at `stage0`, `stage1` and
  `stage2` on T1, T2 and T3 data respectively, which is the exact incremental protocol.
- **Reported:**
  - `k_out` at ε_out ∈ {0.90, 0.95, 0.99};
  - the **parameter-weighted protected fraction**, `Σ_l k_out,l·k_in,l / Σ_l d_out,l·d_in,l` over
    registry weights, for two-sided against GPM;
  - per-half and per-group breakdowns;
  - the union's growth T1→T3, plus a **forecast** to T8, labelled as a forecast; it cannot show
    that saturation will be avoided.
- **Pass:** the parameter-weighted trunk fraction ≤ 0.5 at ε_out = 0.95, against about 0.94 for
  GPM. Fixed in the committed rule.
- **Test:** unit-test the estimator on a 2-layer linear net, where `E[JᵀJ] = W₂ᵀW₂`.
- **This establishes potential capacity savings only.**

**E2, pilots (W4, about 6 h).** Arm types are added to the machinery of
`flowcl/experiments/t5_sweep.py` and `sgp_coupling.py`. The left projector goes into
`_ProjectedLayer` (`flowcl/methods/gpm.py`), with the residual check
‖Uᵀ D M‖ ≤ atol + rtol‖D‖.

**Unit tests:** the projector is idempotent and Frobenius-self-adjoint, removes `U⊗M`, and
reduces to GPM at `U = I`.

| Pilot | Start (memories by the incremental protocol) | Arms | Reference pair (hi, lo) for gap closure | Old-task screening |
|---|---|---|---|---|
| T4 | adaptive seed 0 `stage2.pt`; M = `memory_task2`; U = U₁∪U₂∪U₃ | `two_sided`, `two_sided_random` (random orthonormal U, same rank per layer, fixed seed), plus a new matched-start **`free`** arm (~1.5 h) | hi = adaptive stage 3, which is hard projection from the same start and stream (the stage path with `warm_text_cache` reproduces runner stages bitwise, per the coupling study); lo = `free` | T1–T3 each ≥ adaptive stage 3 − 10 pp |
| T5 | adaptive seed 0 `stage3.pt`; replayed M; U adds U₄ (captured at stage 3) | `two_sided`, `two_sided_random` | hi = sweep `hard`; lo = sweep `free` (same start) | T1–T4 each ≥ `hard` − 10 pp |

Both pilots thus measure gap closure **from a shared starting state**. The T4 closure of the
adaptive-vs-plain-GPM gap (L_G = plain GPM stage 3, a different history) is reported as a
secondary **benchmark comparison**, comparable to the SGP rule's g, and is not decisive.

**Each pilot passes only if all three criteria hold** (point estimates, development seed):
1. **Gap closure:** g = (L_hi − L_2s)/(L_hi − L_lo) ≥ 0.5, on the new task's 16 fixed probe
   batches.
   - g is applicable only if the reference gap is real: L_hi/L_lo ≥ 1.2 **and** the matched-batch
     CI of ln(L_hi/L_lo) is above 0.
   - Otherwise the pilot is **inconclusive on plasticity**, not passed. T5's known gap qualifies
     (0.016/0.0064 = 2.5). T4's matched-start gap is measured by the new `free` arm.
2. **Old-task screening**, as in the table.
3. **Beats random U:** Δ = mean over old tasks of ln(L_random / L_2s), the old-task probe loss
   averaged over each task's 16 matched batches. It needs Δ > 0 with a matched-batch bootstrap
   CI lower bound above 0 (10 000 resamples, stratified by task). Success differences are
   reported, not decisive.

- **The random-U control** separates useful sensitivity selection from simply imposing fewer
  constraints.
- **Scope:** the pilots test switching two-sided on at a late transition. A full run has
  two-sided from T2, which E3 covers.

**Decision D2 (end of W5):**

| Outcome | Next step |
|---|---|
| E1 and both pilots pass | **angle A** |
| E1 fails because `E[JᵀJ]` is near full rank, and Track 2 (frozen at W4) identified critical outputs | **angle B**: `u` restricted to those outputs, rerun E1 and E2 (+1 week) |
| neither passes | **minimum thesis**: E3 runs without a new method (capacity frontier and references only) |

**E3, the 8-task evaluation (W7–W11)** (`configs/analysis/main_eval.yaml`, committed at the end
of W6). The arms, per seed:

| Arm | How | GPU per seed |
|---|---|---|
| seq_ft | resume from stage 3 | ~6 h |
| plain GPM | resume | ~6 h |
| **adaptive GPM f = 0.9 (primary comparator)** | resume | ~6 h |
| adaptive f = 0.7 | full run | ~10 h |
| two-sided method | full run | ~10 h |
| replay | full run | ~10 h |

**Pre-registered definitions:**
- **Capacity** is the parameter-weighted protected fraction after the final task. It is monotone
  non-decreasing under union accumulation, so final equals maximum; it is also reported after
  each task.
- **Overall performance** is ACC over T1–T8. **Retention** is reported as NBT and per-task
  forgetting.
- **Plasticity** is the mean diagonal over T5–T8 (primary), with each task reported.
- **Wording:** plain, f 0.7 and f 0.9 form an **empirical performance–capacity frontier**, not a
  curve.

**Primary claim:** two-sided has higher overall performance (ACC) than adaptive f = 0.9, at ≤ its
capacity, **for these trained policies**.
- It requires the **same sign on all 3 seeds** (directional replication). Effects are shown per
  seed, with seeds 1–2 as the replication.
- Mean T5–T8 plasticity must be non-inferior at −10 pp **on each of seeds 1 and 2 separately**
  (the CI rule above).
- Forgetting (NBT, per-task) is reported alongside, not folded into the claim.

**Resource claims stay separate.** Freed update dimensions are not lower memory or runtime: U
storage and estimation are measured and reported.

**D3 (end of W8, seed 0, against adaptive f = 0.9 seed 0, point estimates):**
- **"Positive"** requires all three:
  - ACC(two-sided) > ACC(adaptive);
  - final parameter-weighted capacity(two-sided) ≤ capacity(adaptive);
  - mean T5–T8 plasticity(two-sided) ≥ plasticity(adaptive) − 10 pp.
- **Anything else is "negative".** It **stops that arm**, and its record is written.
- The seq_ft, GPM, adaptive, f 0.7 and replay arms continue to seeds 1–2 regardless.

## Workstream F: Stage B hardware (required)

- **F0, readiness (W1–W2, bounded to about 2 days):**
  - inspect the teleop log format (observations, action convention, timestamps and control rate);
  - confirm lab and robot access, **including W12–W13**;
  - time a scripted reset;
  - decide between training from scratch and transferring the trunk from LIBERO (spec §3.1).
- **F1, engineering plus end-to-end proof (W6–W8):**
  - build `configs/embodiment/agilex_dual.yaml` (D_action not padded), `flowcl/data/agilex_adapter.py`
    and `flowcl/envs/hw_env.py` (closed loop, scripted reset, initial-condition photos), with
    adapter tests;
  - then the **single-task proof**, a gate that must pass before F2 starts. About 15 demos of one
    task → train → 10 closed-loop rollouts on the fixed initial conditions. It **passes** only if
    all of the following hold (the thresholds are a judgement to confirm at D1):
    - open-loop replay of 3 recorded demos reproduces their trajectories within a tolerance set
      in F0. This is the hardware analogue of the §10.1 replay test: action convention, frame and
      timing;
    - the action and observation statistics are recorded and match F0's log inspection;
    - **≥ 6/10 successes**;
    - a measured **episode-cycle time** (execution, reset and annotation) ≤ 4 min.
  - **Repair window:** W8 is reserved for fixes.
  - **D-hw (end of W8) separates two cases:**
    - **Proof failed** (conventions, control or learning). No continual evaluation starts. Repair
      continues, and the supervisor decides by W9 whether Stage B goes ahead, is shortened, or is
      replaced with a documented single-task transfer study. A smaller scope does not fix a broken
      pipeline, so scope reduction is **not** the remedy here.
    - **Proof passed, robot time short.** Choose the evaluation plan from the budget table below,
      using the measured cycle time.
- **F2, data (W9–W10):** 3 tasks, including **at least one bimanual coordination task**, about 50
  demos each. The collection time is budgeted separately from the rollouts.
- **F3, single-task references (W11):** 3 tasks × 20 rollouts.
- **F4, continual runs (W14–W16):**
  - **Methods:** the method frozen at D4: adaptive GPM, unless two-sided is "positive" (the D3
    rule) on seed 0 **and on seed 1**, the first scheduled replication, by W11. Seed 2's result
    is not used for D4. Plus seq_ft and replay.
  - **Protocol:** seen-only evaluation, 20 rollouts per cell, failure taxonomy, no hyperparameter
    search.

**Robot-time budget, bottom-up.** Budgets use the **total episode-cycle time** c (execution, reset
and annotation), measured in F0 and confirmed in the F1 proof. A 3-task seen-only run has
6 cells × 20 = 120 rollouts; a 2-task run has 3 × 20 = 60.

| Plan | Scope | Rollouts | Hours at c = 3 min | at c = 4 min |
|---|---|---:|---:|---:|
| **Full** | 3 tasks; method × 2 seeds; seq_ft and replay × 2 seeds; 3 single-task refs | 6 × 120 + 60 = 780 | 39 | 52 |
| **Fewer-seeds** | 3 tasks; method × 2 seeds; seq_ft and replay × 1 seed; 3 refs | 4 × 120 + 60 = 540 | 27 | 36 |
| **Two-task** | 2 tasks (one bimanual); method and seq_ft × 1 seed; 2 refs | 2 × 60 + 40 = 160 | 8 | 11 |

- Demo collection (F2) and engineering are budgeted separately.
- The plan is chosen at D-hw from the measured c and the confirmed robot access. It is used
  **only when the proof passed**.

**Precision.** At n = 20, a 50% success rate has a 95% CI of about ±22 pp. Hardware therefore
gives **transfer evidence and failure analysis, not fine method rankings**, and the thesis says so.

## Workstream G: writing (continuous)

| Weeks | Writing |
|---|---|
| W2–W4 | Setup and protocol (including the metric naming fix) |
| W4–W6 | Characterization |
| W8–W10 | GPM / adaptive / controls / reverse order |
| W10–W11 | SGP |
| W15–W17 | Baselines and main results |
| W18 | Hardware |
| W18–W20 | Introduction, related work, discussion |
| W21 | Buffer |

## Timeline

Priority in W1–W5, if overloaded: E1/E2 and replay/EWC/LoRA first.

**The baseline queue waits for Track 2's freeze only until Fri 30 Oct.** After that it starts
regardless, and Track 2 drops its held-out claim (Workstream D). Required work never waits on
optional work. Only the held-out test analysis may move to W6.

ConSFT is implemented in W5–W6 and runs in W7–W9.

| Week (Mon) | GPU queue | Build / analyze | Decision |
|---|---|---|---|
| W0 (Oct 2) | T5 sweep finishes | T5 record; close SGP | |
| W1 (Oct 5) | reverse seed 0; Gate 0 T6–T8 | A1–A6; replay; EWC; **F0** | **D1:** supervisor: accept the minimum thesis; the form of the final statistical claim; hardware scope and proof-gate thresholds |
| W2 (Oct 12) | reverse seeds 1–2; e99 × 3 | LoRA; literature and novelty delta; Track 2 manifest; **F0** | |
| W3 (Oct 19) | EWC/LoRA brackets; E1 | Track 2 measures; projector code | |
| W4 (Oct 26) | E2 pilots; then (after the freeze) baselines seed 0 | **Track 2 freeze, Wed 28 Oct**; ConSFT notes | |
| W5 (Nov 2) | baselines seeds 1–2 | ConSFT; Track 2 held-out test | **D2** |
| W6 (Nov 9) | (angle B reruns) | method in `run_continual`; commit `main_eval.yaml`; F1 | |
| W7–W8 | E3 seed 0; ConSFT runs in the gaps | F1 proof gate; W8 repair window; writing | **D3**, **D-hw** |
| W9–W11 | E3 seeds 1–2; ConSFT seeds | F2; F3 | **D4:** hardware method |
| W12–W13 | holidays: buffer and reruns only | writing | |
| W14–W16 | hardware training | F4 | |
| W17 (Jan 25) | **experiments frozen** | tables and figures | **D5** |
| W18–W21 | none | writing; submit ~Mar 1 | |

**GPU budget** (single RTX 4090):

| Item | Hours |
|---|---:|
| reverse order | ~48 |
| e99 control | ~15 |
| brackets | ~10 |
| four baselines | ~85, Fisher estimation included |
| Gate 0 T6–T8 | ~3 |
| Track 2 extraction | ~8, ODE sampling included |
| E1/E2, U estimation and the T4 `free` arm included | ~10 |
| E3 (48 h per seed × 3) | ~144 |
| hardware training | ~20 |
| **total** | **≈ 340** |

That is about 13% of calendar time, or about 53% of a 40-hour working week, over 16 weeks.
Implementation, not GPU time, is the W1–W5 bottleneck.

## Verification

- **CPU tests** (`uv run pytest -m 'not sim and not gpu'`):
  - replay mixing and its T1 no-op;
  - the EWC Fisher on a quadratic toy and the fp32 penalty;
  - LoRA merge exactness;
  - ConSFT weighting (once specified);
  - the Hutchinson estimator;
  - the two-sided projector algebra and random-U rank matching;
  - resume identity (weights, memory, eval successes);
  - the ε schedule;
  - the bootstrap on known cases;
  - the lineage-cluster bootstrap;
  - the metric-name mapping.
- **GPU checks** before each queue: a 200-step smoke run into scratch, with the rule SHA unchanged.
- **Every run:** the T1 pairing or identity check where it applies; the sequence-report provenance
  checks; a `docs/runs/` record with verdict, per-seed tables and disclosures.

## Risks

- **Both the method and Track 2 are negative.** The minimum thesis is defined and confirmed at D1.
- **W1–W5 overload.** The priority order is above; ConSFT and Track 2's analysis are the first to
  slip.
- **Union saturation over 8 tasks.** E1 projects it. If it saturates, record it, and leave the
  exact `∪(U_t ⊗ M_t)` projection as a candidate for a further plan.
- **Crashes.** Stage-boundary resume and sequential queues. A UPS if possible.
- **Hardware.** F0 bounds the unknowns in W1–W2, and the F1 proof gate comes before full
  collection.
  - **A failed proof** goes to repair and a supervisor decision.
  - **A short robot budget** means choosing the Fewer-seeds or Two-task plan.
- **SGP pull-back.** SGP is closed; any follow-up needs an explicit decision against this plan.
