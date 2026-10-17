# E1-B, the angle-B capacity gate (adaptive GPM seed 0): `fail`, near full rank

**Status: a pre-registered measurement, no training, one seed (the development seed).** The
verdict is **`fail`**, with the label **`near_full_rank`**.

**What it measured.** The recovery study (`runs/2026-10-17_recovery_study.md`) found that closed-loop
harm concentrates in position offsets and in gripper flips while carrying. E1-B weighted the
executed action prefix's sensitivity by that measured harm, then reran E1's capacity question.

**The result.**
- **The decisive quantity is unchanged.** At the T4 pilot's start state, protecting
  `(U_1∪U_2∪U_3)_R ⊗ M` would still cover **0.861** of the trunk's weight directions,
  parameter-weighted. The threshold is ≤ 0.50. GPM covers 0.933.
  - E1, with whole-velocity sensitivity, gave **0.861** at the same point. The two differ in the
    fourth decimal (0.8611 against 0.8608).
- **The weighting does not matter.** Every variant lies between **0.850 and 0.870**:
  - R (injury-weighted): 0.861;
  - I (uniform): 0.870;
  - S (shuffled): 0.850;
  - P (phase-pooled): 0.864;
  - R0 (no gripper): 0.861;
  - Ri (independent-step): 0.868.
- **No label besides `near_full_rank` is set.** R's trunk ρ_out is 0.924.
- **Every check passed.** On all four tasks, the gradient-enabled integration reproduced
  `policy.sample` bitwise on every batch (524 of 524). Weights and gradients were unchanged, every
  Gram is finite, and every layer has enough rows (§2).

**In short:**
- **The trunk's output side.** It is near full rank under both output metrics tested so far:
  - E1's whole velocity: 7 dimensions × 16 steps, at a random flow time;
  - this injury-weighted, temporally coherent executed prefix: 7 directions, through the whole
    sampler.

  How many outputs are measured, and how they are weighted, moves the decisive fraction by at most
  2 pp. The reading, consistent with the expectation written in the plan but not established here:
  the occupancy comes from the variety of states and positions, not from the output target.
- **The decoder** becomes about twice as compact as under E1: two-sided 0.064 against 0.128, ρ_out
  0.18 against 0.33 after T3. But the decoder is not saturated (GPM 0.374), so this frees nothing
  where capacity is short.
- **The pre-written consequence applies.** The recovery line stops; this is a scheduling decision.
  D2 resolves to the **minimum thesis**.
- **The record's sentence, as pre-registered:** "the tested surrogate did not meet the
  preregistered 50% capacity budget under this capture and accumulation rule". It does not show
  that every recovery-derived Kronecker metric lacks useful capacity.
- **The recovery study's GO stands** as a measured finding.

**Date (local, CEST):** Sat 17 Oct 2026, 16:39:09 → 18:48:19.
- captures 16:39 → 18:14 (T1 18 min, T2 19 min, T3 22 min, T4 36 min);
- summary 18:14 → 18:48 (34 min).

**Code:** `cbc27ae` (`cbc27ae7…`), clean at queue start. The rule
`configs/analysis/twosided_angleb.yaml` (SHA-256 `17663228…`) was committed with the implementation
at 16:38:59, 10 s before the queue (Disclosure 1).

**Cite:** `results/twosided_angleb/summary.json` (SHA-256 `a97ddcd5…`). It holds:
- the verdict and labels, and the validity of every capture;
- the weight tables;
- all tables (variant × rule × ε_out × stage × half/group);
- E1's numbers (pinned), Ri's stability, the alignment check and the forecast;
- the SHAs of the saved bases.

---

## 1. Design (as pre-registered; the rule header is the specification)

**The question.** Does the trunk's output side become compact under a **continuous executed-prefix
surrogate weighted by observed injury**? E1 failed on whole-velocity sensitivity; this is D2's
angle-B row ("`u` restricted to those outputs, rerun E1"). The surrogate ignores clipping and the
gripper's sign processing. It is a defensible hypothesis, not the controller.

**The source:** E1's, unchanged and SHA-pinned. That is adaptive GPM seed 0, `stage0–3.pt` and
`memory_task0–3.pt`.
- Capture i uses `stage{i}.pt` on task T(i+1)'s whole demo dataset.
- E1's text-cache state is reused: T1 under AMP, the later tasks in fp32.
- M is GPM's stored `memory_task{i}`, which E1 replayed bitwise.

**The output: the integrated executed prefix.**
- `integrate` is the arithmetic of `FlowHead.sample` with autograd on. The trunk encodes once, then
  10 Euler steps run from noise drawn per batch on the CPU. It runs fp32, with autocast off, in
  `eval()`.
- The prefix is chunk steps 0–7, the executed ones.

**The exact coherent capture.**
- **One VJP per action dimension d,** of `T_d = Σ_{k<8} a_{k,d} / √8`. This is a constant offset across
  the prefix, the temporal direction the recovery study perturbed.
- **Each row is scaled by** `c_d = std_d / σ_{i,d}`:
  - `std_d` is the checkpoint's frozen normalization;
  - `σ_{i,d}` is the recovery study's demo SD, so harm is measured per demo SD, as perturbed.
- **Rows are accumulated** in float64 into 18 Grams per layer, one per (phase bin b × group g), where
  φ = t / L_episode of the demo frame.
- **Every coherent variant** is `G_v = Σ w_v[b, g] G_{b,g}`. It is exact over output directions,
  conditional on the captured samples and rows.

**The variants:**

| Variant | Weights `w_v[b, g]` | Role |
|---|---|---|
| **R (decisive)** | The recovery study's 3σ injury rate in (task, bin, group), over effective, at-risk perturbations | Injury-weighted surrogate |
| I | 1 | `uniform_integrated_suffices` if ≤ 0.50 |
| S | R's 18 values permuted per task (seeded) | `concentration_suffices` if ≤ 0.50 |
| P | Per-group task-level rates, phase-independent | `phase_mapping_sensitive` if its verdict differs from R's |
| R0 | R with the gripper weight 0 | Reported |
| Ri | Hutchinson, 2 probes (A, B), independent per step, scaled by `√w_R · c` | Reported; `estimator_sensitive` if A and B straddle |

**Rows.**
- A decoder layer runs once per Euler step, so every run's positions are candidate rows.
- Per sample and layer, up to 16 rows are drawn among eligible (call, position) pairs. The trunk
  uses E1's rule.
- **Eligible** means that the gradient of a generic random combination of the 8 × 7 prefix outputs
  is nonzero (Disclosure 3).
- The rows are drawn once per batch and serve every VJP.

**The U rule and the decisive quantity:** E1's own.
- `U` is accumulated task by task with `extend_basis` over the per-task `G_v`. The rule is adaptive,
  ε_out 0.95, f 0.9.
- **The decisive quantity** is the parameter-weighted trunk two-sided fraction of
  `(U_1∪U_2∪U_3)_R ⊗ memory_task2`, the start of the T4 pilot.
- **The verdict, first match:** `invalid`, `incomplete`, `pass` (≤ 0.50), then `fail`.

## 2. Validity (all passed)

| Check | T1 | T2 | T3 | T4 |
|---|---|---|---|---|
| Integration identity: grad-enabled integration vs `policy.sample`, **bitwise** | 92/92 | 114/114 | 110/110 | 208/208 |
| Worst absolute difference | 0.0 | 0.0 | 0.0 | 0.0 |
| Rows per output dimension, worst layer (≥ 10) | 11.5 | 14.2 | 13.7 | 26.0 |
| Grams finite; weights and `.grad` unchanged | yes | yes | yes | yes |
| Text cache | AMP | fp32 | fp32 | fp32 |
| Dataset samples / batches (64) | 5,882 / 92 | 7,281 / 114 | 7,027 / 110 | 13,298 / 208 |
| Peak GPU memory | 11.8 GB | 11.8 GB | 11.8 GB | 11.8 GB |

- **The worst sample ratio** is `trunk.state_projection`, which has one row per sample, as in E1.
- **Every capture's `git_sha`** is `cbc27ae`, clean. The queue log shows 0 changes at start.
- **The pinned inputs** were checked on every load:
  - E1's source, its summary (`03d4359d…`);
  - the recovery rule (`88a8a72b…`) and analysis (`33ac9f44…`);
  - each task's recovery work files, bound to that rule.
- **No weight cell needed the fallback.** None of the 72 (task × bin × group) cells was empty.

## 3. Results

**The decisive chain** (R, adaptive, ε_out = 0.95). Parameter-weighted fractions after each task;
E1's trunk value is in brackets:

| After | Trunk two-sided [E1] | Trunk GPM | Trunk ρ_out [E1] | Decoder two-sided [E1] | Decoder GPM | All two-sided | All GPM |
|---|---:|---:|---:|---:|---:|---:|---:|
| T1 | 0.250 [0.234] | 0.477 | 0.538 [0.496] | 0.003 [0.006] | 0.077 | 0.151 | 0.317 |
| T2 | 0.629 [0.612] | 0.781 | 0.809 [0.785] | 0.025 [0.046] | 0.227 | 0.387 | 0.559 |
| **T3 (decisive)** | **0.861** [0.861] | **0.933** | **0.924** [0.922] | 0.064 [0.128] | 0.374 | 0.542 | 0.709 |
| T4 | 0.944 [0.954] | 0.980 | 0.964 [0.974] | 0.127 [0.245] | 0.528 | 0.617 | 0.799 |

- **The trunk.** The narrower target gives a slightly *higher* trunk occupancy at T1 and T2, and the
  same at T3. Examples of `k_out`, R [E1], over T1–T4:
  - `trunk.blocks.0.mlp.fc1`: 1529 → 1901 → 2022 → 2044 of 2048 [1278 → 1773 → 1982 → 2037];
  - `trunk.blocks.7.mlp.fc1`: 963 → 1596 → 1902 → 2007 [969 → 1624 → 1919 → 2023];
  - `trunk.blocks.0.attn.q_proj`: 245 → 383 → 473 → 504 of 512 [200 → 327 → 438 → 493].
- **The decoder** is about half of E1's at every stage.

**Every variant, trunk two-sided / ρ_out** (adaptive, ε_out 0.95):

| Variant | T1 | T2 | **T3 (decisive point)** | T4 |
|---|---|---|---|---|
| **R** | 0.250 / 0.538 | 0.629 / 0.809 | **0.861 / 0.924** | 0.944 / 0.964 |
| I | 0.257 / 0.552 | 0.641 / 0.823 | **0.870** / 0.933 | 0.934 / 0.953 |
| S | 0.245 / 0.526 | 0.622 / 0.799 | **0.850** / 0.912 | 0.919 / 0.938 |
| P | 0.250 / 0.538 | 0.632 / 0.812 | **0.864** / 0.926 | 0.945 / 0.965 |
| R0 | 0.250 / 0.538 | 0.629 / 0.808 | **0.861** / 0.923 | 0.947 / 0.967 |
| Ri (A + B) | 0.256 / 0.549 | 0.638 / 0.819 | **0.868** / 0.930 | 0.945 / 0.964 |
| *E1 (whole velocity)* | *0.234 / 0.496* | *0.612 / 0.785* | ***0.861** / 0.922* | *0.954 / 0.974* |

- **Concentration does not compress.** R and S use the same 18 weights in different cells and differ
  by 1.1 pp. Uniform weighting (I) differs from R by 0.9 pp. The variants' decoder fractions at T3
  span 0.062–0.075.

**By group, at the decisive point** (R; E1 in brackets):

| Group | Two-sided | GPM | ρ_out |
|---|---:|---:|---:|
| trunk_attn (32) | 0.816 [0.816] | 0.906 | 0.900 [0.898] |
| trunk_mlp (16) | 0.884 [0.883] | 0.946 | 0.936 [0.935] |
| trunk_input (1) | 0.344 [0.432] | 1.000 | 0.344 [0.432] |
| decoder_input (1) | 0.146 [0.168] | 1.000 | 0.146 [0.168] |
| decoder_self_attn (16) | 0.048 [0.099] | 0.254 | 0.178 [0.373] |
| decoder_cross_attn (16) | 0.148 [0.269] | 0.691 | 0.317 [0.456] |
| decoder_mlp (8) | 0.030 [0.072] | 0.274 | 0.107 [0.242] |
| decoder_output (1) | 0.631 [0.631] | 0.631 | 1.000 (7 of 7) |

**Trunk two-sided fraction, rule × ε_out** (R; reported only; the decisive cell is in bold):

| Rule | ε_out | T1 | T2 | T3 | T4 |
|---|---|---:|---:|---:|---:|
| adaptive | 0.90 | 0.178 | 0.568 | 0.832 | 0.931 |
| adaptive | 0.95 | 0.250 | 0.629 | **0.861** | 0.944 |
| adaptive | 0.99 | 0.382 | 0.732 | 0.912 | 0.970 |
| fixed | 0.90 | 0.178 | 0.432 | 0.597 | 0.664 |
| fixed | 0.95 | 0.250 | 0.555 | 0.733 | 0.802 |
| fixed | 0.99 | 0.382 | 0.718 | 0.886 | 0.943 |

- **Every cell is above 0.5 at T3.** The most lenient, fixed ε_out 0.90, gives 0.597 (E1: 0.598).
- **The uniform variant I** gives 0.613 at that cell, so it fails there too.

**Stability of the independent-step comparison** (Ri; reported):

| Gram | T1 | T2 | T3 | T4 |
|---|---:|---:|---:|---:|
| A only | 0.248 | 0.628 | **0.862** | 0.947 |
| B only | 0.250 | 0.630 | **0.863** | 0.939 |
| A + B | 0.256 | 0.638 | **0.868** | 0.945 |

- **No `estimator_sensitive` label.** The span overlap `‖U_AᵀU_B‖²/max k` has a median of 0.97 in
  the trunk and 0.94 in the decoder.
- **The decisive coherent Grams are exact** and need no stability check.

**The weights** (R, the 3σ injury rate; bins 0–5 at 1/12 … 11/12 of the rollout):

| Task | Position | Rotation | Gripper |
|---|---|---|---|
| T1 Spatial | .51 .61 .26 .35 .50 .39 | .02 .06 .00 .03 .01 .00 | .04 .31 .10 .79 .48 .06 |
| T2 Object | .62 .67 .25 .52 .34 .19 | .14 .11 .01 .00 .02 .00 | .00 .33 .73 .91 .50 .00 |
| T3 Goal drawer | .70 .68 .67 .20 .06 .11 | .00 .03 .03 .04 .00 .00 | all 0 (never closes) |
| T4 libero_10 | .59 .10 .35 .46 .27 .09 | .05 .05 .14 .13 .03 .00 | .00 .02 .02 .05 .00 .11 |

The trials per cell are 80–96 for position and rotation and 40–70 for the gripper.

**The scale** `c_d = std_d / σ_{i,d}`:
- T1 is exactly 1, since the normalization is T1's.
- For later tasks it ranges from 0.31 to 1.52; it is smallest on Goal's rotation dimensions.

**The alignment check** (command-level evidence of phase correspondence, not evidence that an
object is carried). Per bin, the gripper-closed fraction in the demos against the recovery
rollouts' inferred commands:

| Task | Demos | Rollouts | Rollout / demo length |
|---|---|---|---:|
| T1 Spatial | 0 .09 .83 .97 .98 .52 | 0 0 1 1 1 1 | 0.90 |
| T2 Object | 0 .12 .91 1 .97 .49 | 0 .10 1 1 1 1 | 0.97 |
| T3 Goal drawer | 0 0 0 0 0 0 | 0 0 0 0 0 0 | 0.89 |
| T4 libero_10 | .00 .75 .24 .03 .63 .81 | 0 1 0 0 1 1 | 1.03 |

- **The phase structure corresponds.** It holds on every task, including libero_10's two grasps.
- **The one mismatch is bin 5 on T1 and T2.** There the demos have released in about half the
  frames, and the rollouts' queries (at 11/12) still hold.
- **The undetermined queries** (no effective 3σ gripper flip) are 10, 16, 0 and 36.
- **Either way, P moves the decisive fraction by only 0.3 pp.** The phase mapping cannot change the
  verdict.

**The forecast** (R; linear per layer and side, from T2 → T4; **a forecast, not a measurement**):

| Task | Trunk two-sided / GPM | Decoder two-sided / GPM | All two-sided / GPM |
|---|---|---|---|
| T5 | 0.985 / 0.997 | 0.199 / 0.658 | 0.670 / 0.861 |
| T6 | 0.993 / 1.000 | 0.277 / 0.767 | 0.707 / 0.907 |
| T7 | 0.998 / 1.000 | 0.347 / 0.842 | 0.738 / 0.937 |
| T8 | 1.000 / 1.000 | 0.401 / 0.880 | 0.760 / 0.952 |

## 4. The pre-registered judgement

| Quantity | Value | Rule | Result |
|---|---|---|---|
| R's trunk two-sided fraction, T3, adaptive 0.95 | **0.861** | ≤ 0.50 | ✗ |
| I's (uniform) | 0.870 | ≤ 0.50 → `uniform_integrated_suffices` | not set |
| S's (shuffled) | 0.850 | ≤ 0.50 → `concentration_suffices` | not set |
| P's verdict against R's | fail = fail | differ → `phase_mapping_sensitive` | not set |
| R's trunk ρ_out (parameter-weighted), same point | 0.924 | ≥ 0.90 → `near_full_rank` | **label set** |
| Ri's A-only / B-only on opposite sides of 0.50? | 0.862 / 0.863 | → `estimator_sensitive` (Ri only) | not set |

**Verdict: `fail`** (summary exit code 0: a valid finding), labels `[near_full_rank]`.

## 5. What this does and does not show

**It shows,** for adaptive GPM seed 0, under first-order output sensitivity of the integrated
executed prefix weighted by the recovery study's measured injury:
- the trunk's old-task output sensitivity still spans nearly all of its output directions;
- two quite different output metrics give the same decisive fraction to three decimals:
  - E1's velocity at a random flow time, all 16 steps, unweighted;
  - this study's coherent 8-step prefix through the whole sampler, injury-weighted.

  Five weightings move it by at most 2 pp. Which outputs are measured does not decide the trunk's
  output-side occupancy at this ε rule;
- protecting `(∪U) ⊗ (∪M)` instead of `I ⊗ (∪M)` frees 7.2 pp of the trunk at T4's start and
  3.6 pp at T5's start;
- the decoder's output side is about half as occupied as under E1. The decoder is not where
  capacity is short.

**It does not show:**
- **that every recovery-derived Kronecker metric lacks useful capacity.** A different accumulation
  might compress, for example per-phase bases, worst-case rather than energy rules, or another
  surrogate for the gripper's sign. None was tested, and none is planned.
- **anything about retention or learning.** Nothing was trained.
- **that the recovery study's structure is irrelevant to continual learning.** It is irrelevant
  only to this way of freeing weight-space capacity. The GO finding stands as measured.
- **other seeds.** Seed 0 only.

## 6. Disclosures

1. **Rule timing.**
   - The rule and the code were committed in `cbc27ae` at 16:38:59. The production queue started
     at 16:39:09 from a clean tree.
   - **Before the commit:** the 13 new tests passed, and so did E1's and the recovery study's tests.
   - **The full CPU suite ran at the commit concurrently with production,** at the lowest CPU
     priority. It finished at 18:28 with **1134 passed, exit 0**.
   - **This differs from E1's order,** where the suite finished before the queue. No production
     step depended on it.
2. **The smoke run.** It ran before the commit, from a dirty tree at `589dcda`, into the session's
   scratch directory.
   - **What it ran:** task 0, 3 batches, then the summary.
   - **The outcome was the one defined in advance:** `incomplete`, exit 4. The identity was bitwise
     on 3 of 3 batches.
   - **What was read from it:** only time and memory (10.4 s per batch, peak 11.8 GB, about
     10.6 GB of Grams per task), plus the alignment helpers' output, used to check that they ran.
     Its fractions were not read.
   - **The rule was the same.** The smoke's recorded configuration equals production's, field for
     field. The smoke's Grams were deleted.
3. **Row eligibility as implemented.** The plan said a row is eligible when its gradient is nonzero
   for **any of the batch's 9 VJPs**. The implementation uses the gradient of one generic random
   combination of all 56 prefix outputs, drawn from the row generator.
   - That support is almost surely the union of every prefix output's support, so it contains
     every coherent and Hutchinson VJP's support.
   - The rule header and the module docstring state it, before production.
   - A test with disjoint per-dimension supports checks that no direction's rows are missed.
   - Like E1's rule, it changes how many informative rows are kept, not what any Gram means.
4. **The weights transfer between policies and clocks.**
   - **Where they were measured:** on seq_ft's task-i policies, in rollout-normalized time.
   - **Where they are applied:** to GPM's stage-i policies, in demo-normalized time.
   - The phase-pooled variant P and the alignment check bear on this. Neither can change the
     verdict, since P is within 0.3 pp of R.
5. **The continuous surrogate.** The prefix ignores the controller's clipping and the gripper's
   sign processing. R0, without the gripper, moves the decisive fraction by 0.04 pp.
6. **Scaling.** `c_d` makes harm per demo SD, as the recovery study perturbed it. Its "rotation is
   nearly inert" finding is scaling-dependent (that record's caveat). Here this cannot matter: the
   uniform variant I and E1, which has no σ scaling, both fail.
7. **The comparator** is GPM's parameter-weighted fraction (0.933), as in E1.
8. **T4** (`U₄` with `memory_task3`) is reported only, as in E1.
9. **Development seed.** Seed 0 only.

## 7. Consequences (pre-written in the rule)

- **The recovery line stops.** This is a scheduling decision. E2-B, the T4/T5 matched-start pilots
  with the two-sided projector, is not planned.
- **D2 resolves to the minimum thesis.** Angle A is off (E1), and angle B is off (E1-B). E3 runs
  the capacity frontier and the reference methods without a new method. The formal D2 checkpoint
  (end of W5) records this.
- **For the thesis story** ("which directions actually need protection?"):
  - closed-loop vulnerability is structured by action component and phase (the recovery study);
  - that structure does not compress the trunk's weight-space output protection;
  - under whole-velocity and injury-weighted executed-prefix sensitivity alike, the trunk needs
    nearly all of its output directions.

  These are reported together, as two registered findings.
- **Disk.** `results/twosided_angleb/` holds 48 GB.
  - The 18-bin Grams (about 42 GB) were saved for E2-B's rank-budgeted controls, which will not
    run.
  - `summary.json` cites everything reported here. The Grams are needed only to recompute it.
  - Deleting them is the user's decision.

## 8. Artifact map

```
results/twosided_angleb/summary.json                  # verdict, labels, validity per capture, weights,
                                                      # tables (variant x rule x eps_out x stage x
                                                      # half/group), E1's numbers, stability, alignment,
                                                      # forecast, bases SHAs
results/twosided_angleb/capture/task{0..3}/           # grams_coherent.pt (bins x groups x d x d per layer,
                                                      # float64), grams_hutch_{A,B}.pt, capture.json
                                                      # (seeds, cache state, weights, std, scale, demo
                                                      # lengths and gripper profile, rows, checks, timings)
results/twosided_angleb/bases/{R,I,S,P}_adaptive_e0.95_task{0..3}.pt   # the chained unions, (d_out, k_out)
results/logs/queue_20261017_163909_twosided_angleb/   # queue.log and one log per step
configs/analysis/twosided_angleb.yaml (cbc27ae; SHA-256 17663228...)
flowcl/analysis/integrated_sensitivity.py, flowcl/analysis/recovery.py (phase_group_weights,
gripper_closed_by_bin, demo_gripper_closed_by_bin), flowcl/experiments/twosided_angleb.py,
scripts/twosided_angleb.py, scripts/queue_twosided_angleb.sh, tests/test_twosided_angleb.py
```

**Inputs** (read only, SHA-256-pinned in the config):
- `results/seq_hetero__gpm_projected_adam_ne90__seed0/`: `stage0–3.pt`, `memory_task0–3.pt`,
  `result.json`;
- `results/twosided_gonogo/summary.json`;
- `configs/analysis/recovery_study.yaml`, `results/recovery_study/analysis/analysis.json` and
  `results/recovery_study/.work/`.
