# Four-task run — `gpm_projected_adam`, `seq_hetero`, seed 0, ε = 0.95 (build step 8)

**Status:** the pre-registered plasticity criteria are **met on all four tasks** (T4 borderline).
Final retention is **met for T1 only**: T2 (Object) ends at 0%, T3 (Goal) at 56%. The SGP fallback
is **not triggered**. The pre-registered rules did not cover a retention failure; §14 sets the next
step: no-training diagnostics under a newly pre-registered decision rule.  
**Date (local, CEST / UTC+2):** Thu 24 Sep 2026, 12:41 → 17:38 (4 h 58 min). A first attempt was lost
to a machine crash on 23 Sep at 16:04, during stage 2 (§10).  
**Machine:** `gamma`, RTX 4090, AMP on (Gate 1 recipe)  
**Seed:** 0 (one seed only)  
**Code:** `0184293` (clean tree, recorded in `git_sha`)  
**Run:** `results/seq_hetero__gpm_projected_adam__seed0`. Reference: `results/seq_hetero__seq_ft__seed0`  
**Companion records:** Gate 1 (`2026-09-22_gate1.md`), Gate 2 (`2026-09-22_gate2.md`), Gate 3
(`2026-09-22_gate3.md`), pilot (`2026-09-23_gpm_pilot.md`)

This note accompanies `results/gpm_seq/report.json`, written by `scripts/sequence_report.py`. Cite
the JSON for numbers. Every number here is in it; where the key is not obvious, it is named. The
first draft of this note computed several of them with one-off scripts; they now come from the
report (§12, item 8).

---

## 1. What the run answers

The pilot showed that T1→T2 works against a fixed T1 memory. This run adds GPM's incremental memory
(paper Eq. 8–9, spec §7.2) and trains all four tasks. It was built to probe one risk: that the
accumulated memory exhausts the free subspace, so that T3/T4 plasticity collapses. That collapse is
the SGP-fallback trigger.

| | `gpm_projected_adam` | seq_ft (Gate 1 reference) |
|---|---|---|
| T1 | unconstrained, all 53 813 767 trainable parameters | the same; stage-0 weights are **bitwise identical** |
| T2–T4 | registry weights only (41 954 304). The gradient is projected before AdamW **and** AdamW's realized update after the step, both against the accumulated memory | all parameters, no projection |
| memory | after every task: capture at the post-training weights, then extend to ε = 0.95 of that task's input energy (Eq. 8–9), float64 | — |
| streams | seq_ft's own: data, `s` and `A_0` from `derive_seed(seq_ft run id, task_key, stage)`; rollouts in seq_ft's seed namespace | its own |

Rollout seeds are identical in every cell of both runs, so all differences below are **paired**.

**Pre-registered criteria.** These are in `configs/analysis/sequence_report.yaml`, committed in
`0184293` before the run. References come from Gate 1's diagonal (90 / 78 / 100 / 98%), minus 15 pp:

| Criterion | Rule | Thresholds T1…T4 |
|---|---|---|
| Plasticity OK for task j | `R[j][j] ≥ R_seqft[j][j] − 15 pp` | 75 / 63 / 85 / 83% |
| Final retention OK for task j < 3 | `R[3][j] ≥ R_seqft[j][j] − 15 pp` | 75 / 63 / 85% |
| SGP fallback triggered | plasticity fails on T3 **or** T4 | — |

Point estimates decide. A CI that straddles its threshold is flagged `borderline`.

---

## 2. Decision

**Overall.** This run falsified the capacity-collapse hypothesis it was built to test. Fixed
total-energy GPM strongly protected T1 but allocated substantially less coverage to the novel
input-energy components introduced by T2–T4. T2 and T3 were subsequently forgotten; the planned
diagnostics (§14) test whether residual under-protection explains those failures.

| Question | Answer |
|---|---|
| Pre-registered outcome | **Plasticity OK on all four tasks** (T4 borderline). **Final retention OK on T1, FAIL on T2 (0%) and T3 (56%).** SGP fallback **not triggered**. |
| Did memory exhaust the free subspace in T3/T4? This is the risk the step was built to probe. | **Capacity was not globally exhausted.** After T4 the median occupancy is 0.615 in the trunk and 0.058 in the decoder. Only `action_in` (`d_in = 7`) is full, and it has been full since T1. But `trunk.blocks.{1..7}.mlp.fc2` already sit at ρ = 0.75–0.82, so stronger protection could make capacity a real T3/T4 constraint (§5). |
| Did plasticity collapse? | **No.** 90 / 78 / 94 / 90% against thresholds of 75 / 63 / 85 / 83%. A small cost may be emerging: −6 and −8 pp against seq_ft on T3 and T4, with CIs reaching 0 (§8). |
| Is T1 retained? | **Yes.** 88% after three more tasks (90% at stage 0). That is +88 pp over seq_ft. |
| Are T2 and T3 retained? | **No.** Object fell from 78% to 2% as soon as Goal was trained, then to 0%, the same end state as seq_ft. Goal fell from 94% to 56% while LIBERO-10 was trained. |
| Is the forgetting a pipeline defect? | **No evidence of one** (§7). Every realized update met the orthogonality bound, at worst 2.5% of it. All 565 state-dict tensors outside the allowlist are bit-identical from stage 0 to stage 3. Normalization stats were frozen at T1. Each task's projector was built from the accumulated memory. The forgetting enters through the projected layers, within GPM's known approximations. |
| Why is T1 protected but not T2/T3? | **Not established.** Measured (§6): ε is applied to a task's *total* input energy. Later tasks share 90–94% of theirs with memory, so only 22–49% of what is **new** in each later task was protected, against 95% for T1. Whether this under-protection explains the forgetting remains to be tested. The diagnostics localize the likely mechanism; the adaptive-GPM intervention is the causal test. |
| What does the pre-registered rule say to do next? | **Nothing.** It covers plasticity collapse (→ SGP). SGP relaxes protection, the opposite of what failed here. §14 sets the next step: no-training diagnostics under a newly pre-registered rule. |

---

## 3. Headline results

Success rate (%) with bootstrap 95% CIs over 50 rollouts. Row: after training on that task.
Column: evaluated task.

**`gpm_projected_adam`**

| after \ on | Spatial | Object | Goal | LIBERO-10 |
|---|---:|---:|---:|---:|
| Spatial | **90.0** [80, 98] | 0.0 | 0.0 | 0.0 |
| Object | 94.0 [86, 100] | **78.0** [66, 88] | 0.0 | 0.0 |
| Goal | 90.0 [80, 98] | 2.0 [0, 6] | **94.0** [86, 100] | 0.0 |
| LIBERO-10 | **88.0** [78, 96] | **0.0** [0, 0] | **56.0** [42, 70] | **90.0** [82, 98] |

**seq_ft** (Gate 1): diagonal 90.0 / 78.0 / 100.0 / 98.0. **Every** off-diagonal cell is 0.0 [0, 0].

Paired differences, `gpm_projected_adam` − seq_ft:

| | Spatial | Object | Goal | LIBERO-10 |
|---|---:|---:|---:|---:|
| plasticity `R[j][j]` | +0.0 [0, 0] | +0.0 [−14, +14] | −6.0 [−14, 0] | −8.0 [−18, 0] |
| final row `R[3][j]` | **+88.0 [+78, +96]** | +0.0 [0, 0] | **+56.0 [+42, +70]** | −8.0 [−18, 0] |

Summary metrics. These are point values: `summarize` computes no CIs (§11).

| | `gpm_projected_adam` | seq_ft |
|---|---:|---:|
| F_1 (final average success) | **58.5%** | 24.5% |
| NBT, mean of `R[j][j] − R[3][j]` over j = 0..2 | **39.3 pp** | 89.3 pp |
| per-task forgetting, Spatial / Object / Goal | 2 / **78** / 38 pp | 90 / 78 / 100 pp |
| AUC (per-stage average: 90 → 86 → 62 → 58.5) | **74.1%** | 46.7% |
| FWT | −97.3 pp | −97.3 pp (Gate 1, derived) |

FWT carries no information here. Every zero-shot cell `R[j−1][j]` is 0 in both runs, so FWT
reduces to minus the mean Gate 0 baseline (92 / 100 / 100%).

---

## 4. Where the forgetting happens (episode level)

The same 50 episode seeds are used at every stage, so each transition is paired (report
`episode_transitions`, which raises if seeds differ between stages):

| Task | Transition | both succeed | lost | gained | median success horizon |
|---|---|---:|---:|---:|---|
| Spatial | stage 0 → 1 | 42 | 3 | 5 | 103 → 108 |
| | stage 1 → 2 | 43 | 4 | 2 | 108 → 102 |
| | stage 2 → 3 | 40 | 5 | 4 | 102 → 105 |
| | **stage 0 → 3** | **41** | **4** | **3** | 103 → 105 |
| Object | **stage 1 → 2** | **1** | **38** | **0** | 136 → 312 (the one survivor) |
| | stage 2 → 3 | 0 | 1 | 0 | — |
| Goal | **stage 2 → 3** | **26** | **21** | **2** | 125 → 152.5 |

Every failure, at every stage, is a 600-step timeout.

Three distinct patterns:
- **Spatial is perturbed, not eroded.** Each stage churns 3–5 episodes in both directions and the
  horizon does not drift. This matches the pilot's §6 reading, now over three later tasks.
- **Object is erased in one stage.** Goal training removes 38 of 39 successes. The survivor takes
  2.3× the median horizon.
- **Goal is degraded.** 21 episodes are lost, and the surviving successes are slower: +19 steps
  median and +52 mean on the 26 shared successes.

---

## 5. Capacity: did memory exhaust the free subspace?

**Not globally.** Occupancy `ρ_l = k_l / d_l` is dimension-based, never decreases, and is checked
(§7). Values are the median over the group's layers, with [min, max]:

| Group (layers) | after T1 | after T2 | after T3 | after T4 |
|---|---:|---:|---:|---:|
| trunk_attn (32) | 0.428 [0.32, 0.49] | 0.506 [0.35, 0.55] | 0.555 [0.38, 0.60] | 0.598 [0.40, 0.64] |
| trunk_mlp (16) | 0.483 [0.35, 0.67] | 0.548 [0.40, 0.74] | 0.596 [0.45, 0.79] | 0.645 [0.48, 0.82] |
| trunk_input (`state_projection`, d = 8) | 0.625 | 0.750 | 0.750 | 0.750 |
| decoder_self_attn (16) | 0.040 [0.02, 0.06] | 0.043 | 0.044 | 0.050 [0.02, 0.07] |
| decoder_cross_attn (16) | 0.289 [0.02, 0.31] | 0.444 [0.02, 0.48] | 0.486 [0.02, 0.53] | 0.562 [0.02, 0.62] |
| decoder_mlp (8) | 0.029 [0.01, 0.06] | 0.032 [0.01, 0.08] | 0.034 [0.01, 0.10] | 0.037 [0.01, 0.12] |
| decoder_output (`action_out`) | 0.078 | 0.084 | 0.092 | 0.107 |
| decoder_input (`action_in`, d = 7) | **1.000** | 1.000 | 1.000 | 1.000 |
| **trunk median ρ** (report) | **0.439** | **0.521** | **0.577** | **0.615** |
| **decoder median ρ** (report) | **0.044** | **0.049** | **0.051** | **0.058** |
| `k_added`, trunk / decoder | 18 278 / 2 367 | 2 621 / 1 006 | 2 050 / 349 | 1 519 / 618 |

Report `capacity[τ][group]` for the group rows and `capacity[τ][trunk|decoder]` for the half
medians and `k_added`; all come from `memory_history` in `memory_task3.pt`.

- **The most occupied layers** are `trunk.blocks.{1..7}.mlp.fc2`, at 0.75–0.82 after T4. For
  example, `blocks.5.mlp.fc2` holds 1 680 of 2 048 dimensions.
- **Memory growth slows with each task.** Trunk additions fall from 2 621 to 2 050 to 1 519
  dimensions: 14%, 11% and 8% of T1's 18 278.
- **The decoder barely grows**, except for cross-attention, which reads the trunk's tokens and fills
  up like the trunk does.
- **Capacity was not globally exhausted**, so the risk this step was built to probe did not
  materialize over four tasks. But `trunk.blocks.{1..7}.mlp.fc2` already sit at ρ = 0.75–0.82: a
  stronger protection rule could make capacity a real T3/T4 constraint.

---

## 6. Why later tasks are protected less: ε on total energy (hypothesis)

At the end of task τ, Eq. 9 measures how much of τ's input energy already lies in memory
(`proj_energy_fraction`). Eq. 8 then adds directions until the total reaches ε = 0.95. Each task
therefore leaves about 5% of its **total** energy unprotected: `captured_energy_fraction` in
`memory_history` is ≥ 0.95 in every layer. How that 5% compares with the part of the task that is
**new**, meaning not already in memory, differs sharply between T1 and later tasks (medians over
layers; report `capacity[τ][half].median_new_energy_protected`):

| Task | trunk: energy already in memory | trunk: share of the new energy protected | decoder: already in memory | decoder: share of the new energy protected |
|---|---:|---:|---:|---:|
| T1 Spatial | 0 | **0.95** | 0 | **0.95** |
| T2 Object | 0.903 | **0.49** | 0.939 | **0.24** |
| T3 Goal | 0.921 | **0.38** | 0.943 | **0.22** |
| T4 LIBERO-10 | 0.926 | **0.33** | 0.935 | **0.27** |

The protected share of new energy is `(captured − proj) / (1 − proj)`, computed per layer. Layers
whose new energy is numerically zero (`proj ≥ 1 − 1e-12`) have no share and are left out of the
median.

**Reading.**
- For T1, the unprotected 5% is 5% of everything the task has.
- For Object, the same 5% is **about half of its trunk-specific energy and three quarters of its
  decoder-specific energy**.
- That residual lies in the complement of memory, **exactly where every later task's update is
  allowed to act.**
- *If* what distinguishes Object's behaviour lives mainly in its new directions, and not in the ones
  it shares with Spatial, its protection is much weaker than ε = 0.95 suggests. T1's is not.

This fits the pattern in §4: T1 is intact, while T2 and T3 are damaged by the very next task.
**Whether this under-protection explains the forgetting remains to be tested.** The diagnostics
localize the likely mechanism; the adaptive-GPM intervention is the causal test. Three alternatives
are not excluded:
1. **Stale bases (compounding drift).** Each memory describes a task's inputs at the weights where
   it was captured. Changes in earlier layers shift the inputs that later layers receive, and this
   compounds. It would hit T1 too, but T1 has had more of its energy protected from the start.
2. **Task fragility.** Object was the weakest task (78% even unconstrained). Closed-loop success is
   a cliff, so a modest loss increase can erase it. A complete 39 → 1 collapse is hard to explain
   this way alone.
3. **Task order or content.** This run cannot separate "second task" from "Object". That would need
   `seq_hetero_reverse`.

§14 gives the pre-registered diagnostics that discriminate between them.

---

## 7. Mechanism and numerical guarantees

The median over layers of the raw-gradient overlap `c_l` with memory, and of AdamW's step before the
update projection, logged every 100 steps (report `dynamics[τ].gradient_c_per_step` and
`update_c_per_step`). Each cell is trunk / decoder:

| Stage (memory) | step 0 | 100 | 1 000 | 5 000 | 15 000 | 25 000 | 29 900 |
|---|---|---|---|---|---|---|---|
| T2, raw gradient (T1 memory) | 0.947 / 0.898 | 0.927 / 0.891 | 0.923 / 0.881 | 0.937 / 0.913 | 0.946 / 0.968 | 0.937 / 0.957 | 0.932 / 0.912 |
| T3, raw gradient (T1 + T2) | 0.954 / 0.925 | 0.937 / 0.921 | 0.938 / 0.931 | 0.931 / 0.935 | 0.936 / 0.926 | 0.936 / 0.932 | 0.942 / 0.951 |
| T4, raw gradient (T1 + T2 + T3) | 0.966 / 0.942 | 0.944 / 0.919 | 0.943 / 0.894 | 0.946 / 0.935 | 0.945 / 0.935 | 0.944 / 0.938 | 0.944 / 0.936 |
| T2, AdamW step | — | 0.192 / 0.169 | 0.154 / 0.131 | 0.093 / 0.083 | 0.075 / 0.084 | 0.070 / 0.088 | 0.461 / 0.163 |
| T3, AdamW step | — | 0.175 / 0.185 | 0.155 / 0.156 | 0.096 / 0.105 | 0.076 / 0.096 | 0.074 / 0.097 | 0.525 / 0.183 |
| T4, AdamW step | — | 0.190 / 0.164 | 0.165 / 0.152 | 0.094 / 0.100 | 0.077 / 0.094 | 0.072 / 0.090 | 0.532 / 0.166 |

1. **The raw overlap barely grows with memory.** Trunk occupancy rises from 0.44 to 0.58, yet the
   trunk overlap stays at 0.93–0.95. The component left for learning is roughly as large in every
   stage, consistent with plasticity holding.
2. **The T2 row reproduces the pilot's**: trunk 0.947 at step 0 in both. The decoder values differ
   slightly because the decoder memory differs (§9).
3. **AdamW's step leaks as in the pilot**, 16–19% early and 7–10% later. The post-step projection
   removes it. The rise at step 29 900 (lr ≈ 2.7e-9) repeats in every stage, which is consistent
   with the pilot's untested weight-decay explanation.

**Guarantees checked** (report `provenance_checks` and `dynamics`):
- The residual `‖D M‖ ≤ 1e-6‖W‖ + 1e-3‖D‖` held on every step for every layer. Worst per stage:
  2.44%, 2.49% and 2.45% of the bound. The median over layers was 1.5%, 1.7% and 1.9%
  (`median_residual_over_bound`).
- Occupancy never decreases.
- Frozen tensors (`frozen_from_stage1`, which now covers the whole state dict): all 565 state-dict
  tensors outside the allowlist are bit-identical to stage 0 in every later checkpoint. That covers
  T1-trained parameters, never-trained backbones and persistent buffers.
- 90 of the 91 allowlisted layers moved at every stage; `action_in`, whose projector is exactly 0,
  did not (`moved_allowlisted`, `unmoved_allowlisted`).
- Normalization stats were fitted on T1 only and asserted against their fingerprint before every
  stage.
- Each task's projector was built from the memory accumulated so far (`memory_chained`, PASS):
  - every task's `k_before` equals the previous task's `k_after`, and each stored basis has its
    recorded rank;
  - the previous memory lies inside the next: `‖M_{τ−1}ᵀ M_τ‖²_F / k_{τ−1}` = 1 − 7e-16 at every
    step. Equal ranks alone would not show this;
  - it is also the exact prefix of the next (`prefix_identical`). In the code, `_update_memory`
    writes the extended basis back into `_memory`, which `on_task_start` then uses
    (`flowcl/methods/gpm.py:260`, `:325`).

---

## 8. Plasticity cost

| Task | `R[j][j]` gpm − seq_ft | last-50 training loss, gpm vs seq_ft |
|---|---:|---|
| T2 Object | +0.0 pp [−14, +14] | 0.00858 vs 0.00759 (+13%) |
| T3 Goal | −6.0 pp [−14, 0] | 0.00796 vs 0.00706 (+13%) |
| T4 LIBERO-10 | −8.0 pp [−18, 0] | 0.00867 vs 0.00761 (+14%) |

- **Training loss is about 13% higher in every projected stage.** That is roughly the pilot's gap,
  and it does not grow with memory.
- **Rollouts show a small cost that may be growing**: 0, then −6, then −8 pp. Each CI reaches 0, and
  T4's 90% is borderline against its 83% threshold.

The trend is worth watching over seeds. It is not a finding.

---

## 9. Comparison with the pilot: a warning about single-run variance

Stage 1 of this run repeats the pilot's T1→T2 experiment:
- **Identical:** stage-0 weights (bitwise), the T2 stream, the recipe, the frozen set and the
  rollout seeds.
- **Different:** only the T1 memory. The pilot used Gate 2's stored bases; this run captured afresh
  under the `gpm_memory::` seed namespace.

| Stage 1 | pilot | this run | paired difference |
|---|---:|---:|---:|
| Spatial | 80.0 [68, 90] | 94.0 [86, 100] | **+14 pp [+2, +26]** |
| Object | 88.0 [78, 96] | 78.0 [66, 88] | −10 pp [−24, +4] |

The two T1 memories were compared directly at ε = 0.95 (report
`pilot_comparison.t1_memory_overlap`):
- **Notation.** `a` is the pilot's basis: Gate 2's `U`, cut to its ε = 0.95 prefix exactly as the
  pilot used it (Gate 2 stores `U` up to ε = 0.99). `b` is this run's `memory_task0`. With
  `s = ‖M_aᵀ M_b‖²_F`:
  - `s/k_a = 1` means the pilot's basis lies inside this run's;
  - `s/k_b = 1` means the reverse.
- **How the capture works.** It iterates the full T1 dataset in a fixed order, and its token
  subsample has a fixed seed. The capture seed draws only the flow time `s` and the noise that forms
  the decoder's noisy-action input.
- **The trunk memories are identical.** `s/k_a` = 1.000 and `s/k_b` = 1.000 at the median (min
  0.999); ranks differ by at most one dimension.
- **The decoder memories differ slightly**, in both directions:

  | Decoder group | `s/k_a` median (min) | `s/k_b` median (min) | `k_b − k_a` |
  |---|---:|---:|---|
  | self-attention | 0.998 (0.888) | 0.988 (0.940) | −1 to +1 |
  | MLP | 0.997 (0.975) | 0.989 (0.942) | 0 to +4 |
  | cross-attention | 1.000 (0.983) | 1.000 (0.900) | 0 to +1 |
  | `action_out` | 0.984 | 0.910 | +3 (40 vs 37) |

  For `action_out`, the pilot's basis lies almost entirely inside this run's, which adds three
  directions. The self-attention bases are slightly rotated relative to each other near the ε cutoff.

**Consequence.** A perturbation confined to the low-energy edge of the decoder memory moved stage-1
outcomes by 10–14 pp, and one of the two differences lies outside its CI. Rollout CIs cover
evaluation noise only; training-path variance is at least as large. Two readings follow:
- The pilot's borderline Spatial 80% was not a stable number.
- No single cell of this run is either, **except** the qualitative pattern: T1 held while T2 was
  erased, and a 78 → 0 collapse is not a 10 pp fluctuation.

---

## 10. Reproducibility and provenance

**The crash.** The first attempt started on 23 Sep from the same commit (`git_sha` `01842933…`). The machine hard-crashed
at 16:04 during stage 2: the journal ends with no shutdown record and no kernel error. By then
stages 0–1 were complete, hashed and verified. The partial run is kept, not deleted, as
`results/seq_hetero__gpm_projected_adam__seed0_crashed_20260923/`. The rerun started from scratch
on 24 Sep.

**Bitwise reproduction.** The rerun's stages 0–1 equal the crashed run's:
- identical SHA-256 for `checkpoints/stage{0,1}.pt`, `method/memory_task{0,1}.pt`,
  `method/gpm_logs_task{0,1}.json` and `t1_pairing.json`;
- `eval/stage{0,1}.json` differ only in `wall_clock_s`. Successes, step counts and seeds are
  identical.

Training, memory capture and rollouts are therefore deterministic on this machine, so the crash cost
time and nothing else.

| Provenance check (report) | Result |
|---|---|
| `clean_git_sha` | PASS: `01842933…` |
| `seed_namespace` | PASS: `seq_hetero__seq_ft__seed0` |
| `t1_pairing` | PASS: relative weight difference **0.0** overall and in every group; final and last-50 losses equal to seq_ft's |
| `occupancy_non_decreasing` | PASS |
| `memory_chained` | PASS: each memory contains the previous one (containment 1 − 7e-16) as its exact prefix, and the ranks chain |
| `residuals_within_bound` | PASS: worst 0.0249 of the bound |
| `artifact_hashes_match` | PASS: every `method_artifacts` SHA-256 in every checkpoint matches the file on disk |
| `frozen_from_stage1` | PASS: all 565 state-dict tensors outside the allowlist equal stage 0; 90 of 91 allowlisted layers moved per stage |

---

## 11. Cost

| | `gpm_projected_adam` | seq_ft (Gate 1) |
|---|---:|---:|
| total wall clock | 4 h 58 min (17 865 s) | 5 h 34 min (20 014 s) |
| training per stage, including memory capture (s) | 2 455 / 2 624 / 2 615 / 2 694 | 2 544 / 2 558 / 2 553 / 2 553 |
| stored memory (§8.2, float32 equivalent) | 140.1 MB after T4 (T1 alone: 103.6 MB) | 0 |
| memory artifacts on disk (float64 plus spectra) | 208 / 239 / 263 / 281 MB | — |

- **Training time.** Stage 0 trains identically in both runs, and this run's stage 0 also includes
  the T1 memory capture. It was still 3.5% faster, which sets the run-to-run noise. Against that,
  stages 1–3 are 2–6% slower than seq_ft, capture included. The capture is not timed separately.
- **Total time.** It is shorter than seq_ft's because fewer rollouts time out at 600 steps.

---

## 12. Protocol deviations and caveats (§11)

1. **Crash and full rerun** (§10). No results from the crashed attempt are used.
2. **Not canonical GPM.** AdamW with gradient and update projection. Always named
   `gpm_projected_adam`.
3. **ε = 0.95 is fixed for every task and layer.** The paper's per-task increase of the threshold is
   not used. Given §6, this choice probably matters.
4. **Frozen protocol.** From T2 on, the 11 859 463 non-registry parameters are frozen, unlike in
   seq_ft. The pilot's `freeze_only` arm showed that freezing alone neither helps nor hurts on T1→T2.
   It is still a listed difference for the Stage A table.
5. **Summary metrics have no CIs.** F_1, NBT and AUC are point values, as in Gate 1. Spec §11 asks
   for a CI on every number, and a paired bootstrap over rollouts is not yet implemented.
6. **FWT is uninformative** on this curriculum (§3).
7. **The pre-registration had a gap.** It anticipated plasticity collapse, not retention failure
   with plasticity intact. For the next step, the gap is closed by a separate decision rule,
   committed before the diagnostics run (§14).
8. **One-off numbers were moved into the report.** The first draft of this note computed several
   numbers with one-off scripts. `sequence_report.py` now computes all of them, and the report was
   regenerated:
   - the episode transitions (§4);
   - the per-group capacity and new-energy shares (§5, §6);
   - the per-step `c_l` (§7);
   - the whole-state-dict frozen check and the memory chain (§7);
   - the T1 basis overlaps (§9).

   Two values changed:
   - T4's decoder new-energy median went from 0.26 to 0.27, because the one-off median did not
     exclude layers whose share is undefined.
   - The §9 overlaps are now the directional `s/k_a` and `s/k_b`, which replace an ad-hoc `max(k)`
     normalization.

---

## 13. What this run does *not* show

1. **Anything beyond one seed.** §9 shows single cells move 10–14 pp from a decoder-memory
   perturbation alone.
2. **Why T2 and T3 are forgotten.** §6 measures an under-protection, but whether it explains the
   forgetting is open, with three live alternatives.
3. **Whether a stricter memory criterion fixes retention** without costing plasticity or capacity.
4. **Loss-level forgetting.** This run has no fixed-batch probe-loss matrix; the pilot had one. The
   diagnostics add it (§14).
5. **Whether the pattern depends on task order** (no `seq_hetero_reverse` run).
6. **Anything about `gpm_grad_only`, SGP or other ε values.**

---

## 14. Consequences and next step (decided)

**Where the pre-registered plan stands.**
- The build-step-8 plan said: more seeds if the criteria pass, the SGP fallback if they fail. The
  fallback was defined as a response to T3/T4 **plasticity** collapse, and it did not trigger.
- The retention criteria failed, so "more seeds" is not licensed either. The outcome falls outside
  the pre-registered tree.
- **SGP is not indicated by this failure mode.** It scales protection down by importance, trading
  stability for plasticity. Here plasticity held and stability failed.

**Next step: no-training diagnostics.**
- **How they run.** `scripts/forgetting_diagnostics.py`, on the existing checkpoints. The decision
  rule is pre-registered in `configs/analysis/forgetting_diagnostics.yaml` and committed before the
  run; the script refuses a dirty tree.
- **What they can show.** They localize the likely mechanism. They cannot establish causality.

1. **A fixed-batch loss matrix** `L[i][j]` for all four checkpoints of both runs, using the pilot's
   exact probe. seq_ft's stage-0/1 cells must reproduce the pilot's recorded values; the script
   fails loudly otherwise.
2. **Per-layer activation interference** `r_l = ‖ΔW X‖ / ‖W X‖`, computed exactly from input Grams.
   The primary comparison takes the update that trained Goal (stage 1 → 2) and applies it to:
   - Object's inputs (the target, forgotten);
   - Spatial's inputs (the control, which survived the same update).

   Each is measured two ways:
   - **direct**, on stage-1 activations: the unprotected residual;
   - **after drift**, on stage-2 activations: the same update on drifted inputs.

   The capture seeds do not depend on the stage, so both measurements see identical observations,
   `s` and noise. A secondary comparison repeats this for Goal across LIBERO-10's update
   (stage 2 → 3).

**Pre-registered decision rule** (as committed in the config):

The quantities:
- the loss ratios `R_O = L[2][Object] / L[1][Object]` and `R_S = L[2][Spatial] / L[1][Spatial]`;
- per layer, `q_l = r_l(Object) / r_l(Spatial)`, with explicit zero and inf rules:
  - both zero, or both inf: the layer is excluded;
  - control zero or target inf: `q_l` is inf;
  - target zero or control inf: `q_l` is 0;
- `Q_direct` and `Q_drift`, the median of `q_l` per half on stage-1 and stage-2 activations.
  "Large" means `Q ≥ 2` in either half.

| Loss state | Condition (inclusive) |
|---|---|
| approximately stable | `R_O < 2` |
| non-selective worsening | `R_O ≥ 2` and `R_O / R_S < 2` |
| selective worsening | `R_O ≥ 2` and `R_O / R_S ≥ 2` |

Exactly one case applies:

| Case | Condition | Interpretation → next step |
|---|---|---|
| C | approximately stable, whatever Q is | **Rollout / probe-loss mismatch.** Closed-loop fragility is one candidate, not a conclusion → inspect rollouts and task order. Stronger projection is not yet justified. |
| E | non-selective worsening | **Non-selective degradation.** The control comparison is inconclusive. |
| A | selective, `Q_direct` large | **Direct unprotected interference** → test adaptive GPM. |
| B | selective, `Q_direct` small, `Q_drift` large | **Compounded representation drift / stale bases.** |
| D | selective, both Q small | **Sparse-layer or downstream interference** that the medians miss → inspect the tail and the top layers. |

Low medians alone are never read as fragility. Always reported, but not decisive:
- the 75th and 90th percentiles and the maximum of `q_l`;
- the share of layers with `q_l ≥ 2`;
- the top five layers with their groups;
- the seq_ft anchor `r_gpm / r_seqft`;
- the energy outside memory before and after drift;
- the Goal replication, flagged if it disagrees with the primary case.

The thresholds are judgements, not calibrated bounds.

**If case A: the adaptive variant, which is the causal test.** This is a single variant, not an
arbitrary switch of every layer to ε = 0.99:

    c_target = max(0.95, p + 0.90 (1 − p)),    p = proj_energy_fraction

- **What it protects.** At least 90% of each task's new energy. At p ≈ 0.90 its effective total
  threshold is ≈ 0.99.
- **A built-in check.** At T1 (p = 0) the target is 0.95, so stage 0 and T1's memory must be bitwise
  identical to this run's.
- **What it would show.** Under-protection explains the forgetting only if this variant improves T2
  retention.
- **What it risks.** It costs capacity. With `mlp.fc2` layers already at 0.82, capacity could become
  a real T3/T4 constraint, so the capacity table is part of its readout.
- It gets its own plan and pre-registration.

**Seeds 1–2 wait.**
- A 78 → 0 collapse is not plausibly explained by the 10–14 pp run-to-run variance in §9.
- The protection rule is settled first; then the final configuration is repeated across seeds.
- That needs seq_ft seeds 1–2 for pairing (about 5.5 h each), plus about 5 h per method run.

The results will be recorded in `docs/runs/2026-09-2X_gpm_forgetting_diagnostics.md`.

---

## 15. Suggested thesis wording (draft)

> We trained `gpm_projected_adam` over the four-task `seq_hetero` curriculum with GPM's incremental
> memory (ε = 0.95 per task and layer; one seed). Plasticity was preserved: every task was learned to
> within 15 pp of unconstrained fine-tuning (90, 78, 94 and 90% against 90, 78, 100 and 98%).
> Retention was not uniform. The first task kept 88.0% [78.0, 96.0] after three further tasks, where
> fine-tuning kept 0%. The second task (Object) fell from 78% to 2% as soon as the third was trained,
> and the third (Goal) fell from 94% to 56.0% [42.0, 70.0] during the fourth. Capacity was not
> globally exhausted: after four tasks the median layer protected 62% of the trunk's input
> dimensions, although the most occupied MLP layers reached 82%. Fixed total-energy GPM strongly
> protected the first task but allocated substantially less coverage to the novel input-energy
> components introduced by the later tasks. Those tasks shared 90–94% of their input energy with
> earlier ones, so a criterion of 95% on total energy protected only 22–49% of the energy new to
> each of them, against 95% for the first. The second and third tasks were subsequently forgotten;
> the diagnostics and the intervention described next test whether residual under-protection
> explains those failures.

Do not write "GPM prevents forgetting", "GPM fails", or that under-protection *caused* the
forgetting. It retained the first task and lost the later ones, from one seed, and the mechanism is
not yet verified.

---

## 16. Artifact map

```
results/gpm_seq/report.json                   # canonical: criteria, both R matrices, paired cells,
                                              # metrics, capacity (per half and group, new-energy
                                              # shares), episode transitions, dynamics (per step),
                                              # pilot comparison (incl. T1 memory overlap),
                                              # provenance checks (incl. memory_chained)
results/seq_hetero__gpm_projected_adam__seed0/
    config.yaml  git_sha  libero_submodule_sha  requirements.txt  seed.json  stats.json
    result.json                               # R matrix, F_1 / NBT / AUC / FWT, per-stage losses and
                                              # timings, t1_pairing, both run ids
    t1_pairing.json
    checkpoints/stage{0..3}.pt                # extra: method_run_id, seed_namespace_run_id,
                                              # method_artifacts [{path, sha256}]
    method/memory_task{0..3}.pt               # accumulated memory after task τ (float64) plus
                                              # memory_history (k, ρ, proj/captured energy per layer)
    method/gpm_logs_task{0..3}.json           # c_l every 100 steps, residuals, displacement, memory info
    eval/stage{0..3}.json                     # 4 tasks × 50 rollouts, seeds, both run ids
results/seq_hetero__gpm_projected_adam__seed0_crashed_20260923/   # first attempt; stages 0–1 bitwise equal
results/logs/gpm_seq_hetero_seed0_20260924_1241.log
```

Inputs, read only:
- `results/seq_hetero__seq_ft__seed0/{config.yaml, result.json, eval/, checkpoints/stage0.pt}`
- `results/single__*__seed0/eval.json` (FWT)
- `results/gpm_pilot/pilot.json`
- `results/seq_hetero__seq_ft__seed0/bases/task0.pt` (Gate 2's T1 basis, §9)
