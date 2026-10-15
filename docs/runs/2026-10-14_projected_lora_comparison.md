# Projected low-rank adaptation, T1 → T2 on seed 0: all three families keep Spatial at 0.90–0.98 under GPM's memory (plain LoRA: 0.52). The InfLoRA adaptation and its trained-A ablation match `gpm_projected_adam` on both tasks with 1–2M trainable parameters; random-init projected LoRA learns Object least

**Status: three brackets done, selected and verified** (Tue 13 Oct 01:52 – Wed 14 Oct 07:44).
- **Descriptive, one seed, one transition (Spatial → Object), 50 rollouts per cell.** The development
  seed also enters the final evaluation. Nothing here is a non-inferiority claim.
- **Claim scope (unchanged from the rules):** we claim no new projection mechanism. The contribution is
  a controlled comparison of fixed against trainable low-rank adaptation under one shared protection
  memory, with measured costs and diagnostics. Prior work (InfLoRA, Janus-LoRA, LoDA, KeepLoRA,
  OPLoRA, LoRA-Null, OrthoSkillVLA) is context.

**Provenance:**
- **Rules,** committed before any arm (`d213ddb`):
  - `configs/analysis/baseline_selection_projected_lora.yaml` (trained A);
  - `…_inflora.yaml` (fixed A);
  - `…_pca_trained.yaml` (the matched ablation).
- **Method:** `flowcl/methods/projected_lora.py`.
- **Queue:** `results/logs/queue_20261012_232356_tonight`.
  - Run at `99e8083`, with the full CPU suite green at that commit (1091 passed).
  - The arm rows record `6df9c65`, a docs-only commit made during the first arm.
- **Bundles** (`results/baseline_selection_<family>/lora/selection/selection.json`):

  | family | bundle SHA-256 | verified |
  |---|---|---|
  | `projected_lora` | `e60d637c…` | yes |
  | `inflora` | `dda181af…` | yes |
  | `pca_trained` | `88e735f2…` | yes |

- **Shared inputs, bound in every row:**
  - the start, seq_ft seed 0's `stage0.pt`, asserted bitwise equal to `gpm_projected_adam`'s;
  - that run's Task-1 memory `memory_task0.pt` (`93b69935…`, eps 0.95);
  - K_old (`results/baseline_selection_projected/kold_seq_ft_seed0_stage0.pt`, `fc795a1d…`).

**Two bugs found before any arm,** both in `projected_lora.py`, both fixed with tests:
- **The InfLoRA design capture ran after the base was frozen,** so its reachability probe had no
  gradient. The fake-world bracket test caught this.
- **In 4-task runs, GPM's task-end memory update ran on the still-frozen base.** The first smoke caught
  it; the queue was stopped and relaunched at the fix. The trainable flags are now restored after the
  merge. Bracket arms never reach this path, since they do not update the memory.

All three smokes then passed:
- the runner's stage 0 equals seq_ft's;
- the Task-1 memory equals GPM's on 91 of 91 layers;
- a resume at stage 3 reproduces the run (tensors, episodes, method state);
- the bracket path reproduces the runner's stage 1 and history.

## The selected arms

Paired differences against `gpm_projected_adam` stage 1 (same 50 initial states), 95% bootstrap CIs:

| | rank, lr | Spatial (T1, kept) | Object (T2, learned) | Spatial − GPM | Object − GPM | trainable params |
|---|---|---|---|---|---|---|
| `gpm_projected_adam` | — | 0.94 | 0.78 | | | 42.0M (dense registry weights) |
| LoRA v2 (selected) | 16, 3e-5 | 0.52 | 0.70 | −0.42 [−0.56, −0.28] | −0.08 [−0.24, 0.08] | 2.0M |
| **`projected_lora`** | 64, 3e-4 | 0.94 | 0.70 | 0.00 [−0.08, 0.08] | −0.08 [−0.24, 0.08] | 8.2M |
| **`inflora`** | 16, 1e-4 | 0.90 | 0.82 | −0.04 [−0.16, 0.08] | +0.04 [−0.12, 0.20] | 1.0M |
| **`pca_trained`** | 16, 1e-4 | 0.98 | 0.82 | +0.04 [−0.04, 0.12] | +0.04 [−0.12, 0.20] | 2.0M |

Selection details:
- **`projected_lora`:**
  - 6 of 13 arms are eligible (Object ≥ 0.68).
  - Spatial ties at 0.94 between (16, 1e-3), (64, 1e-4) and (64, 3e-4); the lower Spatial probe loss
    picks (64, 3e-4).
  - That cell lies on the rank edge, so the rank-256 extension was owed. Its Spatial is 0.92, so the
    grid's choice stands (flag `edge_extension`).
- **`inflora`:** 4 of 12 eligible; interior, no extension.
- **`pca_trained`:** 4 of 4 eligible; interior.

All three selections are "the best among tested settings" on seed 0. Every paired CI against GPM
includes 0. **Read against the rules' screening threshold (within 10 pp of `gpm_projected_adam`):**
- all three are within 10 pp on Spatial;
- `inflora` and `pca_trained` are within 10 pp on Object;
- `projected_lora` is at −0.08 on Object.

None of this establishes non-inferiority at 50 rollouts.

## Every arm

Spatial / Object success:

| family | rank | lr 3e-5 | lr 1e-4 | lr 3e-4 | lr 1e-3 |
|---|---|---|---|---|---|
| `projected_lora` | 4 | 0.96 / 0.52 | 0.84 / 0.76 | 0.92 / 0.56 | 0.86 / 0.62 |
| | 16 | 0.94 / 0.60 | 0.92 / 0.64 | 0.96 / 0.50 | 0.94 / 0.78 |
| | 64 | 0.96 / 0.56 | 0.94 / 0.74 | **0.94 / 0.70** | 0.90 / 0.88 |
| | 256 (ext.) | | | 0.92 / 0.78 | |
| `inflora` | 4 | 0.98 / 0.54 | 0.96 / 0.58 | 0.94 / 0.64 | 0.94 / 0.58 |
| | 16 | 0.84 / 0.56 | **0.90 / 0.82** | 0.86 / 0.48 | 0.82 / 0.76 |
| | 64 | 0.96 / 0.58 | 0.90 / 0.66 | 0.84 / 0.92 | 0.72 / 0.90 |
| `pca_trained` | 16 | 0.86 / 0.72 | **0.98 / 0.82** | 0.86 / 0.78 | 0.92 / 0.84 |

- **Retention is high everywhere.** Spatial is 0.72–0.98 on all 32 arms, and 0.82 or higher on 31.
  Plain LoRA's bracket kept 0.52 at best.
- **Acquisition is the binding side.**
  - Low lrs (3e-5) fall below the Object floor in both families with a fixed or random A.
  - Object is not monotone in lr within a rank. For example, `projected_lora` at rank 16 gives
    0.60 / 0.64 / 0.50 / 0.78 across the four lrs.
  - A single cell's interval is about ±0.13, so these wiggles are within noise. The rule, not the
    shape, decides.
- **The trade appears only at the largest updates.** `inflora` at (64, 1e-3) learns Object to 0.90 but
  keeps Spatial at only 0.72, the lowest of any arm.

## The matched ablation: unfreezing the down-projection (rank 16, same lr, same start)

| lr | `inflora` S / O | `pca_trained` S / O | f_outside (pca_trained) |
|---|---|---|---|
| 3e-5 | 0.84 / 0.56 | 0.86 / 0.72 | 0.11 |
| 1e-4 | 0.90 / 0.82 | 0.98 / 0.82 | 0.19 |
| 3e-4 | 0.86 / 0.48 | 0.86 / 0.78 | 0.41 |
| 1e-3 | 0.82 / 0.76 | 0.92 / 0.84 | 0.78 |

- **`pca_trained` is equal or higher on both tasks at all four lrs** (8 of 8 comparisons, 2 ties). Each
  single difference is within the ±0.13 cell noise. The consistency across lrs is the observation, on
  one seed.
- **As pre-written, this measures the effect of unfreezing the down-projection under the same
  protection constraint.** It is not geometric movement alone:
  - training A also doubles the trainable parameters at scale 1, and changes the optimisation;
  - the row space does move: f_outside, the share of the merged update outside the initial A_eff row
    space, rises with lr from 0.11 to 0.78 (median over layers);
  - `inflora`'s f_outside is 0 by construction.
- **Against random-init `projected_lora` at rank 16** (scale α/r = 1, the same as `pca_trained`'s): the
  PCA start learns Object better at every lr (0.72 / 0.82 / 0.78 / 0.84 against 0.60 / 0.64 / 0.50 /
  0.78). Across the two trained-A variants, the initialisation matters more than whether A trains.
  - Random-init A ends far from its start (f_outside 0.64–0.93 in every arm), so the training largely
    discards the random subspace.
  - The stricter control (A trained with its row space fixed) waits until it is justified.

## Diagnostics

**The constraint held on every arm.**
- The worst layer's residual ‖ΔW M‖ over its tolerance (atol 1e-5 ‖W‖ + rtol 1e-4 ‖ΔW‖) is
  0.0010–0.0019, on both the intended and the realized merged updates.
- One of the 91 layers has a full-rank memory and is not adapted, in every arm.

**Layer-output change on fixed Task-1 inputs:** tr(D K_old Dᵀ) / tr(W₀ K_old W₀ᵀ), summed over layers.
- It is the general form, valid for any update; for the projected updates the sandwiched form agrees.
- It measures single layers on fixed inputs, not the network.

| | total | max over layers |
|---|---|---|
| `gpm_projected_adam` stage 1 | 0.83% | 7.1% |
| LoRA v2 selected | 4.97% | 74% |
| `projected_lora` selected | 1.20% | 47% |
| `inflora` selected | 0.44% | 14% |
| `pca_trained` selected | 0.56% | 39% |

- The change grows with lr in every family (`projected_lora` rank 4: 0.44% → 6.8%).
- **It is not a retention score.** `projected_lora` (4, 1e-3), at 6.8%, still keeps Spatial at 0.86.
- **A residual of zero with nonzero output change** is what the rule anticipated: the memory protects
  95% of the captured Task-1 input energy, and the rest is unprotected.

**The update is not small.** The median ‖ΔW‖/‖W‖ over layers is 0.08–0.78 across arms, rising with
lr. The projection constrains the update's direction, not its size.

## Costs (T2 only; T1 is full fine-tuning in every family)

| | peak GPU (arm) | ms / step (benchmark, 300 steps) | training time / arm | design capture / task | trainable params |
|---|---|---|---|---|---|
| full fine-tuning (benchmark) | 3.53–3.62 GiB | 78–89 | | | 139.0M (all) |
| `projected_lora`, rank 4 / 16 / 64 | 3.26 / 3.30 / 3.45 GiB | 99–101 | 49–50 min | — | 0.5M / 2.0M / 8.2M |
| `inflora`, rank 4 / 16 / 64 | 2.25 / 2.28 / 2.40 GiB | 95–104 | 47–57 min | 93–179 s | 0.26M / 1.0M / 4.1M |
| `pca_trained`, rank 16 | 3.30 GiB | 98 | 48–49 min | 95–101 s | 2.0M |

- **Memory:**
  - `inflora` saves about 35% of peak GPU memory against full fine-tuning. A fixed A and a frozen
    base mean autograd keeps no layer input for a weight gradient.
  - The trained-A variants save 2–8%: A's gradient needs those inputs, and activations dominate at
    this model size.
- **Optimiser state:** about 16 B per trainable parameter (parameter, gradient, two AdamW moments):
  16–130 MB for the adapters, against about 670 MB for GPM's 42.0M dense registry weights.
- **Speed:** every variant is 10–25% slower per step than full fine-tuning, because A_eff = A − (AM)Mᵀ
  is formed every forward pass.
- **Not measured:** `gpm_projected_adam`'s own peak memory and step time (it also holds per-layer
  projectors).
- **Stored subspace:** the memory is the same file for all, 208 MB. `inflora` additionally fixes A
  (rank × d_in floats per layer).

## What this means

**Pre-written interpretations, applied:**
- **The claim this supports, on seed 0:** "comparable retention and acquisition to `gpm_projected_adam`
  with measured savings in trainable parameters (1.0–2.0M against 42.0M), optimiser state, and, for
  `inflora`, about 35% of peak GPU memory." It holds for `inflora` and `pca_trained`.
  - For `projected_lora` it holds on retention, with a −0.08 Object difference.
  - Not supported: any speed advantage (all variants are slower per step), or memory savings for the
    trained-A variants beyond a few percent.
- **The lever is the low-rank update confined to P⊥, not the low rank alone.** Plain LoRA at the same
  rank and a similar trainable count (2.0M) kept 0.52 Spatial.
- **The fixed versus trainable question has a seed-0 answer:** training A from InfLoRA's start helped
  at every lr; training A from a random start helped least. This is a statement about this protocol
  (Task-1 full fine-tuning, our eps-0.95 memory rule in place of DualGPM's), not about InfLoRA as
  published.

**Limits:**
- one seed;
- one transition;
- one memory, Task 1's;
- 50 rollouts per cell, so single-cell differences under about 0.13 are noise.

The 4-task runs are a separate step: they test whether the retention holds as the memory grows (more
layers become full rank and are skipped). They are reserved for Track 2's test set
(`projected_lora`, `inflora`), so they run after its freeze. They would include adaptive GPM as the
retention reference and separate rollout uncertainty from seed robustness.

**Track 2:** every bracket cell above is in neither the development nor the held-out set.
