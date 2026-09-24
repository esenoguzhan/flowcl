# Forgetting diagnostics — `gpm_projected_adam`, `seq_hetero`, seed 0 (no training)

**Status:** the pre-registered outcome is **case D in both comparisons**, with no disagreement:
- primary: Object across Goal's update;
- secondary: Goal across LIBERO-10's update.

Case D prescribes an inspection of the tail and the top layers. It finds a **flat tail**: no layer
gives the forgotten task even twice the control's interference. The maximum ratio is 1.50 on
activations before the update and 1.82 after it.  
**Date (local, CEST):** Thu 24 Sep 2026, finished 19:57. The run took 18 min and trained nothing.  
**Code:** `e722524`, clean tree (`allow_dirty` false). The decision rule
(`configs/analysis/forgetting_diagnostics.yaml`) was committed in that commit, before the run.  
**Inputs:** `results/seq_hetero__gpm_projected_adam__seed0` and
`results/seq_hetero__seq_ft__seed0`. The report records every checkpoint and memory artifact it
read by SHA-256.  
**Companion record:** `2026-09-24_gpm_seq_hetero.md`: §6 states the hypothesis, §14 the rule.

This note accompanies `results/forgetting_diag/report.json`; cite the JSON for numbers. **These
diagnostics localize a likely mechanism. They cannot establish causality.**

---

## 1. What was measured

- **The loss matrix `L[i][j]`.** This is the fixed-batch masked flow-matching loss, for both runs,
  all four stages and all four tasks.
  - It uses the pilot's exact probe: 16 × 64 samples per cell, with the same batches, `s` and noise
    for every checkpoint.
  - **Instrument check:** seq_ft's four stage-0/1 cells reproduce the pilot's recorded probe losses
    exactly (relative difference 0.0). `L` is therefore the same instrument as pilot §3.
- **Per-layer activation interference `r_l = ‖ΔW X‖ / ‖W X‖`.** It covers all 91 registry layers and
  is computed exactly from input Grams over each task's full dataset.
  - `ΔW` is one training stage's update; `X` is the evaluated task's inputs to the layer.
  - **Direct** uses `X` from the network before the update.
  - **After drift** uses `X` from the network after it. Only `X` changes between the two.
  - The capture seeds do not depend on the stage, so both see identical observations, `s` and
    noise.

| Comparison | Update (trains) | Target (forgotten) | Control (retained) |
|---|---|---|---|
| primary (decides) | stage 1 → 2 (Goal) | Object: 78% → 2% | Spatial: 94% → 90% |
| secondary | stage 2 → 3 (LIBERO-10) | Goal: 94% → 56% | Spatial: 90% → 88% |

The seq_ft run supplies the scale anchor: the same transitions, measured on the target only.

---

## 2. Decision

| Question | Answer |
|---|---|
| Pre-registered case | **D in both comparisons.** By the pre-registered ratio, the loss damage is selective: `R_O/R_S` = 5.1 (primary) and 15.4 (secondary). Both interference medians are small, however: `Q` is between 0.98 and 1.17. |
| Does the forgotten task receive more interference than the control? (case A) | **No.** The median per-layer ratio `q_l` is 1.11 in the trunk and 1.03 in the decoder, on stage-1 activations. No layer reaches 2; the maximum is 1.50. |
| Does representation drift amplify it? (case B) | **Barely.** Drift raises Object's interference by 3% (trunk) and 11% (decoder), and Spatial's by 3% and 7%. `Q_drift` ≈ `Q_direct`. |
| Is the damage concentrated in a few layers? (the case D inspection) | **No.** The tail is flat. The 90th percentile of `q_l` is ≤ 1.35, the maximum is 1.64 in the primary and 1.82 in the secondary, and no layer has `q_l` ≥ 2. The largest ratios, 1.3–1.8, are in the decoder's cross-attention key projections: a mild but consistent excess. |
| How large is GPM's residual interference? | About a **9% relative change** in each layer's output, for the target and the control alike. That is about 30% of seq_ft's per-layer interference, and 3–13% of seq_ft's loss increase. |
| Is the cause of the forgetting established? | **No.** The diagnostics rule out two localized mechanisms. They do not show what causes the forgetting. |

---

## 3. Loss matrix

**`gpm_projected_adam`.** The diagonal is the task just trained.

| after \ on | Spatial | Object | Goal | LIBERO-10 |
|---|---:|---:|---:|---:|
| Spatial | **0.00986** | 1.62007 | 1.80034 | 1.45537 |
| Object | 0.04216 | **0.00717** | 1.72154 | 1.30250 |
| Goal | 0.10161 | 0.08853 | **0.00657** | 1.54529 |
| LIBERO-10 | 0.09624 | 0.12122 | 0.09574 | **0.00853** |

**seq_ft**

| after \ on | Spatial | Object | Goal | LIBERO-10 |
|---|---:|---:|---:|---:|
| Spatial | **0.00986** | 1.62007 | 1.80034 | 1.45537 |
| Object | 0.72515 | **0.00638** | 1.63018 | 1.50987 |
| Goal | 2.67954 | 1.42680 | **0.00588** | 2.36495 |
| LIBERO-10 | 1.34378 | 1.03112 | 0.69543 | **0.00772** |

- **Old-task losses.** Under GPM they stay at 0.04–0.12; under seq_ft they reach 0.70–2.68.
- **GPM's loss increase as a fraction of seq_ft's**, per transition (`loss_forgetting_fraction`):
  - Spatial: 4.5% at T2 and 3.0% at T3;
  - Object: 5.7% at T3;
  - Goal: 12.9% at T4.

  It is undefined where seq_ft's own loss fell (2 → 3, for Spatial and Object).

**Probe loss against rollout success** (GPM, same checkpoint):

| Task | Stage | Probe loss | Success |
|---|---:|---:|---:|
| Spatial | 1 | 0.042 | 94% |
| Spatial | 2 | 0.102 | 90% |
| Spatial | 3 | 0.096 | 88% |
| Object | 1 | 0.0072 | 78% |
| **Object** | **2** | **0.089** | **2%** |
| Object | 3 | 0.121 | 0% |
| Goal | 2 | 0.0066 | 94% |
| **Goal** | **3** | **0.096** | **56%** |

At a similar probe loss of about 0.09–0.10, **Spatial succeeds 88–90%, Goal 56% and Object 2%.**

---

## 4. The classification

| | primary: Object vs Spatial, Goal's update | secondary: Goal vs Spatial, LIBERO-10's update |
|---|---|---|
| `R_O` (target) | 12.34 (0.0072 → 0.0885) | 14.57 (0.0066 → 0.0957) |
| `R_S` (control) | 2.41 (0.0422 → 0.1016) | 0.95 (0.1016 → 0.0962) |
| `R_O / R_S` (`F_loss`) | 5.12 (1.63) → **selective** | 15.38 (2.73) → **selective** |
| `Q_direct`, trunk / decoder | 1.109 / 1.028 | 1.169 / 0.977 |
| `Q_drift`, trunk / decoder | 1.111 / 1.048 | 1.167 / 1.043 |
| **Case** | **D** | **D** |

**The tail of `q_l`** over all 90 moving layers. `action_in` does not move, so it is excluded by
the zero rule.

| | p75 | p90 | max | share ≥ 2 |
|---|---:|---:|---:|---:|
| primary, direct | 1.13 | 1.24 | 1.50 | 0 |
| primary, after drift | 1.14 | 1.27 | 1.64 | 0 |
| secondary, direct | 1.19 | 1.27 | 1.73 | 0 |
| secondary, after drift | 1.19 | 1.30 | 1.82 | 0 |

**Top layers.**
- **Primary, direct:**

  | Layer | `q` | `r` Object | `r` Spatial |
  |---|---:|---:|---:|
  | `flow_head.blocks.0.cross_attn.k_proj` | 1.50 | 0.075 | 0.050 |
  | `flow_head.blocks.1.cross_attn.k_proj` | 1.36 | | |
  | `trunk.state_projection` | 1.36 | 0.014 | 0.010 |
  | `flow_head.blocks.3.cross_attn.k_proj` | 1.34 | | |
  | `trunk.blocks.6.attn.q_proj` | 1.30 | | |

- **Primary, after drift:** the four decoder cross-attention `k_proj` layers lead, at 1.41–1.64.
- **Secondary:** the same `k_proj` layers lead, at 1.36–1.82.

These layers read the trunk's observation tokens into the action decoder. This is the one place
where the forgotten task is consistently more exposed, and the excess is modest.

**Group medians** are 0.95–1.17. The one exception is the 8-dimensional state projection: 1.36 in
the primary, 0.68 in the secondary.

---

## 5. Interference magnitudes (reported, not decisive)

The primary comparison. Each cell is the median over layers, trunk / decoder.

| | Object (target) | Spatial (control) | seq_ft, Object |
|---|---:|---:|---:|
| `r`, direct | 0.090 / 0.094 | 0.080 / 0.093 | 0.320 / 0.343 |
| `r`, after drift | 0.094 / 0.106 | 0.083 / 0.102 | 0.321 / 0.371 |
| energy outside memory, direct | 0.050 / 0.048 | 0.043 / 0.045 | — |
| energy outside memory, after drift | 0.053 / 0.057 | 0.045 / 0.049 | — |

- **Scale against seq_ft.** The ratio `r_gpm / r_seqft` on the same transition is 0.28–0.33 in the
  primary and 0.30–0.35 in the secondary.
- **The secondary comparison looks the same.**
  - `r`, direct: Goal 0.092 / 0.091 against Spatial 0.078 / 0.094.
  - Energy outside memory: Goal 0.050 / 0.048 against Spatial 0.038 / 0.048.
  - Drift amplification: 1.03 / 1.11 for Goal, 1.04 / 1.03 for Spatial.

---

## 6. Reading

**What the per-layer measures rule out, as far as they reach:**
1. **Direct unprotected interference that is larger for the forgotten task.**
   - Object's inputs carry about as much energy outside memory as Spatial's: 5.0% against 4.3% in
     the trunk.
   - The update perturbs both tasks' layer outputs by about the same relative amount.
   - The observed ratios are close to what the difference in unprotected energy alone predicts:
     trunk `√(0.050/0.043)` = 1.08 against `Q` = 1.11; decoder 1.03 against 1.03.
2. **Drift amplification.** It adds only 3–11%.
3. **Object-specific damage concentrated in a few layers.** The tail is flat.

**What the numbers point to (not established).**
- **The residual interference looks roughly uniform across tasks.** ε = 0.95 leaves every task about
  5% of its input energy outside memory. The next update acts on that remainder, changing each
  layer's output by about 9%.
- **Tasks appear to differ in how much of it they tolerate in closed loop.** At a similar probe loss,
  Spatial keeps 88–90%, Goal 56% and Object 2% (§3).
- This is the reading case C describes, a rollout/probe-loss mismatch. It was reached through case D
  because the pre-registered ratio called the loss damage selective. See the caveat below.

**A caveat on "selective" that the pre-registration did not anticipate.**
- **The two baselines differ.** `R_O` is measured from the target's freshly trained minimum. `R_S`
  starts from a control that was already perturbed: Spatial's loss at stage 1 was 0.042, 4.3× its
  trained 0.0099.
- **So the ratio statistic favours "selective" for any freshly trained target.** At a minimum, any
  perturbation raises the loss. Away from it, a perturbation can raise or lower it.
- **In absolute terms, the primary damage is similar for both tasks:** +0.081 for Object against
  +0.059 for Spatial, a factor of 1.4.
- **In the secondary, Spatial's loss fell slightly** (−0.005) while Goal's rose by 0.089.

The classification stays D, as pre-registered. The baseline asymmetry is recorded as a limitation
of the rule (§7).

**Consequence for the run record's §6 hypothesis.** `r` is energy-weighted: it measures *how much*
a layer's output moves, not *which* directions move. The diagnostics therefore neither confirm nor
refute the hypothesis in its specific form: that the unprotected part of a later task is its
task-specific part.

They do show that the **amount** of residual interference is not larger for the forgotten tasks.
The adaptive-GPM intervention remains the causal test of whether more protection for later tasks
improves their retention.

---

## 7. Caveats

1. **The thresholds are judgements**: `R_O ≥ 2`, `R_O/R_S ≥ 2` and `Q ≥ 2`.
2. **The control's baseline is asymmetric** (§6). A future rule should compare absolute loss
   changes, or use a control that sits at its own minimum.
3. **All measurements are in-sample.** `X` comes from the training demos; there are no held-out
   demos. The trunk inputs match the memory capture's. The decoder inputs use different `s` and
   noise draws.
4. **`r` averages over a layer's inputs and outputs, weighted by energy.** A small perturbation in a
   direction critical for behaviour is invisible to it.
5. **The probe loss averages over flow time, action components and chunk positions.** Closed-loop
   success may hinge on a small part of that, such as grasp timing.
6. **One seed and one curriculum order.**

---

## 8. Next step (decision needed)

Case D sends the rule to an inspection of the tail and top layers. That inspection is done (§4) and
found a flat tail. No pre-registered branch goes further. The options:

1. **Adaptive-GPM run as the causal test** (recommended; about 5 h). Its target is
   `c_target = max(0.95, p + 0.90(1 − p))`.
   - **What the diagnostics predict.** Raising later tasks' captured energy from 0.95 to about 0.99
     cuts their energy outside memory from about 5% to about 1%. Object's per-layer interference
     under Goal's update should then fall to roughly `√(0.01/0.05)` ≈ 0.45 of today's value, which
     is below Spatial's. This is approximate: updates confined to a smaller free subspace may grow.
   - **How to read the outcome.**
     - If Object's retention after T3 then rises substantially, the amount of residual interference
       matters for the less tolerant tasks.
     - If interference falls as predicted but retention does not, residual under-protection does not
       explain the forgetting.
   - **Built-in checks.** Stage 0 and T1's memory must be bitwise identical to this run's. Capacity
     is watched, since the `mlp.fc2` layers are already at ρ = 0.82.
   - It gets its own short plan and pre-registration.
2. **Downstream inspection** (cheap, under 1 h, no training).
   - Break the loss down by action component (position, rotation, gripper) and by chunk position,
     for Object and Spatial at stages 1–2.
   - Record a few paired Object rollouts at stages 1 and 2 to see where episodes fail.

   This would say what "less tolerant" means for Object. It would not change whether option 1 is
   worth running.
3. **A task-order control** (`seq_hetero_reverse`: seq_ft plus GPM, about 11 h). It separates
   "Object is fragile" from "the second position is fragile".

Seeds 1–2 still wait until the protection rule is settled.

---

## 9. Artifact map

```
results/forgetting_diag/report.json     # canonical: loss matrices (both runs), instrument check,
                                        # loss_forgetting_fraction, per comparison: classification,
                                        # q summaries (tails, top layers, group medians), reported
                                        # magnitudes, per-layer rows; input SHA-256s; git_sha; timings
configs/analysis/forgetting_diagnostics.yaml   # the pre-registered rule (committed in e722524)
flowcl/experiments/forgetting_diagnostics.py   # measurement and classification
```
