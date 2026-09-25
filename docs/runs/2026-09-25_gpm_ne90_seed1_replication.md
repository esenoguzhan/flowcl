# Seed-1 replication — adaptive GPM (`gpm_projected_adam_ne90`) against plain GPM, `seq_hetero`

**Status:** the pre-registered replication outcome is **replicated**, meaning both seeds show
causal transition support. It is **not durably replicated**:
- seed 0 was `durable_support`;
- seed 1 is `transition_support`, flagged *T4 plasticity below threshold*. LIBERO-10 reached 74%
  against its 83% threshold, and 96% under plain GPM.

The retention effect replicates almost exactly. The capacity cost that seed 0 hinted at shows up as
a real T4 plasticity loss on seed 1.  
**Date (local, CEST):** Fri 25 Sep 2026, 10:32 → 20:55 (one queue: `results/logs/queue_20260925_103234/`).  
**Code:** `e2fc715` (clean). Per-seed thresholds, the premise gate and the replication rule were
committed before any seed-1 method result existed.  
**Runs** (all under `results/`):
- `seq_hetero__gpm_projected_adam__seed1` (baseline);
- `seq_hetero__gpm_projected_adam_ne90__seed1` (variant);
- `seq_hetero__seq_ft__seed1` (reference).

**Companion records:** `2026-09-25_gpm_ne90_seq_hetero.md` (seed 0) and
`2026-09-24_gpm_forgetting_diagnostics.md`.

Cite these reports:
- `results/adaptive_gpm_seed1/report.json` (seed-1 verdict);
- `results/adaptive_gpm/replication.json` (replication);
- `results/gpm_seq_seed1/report.json` and `results/gpm_seq_ne90_seed1/report.json` (sequence reports);
- `results/forgetting_diag_seed1/report.json` and `results/forgetting_diag_ne90_seed1/report.json`
  (diagnostics).

---

## 1. Provenance and pairing

| Check | Result |
|---|---|
| T1 pairing, both runs, against seq_ft seed 1 | relative weight difference **0.0**; final loss identical (0.0022294) |
| Variant identity against plain GPM seed 1 | **0 of 656** tensors differ at stage 0 and at stage 1. T1 memory is equal; `memory_task1` differs, as intended |
| Sequence-report provenance, both runs | all pass: clean SHA, seed namespace, occupancy, `memory_chained` (prefix-identical), residuals ≤ 2.8% of the bound, artifact hashes, 565 frozen tensors |
| Thresholds | seed 1's own: 96 / 90 / 100 / 98 − 15 pp = **81 / 75 / 85 / 83** |
| Pilot comparison | skipped: the pilot was rolled out under seed 0's namespace |
| Diagnostics instrument check | **not applicable**. The pilot's probe references exist only for seed 0, where the diagnostics were validated exactly. The checks above verify checkpoint and seed provenance, but do not replace the probe-loss instrument check. |

---

## 2. Results

Success rate (%), 50 paired episodes per cell.

**Plain GPM, seed 1** (the premise):

| after \ on | Spatial | Object | Goal | LIBERO-10 |
|---|---:|---:|---:|---:|
| Spatial | 96 | 0 | 0 | 0 |
| Object | 98 | **78** | 0 | 0 |
| Goal | 86 | **2** | **98** | 0 |
| LIBERO-10 | 92 | **0** | **20** | **96** |

**Adaptive GPM, seed 1:**

| after \ on | Spatial | Object | Goal | LIBERO-10 |
|---|---:|---:|---:|---:|
| Spatial | 96 | 0 | 0 | 0 |
| Object | 98 | **78** | 0 | 0 |
| Goal | 98 | **66** | **100** | 0 |
| LIBERO-10 | 94 | **82** | **72** | **74** |

**Paired differences, variant − baseline, seed 1** (with seed 0 alongside):

| Cell | seed 1 | seed 0 |
|---|---:|---:|
| Object after Goal `R[2][1]` (transition) | **+64 [+50, +78]** | +66 [+52, +78] |
| Object final `R[3][1]` | **+82 [+70, +92]** | +84 [+72, +94] |
| Goal final `R[3][2]` | +52 [+34, +70] | +38 [+22, +54] |
| Spatial after Goal `R[2][0]` | +12 [+4, +22] | — |
| Goal learned `R[2][2]` (T3 plasticity) | +2 [0, +6] | 0 [−6, +6] |
| **LIBERO-10 learned `R[3][3]` (T4 plasticity)** | **−22 [−36, −8]** | +2 [−10, +14] |

| | variant s1 | baseline s1 | variant s0 | baseline s0 |
|---|---:|---:|---:|---:|
| F_1 | 80.5% | 52.0% | 92.0% | 58.5% |
| NBT | 8.7 pp | 53.3 pp | −4.7 pp | 39.3 pp |
| AUC | 88.1% | 74.5% | 88.5% | 74.1% |

---

## 3. Pre-registered verdicts

**Seed 1** (`adaptive_gpm_seed1/report.json`): **`transition_support`**, flagged *T4 plasticity
below threshold*.

| Check | seed 1 |
|---|---|
| identity | pass |
| per-layer energy | pass: 269 applicable, 0 failures, minimum new-energy share 0.9000006 |
| premise (the baseline forgot Object) | pass: 2% < 75% |
| interference (≤ 0.70 per half) | pass: trunk 0.67, decoder 0.41 |
| transition gain and T3 plasticity | pass × pass: +64 pp [+50, +78]; Goal 100% |
| durable Object ≥ 75% | pass: 82% |
| **T4 plasticity ≥ 83%** | **fail: 74%** (borderline, CI [62, 86]) |

**Replication** (`adaptive_gpm/replication.json`): **`replicated`**. Both seeds show causal
transition support. Because seed 1 is not `durable_support`, the outcome is not
`durably_replicated`.

**The sequence criteria** (`sequence_report.yaml`, applied to the variant seed 1) fail twice:
- T4 plasticity: 74% < 83%. Under the step-8 rule, a plasticity failure on T4 is the SGP-fallback
  trigger, so it is flagged for the variant.
- Goal final retention: 72% < 85%.

Object passes (82% ≥ 75%) and Spatial passes (94% ≥ 81%).

Plain GPM seed 1 fails final retention on Object (0%) and Goal (20%), with no plasticity failure.
That is the same pattern as seed 0.

---

## 4. What replicated, and what did not

**Replicated, with nearly the same magnitude:**
- **The baseline's forgetting.** Object goes from 78% to 2% to 0% in both seeds. Goal ends at 20%
  (seed 1) and 56% (seed 0).
- **The diagnostics.** Plain GPM seed 1 is again case D in both comparisons, computed independently,
  so it agrees with seed 0. Its interference ratios `Q` are 1.11 / 1.03 (seed 0: 1.11 / 1.03).
- **The manipulation.** The share of new energy protected is 0.90 in every applicable layer. Object's
  interference ratio is 0.67 / 0.41 (seed 0: 0.67 / 0.43).
- **The causal effect on Object.** The transition gain is +64 pp against +66 pp, and the final gain
  +82 pp against +84 pp.

**Not replicated: T4 plasticity.**
- In seed 1 LIBERO-10 reached **74%**, against 96% under plain GPM (−22 pp, CI excluding 0). Seed 0
  gave 92% against 90%.
- **At the loss level the cost is the same in both seeds.** The variant's T4 last-50 training loss
  is **0.0136** in both. The baseline's is 0.0083–0.0087, and seq_ft's 0.0073.
- So T4 learns worse under the variant in *both* seeds. Only seed 1's rollouts crossed the
  threshold, and seed 0's pass looks partly fortunate.
- T3 shows no such cost: the loss is 0.0087–0.0091 against 0.0077–0.0080, the same gap as the
  baseline's own gap to seq_ft.

**Why T4.**
- Entering T4, the variant's memory already covers **94% of the trunk's input dimensions** at the
  median:
  - `trunk_mlp` median 0.96, max 0.98;
  - `trunk_attn` 0.94;
  - decoder cross-attention 0.89;
  - `trunk.state_projection` full.
- During T4, **99.5% of the trunk gradient norm points into memory.** LIBERO-10 must be learned
  almost entirely through the decoder's self-attention and MLP layers, which are about 30–35%
  occupied.
- This is the capacity-collapse risk that build step 8 was designed to detect. The fixed-ε baseline
  never reached it. The adaptive rule does, at the fourth task.

**Reading.**
- Across two seeds, stronger protection of later tasks' new energy **causally prevents the
  forgetting of the protected tasks**. For Object this holds with an almost identical effect size.
- **The protection costs capacity.** After three tasks the trunk is nearly exhausted, and the fourth
  task's learning degrades: consistently at the loss level, and in one of two seeds past the
  pre-registered threshold.
- This is a stability–plasticity trade-off, governed by capacity. It is not a free improvement.

---

## 5. Consequences for the thesis claim

- **Supported, over 2 seeds:** fixed total-energy GPM under-protects later tasks, and that
  under-protection causally explains their forgetting.
  - Across the next task, the adaptive intervention removes most of Object's forgetting. Object goes
    from 78% to 66–68%, instead of from 78% to 2%.
  - Object ends at 82–84%, against 0% for plain GPM.
- **Not supported as a working solution for longer sequences:** with four heterogeneous tasks the
  adaptive rule nearly exhausts the trunk, and the last task's plasticity suffers.
- Say **"a capacity-bounded trade-off"**, not "adaptive GPM solves forgetting".

---

## 6. Next steps (decided 25 Sep)

**The measured immediate mechanism of the T4 loss is capacity saturation.** The trunk is 94%
occupied entering T4, and 99.5% of its T4 gradient lies in the protected subspace.

**Sequence:**
1. **The seed-2 set** (seq_ft, plain and adaptive GPM seed 2), as pre-registered. It shows how
   consistently the T4 cost crosses the behavioural threshold.
   - Seed 2's thresholds are derived by the pre-registered per-seed rule, `θ_j = R_seqft[j][j] − 0.15`.
     This is an explicit, fail-closed entry in `sequence_report.yaml`.
2. **SGP**, in its own plan and pre-registration.
   - The step-8 SGP trigger (plasticity failure on T3/T4) fired for the **adaptive variant on seed
     1**, not for plain GPM on either seed.
   - SGP is also next in the spec's build order.
3. **Gate 4** (the `s`-binned characterization over 3 seeds, no training), early. It decides whether
   the thesis extension (`s`-binned SGP, README §7.5) is pursued.
4. **The required baselines and core ablations:** `replay`, `lora`, `ewc`, then `consft`.
5. **The high-protection control:** ε = 0.95 at T1, then a fixed 0.99 from T2 on. It keeps T1 and T2
   identical, which a global ε = 0.99 would not, and separates the *adaptive* allocation rule from
   stronger protection in general.
6. **The rest of the Stage A matrix:** `seq_correlated` and `seq_hetero_reverse`, each with 3 seeds.
   Both are required by README §5.

**Not doing:**
- an improvised occupancy cap or a smaller `f` (post-hoc method tuning);
- a fifth-task stress test (the capacity limit is already visible at T4).

**SGP naming, fixed now.** The SGP plan must say which of these it runs:

| Name | Memory | Projection |
|---|---|---|
| **Paper SGP** (baseline) | standard GPM memory | `G' = G(I − M Λ Mᵀ)`, with basis-wise importance `λ_i = (α+1)σ_i / (α σ_i + max σ)`, accumulated across tasks and capped at 1 (Saha & Roy, AAAI 2023, Eq. 2–10) |
| **Layerwise scaled projection** | — | README §6 item 7: `G' = G_⊥ + α_l G_∥`, one `α_l` per layer |
| **Adaptive SGP** (thesis method) | adaptive ne90 memory | basis-wise or layerwise scaling; our hybrid extension, not the published method |

Notes for the SGP plan:
- **The paper's cross-task importance** uses surrogate singular values. In our Gram pipeline they are
  `σ'_i² = m_iᵀ K m_i`.
- **With Adam, the paper projects Adam's output** (Adam-GP, App. E). That matches our update
  projection.
- **α must be fixed a priori.** The paper used values from 1 to 25.
- **Deviations from the spec's per-layer `α_l`** are documented.
