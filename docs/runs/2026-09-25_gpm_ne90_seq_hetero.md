# Adaptive-GPM run — `gpm_projected_adam_ne90`, `seq_hetero`, seed 0 (the causal test)

**Status:** pre-registered verdict **strong / durable support**. Every gate and every outcome
check passed:
- **identity:** stages 0–1 are bitwise equal to the baseline;
- **per-layer energy target:** met in all 269 applicable layer-extensions;
- **interference:** reduced below 0.70× the baseline's;
- **Object's retention across T3:** +66 pp;
- **T3 and T4 plasticity:** both held;
- **Object's final retention:** 84%.

**Main caveat:** the trunk's memory capacity is **nearly exhausted** after four tasks (§5).  
**Date (local, CEST):** Thu 24 Sep 2026 20:50 → Fri 25 Sep 01:35 (4 h 44 min). The analyses
finished at 01:53. An overnight queue ran it, and seq_ft seed 1 followed until 07:00.  
**Code:** `9701620` (clean). The rule `configs/analysis/adaptive_gpm.yaml` was committed before the
run.  
**Run:** `results/seq_hetero__gpm_projected_adam_ne90__seed0`.
**Baseline:** `…gpm_projected_adam__seed0`. **Reference:** `…seq_ft__seed0`.  
**Companion records:**
- `2026-09-24_gpm_seq_hetero.md` (the baseline);
- `2026-09-24_gpm_forgetting_diagnostics.md` (why this test).

Cite `results/adaptive_gpm/report.json` (the verdict), `results/gpm_seq_ne90/report.json` (the
sequence report) and `results/forgetting_diag_ne90/report.json` (the diagnostics).

---

## 1. What was tested

The baseline kept T1 but lost T2 and T3. At ε = 0.95 on total energy, only 22–49% of each later
task's *new* input energy was protected. This run changes one thing, the memory target after each
task:

    c_target = max(0.95, p + 0.90 (1 − p)),    p = the task's energy already in memory

- **At T1, p = 0,** so the target is exactly 0.95. Stages 0 and 1 are therefore identical to the
  baseline by construction, and the run checked this fail-fast.
- **The variant only diverges at the memory extension after T2.** Everything downstream of it is
  the effect of the manipulation.

Everything else is identical to the baseline:
- the data, `s` and noise streams (seq_ft's namespace) and the rollout episode seeds;
- the recipe;
- ε;
- the frozen set.

---

## 2. Verdict (pre-registered)

| Check | Rule | Result |
|---|---|---|
| **Identity** | stage-0/1 state dicts and the T1 memory equal the baseline's | **pass**: 0 of 656 tensors differ at stages 0 and 1; all 91 T1 bases are equal; `memory_task1` differs, as intended |
| **Per-layer energy** | every applicable layer: `captured ≥ target − 1e-6` (target recomputed and matching the recorded one) **and** new-energy share ≥ 0.90 | **pass**: 269 applicable, 4 not applicable, 0 failures. The minimum share is 0.90001 |
| **Interference** | variant/baseline median direct `r_Object` under Goal's update ≤ 0.70 in both halves | **pass**: trunk **0.67**, decoder **0.43** (predicted ≈ 0.45) |
| **Transition retention × T3 plasticity** | Δ`R[2][Object]` ≥ +20 pp with CI lower > 0, **and** `R[2][2]` ≥ 0.85 | **pass × pass**: **+66 pp [+52, +78]**; Goal learned to 94% |
| **Durable** | `R[3][Object]` ≥ 0.63 **and** `R[3][3]` ≥ 0.83 | **pass**: Object **84%**; LIBERO-10 92% |
| **Verdict** | | **strong / durable support** |

**In words.**
- Raising later tasks' protection is followed by their retention, with learning unaffected.
- In this seed and curriculum, that supports the claim that **under-protection of later tasks
  explains their forgetting under fixed-ε GPM**.

**Scope.**
- It supports "later tasks were under-protected". It does not support the narrower claim that
  *task-specific* directions were the unprotected part, because the variant protects more energy
  overall.
- It is one seed.

---

## 3. Headline results

Success rate (%), 50 rollouts per cell, with the same episode seeds as the baseline and seq_ft.

**`gpm_projected_adam_ne90`**

| after \ on | Spatial | Object | Goal | LIBERO-10 |
|---|---:|---:|---:|---:|
| Spatial | **90** [80, 98] | 0 | 0 | 0 |
| Object | 94 [86, 100] | **78** [66, 88] | 0 | 0 |
| Goal | 96 [90, 100] | **68** [54, 80] | **94** [86, 100] | 0 |
| LIBERO-10 | **98** [94, 100] | **84** [72, 94] | **94** [86, 100] | **92** [84, 98] |

**Baseline (`gpm_projected_adam`), for comparison:** Object after Goal 2%, final 0%. Goal final
56%. Spatial final 88%.

Paired differences, variant − baseline:

| Cell | Δ (pp) |
|---|---:|
| Object after Goal `R[2][1]` | **+66 [+52, +78]** |
| Object final `R[3][1]` | **+84 [+72, +94]** |
| Goal final `R[3][2]` | **+38 [+22, +54]** |
| Goal learned `R[2][2]` | +0 [−6, +6] |
| LIBERO-10 learned `R[3][3]` | +2 [−10, +14] |

**All pre-registered sequence criteria pass** (`sequence_report.yaml`; the baseline failed two of
them):
- plasticity: 90 / 78 / 94 / 92%, against thresholds of 75 / 63 / 85 / 83%;
- final retention: 98 / 84 / 94%, against 75 / 63 / 85%.

| | variant | baseline | seq_ft |
|---|---:|---:|---:|
| F_1 | **92.0%** | 58.5% | 24.5% |
| NBT | **−4.7 pp** | 39.3 pp | 89.3 pp |
| AUC | **88.5%** | 74.1% | 46.7% |

- **Negative NBT means the old tasks ended *above* their just-trained level.** Spatial went from 90
  to 98%, and Object from 78 to 84%.
- **Object dipped, then recovered.** It fell to 68% after Goal, with 9 episodes lost and 4 gained,
  and recovered to 84% after LIBERO-10, with 9 gained and 1 lost.
- These metrics are point values, with no CIs, as in the earlier runs.

**Probe loss** (fixed batches; `results/forgetting_diag_ne90/report.json`):

| Old task | Variant | Baseline |
|---|---:|---:|
| Object after Goal | 0.023 | 0.089 |
| All old-task cells at the end | 0.023–0.059 | 0.04–0.12 |

---

## 4. Manipulation: what actually changed

| | after T2 | after T3 | after T4 |
|---|---:|---:|---:|
| new-energy share protected, median trunk / decoder | 0.90 / 0.90 | 0.90 / 0.90 | 0.91 / 0.90 |
| (baseline) | 0.49 / 0.24 | 0.38 / 0.22 | 0.33 / 0.27 |
| energy left unprotected, median trunk / decoder | 0.010 / 0.006 | 0.003 / 0.001 | 0.001 / 0.001 |
| (baseline) | 0.050 / 0.048 | 0.050 / 0.047 | 0.050 / 0.047 |

- **Interference fell as intended.** Object's median per-layer interference under Goal's update fell
  from 0.090 / 0.094 to 0.061 / 0.041 (trunk / decoder).
- **The decoder matched the prediction** (0.43, against ≈ 0.45).
- **The trunk fell less** (0.67). It still passed the pre-registered 0.70, but with little margin.
- **By the diagnostics rule, the variant is still case D.** Object's loss rose 3.2× (0.0072 → 0.0231),
  but its per-layer interference is now *below* Spatial's: `Q_direct` is 0.88 / 0.84.

---

## 5. Capacity — the main caveat

Median occupancy ρ, with the max in brackets (`adaptive_gpm/report.json`, `capacity.variant`):

| Group | after T1 | after T2 | after T3 | after T4 | baseline after T4 |
|---|---:|---:|---:|---:|---:|
| trunk_attn (32) | 0.43 | 0.76 (0.83) | 0.94 (0.96) | **0.99 (0.99)** | 0.60 |
| trunk_mlp (16) | 0.48 | 0.82 (0.90) | 0.96 (0.98) | **0.99 (1.00)** | 0.65 |
| decoder_cross_attn (16) | 0.29 | 0.68 | 0.89 | **0.97** | 0.56 |
| decoder_self_attn (16) | 0.04 | 0.15 | 0.29 | 0.46 | 0.05 |
| decoder_mlp (8) | 0.03 | 0.12 | 0.25 | 0.48 | 0.04 |
| decoder_output | 0.08 | 0.35 | 0.63 | 0.84 | 0.11 |
| **trunk median** | 0.44 | 0.78 | 0.94 | **0.99** | 0.62 |

- **The trunk is effectively full after four tasks.** It has 1.2% of its input dimensions free at the
  median.
  - `trunk.state_projection` is exhausted from T3 on, and `action_in` since T1.
  - At T4 the raw trunk gradient already put **99.5% of its norm** into memory (`c_l` 0.995), and LIBERO-10 was
    still learned to 92%. Most of that learning must have gone through the decoder, whose
    self-attention and MLP layers are still about half free.
- **The rule extends every layer with any new energy left** (the consequence noted in the plan), and
  later tasks are 90–99% inside memory. Protecting 90% of their small new energy therefore still adds
  many dimensions.
- **Consequence for the thesis claim.**
  - This rule works for a four-task curriculum. It is **not** shown to scale: a fifth heterogeneous
    task would find almost no free trunk capacity.
  - Part of the retention gain may also come from the trunk being nearly frozen by T4. T1 rising
    from 88% to 98% fits that reading.
- **Stored memory** is 231 MB (float32 equivalent), against 140 MB for the baseline.

---

## 6. Provenance

All sequence-report checks pass:
- the clean SHA;
- the seed namespace;
- the T1 pairing (relative weight difference 0.0000);
- occupancy non-decreasing;
- `memory_chained` (prefix-identical);
- residuals: worst 2.8% of the bound;
- artifact hashes;
- frozen tensors: 565 unchanged.

Allowlisted layers that moved per stage: 90 / 90 / 89. `trunk.state_projection` stopped moving
at T4 because it is full.

---

## 7. seq_ft seed 1 (the second job in the queue)

It finished at 07:00 (`results/seq_hetero__seq_ft__seed1`).
- **Diagonal:** 96 / 90 / 100 / 98%. Every off-diagonal cell is 0.
- **Summary:** F_1 24.5%, NBT 95.3 pp.
- **Checkpoints:** all four stage checkpoints exist, so it can serve as the pairing reference for
  any method's seed 1.

**Decision needed before any seed-1 method run.** Seed 1's diagonal differs from seed 0's
(Object 90% against 78%). The thresholds are `R_seqft[j][j] − 15 pp`, and
`sequence_report.yaml` asserts seed 0's values (75 / 63 / 85 / 83). A seed-1 report would
therefore raise `reference run changed`. There are two options:
- keep the thresholds fixed at seed 0's values;
- recompute them from each seed's own reference, which would give 81 / 75 / 85 / 83 for seed 1.

This must be pre-registered before seed-1 results exist.

---

## 8. What this does *not* show

1. **Replication.** It is one seed. The run-to-run spread seen so far is 10–14 pp per cell, while the
   observed effects are +66 and +84 pp.
2. **Scalability.** Trunk capacity is nearly exhausted at four tasks (§5).
3. **Whether the *adaptive* rule matters, or simply more energy.** A plain ε = 0.99 run would
   separate the two. The adaptive rule's advantage is that T1 is untouched and protection scales
   with what is actually new.
4. **Task order** (no `seq_hetero_reverse`).
5. **`gpm_grad_only`, SGP.**

---

## 9. Next steps (decision needed)

1. **Replicate the causal comparison on seed 1.**
   - Run `gpm` seed 1 (the baseline) and `gpm_ne90` seed 1, each about 5 h, paired to seq_ft seed
     1, which is done.
   - It needs the threshold decision in §7, and seed-1 identity checks (the variant against the
     plain GPM seed 1 at stages 0–1).
   - Overnight, both fit in one queue (about 10 h).
2. **Capacity-aware reporting** for the thesis: state ρ after every task next to retention.
   Optionally, a fifth task to show the limit.
3. **Optionally, a plain ε = 0.99 control** (about 5 h), to attribute the gain to the adaptive rule
   rather than to more protection in general.
