# E1, two-sided protection go/no-go (adaptive GPM seed 0): `fail`, near full rank

**Status: a pre-registered measurement, no training, one seed (the development seed).** The
verdict is **`fail`**, with the label **`near_full_rank`**.
- **The decisive quantity misses by far.** At the T4 pilot's start state, protecting
  `(U_1∪U_2∪U_3) ⊗ M` would still cover **0.861** of the trunk's weight directions,
  parameter-weighted. The threshold is ≤ 0.50; GPM covers 0.933.
- **The reason is the output side itself.** The trunk's output sensitivity is near full rank: the
  union `U` spans 0.92 of the trunk's output dimensions (parameter-weighted) after T3, and 0.50
  already after T1 alone.
- **Every check passed, bitwise.** On all four tasks the capture rebuilt GPM's memory exactly, and
  the grad-enabled pass reproduced GPM's input Gram exactly (§2).
- **The verdict does not depend on the estimator or the rule.** The two Hutchinson probes alone
  give 0.834 each, and every reported rule × ε_out cell is above 0.5 at T3 (the most lenient,
  fixed ε_out = 0.90, gives 0.598).

**In short:**
- Two-sided protection frees **7.2 pp** of the trunk at T4's start (0.933 → 0.861) and **2.5 pp**
  at T5's start (0.980 → 0.954). The trunk's free fraction roughly doubles (6.7% → 13.9%; 2.0% →
  4.6%), but stays small.
- In the decoder, which is not saturated, it would free much more (0.374 → 0.128 after T3).
- The forecast (labelled, linear) has both rules covering the whole trunk by T6.
- **For D2** (`docs/thesis_plan.md`): angle A, which needed E1 and both pilots to pass, is off.
  - The `near_full_rank` label matches the D2 row for **angle B**: `u` restricted to critical
    outputs, but only if Track 2 identifies them.
  - Otherwise the row is the **minimum thesis**: E3 without a new method.
- **This establishes potential capacity only, here its absence.** It says nothing about whether
  two-sided protection would retain or learn, since nothing was trained.

**Date (local, CEST):** Fri 2 Oct 2026, 01:20:41 → 01:52:48.
- captures 01:20 → 01:47 (T1 5 min, T2 6 min, T3 6 min, T4 10 min);
- summary 01:47 → 01:52.

**Code:** `784ece1` (`784ece12…`), clean at queue start. The rule
`configs/analysis/twosided_gonogo.yaml` (SHA-256 `59dad7d7…`) was committed with the
implementation at 00:52:24, 28 min before the queue.

**Cite:** `results/twosided_gonogo/summary.json`. It holds the verdict, the validity of every
capture, all tables (rule × ε_out × stage × half/group), the stability, the forecast, and the
SHAs of the decisive bases.

---

## 1. Design (as pre-registered)

**The question.** GPM protects `d_out × k_in` weight directions per layer. Two-sided protection
protects only `k_out × k_in`, where `U` spans the output directions the policy's velocity is
sensitive to. Does old-task output sensitivity occupy few enough directions to free useful
capacity in the saturated trunk?

**The source.** Adaptive GPM seed 0 (`seq_hetero__gpm_projected_adam_ne90__seed0`): `stage0–3.pt`
and `memory_task0–3.pt`, SHA-pinned (the same files as the T5 sweep).

**The capture,** per task index i (`stage{i}.pt` on task T(i+1)'s whole dataset; the pairing
GPM's `memory_task{i}` was built from):
1. **The production pass.** GPM's own forward-only capture (`gate2.capture_task_grams`, GPM's
   seeds) gives the input Gram `K_i` and the reachability views.
2. **The output pass.** The same data, in the same order, with the same s/noise stream, but
   grad-enabled. It runs fp32, with autocast off, in `eval()`.
   - **Hutchinson, R = 2 probes (A, B):** `u ~ N(0, I)` over the velocity, masked by
     `action_mask`, from its own seed.
   - `g = ∂⟨u, v⟩/∂y` at every registry layer's output, accumulated as `Σ g gᵀ` (`d_out × d_out`,
     float64). It uses 16 positions per sample, drawn among the primary view's positions whose
     gradient is not exactly zero.
   - `E_u[g gᵀ] = JᵀJ`, the output factor of the Gauss-Newton matrix.
3. **The text cache** is the runner's: T1's instruction under AMP, the later ones in fp32.

**The U rule** (chosen before any run). `U` is accumulated task by task with `extend_basis`, the
call that builds `M`. The decisive rule is **M's own: adaptive, ε_out = 0.95, f = 0.9**, on
`G_A + G_B`. The fixed-ε rule is reported as a comparison.

**The decisive quantity** is the parameter-weighted trunk fraction
`Σ k_out·k_in / Σ d_out·d_in` over the 49 trunk weights. It uses `U = U_1∪U_2∪U_3` with
`M = memory_task2`, the start of E2's T4 pilot.
- **Verdict, first match:** `invalid`, `incomplete`, `pass` (≤ 0.50), then `fail`.
- **Labels:**
  - `estimator_sensitive`: A-only and B-only fall on opposite sides of 0.50;
  - `near_full_rank`: trunk `k_out/d_out`, parameter-weighted, is ≥ 0.9.

## 2. Validity (all passed)

| Check | T1 | T2 | T3 | T4 |
|---|---|---|---|---|
| Memory replay (`memory_task{i}` rebuilt) | pass, **bitwise** | pass, **bitwise** | pass, **bitwise** | pass, **bitwise** |
| Forward identity (grad-enabled `K̃` vs `K`; gate 1e-5) | **bitwise** | **bitwise** | **bitwise** | **bitwise** |
| Rows per output dimension, worst layer (≥ 10) | 11.5 | 14.2 | 13.7 | 26.0 |
| Output Grams finite; weights and `.grad` unchanged | yes | yes | yes | yes |
| Text cache | AMP | fp32 | fp32 | fp32 |
| Dataset samples / batches | 5,882 / 92 | 7,281 / 114 | 7,027 / 110 | 13,298 / 208 |

- **The worst sample ratio** is `trunk.state_projection`, which has one row per sample.
- **Every capture's `git_sha`** is `784ece1`, clean. The queue log shows 0 changes at start.

## 3. Results

**The decisive chain** (adaptive, ε_out = 0.95). Parameter-weighted fractions after each task:

| After | Trunk two-sided | Trunk GPM | Trunk ρ_out | Decoder two-sided | Decoder GPM | All two-sided | All GPM |
|---|---:|---:|---:|---:|---:|---:|---:|
| T1 | 0.234 | 0.477 | 0.496 | 0.006 | 0.077 | 0.143 | 0.317 |
| T2 | 0.612 | 0.782 | 0.785 | 0.046 | 0.227 | 0.386 | 0.560 |
| **T3 (decisive)** | **0.861** | **0.933** | **0.922** | 0.128 | 0.374 | 0.568 | 0.709 |
| T4 | 0.954 | 0.980 | 0.974 | 0.245 | 0.528 | 0.671 | 0.799 |

**The trunk's output side fills like its input side.**
- ρ_out (0.50 → 0.78 → 0.92 → 0.97) tracks GPM's input occupancy (0.48 → 0.78 → 0.93 → 0.98).
- Examples of `k_out` over T1–T4:
  - `trunk.blocks.0.mlp.fc1`: 1278 → 1773 → 1982 → 2037 of 2048;
  - `trunk.blocks.7.mlp.fc1`: 969 → 1624 → 1919 → 2023;
  - `trunk.blocks.0.attn.q_proj`: 200 → 327 → 438 → 493 of 512.

**By group, at the decisive point:**

| Group | Two-sided | GPM | ρ_out |
|---|---:|---:|---:|
| trunk_attn (32) | 0.816 | 0.906 | 0.898 |
| trunk_mlp (16) | 0.883 | 0.946 | 0.935 |
| trunk_input (1) | 0.432 | 1.000 | 0.432 |
| decoder_self_attn (16) | 0.099 | 0.254 | 0.373 |
| decoder_cross_attn (16) | 0.269 | 0.691 | 0.456 |
| decoder_mlp (8) | 0.072 | 0.274 | 0.242 |
| decoder_output (1) | 0.631 | 0.631 | 1.000 (7 of 7 outputs) |

**Trunk two-sided fraction, rule × ε_out** (reported only; the decisive cell is in bold):

| Rule | ε_out | T1 | T2 | T3 | T4 |
|---|---|---:|---:|---:|---:|
| adaptive | 0.90 | 0.163 | 0.545 | 0.825 | 0.942 |
| adaptive | 0.95 | 0.234 | 0.612 | **0.861** | 0.954 |
| adaptive | 0.99 | 0.370 | 0.729 | 0.915 | 0.975 |
| fixed | 0.90 | 0.163 | 0.422 | 0.598 | 0.691 |
| fixed | 0.95 | 0.234 | 0.551 | 0.741 | 0.824 |
| fixed | 0.99 | 0.370 | 0.720 | 0.892 | 0.950 |

- Even the fixed rule's most lenient cell (ε_out = 0.90) is above 0.5 at T3.
- The fixed rule is the one this project showed under-protects later tasks (its decoder fraction
  barely grows: 0.034 at T3 at ε_out 0.90).

**Stability** (decisive rule; reported):

| Gram | T1 | T2 | T3 | T4 |
|---|---:|---:|---:|---:|
| A only | 0.210 | 0.572 | **0.834** | 0.946 |
| B only | 0.209 | 0.572 | **0.834** | 0.946 |
| A + B (decisive) | 0.234 | 0.612 | **0.861** | 0.954 |

- **No `estimator_sensitive` label.** The two probes agree to the third decimal, and both lie far
  above 0.50.
- **The span overlap `‖U_AᵀU_B‖²/max k`** at T3 has a median of 0.93 in the trunk and 0.93 in the
  decoder. The lowest is `trunk.state_projection` (0.75).
- **The estimate rose with more probes** (0.834 → 0.861). That is consistent with finite-sample
  concentration of the spectrum, which under-counts the directions needed. It is not established
  here. Either way, more probes would not be expected to bring the fraction toward 0.5.

**The forecast** (linear per layer and side, from T2 → T4; **a forecast, not a measurement**):

| Task | Trunk two-sided / GPM | Decoder two-sided / GPM | All two-sided / GPM |
|---|---|---|---|
| T5 | 0.992 / 0.997 | 0.382 / 0.658 | 0.748 / 0.861 |
| T6 | 1.000 / 1.000 | 0.512 / 0.767 | 0.805 / 0.907 |
| T7 | 1.000 / 1.000 | 0.625 / 0.842 | 0.850 / 0.937 |
| T8 | 1.000 / 1.000 | 0.709 / 0.880 | 0.884 / 0.952 |

## 4. The pre-registered judgement

| Quantity | Value | Rule | Result |
|---|---|---|---|
| Trunk two-sided fraction, T3, adaptive 0.95 | **0.861** | ≤ 0.50 | ✗ |
| Trunk ρ_out (parameter-weighted), same point | 0.922 | ≥ 0.90 → `near_full_rank` | label set |
| A-only / B-only on opposite sides of 0.50? | 0.834 / 0.834 | → `estimator_sensitive` | not set |

**Verdict: `fail`** (summary exit code 0: a valid finding), labels `[near_full_rank]`.

## 5. What this does and does not show

**It shows,** for adaptive GPM seed 0 under first-order (Gauss-Newton) output sensitivity of the
whole velocity chunk:
- the trunk's old-task output sensitivity is spread over nearly all of its output directions;
- protecting `(∪U) ⊗ (∪M)` instead of `I ⊗ (∪M)` frees only 7 pp of the trunk at T4's start and
  2.5 pp at T5's start;
- the decoder, which is not saturated, has a lower-rank output side (ρ_out 0.33 after T3).

**It does not show:**
- **that two-sided protection would retain or learn worse.** Nothing was trained. E1 measures
  capacity, and here finds too little of it to justify the pilots.
- **anything about a narrower sensitivity target.** `u` spans every velocity component, all 16
  chunk steps and the training distribution of s. A target restricted to the outputs that decide
  success (angle B) could have a lower rank. That needs Track 2's critical outputs.
- **the exact union `∪_t(U_t ⊗ M_t)`.** It is a subset of the conservative `(∪U) ⊗ (∪M)` measured
  here and was not computed (`docs/thesis_plan.md`, Risks).
- **other seeds.** Seed 0 only.

## 6. Disclosures

1. **Rule timing.** The rule and the code were committed in `784ece1` at 00:52:24. The queue
   started at 01:20:41 from a clean tree, after the CPU suite finished (777 passed; the 17 new E1
   tests passed separately).
2. **The smoke runs** went into the session's scratch directory, before the commit, and were
   discarded. The rule's SHA-256 (`59dad7d7…`) was identical before and after both.
   - **E1:** T1, 3 batches, then the summary.
     - The outcome was the one defined in advance: `incomplete`, exit 4.
     - The forward identity was enforced even in the smoke run, which is stricter than the plan's
       "unverified". It was bitwise.
     - Only throughput was read from it (about 1.4 s per batch for the production pass and 3.5 s
       per batch for the output pass, under CPU contention).
   - **Gate 0:** 200 steps and 1 episode on all six T6–T8 tasks. `results/gate0/gate0.json` and the
     existing single-task runs were unchanged afterwards (SHA and mtime).
3. **The comparator.** `docs/thesis_plan.md` quotes "about 0.94 for GPM". That is the median over
   trunk layers of `k_in/d_in` (0.941). This rule compares parameter-weighted fractions on both
   sides, so GPM is 0.933 here, which the config header states.
4. **The U rule** (adaptive, M's own) was chosen during planning, before any capture. The plan's
   text, read literally, allowed the fixed rule. That rule is reported in full, and it also
   fails (0.741 at ε_out 0.95).
5. **Eligible positions.** Output positions whose gradient is exactly zero were not eligible for
   the per-sample subsample. Examples are the last trunk block's non-context positions. This is
   stated in the rule header before the run; it changes no Gram, only how many informative rows
   are kept.
6. **T4 (U₄ with `memory_task3`)** is reported only. The plan's E1 named U₁–U₃, and U₄ is the
   start of the T5 pilot.
7. **Development seed.** Seed 0 is the development seed. No other seed was measured.

## 7. Consequences

- **E2 (angle A) is not triggered.** D2's angle A needed E1 and both pilots to pass. The T4/T5
  two-sided pilots and the T4 `free` arm are not run.
- **D2 (end of W5) now reads from two rows:**
  - **Angle B** applies only if Track 2 (optional; freeze by 30 Oct) identifies the outputs that
    decide success. Then `u` is restricted to them, and E1 is rerun (no training, about 35 min of
    GPU), followed by E2 if it passes.
  - **Otherwise the minimum thesis.** E3 runs the capacity frontier and the reference methods
    without a new method.
- **For the thesis story** ("which directions actually need protection?"):
  - under whole-velocity sensitivity, the answer for the trunk is "almost all of them";
  - the saturation the adaptive rule causes is not an artifact of protecting insensitive output
    directions;
  - this is reported as a negative result with its registered rule.
- **The decisive bases are kept** in `results/twosided_gonogo/bases/`. Angle B or E3's capacity
  reporting can compare against them without new captures.

## 8. Artifact map

```
results/twosided_gonogo/summary.json                 # verdict, labels, validity per capture, tables
                                                     # (rule x eps_out x stage x half/group), stability,
                                                     # forecast, bases SHAs
results/twosided_gonogo/capture/task{0..3}/          # grams_A.pt, grams_B.pt (output Grams, float64),
                                                     # capture.json (seeds, cache state, reachability,
                                                     # per-layer dims/views/counts/identity, checks)
results/twosided_gonogo/bases/U_adaptive_e0.95_task{0..3}.pt   # the decisive unions, (d_out, k_out)
results/logs/queue_20261002_012041_twosided_gonogo/  # queue.log and one log per step
configs/analysis/twosided_gonogo.yaml (784ece1; SHA-256 59dad7d7...)
flowcl/analysis/sensitivity.py, flowcl/experiments/twosided_gonogo.py, scripts/twosided_gonogo.py,
scripts/queue_twosided_gonogo.sh, tests/test_twosided_gonogo.py
```

**Inputs** (read only, SHA-256-pinned in the config):
`results/seq_hetero__gpm_projected_adam_ne90__seed0/`: `stage0–3.pt`, `memory_task0–3.pt`,
`result.json`.
