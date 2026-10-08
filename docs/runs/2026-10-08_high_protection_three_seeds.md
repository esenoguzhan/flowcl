# C1 high-protection control, three seeds: `unresolved` on every seed. A flat 0.99 target at matched T2 protection reproduces the Object transition gain on all three, is never distinguishable from adaptive GPM (D = +0.04 / −0.02 / −0.08), and its non-inferiority at 10 pp is never shown

**Status: the pre-registered control, complete.**
- **Every seed:** `unresolved`, flag `within_margin_by_point_only`. Every gate passed on every seed.
- **The replication label is `mixed`.** This is a gap in the registered rule, disclosed here and not
  corrected. The rule had rows for all `protection_suffices`, all `flat_control_inferior` and all
  `plasticity_failure`, but none for all `unresolved`, so the outcome fell to "otherwise → mixed".
  Its printed text ("the seeds' verdicts differ") is wrong for this case: the three verdicts are
  identical.
- Seed 0's own record is `2026-10-08_high_protection_seed0.md`.

Object after Goal, `R[2][1]`, rollout-paired 95% CIs:

| Seed | plain | adaptive | e99 | **G** = e99 − plain | **D** = e99 − adaptive | adaptive − plain (reference gate) |
|---|---:|---:|---:|---|---|---|
| 0 | 0.02 | 0.68 | 0.72 | **+0.70 [+0.56, +0.82]** | +0.04 [−0.10, +0.18] | +0.66 [+0.52, +0.78] |
| 1 | 0.02 | 0.66 | 0.64 | **+0.62 [+0.48, +0.76]** | −0.02 [−0.18, +0.14] | +0.64 [+0.50, +0.78] |
| 2 | 0.22 | 0.78 | 0.70 | **+0.48 [+0.30, +0.66]** | −0.08 [−0.22, +0.06] | +0.56 [+0.42, +0.70] |

- **The flat control reproduces the transition gain on every seed.** G meets the adaptive rule's
  own criterion (≥ +0.20, CI low > 0) three times out of three. Goal was learned under the 0.99
  memory every time (0.92 / 1.00 / 1.00).
- **It is never distinguishable from adaptive.**
  - D's sign is +, −, −; the mean of the three point estimates is −0.02 (descriptive).
  - No CI excludes 0, so neither `adaptive_advantage` nor `e99_advantage` is set on any seed.
- **Non-inferiority within 10 pp is never shown.** D's CI lower bound is −0.10 / −0.18 / −0.22,
  never strictly above −0.10.
  - At 50 rollouts the paired CIs are ±14–16 pp wide, so this design could only show it if e99
    were ahead on the point estimate.
  - Inferiority beyond the margin is never shown either: the CI upper bounds are +0.18 / +0.14 /
    +0.06.
- **Total protection after T2 was matched on every seed, exactly as predicted before the runs:**

| Seed | e99 | adaptive | e99 − adaptive (global / trunk / decoder, pp) | T2 ranks equal to the prediction |
|---|---:|---:|---|---:|
| 0 | 55.85% | 55.95% | −0.10 / +0.54 / −1.06 | 91 / 91 |
| 1 | 57.06% | 57.20% | −0.15 / +0.68 / −1.39 | 91 / 91 |
| 2 | 56.31% | 56.19% | +0.12 / +0.79 / −0.88 | 91 / 91 |

  `protection_matched` holds on all three (tolerance 0.5 pp). The comparison is therefore between
  two allocations at approximately matched total protection.

**What the pre-registered rule lets this record say:**
- **Neither pre-written conclusion is drawn.**
  - Not "adaptive targets are unnecessary for this cell": that needs non-inferiority.
  - Not "adaptive's allocation is better than the flat schedule": that needs D's CI below 0.
- **What it can say, worded within the result:**
  - At approximately matched total protection after T2, a flat 0.99 target reproduced the Object
    transition benefit on all three seeds.
  - Its retention of Object across Goal could not be distinguished from the adaptive allocation's.
  - The experiment, at 50 rollouts per cell, cannot resolve a difference of 10 pp in either
    direction.

**Descriptive only (never part of the claim).** After T2 the arms differ in protection strength
(0.99 against adaptive's median 0.998 / 0.999 at T3 / T4) and in history.
- **Object at the end:** e99 keeps less on every seed, with CIs excluding 0 on seeds 1–2.
- **Capacity:** e99 protects about 15 pp less of the model after T4 on every seed.
- **Final Goal, T4 plasticity and ACC:** no consistent direction (§3).

**Date (local, CEST):** Wed 7 Oct 23:17:52 → Thu 8 Oct 13:39:01 (14 h 21 min); `QUEUE DONE ok`.

| Seed | e99 run | Total, run + reports |
|---|---|---|
| 0 | 4.80 h | 23:17:52 → 04:06:26 |
| 1 | 4.72 h | 04:06:26 → 08:50:36 |
| 2 | 4.79 h | 08:50:36 → 13:39:00 |

- Training took 2.9 h per run. Each sequence report took about 12 s and each seed report about
  23 s; the replication took 1 s.
- **The machine rebooted at 21:09 on Thu 8 Oct,** 7.5 h after the queue ended. It affected nothing.

**Code:** `6eb035b`, clean at queue start. Every run's and report's `git_sha` is `6eb035bc…`.
- The rule `configs/analysis/high_protection.yaml` has SHA-256 `baf871db…`; every seed report
  carries it, and the replication checked it.
- The smoke run, the CPU suite (970 passed) and the launch-before-suite disclosure are in the seed-0
  record.

**Cite:**
- `results/high_protection/replication.json` (SHA-256 `5105f0cb…`);
- `results/high_protection/seed{0,1,2}/report.json` (`773b0fc1…`, `c578310a…`, `cb5b2490…`);
- `results/gpm_seq_e99_seed{0,1,2}/report.json` (`ac8795b4…`, `dc50fde3…`, `ab708dc1…`);
- the runs `results/seq_hetero__gpm_projected_adam_e99__seed{0,1,2}`;
- the queue log `results/logs/queue_20261007_231752_high_protection/queue.log`.

---

## 1. Validity, seeds 1–2 (all passed; seed 0 is in its record)

| Check | Seed 1 | Seed 2 |
|---|---|---|
| Identity against plain GPM (stages 0, 1) | 0 / 0 of 656 tensors differ; `memory_task0` equal | 0 / 0 of 656; `memory_task0` equal |
| T1 pairing against seq_ft | relative weight difference 0.0 | 0.0 |
| Energy (364 layer-task entries) | targets 0.95 / 0.99 / 0.99 / 0.99 on every layer, every capture ≥ target | the same |
| Production, all four runs | passed | passed |
| Matrices, statistics, episode pairing | complete; one fingerprint; identical seeds | the same |
| Sequence reports (e99, plain, adaptive): 8 required checks each | all passed; e99's worst residual is 2.7% of the bound | all passed; 2.7% |
| Thresholds | registered 0.81 / 0.75 / 0.85 / 0.83 | derived 0.81 / 0.77 / 0.83 / 0.75 |
| Premise (plain forgot Object) | 0.02 < 0.75 | 0.22 < 0.77 |

## 2. Retention matrices, seeds 1–2

Rows are after each stage; columns are Spatial / Object / Goal / LIBERO-10; values in %. Rows 1–2
of the three GPM arms are the same policies (identity).

| Seed | After | plain GPM | adaptive GPM | e99 |
|---|---|---|---|---|
| 1 | Spatial | 96 / 0 / 0 / 0 | 96 / 0 / 0 / 0 | 96 / 0 / 0 / 0 |
| 1 | Object | 98 / 78 / 0 / 0 | 98 / 78 / 0 / 0 | 98 / 78 / 0 / 0 |
| 1 | Goal | 86 / **2** / 98 / 0 | 98 / **66** / 100 / 0 | 98 / **64** / 100 / 0 |
| 1 | LIBERO-10 | 92 / 0 / 20 / 96 | 94 / 82 / 72 / 74 | 98 / 66 / 94 / 76 |
| 2 | Spatial | 96 / 0 / 0 / 0 | 96 / 0 / 0 / 0 | 96 / 0 / 0 / 0 |
| 2 | Object | 94 / 86 / 0 / 0 | 94 / 86 / 0 / 0 | 94 / 86 / 0 / 0 |
| 2 | Goal | 98 / **22** / 98 / 0 | 96 / **78** / 100 / 0 | 98 / **70** / 100 / 0 |
| 2 | LIBERO-10 | 96 / 24 / 96 / 94 | 94 / 78 / 98 / 80 | 100 / 56 / 98 / 90 |

## 3. Descriptive, all three seeds

**e99 − adaptive** (rollout-paired):

| Cell | Seed 0 | Seed 1 | Seed 2 |
|---|---|---|---|
| Object final `R[3][1]` | −0.14 [−0.28, 0.00] | −0.16 [−0.30, −0.02] | −0.22 [−0.40, −0.06] |
| Goal final `R[3][2]` | −0.16 [−0.30, −0.02] | +0.22 [+0.12, +0.34] | 0.00 [−0.06, +0.06] |
| T4 `R[3][3]` | 0.00 [−0.10, +0.12] | +0.02 [−0.12, +0.16] | +0.10 [−0.04, +0.24] |
| Spatial final `R[3][0]` | −0.06 [−0.14, +0.02] | +0.04 [−0.04, +0.12] | +0.06 [0.00, +0.14] |

**Summary metrics:**

| Seed | ACC: plain / adaptive / e99 | NBT: adaptive / e99 |
|---|---|---|
| 0 | 0.585 / 0.920 / 0.830 | −0.047 / 0.067 |
| 1 | 0.520 / 0.805 / 0.835 | 0.087 / 0.053 |
| 2 | 0.775 / 0.875 / 0.860 | 0.040 / 0.093 |

**Capacity after T4:**

| Seed | Protected fraction, % (global / trunk / decoder): plain | adaptive | e99 | Trunk median ρ: adaptive / e99 |
|---|---|---|---|---|
| 0 | 44.0 / 64.1 / 13.8 | 79.9 / 98.0 / 52.8 | 64.5 / 89.1 / 27.5 | 0.988 / 0.885 |
| 1 | 44.6 / 64.3 / 15.1 | 81.9 / 98.0 / 57.7 | 66.0 / 89.2 / 31.2 | 0.988 / 0.895 |
| 2 | 43.9 / 63.9 / 13.9 | 80.2 / 98.0 / 53.6 | 65.2 / 89.0 / 29.6 | 0.988 / 0.893 |

**What these show:**
- **The final Object cell is the one consistent difference.** Adaptive's higher T3/T4 targets keep
  more Object at the end: same sign on all three seeds (directional replication), with CIs
  excluding 0 on two.
- **Its price is about 15 pp more of the model protected.** That buys no consistent final Goal or
  T4 plasticity gain for either arm.
  - Adaptive's T4 cost against plain (seeds 1–2: 0.74 and 0.80 against 0.96 and 0.94) is not
    relieved on seed 1 (e99 0.76).
  - It is partly relieved on seed 2 (e99 0.90).
- **The later cells compare different schedules and histories,** so they are not a test of the
  allocation.

## 4. Implications for the thesis story (for discussion; nothing here is a registered claim)

- **"Adaptive allocation fixes retention" needs narrower wording.** The Object transition effect is
  reproduced by a flat 0.99 T2 target, which protects about the same 56% of the registry weights as
  adaptive's T2 memory (one other allocation tested, not any allocation).
  - Whether that energy is allocated per layer by new energy or by a flat total target made no
    resolvable difference at this precision.
  - What C1 supports is "under-protection at the fixed 0.95 target causes the forgetting; raising
    the T2 target removes it". The new-energy rule is one way to raise it, not shown to be a
    better one.
- **The adaptive rule's distinct, consistent descriptive effect is at the end of the sequence:**
  more final Object retention at about 15 pp more capacity. It belongs in the capacity frontier
  (E3), not in C1's claim.
- **Resolving ±10 pp at `R[2][1]` would need more rollouts.** About 150 per arm gives paired CIs of
  roughly ±8–9 pp.
  - It would be an evaluation-only follow-up on the existing stage-2 checkpoints of e99 and
    adaptive, 3 seeds, about 1–2 GPU h.
  - It would need its own pre-registered rule before any rollout. It is not decided here.
