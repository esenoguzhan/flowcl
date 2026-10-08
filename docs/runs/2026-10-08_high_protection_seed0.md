# C1 high-protection control, seed 0: `unresolved` by the smallest possible margin. A flat 0.99 target reproduces the transition gain (+0.70) at matched T2 protection, and its CI lower bound against adaptive sits exactly on −0.10

**Status: the pre-registered control, development seed.**
- **Verdict:** `unresolved`, flag `within_margin_by_point_only`. Every gate passed: identity,
  energy and all 11 validity checks.
- **Seeds 1–2** are queued in the same run (`scripts/queue_high_protection.sh`). The replication
  follows them.

| Object after Goal, `R[2][1]` (seed 0) | Value | Paired difference [95% CI] |
|---|---:|---|
| plain GPM | 0.02 | |
| adaptive GPM (`ne90`) | 0.68 | adaptive − plain **+0.66 [+0.52, +0.78]** (reference gate: passed) |
| **e99** (flat 0.99 after T1) | **0.72** | **G** = e99 − plain **+0.70 [+0.56, +0.82]**: gain reproduced |
| | | **D** = e99 − adaptive **+0.04 [−0.10, +0.18]** |

- **Why `unresolved`:**
  - `protection_suffices` needs D's CI lower bound **strictly** above −0.10. It is exactly −0.10.
  - `flat_control_inferior` needs the CI upper bound below −0.10; it is +0.18.
  - The point estimate favours e99 (+4 pp), and neither direction is significant: there is no
    `adaptive_advantage` and no `e99_advantage` flag.
- **e99 retained Object almost completely across Goal:** 0.72 against its acquisition `R[1][1]` of
  0.78. Goal was learned (0.92 ≥ θ 0.85).
- **Protection after T2 came out exactly as predicted.**
  - e99 protected 55.85% of the registry weights; adaptive protected 55.95%.
  - That is −0.10 pp overall, +0.54 pp in the trunk and −1.06 pp in the decoder, all three equal to
    the pre-registered prediction.
  - 91 of 91 layers' T2 ranks equal the offline prediction.
  - So "approximately matched total protection" applies (`protection_matched: true`).

**Descriptive only (later cells compare different schedules and histories).**
- **Final retention:** by the end, e99 keeps less than adaptive.
  - Object: 0.70 against 0.84, −0.14 [−0.28, 0.00].
  - Goal: 0.78 against 0.94, −0.16 [−0.30, −0.02].
- **ACC** (final average success): e99 0.83, adaptive 0.92, plain 0.585.
- **Capacity:** e99 uses much less.
  - Protected fraction after T4: 64.5% against 79.9%.
  - Trunk median occupancy after T4: 0.885 against 0.988.
- **T4 plasticity is no better:** LIBERO-10 is 0.92 in both arms.

**Date (local, CEST):** Wed 7 Oct 23:17:52 → Thu 8 Oct 04:06:26.
- **The e99 run:** 4 h 48 min, including 2 h 56 min of training.
- **The sequence report:** 12 s. **The seed report:** 23 s.

**Code:** `6eb035b`, clean at queue start; the run's `git_sha` is `6eb035bc…`.
- The rule `configs/analysis/high_protection.yaml` has SHA-256 `baf871db…`; the method config
  `configs/method/gpm_e99.yaml` has `1fbaffde…`.
- Both SHAs were recorded before the smoke run and were unchanged at commit and launch.
- **Smoke run** (scratch, 22:14–23:10, seed 2, 200 steps, 1 episode, dirty tree): `QUEUE DONE ok`.
  - The smoke identity against the smoke plain run passed at stages 0–1 (0 of 656).
  - The recorded targets were 0.95 / 0.99 / 0.99 / 0.99 on all 91 layers.
  - All 11 validity checks passed.
  - 91 of 91 T2 ranks equalled the offline prediction.
- **CPU suite: 970 passed.**
  - **Disclosure:** the launch came before the full suite finished. The queue started at 23:17,
    when the suite was about 25% through; it finished at 23:59 with no failure.
  - The 93 tests covering the new code had already passed before the launch.

**Cite:**
- `results/high_protection/seed0/report.json` (SHA-256 `773b0fc1…`; it pins its own inputs);
- `results/gpm_seq_e99_seed0/report.json` (`ac8795b4…`);
- the run `results/seq_hetero__gpm_projected_adam_e99__seed0`;
- the queue log `results/logs/queue_20261007_231752_high_protection/queue.log`.

---

## 1. Design (as pre-registered)

**The question** (`docs/thesis_plan.md`, Workstream C, C1): does the adaptive allocation matter, or
only more protection?

**The arm.** `gpm_projected_adam_e99`: T1's memory is built to ε = 0.95, as in plain and adaptive
GPM. Every later task's memory is built to a flat 0.99 of its input energy (A4, `eps_later`).

**Why `R[2][1]` isolates the T2 allocation:**
- Object after Goal depends only on the memories after T1 (equal in all arms) and T2.
- e99's stage-1 weights and capture seeds equal adaptive's, so its T2 Gram is the same.
- Its T2 ranks were therefore predicted before the run from the adaptive run's stored residual
  spectra: e99 55.85% against adaptive 55.95% of the registry weights after T2.

**The rule:**
- **Gates, in order:** identity, energy, validity (of e99 and every comparator), premise,
  adaptive reference, Goal learned.
- **The verdict:**
  - `protection_suffices` if the gain over plain is reproduced and D's CI low > −0.10;
  - `flat_control_inferior` if D's CI high < −0.10;
  - otherwise `unresolved`.
- **Flags:** independent of the verdict.

## 2. Validity (all passed)

| Check | Result |
|---|---|
| Identity against plain GPM seed 0 (trained 23 Sep at `0184293`) | stage 0: 0 of 656 tensors differ; stage 1: 0 of 656; `memory_task0` equal; `memory_task1` differs (as designed). The runner's fail-fast check and the report's re-verification agree |
| T1 pairing against seq_ft | relative weight difference 0.0 |
| Energy | 364 layer-task entries (91 layers × tasks 0–3), exactly plain's registry. Recorded targets 0.95 at task 0 and 0.99 at tasks 1–3; every captured fraction ≥ target − 1e-6; all values finite; memory config `eps_later` 0.99 |
| Production | all four runs: curriculum, seed, recipe (30k steps, batch 64, lr 1e-4, AMP), 50 episodes, method config equal to its file, clean `git_sha`. **Excused, as registered:** seq_ft seed 0's `config.yaml` predates the `seed_namespace_run_id` field; it is the namespace run itself |
| Matrices | four complete 4 × 4 matrices, 50 rollouts per cell |
| Statistics | one fingerprint across the four runs and all 16 stage checkpoints |
| Episode pairing | identical per-episode seeds, e99 against seq_ft, plain and adaptive, in every cell |
| Sequence reports (e99, plain, adaptive) | all 8 required checks present and passed in each. For e99, the worst residual is 2.7% of the bound |
| Thresholds | e99's and adaptive's threshold blocks agree, and with the current seq_ft `result.json`: θ = 0.75 / 0.63 / 0.85 / 0.83 (registered) |

## 3. Results

**Retention matrices, seed 0.** Rows are after each stage; columns are Spatial / Object / Goal /
LIBERO-10; values in %:

| After | seq_ft | plain GPM | adaptive GPM | e99 |
|---|---|---|---|---|
| Spatial | 90 / 0 / 0 / 0 | 90 / 0 / 0 / 0 | 90 / 0 / 0 / 0 | 90 / 0 / 0 / 0 |
| Object | 0 / 78 / 0 / 0 | 94 / 78 / 0 / 0 | 94 / 78 / 0 / 0 | 94 / 78 / 0 / 0 |
| Goal | 0 / 0 / 100 / 0 | 90 / **2** / 94 / 0 | 96 / **68** / 94 / 0 | 96 / **72** / 92 / 0 |
| LIBERO-10 | 0 / 0 / 0 / 98 | 88 / 0 / 56 / 90 | 98 / 84 / 94 / 92 | 92 / 70 / 78 / 92 |

Rows 1–2 of the three GPM arms are the same policies (identity).

**Descriptive cells** (e99, paired against plain and adaptive):

| Cell | e99 [CI] | plain | adaptive | e99 − plain | e99 − adaptive |
|---|---|---:|---:|---|---|
| Goal learned `R[2][2]` | 0.92 [0.84, 0.98] | 0.94 | 0.94 | −0.02 [−0.12, +0.06] | −0.02 [−0.12, +0.06] |
| Object final `R[3][1]` | 0.70 [0.58, 0.82] | 0.00 | 0.84 | +0.70 [+0.58, +0.82] | −0.14 [−0.28, 0.00] |
| Goal final `R[3][2]` | 0.78 [0.66, 0.88] | 0.56 | 0.94 | +0.22 [+0.12, +0.34] | −0.16 [−0.30, −0.02] |
| T4 `R[3][3]` | 0.92 [0.84, 0.98] | 0.90 | 0.92 | +0.02 [−0.10, +0.14] | 0.00 [−0.10, +0.12] |
| Spatial final `R[3][0]` | 0.92 [0.84, 0.98] | 0.88 | 0.98 | +0.04 [−0.06, +0.14] | −0.06 [−0.14, +0.02] |

**ACC / NBT / AUC:**

| Arm | ACC | NBT | AUC |
|---|---:|---:|---:|
| seq_ft | 0.245 | 0.893 | 0.467 |
| plain | 0.585 | 0.393 | 0.741 |
| adaptive | 0.920 | −0.047 | 0.885 |
| e99 | 0.830 | 0.067 | 0.864 |

**Capacity:** the parameter-weighted protected fraction, in % (global / trunk / decoder), and the
trunk's median occupancy ρ:

| After | plain | adaptive | e99 | trunk ρ: plain / adaptive / e99 |
|---|---|---|---|---|
| T1 | 31.7 / 47.7 / 7.7 | 31.7 / 47.7 / 7.7 | 31.7 / 47.7 / 7.7 | 0.44 / 0.44 / 0.44 |
| T2 | 37.1 / 54.7 / 10.8 | 55.9 / 78.2 / 22.7 | **55.8 / 78.7 / 21.6** | 0.52 / 0.78 / 0.78 |
| T3 | 40.8 / 60.0 / 11.9 | 70.9 / 93.3 / 37.4 | 61.4 / 86.0 / 24.5 | 0.58 / 0.94 / 0.86 |
| T4 | 44.0 / 64.1 / 13.8 | 79.9 / 98.0 / 52.8 | 64.5 / 89.1 / 27.5 | 0.62 / 0.99 / 0.89 |

**The T2 allocation:**
- Adaptive's per-layer target exceeds 0.99 on 55% of trunk layers and 71% of decoder layers. The
  median excess is +0.0003 in the trunk and +0.0039 in the decoder.
- e99 added more trunk directions than adaptive (11 247 against 10 903) and fewer decoder
  directions (4 354 against 4 566).

## 4. Reading (the pre-written interpretation, applied)

- **The rule's verdict is `unresolved`. Neither of the pre-written conclusions applies on this
  seed.** The record does not say that a flat target suffices, nor that adaptive's allocation is
  better.
- **What the numbers show, with that verdict:**
  - At approximately matched total protection after T2 (−0.10 pp overall), a flat 0.99 target kept
    Object across Goal at least as well as adaptive's allocation by point estimate (+4 pp).
  - The CI's lower bound reaches exactly the −10 pp margin. Paired differences move in steps of
    0.02 (k/50), so the bound can land on the margin exactly; the strict inequality was
    pre-registered and is applied as written.
- **Later cells (descriptive; the arms then differ in strength and history):**
  - e99's lower T3/T4 targets (0.99 against adaptive's median 0.998/0.999) leave more Goal and
    Object forgetting at the end, and about 15 pp less of the model protected.
  - On this seed they buy no T4 plasticity: LIBERO-10 0.92 in both arms.
  - This bears on the capacity story (does the extra protection pay later?) but is not part of
    C1's claim.
- **Seed 0 is the development seed.** Seeds 1–2 and the replication decide the C1 claim.
