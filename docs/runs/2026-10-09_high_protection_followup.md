# C1 follow-up: three repetitions of Object's 50 initial states narrow D's CIs to ±6–11 pp. The verdicts are `protection_suffices` (seed 0), `unresolved` (seed 1), and `unresolved` with a significant adaptive advantage (seed 2); replication `mixed`

**Status: a pre-registered second-stage test of C1's primary cell (Object after Goal, `R[2][1]`).**
- It was decided after C1's `unresolved` verdict and is reported as such. C1's own verdict and
  records are unchanged.
- **Every gate passed on every seed.** Repetition 0 reproduced C1's 50 stored episodes exactly in
  all 6 cells (successes, step counts, seeds), so D from repetition 0 equals C1's D and CI exactly.

| Seed | D = e99 − adaptive, 50 states × 3 repetitions [95% CI over states] | C1 (one repetition) | Verdict | Flags |
|---|---|---|---|---|
| 0 | **+0.033 [−0.027, +0.100]** | +0.04 [−0.10, +0.18] | **`protection_suffices`** | none |
| 1 | −0.053 [−0.160, +0.053] | −0.02 [−0.18, +0.14] | `unresolved` | `within_margin_by_point_only` |
| 2 | **−0.100 [−0.193, −0.007]** | −0.08 [−0.22, +0.06] | `unresolved` | **`adaptive_advantage`**, `within_margin_by_point_only` |

**Replication:** `mixed`. Here the seeds really differ.
- `adaptive_advantage` holds on seed 2 only; no seed shows `e99_advantage`.
- Point estimates are +, −, −; their mean is −0.04 (descriptive).

**What this lets the thesis say** (C1's pre-written wording, applied per seed):
- **Seed 0:** "A flat 0.99 target reproduces the transition benefit within the pre-specified margin;
  adaptive targets are unnecessary for this cell under this protocol."
- **Seed 2:** adaptive's allocation keeps significantly more Object than this flat schedule, by
  between 0.7 and 19 pp. That is not beyond the 10 pp margin; the CI's upper bound is −0.007.
- **Seed 1:** neither.
- **Across seeds, no single conclusion replicates.** The difference the allocation makes at this
  cell is small and seed-dependent: every CI lies within [−0.19, +0.10].
  - That contrasts with the gain both arms share over plain GPM: G = +0.70 / +0.62 / +0.48 in C1, and
    adaptive − plain = +0.66 / +0.64 / +0.56.
  - So most of the Object effect comes from raising T2 protection to about 56% of the registry
    weights (C1). The allocation contributes, at most, a small seed-dependent part.
  - That supports C1's narrowed wording, with one addition: on one seed of three the new-energy
    allocation has a measurable edge.

**Date (local, CEST):** Fri 9 Oct 00:36:50 → 02:22:18 (1 h 45 min). 18 evaluations of 50 rollouts
(about 5.5–6.5 min each), reports about 5 s each, replication 1 s.

**Code:** `3442743`, clean at queue start; every report's `git_sha` is `34427435…`.
- The rule `configs/analysis/high_protection_followup.yaml`, revision 2, has SHA-256 `538d16c8…`. It
  was recorded before the smoke run and unchanged at commit.
- **Smoke** (seed 0, 3 episodes, repetitions 0–1, scratch): `QUEUE DONE ok`, every gate passed. The
  fresh-noise check saw adaptive's outcome change on 1 of 3 states between repetitions.
- **Tests:** 78 passed (the follow-up and C1 suites). No existing code changed; only new files were
  added.

**Revision 1 failed before producing anything (disclosed).**
- Revision 1 (`33fcc2e`, launched 23:21 Thu) asked for episodes 0–149 in one namespace. LIBERO has
  only **50 fixed initial states per task**, so all six evaluations stopped with `IndexError` at
  episode 50, before writing a file. The log is
  `results/logs/queue_20261008_232117_high_protection_followup`.
- Those logs show only episodes 0–49, which are C1's own. No new outcome existed when the rule was
  revised; the user chose the revision.
- The revision's header records all of this.

**Cite:**
- `results/high_protection_followup/replication.json` (SHA-256 `b43d18e6…`);
- `results/high_protection_followup/seed{0,1,2}/report.json` (`6dd53b77…`, `c95653bc…`,
  `81986bc8…`), each pinning its rule, C1 report, checkpoint and evaluation-file hashes;
- the evaluations `results/high_protection_followup/seed{S}/{e99,adaptive}.rep{0,1,2}.eval.json`;
- the queue log `results/logs/queue_20261009_003650_high_protection_followup/queue.log`.

---

## 1. Design (revision 2, as pre-registered)

- **What runs:** the existing stage-2 checkpoints of e99 and adaptive GPM, seeds 0–2, on Object
  only. No training.
- **The repetitions:** 3 repetitions of the task's 50 fixed initial states.
  - Repetition 0 uses the C1 namespace `seq_hetero__seq_ft__seed{S}`, which gives C1's own episodes.
  - Repetition r uses `…__rep{r}`: the same initial states with fresh sampling noise.
  - Both arms share every (state, repetition) seed.
- **The estimand:** success on LIBERO's fixed 50 initial states, averaged over sampling noise.
- **D and its CI:** D is the mean over states of each state's mean difference across repetitions.
  Its CI resamples the 50 states with their repetitions kept together. With one repetition this is
  exactly C1's paired CI, and that reduction is tested.
- **Gates:**
  - **Reproduction:** repetition 0 equals C1's stored cell exactly.
  - **Provenance:**
    - each checkpoint's `run_id` and stage;
    - each evaluation's namespace, stage and 50 rollouts;
    - the arms share seeds within a repetition;
    - the seeds differ across repetitions.
  - **The C1 report:** it is valid and under C1's rule SHA.
- **The verdict:** C1's own `classify` (margin 10 pp), using G and the checks from C1 and D from all
  repetitions.
- **Replication:** has the all-`unresolved` row that C1's rule lacked.

## 2. Validity (all passed, every seed)

| Gate | Seed 0 | Seed 1 | Seed 2 |
|---|---|---|---|
| Reproduction, e99 and adaptive (all 50 episodes of repetition 0) | equal | equal | equal |
| Checkpoints (`run_id`, stage 2); SHA-256 e99 / adaptive | `99c9c5d2…` / `f3f5202e…` | `4db8b84c…` / `01d200e0…` | `ae464c65…` / `d9b35f55…` |
| Evaluations (namespace, stage, 50 rollouts) × 6 | passed | passed | passed |
| Episode pairing within each repetition; fresh noise across repetitions | passed | passed | passed |
| C1 seed report (valid, rule `baf871db…`, same runs) | passed | passed | passed |

## 3. Results

| Seed | e99 success [CI] | adaptive success [CI] | D per repetition (0 / 1 / 2) | D, repetitions 1–2 only | States with the same outcome in every repetition, e99 / adaptive |
|---|---|---|---|---|---|
| 0 | 0.713 [0.613, 0.807] | 0.680 [0.573, 0.780] | +0.04 / +0.02 / +0.04 | +0.030 [−0.060, +0.120] | 60% / 66% |
| 1 | 0.593 [0.473, 0.707] | 0.647 [0.540, 0.747] | −0.02 / −0.18 / +0.04 | −0.070 [−0.200, +0.060] | 70% / 60% |
| 2 | 0.720 [0.633, 0.800] | 0.820 [0.740, 0.893] | −0.08 / −0.10 / −0.12 | −0.110 [−0.240, +0.010] | 50% / 66% |

**Precision.**
- The repetitions narrowed D's CI from ±14–16 pp (C1) to ±6.3 / ±10.7 / ±9.3 pp.
- Between 30% and 50% of initial states changed outcome across repetitions, so sampling noise, not
  only the state, decides many episodes. That is why the repetitions added information.
- The precision was not known in advance (the rule said so). Tripling independent rollouts would
  give about C1's width ÷ √3: ±8.1 / ±9.2 / ±8.1 pp. Seed 0 narrowed more than that; seeds 1–2
  narrowed less.
- **Seed 1 varies most across repetitions:** D is −0.18 in repetition 1, against −0.02 and +0.04 in
  the others.

## 4. Reading and limits

- **The estimand differs slightly from C1's.** It averages over sampling noise on the fixed 50
  states, where C1 had one noise draw per state. Repetition 0 is C1 itself.
- **This is a second-stage test, decided after seeing C1's `unresolved` verdicts,** and two
  `unresolved` verdicts remain.
  - The one non-inferiority (seed 0) and the one significant adaptive advantage (seed 2) are each
    single-seed findings.
  - The replication rule calls the whole `mixed`. The thesis should report the three seeds
    separately, as here, and not pool them.
- **The rollout CIs describe these trained policies** on the fixed initial-state set. Three seeds
  give directional information only, and the directions disagree.
- **For the thesis story (C1 record §4):** the narrowed wording stands. Raising T2 protection removes
  the Object forgetting. The new-energy allocation adds at most a small, seed-dependent amount at
  this cell, significant on one seed of three. Adaptive's consistent descriptive edge remains the
  **final** Object retention, at about 15 pp more capacity (C1).
