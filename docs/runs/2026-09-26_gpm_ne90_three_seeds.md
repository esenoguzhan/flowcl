# Seed 2 and the three-seed result — adaptive GPM (`gpm_projected_adam_ne90`) against plain GPM

**Status:** seed 2 gives **`durable_support`**. The pre-registered replication over seeds 0, 1 and 2
is **replicated**, meaning all three seeds show causal transition support. It is not *durably*
replicated, because seed 1 was `transition_support`.

The Object effect is robust across all three seeds. T4 plasticity has a cost:
- it is consistent at the loss level;
- it is significant in the rollouts in 2 of 3 seeds;
- it crossed the pre-registered threshold in 1 of 3.

**Dates (local, CEST):**
- the first queue ran 25 Sep 23:28 → 26 Sep 10:23, when the machine hard-stopped;
- it was resumed with `--from-step 4` at 13:28 and was done at 18:26.

**Code:** `b150a6d` for seq_ft and plain GPM seed 2; `9139d99` for the resumed steps, which only
added `--from-step`. Both are clean.

**Runs** (under `results/`):
- `seq_hetero__seq_ft__seed2`;
- `seq_hetero__gpm_projected_adam__seed2`;
- `seq_hetero__gpm_projected_adam_ne90__seed2`. The partial pre-crash attempt is kept as
  `…_ne90__seed2_crashed_20260926`: config only, no checkpoints, not used.

**Companion records:**
- `2026-09-25_gpm_ne90_seq_hetero.md` (seed 0);
- `2026-09-25_gpm_ne90_seed1_replication.md` (seed 1; its §6 holds the decided roadmap);
- the tag `stageA-gpm-adaptive-2seeds` (the frozen two-seed state).

Cite these reports:
- `results/adaptive_gpm_seed2/report.json` (seed-2 verdict);
- `results/adaptive_gpm/replication.json` (three-seed replication);
- `results/gpm_seq_seed2/report.json` and `results/gpm_seq_ne90_seed2/report.json` (sequence reports);
- `results/forgetting_diag_seed2/report.json` and `results/forgetting_diag_ne90_seed2/report.json`
  (diagnostics).

---

## 1. Interruption and provenance

**The interruption.**
- The machine went down at 10:23 on 26 Sep, with no shutdown record, and rebooted at 10:34. This is
  the second hard stop in three days; the first was on 23 Sep at 16:04.
- The adaptive run was at T1 step 26 951 of 30 000 and had saved no checkpoint.
- Steps 0–3 (seq_ft, plain GPM, and plain GPM's reports) had finished and were reused.
- The queue was resumed with the new, tested `--from-step 4`, which checks each earlier step's output
  on disk. The adaptive run restarted from scratch.

| Check | Result |
|---|---|
| Seed-2 thresholds | **derived** under the pre-registered explicit entry: seq_ft seed 2 (96 / 92 / 98 / 90) − 0.15 = **81 / 77 / 83 / 75** |
| T1 pairing against seq_ft seed 2, both runs | relative weight difference **0.0** |
| Adaptive identity against plain GPM seed 2 | **0 of 656** tensors differ at stage 0 and at stage 1 |
| Threshold blocks | the two sequence reports agree on reference, hash, thresholds, margin, namespace and task order. The adaptive report consumed them. |
| Sequence-report provenance, both runs | all pass (clean SHA, seed namespace, T1 pairing, occupancy, `memory_chained`, residuals ≤ 2.8% of the bound, artifact hashes, frozen tensors) |
| Diagnostics instrument check | not applicable, as pre-registered: the pilot's references exist only for seed 0 |

---

## 2. Seed-2 results

Success rate (%), 50 paired episodes per cell.

**Plain GPM, seed 2:**

| after \ on | Spatial | Object | Goal | LIBERO-10 |
|---|---:|---:|---:|---:|
| Spatial | 96 | 0 | 0 | 0 |
| Object | 94 | **86** | 0 | 0 |
| Goal | 98 | **22** | **98** | 0 |
| LIBERO-10 | 96 | **24** | 96 | **94** |

**Adaptive GPM, seed 2:**

| after \ on | Spatial | Object | Goal | LIBERO-10 |
|---|---:|---:|---:|---:|
| Spatial | 96 | 0 | 0 | 0 |
| Object | 94 | **86** | 0 | 0 |
| Goal | 96 | **78** | **100** | 0 |
| LIBERO-10 | 94 | **78** | 98 | **80** |

**How the plain baseline forgets differently on seed 2:**
- Object falls from 86% to 22%, not to 2% as in seeds 0 and 1. That is still large, and the premise
  holds: 22% is below 77%.
- **Goal is not forgotten** (98 → 96%). In seeds 0 and 1 it fell to 56% and 20%.
- The diagnostics are again case D in both comparisons, computed independently.

**Seed-2 verdict: `durable_support`.** Every check passes:

| Check | Result |
|---|---|
| identity | pass |
| per-layer energy | pass: 269 applicable, 0 failures, minimum share 0.90001 |
| premise | pass |
| interference ratio | trunk 0.67, decoder 0.46 |
| transition gain | **+56 pp [+42, +70]** |
| T3 plasticity | 100% |
| durable Object | 78% ≥ 77% |
| T4 plasticity | 80% ≥ 75% |

T4 is **borderline**: its CI [68, 90] straddles 75%. It is also **significantly below plain GPM:
−14 pp [−26, −2]**.

---

## 3. Across three seeds

| | seed 0 | seed 1 | seed 2 |
|---|---:|---:|---:|
| Plain GPM: Object learned → after Goal → end | 78 → 2 → 0 | 78 → 2 → 0 | 86 → 22 → 24 |
| **Object transition gain** (adaptive − plain, after Goal) | **+66 [+52, +78]** | **+64 [+50, +78]** | **+56 [+42, +70]** |
| Object final gain | +84 | +82 | +54 |
| Goal final gain | +38 | +52 | +2 (the baseline kept Goal) |
| T3 plasticity, adaptive (Goal learned) | 94% | 100% | 100% |
| **T4 plasticity, adaptive vs plain** (LIBERO-10) | 92 vs 90: **+2 [−10, +14]** | 74 vs 96: **−22 [−36, −8]** | 80 vs 94: **−14 [−26, −2]** |
| T4 threshold (per seed) | 83% (pass) | 83% (**fail**) | 75% (pass, borderline) |
| T4 last-50 training loss, adaptive / plain | 0.01365 / 0.00867 | 0.01362 / 0.00833 | 0.01451 / 0.00931 |
| Trunk occupancy entering T4 (median) | 0.94 | 0.94 | 0.94 |
| Verdict | `durable_support` | `transition_support` | `durable_support` |
| F_1, adaptive / plain / seq_ft | 92.0 / 58.5 / 24.5 | 80.5 / 52.0 / 24.5 | 87.5 / 77.5 / 22.5 |

**Reading.**
1. **The causal effect replicates over three seeds.** Protecting at least 90% of each later task's
   new input energy raises Object's retention across the next task by +56 to +66 pp, with every CI
   excluding 0. It holds even on seed 2, where the baseline forgot less.
2. **T4 plasticity has a real cost.**
   - At the loss level it is consistent: the last-50 loss is about 1.6× plain GPM's in every seed.
   - In rollouts it is significant in 2 of 3 seeds (−22 and −14 pp).
   - It crossed its pre-registered threshold only on seed 1.
   - The measured immediate mechanism is the same in all three seeds: 94% trunk occupancy entering
     T4, and 99.5% of the trunk gradient inside memory.
3. **A caveat on "passes":** seed 2's T4 threshold is lower (75%) than seed 1's (83%), because
   seq_ft seed 2 itself learned LIBERO-10 only to 90%. Under seed 1's threshold, seed 2's 80% would
   fail. The per-seed rule was fixed in advance and is applied as registered, but the difference
   is reported so that the count of 1 in 3 failures is not over-read.

**Thesis claim, three seeds.** Under fixed total-energy GPM, the under-protection of later tasks
causally explains their forgetting. Allocating protection by new energy removes most of it. The
price is capacity: after three heterogeneous tasks the trunk is nearly exhausted, and the fourth
task learns measurably worse, consistently in loss and in 2 of 3 seeds in success. This is a
capacity-bounded stability–plasticity trade-off.

---

## 4. Next (per the decided roadmap in the seed-1 record, §6)
1. **SGP plan and pre-registration.** It directly targets the T4 capacity cost:
   - paper SGP as the baseline;
   - adaptive SGP as the thesis method;
   - α fixed a priori.
2. **Gate 4**, now runnable, since three seq_ft seeds exist.
3. **The baselines** (`replay`, `lora`, `ewc`, `consft`), then the high-protection control, then
   the rest of the Stage A matrix.
4. **Machine stability.** Two hard stops in three days. `--from-step` recovers at step granularity,
   but a run that is cut mid-training is lost, about 5 h each time. Options are a UPS, or
   checkpoint-resume inside `run_continual`, a separate feature.
