# Instruction negatives at seeds 1–2 (stove and cabinet top): stove replicates (plate 0.92 and 0.88, fresh plain GPM 0.00); cabinet top is `mixed`, a small gain at every seed (+0.10, +0.28; seed 0 +0.04) that is far from retention

**Status: ran on the GPU Thu 19 Nov 01:38–18:50, with a reboot at 14:52.** The queue ended
`QUEUE DONE ok` and the summary is `valid`.
- **The rule:** `configs/analysis/instruction_negatives_seeds.yaml` (SHA-256 `fdb9354a…`),
  committed before the smoke (`9e2d781`).
- **Scope:** two new seeds. At each, a new negatives base trained on stove and cabinet top resumed
  from it. **Every control is fresh plain GPM, same code, same namespace**, so nothing in seeds 1–2
  leans on a historical run. Seed 0's rows (stove pilot, cabinet follow-up) stand beside them,
  pinned, with their historical controls.

**The question.** At seed 0, GPM + negatives kept plate at 0.70 through stove and at 0.04 through
cabinet top (`runs/2026-11-17_instruction_negatives.md`,
`runs/2026-11-18_instruction_negatives_curricula.md`). Is 0.70 or 0.04 the typical case, and does
either replicate against fresh controls?

**What was found:**
- **Stove: `replicated`.** Plate after stove, GPM + negatives against fresh plain GPM:

  | Seed | GPM + neg | plain GPM | Paired difference | `effect_present` | `retains` |
  |---|---:|---:|---|---|---|
  | 0 (pilot) | 0.70 | 0.00 (historical) | +0.70 [+0.56, +0.82] | effect_present | not_shown (forgetting 0.24) |
  | 1 | **0.92** | 0.00 | +0.92 [+0.84, +0.98] | effect_present | **retains** (forgetting 0.06) |
  | 2 | **0.88** | 0.00 | +0.88 [+0.78, +0.96] | effect_present | **retains** (forgetting 0.06) |

  Seed 0 was the low seed. Stove is learned equally well in both arms at every seed (0.98 / 0.98,
  0.98 / 1.00).
- **Cabinet top: `mixed`.**

  | Seed | GPM + neg | plain GPM | Paired difference | `effect_present` | Forgetting (neg / ctrl) |
  |---|---:|---:|---|---|---|
  | 0 | 0.04 | 0.00 (historical) | +0.04 [0.00, +0.10] | not_shown | 0.90 / 0.94 |
  | 1 | 0.10 | 0.00 | +0.10 [**+0.02**, +0.18] | not_shown (below +0.20) | 0.88 / 0.94 |
  | 2 | **0.28** | 0.00 | +0.28 [+0.16, +0.42] | **effect_present** | 0.66 / 0.90 |

  `retains` is `not_shown` at every seed. Cabinet top is learned equally well (0.96–0.98 in both
  arms).
- **Outside the rule: the user's reading that these are improvements holds as a sign, not as
  retention.**
  - **The sign:** the negatives arm is ahead on cabinet top at all three seeds. The CI low is above 0
    at seeds 1 and 2 (`improvement_ci_low_above_0` in the wine pair's terms), and exactly 0.00 at
    seed 0.
  - **Every discordant rollout favours the negatives:** 2 + 5 + 14 = 21 of 150 paired rollouts, and
    none the other way (the control never reaches plate after cabinet top).
  - **The size:** 2 to 28 pp, against a starting point of 0.94–0.98. So the negatives keep a little
    of plate through cabinet top, not plate.
  - **The unit is the seed.** Three of three positive is p = 1/8 by a one-sided sign test over seeds.
    The rollout count above is descriptive.
- **The seed-0 loss gap does not hold up as the explanation (§3.4).** At seeds 1–2, A's probe loss
  after cabinet top (0.102, 0.076) equals stove's (0.094, 0.083). So did the chunk contrast. What
  still separates the two B tasks is the release: under the plate instruction in B's scenes, the arm
  reopens the gripper in 38–49 of 50 rollouts after stove, and in 13–31 after cabinet top.

## 1. What was run

The queue was `bash scripts/queue_instruction_negatives_seeds.sh`, in tmux session
`instrnegseeds`, at `9e2d781` with a clean tree and 248 GB free.

| Step | What | Start → end | Time |
|---|---|---|---|
| 0 | preflight (pins, hold record, seed-0 summaries, clean tree, idle GPU, disk, targets absent) | 01:38:54 → 01:38:56 | 2 s |
| 1 | seed 1, stove, `gpm_neg`, full run | → 03:46:42 | 2 h 08 |
| 2 | seed 1, panel and gate on its `stage0.pt` | → 04:24:08 | 37 min |
| 3 | seed 1, cabinet top, `gpm_neg`, resumed from step 1's `stage0.pt` (`fbcd10df…`) | → 05:21:18 | 57 min |
| 4 | seed 1, stove, plain GPM, full run | → 07:13:55 | 1 h 53 |
| 5 | seed 1, cabinet top, plain GPM, resumed from step 4's `stage0.pt` (`0c62dcff…`) | → 08:11:41 | 58 min |
| 6–9 | seed 2: the same (bases `4973ff1e…`, `a2e6cc01…`) | → 13:47:29 | 5 h 36 |
| 10 | seed 2, cabinet top, plain GPM, resumed | started 13:47:29; **lost at the 14:52 reboot** | |
| 10 (again) | the same, after the relaunch | 16:12:11 → 17:08:46 | 57 min |
| 11 | probes (3 per seed and B, 12) | → 17:22:50 | 14 min |
| 12 | summary | → 17:23:08 | 18 s |
| 13 | stage-1 swaps of the four negatives runs | → 18:49:55 | 4 × 22 min |
| 14 | summary | → 18:50:13 | 18 s |

**Runs:** `results/similar_ab_{stove,cabinet}__{gpm_projected_adam_neg,gpm_projected_adam}__seed{1,2}`.
All eight use the namespace `similar_ab_stove__seq_ft__seed<N>`, so the two arms at a seed draw the
same batches, flow times, noise and rollout seeds.

**Study artifacts:** `results/instruction_negatives_seeds/{preflight.json,summary.json,seed{1,2}/{stove/{panel,probes,swaps},cabinet/{probes,swaps}}}`.

**Queue logs:** `results/logs/queue_20261119_013854_instruction_negatives_seeds/` and, after the
reboot, `results/logs/queue_20261119_161206_instruction_negatives_seeds/`.

**The reboot.**
- **What happened:** the user rebooted the machine at 14:52 (`last -x`: boot 14:52, shutdown 14:58,
  boot 15:00), about an hour into step 10. Steps 0–9 had finished, each with its `result.json` or
  `panel.json`.
- **What was done:**
  - the partial run was moved aside to
    `results/interrupted/similar_ab_cabinet__gpm_projected_adam__seed2_reboot_20261119`;
  - before the relaunch, by hand: clean tree at `9e2d781`, 248 GB free, an idle GPU, no other
    queue, `~/.libero/config.yaml` pointing at `~/flowcl`;
  - the queue was relaunched with `--from-step 10`. It accepted steps 0–9 as `PRIOR ok` and reran
    step 10 in full, from the same pinned source (`a2e6cc01…`).
- **The summary checks every reused output** (below), and all pass.

## 2. Validity (all pass)

| Check | Result |
|---|---|
| Stove and curricula rules, the bank (`6f9affc7…`), the hold record and seed 0's two summaries at their pins | pass |
| Gate per seed: R0[A] ≥ 0.80; plate reached < 0.50 under each held-out task | pass at both seeds (§3.1) |
| Negatives runs, stove (2): namespace, method, clean commit, recipe, negatives active at stage 0 with the bank's SHA | pass; negative loss 1.15 → 0.0004 (seed 1) and 1.15 → 0.0002 (seed 2), 301 logged steps |
| Negatives runs, cabinet top (2): resumed from the pinned base; no own `stage0.pt`; B task in config, `result.json` and `eval/stage1.json`; negatives logged at stage 1 only as inactive | pass |
| Control runs (4): namespace, plain GPM without negative keys, clean commit, recipe; the cabinet top ones resumed from their pinned stove base | pass |
| GPM memory (8 runs): `memory_check` worst residual over bound; restored memory at the pin (resumed runs) | pass, 0.025–0.026 |
| Row pairing: per row the task key, episode index and rollout seed equal across arms, within each B and for plate across the two Bs | pass |
| Panel, probe and swap artifacts: checkpoint SHA, weights unchanged | pass |

**How the rows pair.** Each row's rollout seed is `derive_seed(namespace, task, episode index)`,
and the episode index selects the initial state. So equal seeds mean the same initial state and the
same policy noise in both arms.

## 3. Results

Every difference is neg − ctrl, rollout-paired, with a 10 000-resample bootstrap. "Rollouts that
differ" counts discordant pairs.

### 3.1 The gate and the panel (each seed's negatives base, before any B)

| | Seed 0 (pilot) | Seed 1 | Seed 2 |
|---|---:|---:|---:|
| R0[A], plate (50) | 0.94 | 0.98 | 0.94 |
| Own instruction (panel, 20) | 0.95 | 0.95 | 1.00 |
| Paraphrases "place the bowl on the plate" / "put the bowl onto the plate" (20 each) | 0.95 / 0.90 | 1.00 / 0.95 | 0.95 / 1.00 |
| Plate reached under "put the bowl on the stove" (50) | 0.00 | 0.00 | 0.00 |
| Plate reached under "put the bowl on top of the cabinet" (50) | 0.00 | 0.00 | 0.00 |
| Plate reached under the two trained negatives and "turn on the stove" (20 each) | 0.00 | 0.00 | 0.00 |

- **Both gates pass.** As at seed 0, the base does plate under its instruction and paraphrases, and
  abstains under everything else.
- **Plate before B, neg − ctrl:** +0.04 [−0.04, +0.12] at seed 1 (0.98 against 0.94) and
  +0.04 [−0.06, +0.14] at seed 2 (0.94 against 0.90). So the arms start level.

### 3.2 Stove

| Cell (50 rollouts) | Seed 1: neg | Seed 1: ctrl | Seed 2: neg | Seed 2: ctrl |
|---|---:|---:|---:|---:|
| R0[A] plate before stove | 0.98 | 0.94 | 0.94 | 0.90 |
| **R1[A] plate after stove** | **0.92** | 0.00 | **0.88** | 0.00 |
| R1[B] stove | 0.98 | 0.98 | 0.98 | 1.00 |
| Plate forgetting | 0.06 | 0.94 | 0.06 | 0.90 |

- **Acquisition `clean` at both seeds:** stove neg − ctrl 0.00 [−0.06, +0.06] and
  −0.02 [−0.06, 0.00].
- **`effect_present`:** +0.92 [+0.84, +0.98] (46 rollouts differ) and +0.88 [+0.78, +0.96] (44).
- **`retains` at both seeds**, which it was not at seed 0.
- **Across seeds: `replicated`.** Mean over seeds 0–2: 0.83 against 0.00.

### 3.3 Cabinet top

| Cell (50 rollouts) | Seed 1: neg | Seed 1: ctrl | Seed 2: neg | Seed 2: ctrl |
|---|---:|---:|---:|---:|
| R0[A] plate before cabinet top (the seed's base) | 0.98 | 0.94 | 0.94 | 0.90 |
| **R1[A] plate after cabinet top** | **0.10** | 0.00 | **0.28** | 0.00 |
| R1[B] cabinet top | 0.96 | 0.98 | 0.98 | 0.98 |
| Plate forgetting | 0.88 | 0.94 | 0.66 | 0.90 |

- **Acquisition `clean` at both seeds:** −0.02 [−0.08, +0.04] and 0.00 [−0.06, +0.06].
- **Seed 1:** +0.10 [+0.02, +0.18], 5 rollouts differ, all for the negatives. `not_shown`: the CI
  low is above 0, but the point is below the rule's +0.20.
- **Seed 2:** +0.28 [+0.16, +0.42], 14 rollouts differ, all for the negatives. `effect_present`.
- **Across seeds: `mixed`.** Mean over seeds 0–2: 0.14 against 0.00.

### 3.4 Probes (descriptive)

On A's data under B's instruction. The seed-0 values are the earlier records' (stove pilot and
cabinet follow-up, §3.5 and §3.7 there).

| | Stove s0 | Stove s1 | Stove s2 | Cabinet s0 | Cabinet s1 | Cabinet s2 |
|---|---:|---:|---:|---:|---:|---:|
| **Base:** chunk divergence, plate ↔ B | 0.85 | 0.85 | 0.73 | 0.91 | 0.91 | 0.92 |
| **Base:** trunk signal outside GPM's memory | 0.14 | 0.09 | 0.13 | 0.12 | 0.07 | 0.09 |
| **After B, neg:** chunk divergence | 0.25 | 0.25 | 0.30 | 0.31 | 0.31 | 0.31 |
| **After B, neg:** probe loss on A | 0.096 | 0.094 | 0.083 | 0.133 | 0.102 | 0.076 |
| **After B, neg:** action drift on A, stage 0 → 1 | 0.083 | 0.087 | 0.082 | 0.10 | 0.096 | 0.090 |
| **After B, ctrl:** chunk divergence | —¹ | 0.0018 | 0.0013 | 0.0024¹ | 0.0026 | 0.0024 |
| **After B, ctrl:** probe loss on A | 0.45¹ | 0.50 | 0.48 | 0.46¹ | 0.56 | 0.53 |
| **Plate success after B, neg** | 0.70 | 0.92 | 0.88 | 0.04 | 0.10 | 0.28 |

¹ Seed 0's control is historical: GPM at `98f143e` (`runs/2026-10-05_similar_cl.md` §3.3 for the
probe loss). Its chunk divergence after stove was not reported.

**Reading:**
- **The negatives keep the instruction contrast and A's loss on both B tasks, at every seed.**
  Plain GPM collapses the contrast to about 0.002 and A's loss to about 0.5.
- **At seeds 1–2, the probes do not separate the two B tasks.** After cabinet top, A's loss is the
  same as after stove (0.102 against 0.094; 0.076 against 0.083), and the contrast is the same or
  larger. Yet plate succeeds at 0.10 and 0.28 against 0.92 and 0.88.
- **So seed 0's loss gap (0.133 against 0.096) was not a stable separator.** The idea that ε = 0.99
  could close it, parked in the curricula record's §6, loses its premise.
- **The base is again not why:** its plate ↔ cabinet top contrast is the larger one at every seed.
- **Drift is slightly higher after cabinet top at every seed** (by 0.008–0.017). It is a small,
  consistent difference, not a tested one.

### 3.5 Stage-1 swaps of the negatives runs

| Swap (50 rollouts, the evaluation's seeds) | Stove s0 | Stove s1 | Stove s2 | Cabinet s0 | Cabinet s1 | Cabinet s2 |
|---|---:|---:|---:|---:|---:|---:|
| A's states, B's instruction: B reached | 0.98 | 0.92 | 0.94 | 0.96 | 0.94 | 1.00 |
| A's states, B's instruction: plate reached | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| B's states, plate instruction: **plate reached** | **0.78** | **0.96** | **0.80** | **0.06** | **0.06** | **0.20** |
| B's states, plate instruction: B reached | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| B's states, plate instruction: **gripper reopened in the first 100 steps** (≥ 2 sign changes, of 50) | **49** | **43** | **38** | **13** | **16** | **31** |
| B's states, plate instruction: median max displacement, first 100 steps | 35.8 cm | 36.4 cm | 35.9 cm | 38.5 cm | 38.8 cm | 38.2 cm |

- **On both B tasks, at every seed, the final policy selects by instruction one way.** Under B's
  instruction it does B. Under the plate instruction it never does B.
- **On cabinet top it then fails to complete plate.** It moves as far as on stove, and a plate
  success took a median 86 steps at seed 0, so the release falls inside the window. But it reopens the
  gripper in 13–31 of 50 rollouts, against 38–49 after stove.
- **The ordering matches the success rates.** Cabinet top's best seed (seed 2: 0.20 here, 0.28 on
  plate's own states) is the one with the most reopenings (31).
- **What the arm does after step 100 is not measured,** and there are no videos.

### 3.6 Cost

- **Stage A:** the negatives base trains in 3803 s on average, against 2455 s for plain GPM
  (+55%; seed 0: +57%). It is paid once per seed and shared by both B tasks.
- **Stage B:** 2589–2610 s with negatives, 2531–2606 s without, so within 3%.

## 4. Interpretation (not tested by the rule)

- **Stove is now a replicated result against same-code controls.**
  - At three seeds, GPM + negatives keeps plate through stove at 0.70, 0.92 and 0.88; plain GPM
    keeps 0.00.
  - At seeds 1–2 it meets the rule's `retains` (forgetting 0.06).
  - This removes the main limit of the pilot, which leaned on one seed and a historical control.
- **The stove/cabinet top difference is not a seed artifact.** It holds at all three seeds. So the
  curricula record's "the seed" candidate is ruled out, and stove's 0.70 was its low seed, not its
  high one.
- **On cabinet top, the negatives help a little and consistently.** The sign is positive at every
  seed, and no rollout favours the control. But the size varies (4–28 pp) and stays far from
  retention, and the rule's reading is `mixed`.
- **Where cabinet top fails is the place-and-release phase, not instruction selection.**
  - The contrast survives, A's loss is as low as on stove, and the policy never does cabinet top
    under the plate instruction.
  - What differs is the gripper release in B's scenes.
  - This is consistent with the curricula record's remaining candidate, that cabinet top's update
    overlaps with plate's place phase: both B tasks pick up the bowl, but only cabinet top ends in
    a place-and-release like plate's. It is consistent, not tested.
- **For the thesis:** instruction negatives make GPM retain A through a similar B when the B
  differs from A in where the motion goes (stove). They help only weakly when B shares A's final
  phase (cabinet top). The wine pair (next) asks whether the stove result is particular to this
  object and scene.

## 5. Limits

- **Three seeds,** two of them new. The CIs are over rollouts; the across-seed reading is a count.
- **Within a seed, stove and cabinet top share a base,** by design. So the two B tasks are compared
  on equal starts, and their seeds are not independent of each other.
- **Seed 0's rows** keep their historical controls and flags.
- **The reboot** cost one partial run, which was rerun in full from its pinned source. Steps 0–9
  were reused after the summary's checks.
- **The gripper statistic** is a 100-step window; there are no videos.

## 6. Possible next steps (parked, none decided)

- **The wine pair** (wine bottle → cabinet top, then → rack, seeds 0–2), already planned and
  approved. It tests the recipe on another A, with its own bank.
- **Videos of the cabinet-top negatives arm under the plate instruction** (one evaluation pass),
  to see what happens to the release.
- **ε = 0.99 for A's memory, on cabinet top.** Its premise, a loss gap, did not survive seeds 1–2
  (§3.4), so it is now a weaker candidate.
- **Not in the plan.** None of these displaces Track 2 step 4, `main_eval.yaml` or E3
  (`status.md` §4).
