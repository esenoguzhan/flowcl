# Instruction negatives on cabinet top and milk (exploratory, seed 0): the stove effect does not carry over to cabinet top (plate 0.04, historical GPM 0.00); on milk, where GPM already kept plate, the negatives change nothing measurable

**Status: ran on the GPU Wed 18 Nov 01:52–06:03. The queue ended `QUEUE DONE ok` and the summary is
`valid`.**
- **The rule:** `configs/analysis/instruction_negatives_curricula.yaml` (SHA-256 `8b9769de…`),
  committed before the smoke.
- **Scope:** one seed, and all three B tasks share one negatives base, so this tests generality over
  B, not replication. The negatives-off cells are historical controls.

**The question.** The stove pilot (`runs/2026-11-17_instruction_negatives.md`) found that with
negatives at stage A, GPM keeps plate through stove at 0.70 (historical GPM: 0.00). This run trains
the other two B tasks of the 5 Oct study from that pilot's stage-A checkpoints:
- **cabinet top**, the second similar task (same scene and object). Historically every arm lost
  plate there;
- **milk → basket**, the diverse task (another suite and scene). Historically GPM already kept plate
  there (0.92).

**What was found:**
- **Cabinet top: the effect does not hold.**
  - **Plate after cabinet top:** 0.04 with GPM + negatives, against 0.00 for historical GPM
    (+0.04 [0.00, +0.10]). It is 0.00 for allowlist + negatives.
  - **Cabinet top is learned equally well:** 0.96 in both new arms.
  - **Every retention label is `not_shown`.**
- **What survives cabinet top:**
  - **The instruction contrast:** chunk divergence 0.31 against stove's 0.25 (historical GPM:
    0.0024).
  - **The policy stays off the cabinet:** under the plate instruction in cabinet's scenes it reaches
    the cabinet in 0 of 50 rollouts.
  - **What is lost is plate itself.** The arm grasps and carries the bowl. Within 100 steps it
    reopens the gripper in 13 of 50 rollouts; in the stove pilot it did in 49 of 50.
- **Milk: neutral where GPM already worked.**
  - **Plate:** 0.92, the same as historical GPM; 0.00 [−0.08, +0.08] is labelled
    `no_cost_beyond_10pp`.
  - **Milk:** 0.82 against 0.76; +0.06 [−0.10, +0.22] is labelled
    `within_10pp_on_observed_rollouts`.
  - **So `milk_no_harm` is `not_shown`**, only because milk's CI low (−0.1005) reaches the −0.10
    bound.
  - The negatives alone keep 0.00 on milk too, so projection is still what protects plate there.
- **Across the three B tasks, GPM + negatives keeps plate at:**
  - stove 0.70;
  - cabinet top 0.04;
  - milk 0.92 (unchanged).

  The stove result depended on the B task.
- **The base is not why (addendum, §3.7).** Probed under the cabinet-top instruction, it separates
  plate from cabinet top at least as strongly as from stove:
  - chunk contrast 0.91, against 0.85;
  - slightly more of the signal inside GPM's memory (outside share 0.12, against 0.14).

  So cabinet top's loss of plate happens while learning B.

## 1. What was run

The queue was `bash scripts/queue_instruction_negatives_curricula.sh`, in tmux session `instrnegcur`.
- **Code:** the tree was clean at `64445a9`.
  - `ea4e46d`: the rule, the study module, the CLI, the queue and the tests.
  - `64445a9`: the fix of the summary's stage-0 cells, found by the GPU smoke (below).
- **Free disk:** 254 GB at launch.

| Step | What | Start → end | Time |
|---|---|---|---|
| 0 | preflight (pins, sources bitwise equal, historical files, clean tree, idle GPU, disk, targets absent) | 01:52:34 → 01:52:37 | 3 s |
| 1 | cabinet top, `gpm_neg`, resumed from the stove `gpm_neg` run's `stage0.pt` (`--resume-sha256 91f2ade7…`) | → 02:50:15 | 58 min |
| 2 | cabinet top, `allowlist_ft_neg`, resumed from the stove `allowlist_ft_neg` run's `stage0.pt` (`ba8af0bb…`) | → 03:38:55 | 49 min |
| 3 | milk, `gpm_neg`, resumed | → 04:32:22 | 53 min |
| 4 | milk, `allowlist_ft_neg`, resumed | → 05:22:24 | 50 min |
| 5 | cabinet stage-1 swaps, GPM then allowlist | → 05:59:12 | 22 + 15 min |
| 6 | probes on the 4 stage-1 checkpoints | → 06:02:45 | 4 min |
| 7 | summary | → 06:03:05 | 20 s |

**Runs:** `results/similar_ab_{cabinet,milk}__{gpm_projected_adam_neg,allowlist_ft_neg}__seed0`.

**Study artifacts:** `results/instruction_negatives_curricula/{preflight.json,summary.json,cabinet/{swaps,probes},milk/probes}`.

**Queue log:** `results/logs/queue_20261118_015232_instruction_negatives_curricula/`.

**How the runs were set up:**
- **Resumed at stage 1.** Each run resumes from its stove source's `stage0.pt` and inherits the
  stove namespace (`similar_ab_stove__seq_ft__seed0`), as the 5 Oct cabinet and milk runs did. So
  the per-episode seeds equal the historical runs'.
- **Recipe:** the stove pilot's (30k steps, batch 64, lr 1e-4, AMP, 50 rollouts per cell).
- **Stage 1** is plain training of B, with the negatives inactive.

**The GPU smoke (01:38–01:50, scratch folder):**
- **Source:** the stove smoke's 200-step runs, 1 rollout per cell.
- **First result: the summary returned `incomplete`.**
  - It had asked stage 0 for B's cell. A resumed run's stage-0 evaluation is its source's, plate and
    stove.
  - Fixed in `64445a9`, with a test. The summary re-run was `valid`.
- **Resume equality:** a `gpm_neg` run resumed into `similar_ab_stove` equalled its source bit for
  bit.
  - 0 of 656 tensors differ.
  - `compare_resume` passed on the checkpoint, the memory and the rollouts.
- **No other GPU process ran** between 01:52 and 06:03.

## 2. Validity (all pass)

| Check | Result |
|---|---|
| Stove rule and summary at their pinned SHA-256s; summary `valid`, gate passed; `gate.json` pinned | pass |
| The two sources' `stage0.pt` at their pins, and bitwise equal to each other | pass, **0 of 656 tensors differ** |
| GPM source memory at its pin (`366ccc24…`) | pass |
| The 12 historical files (`eval/stage0.json`, `eval/stage1.json`, `result.json` × 2 curricula × 2 arms) at their pins | pass |
| Each resumed run (4): `resume.json` names the pinned source and SHA; no own `stage0.pt`; `eval/stage0.json` byte-equal to the source's | pass |
| Clean `git_sha`, namespace, method config, recipe (4 runs) | pass |
| Negatives inactive at stage 1, bank SHA-256 `6f9affc7…` in the method state | pass (4 runs) |
| B task: config, `result.json` task keys and `eval/stage1.json` all `[plate, B]`, 50 rollouts | pass (4 runs) |
| GPM arms: restored memory at the pin; `memory_check` worst residual over bound | pass, 0.025 (both) |
| allowlist arms: `frozen_check` against the source; freeze report | pass, 0 changed frozen tensors, 91 trainable tensors |
| Row pairing: task key, episode index and seed per row, across new and historical runs; A's rows against the stove runs' | pass (both curricula) |
| Swap and probe artifacts: checkpoint SHA, weights unchanged | pass |

**The projection basis** (summary `projection_basis`, both curricula; all checks pass):
- **Start:** the two arms start from bitwise-equal weights.
- **Trainable set:**
  - in both arms, every tensor outside the 91 allowlist names is unchanged from the source;
  - both arms train 41 954 304 parameters;
  - allowlist's freeze report lists exactly the 91.
- **Training config:** the stage-1 `train_config` records are identical (AdamW, lr 1e-4, weight decay
  1e-4, grad clip 1.0, 100 warm-up steps, cosine, AMP, batch 64, 30k steps). Their diff is empty.
- **Namespace:** both arms use `similar_ab_stove__seq_ft__seed0`, so batches, flow times and noise
  come from the same streams.
- **What remains:** projection (GPM's memory and its projected gradient and update).

## 3. Results

Every difference below is rollout-paired, with a 10 000-resample bootstrap. Differences against a
historical cell carry the flag `against_historical_control` (§5).

### 3.1 Cabinet top (similar): the stove rule's labels

| Cell (50 rollouts) | GPM + neg | allowlist + neg | GPM, historical | allowlist, historical |
|---|---:|---:|---:|---:|
| R0[A] plate before cabinet top | 0.94 [0.86, 1.00] | 0.94 (identical weights) | 0.94 [0.86, 1.00] | 0.94 |
| **R1[A] plate after cabinet top** | **0.04** [0.00, 0.10] | 0.00 | 0.00 | 0.00 |
| R1[B] cabinet top | 0.96 [0.90, 1.00] | 0.96 [0.90, 1.00] | 0.98 [0.94, 1.00] | 0.96 [0.90, 1.00] |
| Plate forgetting R0 − R1 | 0.90 | 0.94 | 0.94 | 0.94 |

- **Acquisition is `clean`:** GPM − allowlist on cabinet top is 0.00 [−0.08, +0.08].
- **Against each arm's historical control, on cabinet top:**
  - GPM: −0.02 [−0.06, 0.00] (1 of 50 rollouts differs);
  - allowlist: 0.00 [−0.08, +0.08].
- **The labels:**

  | Label | Result | Basis |
  |---|---|---|
  | `effect_present` | not_shown | GPM + negatives − historical GPM: +0.04 [0.00, +0.10]. The rule needs ≥ +0.20, with the CI low > 0 |
  | `retains` | not_shown | Forgetting 0.90 |
  | `projection_contributes` | not_shown | GPM − allowlist, both with negatives: +0.04 [0.00, +0.10] |
  | `allowlist_without_projection_retains` | not_shown | Forgetting 0.94 |

- **The interaction** is +0.04 [0.00, +0.10]; the off cells are historical.
- **GPM's two plate successes are new.** They are rollouts the historical run failed; no rollout goes
  the other way.
- **All 48 failed plate rollouts time out** (600 of 600 steps). The 2 successes take 86 and 118
  steps.

### 3.2 Milk (diverse): the no-harm reading

| Cell (50 rollouts) | GPM + neg | allowlist + neg | GPM, historical | allowlist, historical |
|---|---:|---:|---:|---:|
| R0[A] plate before milk | 0.94 [0.86, 1.00] | 0.94 | 0.94 | 0.94 |
| **R1[A] plate after milk** | **0.92** [0.84, 0.98] | 0.00 | 0.92 [0.84, 0.98] | 0.00 |
| R1[B] milk | 0.82 [0.70, 0.92] | 0.80 [0.68, 0.90] | 0.76 [0.64, 0.88] | 0.86 [0.76, 0.94] |
| Plate forgetting R0 − R1 | 0.02 | 0.94 | 0.02 | 0.94 |

- **Acquisition is `assessed`:** both arms reach ≥ 0.50 on milk.
- **The endpoints**, each GPM + negatives against historical GPM:

  | Endpoint | Difference | Rollouts that differ | Label |
  |---|---|---:|---|
  | Plate after milk | 0.00 [−0.08, +0.08] | 4 (2 each way) | `no_cost_beyond_10pp` |
  | Milk | +0.06 [−0.1005, +0.22] | 19 (11 vs 8) | `within_10pp_on_observed_rollouts` |

- **`milk_no_harm`: not_shown.** It needs both endpoints `no_cost_beyond_10pp`. Milk's paired CI
  low is −0.1005, at the −0.10 bound: the 2.5th percentile falls between −0.10 and −0.12. With
  19 discordant rollouts, 50 rollouts cannot rule out a 10 pp cost on milk, although the point
  favours the negatives.
- **The other labels:**
  - **`retains`: retains** (forgetting 0.02).
  - **`projection_contributes`: projection_contributes.** GPM − allowlist on plate is
    +0.92 [+0.84, +0.98].
  - **`allowlist_without_projection_retains`: not_shown.**
- **The interaction** is 0.00 [−0.08, +0.08].
- **allowlist's milk against its historical control:** −0.06 [−0.18, +0.06].

### 3.3 Across the three B tasks (reported, not judged)

| B | Plate before B (all arms) | Plate after B: GPM + neg / allow + neg / GPM hist / allow hist | B: GPM + neg / allow + neg / GPM hist / allow hist |
|---|---:|---|---|
| stove (pilot) | 0.94 | **0.70** / 0.00 / 0.00 / 0.00 | 0.96 / 0.96 / 0.98 / 0.90 |
| cabinet top | 0.94 | **0.04** / 0.00 / 0.00 / 0.00 | 0.96 / 0.96 / 0.98 / 0.96 |
| milk | 0.94 | **0.92** / 0.00 / 0.92 / 0.00 | 0.82 / 0.80 / 0.76 / 0.86 |

- **Plate before B** is the same base in all three, and equal in rate to the historical runs'.
  The paired difference is 0.00 [−0.08, +0.08]; 4 of 50 rollouts differ, as in the pilot.
- **So the differences after B are in forgetting, not in the start.**

### 3.4 Cabinet stage-1 swaps: what each final policy does with the two instructions

| Swap (50 rollouts, the evaluation's seeds) | GPM + neg | allowlist + neg | Stove pilot: GPM + neg / allow + neg |
|---|---:|---:|---|
| A's initial states, cabinet-top instruction: cabinet top reached | 0.96 | 0.96 | stove reached 0.98 / 0.92 |
| A's initial states, cabinet-top instruction: plate reached | 0.00 | 0.00 | 0.00 / 0.00 |
| B's initial states, plate instruction: **plate reached** | **0.06** | 0.00 | **0.78** / 0.00 |
| B's initial states, plate instruction: cabinet top reached | **0.00** | 0.96 | stove reached 0.00 / 0.94 |

- **The GPM arm still uses the instruction to avoid B.** Under the plate instruction it never puts
  the bowl on the cabinet.
- **It no longer completes A.**
  - **Motion:** in B's states under the plate instruction, it moves as far as a successful
    episode (median maximum displacement 38.5 cm and path 0.89 m over the first 100 steps). The
    stove pilot's arm moved 35.8 cm.
  - **The gripper, in the same window:**
    - here, 37 of 50 rollouts close the gripper once and have not reopened it;
    - in the stove pilot, 49 of 50 closed and reopened it (38 with two sign changes, 11 with
      more).
  - A plate success takes a median 86 steps, so the release falls inside the window.
- **What the arm does after step 100 is not measured.** All 50 rollouts run to 600 steps, and
  there are no videos.
- **The allowlist arm became a cabinet-top policy,** as it became a stove policy in the pilot.

### 3.5 Probes (descriptive)

These are the stove pilot's statistics, computed on A's data under B's instruction.

| | Stove: GPM + neg | Stove: allow + neg | **Cabinet: GPM + neg** | Cabinet: allow + neg | Milk: GPM + neg | Milk: allow + neg |
|---|---:|---:|---:|---:|---:|---:|
| Velocity sensitivity, plate ↔ B | 0.40 | 0.0011 | **0.45** | 0.012 | 1.43 | 0.10 |
| Chunk relative divergence, plate ↔ B | 0.25 | 0.016 | **0.31** | 0.050 | 0.91 | 0.33 |
| Trunk instruction signal (median size) | 0.13 | 0.034 | **0.25** | 0.11 | 0.63 | 0.34 |
| Share of that signal outside GPM's memory (trunk median) | 0.14 | 0.23 | 0.12 | 0.24 | 0.11 | 0.22 |
| **Probe loss on A's data** | **0.096** | 1.15 | **0.133** | 1.28 | 0.039 | 1.13 |
| Historical GPM's probe loss on A after B (`runs/2026-10-05_similar_cl.md` §3.3) | 0.45 | | 0.46 | | 0.047 | |
| Action drift on A, stage 0 → 1 (relative) | 0.083 | 0.42 | 0.10 | 0.44 | 0.058 | 0.45 |

**Reading:**
- **On cabinet top, the negatives did what they did on stove.** They kept A's instruction contrast
  through B:
  - chunk divergence 0.31; historical GPM's was 0.0024;
  - most of it inside the memory (outside share 0.12).
  - They cut A's loss after B from 0.46 to 0.13, against stove's 0.45 to 0.096.
- **The probes do not predict the rollouts.**
  - A probe loss 1.4× stove's goes with 0.04 plate success against 0.70.
  - The contrast is larger than stove's.
- **Milk is not comparable:** its instruction is another suite's task, so a large contrast is
  expected there.
- **The base's plate ↔ cabinet contrast was probed after the queue** (§3.7).

### 3.6 Cost

| Stage-1 wall clock | GPM + neg | allowlist + neg | GPM, historical | allowlist, historical |
|---|---:|---:|---:|---:|
| Cabinet top | 2604 s | 2036 s | 2571 s | 1999 s |
| Milk | 2638 s | 2024 s | 2591 s | 2003 s |

- **Stage B is unchanged by the negatives,** within 2%.
- **Stage A's +57% was paid once,** in the pilot, and is shared by all three B tasks.

### 3.7 Addendum: the base under the cabinet-top instruction (descriptive, not in the rule)

**What was run:** Wed 18 Nov, after the record above.
- **Command:** `scripts/instruction_negatives_curricula.py --part base-probe --curriculum cabinet`,
  at `6ffa0fd` (clean tree). It took 106 s.
- **What it probes:** the negatives base, i.e. the GPM source's `stage0.pt` at its pin `91f2ade7…`,
  with the pinned memory `366ccc24…`. It uses the stove pilot's base probe, with cabinet top as B.
- **Output:** `results/instruction_negatives_curricula/cabinet/probes/neg_stage0/probe.json`.

**Checks:**
- weights unchanged;
- A's probe loss (0.0069) and the memory coverage are identical to the pilot's base probe, as they
  must be (same checkpoint, same data);
- the cabinet-top instruction is encoded in fp32, as stove's was.

| Negatives base, before any B | plate ↔ stove (pilot) | **plate ↔ cabinet top** |
|---|---:|---:|
| Velocity sensitivity (A's data, B's instruction) | 1.67 | **1.63** |
| Chunk relative divergence (A's data, B's instruction) | 0.85 | **0.91** |
| Chunk relative divergence (B's data, A's instruction) | 0.84 | **0.87** |
| Trunk instruction signal, median size (A's data / B's data) | 0.68 / 0.67 | **1.41 / 1.04** |
| Share of that signal outside GPM's memory, trunk median (A's data / B's data) | 0.14 / 0.14 | **0.12 / 0.13** |
| Probe loss on B's data (B not yet trained) | 2.25 | 2.38 |
| After B, GPM + neg: chunk divergence (share of the base's kept) | 0.25 (29%) | 0.31 (34%) |
| After B, GPM + neg: plate success | 0.70 | 0.04 |

- **The base separates plate from cabinet top at least as strongly as from stove.** The chunk
  contrast is equal or larger in both directions, and the trunk signal is 1.5–2× larger.
- **Slightly more of it lies inside GPM's memory:** 12–13% outside, against 14%.
- **Through B, GPM keeps a similar share of the contrast** on both tasks (29% and 34%).
- **So the base candidate in §4 is ruled out.** Neither the size of the contrast nor its placement
  relative to the memory explains why plate survives stove and not cabinet top. The difference
  arises during stage B.

## 4. Interpretation (not tested by the rule)

- **The stove result depended on the B task.**
  - **Held fixed:** the base, the memory, the recipe and the seed streams. Only B changed.
  - **The outcome:** GPM + negatives kept plate through stove (0.70) but not through cabinet top
    (0.04).
  - **So** instruction negatives plus projection are not a general fix for forgetting between
    similar tasks: here they fixed 1 of 2 similar B tasks, at one seed.
- **On cabinet top, the instruction dependence survives but plate's action does not.**
  - **What survives:** the policy still reads the noun. It avoids the cabinet under the plate
    instruction, and its chunk contrast and memory share match stove's.
  - **What fails:** the plate behaviour. Its release does not happen in the window where it should.
  - **So** keeping the contrast is necessary, as the pilot's allowlist arm showed, but not
    sufficient.
- **Without negatives, stove and cabinet top hurt A equally** (probe loss 0.45 and 0.46). With them,
  the two differ modestly in loss (0.096 against 0.133) and greatly in success (0.70 against
  0.04).
  - This suggests closed-loop success drops steeply somewhere in that range of A's loss.
  - It is one pair of points, not a measured threshold.
- **Why cabinet top and not stove is open.** One candidate is ruled out; two remain, untested:
  - **Ruled out, the base** (§3.7). Its plate ↔ cabinet contrast is as large as its plate ↔ stove
    contrast, and as far inside the memory.
  - **Overlap:** cabinet top's update overlaps more with plate's place phase.
  - **The seed:** stove's 0.70 is the seed-0 outlier.
- **On milk the negatives are neutral,** as they should be where GPM already separates A from B.
  The negatives alone do not protect plate even against a diverse B (allowlist + negatives: 0.00).

## 5. Limits

- **Exploratory scope:**
  - one seed, and one shared stage-A base for all three B tasks;
  - CIs are over rollouts, not seeds;
  - the point conventions were fixed in the rule before the run.
- **Historical controls:**
  - **What they are:** the negatives-off cells come from runs at `98f143e`.
  - **The basis for using them:**
    - today's code reproduces September's plain-GPM runs bitwise at stages 0–1, with all 48
      stage-0/1 evaluation cells equal rollout for rollout (`runs/2026-11-18_ne70_4task.md`);
    - `allowlist_ft.py` is unchanged.
  - **What that does not establish:** that the October cabinet and milk paths behaved as today's
    code would. So the evidence is indirect, and every comparison with them is flagged.
- **Milk's no-harm reading** is limited by 50 rollouts: the paired milk CI is 32 pp wide.
- **No milk swaps:** milk's scene has no plate.
- **No videos.** The swap motion statistics cover the first 100 steps only.

## 6. Possible next steps (parked, none decided)

- **Seeds 1–2 for stove and cabinet top.** This needs a new negatives base per seed: about 5.5 h for
  the stove pair, plus about 1.8 h per further B. It would tell whether stove's 0.70 or cabinet top's 0.04
  is the typical case.
- **Done: the base probe under the cabinet-top instruction** (§3.7). It rules out the base.
- **ε = 0.99 for A's memory, on cabinet top.**
  - **The cost:** a new memory from the same `stage0.pt`, so it is not a plain resume, and the
    runner would need to support it.
  - **The question:** whether the loss gap (0.13 against 0.096) is what separates the two B tasks.
- **Videos of the cabinet-top GPM arm under the plate instruction** (one evaluation pass).
- **Not in the plan.** None of these displaces Track 2 step 4, `main_eval.yaml` or E3
  (`status.md` §4).
