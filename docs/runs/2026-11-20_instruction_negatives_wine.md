# Instruction negatives on the wine pair (wine bottle → cabinet top, then → rack; seeds 0–2): the rule's gate fails at every seed, so the reading is `not_assessed`; descriptively, GPM + negatives keeps A at 0.88, 0.94 and 0.16 against 0.02–0.04 for plain GPM

**Status: ran on the GPU Thu 19 Nov 19:14 → Fri 20 Nov 10:35.** The queue ended `QUEUE DONE ok`, and
the final summary is `valid` (SHA-256 `3f00d9b3…`).
- **The rule:** `configs/analysis/instruction_negatives_wine.yaml` (SHA-256 `278ff2aa…`), committed
  before the smoke (`9a5fc67`). The plan was approved Thu 19 Nov after two GPT reviews.
- **Scope:** a second A → B pair built like the bowl pair: the same object, another target, one
  scene. It has its own bank of 15 wrong instructions, in which the words "rack" and "bowl" never
  occur. Three seeds, each with a new negatives base and a fresh plain-GPM control (same code, same
  namespace).

**The question.** On the bowl pair, GPM + negatives kept A through stove at 0.70 / 0.92 / 0.88
against 0.00 (`runs/2026-11-19_instruction_negatives_seeds.md`). Does the same recipe help GPM keep
A on another pair?

**What was found:**
- **By the rule: `not_assessed` at every seed.** The gate requires a base that has learned A
  (R0[A] ≥ 0.80) and abstains under held-out tasks (A's goal < 0.50 under each).
  - **What failed:** under B's instruction, "put the wine bottle on the rack", the base still put
    the bottle on the cabinet: 0.90, 0.92 and 0.68 of 50 rollouts.
  - **Also:** A itself was learned below the floor at seeds 0 and 2 (0.78, 0.52).
  - **What held:** the base held still under everything else, including "put the bowl on top of
    the cabinet" (0.00 at every seed).
- **Reported, not judged:** the paired neg − ctrl difference on A after B, with fresh controls.

  | Seed | A before B: neg / ctrl | **A after B: neg / ctrl** | Paired difference, A after B | B after B: neg / ctrl |
  |---|---|---|---|---|
  | 0 | 0.78 / 0.98 | **0.88** / 0.02 | +0.86 [+0.76, +0.94] | 0.90 / 0.94 |
  | 1 | 0.90 / 0.78 | **0.94** / 0.04 | +0.90 [+0.80, +0.98] | 0.84 / 0.88 |
  | 2 | 0.52 / 0.94 | **0.16** / 0.02 | +0.14 [+0.04, +0.26] | 0.76 / 0.90 |

  - `improvement_ci_low_above_0` is true at all three seeds.
  - At seeds 0–1 the negatives arm loses nothing of A through B (forgetting −0.10 and −0.04).
    Plain GPM loses almost all of it (0.96, 0.74).
- **The route differs from the bowl pair's.**
  - **Bowl:** the negatives base already separated A from B's instruction (chunk contrast
    0.73–0.85), and B wore that down.
  - **Wine:** the base barely separated them (0.03–0.10), and the contrast grew during B, to
    0.11–0.28.

  So on this pair, the abstention the gate checks for was not needed for retention.
- **The costs:**
  - **A is learned less well with the negatives at two seeds:** A before B is −0.20
    [−0.32, −0.08] and −0.42 [−0.56, −0.28] against the control. At seed 1 it is +0.12.
    On the bowl pair the negatives cost A nothing.
  - **B is slightly lower at every seed:** −0.04, −0.04 and −0.14 [−0.28, 0.00].
  - The seed with the weakest A (seed 2, 0.52) is the one where retention largely fails.

## 1. What was run

The queue was `bash scripts/queue_instruction_negatives_pair.sh --rule wine`, in tmux session
`instrnegwine`, at `9a5fc67` with a clean tree and 245 GB free.

| Step | What | Start → end | Time |
|---|---|---|---|
| 0 | preflight (pins, clean tree, idle GPU, disk, targets absent) | 19:14:32 → 19:14:34 | 2 s |
| 1 | hold check (A's demos 0–4, no policy) | → 19:15:09 | 35 s |
| 2 | seed 0, `gpm_neg_wine`, full run | → 21:25:43 | 2 h 11 |
| 3 | seed 0, panel and gate on its `stage0.pt` | → 21:59:44 | 34 min |
| 4 | seed 0, plain GPM, full run | → 23:52:40 | 1 h 53 |
| 5–7 | seed 1: the same | → 04:32:27 | 4 h 40 |
| 8–10 | seed 2: the same | → 09:20:39 | 4 h 48 |
| 11 | probes (3 per seed, 9) | → 09:31:59 | 11 min |
| 12 | summary, preliminary | → 09:32:13 | 14 s |
| 13 | stage-1 swaps of the three negatives runs | → 10:35:40 | 22 + 22 + 19 min |
| 14 | summary, final | → 10:35:53 | 13 s |

- **Runs:** `results/similar_wine_rack__{gpm_projected_adam_neg,gpm_projected_adam}__seed{0,1,2}`,
  in the namespace `similar_wine_rack__seq_ft__seed<N>`.
- **Study artifacts:** `results/instruction_negatives_wine/{preflight.json,hold/,seed{0,1,2}/{panel,probes,swaps},summary_preliminary.json,summary.json}`.
- **Queue log:** `results/logs/queue_20261119_191432_instruction_negatives_wine/`.
- **Nothing interrupted it.** No reboot (`last -x`), and no other GPU process. A backup upload ran
  in another tmux session from 00:10; it does not use the GPU.

**The GPU smoke (18:59–19:14, scratch folder):**
- seed 0, 200 steps, 1 rollout per cell;
- it ended `QUEUE DONE ok`, with both summaries `valid`.
- The rule file was unchanged between the smoke and the launch (`278ff2aa…`).

## 2. Validity (all pass)

| Check | Result |
|---|---|
| The bank at its pin (`09c420b4…`); the method config names it; the control's config has no negative keys; the curriculum's tasks are [A, B] | pass |
| Hold check: A's demos 0–4, initial and closed states (10) | pass; end-effector within 0.9 mm, fingers within 1.3 mm |
| Negatives runs (3): namespace, method, clean commit, recipe, negatives active at stage 0 with the bank's SHA, `memory_check` | pass; negative loss 1.15 → 0.0002–0.0003 (301 logged steps); worst residual over bound 0.024–0.025 |
| Control runs (3): namespace, plain GPM without negative keys, clean commit, recipe, `memory_check` | pass; 0.024 |
| Row pairing: task key, episode index and rollout seed equal per row across the arms | pass |
| The panel's 20 own-instruction rows reproduce A's stage-0 evaluation (rollout seed, success, step count) | pass at every seed |
| The policy's texts: A "put the wine bottle on top of the cabinet", B "put the wine bottle on the rack" | recorded; as planned |
| Panel, probe and swap artifacts: checkpoint SHA, weights unchanged | pass |
| Commits of all outputs | all `9a5fc67` |

**The `.bddl` question from the first GPT review is settled:** the text the policy received for A
is the demo file's and the benchmark's, not the `.bddl` line about a drawer.

## 3. Results

Every difference is neg − ctrl, rollout-paired, with a 10 000-resample bootstrap. "Rows that
differ" counts discordant pairs.

### 3.1 The gate and the panel (each seed's negatives base, before B)

| Condition (A's initial states) | n | Seed 0 | Seed 1 | Seed 2 |
|---|---:|---:|---:|---:|
| R0[A], A's evaluation | 50 | **0.78** | 0.90 | **0.52** |
| A's own instruction | 20 | 0.75 | 0.90 | 0.55 |
| Paraphrase "place the wine bottle on top of the cabinet" | 20 | 0.75 | 0.95 | 0.60 |
| Paraphrase "put the wine bottle on the top of the cabinet" | 20 | 0.70 | 0.90 | 0.60 |
| **Held out, B: "put the wine bottle on the rack"** (A's goal reached) | 50 | **0.90** | **0.92** | **0.68** |
| Held out: "put the bowl on top of the cabinet" | 50 | 0.00 | 0.00 | 0.00 |
| Held out, unrelated: "turn on the stove" | 20 | 0.00 | 0.00 | 0.00 |
| Trained negatives: "put the wine bottle on the plate", "push the plate to the front of the stove" | 20 each | 0.00 | 0.00 | 0.00 |

- **Gate: failed at every seed.** A's goal under B's instruction is above 0.50 at all three. R0[A]
  is below 0.80 at seeds 0 and 2.
- **The base treats "rack" as "cabinet" for the wine bottle.** Under B's instruction it moves as
  under its own (median maximum displacement 28–30 cm, against 30 cm) and puts the bottle on the
  cabinet. Under every other untrained or trained wrong instruction it holds (0.5–2.0 cm).
- **On the bowl pair the base abstained under B's instruction** at every seed (A's goal 0.00 under
  "put the bowl on the stove").

### 3.2 Retention and acquisition (reported for every seed; not judged)

| Cell (50 rollouts) | s0 neg | s0 ctrl | s1 neg | s1 ctrl | s2 neg | s2 ctrl |
|---|---:|---:|---:|---:|---:|---:|
| R0[A] before B | 0.78 | 0.98 | 0.90 | 0.78 | 0.52 | 0.94 |
| **R1[A] after B** | **0.88** | 0.02 | **0.94** | 0.04 | **0.16** | 0.02 |
| R1[B] | 0.90 | 0.94 | 0.84 | 0.88 | 0.76 | 0.90 |
| Forgetting of A, from its own start | −0.10 | 0.96 | −0.04 | 0.74 | 0.36 | 0.92 |

| neg − ctrl | Seed 0 | Seed 1 | Seed 2 |
|---|---|---|---|
| **A after B** | **+0.86 [+0.76, +0.94]**, 43 rows differ | **+0.90 [+0.80, +0.98]**, 45 | **+0.14 [+0.04, +0.26]**, 9 |
| B after B | −0.04 [−0.16, +0.06], 8 | −0.04 [−0.18, +0.10], 12 | −0.14 [−0.28, 0.00], 13 |
| A before B | −0.20 [−0.32, −0.08], 12 | +0.12 [0.00, +0.24], 10 | −0.42 [−0.56, −0.28], 21 |

**Mean over the three seeds:**

| | A before B | A after B | B after B |
|---|---:|---:|---:|
| neg | 0.73 | 0.66 | 0.83 |
| ctrl | 0.90 | 0.03 | 0.91 |

**Within the negatives arm, A from stage 0 to stage 1** (the same rollout seeds and initial states
at both stages; paired, computed for this record, not in the summary):

| Seed | Change | Rows gained / lost |
|---|---|---|
| 0 | +0.10 [0.00, +0.22] | 7 / 2 |
| 1 | +0.04 [−0.04, +0.12] | 3 / 1 |
| 2 | −0.36 [−0.52, −0.18] | 4 / 22 |

The plain-GPM arm loses 48, 38 and 46 of its rollouts, and gains 0, 1 and 0.

### 3.3 Probes (descriptive)

Computed on A's data under B's instruction, like the bowl studies' probes.

| | s0 | s1 | s2 | Bowl pair, stove (s0 / s1 / s2) |
|---|---:|---:|---:|---|
| **Base:** chunk divergence, A ↔ B | **0.10** | **0.045** | **0.031** | 0.85 / 0.85 / 0.73 |
| **Base:** velocity sensitivity, A ↔ B | 0.16 | 0.030 | 0.013 | 1.67 / 1.58 / 1.50 |
| **Base:** trunk signal outside GPM's memory | 0.14 | 0.15 | 0.16 | 0.14 / 0.09 / 0.13 |
| **After B, neg:** chunk divergence | **0.28** | **0.22** | **0.11** | 0.25 / 0.25 / 0.30 |
| **After B, neg:** probe loss on A | 0.067 | 0.051 | 0.077 | 0.096 / 0.094 / 0.083 |
| **After B, neg:** action drift on A, stage 0 → 1 | 0.098 | 0.089 | 0.107 | 0.083 / 0.087 / 0.082 |
| **After B, ctrl:** chunk divergence | 0.002 | 0.002 | 0.001 | — / 0.0018 / 0.0013 |
| **After B, ctrl:** probe loss on A | 0.166 | 0.170 | 0.165 | 0.45 / 0.50 / 0.48 |
| **A after B, neg** | 0.88 | 0.94 | 0.16 | 0.70 / 0.92 / 0.88 |

**Reading:**
- **The contrast was built during B, not before it.** The base's A ↔ B contrast is 8–24 times
  smaller than the bowl base's. Through B, the negatives arm's contrast grows about 3–5 fold, to
  roughly the bowl pair's level after stove at seeds 0–1. Plain GPM's ends at about 0.002.
- **The size of the grown contrast orders the seeds:** 0.28 and 0.22 go with A kept (0.88, 0.94),
  0.11 with A mostly lost (0.16).
- **Probe loss does not predict success.** Seed 2's loss on A (0.077) is close to seed 0's (0.067),
  but its success is 0.16 against 0.88.
- **Plain GPM's loss on A after B is lower here than on the bowl pair** (0.17 against about 0.5),
  yet its success is just as low (0.02–0.04).

### 3.4 Stage-1 swaps of the negatives runs

| Swap (50 rollouts, the evaluation's seeds) | Seed 0 | Seed 1 | Seed 2 |
|---|---:|---:|---:|
| A's states, B's instruction: B (rack) reached | 0.86 | 0.92 | 0.74 |
| A's states, B's instruction: A (cabinet) reached | 0.00 | 0.00 | 0.02 |
| B's states, A's instruction: **A (cabinet) reached** | **0.88** | **0.92** | **0.08** |
| B's states, A's instruction: B (rack) reached | 0.00 | 0.00 | 0.36 |

- **At seeds 0–1 the final policy selects by instruction both ways,** from either task's initial
  states, at 0.86–0.92. That is comparable to the bowl pair's after stove (0.78–0.98).
- **At seed 2 it is mostly a rack policy.** Under the cabinet instruction in B's states it goes to
  the rack in 36% of rollouts and reaches the cabinet in 8%.

### 3.5 Cost

- **Stage A:** the negatives base trains in 3712 s on average, against 2404 s for plain GPM (+54%;
  bowl pair +55–57%).
- **Stage B:** 2569–2636 s with negatives, 2574–2619 s without, so the same.

## 4. Interpretation (not tested by the rule)

- **The rule cannot credit this result,** and that is a property of the rule, not of the runs.
  - **What the gate was for:** to confirm that the base had learned to tell A from B's instruction
    before B, so that retention could be put down to that.
  - **What happened:** on this pair the base did not tell them apart, yet the negatives arm kept A
    at two of three seeds.
  - **So** the gate's premise, that abstention at stage A is how the negatives work, is not what
    happens on this pair.
- **What the negatives seem to give, on both pairs, is a live instruction pathway.**
  - A single-task base ignores its instruction. The negatives base responds to it: it holds under
    wrong instructions, including on this pair.
  - Plain GPM ends both pairs with no instruction contrast (about 0.002) and no A.
  - The negatives arm ends both pairs with a contrast of 0.11–0.31, built or kept.
  - **The contrast is necessary at most, not sufficient.** On the wine pair its size orders the
    seeds, but on the bowl pair cabinet top keeps the largest contrast (0.31) and loses A.

  This is a reading across two pairs and nine conditions, not a test.
- **Why the wine base does A under "rack" is open.** Candidates, untested:
  - the text encoder places "rack" close to "cabinet" for this object;
  - the bank's five wine-bottle noun swaps all hold, and "rack" falls on A's side of them;
  - the two goals share more of their motion.
- **The cost on this pair is real.** At seeds 0 and 2 the negatives made A harder to learn (−0.20,
  −0.42), and B is a little lower at every seed. The weakest base (0.52) is the one where retention
  failed. So A's acquisition under the negatives looks like the limiting factor here, not B.
- **For the thesis:** across nine conditions (two A tasks, three similar B tasks, three seeds each),
  GPM + negatives against plain GPM gives a positive paired difference on A after B in every one:
  - bowl → stove: +0.70, +0.92, +0.88;
  - bowl → cabinet top: +0.04, +0.10, +0.28;
  - wine → rack: +0.86, +0.90, +0.14.

  Bowl seed 0's controls are historical. The difference is large in 5 of the 9 conditions. The rules give `replicated` (stove), `mixed`
  (cabinet top) and `not_assessed` (wine). The wine gate failed for a reason that the result itself
  argues against.

## 5. Limits

- **Three seeds.** The CIs are over rollouts.
- **The pair and the bank change together,** so their separate contributions stay open.
- **The labels are `not_assessed`.** Everything in §3.2–§3.4 is descriptive. The within-arm
  stage 0 → 1 change in §3.2 was computed for this record, outside the summary.
- **No allowlist arm,** so this pair does not test whether projection is needed. On the bowl pair it
  was (allowlist + negatives 0.00).
- **No videos.** The swap motion statistics cover the first 100 steps only.

## 6. Possible next steps (parked, none decided)

- **A gate that tests what matters.** For example, the base's instruction sensitivity (chunk
  contrast to any wrong instruction, or holding under trained negatives) instead of abstention under
  B. It would be pre-registered for the next pair, not applied to this one after the fact.
- **The text-encoder distance** between the held-out instructions and A's, for both pairs (no GPU
  rollouts; minutes).
- **A's acquisition under the negatives.** For example, the negative loss weight or a longer stage
  A on the wine pair, judged on R0[A].
- **Not in the plan.** None of these displaces Track 2 step 4, `main_eval.yaml` or E3
  (`status.md` §4).
