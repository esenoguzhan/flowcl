# Instruction negatives (exploratory pilot, seed 0): with noun-swap negatives at stage A, GPM keeps the plate task through stove at 0.70 (historical GPM: 0.00); the same negatives without projection keep nothing

**Status: run on GPU Tue 17 Nov 03:22–08:55, queue `QUEUE DONE ok`, summary `valid`.** The rule is
`configs/analysis/instruction_negatives.yaml` (SHA-256 `a1f0a38b…`), committed before the smoke.
One seed, one curriculum pair, and the negatives-off cells are historical controls. This is a pilot,
not a thesis result.

- **The question:** A (bowl → plate) and B (bowl → stove) share a scene and differ by one noun.
  In `runs/2026-10-05_similar_cl.md`, every arm lost A after learning B (0.94 → 0.00), GPM
  included. Do noun-swap negatives at stage A make a plate/stove distinction that survives B, and
  does projection add to it?
- **The method:** at stage A only, a second flow loss on the same observations. It pairs a wrong
  instruction from a fixed bank with a "hold" target. Stage B is unchanged.
- **The stage-A policy discriminates.**
  - Plate success 0.94 [0.86, 1.00].
  - Under the held-out instructions "put the bowl on the stove" and "…on top of the cabinet", it
    reaches the plate goal in **0 of 50** rollouts each.
  - It mostly holds still: median displacement 3.1 cm and 0.5 cm. Two plate paraphrases still
    succeed (0.95, 0.90).
- **After stove:**

  | Plate after stove (R1[A]) | allowlist_ft | GPM |
  |---|---:|---:|
  | negatives off (historical, `98f143e`) | 0.00 | 0.00 |
  | **negatives on (this run)** | 0.00 | **0.70** [0.56, 0.82] |

  - **Stove is learned equally well:** 0.96 in both new arms (historical: GPM 0.98, allowlist 0.90).
  - **Neither ingredient alone keeps any plate success; together they keep 0.70.** The
    interaction is +0.70 [0.56, 0.82]. It is flagged `against_historical_control`.
- **The labels** (acquisition `clean`):

  | Label | Result | Basis |
  |---|---|---|
  | `effect_present` | **effect_present** | GPM + negatives − historical GPM: +0.70 [+0.56, +0.82] |
  | `retains` | not_shown | Plate forgetting 0.24 [0.10, 0.38] > 0.10 |
  | `projection_contributes` | **projection_contributes** | GPM − allowlist, both with negatives: +0.70 [+0.56, +0.82] |
  | `allowlist_without_projection_retains` | not_shown | Forgetting 0.94 |

- **After stove, the GPM arm still selects by instruction, both ways.**
  - In stove's initial states, the plate instruction reaches the plate in 0.78 of rollouts.
  - In plate's initial states, the stove instruction reaches the stove in 0.98.
  - The allowlist arm became a stove policy: under the plate instruction it puts the bowl on the
    stove (0.94) and never on the plate.
  - The jointly trained base of `runs/2026-10-06_language_cl.md` behaved differently under GPM:
    with the plate instruction it reached neither goal.
- **Cost:** stage A took 63 min of wall clock against 40 min for the historical GPM run (+57%),
  from the second forward pass.

## 1. What was run

`bash scripts/queue_instruction_negatives.sh` in tmux session `instrneg`. The tree was clean at
`7544eb2`, the 8-task smoke's record commit (docs only). The code is:
- `a8cb634`: the method, configs and bank;
- `16242e0`: the rule, study module, CLI, queue and the rollout motion flag;
- `3973b36` and `27ed0bd`: hold-check revision 2 (§2).

| Step | What | Start → end | Time |
|---|---|---|---|
| 0 | hold check (no policy) | 03:22:52 → 03:23:28 | 36 s |
| 1 | `gpm_neg`, `--curriculum similar_ab_stove --seed 0 --amp --no-t1-check` | → 05:33:14 | 2 h 10 min |
| 2 | stage-A panel and gate on its `stage0.pt` | → 06:11:20 | 38 min |
| 3 | `allowlist_ft_neg`, T1 reference and identity reference = step 1's run, `--identity-stages 0` | → 08:15:24 | 2 h 04 min |
| 4 | stage-1 swaps, GPM then allowlist | → 08:51:38 | 22 + 15 min |
| 5 | probes (negatives base, both stage-1 checkpoints) | → 08:55:15 | 4 min |
| 6 | summary | → 08:55:22 | 7 s |

**Runs:**
- `results/similar_ab_stove__gpm_projected_adam_neg__seed0`;
- `results/similar_ab_stove__allowlist_ft_neg__seed0`.

**Study artifacts:** `results/instruction_negatives/{hold,panel,swaps/{gpm,allow},probes/*,summary.json}`.
The queue log is in `results/logs/queue_20261117_032252_instruction_negatives/`.

**Recipe:** that of the historical runs, 30k steps per task, batch 64, lr 1e-4, AMP and 50 rollouts per
cell. The seed namespace is `similar_ab_stove__seq_ft__seed0`, so the per-episode seeds are identical
across the two new runs and the two historical ones (checked).

**The method** (`flowcl/methods/instruction_negatives.py`; `gpm_neg`, display
`gpm_projected_adam_neg`, and `allowlist_ft_neg`):
- **The negative loss:** at stage 0, `loss + 1.0 × flow_loss(negative batch)`. The negative batch
  has the same images and state, an instruction drawn from the bank, and a hold target.
- **The hold target:** zero pose delta, with the gripper at the previous step's command (open at
  t = 0) and a mask of ones.
- **The bank:** `configs/analysis/instruction_negatives_bank.yaml`, 15 strings, none containing
  "stove" or "cabinet": 8 noun-slot swaps of the plate instruction and 7 unrelated instructions.
- **Randomness:** drawn from the method's own generator, so the positive batches, flow times and
  noise are those of the plain run.
- **Text encoding:** the bank is encoded in fp32 at the stage start.
- **The negative loss fell from 1.13 to 0.0002** (301 logged intervals). At stage 1 it is
  inactive in both runs.
- **GPM's memory comes from A's dataset, positives only,** at ε = 0.95. Its median rank fraction is
  0.31 over 91 layers.
- **allowlist_ft_neg** trains the same parameter set GPM projects, but without projection.

**Order with the other session:**
- The pilot was decided to go first on the GPU tonight. The other session's 8-task smoke
  (`t8smoke`, 00:54–03:11) was already running, so this queue waited for it.
- The GPU smoke ran 03:11–03:21 in a scratch folder: 200 steps, 1 rollout, both arms whatever the
  gate said. It ended `QUEUE DONE ok`, with identity 0 of 656, T1 pairing 0.0 and the negative loss
  falling.
- No other GPU process ran between 03:22 and 08:55.

## 2. Validity (all pass)

| Check | Result |
|---|---|
| Clean `git_sha`, namespace, method configs, recipe (both runs) | pass |
| Per-episode seeds identical across new and historical runs | pass |
| allowlist stage 0 against GPM's, bit for bit | **0 of 656 tensors differ** |
| T1 pairing against GPM's run | rel weight diff **0.0** |
| Negatives active at stage 0 and inactive at stage 1; bank SHA-256 `6f9affc7…`; bank cached in fp32 | pass (both runs) |
| GPM memory SHA-256 and worst residual over bound | pass, 0.026 |
| allowlist frozen set and freeze report | pass, 0 changed frozen tensors |
| Panel and swap artifacts: checkpoint SHA, weights unchanged, eval seeds | pass |
| Hold check | pass (below) |

**Hold check: revision 2 was made before any training run, and is disclosed in the rule's header.**
- **What it checks:** whether the hold action really holds the arm, without any policy. It covers
  demos 0–4 of A at two states each:
  - the initial state;
  - the first step after the gripper has been commanded closed for 20 steps, reached by replaying
    the demo's actions.
- **Revision 1 failed** in the smoke at all 5 mid-grasp states: 2.7–3.6 cm from the hold's first
  step.
  - A diagnostic in the simulator (demos 0 and 3) showed this was the arm stopping. The demo lifts
    at nearly full speed; the arm comes to rest within 10–20 steps and then moves < 0.05 cm.
  - Revision 2 (01:09) measures the end-effector, like the fingers, from the settle step (15). The
    bounds are unchanged (1 cm, 2 mm).
- **In production:** after settling, the end-effector moved ≤ 1.24 mm and each finger ≤ 1.20 mm at
  all 10 states. The stopping distance was 0.15–0.76 cm at the initial states and 2.7–3.6 cm
  mid-grasp.

## 3. Results

### 3.1 Stage A: the discrimination gate and the panel (GPM run's `stage0.pt`)

**Gate: passed.**
- R0[A] = 0.94, against the required ≥ 0.80.
- Under each held-out task instruction, A's goal was reached in 0.00 of 50 rollouts, against the
  required < 0.50.

| Condition (A's initial states) | n | A's goal reached | Requested goal reached | Median max displacement | Median path |
|---|---:|---:|---:|---:|---:|
| A's own instruction (motion reference) | 20 | 0.95 | — | 36.8 cm | 76 cm |
| Paraphrase "place the bowl on the plate" | 20 | 0.95 | — | 36.5 cm | 76 cm |
| Paraphrase "put the bowl onto the plate" | 20 | 0.90 | — | 36.3 cm | 77 cm |
| Held out: "put the bowl on the stove" | 50 | **0.00** | 0.00 | 3.1 cm | 3.5 cm |
| Held out: "put the bowl on top of the cabinet" | 50 | **0.00** | 0.00 | 0.5 cm | 0.5 cm |
| Held out, unrelated: "turn on the stove" | 20 | 0.00 | 0.00 | 0.6 cm | 0.6 cm |
| Trained negative: "put the bowl on the rack" | 20 | 0.00 | — | 0.6 cm | 0.6 cm |
| Trained negative: "put the wine bottle on the rack" | 20 | 0.00 | — | 0.6 cm | 0.6 cm |

Motion is measured over the first 100 executed steps. A plate success ends its episode early, so
paths are not compared between the plate rows and the others.

- **What the base learned: the plate instruction and its paraphrases → plate; anything else →
  hold.**
  - This is abstention that generalises to untrained instructions. It is not task selection: the
    held-out tasks were never trained, so their requested goals are 0.
- **"Stove", the instruction one noun away, is the least still.**
  - Displacement quantiles 1.3 / 3.1 / 6.9 cm (10th / 50th / 90th percentile); 16 of 50 rollouts
    moved > 5 cm, and the maximum is 10.4 cm.
  - None reached the plate.
  - Under "cabinet top" every rollout stayed within 0.9 cm.
- **Stove's zero-shot cell at stage A is 0.00.**

### 3.2 Stage B: retention and acquisition

| Cell (50 rollouts) | GPM + neg | allowlist + neg | GPM, historical | allowlist, historical |
|---|---:|---:|---:|---:|
| R0[A] plate before stove | 0.94 [0.86, 1.00] | 0.94 (identical weights) | 0.94 [0.86, 1.00] | 0.94 |
| **R1[A] plate after stove** | **0.70 [0.56, 0.82]** | 0.00 | 0.00 | 0.00 |
| R1[B] stove | 0.96 [0.90, 1.00] | 0.96 [0.90, 1.00] | 0.98 [0.94, 1.00] | 0.90 [0.80, 0.98] |
| Plate forgetting R0 − R1 | 0.24 [0.10, 0.38] | 0.94 [0.86, 1.00] | 0.94 | 0.94 |

Every difference below is rollout-paired, with a 10 000-resample bootstrap.
- **Acquisition is `clean`:** GPM − allowlist on stove is 0.00 [−0.06, +0.06]. Against its
  historical counterpart, GPM is −0.02 [−0.10, +0.04] and allowlist +0.06 [−0.04, +0.16].
- **The negatives-on and historical stage-0 GPM cells** have the same rate but differ in 4 of 50
  rollouts. The weights differ, as they should: the negatives change stage-0 training.
- **GPM's 15 failed plate rollouts after stove all time out** (600 of 600 steps). Its successes
  take a median of 86 steps.
- **The interaction** is (GPM − allowlist | on) − (GPM − allowlist | off) = 0.70 − 0.00 =
  **+0.70 [+0.56, +0.82]**. The off cells are historical.

### 3.3 Stage-1 swaps: what each final policy does with the two instructions

| Swap (50 rollouts, the evaluation's seeds) | GPM + neg | allowlist + neg |
|---|---:|---:|
| A's initial states, stove instruction: stove reached | 0.98 | 0.92 |
| A's initial states, stove instruction: plate reached | 0.00 | 0.00 |
| B's initial states, plate instruction: **plate reached** | **0.78** | 0.00 |
| B's initial states, plate instruction: stove reached | 0.00 | **0.94** |

- **The GPM arm selects between the two goals by instruction** in both directions.
- **The allowlist arm became a stove policy.** It puts the bowl on the stove under either
  instruction, as seq_ft did from the joint base in `runs/2026-10-06_language_cl.md` §3.3.
- **For reference,** GPM from that joint base reached neither goal under the plate instruction
  (0.00 plate, 0.00 stove; all time out).

### 3.4 Probes (descriptive)

The statistics are those of `runs/2026-10-05_similar_cl.md` and `runs/2026-10-06_language_cl.md`,
with the same seed tags. They are computed on A's data under B's instruction; B's data under A's
instruction is similar.

| | A-only base (similar_cl) | Joint base (language_cl) | **Negatives base** | GPM + neg, stage 1 | allowlist + neg, stage 1 |
|---|---:|---:|---:|---:|---:|
| Velocity sensitivity, plate ↔ stove | 2.4e-6 | — | **1.67** | 0.40 | 0.0011 |
| Chunk relative divergence, plate ↔ stove | 0.0004 | 0.035 | **0.85** | 0.25 | 0.016 |
| Trunk instruction signal (median size) | 0.003 | 0.036–0.043 | **0.68** | 0.13 | 0.034 |
| Share of that signal outside GPM's memory (trunk median) | 0.27 | 0.08 | **0.14** | 0.14 | 0.23 |
| Probe loss on A's data | 0.0069 | 0.0063 | 0.0069 | **0.096** | 1.15 |
| Action drift on A, stage 0 → 1 (relative) | | | | **0.083** | 0.42 |

- **The negatives made plate ↔ stove one of the largest contrasts the policy has.** Swapping the
  noun changes sampled chunks by 85% of their scale. In the A-only base it changed them by 0.04%,
  and in the joint base, which never trained this contrast, by 3.5%.
- **About 86% of that signal lies inside GPM's memory,** although the memory was built from
  positives only. Projection then protects it.
- **After stove, the GPM arm keeps 30% of the chunk contrast** (0.85 → 0.25). The allowlist arm
  keeps 2% (0.016) and loses the plate behaviour with it (probe loss 1.15, as seq_ft's 1.16 in
  language_cl).

### 3.5 Cost

| Stage wall-clock | GPM + neg | allowlist + neg | GPM, historical | allowlist, historical |
|---|---:|---:|---:|---:|
| Stage 0 (A, negatives on) | 3802 s | 3731 s | 2421 s | 2345 s |
| Stage 1 (B) | 2613 s | 2039 s | 2551 s | 2004 s |

- **The negative term adds about 23 min (+57%) at stage A.** Stage B is unchanged.
- **The historical runs ran on other days,** so their times are indicative.

## 4. Interpretation (not tested by the rule)

- **The 10-05 failure was not a limit of projection.** A single-task stage A gives the policy no
  reason to read the noun, so GPM has nothing instruction-specific to protect. With negatives, A's
  behaviour becomes conditional on the instruction, and most of that conditioning sits in the
  directions GPM keeps.
- **The negatives alone do not survive stove.** Plain fine-tuning of the same parameters washes the
  instruction dependence out within one task (chunk contrast 0.85 → 0.016). This matches seq_ft
  from the joint base.
- **Retention is partial: forgetting 0.24.** The failed rollouts time out, so after stove some plate
  episodes neither place nor go to the stove. A higher ε for A's memory might close part of the
  gap; it was not tested.
- **The stage-A behaviour is abstention, not understanding.** The policy holds under every untrained
  instruction. That suffices for two tasks. With more tasks, each new stage would need its own
  negatives, or replay, for the earlier tasks to keep holding under the new instructions. Neither
  was tested.
- **Prior art:** a wrong instruction trained to a null action is in Leap-VLA (arXiv 2611.01011,
  §V), whose negatives name absent objects. What this pilot adds is the combination with
  projection-based retention, on instructions that are feasible in the scene and were never
  negatives.

## 5. Limits

- **Exploratory:** one seed, one task pair (plate → stove), and point conventions fixed before the
  run. The CIs are over rollouts, not seeds.
- **Historical controls:** the negatives-off cells come from runs at `98f143e`. They share the
  namespace, seeds and recipe, but are not bitwise reruns on this code. The interaction inherits
  that flag.
- **Narrow panel:** the panel's held-out instructions are two tasks, plus one unrelated instruction
  and two paraphrases.
- **The hold check** validates 10 states, not every training timestep.
- **No videos.** What GPM's timed-out plate rollouts do is not labelled.

## 6. Possible next steps (parked, none decided)

- **Seeds 1–2:** about 5.5 h each, the same queue with `--seed`. This would need the rule's seed
  field and namespace extended first.
- **The other similar curricula:** `similar_ab_milk` and `similar_ab_cabinet` (10-05).
- **More protection for A:** ε = 0.99 for A's memory (`eps: 0.99`, not `gpm_e99`), to see whether
  forgetting falls below 0.10.
- **A third task:** negatives at every stage, to see whether abstention carries over a longer
  sequence.
- **Fresh negatives-off controls** on the current code, to drop the historical flag.
- **Not in the plan.** None of these displaces E3, the f = 0.7 queue or Track 2 (`status.md` §4).
