# Language-conditioned base: the policy does select between similar tasks by instruction when trained jointly; from that base GPM keeps the cabinet task (0.98) but still loses the plate task to the stove (0.90 → 0.00)

**Status: exploratory, one seed, one base, two arms, no gate.** The rule is
`configs/analysis/language_cl.yaml`. Every run and artifact is valid; the summary's status is
`valid` (exit 0).

**Setup:**
- **Stage 0 (the base):** A (bowl → plate) co-trained with S2 (bowl → top of the cabinet), 60k
  steps. The two tasks share the LIBERO-Goal scene and object, so only the instruction tells them
  apart.
- **Stage 1:** S (bowl → stove), 30k steps.
- **Arms:** seq_ft and plain GPM. GPM's memory is built from the joint stage-0 data.
- 50 rollouts per cell.

| | Stage 0 (base) | Stage 1, seq_ft | Stage 1, GPM |
|---|---:|---:|---:|
| **A** bowl → plate | 0.90 [0.80, 0.98] | **0.00** | **0.00** |
| **S2** bowl → cabinet top | 1.00 | **0.00** | **0.98** [0.94, 1.00] |
| **S** bowl → stove | 0.00 (zero-shot) | 0.92 [0.84, 0.98] | 0.96 [0.90, 1.00] |

GPM's stage 0 equals seq_ft's bit for bit (0 of 656 tensors differ), so the stage-0 column holds
for both arms.

**Answers to the questions that prompted the study:**
- **Can the policy select between similar tasks by instruction? Yes, when stage 0 requires it.**
  - Swapping the instruction on the base flips the behaviour (A's initial states under the cabinet
    instruction, and the reverse):
    - **Original success:** drops by 0.90 [0.80, 0.98] and 1.00.
    - **Requested goal reached:** 1.00 and 0.94 [0.86, 1.00] of swapped rollouts.
    - **L1a:** `instruction_dependence_demonstrated`. **L1b:** `task_selection_threshold_met`.
  - Instruction sensitivity rose by 5 orders of magnitude in velocity, from 2–4×10⁻⁶ for the A-only
    policy to 0.68–0.85 for the joint base. It rose about 400× in sampled chunks (0.0006–0.0009 →
    0.29–0.32).
  - **So the architecture's language pathway works.** The near-zero sensitivity in
    `2026-10-05_similar_cl.md` came from training one task per stage, not from the architecture.
    This answers GPT's joint A+B control.
- **Does a language-conditioned base let GPM keep A while it learns a similar task? Not for A.**
  - **L3 for A: `no_demonstrated_advantage`.** GPM loses A completely (0.90 → 0.00), as seq_ft
    does, and as GPM did from the single-task base.
  - **L3 for S2: `protects`.** GPM keeps the co-trained cabinet task: forgetting 0.02 [0.00, 0.06],
    +0.98 [0.94, 1.00] over seq_ft.
  - **No plasticity cost:** S under GPM − seq_ft = +0.04 [−0.04, +0.14].
- **Which old task survives seems to track how close its instruction is to the new one in the
  base's representation** (descriptive, §4).
  - The base was trained to tell plate from cabinet top. That contrast is large in its activations.
  - It was never trained to tell plate from stove. Those instructions differ by one noun at the
    same position, and their contrast is about 6× smaller.
  - GPM spared the task far from stove and lost the one close to it.
  - Motion similarity is confounded with this: plate and stove are both table-level placements, the
    cabinet top is not.

**Date (local, CEST):** Mon 5 Oct 2026, 22:24:30 → Tue 6 Oct, 04:56:43 (6 h 32 min).
- **seq_ft:** 2 h 41. Stage 0 trained 79 min, stage 1 39 min; the rest is evaluation, 3 cells of 50
  rollouts per stage.
- **Stage-0 swaps:** 29 min.
- **GPM:** 2 h 38 (stage 1 trained 42 min).
- **Stage-1 swaps:** 15 min (seq_ft) and 22 min (GPM).
- **Probes:** 8 min. **Summary:** 7 s.

**Code:**
- The rule and the implementation are in `06c63e0`. The probe refinement (per-position token
  contrast) is in `6b42712`.
- The queue started clean at `6b42712`, and every run's `git_sha` is `6b427122…`.
- The rule `configs/analysis/language_cl.yaml` has SHA-256 `9183710a…`. The curriculum
  `langbase_cabinet_stove.yaml` has `1d0ba4bd…`.
- Both SHAs were locked before the smoke run (scratch; summary `valid`) and were unchanged at
  launch.
- CPU suite: 915 passed. The sim test of the goal predicates passed.

**Cite:**
- `results/language_cl/summary.json` (SHA-256 `e768cc60…`);
- `results/language_cl/swaps/{stage0_seq_ft,stage1_seq_ft,stage1_gpm}/swaps.json`;
- `results/language_cl/probes/{a_only_stage0,base_stage0,seq_ft_stage1,gpm_stage1}/probe.json`;
- the runs `results/langbase_cabinet_stove__{seq_ft,gpm_projected_adam}__seed0`;
- the queue log `results/logs/queue_20261005_222430_language_cl/queue.log`.

---

## 1. Design (as pre-registered)

**Why a joint stage 0:**
- In a single-task stage the instruction is a constant input, so nothing in the training contrast
  requires selecting behaviour by it.
- The language weights do get gradients, but dependence on another instruction is underdetermined.
- A single-task policy's ~0 sensitivity is therefore uninformative about the pathway. This study
  gives stage 0 the contrast.

**Curriculum `langbase_cabinet_stove`** (new runner support: `co_train`, `steps_factor`):
- **Stage 0:** A + S2, 2 × 30k steps, uniform over timesteps. Stats are fitted on A; the
  fingerprint `255ca42f…` equals the A-only lineage's.
- **Exposure:** A has 4,669 samples and S2 5,094 (47.8 / 52.2 %). That is 0.96× and 1.04× a
  single-task stage, recorded rather than claimed equal.
- **Evaluation:** A, S2 and S after every stage. S2 is not a retention-matrix column.

**Swap rollouts:**
- A task's own scene, initial states and success predicate, under another task's instruction.
- **Stopping rule:** the normal one (own goal or timeout).
- **Requested goal:** the other task's BDDL goal is scored after every step without ending the
  episode.
- **Stage 0:** A ↔ S2 on the base. The correct arm was rerun and reproduced `eval/stage0.json`
  exactly (0 of 50 different, both directions).
- **Stage 1:** A ↔ S on each arm. The correct arm is the eval's, with seeds checked.

**Probes** (descriptive): (a) the A-only base of similar_cl with its GPM memory; (b) this base with
its joint GPM memory; (c, d) both stage-1 checkpoints.
- **Probe (a):** reproduced `results/similar_cl/stage0/stage0.json` exactly (worst relative
  difference 0.0).

**Rules:**
- **L1a:** instruction dependence: both constituents ≥ 0.80, and both swap drops ≥ 0.15 with CI
  low > 0.
- **L1b:** task selection: requested goal reached in ≥ 0.50 of swapped rollouts in both
  directions.
- **L3:** retention, with acquisition required in both arms and the plasticity cost disclosed.
- **L4:** reported quantities.
- Every point threshold is an exploratory convention; the CIs are shown.

## 2. Validity (all passed)

- **Cells:** complete (50 rollouts) for A, S2 and S at both stages in both runs.
- **Production settings:** the joint stage as declared: co-train S2, factor 2, 60k + 30k steps
  trained, train keys [A, S2] in both `stage0.pt`. Clean `git_sha`; one stats fingerprint;
  identical per-episode seeds across arms.
- **GPM:**
  - identity at stage 0: 0 of 656 tensors;
  - T1 pairing: relative weight difference 0.0;
  - memory artifacts at both stages match their SHA-256;
  - worst residual over bound: 0.025.
- **Swap artifacts** (all three valid):
  - checkpoint SHA, weights unchanged;
  - text-cache precision as the runner's: A and S2 AMP, S fp32;
  - seeds equal to the eval's;
  - the three Goal envs have identical model signatures (joint names and qpos/dof addresses,
    bodies, nq/nv, state size);
  - every requested goal is scorable;
  - no requested goal held at a rollout's start.
- **Probe artifacts** (all four valid), including (a)'s reproduction.

## 3. Results

### 3.1 The base selects by instruction (L1a, L1b)

| Swap at stage 0 | Original success, correct → swapped | Drop | Requested goal reached (holds at end) |
|---|---|---:|---:|
| A's initial states, cabinet instruction | 0.90 → 0.00 | 0.90 [0.80, 0.98] | **1.00** (50/50) |
| S2's initial states, plate instruction | 1.00 → 0.00 | 1.00 [1.00, 1.00] | **0.94** [0.86, 1.00] (39/50) |

- The requested goal was first reached at a median step of 91 and 84.
- The episodes run on to the timeout. In 8 of the 47 plate hits the bowl did not stay on the
  plate.

### 3.2 Retention while S is learned (L3, L4)

| Paired difference (95% CI) | A | S2 |
|---|---:|---:|
| GPM − seq_ft at stage 1 (ΔR) | +0.00 [0, 0] | **+0.98 [+0.94, +1.00]** |
| Forgetting, seq_ft | 0.90 [0.80, 0.98] | 1.00 [1.00, 1.00] |
| Forgetting, GPM | 0.90 [0.80, 0.98] | **0.02 [0.00, 0.06]** |
| **Outcome** | `no_demonstrated_advantage` | `protects` |

- **Acquisition of S:** seq_ft 0.92, GPM 0.96. ΔP = +0.04 [−0.04, +0.14].
- **A-only lineage, quoted unpaired:** GPM, A after S = 0.00. Unchanged.
- **A's rollouts after S time out** (median 600 of 600 steps) in both arms.

### 3.3 What the stage-1 policies do with the two instructions

| Stage-1 swap | seq_ft | GPM |
|---|---|---|
| A's initial states, stove instruction: stove reached | 0.94 | 0.96 |
| S's initial states, plate instruction: own (stove) success | 0.92 (no drop) | **0.00** (drop 0.96) |
| S's initial states, plate instruction: plate reached | 0.00 | 0.00 |

- **seq_ft became a stove policy:** it puts the bowl on the stove under either instruction.
- **GPM kept the plate/stove distinction but not the plate behaviour.** Under the plate instruction
  it goes to neither goal (all 50 time out). What it does instead is not labelled.

### 3.4 Mechanism probes (descriptive)

| | A-only base (a) | Joint base (b) | seq_ft stage 1 | GPM stage 1 |
|---|---:|---:|---:|---:|
| Chunk sensitivity A ↔ S2 (A's data / S2's data) | 0.0006 / 0.0009 | **0.29 / 0.32** | 0.056 / 0.058 | 0.27 / 0.33 |
| Chunk sensitivity A ↔ S | 0.0004 / 0.0008 | 0.035 / 0.11 | 0.020 / 0.008 | 0.15 / 0.09 |
| Trunk instruction signal A ↔ S2 (size; share outside memory) | 0.009; 0.20 | **0.23; 0.04** | 0.10; – | 0.23; – |
| Trunk instruction signal A ↔ S (size; share outside memory) | 0.003; 0.27 | 0.036–0.043; 0.08 | 0.017; – | 0.046; – |
| Probe loss A / S2 / S | 0.0069 / 1.89 / 2.01 | 0.0063 / 0.0054 / 1.85 | 1.16 / 1.62 / 0.0037 | **0.197 / 0.127** / 0.0045 |

Trunk values are layer medians.

- **The trained contrast (plate vs cabinet top)** became about 25× larger in the trunk's inputs
  and lies almost entirely **inside** the joint memory (4% outside). A useful base can place its
  instruction distinctions inside what it protects.
- **The untrained contrast (plate vs stove)** grew about 13× but stays about 6× smaller than the
  trained one.
- **Token contrast:**
  - **Raw CLIP:** A and S differ only from token 6 (the noun) on, by 0.20–0.97 per position. A and
    S2 differ from token 5 on, by 0.38–1.21. AMP/fp32 rounding (A cached in AMP, S in fp32) adds about 0.001 per
    position.
  - **Projected:** the joint base's `token_projection` amplifies the trained contrast (relative
    difference 1.04, raw 0.68; the A-only base gave 0.38). A ↔ S stays near its raw size (0.43).
- **Memory:**
  - The joint memory protects a parameter-weighted 0.51 of the trunk, against 0.45 for the A-only
    memory.
  - It covers each constituent equally: 0.950 of A's and 0.950 of S2's trunk energy, and 0.952 /
    0.951 in the decoder.
- **GPM's blocked gradient share for S** (median over logged steps and trunk layers): 0.955 from
  the joint base, against 0.951 from the A-only base and 0.918 for the diverse task there. S's
  gradient overlaps the memory as much as before.
- **A's probe loss under GPM** is 0.197 (31× the base's) against 1.16 under seq_ft. Its velocity
  field is partly preserved, as from the A-only base (0.45), but not enough for a success.

## 4. Interpretation (not tested by the rule)

- **The language failure in similar_cl was a property of single-task training, not of the
  architecture.** With a contrast in stage 0, the same model reads the instruction and selects the
  task.
- **GPM's protection of a similar task depends on that task's inputs differing from the new
  task's in the residual directions.**
  - S2's instruction was made distinctive by training, so learning S in the unprotected ~5% moved
    S2 little.
  - A's instruction was never contrasted with "stove". A's and S's inputs therefore stay close, and
    S's update moved A too.
  - Consistent with this, GPM still learned to make the stove instruction behave differently from
    the plate instruction. It could not keep what the plate instruction used to produce.
- **Confound:** plate and stove are also the two table-level placements (the cabinet top is high).
  Language distance and motion distance are not separated by this design.
- **For the thesis story:**
  - The A-only result ("GPM fails on similar tasks") is partly a conditioning artefact. The
    co-trained task survives.
  - It is also partly real: a new task close to an old one, in a direction the base never learned to
    separate, still erases it.
  - **A positive result here supports "joint base + GPM (freezing + projection)" only.**
    Attributing S2's protection to projection needs allowlist_ft. Attributing it to language
    needs the diverse-base control.

## 5. Limits and disclosures

- **Scope:** one seed, one base, one new task. Similarity in language and in motion is not varied
  independently.
- **The A-only lineage comparison is unpaired:** different stage-0 policies and namespaces.
- **What GPM's policy does under the plate instruction after S is not labelled:** neither the plate
  nor the stove goal is reached; there are no videos.
- **"Requested goal reached" counts the first time the goal held.** The episode then continues, and
  in 8 of 47 plate hits (S2's scene) the goal no longer held at the end.
- **Exposure is 0.96× (A) and 1.04× (S2)** of a single-task stage.
- **Thresholds:** "≤ 0.10" forgetting and "≥ 0.50" requested goal are exploratory point
  conventions, not equivalence or reliability claims.
- **Probe statistics are influence measures, not causal attributions.** The `token_projection`
  gradient norms are in the probe files and are not interpreted.
- **This study is outside `docs/thesis_plan.md`** (one night of GPU).

## 6. Follow-ups named by the rule

**A: `no_demonstrated_advantage`, S2: `protects`.** Candidates, not decided here:
- **Joint A + S2 + S control:** whether all three coexist (capacity, expected from §3.1).
- **allowlist_ft from this base:** does freezing alone keep S2? About 2.7 h.
- **Diverse-base control:** A + milk → S, GPM. It separates language from two-task memory.
- **A broader Goal base** (A plus Goal tasks that also mention the stove and plate in other roles):
  whether a base that reads the target noun in general lets GPM keep A. This tests §4's
  explanation directly.
- **Replay (Workstream B):** it supplies the A/S contrast during stage 1 and is the in-plan remedy.
  Add A → S to its runs.
