# Similar-vs-diverse continual study: on a similar task GPM pays no plasticity cost (P1 not supported); instead it fails to protect A at all, while on a diverse task it keeps A at 92%

**Status: exploratory, one seed, one A state, three selected B tasks, no gate.** The rule is
`configs/analysis/similar_cl.yaml`. Every run is valid; the summary's status is `valid` (exit 0).

**Setup:**
- **A:** `libero_goal/put_the_bowl_on_the_plate`, trained once.
- **Bs:**
  - **S** (similar): bowl → stove.
  - **D** (diverse): Object, milk → basket.
  - **S2** (similar): bowl → top of the cabinet.
- **Arms:**
  - **seq_ft**;
  - **allowlist_ft**: from task 2 on, only GPM's trainable weights train, with no projection;
  - **plain GPM**.
- **Recipe:** 30k steps, 50 rollouts per cell.

| | B zero-shot `Z` | B after B `R[1][1]`<br>seq_ft / allowlist / GPM | A after B `R[1][0]`<br>seq_ft / allowlist / GPM |
|---|---:|---:|---:|
| **S** bowl → stove | 0.00 | 0.98 / 0.90 / 0.98 | **0.00 / 0.00 / 0.00** |
| **D** milk → basket | 0.00 | 0.80 / 0.86 / 0.76 | 0.00 / 0.00 / **0.92** |
| **S2** bowl → cabinet top | 0.00 | 0.94 / 0.96 / 0.98 | **0.00 / 0.00 / 0.00** |

A before B is `R[0][0]` = **0.94 [0.86, 1.00]**. It is the same A state, bit-identical in every arm.

**Answers to the question that prompted the study** ("if tasks are similar, GPM might fail"):
- **GPM did fail on the similar tasks, but not the way the diagnostic predicted.**
  - The prediction was that projection would block learning a similar B.
  - **Instead, B was learned fully** (0.98 on both similar tasks), and **A was lost completely**
    (0.94 → 0.00), exactly as under seq_ft.
- **On the diverse task, GPM worked as designed.** It kept A at 0.92 (forgetting +0.02
  [−0.08, +0.12]). Projection cost some of B against allowlist FT (−0.10 [−0.24, +0.04]; the CI
  includes 0).
- **Freezing alone protects nothing.** allowlist_ft forgets A completely in every curriculum. All of
  GPM's protection on D comes from the projection.
- **The policy barely uses its instruction** (P4). In the two similar curricula, A and B share the
  scene and the object; the instruction is the only input that differs. Learning B there rewrote the
  behaviour A needs, and GPM's projection did not prevent it. This is an interpretation, discussed
  in §4.

**Date (local, CEST):** Mon 5 Oct 2026, 01:00:20 → 12:05:03 (11 h 05 min).
- S's three full runs: 1 h 46, 1 h 41, 1 h 51;
- the stage-0 extras: 23 min (two zero-shot cells of 50 rollouts, plus the probes);
- the six resumed runs: 48–57 min each;
- the probes: 5 min and 2.6 min;
- the summary: 17 s.

**Code:** `98f143e`, clean at queue start; every run's `git_sha` is `98f143e0…`. The rule
`configs/analysis/similar_cl.yaml` (SHA-256 `db5f54dc…`) was committed with the implementation.
Its SHA was locked before the smoke run and was unchanged after it.

**Cite:**
- `results/similar_cl/summary.json` (SHA-256 `a6d7affa…`);
- `results/similar_cl/stage0/stage0.json` (`4f22d146…`) and `zero_shot.json` (`18545b3b…`);
- the stage-1 probes in `results/similar_cl/probes/<run>/probe.json`;
- the nine runs `results/similar_ab_{stove,milk,cabinet}__{seq_ft,allowlist_ft,gpm_projected_adam}__seed0`;
- the queue log `results/logs/queue_20261005_010020_similar_cl/queue.log`.

---

## 1. Design (as pre-registered)

**One A state.**
- S's three runs train both stages.
- allowlist_ft and GPM equal S's seq_ft at stage 0 bit for bit. The runner's identity check found
  **0 of 656 tensors different**, and GPM's T1 pairing gave a relative difference of 0.0.
- D's and S2's runs **resume at stage 1** from S's `stage0.pt` of the same arm. The SHA-256 is
  expected at launch and recorded in each `resume.json`:

  | Arm | Source `stage0.pt` SHA-256 |
  |---|---|
  | seq_ft | `03c92d2b…` |
  | allowlist_ft | `71b01fb3…` |
  | GPM | `4028cbf6…` |

- Every run inherits S's seed namespace, so:
  - **A's rollouts are paired** across all nine runs;
  - **B's rollouts are paired** across the three arms and with B's zero-shot cell.

**allowlist_ft** (`flowcl/methods/allowlist_ft.py`):
- **Task 1:** trains every parameter, as GPM does.
- **From task 2:** runs `freeze_to_allowlist`. It trains the **91 registry weight tensors (42.0M
  parameters)** and freezes **146 tensors (11.9M)** that seq_ft would train: the biases, norms,
  encoder projections, context queries, positional and chunk embeddings, and modulation layers.
- **Comparisons it enables:**
  - GPM against allowlist_ft isolates **projection**;
  - allowlist_ft against seq_ft isolates **freezing**.

**Measured at stage 0 (no training):**
- B's zero-shot cells;
- a fixed-batch probe loss on A and every B;
- an instruction probe for each A ↔ B pair, in both directions. It has two statistics:
  - the diagnostic's velocity statistic;
  - sampled-chunk sensitivity: `policy.sample()` from identical observations and noise.

**Measured at each run's stage 1:** the same probes for A and its B. The text cache was rebuilt as
in the runner before every probe, with A under AMP and each B in fp32; all ten probe files record
this.

**Not run:** adaptive GPM. It equals plain GPM through A's memory and B's training.

## 2. Validity (all passed)

- **Every cell** `R[0][0]`, `R[1][0]`, `R[1][1]` has 50 rollouts.
- **Production settings, namespace and stats:** production recipe and clean `git_sha` in all nine
  runs; S's namespace everywhere; one stats fingerprint, fitted on A, matching every checkpoint.
- **Resume:** every `resume.json` names S's same-arm `stage0.pt` with its current SHA. D and S2 have
  no own `stage0.pt`, and their `eval/stage0.json` is byte-identical to the source's.
- **Paired seeds:** per-episode seeds are identical across arms, and against the zero-shot file.
- **Frozen tensors** (allowlist_ft and GPM): the 565 state-dict tensors outside the allowlist are
  unchanged between stage 0 and stage 1.
  - allowlist_ft's recorded trainable set equals the allowlist.
  - **GPM never moved `flow_head.action_in.weight`.** Its input, the 7-dimensional noisy action, is
    spanned entirely by A's memory. This is reported, not failed, as pre-registered.
- **GPM's memory:** artifacts match their recorded SHA. The worst residual over bound is
  0.025 / 0.025 / 0.026 (S / D / S2).
- **Probe artifacts:** the stage-0 file and all nine stage-1 files are valid (checkpoint SHA,
  weights unchanged, cache precision).
- **No flags:** A is not weak (0.94), and every comparator acquired B (≥ 0.80).

## 3. Results

### 3.1 Paired differences (rollout-paired bootstrap, 95%)

| | ΔP_proj<br>GPM − allowlist (B) | ΔP_freeze<br>allowlist − seq_ft (B) | ΔP_tot<br>GPM − seq_ft (B) | ΔR_tot<br>GPM − seq_ft (A) | ΔR_proj<br>GPM − allowlist (A) |
|---|---:|---:|---:|---:|---:|
| **S** | +0.08 [+0.00, +0.18] | **−0.08 [−0.16, −0.02]** | +0.00 [−0.06, +0.06] | +0.00 [0, 0] | +0.00 [0, 0] |
| **D** | −0.10 [−0.24, +0.04] | +0.06 [−0.08, +0.20] | −0.04 [−0.20, +0.12] | **+0.92 [+0.84, +0.98]** | **+0.92 [+0.84, +0.98]** |
| **S2** | +0.02 [+0.00, +0.06] | +0.02 [−0.04, +0.10] | +0.04 [+0.00, +0.10] | +0.00 [0, 0] | +0.00 [0, 0] |

**Forgetting of A**, `R[0][0] − R[1][0]`:
- **+0.94 [+0.86, +1.00]** for every arm in every curriculum,
- **except GPM on D:** +0.02 [−0.08, +0.12].

### 3.2 Predictions

- **P1 (projection costs more plasticity on a similar B): not supported.** The basis is both
  similar curricula (S, S2) against D.
  - The contrast is in the **opposite** direction: mean ΔP_proj(similar) − ΔP_proj(D) = **+0.15**
    (S − D = +0.18, S2 − D = +0.12).
  - Projection cost nothing on the similar tasks. The only negative ΔP_proj is on the diverse task,
    and its CI includes 0.
- **P2 (GPM retains A better than seq_ft in every curriculum): fails.**
  - **Holds on D:** +0.92 [+0.84, +0.98].
  - **Fails on S and S2:** +0.00. GPM retained nothing.
- **P2b (projection beyond freezing):** on D, the whole retention gain is projection's (+0.92). On
  S and S2 there is none.
- **P3 (seq_ft's forgetting, similar against diverse):** complete in all three curricula (0.94 →
  0.00). No difference to report.
- **P5:**
  - **Freezing** cost 8 pp of B on S (CI excludes 0). On D and S2 its CIs include 0, with opposite
    signs, so freezing has no consistent cost.
  - **Zero-shot:** A's policy performs no B at all (Z = 0.00 for all three).

### 3.3 Probe losses (fixed batches, fp32; A at stage 0 = 0.0069)

| | A after B<br>seq_ft / allowlist / GPM | B after B<br>seq_ft / allowlist / GPM | B at stage 0 |
|---|---:|---:|---:|
| **S** | 1.14 / 1.13 / **0.45** | 0.0045 / 0.0048 / 0.0052 | 2.01 |
| **D** | 1.11 / 0.93 / **0.047** | 0.0087 / 0.0090 / 0.0096 | 2.92 |
| **S2** | 1.16 / 1.12 / **0.46** | 0.0044 / 0.0048 / 0.0052 | 1.89 |

- **B is fitted about equally well by every arm.** GPM's loss is 10–18% above seq_ft's.
- **On A's own data, GPM held A's loss** at 0.45–0.46 for the similar tasks, against 0.047 for the
  diverse one.
  - That is about 65× A's stage-0 loss, against 7× for the diverse task.
  - It is also about 2.5× lower than seq_ft's.
- **So projection did preserve part of A's velocity field on similar tasks**, but not enough for a
  single successful rollout.

### 3.4 Instruction dependence (P4, reported)

The statistics below are for A's data under B's instruction. B's data under A's instruction is
similar.

| | Velocity sensitivity<br>stage 0 → seq_ft / allowlist / GPM | Chunk relative divergence<br>stage 0 → seq_ft / allowlist / GPM |
|---|---|---|
| **S** | 2.4e-6 → 1.1e-6 / 1.9e-6 / **1.3e-5** | 0.0004 → 0.0005 / 0.0007 / **0.0015** |
| **D** | 2.0e-5 → 1.9e-5 / 2.6e-5 / 2.7e-5 | 0.0012 → 0.0023 / 0.0025 / 0.0017 |
| **S2** | 3.7e-6 → 2.2e-6 / 5.3e-6 / **2.5e-5** | 0.0006 → 0.0009 / 0.0009 / **0.0024** |

- **Every value is tiny.** Swapping the instruction changes sampled chunks by **≤ 0.45%** of their
  scale. For reference (not part of this rule), `language_check` calls a language pathway alive
  from 2%.
- **On the similar tasks, GPM raised instruction dependence** about 5–7× in velocity and 3–4× in
  chunks. seq_ft did not.
- **The raised values are still about an order of magnitude** below anything that could select
  between two behaviours in one scene.

**A's rollouts after B fail by timeout** (median 600 of 600 steps) in every similar-curriculum arm.
Whether the robot executes B's behaviour in A's scene is not labelled: there are no videos and no
swap rollouts.

## 4. Interpretation (not tested by the rule)

- **The diagnostic's mechanism was real, but its consequence was the opposite of the prediction.**
  - GPM's training logs confirm that a similar B's gradient lies mostly inside A's memory. The
    median per-layer norm share is 0.95–0.96 in the trunk for similar B, against 0.91–0.94 for
    diverse B.
  - B was nevertheless learned to 98% through what remained:
    - ε = 0.95 leaves about 5% of each layer's input energy unprotected;
    - the effects of small changes compound across layers.
  - **A shares those residual directions with B.** Same scene, same object, and an instruction the
    network barely reads. So A's outputs moved with B's.
  - **In the diverse case, B's inputs differ from A's** (another scene and object). B can be learned
    along directions A does not use, so A stays protected.
- **In this setting, separating A from a similar B requires the policy to condition on the
  instruction**, and it does not. Hard projection cannot create that separation; it can only stop
  updates along A's inputs.
- **A softer projection (SGP) would not help here.** It relaxes protection further, and plasticity
  is not the bottleneck.
- **Candidate remedies are outside this study:**
  - instruction-dependent capacity (e.g. stronger language conditioning or task-specific inputs);
  - rehearsal.

## 5. Limits and disclosures

- **Scope:** one seed, and one A state shared by identity and resume (all nine runs depend on it).
  Three hand-picked B tasks; similarity and task difficulty are not varied independently.
- **P1 is unpaired:** the similar-minus-diverse contrast compares different B tasks with point
  estimates. Within each curriculum, the differences are rollout-paired.
- **"Fails to protect" is about success rate.** The probe loss shows partial preservation of A's
  velocity field under GPM (§3.3).
- **Instruction influence, not task selection:** the instruction probes measure influence at
  inference. Task selection would need instruction-swap rollouts, which were not run.
- **What A's failed rollouts do is unlabelled.**
- **Adaptive GPM was omitted** as identical to plain GPM over two tasks.
- **GPM's unmoved `flow_head.action_in.weight`** is legitimate (its input is fully spanned) and is
  reported.
