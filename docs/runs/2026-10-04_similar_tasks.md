# Similar-task diagnostic: similar tasks overlap more with A's GPM memory (P1, P2 hold), but by modest margins; the instruction barely reaches the output

**Status: exploratory, one A policy, one seed, no continual training, no gate.** The rule is
`configs/analysis/similar_tasks.yaml`. All measurements are at a single-task policy A
(`libero_goal/put_the_bowl_on_the_plate`, Gate 0 **98% [94, 100]**) against its input memory
`M_A` (ε = 0.95).

| Category | Task B | Trunk input overlap (median) | Blocked gradient c, trunk (energy) | Norm surviving `√(1 − c²)` | Energy removed `c²` |
|---|---|---:|---:|---:|---:|
| similar | bowl → stove | **0.937** | **0.949** | 0.315 | 0.901 |
| similar | bowl → top of cabinet | **0.936** | **0.961** | 0.278 | 0.923 |
| same scene | open the middle drawer | 0.906 | 0.947 | 0.321 | 0.897 |
| same object and goal | Spatial: black bowl → plate | 0.913 | 0.942 | 0.335 | 0.888 |
| diverse | Object: milk → basket | 0.874 | 0.923 | 0.386 | 0.851 |
| diverse | LIBERO-10: stove + moka pot | 0.880 | 0.926 | 0.378 | 0.857 |
| **similar − diverse** | | **+0.059** | **+0.031** | −0.086 | +0.058 |

**Answers to the user's question (descriptive, at this checkpoint):**
- **Similar tasks do sit more inside A's protected space,** but the margin is small, because even
  diverse tasks already overlap heavily.
  - Hard projection would remove about **90–92%** of a similar task's trunk gradient energy, leaving
    28–32% of its norm. For a diverse task it would remove about **85%**, leaving 38–39%.
  - The follow-up threshold (similar-task mean c ≥ 0.95) is met by the **mean, 0.955**. That is
    borderline: one similar task is at 0.949, the other at 0.961.
- **A's policy already partly fits the similar tasks.** Their trunk gradient norms at A are small
  (0.64–0.93), against 3.8–7.3 for the diverse tasks. That is consistent with shared skills; it is
  not a transfer measurement.
- **The instruction is the only signal that tells A from a similar B, and it is weak.**
  - Swapping "plate" for "stove" changes the trunk inputs by **0.3–0.8%** of their energy (diverse
    instructions: 2.4–4.0%).
  - **19–27%** of that change lies outside `M_A` (diverse: 14–15%), so there is a little free room
    in those directions.
  - **The policy's output barely responds.** The velocity changes by 2–3×10⁻⁶ of its energy for the
    similar swaps, and 1–3×10⁻⁵ for the diverse ones. A single-task policy has essentially learned
    to ignore its instruction.
- **Gradient alignment (P3) cannot be judged.** A has converged, so its gradient is noise: norm
  0.019 and split-half self-consistency −0.02. Every A–B cosine is therefore `unstable` by the
  declared rule. The raw values sit near 0 (−0.11 to +0.03).

**Date (local, CEST):** Sun 4 Oct 2026, 22:29:19 → 23:39:47.
- A's training and evaluation 43 min;
- the memory step 2.5 min;
- six measurements 3–5 min each;
- the summary 2 s.

**Code:** `044048c`, clean at queue start; every step's `git_sha` is `044048cf…`. The rule
`configs/analysis/similar_tasks.yaml` (SHA-256 `6827a695…`) was committed with the implementation.
Its SHA was locked before the smoke run and was unchanged after it.

**Cite:**
- `results/similar_tasks/summary.json` (SHA-256 `ddc126d8…`);
- the per-B results in `results/similar_tasks/tasks/<task>/result.json`;
- `results/similar_tasks/memory/memory.json`: A's checkpoint SHA `ad72cf8b…`, `M_A` SHA `8b5c8a45…`,
  A's gradients SHA `67340a45…`.

---

## 1. Design (as pre-registered)

- **Task A** was trained with the Gate 0 recipe (30k steps, lr 1e-4, batch 64, AMP, seed 0, all
  parameters).
- **`M_A`** is A's ε-thresholded input basis per registry layer, from A's own data (primary view).
  It is a **controlled fp32 diagnostic** using GPM's capture function and ε, not a bitwise
  reproduction of a GPM run's memory: this study's seed tags, and every instruction encoded in fp32
  before any capture.
- **For each task B, at A's checkpoint, with A's frozen statistics:**
  1. the input overlap `tr(M_Aᵀ K_B M_A)/tr K_B`;
  2. Gate 3's `c = ‖G_∥‖/‖G‖` (primary: trunk, pooled energy; full-dataset gradient reported);
  3. the local gradient alignment (cosine of the full-dataset gradients), with split-half stability;
  4. a paired instruction swap:
     - the same observations and s/noise under the native instruction and the other task's;
     - per layer, the size `tr D / tr K_ref` and the share outside memory;
     - per comparison, the output-velocity sensitivity, pooled over valid action positions.
- **Predictions:**
  - P1: similar overlap ≥ diverse;
  - P2: similar c > diverse;
  - P3: similar alignment > 0, diverse ≈ 0;
  - P4: the instruction signal is measured, with no direction predicted.

## 2. Validity (all passed)

| Check | Result |
|---|---|
| A learned the task | Gate 0 98% [94, 100] (≥ 80%: not `A_weak`) |
| Self-control on the Gram `M_A` was built from | every layer ≥ 0.95 − 1e-6 (worst 0.9500002); median trunk 0.9502, decoder 0.9504 |
| Independent stochastic recapture | trunk 0.9502 (identical by construction, §6.3), decoder median 0.9507 |
| A's checkpoint and `M_A` pinned | every measurement verified both SHAs |
| Weights unchanged | every step |
| Samples per dimension ≥ 10 | enforced (production) |
| Clean provenance | queue at `044048c`, 0 changes; `git_sha` clean in every record |
| Smoke run (scratch, discarded) | first attempt: the LIBERO-10 measurement failed (the 192-sample smoke subset held no padded samples for the reachability probe), so the smoke subset was changed to take the dataset's tail. Second attempt: `QUEUE DONE ok`. Production uses full datasets and was unaffected. |

## 3. Results

**Input overlap with `M_A`** (fraction of B's input energy inside A's memory):

| Task B | Trunk median | Trunk parameter-weighted | Trunk median at ε = 0.99 | Decoder median |
|---|---:|---:|---:|---:|
| bowl → stove (similar) | 0.937 | 0.934 | 0.986 | 0.923 |
| bowl → cabinet top (similar) | 0.936 | 0.933 | 0.986 | 0.927 |
| open middle drawer (same scene) | 0.906 | 0.901 | 0.977 | 0.873 |
| Spatial bowl → plate (same object and goal) | 0.913 | 0.910 | 0.980 | 0.925 |
| Object milk → basket (diverse) | 0.874 | 0.869 | 0.971 | 0.892 |
| LIBERO-10 stove + moka (diverse) | 0.880 | 0.875 | 0.971 | 0.867 |

The forward GPM run's diverse transitions logged 0.90–0.93 (trunk). That is unpaired context: a
different checkpoint and memory.

**Blocked gradient (Gate 3, at ε = 0.95):**

| Task B | Trunk c (energy / full) | Decoder c (energy) | Trunk gradient norm at A |
|---|---|---:|---:|
| bowl → stove | 0.949 / 0.965 | 0.973 | 0.93 |
| bowl → cabinet top | 0.961 / 0.983 | 0.977 | 0.64 |
| open middle drawer | 0.947 / 0.957 | 0.964 | 4.52 |
| Spatial bowl → plate | 0.942 / 0.959 | 0.964 | 0.92 |
| Object milk → basket | 0.923 / 0.925 | 0.962 | 7.27 |
| LIBERO-10 stove + moka | 0.926 / 0.935 | 0.957 | 3.78 |
| A on A (self) | 0.962 / 0.961 | 0.928 | 0.008 |

**Local gradient alignment** (trunk; A's split-half self-consistency −0.025; B's 0.95–1.00):

| Task B | cos(G_A, G_B) | Flag |
|---|---:|---|
| bowl → stove | −0.046 | unstable |
| bowl → cabinet top | −0.028 | unstable |
| open middle drawer | −0.046 | unstable |
| Spatial bowl → plate | −0.009 | unstable |
| Object milk → basket | +0.027 | unstable |
| LIBERO-10 stove + moka | +0.030 | unstable |

**Instruction signal** (paired swap; trunk medians; the reference is the native instruction):

| Swap | On | Size `tr D / tr K_ref` | Share outside `M_A` | Output sensitivity |
|---|---|---:|---:|---:|
| plate → stove | A's data | 0.0026 | 0.263 | 2.8e-6 |
| stove → plate | B's data | 0.0025 | 0.274 | 1.7e-6 |
| plate → cabinet top | A's data | 0.0082 | 0.192 | 3.3e-6 |
| cabinet top → plate | B's data | 0.0081 | 0.202 | 2.4e-6 |
| → open middle drawer | A's data | 0.0358 | 0.190 | 3.0e-5 |
| → Spatial bowl → plate | A's data | 0.0244 | 0.140 | 9.4e-6 |
| → milk → basket | A's data | 0.0267 | 0.154 | 1.6e-5 |
| → LIBERO-10 stove + moka | A's data | 0.0403 | 0.138 | 2.5e-5 |

- **Defined layers:** 48 of 49 in the trunk (`trunk.state_projection`'s input does not change) and
  36 of 42 in the decoder.
- **Decoder sizes are tiny** (3×10⁻⁶ to 4×10⁻⁵), with about 36–41% of them outside memory.
- **The similar swaps are a few words long and change the activations least.** A larger fraction of
  their (small) difference is outside the memory.

**Categories** (mean over tasks; differences against diverse):

| Quantity | similar | same scene | same object and goal | diverse | similar − diverse |
|---|---:|---:|---:|---:|---:|
| trunk overlap | 0.937 | 0.906 | 0.913 | 0.877 | **+0.059** |
| trunk c (energy) | 0.955 | 0.947 | 0.942 | 0.924 | **+0.031** |
| instruction signal outside `M_A` (A's data) | 0.227 | 0.190 | 0.140 | 0.146 | +0.081 |
| output sensitivity | 3.0e-6 | 3.0e-5 | 9.4e-6 | 2.1e-5 | −1.8e-5 |

## 4. Predictions

| | Prediction | Result |
|---|---|---|
| P1 | similar trunk overlap ≥ diverse | **holds** (0.937 against 0.877) |
| P2 | similar blocked c > diverse | **holds** (0.955 against 0.924) |
| P3 | similar alignment > 0, diverse ≈ 0 | **not assessable**: every cosine is `unstable` (A's converged gradient is noise) |
| P4 | instruction signal (measured) | small for similar swaps (0.3–0.8% of trunk input energy), 19–27% of it outside `M_A`; output response 2–3×10⁻⁶ |

**The follow-up threshold** (similar mean trunk c ≥ 0.95) is **met**, at 0.955: borderline, with
the two similar tasks at 0.949 and 0.961. The candidate study is a two-task A → B run, **seq_ft
against GPM** (about 2.5 h each). It would be decided separately, under its own rule.

## 5. What this does and does not show

**It shows,** at this A checkpoint, for these tasks:
- similar tasks (the same scene, object and kind of goal) have more activation and raw-gradient
  overlap with A's protected space than diverse tasks;
- hard projection would remove about 90–92% of a similar task's gradient energy (diverse: about
  85%);
- similar tasks have much smaller gradients at A;
- the instruction-induced difference between similar tasks is small, is partly outside the memory,
  and moves the output velocity by almost nothing.

**It does not show:**
- **that GPM would fail to learn a similar task,** or would learn it through the free directions.
  Raw-gradient overlap is not an optimizer step, and free representational room is not learnability.
- **anything about SGP:** the AdamW step-budget coupling concerns optimizer steps, which were not
  measured.
- **positive transfer:** the alignment is not assessable, and gradient norms are only suggestive.
- **generality:** one A policy, one seed, two similar tasks.

**For the user's hypothesis** ("similar tasks share spaces, so GPM might fail and SGP might work"):
- the shared-space part holds here, but modestly;
- the instruction, the only thing that separates the tasks, barely affects a single-task policy's
  output. A continual learner would have to *create* that dependence while learning B. That is
  exactly what a two-task study would test.

## 6. Disclosures

1. **Rule timing.** The rule, the code and the Gate 3 extension were committed in `044048c` before
   the registered run. The rule's SHA was unchanged across both smoke attempts. The smoke fix
   (`TailSubset`) changed the smoke path only.
2. **Controlled diagnostic.** `M_A` is not a GPM run's memory, and the text cache is fp32 for every
   instruction (the config header states both).
3. **The recapture.** It varies only the capture seed, which drives the flow time and the noise.
   The trunk never sees either, and the token subsample seed is fixed by the capture config. So the
   trunk recapture equals the self-control by construction (0.9502 both). Only the decoder
   recapture varies (median 0.9507 against 0.9504). It is therefore not an independent check of the
   trunk.
4. **The summary's printed output sensitivities read 0.0** (rounded to four decimals). The exact
   values, 1.7×10⁻⁶ to 3.0×10⁻⁵, are in the per-task results and in this record.
5. **The follow-up threshold** is met by the mean of two similar tasks that sit on either side of
   0.95. It is a descriptive trigger; nothing ran automatically.
6. **Statistical scope.** Gate 3's bootstrap CIs (per layer, in the results) describe batch
   sampling at this checkpoint, not variation across seeds or task families.

## 7. Artifact map

```
results/similar_tasks/summary.json                        # (SHA-256 ddc126d8...)
results/similar_tasks/memory/{memory.json, M_A.pt, grad_A.pt}
results/similar_tasks/tasks/<suite>__<task>/result.json   # six task-B results
results/single__libero_goal__put_the_bowl_on_the_plate__seed0/   # task A's policy (Gate 0)
results/gate0_similar/gate0.json                          # A's 50 rollouts
results/logs/queue_20261004_222919_similar_tasks/
configs/analysis/similar_tasks.yaml (044048c; SHA-256 6827a695...), flowcl/experiments/similar_tasks.py,
flowcl/experiments/gate3.py (keep_split_gradients), scripts/similar_tasks.py, scripts/queue_similar_tasks.sh,
tests/test_similar_tasks.py
```
