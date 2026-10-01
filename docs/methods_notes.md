# Methods notes

These are implementation facts that the thesis methods section must state. Each was found while
running a study, and each note points to the record that measured it. They change no reported
result unless the note says so.

## 1. AdamW's weight decay is a no-op in every registered run

**The fact.** AdamW's decoupled decay multiplies the weights by `1 − lr·wd` at every step.
- The registered recipe has lr 1e-4 and wd 1e-4, so the factor is `1 − 1e-8`. That rounds to
  exactly 1.0 in fp32, so the decay never changes a weight.
- At the low-update rate (lr 3e-5) the factor is `1 − 3e-9`, which also rounds to 1.0.

**The measurement.** In the optimizer study, a wd-0 run was bitwise identical to the wd-1e-4 run:
0 of 656 tensors differ.
- wd 1e-2 is resolvable (237 tensors change), but its effects are negligible.
- Source: `docs/runs/2026-09-29_optimizer_study.md` §6.

**For the thesis.** The recipe table must say that the weight decay was effectively off throughout
the project. Reporting "AdamW, wd 1e-4" without that note would imply a regularizer that did
nothing. No result changes.

## 2. SGP under AdamW: the scaled directions take Adam's step budget

**The adaptation.** Our SGP removes only the exactly hard bases (λ = 1) from the gradient before
AdamW. It scales the other memory directions *after* the step.
- Adam's moments therefore still see the protected components.
- Those components dominate the raw gradient (fine-tuning's realized trunk update is about 85% in
  the memory by norm at T1 → T2, and about 99% at T4 and T5).
- So Adam's normalized step lies mostly inside the memory, and the post-step scaling removes most
  of it. What reaches the free directions is smaller than under GPM, which removes every memory
  direction *before* Adam.

**Measurement points (one seed each; associative except where stated):**

| Transition | Record | What was measured |
|---|---|---|
| T1 → T2 | `docs/runs/2026-09-29_sgp_sensitivity.md` §4 | At α = 1600 almost every λ ≥ 0.99, yet the trunk's relative update is 0.23, against GPM's 0.39. Hard GPM is not the α → ∞ limit of our SGP. |
| T4 | `docs/runs/2026-10-01_sgp_coupling.md` §7 | Under AdamW the scaled projection's free-direction update is 35% smaller in the trunk than hard projection's (37.3 against 57.3), with a ×1.26 worse T4 fit. Under the exact SGD rule the free component is about unchanged, and the cost is gone (Δ = −0.27 [−0.35, −0.22]). This is the one **controlled** contrast: a change of update rule, not a cross-arm association. |
| T5 (saturated) | `docs/runs/2026-10-01_t5_sweep.md` §5 | AdamW's raw step lies 87–93% inside the memory, against 27–39% for hard. The free component grows with α (trunk 22.3 → 39.9) but stays below hard's 58.2 at every α. |

**For the thesis.**
- "SGP" in the tables means **our AdamW adaptation** of SGP. Differences from GPM at high α are
  optimizer-coupling effects, not effects of softness.
- The exact SGD rule removes the T4 cost but is not practical here: it underfits T4 (78–80%; probe
  loss 3× AdamW's).
- A coupling-aware AdamW variant, which would remove the scaled components from what Adam's
  moments see, would be a new method. It is not planned (SGP is closed, `docs/thesis_plan.md` A7).

## 3. The text-encoder cache keeps the precision of the first encoding

**The mechanism.** The frozen text encoder (`flowcl/models/encoders.py`, `CachedTextEncoder`)
caches each instruction's embedding on the CPU the first time it sees the string. It keeps
whatever precision was active at that moment, AMP or fp32, and every later use gets that
embedding.

**In a continual run** (the runner):
- T1's instruction is first encoded inside T1's first AMP training step.
- Every later task's instruction is first encoded in fp32, by the evaluation after T1. The runner
  evaluates every task after every stage.

**A fresh stage process** (the stage path used by the hparam, low-update, optimizer, sensitivity
and projection-pilot studies) instead first encoded the stage's instruction inside an AMP training
step. The embedding then differs in its last bits, and the trajectory drifts from step 0.

**The fix and its scope** (`1e5ebf6`).
- `warm_text_cache` encodes the stage's instruction in fp32 before training, as the runner's
  evaluations did.
- `warm_runner_cache` (`flowcl/experiments/low_update.py`) recreates the runner's whole state:
  T1 under AMP, the rest in fp32.
- With it, stage-path training reproduces a runner stage bitwise: 0 of 656 tensors differ, and all
  601 printed losses and 54,600 c-values match (`docs/runs/2026-10-01_sgp_coupling.md` §2–3).
- With it, a replay of GPM's memory capture reproduces `memory_task{t}` bitwise
  (`docs/runs/2026-10-01_t5_sweep.md` §2).

**For the thesis.**
- The earlier stage-path studies trained without the fix. Their new runs therefore differ from
  runner-trained stages at the level of a different floating-point realization: a few episodes per
  task.
- This is not a systematic bias. It matters only where those studies compared a new run with a
  reused runner arm, and their records are read with that in mind.
- From the coupling study on, the stage path records its cache state in every row.
