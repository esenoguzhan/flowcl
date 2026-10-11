# LoRA rank × lr bracket (seed 0, T1 → T2): merged LoRA keeps Spatial where EWC kept nothing. The rule selects rank 4, lr 3e-5 (an lr-axis extension, `optimum_at_boundary`): Spatial 0.48, Object 0.72. Retention falls steeply with the adapter learning rate at every rank

**Status: the pre-registered bracket, complete.**
- The queue ended `QUEUE DONE ok`, and the verification passed: inputs, checkpoint fingerprints, and
  the selection recomputed from the validated rows.
- **The registered selection is rank 4, lr 3e-5 (alpha 16), flagged `edge_extension`,
  `optimum_at_boundary`.**
- **How the rule got there:**
  1. On the grid, the selection was (4, 1e-4), with Spatial 0.46, the highest eligible. It lies at
     both the low-rank end and the low-lr end.
  2. That owed two extensions, (1, 1e-4) and (4, 3e-5), which ran in that order.
  3. Re-applied once over all 11 arms, the rule selects (4, 3e-5): Spatial 0.48, Object 0.72.
  4. (1, 1e-4) is ineligible (Object 0.66).
- **This is "the best among tested settings", not a resolved optimum.** The selected arm is on the
  boundary of the lr axis. The three best eligible arms (0.48, 0.46, 0.44) differ by 1–2 of 50
  rollouts, with heavily overlapping intervals.

| rank | lr | Spatial `R[1][0]` [95% CI] | Spatial − seq_ft [CI] | Object `R[1][1]` (floor 0.68) | Object − seq_ft [CI] | Spatial probe loss | Object probe loss | merge ‖ΔW‖/‖W‖, median (max) | adapter params |
|---:|---:|---|---|---:|---|---:|---:|---|---:|
| 1 *(ext.)* | 1e-4 | 0.02 [0.00, 0.06] | +0.02 [0.00, +0.06] | 0.66 ✗ | −0.12 [−0.32, +0.08] | 0.115 | 0.0363 | 0.34 (1.45) | 0.13M |
| **4** *(ext.)* | **3e-5** | **0.48 [0.34, 0.62]** | +0.48 [+0.34, +0.62] | **0.72** | −0.06 [−0.22, +0.10] | **0.087** | 0.0478 | 0.12 (0.46) | 0.51M |
| 4 | 1e-4 | 0.46 [0.32, 0.60] | +0.46 [+0.32, +0.60] | 0.72 | −0.06 [−0.20, +0.10] | 0.113 | 0.0288 | 0.22 (0.71) | 0.51M |
| 4 | 3e-4 | 0.28 [0.16, 0.42] | +0.28 [+0.16, +0.42] | 0.72 | −0.06 [−0.22, +0.10] | 0.133 | 0.0167 | 0.40 (1.33) | 0.51M |
| 4 | 1e-3 | 0.04 [0.00, 0.10] | +0.04 [0.00, +0.10] | 0.68 | −0.10 [−0.28, +0.06] | 0.349 | 0.0129 | 0.90 (2.86) | 0.51M |
| 16 | 1e-4 | 0.44 [0.30, 0.58] | +0.44 [+0.30, +0.58] | 0.88 | +0.10 [−0.02, +0.22] | 0.116 | 0.0274 | 0.16 (0.61) | 2.06M |
| 16 | 3e-4 | 0.20 [0.10, 0.32] | +0.20 [+0.10, +0.32] | 0.64 ✗ | −0.14 [−0.28, 0.00] | 0.157 | 0.0131 | 0.28 (1.17) | 2.06M |
| 16 | 1e-3 | 0.02 [0.00, 0.06] | +0.02 [0.00, +0.06] | 0.76 | −0.02 [−0.18, +0.14] | 0.388 | 0.0086 | 0.65 (2.16) | 2.06M |
| 64 | 1e-4 | 0.20 [0.10, 0.32] | +0.20 [+0.10, +0.32] | 0.90 | +0.12 [+0.02, +0.24] | 0.119 | 0.0269 | 0.14 (0.53) | 8.23M |
| 64 | 3e-4 | 0.08 [0.02, 0.16] | +0.08 [+0.02, +0.16] | 0.74 | −0.04 [−0.20, +0.12] | 0.156 | 0.0123 | 0.25 (1.11) | 8.23M |
| 64 | 1e-3 | 0.00 [0.00, 0.00] | 0.00 [0.00, 0.00] | 0.72 | −0.06 [−0.22, +0.12] | 0.393 | 0.0075 | 0.54 (1.84) | 8.23M |
| *seq_ft (reference)* | *1e-4* | *0.00* | | *0.78* | | *0.725* | *0.0064* | | *(all, 53.8M)* |
| *plain GPM (reference)* | *1e-4* | *0.94* | | *0.78* | | *0.042* | *0.0072* | | |
| *EWC v2, selected (r = 100)* | *1e-4* | *0.00* | | *0.72* | | *0.165* | *0.0096* | | |

**What it shows:**
- **The adapter learning rate dominates retention.** At every rank, Spatial falls as lr rises:
  - rank 4: 0.48 → 0.46 → 0.28 → 0.04 (lr 3e-5 → 1e-3);
  - rank 16: 0.44 → 0.20 → 0.02 (lr 1e-4 → 1e-3);
  - rank 64: 0.20 → 0.08 → 0.00.

  The merged update grows with lr: ‖ΔW‖/‖W‖ has a median of 0.12–0.22 at the lowest lr and 0.54–0.90
  at 1e-3.
- **At lr 1e-4, rank trades Spatial for Object.**
  - Rank 4 gives Spatial 0.46, Object 0.72.
  - Rank 16 gives 0.44 and 0.88.
  - Rank 64 gives 0.20 and 0.90, the only arm whose Object CI excludes 0, above seq_ft.

  At this lr, rank 16 keeps almost rank 4's Spatial with 16 points more Object. The rule ranks
  Spatial first, so the point estimates decide.
- **Lowering lr below 1e-4 adds little Spatial and costs Object loss.** From lr 1e-4 to 3e-5 at
  rank 4, Spatial goes 0.46 → 0.48 (one rollout). Object success holds at 0.72, but Object's probe
  loss rises from 0.029 to 0.048, 7.5 times seq_ft's.
- **Rank 1 is too small** at lr 1e-4: Spatial 0.02, Object 0.66, below the floor.
- **Every arm whose Spatial CI excludes 0 still loses to GPM:**
  - The paired difference from GPM is −0.46 for the selected arm and −0.48 to −0.94 for the others.
  - GPM keeps 0.94 at the same Object (0.78).
- **Probe loss and success:** the selected arm has the lowest Spatial probe loss of any baseline so
  far (0.087), and 48% success.
  - GPM: 0.042 and 94%.
  - EWC's eligible best: 0.165 and 0%.
  - EWC at r = 1000: 0.137 and 4%.
  - This fits the threshold-like relation between loss and success seen with EWC.

**Pre-written interpretations, applied:**
- The selected (rank 4, lr 3e-5) is **the best among tested settings** on seed 0. It sits at the lr
  boundary (`optimum_at_boundary`), and the rule is re-applied only once, so whether a lower lr would
  keep more is untested.
- Spatial > 0 at eligible arms: **observed retention on the development seed.** Broader claims wait
  for the replication seeds.
- **LoRA and GPM are two training procedures on the same weight support** (the 91 registry weights).
  LoRA is a low-rank merged reparametrization with its own lr, AdamW geometry and weight decay; GPM
  uses projected dense updates. So GPM's advantage (0.94 against 0.48) is not attributable to rank
  alone.
- **Descriptive context, not a paired comparison:** dense fine-tuning of the same weights at lr 1e-4
  kept 0/50 Spatial (Spatial probe 0.575, Object 0.78). That is LoRA's expressive limit (§4.1).
  - Source: `results/seq_hetero__freeze_only_pilot__seed0`, from an older pilot on a dirty tree
    (`f01026c-dirty`).
  - At the same lr, rank 4 to 64 kept 0.20–0.46.

**What the rule now implies:**
- **The 4-task LoRA runs use rank 4, lr 3e-5, alpha 16.** They are committed to
  `configs/method/lora.yaml` in their own commit. T1 trains at the recipe lr (1e-4); adapters from T2
  at 3e-5.
- **Production LoRA seed 0's stage 1 is identity-checked against this arm**
  (`--identity-reference-checkpoint 1=results/baseline_selection_lora/lora/seed0/lorab__r4__lr3e-05__seed0/checkpoints/final.pt`).
  If it reproduces the arm, the cell is excluded from Track 2, matched by tensor fingerprint
  `2a5cdb7b…`.
- The 4-task runs wait for Track 2's freeze.

**Limits:**
- one transition on the development seed, which also enters the final evaluation;
- merged adapters, so per-task adapters are not kept (by definition);
- all registry weights adapted, not only attention weights as in Hu et al.'s main experiments;
- the selected setting is on the boundary of the tested lr range;
- differences among the best arms are within rollout noise;
- the cosine recipe and weight decay 1e-4 are shared with every method, not the paper's.

**Date (local, CEST):** Sun 11 Oct 01:39:18 → 13:35:02 (11 h 56 min).
- **Each arm took 61.4–67.0 min**, of which training was 45.9–50.1 min. Full fine-tuning's T2 takes
  about 43 min.
- **The smoke's benchmark** measured ms per step over 30k steps: 85.2 for full fine-tuning, and 101.6,
  97.5 and 104.4 for LoRA at ranks 4, 64 and 256. The adapter hook costs about 15–20% per step; the
  base's weight gradients and optimizer state are saved but the activations are passed over twice
  more.
- The two extensions took 65.3 and 62.4 min. The selections and the verification took 4–13 s.
- **The 15:00 deadline guard** (arm estimate 70 min, extension estimate 72 min) never triggered.

**Code:** `75fb0e1`, clean at queue start. Every row's `git_sha` is `75fb0e1…`.
- **The rule** `configs/analysis/baseline_selection_lora.yaml` has SHA-256 `a6582554…`, committed with
  the code at 01:20, before any arm. There is no calibration.
- **The adapter initialization** is identical for every arm: seed 1599474437, from
  `derive_seed(namespace, "lora_init::<Object>", 1)`. This gives common random numbers across arms.
- **Launch before the suite (disclosed):** the queue launched at 01:39, before the full CPU suite
  finished.
  - The LoRA, bracket, runner, resume, table and method tests had passed (121).
  - The suite then passed at `75fb0e1`: 1064 tests, 0 failures, finished 02:20.
- **Implementation:** the forward hook adds `(α/r)·(x Aᵀ)Bᵀ` to the frozen layer's output, as
  `loralib` computes it. A weight-space form, `F.linear(x, W + (α/r)BA)`, is the same function and
  would avoid the extra activation passes. A draft of it passed the CPU tests but was neither
  benchmarked nor adopted, so that the smoke-validated reference form stayed in place.

**The smoke** (scratch, 01:20–01:38, on `75fb0e1`, 200 steps, 1 episode; `QUEUE DONE ok`):
- **LoRA is bitwise seq_ft at stage 0** (0 of 656 tensors), with T1 pairing at relative difference
  0.0000.
- **The per-stage records:** lr 1e-4 at T1 and 1e-3 from T2. Trainable parameters were 53.8M at T1
  and 0.51M from T2 at rank 4.
- **Resume equality at stage 3:**
  - the tensors match (0 of 656);
  - the method history is equal (restored tasks 1–2);
  - every evaluation episode is equal.
- **Bracket against runner:**
  - an arm from the smoke's seq_ft stage 0 trained under a TrainConfig equal to the runner's stage 1
    (200-step schedule, lr 1e-3);
  - it reproduced the runner's `stage1.pt` (0 of 656 tensors) and its history.

**Cite:**
- `results/baseline_selection_lora/lora/selection/selection.json` (SHA-256 `a1d62334…`, inside the
  immutable bundle);
- the arms `results/baseline_selection_lora/lora/seed0/lorab__r{1,4,16,64}__lr*__seed0/row.json`;
- the queue log `results/logs/queue_20261011_013918_lora_bracket/queue.log`;
- the smoke log `results/logs/queue_20261011_012042_lora_bracket_smoke/queue.log`.

---

## Design (as pre-registered; `configs/analysis/baseline_selection_lora.yaml`)

- **The method: sequential merged LoRA** (Hu et al., ICLR 2022, checked against the paper and
  `loralib`).
  - T1 is full fine-tuning (bitwise seq_ft).
  - From T2, the base is frozen. Each task gets fresh rank-r adapters on the 91 registry `nn.Linear`
    weights, GPM's T2 weight support.
  - **Initialization:** A uses the released `kaiming_uniform_(a=√5)` on its own generator; B = 0.
  - **The scale** is α/r with α = 16, constant in r and not tuned.
  - **Training:** the adapters train at the method's stage lr and are merged into W in fp32 at the
    task's end.
  - **Recorded deviation:** r × lr is bracketed, not r only, because Hu et al. tune the lr per method.
- **The arms:**
  - seed 0, from seq_ft's `stage0.pt`;
  - T2 (Object) trained with the reference recipe (30k steps, warmup + cosine, weight decay 1e-4) at
    the arm's lr, through `train_stage` with the runner's text cache;
  - cells paired with seq_ft's stage 1.
- **The selection:**
  - **Eligible:** Object ≥ seq_ft − 0.10 (0.68).
  - **Order:** Spatial success, then Spatial probe loss, then the smaller rank (lower adapter
    capacity), then the smaller lr.
  - **None eligible:** the highest Object wins, with no extension.
  - **Extensions:** one beyond each edge on which the grid's selection lies (no combined corner),
    then the rule is re-applied once.
  - **Binding:** rows bind the shared and arm-specific inputs and a tensor-content fingerprint, and
    `verify` recomputes the selection.
- **The queue's GPU-window guard** stops between arms and publishes nothing past the deadline.
- **Track 2 exclusions** are by tensor-content fingerprint:
  - the shared T1 cells;
  - every bracket cell;
  - production cells that reproduce one;
  - scratch smoke outputs.
