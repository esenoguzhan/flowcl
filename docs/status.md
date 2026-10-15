# Status: where the thesis stands (Thu 15 Oct 2026, week W2 of `docs/thesis_plan.md`)

This is the handoff page: what is done, what is parked, what comes next. The plan itself is
`docs/thesis_plan.md`; every result has a record in `docs/runs/`. Update this page when the state
changes.

**Deadline:** submission ~1 Mar 2027. Experiments freeze at W17 (25 Jan).

**Right now:** back on the plan.
- **The Dobot demo was cancelled on Wed 7 Oct:** the robot is not available.
- **AgileX stays the Stage B robot.** The user is setting up the arms in the lab; hardware
  deployment comes later.
- **The Dobot try-out is parked on branch `dobot-hw`** (not merged, never run on the robot).
  Its reusable pieces are listed in §3.
- **C1 and its follow-up are finished** (records `runs/2026-10-08_high_protection_three_seeds.md`,
  `runs/2026-10-09_high_protection_followup.md`).
- **Replay and EWC are implemented, and EWC's λ is decided.**
  - v1 bracket: `runs/2026-10-10_ewc_bracket.md`; Spatial 0/50 at every λ.
  - v2 follow-up (sampled Fisher, per-task normalization, wider grid, probe-loss tie-break),
    finished Sun 11 Oct 00:39: `runs/2026-10-11_ewc_bracket_v2.md`. Spatial is still 0/50 at every
    eligible λ.
  - It selects r = 100 (λ = 0.001227), now in `configs/method/ewc.yaml`.
- **LoRA is implemented and its lr is decided.** Merged LoRA keeps Spatial where EWC kept nothing.
  - v1 bracket: `runs/2026-10-11_lora_bracket.md`; rank 4, lr 3e-5, at the lr boundary.
  - v2 follow-up (post hoc; lr 1e-5 and 3e-5 at ranks 4–64), finished Mon 12 Oct 03:10:
    `runs/2026-10-12_lora_bracket_v2.md`.
  - **Selected: rank 16, lr 3e-5,** inside the grid, with Spatial 0.52 and Object 0.70 (GPM: 0.94
    and 0.78). It is in `configs/method/lora.yaml`.
  - At lr 1e-5, Object falls below the floor at every rank. Retention falls steeply with lr.
- **Order decided on Sun 11 Oct:** the 4-task replay, EWC and LoRA runs are Track 2's held-out
  test, so they run after Track 2's freeze.
- **Track 2 is FROZEN (Thu 15 Oct 21:59; `runs/2026-10-15_track2_selection.md`).** Steps 1–3 are done:
  the manifest (449 cells), the measures M0–M3, and the selection (rule `1a6b6de`, written before any
  fitting; bundle `b397bc0a…`, verified).
  - M0 already ranks within tasks at C_a 0.82 and is miscalibrated across tasks by 8 pp.
  - Carried to the held-out test: `m2_position` (a: +0.008) and `m3_mean` (b: +0.016). Both
    development intervals include 0, and `m3_mean` fails root weighting, so "not shown" is the likely
    held-out result.
  - **Next:** the 4-task baseline runs (the test set) can be queued; then step 4, the held-out test.
- **The projected-LoRA comparison is done** (Wed 14 Oct 07:44; `runs/2026-10-14_projected_lora_comparison.md`).
  No new mechanism is claimed. T1 -> T2 on seed 0, low-rank adapters confined to the complement of
  GPM's Task-1 memory:
  - all three families keep Spatial at 0.90–0.98 (plain LoRA: 0.52);
  - selected: `inflora` (16, 1e-4) 0.90 / 0.82 with 1.0M trainable parameters and about 35% less peak
    GPU memory; `pca_trained` (16, 1e-4) 0.98 / 0.82; `projected_lora` (64, 3e-4) 0.94 / 0.70;
    `gpm_projected_adam` 0.94 / 0.78;
  - unfreezing InfLoRA's A helped at every lr (8 of 8 comparisons equal or higher); every variant is
    10–25% slower per step.
  - The 4-task runs are reserved for Track 2's test set, so they wait for the freeze.

## 1. Plan progress

| Item | State | Record / note |
|---|---|---|
| **A1** stage-boundary resume | done (confirmatory test passes) | `runs/2026-10-02_resume_check.md` |
| **A2** `seq_hetero_t8`, Gate 0 T6–T8 | done (T6 microwave, T7, T8 pudding → drawer) | `runs/2026-10-02_gate0_t8*.md` |
| **A3** seen-task-only evaluation (`eval_unseen: false`) | **not started** | |
| **A4** ε schedule for GPM (`eps_later`, `gpm_projected_adam_e99`) | done | `configs/method/gpm_e99.yaml`; CPU test: stages 0–1 and T1 memory bitwise equal to plain |
| **A5** rollout-level paired bootstrap for ACC / NBT / AUC, per-seed tables | **not started** | |
| **A6** metric naming (ACC, F_first) in tables and README | **not started** | |
| **A7** close SGP | done | `runs/2026-10-01_t5_sweep.md` |
| **B** baselines: replay, EWC, LoRA, ConSFT | **replay and EWC implemented** (`2c2b827`, `aaedf24`, `5f1d124`); smoke passed (stage-0 identity, resume equality). **EWC v1 bracket** (empirical Fisher, r ≤ 100): Spatial 0.00 at every λ. **EWC v2 bracket** (sampled Fisher, normalized, r ≤ 10000): Spatial 0.00 at every eligible λ (r ≤ 100); 2/50 only at r ≥ 1000, where Object falls to 0.44 / 0.36. At equal r, v1 and v2 trace the same trade-off. **Selected: r = 100, λ = 0.001227** (probe-loss tie-break, `all_eligible_spatial_zero`); Object 0.72, Spatial probe loss 0.165 (GPM 0.042). **LoRA** (`75fb0e1`) **v1 bracket:** rank {4,16,64} × lr {1e-4,3e-4,1e-3} plus (1, 1e-4) and (4, 3e-5); Spatial falls with lr at every rank; rank 4, lr 3e-5 at the boundary. **v2** (`620727b`, post hoc): lr 1e-5 and 3e-5 at ranks 4–64; at 1e-5 Object < floor at every rank. **Selected rank 16, lr 3e-5:** Spatial 0.52, Object 0.70; the top eligible arms are within noise (0.52 / 0.48 / 0.46 / 0.44). 4-task replay/EWC/LoRA runs wait for Track 2's freeze; ConSFT not started | `runs/2026-10-10_ewc_bracket.md`, `runs/2026-10-11_ewc_bracket_v2.md`, `runs/2026-10-11_lora_bracket.md`, `runs/2026-10-12_lora_bracket_v2.md` |
| **C1** high-protection control (e99 × 3 seeds) | done: **`unresolved`** ×3 (replication label `mixed`, a rule gap). At matched T2 protection (as predicted, 91/91 ranks) a flat 0.99 reproduces the Object gain over plain on every seed (G +0.70 / +0.62 / +0.48), indistinguishable from adaptive (D +0.04 / −0.02 / −0.08), non-inferiority at 10 pp not shown at 50 rollouts. Descriptive: adaptive keeps more final Object (−0.14 / −0.16 / −0.22 for e99) at ~15 pp more capacity. **Follow-up** (3 repetitions of the 50 initial states, second-stage, decided after C1): `protection_suffices` / `unresolved` / `unresolved` + significant `adaptive_advantage` (D +0.03 / −0.05 / −0.10), replication `mixed`: the allocation's effect at this cell is small and seed-dependent | `runs/2026-10-08_high_protection_seed0.md`, `runs/2026-10-08_high_protection_three_seeds.md`, `runs/2026-10-09_high_protection_followup.md` |
| **C2** reverse order × 3 seeds | done: `inconclusive` / `task` / `both` (no three-seed recurrence; Object lost across Spatial every seed; adaptive trade replicates) | `runs/2026-10-0{3,4}_reverse_order_seed*.md` |
| **D** Track 2 (offline predictors) | **step 2 done: the measures M0–M3** (`d213ddb`/`99e8083`): 449 cells, bundle `5be958cc…`, instrument checks 281/281 within 2.1e-5; descriptive only (`runs/2026-10-13_track2_measures.md`). **Step 1 done: the development manifest** (`2ec4c03`, merged): 449 cells (checkpoint × trained task, 50 rollouts) under 24 roots; seed 0's shared stage 0 holds 214; 63 bitwise duplicates collapsed (one disagrees by 2/50 rollouts); 114 depth-1 / 147 first-divergence branches (branch definition changed in `thesis_plan.md` §D, the literal one kept as sensitivity); held-out key set published. **Next: step 3's rule**, then validation and the freeze | freeze target Wed 28 Oct, **hard deadline Fri 30 Oct**; `runs/2026-10-12_track2_manifest.md` |
| **E1** two-sided go/no-go | done: **`fail`** (`near_full_rank`; trunk 0.861 vs GPM 0.933) | `runs/2026-10-02_twosided_gonogo.md` |
| **E2** pilots | not started; angle A is off | D2 (end of W5) is angle B (only if Track 2 finds critical outputs) or the minimum thesis |
| **F0** hardware readiness | **not started on AgileX**. The Dobot try-out was cancelled before any robot time | branch `dobot-hw`: `runs/2026-10-06_dobot_hw_prep.md` |
| **D1** supervisor: minimum thesis, claim form, hardware scope, proof-gate thresholds | not recorded here | |

## 2. Side line, outside the plan: similar tasks and language (closed for now)

Three studies, 4–6 Oct:
1. **`runs/2026-10-04_similar_tasks.md`:** similar tasks overlap more with A's GPM memory, by
   modest margins. The instruction barely reaches a single-task policy's output.
2. **`runs/2026-10-05_similar_cl.md`:** A → similar B. Every arm, GPM included, learns B and
   loses A completely (0.94 → 0.00). On a diverse B, GPM keeps A at 0.92.
3. **`runs/2026-10-06_language_cl.md`:** a base trained jointly on A + cabinet top **does select
   by instruction**.
   - **Swap drops:** 0.90 and 1.00. **Requested goals reached:** 1.00 and 0.94.
   - So the language pathway works; the earlier blindness came from single-task stages.
   - From that base, GPM keeps the cabinet task (0.98) but still loses A (plate) to stove
     (0.90 → 0.00). The base never learned the plate/stove contrast. Motion similarity is
     confounded with it.

**Parked follow-ups** (not decided; none is in the plan):
- allowlist_ft from the joint base (~2.7 h);
- a broader LIBERO-Goal base;
- the diverse-base control;
- a joint A + S2 + S control.

**The in-plan remedy is replay:** add A → stove to replay's runs once replay exists (~2 h).

**Lesson for hardware task choice (F2):** if two hardware tasks share a setup and differ only by
instruction, single-task stages give the policy no reason to read it, and every method will appear
to forget. Choose tasks that differ visually, or plan for joint/replay training.

## 3. The hardware gate (F0/F1 on AgileX): what it must establish

The plan's gate (`thesis_plan.md`, Workstream F, F1):
1. **The data format:**
   - cameras, proprioception;
   - action space (joint or end-effector, absolute or delta, both grippers);
   - control rate, timestamps.

   This defines `configs/embodiment/<robot>.yaml` (D_action not padded) and the dataset adapter.
2. **Open-loop replay of 3 recorded demos** reproduces their trajectories (action convention,
   frame, timing).
3. **The single-task proof:**
   - ~15 demos of one task, trained from scratch (the LIBERO policy does not transfer to another
     embodiment);
   - 10 closed-loop rollouts on fixed initial conditions;
   - **≥ 6/10 successes**, episode cycle (execution, reset, annotation) **≤ 4 min**.
4. **Record:** what was learned (format, rates, measured cycle time, robot access including
   W12–W13) goes in `docs/runs/` and back into this page.

**Reusable from the parked Dobot try-out** (branch `dobot-hw`; record
`docs/runs/2026-10-06_dobot_hw_prep.md` on that branch):
- **Code:**
  - `TaskRef` suite dispatch for non-LIBERO tasks;
  - a LeRobot v2.1 adapter with an HDF5 cache;
  - `stats_std_floor` in the embodiment spec;
  - `gate0 --no-eval`;
  - a websocket policy server with a status page;
  - a robot-side runner (probe without motion, stop test, replay with tracking metrics, operator
    rollout loop with cycle time; safety checks on every chunk);
  - an offline open-loop check.
- **Lessons for AgileX F1:**
  - Make the training state match exactly what the robot can send live (the Dobot's gripper state
    was the last *command*).
  - Frozen Task-1 stats can put later tasks' joints many stds out on a joint-position embodiment.
  - A `--no-eval` `run_continual` writes no `result.json`, so non-seq_ft methods need
    `--no-t1-check`.
  - h5py step-sliced reads (`[::k]`) are very slow; read contiguously, then subsample.
  - Measure demo joint speed against the robot's per-tick clip before choosing it.
- **Disk:** `datasets/dobot/` (69 GB) and the Dobot runs in `results/` (about 6.7 GB) can be
  deleted if space is needed.

## 4. Next, when back on the plan (priority order)

1. **Track 2 (D):** brought forward (see item 3). The baselines' brackets are done; their 4-task runs
   follow the freeze. Before the 4-task baseline runs, `sequence_report` must become
   method-conditional (`frozen_from_stage1`, the pilot comparison).
2. **Thesis wording from C1:** raising T2 protection removes the Object forgetting; the new-energy
   allocation adds at most a small, seed-dependent amount at that cell (significant on one seed of
   three). Its consistent edge is final Object retention at about 15 pp more capacity (E3's frontier).
3. **Track 2 (D):** manifest and measures, frozen by Fri 30 Oct at the latest (sooner if possible).
   Angle B at D2 depends on it. After the freeze: the 4-task replay, EWC and LoRA runs.
4. **A3, A5, A6:** evaluation and metric fixes, needed before the 8-task runs (E3).
5. **ConSFT notes** (W5).
6. **Hardware:** F1 engineering on AgileX (adapter, `hw_env`, proof) continues in parallel to the GPU work; reuse the `dobot-hw` pieces (§3).
