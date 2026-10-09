# Status: where the thesis stands (Fri 9 Oct 2026, week W1 of `docs/thesis_plan.md`)

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
  `runs/2026-10-09_high_protection_followup.md`). Nothing is queued on the GPU.

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
| **B** baselines: replay, EWC, LoRA, ConSFT | **not started** (no method code yet) | brackets planned for W3 |
| **C1** high-protection control (e99 × 3 seeds) | done: **`unresolved`** ×3 (replication label `mixed`, a rule gap). At matched T2 protection (as predicted, 91/91 ranks) a flat 0.99 reproduces the Object gain over plain on every seed (G +0.70 / +0.62 / +0.48), indistinguishable from adaptive (D +0.04 / −0.02 / −0.08), non-inferiority at 10 pp not shown at 50 rollouts. Descriptive: adaptive keeps more final Object (−0.14 / −0.16 / −0.22 for e99) at ~15 pp more capacity. **Follow-up** (3 repetitions of the 50 initial states, second-stage, decided after C1): `protection_suffices` / `unresolved` / `unresolved` + significant `adaptive_advantage` (D +0.03 / −0.05 / −0.10), replication `mixed`: the allocation's effect at this cell is small and seed-dependent | `runs/2026-10-08_high_protection_seed0.md`, `runs/2026-10-08_high_protection_three_seeds.md`, `runs/2026-10-09_high_protection_followup.md` |
| **C2** reverse order × 3 seeds | done: `inconclusive` / `task` / `both` (no three-seed recurrence; Object lost across Spatial every seed; adaptive trade replicates) | `runs/2026-10-0{3,4}_reverse_order_seed*.md` |
| **D** Track 2 (offline predictors) | **not started** | freeze target Wed 28 Oct, **hard deadline Fri 30 Oct** |
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

1. **Baselines (B):** replay and EWC first, then LoRA. Brackets in W3 on seed 0, T1 → T2.
2. **Thesis wording from C1:** raising T2 protection removes the Object forgetting; the new-energy
   allocation adds at most a small, seed-dependent amount at that cell (significant on one seed of
   three). Its consistent edge is final Object retention at about 15 pp more capacity (E3's frontier).
3. **Track 2 (D):** manifest and measures, frozen by Fri 30 Oct at the latest. Angle B at D2
   depends on it.
4. **A3, A5, A6:** evaluation and metric fixes, needed before the 8-task runs (E3).
5. **ConSFT notes** (W5).
6. **Hardware:** F1 engineering on AgileX (adapter, `hw_env`, proof) continues in parallel to the GPU work; reuse the `dobot-hw` pieces (§3).
