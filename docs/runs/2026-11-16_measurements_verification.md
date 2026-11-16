# §5 runner measurements (per-step loss, peak GPU memory, seen-only evaluation, probe loss): GPU verification `pass`. With every new switch off the 4-task protocol is unchanged for all 8 methods; seen-only evaluation and probes every 50 steps leave training unchanged; the new fields hold what they should

**Status: verified on GPU, committed in `f2a5119` (Mon 16 Nov), pushed.** These are the
measurements E3 needs (`status.md` §5 items 1, 2 and 5, and A3). It was a smoke test of the code,
not an experiment: nothing here is a result about the methods.
- **Default protocol unchanged.** A smoke of all 8 methods with the new code equals a smoke made
  with the code from before the change, in every checkpoint tensor, method artifact and rollout.
- **Seen-only evaluation** (`--no-eval-unseen`; seq_ft, plain GPM, replay) trains exactly as the
  default mode does. Every rollout both modes ran is identical too.
- **Probes every 50 steps** (seq_ft, plain GPM, replay, EWC, projected_lora) leave checkpoints and
  rollouts identical, with 5 records per stage from stage 1.
- **The new fields are right on GPU:**
  - every loss file has 200 entries;
  - replay's new-task loss differs from its batch loss from stage 1;
  - every peak-memory window is set;
  - InfLoRA's T2 training window holds its design-capture peak.
- **The CPU suite passes** (1305 tests).
- **The disk filled up during the run** (§6). The 7 runs it stopped were rerun from scratch.

| Check (`scripts/compare_smokes.py`) | Runs | Result |
|---|---|---|
| 1. Default unchanged: `ref` (old code) against `new` (new code, switches off) | all 8 methods | **pass**: 0 of 656 tensors differ at any stage; method artifacts and every rollout equal |
| 2. Seen-only against default, shared cells only | seq_ft, GPM plain, replay | **pass**: checkpoints and method artifacts equal at every stage; the 10 seen cells' rollouts equal |
| 3. Probes on (`--probe-every 50`) against probes off | seq_ft, GPM plain, replay, EWC, projected_lora | **pass**: checkpoints, method artifacts and every rollout equal |

**What a run is:** `seq_hetero`, seed 2, 200 steps per task (batch 64, as in production), AMP,
1 rollout per task and stage. The rollouts mostly fail at 200 steps. The check still pins every
trajectory, because it compares each rollout's seed, success and step count.

**Dates (local, CET):**

| Smoke | Folder | Run |
|---|---|---|
| Reference (old code) | `~/flowcl_smoke/ref` | Sun 15 Nov 21:03:33 → 22:17:54 |
| New, switches off | `~/flowcl_smoke/new` | Sun 15 Nov 22:28:31 → Mon 16 Nov 00:05:09 |
| Seen-only seq_ft | `~/flowcl_smoke/seen` | → 00:09:30 |
| *Disk full* | | about 00:10; the next 7 runs failed (§6) |
| Seen-only GPM and replay, then the 5 probe runs | `~/flowcl_smoke/seen`, `~/flowcl_smoke/probe` | 00:37:03 → 01:42:12 |

**Code.** Each run imports the working tree at its start, so it ran on that tree's code.
- **The reference runs had the old runner code.**
  - seq_ft ran at `6e4a036` and GPM at `5e22984`, both clean. The other six ran at `5e22984` or
    `22bd039`, with uncommitted changes.
  - Those changes were only the A5 and A6 work (`flowcl/analysis/`), `scripts/make_figures.py`,
    `docs/`, and the untracked `flowcl/utils/gpu_memory.py`, which nothing imported yet.
  - The runner files (`trainer.py`, `continual.py`, `projected_lora.py`, `run_continual.py`) were
    first edited at 22:19:38. The last reference run had ended at 22:17:54. (Edit times come from
    the session transcript and the files' modification times.)
  - **The commit message of `f2a5119` says "the smoke made at 6e4a036". The exact statement is
    this paragraph.**
- **The new, seen-only and probe runs had the code committed in `f2a5119`.** The runner files were
  last modified at 22:27:06, before the first new run started, and were committed unchanged. The
  runs' `git_sha` reads `22bd039-dirty` (new seq_ft) or `8be8f33-dirty` (all the others).
- **The queue scripts** are `~/flowcl_smoke/scripts/verify_smokes.sh` and `verify_smokes2.sh` (the
  rerun): `queue_baselines.sh --smoke` for `new`, then `scripts/run_continual.py` with
  `--allow-dirty`, `--no-eval-unseen` or `--probe-every 50`. `ref` is `queue_baselines.sh --smoke
  ~/flowcl_smoke/ref`. Log: `~/flowcl_smoke/verify.log`. Runs other than seq_ft take `--t1-reference-run
  new/seq_hetero__seq_ft__seed2`.

**Cite** (the comparison reports, in `~/flowcl_smoke/compare/`, outside `results/`):
- `ref_vs_new.json` (SHA-256 `199c5246…`);
- `new_vs_seen_{seq_ft,gpm_projected_adam,replay}.json` (`83235873…`, `2047e760…`,
  `a541792e…`);
- `new_vs_probe.json` (`7b5f34c1…`).

---

## 1. What was tested

The code is `f2a5119`. `status.md` §5 describes each switch.
- **Per-step loss (item 1).** `train/stage{K}_losses.json` holds three series:
  - `loss`, the optimised loss;
  - `batch_loss`, the flow loss before `modify_loss`;
  - `task_loss`, the flow loss of the stage's own samples.

  `result.json` records the file as `loss_file` (path, SHA-256).
- **Peak GPU memory (item 2).** Allocated and reserved bytes are recorded for two windows per
  stage, training and evaluation, in `stages[k].peak_gpu` and `systems.peak_gpu`.
- **A3, seen-only evaluation.** `--no-eval-unseen` evaluates only the tasks trained so far. After
  the stage-0 evaluation it encodes the other instructions in fp32, so the text cache matches the
  default mode's.
- **Probe loss (item 5).** `--probe-every N` writes `train/stage{K}_probes.json`: the flow loss of
  every earlier task on a fixed batch, at the start of each stage and every N steps.

## 2. The comparison contract

`flowcl/experiments/smoke_compare.py` (`bc57779`), for each run pair and stage:
- **Compared:**
  - the checkpoint tensors (`stage_identity_check`);
  - the method artifacts' tensors, and `method_state` without its `timings` (`deep_differences`);
  - for each task cell, the rollout seeds, successes and step counts.
- **Ignored:**
  - wall clocks, `timings`, git SHAs and paths;
  - the new loss, peak and probe fields;
  - with `--shared-cells-only`, cells only one run evaluated.

Its unit tests (`tests/test_smoke_compare.py`, 8):
- an equal pair passes;
- a pair with one changed tensor, success or seed fails;
- a changed wall clock still passes.

## 3. Results of the three checks

- **1. Default unchanged.** All 8 runs pass at all 4 stages. Every rollout is equal: 4 tasks × 4
  stages per run.
- **2. Seen-only.** All 3 runs pass. Their checkpoints and method artifacts are equal at every
  stage, so the pre-warmed text cache gives the same training as the default mode.
  - The seen-only matrices hold only the seen cells: 1, 2, 3, then 4 tasks after each stage, so 10
    cells per run. All 10 rollouts equal the default mode's.
  - `seen_only_text_cache` records T1 as `already cached` and T2–T4 as `fp32`.
  - `config.yaml` and `result.json` record `eval_unseen: false`.
- **3. Probes.** All 5 runs pass.
  - Each probe file from stage 1 has records at steps 0, 50, 100, 150 and 200, for every earlier
    task. Each matches its SHA-256 in `result.json`.
  - Stage 0 has no probe file, since no earlier task exists.
  - Probes changed neither peak memory (equal to 0.01 GB) nor training time (§5).
- **Not checked on GPU:**
  - that each module's mode is restored after a probe. A CPU test checks this
    (`test_probe_loss.py::test_probe_losses_restore_every_module_mode`). On GPU it is covered only
    indirectly, by the bitwise-equal checkpoints.
  - **adaptive GPM (`gpm_ne90`).** The plan named it for check 2, but plain GPM was run instead.
    Both go through the same GPM path in the runner.

## 4. The new fields, from `new`

**Loss series.** All 32 loss files (8 methods × 4 stages) have 200 entries per series. The largest
per-step gap between series in any stage:

| Method | `loss` − `batch_loss` | `task_loss` − `batch_loss` | Reading |
|---|---|---|---|
| seq_ft, GPM, LoRA, InfLoRA, projected_lora | 0 | 0 | one task per batch, no loss term |
| EWC | 0.0007 / 0.0008 / 0.0017 (stages 1–3) | 0 | the penalty term |
| ConSFT | 0.46 / 0.48 / 1.06 (stages 1–3) | 0 | ConSFT's loss term |
| replay | 0 | 0.14 / 0.25 / 0.24 (stages 1–3) | the buffer samples in the batch |

**Peak GPU memory**: the PyTorch allocator only, not EGL rendering or the CUDA context. Values are
GB allocated, per stage:

| Method | Training window | Evaluation window | Max reserved |
|---|---|---|---|
| seq_ft, replay, ConSFT | 3.79 at every stage | 0.80 | 4.22–4.23 |
| EWC | 3.79 / 4.47 / 4.48 / 4.48 | 1.23 | 4.98 |
| GPM plain | 3.79 / 4.31 / 4.32 / 4.33 | 0.79–0.80 | 4.68 |
| LoRA | 3.79 / 3.55 / 3.55 / 3.55 | 0.80 | 4.22 |
| InfLoRA | 3.79 / 3.16 / 2.87 / 2.87 | 0.79 / 0.58 / 0.58 / 0.58 | 4.53 |
| projected_lora | 3.79 / 3.54 / 3.56 / 3.57 | 0.79 / 0.58 / 0.58 / 0.58 | 4.53 |

- **InfLoRA's design capture sets its T2 window.** That capture peaks at 3.16 GB, and training
  after it at 2.37 GB. The window reports 3.16, so the reset inside `projected_lora.py` no longer
  cuts the window short.
  - The T3 and T4 windows (2.87) are slightly above their design peaks (2.84). That is GPM's memory
    update in `on_task_end`, which is inside the window.
  - projected_lora's design peaks are 0.64–0.79 GB. Its windows come from training.
  - Both are stored in each checkpoint's `method_state.timings` as `design_peak_gpu_bytes`.
- **Reserved memory only rises.** The caching allocator keeps what it reserved, so a stage's
  reserved peak includes the earlier stages' cache. Allocated is the column for the systems table.
- The seen-only and probe runs give the same peaks, to 0.01 GB.

## 5. Probe curves and run times (descriptive; 200-step smokes)

**T1 (Spatial) probe loss**: at the start of stage 1, then at the end of stages 1, 2 and 3.

| Method | Start of T2 | End of T2 | End of T3 | End of T4 |
|---|---|---|---|---|
| seq_ft | 0.192 | 0.286 | 1.299 | 0.297 |
| EWC | 0.192 | 0.282 | 1.214 | 0.279 |
| GPM plain | 0.192 | 0.194 | 0.204 | 0.193 |
| projected_lora | 0.192 | 0.200 | 0.209 | 0.207 |
| replay | 0.192 | 0.167 | 0.170 | 0.168 |

Each stage's last record equals the next stage's first. After 200 steps per task these are not
the production curves. They show only that the probe measures what it should.

**Run times** (whole run; training in brackets):
- **The `new` queue ran slower than `ref`:** GPM 18.2 min (13.3 training) against 12.8 (8.7). The
  full CPU suite was running on the same machine at the time.
- **The probe runs** use the same code without that load. They match `ref`: GPM 12.7 (8.7),
  projected_lora 13.2 (9.1), EWC 8.0 (4.0), seq_ft and replay 5.0 (1.1). So probes every 50 steps
  cost no measurable time here.
- **Seen-only saves evaluation time:** seq_ft took 4.2 min and replay 3.9, against 5.0 each in
  `ref` (which had no extra load).

## 6. The full disk

- **What happened.** At about 00:10 the disk (`/dev/sda2`, 1.8 TB, which also holds `/tmp`) had 0
  bytes free.
  - Seen-only GPM failed while saving a checkpoint (`RuntimeError` in `torch.save`).
  - It, seen-only replay and all 5 probe runs exited with rc=120, by 00:15:52.
  - The background CPU test run lost its output.
- **The cause** was not the smokes: about 43 GB of smokes on a disk with 1.7 TB used.
  - The largest folders were `~/flowcl/results` (378 GB), `~/flowcl/datasets` (208 GB) and
    `~/dexbench` (102 GB).
  - On the user's instruction, `~/dexbench` was removed. It was pushed, with nothing uncommitted.
  - That left 97 GB free. After the rerun, 67 GB was free.
- **The rerun.** The partial seen-only GPM folder was deleted. All 7 runs were then rerun from the
  start, on the same code, and all ended rc=0.
- **For E3:** check `df -h /` before any queue. One 4-task smoke run takes about 2.5 GB, mostly
  checkpoints.

## 7. CPU suite

- The first run, `uv run pytest -m 'not sim and not gpu'` (1305 tests), hit its 90 min `timeout`
  after 1283 tests, all passed. It was slowed by the smokes running at the same time.
- The two files it had not reached, `test_twosided_gonogo.py` and `test_video.py`, were rerun
  after the disk was freed: 38 passed.
- The new tests:
  - `test_loss_series.py` (5);
  - `test_gpu_memory.py` (2 on CPU, plus 2 marked `gpu`);
  - `test_seen_only_eval.py` (5), including a seen-only resume, and a check that the text cache and
    every checkpoint equal the default mode's;
  - `test_probe_loss.py` (5), including that probes leave checkpoints unchanged and that a resumed
    run gives the same probe records.

## 8. What this does not cover

- **One seed, one curriculum.** The 8-task curriculum (`seq_hetero_t8`) was not smoked with the new
  code. E3's plan should include an 8-task smoke.
- **Resume on GPU.** Copying loss files on resume, a seen-only resume, and rebuilding probe batches
  on resume are tested on CPU only.
- **Peak memory** is the PyTorch allocator's. EGL rendering and the CUDA context are outside it,
  so it understates what `nvidia-smi` shows.
