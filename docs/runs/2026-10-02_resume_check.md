# A1 stage-boundary resume test (plain GPM seed 0, stage 3): `fail` by the registered rule, on metadata schema drift only

**Status: the registered verdict is `fail`.** One of the three criteria (memory content) fails.
Every *computed* quantity is reproduced exactly:

| Criterion (`configs/analysis/resume_check.yaml`) | Result |
|---|---|
| 1. `stage3.pt` state dict bitwise | **pass**: 0 of 656 tensors differ |
| 2. `memory_task3.pt` content (every tensor and metadata value) | **fail**: 183 differences, **all of them extra metadata keys**; 0 tensor differences; every shared value equal |
| 3. `eval/stage3.json`, every per-episode success and step count | **pass**: all four tasks identical (Spatial 88%, Object 0%, Goal 56%, LIBERO-10 90%) |

- **The cause is diagnosed.** The source run was trained at `0184293` (23 Sep). Commit `9701620`
  (25 Sep, the adaptive GPM variant) later added two metadata fields:
  - `target_fraction` in each layer's memory-history entry (91 layers, mirrored in each basis's
    history);
  - `new_energy_fraction` in the memory's `config`.

  The resumed run, on today's code, writes both, and the source's file lacks them.
  - Every memory tensor is equal, and so are every value both files share and the histories of
    tasks 0–2.
  - The final and mean-last-50 training losses are equal to the source's (0.003032 / 0.008670).
- **The confirmatory test is registered but not yet run** (`configs/analysis/resume_check_ne90.yaml`,
  `53b05cb`). Its source, adaptive GPM seed 0, was trained at `9701620`, so its memory files
  already carry both keys (§4).

**Date (local, CEST):** Fri 2 Oct 2026, 14:27:03 → 15:45:36. The resume took 78 min: training
50 min, then the memory update and four-task evaluation. The check took 5 s.

**Code:** `9a723c6`, clean at queue start. The rule `configs/analysis/resume_check.yaml` was
committed with the implementation, and its SHA-256 was identical before and after the smoke run.

**Cite:** `results/resume_check/report.json` (SHA-256 `e3e84b92…`).

---

## 1. What was tested

**The resume path** (`9a723c6`): `scripts/run_continual.py --resume-run DIR --start-stage K`.
- **The source checks.** It verifies `checkpoints/stage{K-1}.pt` (against `--resume-sha256`)
  and its stage, then the curriculum prefix, method and kwargs, recipe (except device, workers
  and log interval), policy config, seed, stats, every method artifact's SHA, and the earlier
  evaluations.
- **The restore.** GPM's memory comes from the artifact the checkpoint names
  (`GPM.restore_state`, SHA-verified); methods that have artifacts but no `restore_state` refuse.
- **The rest of the state.**
  - It recreates the runner's text cache (T1 under AMP, the rest in fp32).
  - It imports the earlier stages' evaluations and keeps the source's seed namespace.
  - It never writes into the source.

**This test:** resume plain GPM seed 0 (`seq_hetero__gpm_projected_adam__seed0`) at stage 3 from
its own `stage2.pt` (SHA `a0d70622…`), into `results/resume_check/`.
- The runner's own identity check at stage 3 passed before evaluation.
- The restored state was memory tasks 0–2 (`memory_task2.pt`, SHA `b79264ea…`) with
  `memory_extended` {0, 1, 2}.

## 2. Results

- **The checkpoint:** 0 of 656 tensors differ, with none missing and none extra.
- **The evaluation:** identical per episode, successes and step counts, on all four tasks.
- **The memory artifact `memory_task3.pt`.** Bytes differ, and content differs only in keys:
  - 92 differences in `meta`: each layer's `memory_history["3"][layer]` has the extra key
    `target_fraction`, and `config` has the extra key `new_energy_fraction: None`;
  - 91 differences in `bases`: each basis's history mirrors the `target_fraction` key;
  - tensor differences (vectors, singular values): **0**;
  - value differences on shared keys: **0**;
  - tasks 0–2's history: identical.
- **`gpm_logs_task3.json`** (reported only) shows the same extra key in its `memory` block.

## 3. The pre-registered judgement

Criterion 2 requires equality of "every tensor and metadata value". Extra keys violate it.
**Verdict: `fail`.** The rule was written to catch a resume that loses or alters state. The
failure it caught is a schema difference between the code that wrote the source and today's code.
No restored or recomputed value differs.

## 4. Confirmatory test (registered, pending)

**`configs/analysis/resume_check_ne90.yaml`** (`53b05cb`) applies the same procedure and the same
three criteria to adaptive GPM seed 0 at stage 3. That source was trained at `9701620`:
- its memory files carry `target_fraction` and `new_energy_fraction`;
- no later commit changes GPM's memory schema (`_basis_meta` adds nothing for GPM).

**Launch:** `bash scripts/queue_resume_check.sh --config configs/analysis/resume_check_ne90.yaml`.
- It was started at 15:47 and stopped at once. The runner's clean-tree guard refused because the
  working tree had changes that are not this study's: `README.md`, `docs/implementation_notes.md`
  and `docs/assets/`, all written at 15:46:48.
- It will run from a clean tree.

## 5. Consequences

- **For E3** (resuming seq_ft, plain GPM and adaptive GPM from stage 3): the resume path
  reproduces weights, memory tensors and evaluation episodes exactly.
- **Resumed plain-GPM runs** will write the newer metadata schema (two extra keys). Nothing reads
  those keys as a decision input.
- **The A1 sign-off waits for the confirmatory test.**
- **Not covered:**
  - resuming a crashed stage (a stage that crashed is rerun);
  - an extended curriculum (E3's seq_hetero → seq_hetero_t8). The namespace inheritance is in the
    code and the CPU tests, but no GPU run has checked it.

## 6. Artifact map

```
results/resume_check/report.json                                  # the comparison (SHA-256 e3e84b92...)
results/resume_check/seq_hetero__gpm_projected_adam__seed0/        # the resumed run: config (with the
                                                                  # resume block), resume.json, stats,
                                                                  # checkpoints/stage3.pt, method/*_task3.*,
                                                                  # eval/stage0-3.json, identity_stage3.json,
                                                                  # result.json (stages 0-2 imported)
results/logs/queue_20261002_142703_resume_check/                   # queue.log, 0_resume.log, 1_check.log
configs/analysis/resume_check.yaml, configs/analysis/resume_check_ne90.yaml,
flowcl/train/continual.py (load_resume_source, warm_runner_text_cache), flowcl/methods/{base,gpm}.py
(restore_state), flowcl/experiments/resume_check.py, scripts/{run_continual,resume_check}.py,
scripts/queue_resume_check.sh, tests/test_resume.py
```
