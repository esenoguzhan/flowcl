# A1 stage-boundary resume: first test `fail` (metadata schema drift only), confirmatory test `pass`

**Status: A1 is signed off.**
- **The confirmatory test passes all three criteria** (adaptive GPM seed 0, stage 3; §4): the
  checkpoint is bitwise identical, the memory content is equal, and every evaluation episode is
  identical.
- **The first registered test** (plain GPM seed 0, stage 3; §1–3) keeps its verdict, **`fail`**.
  Its one failing criterion, memory content, failed only on two metadata keys that code written
  after its source run adds.

**The first test,** criterion by criterion. Every *computed* quantity is reproduced exactly:

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
- **The confirmatory test** (`configs/analysis/resume_check_ne90.yaml`, registered in `53b05cb`
  before its run) used a source trained at `9701620`, whose memory files already carry both keys.
  It passed (§4).

**Dates (local, CEST), Fri 2 Oct 2026:**

| Test | Run | Duration |
|---|---|---|
| First | 14:27:03 → 15:45:36 | 78 min (training 50 min, then the memory update and four-task evaluation; the check took 5 s) |
| Confirmatory | 18:32:53 → 19:36:11 | 63 min |

**Code:**
- The first test ran on `9a723c6`, clean at queue start. Its rule
  `configs/analysis/resume_check.yaml` was committed with the implementation, and the rule's
  SHA-256 was identical before and after the smoke run.
- The confirmatory test ran on `e6be1fc`, clean at queue start. Between `53b05cb` and `e6be1fc`
  only `README.md`, `docs/` and the first test's record changed: no code, config or test.

**Cite:**
- `results/resume_check/report.json` (SHA-256 `e3e84b92…`);
- `results/resume_check_ne90/report.json` (SHA-256 `c8b81012…`).

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

## 4. Confirmatory test: `pass`

**The test.** `configs/analysis/resume_check_ne90.yaml` (`53b05cb`) applies the same procedure and
the same three criteria to adaptive GPM seed 0 at stage 3. That source was trained at `9701620`:
- its memory files carry `target_fraction` and `new_energy_fraction`;
- no later commit changes GPM's memory schema (`_basis_meta` adds nothing for GPM).

**The restored state.** The run resumed from `stage2.pt` (SHA `f3f5202e…`, the T5 sweep's pinned
file), with memory tasks 0–2.

| Criterion | Result |
|---|---|
| 1. `stage3.pt` bitwise | **pass**: 0 of 656 tensors differ |
| 2. `memory_task3.pt` content | **pass**: no difference in any tensor, key or value (bytes differ, as expected) |
| 3. `eval/stage3.json` per episode | **pass**: identical on all four tasks (98 / 84 / 94 / 92%) |

**Also equal:**
- the final and mean-last-50 training losses (0.007504 / 0.013651);
- `gpm_logs_task3.json` (reported only).
  - The comparison helper listed its 600 NaN c-values (steps with a zero update) as differences,
    because NaN ≠ NaN. A NaN-aware comparison finds 0 real differences.
  - The helper now treats two NaNs as equal (a follow-up commit). This changes no criterion: the
    log is reported only, and the memory and checkpoint contain no NaN.

**The first launch** (15:47) was stopped at once by the runner's clean-tree guard. The working tree
then had a README rewrite in progress that is not this study's. That rewrite was committed
(`e6be1fc`), and the test ran from that clean tree.

## 5. Consequences

- **For E3** (resuming seq_ft, plain GPM and adaptive GPM from stage 3): the resume path
  reproduces weights, memory tensors and evaluation episodes exactly.
- **Resumed plain-GPM runs** will write the newer metadata schema (two extra keys). Nothing reads
  those keys as a decision input.
- **A1 is signed off** on the confirmatory test. The first test's `fail` stays on record with its
  diagnosis.
- **Not covered:**
  - resuming a crashed stage (a stage that crashed is rerun);
  - an extended curriculum (E3's seq_hetero → seq_hetero_t8). The namespace inheritance is in the
    code and the CPU tests, but no GPU run has checked it.

## 6. Artifact map

```
results/resume_check/report.json                                  # the first test (SHA-256 e3e84b92...)
results/resume_check_ne90/report.json                             # the confirmatory test (SHA-256 c8b81012...)
results/resume_check_ne90/seq_hetero__gpm_projected_adam_ne90__seed0/   # its resumed run
results/logs/queue_20261002_183253_resume_check_ne90/              # its queue logs
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
