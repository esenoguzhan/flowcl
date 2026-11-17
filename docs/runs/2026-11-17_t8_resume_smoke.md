# 8-task resume smoke (pre-E3 plan, B): `pass`. Resuming 4-task runs into `seq_hetero_t8`, seen-only, gives the same training from old-code and new-code sources, probes change nothing, a run killed during stage 6 resumes to the same result, and the strict sequence reports pass with the checks named by method

**Status: run on GPU Tue 17 Nov 00:54–03:11, queue `QUEUE DONE ok`.** The code is
`b997c6f` (A1), `4b78e16` (A2), `40366b3` (A3) and `d0f7b49` (B). This is a smoke test of E3's
path, not an experiment. At 200 steps and 1 rollout every success is 0, so nothing here says
anything about the methods.
- **E3's path works end to end.** These runs resume 4-task `seq_hetero` runs at stage 4 into
  `seq_hetero_t8`, with seen-only evaluation: seq_ft, plain GPM, replay, and f = 0.7 from its own
  4-task smoke. They keep the 4-task seed namespace `seq_hetero__seq_ft__seed2`, so every cell
  pairs with the seq_ft resume.
- **Old-code sources resume like new ones.** Sources made before the §5 runner change carry no
  loss, peak or stage-record files, like the production runs. Resumed from them, stages 4–7 are
  bitwise equal to the resumes from the new-code smokes.
- **Probes every 50 steps leave training unchanged** at stages 4–7 (GPM, replay).
- **A real interruption recovers.** A GPM resume was killed by its own PID during stage 6. It left
  what a reboot leaves: stage records 0–5, no record 6 and no `result.json`. Resumed at stage 6 from
  its own folder, its stages 6–7 equal the uninterrupted run's.
- **Strict reports pass for all 8 method runs.** Each check is named by method. Each run follows its
  resume chain, the killed-then-resumed one through two links.
- **The f = 0.7 4-task queue works in smoke mode:**
  - the preflight passed;
  - identity at stages 0–1 against plain GPM passed;
  - the Task-1 memory equals plain GPM's on 91 layers;
  - its strict report passed.

| Check (`flowcl/experiments/t8_smoke.py`, `scripts/check_t8_smoke.py`) | Runs | Result |
|---|---|---|
| Old-code sources against new-code sources (`from_ref` vs `from_new`) | seq_ft, GPM, replay | **pass**: 0 of 656 tensors differ at stages 4–7; method state, method artifacts and every rollout equal |
| Probes on against off (`probe` vs `from_new`) | GPM, replay | **pass**, as above |
| Killed and resumed against uninterrupted (`again` vs `from_new`, stages 6–7) | GPM | **pass**, as above |
| Fields of each resumed run (matrix, stage records, loss files, peaks, probes, `eval_unseen`, text cache) | all 10 resumes | **pass** |
| State the killed run left | GPM `cut/` | **pass**: records 0–5, no record 6, no `result.json` |
| Strict sequence reports against the `from_new` seq_ft resume, checks named by method | 8 method runs | **pass** |

## 1. What was run

`bash scripts/queue_t8_smoke.sh --root ~/flowcl_smoke/t8 --new ~/flowcl_smoke/new --ref ~/flowcl_smoke/ref`
ran in tmux session `t8smoke`, started from a clean tree.

Every resume used `--curriculum seq_hetero_t8 --seed 2 --amp --steps-per-task 200 --n-episodes 1
--no-eval-unseen --start-stage 4 --resume-sha256 <source stage3.pt>`, with no `--allow-dirty`.
Each resume was first run as the same command with `--check-resume-only` (`<k>_check_*`, all rc=0).

**The sources:**
- `new` is the 4-task smokes made with the §5 runner code (Sun 15 – Mon 16 Nov).
- `ref` is the ones made with the code from before it. Both are verified equal in
  `runs/2026-11-16_measurements_verification.md`.

| Step | Run (folder under `~/flowcl_smoke/t8/`) | Source | Start → end | Time |
|---|---|---|---|---|
| 0 | f = 0.7 4-task smoke, `queue_ne70.sh --smoke` (`ne70/`) | — (references `new`) | 00:54:26 → 01:08:30 | 14 min |
| 1 | seq_ft (`from_new/`) | `new` | → 01:15:57 | 7 min |
| 2 | plain GPM (`from_new/`) | `new` | → 01:32:39 | 17 min |
| 3 | replay (`from_new/`) | `new` | → 01:39:46 | 7 min |
| 4 | f = 0.7 (`from_new/`) | `ne70/` (step 0) | → 01:56:37 | 17 min |
| 5–7 | seq_ft, GPM, replay (`from_ref/`) | `ref` | → 02:27:49 | 7 + 17 + 7 min |
| 8–9 | GPM, replay with `--probe-every 50` (`probe/`) | `new` | → 02:51:46 | 17 + 7 min |
| 10 | GPM, killed during stage 6 (`cut/`) | `new` | 02:51:51 → 03:00:07 | 8 min |
| 11 | the killed run resumed at stage 6 (`again/`) | `cut/` | → 03:09:30 | 9 min |
| 12 | checks and reports (`checks.json`, `reports/`) | | → 03:11:30 | 2 min |

The queue took 2 h 17 min, against the 3 h planned. It uses 30 GB on disk: `from_new` 12 GB,
`from_ref` 8 GB, `probe` 5.9 GB, `ne70` 2.5 GB, `cut` and `again` 1.3 GB each. 301 GB was free
before.

## 2. The interruption

- **How it was started.** Step 10 started the run with `.venv/bin/python` directly, without `uv`,
  so `$!` was the training process itself (PID 798991).
- **When it was killed.** The queue waited for `train/stage5_record.json`, which appeared at
  02:59:42. It slept 20 s, then sent `kill 798991` (SIGTERM; exit 143 at 03:00:07). Nothing was
  killed by name or parent.
- **Where the run was at the kill.** Its log shows stage 6's header and all 200 training steps, so
  the kill landed in what follows training (memory update and evaluation). No stage-6 checkpoint,
  evaluation, loss file or record had been written. The resume restarts stage 6 from `stage5.pt`
  in a new folder, so where in stage 6 the kill lands does not matter to it.
- **What it left:**
  - `checkpoints/stage4.pt` and `stage5.pt`, and `eval/stage0–5.json`;
  - `train/stage0–5_record.json`: records 0–3 written at import (02:51:56), 4–5 after each stage;
  - no `result.json`.
- **The resume.** `--check-resume-only` on the killed run passed. Its stage records were read in
  place of the missing `result.json` (A1). The resumed run (`again/`) imports stages 0–5 with
  `imported_from` naming the killed run, its immediate source. Its strict report follows two
  links: `again` → `cut` (stage 6), then `cut` → `new` (stage 4).
- **Bitwise equality.** Each link's source `stage{K-1}.pt` has the recorded SHA-256, the imported
  evaluations are byte-equal to the source's, and the namespaces agree. Its stages 6–7 equal
  `from_new`'s uninterrupted GPM run. The stage-6 and stage-7 memories are equal too, and their
  GPM logs show no differences.

## 3. Fields of the resumed runs

Every resumed run's `retention_matrix` has these cells, with the rest null (22 of 64 cells):
- rows 0–3 (imported from 4-task sources): exactly T1–T4;
- rows 4–7 (seen-only): exactly T1..T(k+1).

The matrix of the plain GPM run from `new` (all values 0 at 200 steps):

```
   0    0    0    0   -    -    -    -
   0    0    0    0   -    -    -    -
   0    0    0    0   -    -    -    -
   0    0    0    0   -    -    -    -
   0    0    0    0    0   -    -    -
   0    0    0    0    0    0   -    -
   0    0    0    0    0    0    0   -
   0    0    0    0    0    0    0    0
```

- **Stage records:**
  - `train/stage0–7_record.json` exist for every resumed run and equal `result.json`'s stage
    entries.
  - Imported stages name the immediate source as `imported_from`: `new/…`, `ref/…`, `ne70/…`, or
    `cut/…` for `again`.
- **Loss files:** for all 8 stages in runs from `new` (including probes and `again`). For stages
  4–7 only in runs from `ref`, whose old-code sources wrote none.
- **Peak GPU memory (training window)** is set for every stage trained here.
  - For stages imported from `ref` it is null, as it was in the source.
  - Plain GPM peaks at 3.53 / 4.02 / 4.02 / 4.03 GiB at stages 0–3 (imported) and 3.99 GiB at
    stages 4–7. Replay is 3.53 GiB at every stage.
- **Probes:** `probe/` stages 4–7 have probe files over T1..Tk, with k = 4, 5, 6, 7 tasks, at steps
  0, 50, 100, 150 and 200.
- **Other fields:** `eval_unseen` is `false` in every result, and the text-cache record covers all
  8 tasks.

## 4. Strict sequence reports

`sequence_report.build_report` ran on 8 method runs against `from_new`'s seq_ft resume
(`seq_hetero_t8__seq_ft__seed2`). The runs are `from_new` GPM, f = 0.7 and replay; `from_ref` GPM
and replay; `probe` GPM and replay; and `again` GPM. Every report passed `strict_failures` with no
`--allow-dirty`.
- **Thresholds:** derived from that reference's diagonal with the 0.15 margin. The smoke registers
  them for this one reference; E3's thresholds belong in `main_eval.yaml`.
- **Namespace:** the reference's established namespace is its recorded field,
  `seq_hetero__seq_ft__seed2`, not its folder name. Every method run has the same.

| Check | GPM (5 runs) | f = 0.7 | replay (3 runs) |
|---|---|---|---|
| `resume_source` (per link: source checkpoint SHA, byte-equal evaluations, `imported_from`, namespace) | pass (`again`: 2 links) | pass | pass |
| `seed_namespace` | pass | pass | pass |
| `t1_pairing` | pass, inherited from the 4-task source | pass, inherited from `ne70/` | pass, inherited |
| `identity` | none configured, none present | pass, stages 0–1, inherited from `ne70/` | pass, stage 0, inherited from the source |
| `memory_chained` | pass, 0→1 … 6→7 including 3→4 | pass, including 3→4 | not applicable |
| occupancy, residuals, `frozen_from_stage1` | pass | pass | not applicable |
| `artifact_hashes_match` (imported stages' artifacts hashed in their owner) | pass | pass | pass |
| `not_evaluated` | 22 cells per run, each with its reason (16 from the 4-task rows, 6 seen-only) | same | same |

The reports are in `~/flowcl_smoke/t8/reports/`; `checks.json` holds the full comparison detail.

## 5. Code state during the queue

The queue started at `0ef11cc` with 0 changes. Another session committed four times during it
(`a8cb634`, `16242e0`, `3973b36`, `27ed0bd`, 00:59–01:11; the instruction-negatives pilot). Each run
records the SHA it started at, none dirty:
- `ne70/`: `0ef11cc`;
- `from_new` seq_ft: `16242e0`;
- all the others: `27ed0bd`.

**Those commits don't change what these runs do.** They change two shared files, and in both the
change is opt-in:
- `flowcl/envs/libero_env.py`: `rollout(motion=0)` records nothing and changes nothing else;
- `flowcl/methods/all.py`: two new methods registered.

The rest is new files. No comparison above spans `0ef11cc` and `a8cb634`, so this rests on the
diff. One thing is shown directly: `from_new` seq_ft (`16242e0`) and `from_ref` seq_ft (`27ed0bd`)
are bitwise equal.

**The GPU wasn't shared.** That session's own GPU smoke started at 03:11:35, after `QUEUE DONE`.

## 6. What this does not cover

- **Production scale.** One seed, 200 steps and 1 rollout per cell. E3 runs 50 rollouts at
  production steps. The production sources were checked only with `--check-resume-only` on CPU:
  - the 12 E3 sources (seq_ft, GPM, f = 0.9, replay × seeds 0–2; A1);
  - the 3 flat-0.99 sources (e99 × seeds 0–2);
  - the f = 0.7 sources, after C.

  All pass; seq_ft seed 0 resolves to `seq_hetero__seq_ft__seed0` through the legacy rule.
- **Untested here:**
  - the f = 0.9 and e99 arms, which are the GPM code path with other memory targets;
  - the old-format interrupted source path, checked only on the real LoRA folder with
    `--check-resume-only`.
- **E3's thresholds** (`main_eval.yaml`, `pinned_reference` for T5–T8) are not written. The mode is
  tested on CPU only.
- **A seq_ft report.** None was made, since seq_ft is the reference. The reference loader
  established its namespace.

**Next:** the f = 0.7 queue (C), `bash scripts/queue_ne70.sh`: 3 seeds, about 14.3 h, from a clean
tree. Nothing in the tree may change until it ends.
