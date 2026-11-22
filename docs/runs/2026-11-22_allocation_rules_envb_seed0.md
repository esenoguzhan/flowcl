# Allocation rules at matched capacity, seed 0, env B: every training-side gate passes on all three arms; the rep-0 rerun of T4 does not reproduce, so all three are `invalid_evaluation` until the gate is decided

**Status: run on ThinkStation-2 (env B) Sun 22 Nov 03:07 → 18:16, queue `QUEUE DONE FAILED:
22_summary`, clean tree at `b7013ba`, rule `configs/analysis/allocation_rules.yaml` (SHA-256
`40cc4f17eaa0…`, env-B amendment of 22 Nov).** Every step before the summary ended rc=0. The
summary exits 1 because no arm has a valid label.
- **What the pilot asks:** at e99's own protected capacity after each extension, does *where*
  GPM protects (three allocation rules) change ACC against flat 0.99? Plan:
  `~/.claude/plans/jazzy-meandering-star.md`; the rule holds the question, gates and labels.
- **Training side, all three arms: every gate passes.**
  - Stages 0–1 equal plain GPM's bit for bit (656/656 tensors at both).
  - The T1 memory equals plain GPM's, and T1 pairs with seq_ft at relative difference 0.0.
  - Each extension reaches e99's count within 0.001 pp (gate 0.05 pp).
  - Every post-T2 rank equals the registered prediction, except one ±1 on the room arm's
    predicted jump layer, which is the declared `prediction_borderline` flag.
  - All three are distinct from e99: 1.19 / 3.49 / 14.86 pp of capacity moved after T2.
  - The three strict reports pass.
- **Evaluation side: one gate fails, the same way for every run.**
  - Provenance, pairing (48 cells) and fresh noise pass.
  - e99's rep-0 rerun equals its stored T1, T2 and T3 cells (150/150 rollouts, steps included).
  - **T4 does not reproduce, for e99 and every arm:** 11–17 of 50 rollouts end at another step, and
    1–4 change outcome (§3).
  - **Two identical fresh processes disagree with each other on 16/50 T4 rollouts as well** (§4).
    The cause is found: the wrist camera's multisampled rendering of the stove burner varies by
    1–2 levels on a few pixels between renders of the same state on this GPU. Physics and the
    policy are deterministic. So the registered check cannot pass for any run.
  - The rule requires exact equality, so all three arms are labelled `invalid_evaluation` and
    none is promoted.
- **The T4 check was amended before any outcome was read** (§5, amendment 2): seeds equal, plus
  an exact McNemar test on the paired outcomes. The summary is rerun under it (§7).

## 1. What was run

`bash scripts/queue_alloc.sh` in tmux session `alloc`, log
`results/logs/queue_20261122_030746_alloc_envb/`. Every path is under `results/envB/`.

| Step | What | Start → end |
|---|---|---|
| 0 | preflight: 10 checks (clean tree, rule committed, idle GPU, disk, targets absent, configs, references by SHA-256, torch 2.9.1+cu128 everywhere, archived e99, prediction) | 03:07:47 → 03:07:50 |
| 1–3 | e99: rep 0 rerun (4 tasks), reps 1–2 | → 03:42:35 |
| 4–9 | novelty: run (4 h 25 min), check, strict report, rep-0 T4 rerun, reps 1–2 | → 08:33:49 |
| 10–15 | size: same (run 4 h 24 min) | → 13:23:14 |
| 16–21 | room: same (run 4 h 26 min) | → 18:16:32 |
| 22 | summary | → 18:16:47, rc 1 |

- **Each run:** `run_continual.py --curriculum seq_hetero --amp --results-root results/envB
  --method gpm_alloc_<rule> --seed 0`, with `--t1-reference-run results/envB/seq_hetero__seq_ft__seed0`
  and `--identity-reference-run results/envB/seq_hetero__gpm_projected_adam__seed0
  --identity-stages 0 1`.
- **Each evaluation:** `evaluate.py` on stage 3 with `--runner-text-cache seq_hetero --amp`. That
  rebuilds the runner's instruction cache: T1 encoded under AMP, the rest fp32 (`b7013ba`).
  - Rep 0 uses the run's own id `seq_hetero__seq_ft__seed0`, so it reruns the stored cell's
    rollout seeds.
  - Reps 1–2 use `…__rep1` / `…__rep2`: the same 50 initial states with fresh noise.
- **The comparator** is env-B e99, `seq_hetero__gpm_projected_adam_e99__seed0`, retrained here on
  21 Nov (`7fe311d`).
- **Where:** the runs `seq_hetero__gpm_projected_adam_alloc_{novelty,size,room}__seed0`; the
  reports `gpm_seq_alloc_{novelty,size,room}_seed0/report.json`; the evaluations and the summary
  in `allocation_rules/seed0/`.

## 2. Training-side gates

| | novelty | size | room |
|---|---|---|---|
| Identity vs plain GPM, stage 0 / 1 | 0 of 656 / 0 of 656 | same | same |
| T1 memory vs plain GPM (91 layers) | equal | equal | equal |
| T1 pairing vs seq_ft | 0.0 | 0.0 | 0.0 |
| Knob after T2 / T3 / T4 | f 0.9011 / 0.6273 / 0.4912 | δ 1.10e-4 / 1.70e-4 / 2.09e-4 | κ 1.0254 / 0.9843 / 0.6738 |
| Matching error after T2 / T3 (gate) | −0.0006 / −0.0005 pp | +0.0001 / 0.0000 pp | −0.0008 / −0.0010 pp |
| Matching error after T4 (E3 requirement only) | +0.0006 pp | −0.0003 pp | −0.0002 pp |
| Post-T2 ranks vs registered prediction | equal on 91/91 | equal on 91/91 | 90/91; −1 on the jump layer `flow_head.blocks.2.self_attn.q_proj` (`prediction_borderline`) |
| Capacity moved vs e99, after T2 / T3 (gate ≥ 1 pp) | 1.19 / 1.59 pp | 3.49 / 3.60 pp | 14.86 / 15.57 pp |
| Subspace overlap with e99 after T3 | 0.993 | 0.994 | 0.994 |
| Strict report (env-B thresholds) | pass | pass | pass |

**Protected, % of the layer group after T4 (arm against e99):**

| | all | trunk | decoder |
|---|---|---|---|
| e99 | 64.45 | 89.22 | 27.31 |
| novelty | 64.45 | 88.23 | 28.80 |
| size | 64.45 | 88.63 | 28.19 |
| room | 64.45 | 67.25 | 60.26 |

- **Room is the only arm that moves capacity in bulk.** After T2 it protects 57.3 % of the trunk
  and 53.7 % of the decoder, against e99's 78.9 / 21.4 %. Novelty and size stay within 1.5 pp of
  e99's split at every stage.
- **The room arm's post-T2 extension is the lower end of its bracket,** 23 448 650 (−352 against
  the target). The prediction took the upper end, 23 449 162 (+160).
  - In the run, the bracket has two jump layers (that `q_proj` and the same block's `v_proj`), not
    one. The in-run spectrum differs from the offline reconstruction in the last digits.
  - The rule declares this case: a ±1 on a predicted jump layer is a flag, and the matching gate
    still applies (−0.0008 pp).
- **After T2, the smaller memory of each layer lies inside the larger one** (overlap with e99
  1.0): every arm extends the same T1 memory from the same Grams, so the rules differ only in how
  many leading directions each layer takes. From T3 on, the trajectories differ, so equal ranks no
  longer mean the same subspace (overlap 0.993–0.994).

## 3. Evaluation gates

| Gate | Result |
|---|---|
| Provenance (checkpoint and eval SHA-256, task list, text cache: T1 AMP, rest fp32), 12 eval files | pass |
| Pairing: state IDs and rollout seeds equal across arms, every task × repetition | pass, 48 cells |
| Fresh noise: reps 1–2 seeds differ from rep 0 | pass |
| e99 rep-0 rerun, T1 / T2 / T3 | equal, 50/50 each, steps included |
| **Rep-0 rerun of T4** (libero_10 stove + moka pot) | **fails on all four runs** |

**T4 rerun against the run's own stage-3 cell (same checkpoint, same seeds):**

| Run | Episodes ending at another step | Outcome changed | Net successes, rerun − stored |
|---|---|---|---|
| e99 | 14 / 50 (first: episode 6) | 1 (ep 45) | −1 |
| novelty | 17 / 50 (first: episode 3) | 4 (eps 19, 30, 31, 33) | +4 |
| size | 17 / 50 (first: episode 5) | 1 (ep 30) | +1 |
| room | 11 / 50 (first: episode 8) | 2 (eps 30, 36) | +2 |

- **Rollout seeds are equal in every comparison,** so this is not a pairing or seed error.
- **The in-run stage-0 and stage-1 cells agree across the 5 training processes** (plain GPM, e99
  and the three arms), rollout for rollout, on all four tasks. That says little about T4: no
  policy solves T4 before training on it, so all 50 episodes run to the step limit there. The
  informative cells are T1 at stages 0–1 and T2 at stage 1.
- **On gamma (env A), T4 reproduced.** The 20 Nov rerun of e99 seed 0 equals its stored T4 cell
  50/50; there only T1 differed, through the instruction cache that `b7013ba` fixed.

## 4. Diagnosis: the wrist camera's multisampled rendering of the stove burner varies

Run Sun 22 Nov 18:19–19:25 on the otherwise idle GPU, at the user's request ("Diagnose T4
first"). Scripts and outputs: `results/envB/allocation_rules/seed0/t4_diagnosis/`.

**Step 1: it is not in-run against standalone.** Two more fresh processes ran e99's T4 rerun with
the queue's command, except `--tasks` T4 only: same checkpoint, run id, seeds and text cache.

| Pair (e99 stage 3, T4, same 50 seeds) | Episodes ending at another step | Outcome changed |
|---|---|---|
| stored in-run cell vs queue rerun (4 tasks) | 14 | 1 |
| stored vs fresh a | 15 | 3 |
| stored vs fresh b | 18 | 4 |
| queue rerun vs fresh a | 18 | 2 |
| queue rerun vs fresh b | 18 | 3 |
| **fresh a vs fresh b** (identical commands) | **16** | **3** |

- Two identical fresh processes disagree as much as either does with the run.
- e99's T4 successes in the four processes are 48, 47, 45 and 46 of 50.

**Step 2: the first divergence is always the wrist-camera image.** `diag.py` reruns T4 exactly as
`evaluate.py` does and hashes:
- at every replan: each camera image, the normalized robot state and the policy's action chunk;
- after every env step: the executed action and the full MuJoCo state.

Two processes each ran episodes 0–24, then repeated episodes 6, 10, 13 and 17 in the same process
(`compare_run1_run2.txt`).
- **Across the processes,** 10 of the 25 episodes diverge. Within a process, the repeats of 13
  and 17 diverge from their own first run too.
- **In every diverging episode,** everything before the divergence is identical: every MuJoCo
  state, every action, every agent-view image and the robot state.
- **The first differing input is always the eye-in-hand image,** at replan 7–29 (env step
  56–232), and the chunk changes with it.
- **The weights and the instruction cache** hash equal in both processes.
- **So the physics, the policy and its GPU inference are deterministic.** The wrist camera's
  image is not.

**Step 3: the rendering itself varies for a fixed state.** `diag_render.py` re-renders both cameras
4–6 times from the unchanged MuJoCo state after each step.
- **Default rendering** (MuJoCo's 4× multisampling), episodes 13 and 16 to step 150:
  - the wrist camera's renders differ at 28 of 302 states, by 1–2 intensity levels on 1–6 pixels;
  - the pixels sit on the rim of the stove burner's rings (`render1/examples.png`);
  - the agent-view camera never varies.
- **Multisampling off** (`vis.quality.offsamples` 0): no render varies at any of 2,665 states.
  That covers episodes 13 and 16 to step 150, and episodes 1, 3, 14, 17, 18 and 19 to their end.

**Reading.**
- **The cause:** the 4× multisample resolve is not deterministic at those edges on this GPU and
  driver (RTX PRO 4000 Blackwell, driver 595.99.02, EGL). A one-level change in a few wrist
  pixels at a replan changes the action chunk. On this contact-rich task the episode then takes
  another course.
- **T1–T3:** their scenes show no such effect (their reruns are exact).
- **Gamma:** its RTX 4090 rendered T4 deterministically.

**What it means.**
- **It is evaluation noise only.** It is the same mechanism for every arm and every run. Training
  (dataset images), the checkpoints and the cross-arm pairing (states and seeds) are unaffected.
- **No PyTorch setting can fix it:** the policy is already deterministic.
- **Turning multisampling off makes the rendering deterministic, but changes the images.** They
  lose their anti-aliasing, and with them the rollouts (episode 13: 424 steps without, 279–291
  with).
  - That is a change to the evaluation protocol. No stored env-B cell would be comparable any
    more (references, e99, these arms), and everything would need re-evaluation.
  - It is not adopted here.
- **Any env-B rule with a bitwise rerun check on a task whose wrist camera sees such geometry**
  will fail it, as this one did. That includes E3 if its rule keeps one; T5–T8 are untested.

## 5. The T4 reproduction gate: amended (amendment 2)

The registered labels were all three arms `invalid_evaluation`, none promoted. Two options were
put to the user:
- **A:** amend the T4 check to a tolerance;
- **B:** re-evaluate everything under deterministic rendering, giving up comparability with every
  stored env-B cell.

**The user chose A** ("Amend the T4 gate", Sun 22 Nov evening).

**Amendment 2** (`configs/analysis/allocation_rules.yaml`, header and `evaluation.rep0_tolerance`)
was committed before the summary was rerun.
- **The new check, for T4:** the rep-0 rerun must keep the stored cell's rollout seeds, and a
  two-sided exact McNemar test on the 50 paired outcomes must not reject at 0.05 (p ≥ 0.05).
- **Kept descriptively:** the bitwise comparison is still computed and kept in the summary.
- **Unchanged:** T1–T3 stay bitwise; so does everything else in the rule.
- **Code and tests:** `tolerant_reproduction_check` and `mcnemar_exact_p` in
  `flowcl/experiments/allocation_rules.py`, with 2 new tests (39 allocation tests pass).
- **What was known when it was written:** the gate data (§3, §4). The discordant outcomes,
  1 / 4 / 1 / 2, all pass this test (novelty's 4, all in one direction: p = 0.125).
- **What was not known:** the outcome (`rep0_paired`, `retention`, ΔACC) had not been read.

## 6. Files

- Rule: `configs/analysis/allocation_rules.yaml` (env-B amendment, `a6bef27`; text cache,
  `b7013ba`).
- Runs: `results/envB/seq_hetero__gpm_projected_adam_alloc_{novelty,size,room}__seed0/`.
- Strict reports: `results/envB/gpm_seq_alloc_{novelty,size,room}_seed0/report.json`.
- Evaluations and summary: `results/envB/allocation_rules/seed0/`.
- Queue log: `results/logs/queue_20261122_030746_alloc_envb/`.
- T4 diagnosis (§4): `results/envB/allocation_rules/seed0/t4_diagnosis/` (scripts `diag.py`,
  `diag_render.py`, `compare.py`, `t4cmp.py`; the two fresh reruns; per-step hash logs; render
  reports with and without multisampling; `render1/examples.png`).
- The gamma launch, cut off by the move: `results/interrupted/seq_hetero__gpm_projected_adam_alloc_novelty__seed0__machine_move_20261120`.
