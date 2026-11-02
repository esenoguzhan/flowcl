# LIBERO calibration of the offline open-loop check (stove + moka pot, 20, 30 and 40 demos). Policies that succeed closed-loop (86–94 %) are at about half of standing still on held-out demos (ratio 0.48). The 30-demo Piper policy is at 1.15–1.22, so by this check its shortfall is real

**Status: a calibration for Piper F1** (`runs/2026-11-01_piper_f1_prep.md` §5, §8–§10).
- **Code:** `b89fd2e`. All three runs record that sha and were trained from a clean tree.
- **Question:** does a sim policy that succeeds closed-loop beat standing still on held-out demos,
  in the same offline check as Piper?
- **Answer:** yes, clearly. The ratio is 0.48–0.49 at every demo count, and below 0.62 on every
  held-out demo.
- **Not settled:**
  - how much of Piper's gap comes from absolute targets rather than from the policy's sense of
    where to move (§5, reading 4);
  - whether a sim policy that fails sits near 1 (§5, reading 6).

## 1. Setup

**Task:** `libero_10/KITCHEN_SCENE3_turn_on_the_stove_and_put_the_moka_pot_on_it`.
- Two stages: the closest LIBERO analogue to sausage → pot → lid.
- Gate 0 trained it on all 50 demos and reached 100.0 [100.0, 100.0]
  (`runs/2026-09-21_gate0.md`). With all 50 trained on, that run has no held-out demo.

**Runs:** `scripts/queue_libero_calibration.sh`, Sun 1 Nov 23:14 to Mon 2 Nov 01:28.
- **Demos:** 20, 30 and 40, i.e. demos 0..n−1.
- **Recipe:** Gate 0's, unchanged: 30k steps, batch 64, lr 1e-4, AMP, seed 0. Then 50 closed-loop
  rollouts on the fixed init states, with Wilson CIs.
- **Time per run:** about 39–41 min of training, 5 min of rollouts and 7 s for the check.
- **Output:** `results/libero_calib/n<N>/` and `results/libero_calib/curve.txt`.

**Offline check:** `scripts/libero_offline_check.py`. The shared core is
`flowcl/analysis/open_loop.py`, which the Piper check now uses too; its output there is
byte-identical.
- **Ticks and noise:** every 8 ticks, the policy samples a 16-tick chunk from the recorded frames
  and state, with noise keyed by the recording.
- **Demos:**
  - held out: 40–49, unseen by every run (325 chunks);
  - training: 0–9, trained on by every run (359 chunks).
  - The script refuses a run whose `n_demos` covers a held-out demo.
- **Metric:** mean absolute error on the 6 OSC pose dims, over the 8 executed ticks. Units are
  LIBERO's action units: deltas in [−1, 1], used as recorded. The gripper is reported apart.
  - **Integrated,** `|Σ(δ̂ − δ)|` from the chunk start: the position error the deltas add up to.
    This is the form of Piper's absolute-target error.
  - **Baselines:**
    - **standing still,** `|Σδ|`, the arm's own displacement since the chunk start. This is
      Piper's "hold the last command": there, the error at tick t is `|a_t − a_{t0−1}|`, the
      sum of the command's increments.
    - **repeating the last delta,** constant velocity. It uses the last recorded action, which
      the policy never sees.
  - **Per tick,** `|δ̂ − δ|`, against zero motion and repeating the last delta. These are reported
    for completeness.

## 2. Results

| Demos | Success (50 rollouts) | Held out: policy | Standing still | Ratio | Repeat last delta | Training: policy | Ratio |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 20 | **94.0** [86.0, 100.0] | 0.3067 | 0.6222 | **0.49** | 0.2047 | 0.0121 | 0.02 |
| 30 | **86.0** [76.0, 94.0] | 0.2973 | 0.6222 | **0.48** | 0.2047 | 0.0147 | 0.02 |
| 40 | **94.0** [86.0, 100.0] | 0.2957 | 0.6222 | **0.48** | 0.2047 | 0.0175 | 0.03 |
| 50 (Gate 0) | 100.0 [100.0, 100.0] | — | — | — | — | — | — |

Integrated pose error, executed 8 ticks, in action units.
- **Training baselines:** standing still 0.6054; repeating the last delta 0.2032.
- **Per held-out demo,** policy over standing still:
  - 20 demos: 0.40–0.61;
  - 30 demos: 0.37–0.61;
  - 40 demos: 0.35–0.59.
- **Per tick** (held out): policy 0.071 / 0.070 / 0.069; zero motion 0.139; repeating the last
  delta 0.061.
- **Gripper** (held out): 0.136 / 0.130 / 0.123 in [−1, 1] units, against 0.001 on training.
  Most of this is the timing of the open/close switch.

**By tick within the chunk** (30 demos, held out, integrated):

| Tick | 0 | 1 | 3 | 7 | 11 | 15 |
|---|---:|---:|---:|---:|---:|---:|
| Policy | 0.071 | 0.140 | 0.272 | 0.512 | 0.712 | 0.873 |
| Standing still | 0.138 | 0.276 | 0.553 | 1.114 | 1.661 | 2.192 |
| Repeat last delta | 0.015 | 0.044 | 0.142 | 0.479 | 0.960 | 1.553 |
| **Policy / standing still** | **0.51** | **0.51** | **0.49** | **0.46** | **0.43** | **0.40** |

## 3. Against Piper

Only ratios compare: LIBERO's errors are in action units, Piper's in rad.

| | LIBERO moka pot, 30 demos | Piper sausage → pot → lid, 30 demos (A / B augmented) |
|---|---:|---:|
| Closed-loop success | 86.0 [76.0, 94.0] | not run yet |
| Samples | about 8,100 at 20 Hz | 13,376 at 15 Hz |
| **Held out: policy / standing still** | **0.48** | **1.22 / 1.15** (against holding the last command) |
| Held out: policy / holding the state | — (standing still starts from the state) | 1.06 / 0.99 |
| Training: policy / standing still | 0.02 | 0.12 / 0.30 |
| Held out over training | 20× | 15× / 5.6× |

**By tick,** policy over standing still (Piper A against holding the last command, record §5):

| Tick | 0 | 1 | 3 | 7 | 11 | 15 |
|---|---:|---:|---:|---:|---:|---:|
| LIBERO, 30 demos | 0.51 | 0.51 | 0.49 | 0.46 | 0.43 | 0.40 |
| Piper A, 30 demos | 2.8 | 1.85 | 1.33 | 1.03 | 0.89 | 0.83 |

## 4. A sanity point from the smoke run

The smoke run trained the same task for 200 steps on 20 demos and went 0/2 closed-loop. Its ratio
was 0.88 on held-out demos and 0.89 on training demos, so the ratio moves with training. It is one
untrained policy and only 2 rollouts, not a calibration point.

## 5. Readings

1. **In sim, 20 demos are enough for this task.**
   - Success is 94 % at 20 demos, 86 % at 30 and 94 % at 40; Gate 0's 50-demo run reached 100 %.
   - The dip at 30 is inside the overlapping CIs of a single seed.
   - From 20 to 40 demos, neither success nor the offline score moves much (0.307 → 0.296).
2. **A policy that succeeds closed-loop clearly beats standing still offline.**
   - Its error is about half the arm's own displacement over the executed chunk: 0.48–0.49 at
     every demo count, and 0.35–0.61 on every held-out demo.
   - The ratio is nearly flat across the chunk (0.51 at tick 0, 0.46 at tick 7).
3. **Piper's policy is outside that.**
   - It is at 1.15–1.22 against holding the last command and 0.99–1.06 against holding the state.
   - By the case the plan fixed before the runs, its shortfall is real, not an artefact of the
     check. More demos or relative actions should come before the robot.
   - This is offline evidence only. A memorising policy can still succeed on a scene close to its
     demos.
4. **Piper's gap sits at the start of the chunk.**
   - Its ratio is 2.8 at tick 0 and 1.03 at tick 7, and below 1 from tick 11.
   - LIBERO's integrated error starts every chunk at zero by construction, because its actions
     are deltas from the current pose. Piper's absolute targets must place tick 0 from a follower
     state that lags the leader by up to 0.06–0.13 rad (p99).
   - So part of Piper's shortfall may be the absolute-target offset rather than the motion.
   - **This can be split offline, without training:** rerun the Piper check with each chunk's
     tick 0 subtracted, from both the predicted and the recorded chunk, and compare predicted
     motion with recorded motion.
     - If that ratio is near LIBERO's, relative actions are the lever.
     - If it stays near 1, the policy does not know where to move, and more demos are the lever.
5. **Memorisation is not what separates them.**
   - The LIBERO policy's held-out error is 20× its training error (Piper A: 15×), and it still
     succeeds.
   - What differs is the held-out level against the arm's motion.
6. **The check is calibrated at one point only.**
   - Success and offline score are both flat over 20–40 demos, and no calibration run fails.
   - So this shows where a successful sim policy sits (about 0.48), not where a failing one does.
     The smoke point (§4) suggests an untrained one sits near 0.9.
7. **Losing to a baseline that sees the last action is normal.**
   - The 86–94 % policies lose to repeating the last delta by 1.44–1.50×, and on every held-out
     demo (1.03–1.85).
   - A check against that baseline would fail every successful sim policy here.

## 6. Caveats

- **Confounds:** sim against real and delta against absolute actions both differ between the two
  sides. A gap that favours LIBERO may come from either. Reading 4 is the step that would separate
  them.
- **Which baseline is the analogue is a choice.**
  - The plan matched baselines physically: in both cases the baseline is "the arm stops", and the
    ratio is position error over the arm's own displacement in the executed chunk.
  - Matched by information instead (each baseline uses one past command the policy never sees),
    Piper's 1.15–1.22 against holding the last command would compare with LIBERO's 1.44–1.50
    against repeating the last delta. By that matching, Piper looks no worse.
  - The tick-0 split in reading 4 decides between the two readings.
- **Held-out demos differ in kind.**
  - LIBERO's held-out demos come from the same init-state distribution and operator as its
    training demos.
  - Piper's held-out episodes are from the faster `v3` session, while two thirds of its training
    demos are from the slower one.
- **Scope:** single seeds, one task, one recipe.

## 7. Files

- `flowcl/analysis/open_loop.py`: the shared core. `predict_chunks`, `summarize`,
  `delta_metrics`, `assert_unseen`.
- `scripts/libero_offline_check.py`: the CLI.
- `scripts/queue_libero_calibration.sh`: the queue.
- `tests/test_open_loop.py`: hand-computed toy cases.
- `results/libero_calib/n{20,30,40}/`:
  - the runs, each with `eval.json`, `offline_check.json` and `checkpoints/final.pt`;
  - their `gate0/gate0.json` (all pass at 0.8).
- `results/libero_calib/curve.txt`; logs in `results/libero_calib/queue/`.
