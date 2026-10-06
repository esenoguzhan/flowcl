# Dobot X-Trainer try-out, preparation: data conventions, training pipeline, deployment path and overnight runs for the first robot session

**Status: side try-out on branch `dobot-hw`, not in `docs/thesis_plan.md`.** AgileX stays the
Stage B robot; whether any of this merges is decided later. The lab's xtrainer repository is
read only, nothing there was changed. This record covers the preparation (Tue 6 Oct evening);
the robot session (Wed 7 Oct) gets its own record.

**Goal:** catch hardware-pipeline problems early (conventions, image path, timing, safety) by
running flowcl policies on the lab's robot: the plan's F0/F1 checks pulled forward, plus a first
look at seq_ft vs adaptive GPM on real tasks.

## 1. Data (checked against the recordings)

- **Source:** HF `Physical-41/dobot_dataset.puzzle.{red,green,yellow}.retrieve`, LeRobot v2.1,
  pinned in `flowcl/data/dobot_tasks.py`.

  | Task | Instruction (parquet `prompt`) | Episodes | Median length |
  |---|---|---:|---:|
  | red | Use the red block to open the lid of the box and retrieve the object inside | 120 | 26.6 s |
  | green | Combine the green sticks and use it to push the object out of the tube | 124 | 43.0 s |
  | yellow | Use the yellow tool to retrieve the object out of the cage | 142 | 39.9 s |

  Each is a single-object scene with wide placement variation. `puzzle_red_retrieve_large` is a
  cluttered variant (all puzzles on the table) and is not used.
- **Rate and signals:** 30 fps, three cameras (top, left wrist, right wrist; h264 640×480).
  - `actions` (14) are absolute joint targets `[L_j1..6, L_grip, R_j1..6, R_grip]` from the
    leader arms; grippers are in [0, 1], 1 = open.
  - `obs.joint_positions` are measured and trail the action by about 3–5 frames.
  - `joint_velocities` are copies of the positions and are not used.
- **Top camera:** the recorder crops `[150:420, 220:480]` and stretches the crop to 640×480
  (`run_control_terminal.py:390-425`), then JPEG q50; the collector writes h264.
- **Video/parquet alignment:**
  - Every video has exactly 2 more frames than its parquet rows. PTS start at 0, and frame `i`
    pairs with row `i` (LeRobot's timestamp rule); the trailing two are dropped.
  - Image motion trails joint motion by 1–2 frames (cross-correlation, 6 episodes). That is
    camera latency or extra frames at the start; the data cannot tell which, and the effect
    (≤ 67 ms) is small next to a 1.6 s chunk.
- **Hold-out:** the last 10 episodes per task are never trained on. They supply the offline
  checks and the rollouts' initial-condition photos (`datasets/dobot/photos/<task>/`).

## 2. Training setup (`configs/embodiment/dobot_xtrainer.yaml`)

- **Rate and chunking:** **10 Hz** (stride 3), H = 16 (1.6 s), execute_k = 8 (0.8 s).
  - 30 Hz would need about 25 GB of RAM per long task, against 62 GB on the machine.
  - The server upsamples each chunk linearly back to 30 Hz.
- **State:** measured joints, with the gripper dims set to the previous 30 Hz gripper *command*.
  This is what the robot side sends live (the lab's adapter does the same).
- **Action:** `a[3i]`, z-scored.
- **Images:** 128×128 RGB, by `flowcl.deploy.images.to_policy_frame`. The training cache and the
  server use this same function.
- **Std floor 0.2 rad (new `stats_std_floor`).** Red barely moves joint 5 (std 0.04–0.05), so
  under red's frozen §3.3 stats yellow's joint 5 reaches large |z|:

  | Floor | Left joint 5 | Right joint 5 |
  |---|---:|---:|
  | none | 22 | 33 |
  | 0.1 | 11 | 15 |
  | 0.2 | ≈ 5 | ≈ 8 |

  - Green's worst dim under red stats is 8.6 at any floor.
  - **This is a deviation:** the floor was chosen from all three tasks' recordings, not from
    Task 1 alone.
  - With no floor set, stats are bit-identical to before (tested), so LIBERO is unaffected.
- **Demos:** all training episodes (110 / 114 / 132), not the plan's ~15-demo proof recipe.
  This is a deviation and is recorded here.
- **Joint speed in the demos vs the robot runner's per-tick clip:**

  | Task | Ticks over 0.03 rad | Ticks over 0.05 rad |
  |---|---:|---:|
  | red | 2.1% | 0.17% |
  | green | 9.8% | 2.4% |
  | yellow | 4.2% | 0.5% |

  Grippers exceed 0.05/tick on about 2% of ticks and 0.1 on 0.3–0.5%, so the runner uses 0.1
  (the lab terminal uses 0.05).
- **Timing:** 0.10 s/step with three cameras (smoke run), so about 50 min per 30k-step task.

## 3. Deployment path

```
4090 PC (10.147.20.55, ZeroTier)                Robot PC (lab, xtrainer env, no torch)
scripts/serve_policy.py   <--- ws:8000 --->     scripts/hw/dobot_run.py {probe,stoptest,replay,rollouts}
```

- **Server** (`flowcl/deploy/server.py`):
  - one checkpoint per process; fp32, 10 Euler steps;
  - JPEG → `to_policy_frame`, then denormalise and upsample 10 → 30 Hz;
  - the handshake carries the training-action envelope (±0.15 rad), the task start poses (median
    first command) and the instructions;
  - it serves recorded episodes for replay;
  - JSONL request log, plus dumps of the first inputs.
- **Runner** (`scripts/hw/dobot_run.py`):
  - imports `flowcl.deploy.{images,wire,safety}` and the lab's robot/camera helpers, read-only;
  - camera threads keep the latest frame;
  - it sends exactly what the recorder stored (crop/stretch for top, BGR, JPEG q50).
- **Safety:**
  - a 1 s inference deadline;
  - every executed 24-tick prefix is checked before its first tick: finite values, the
    envelope, the lab's J3/J4 limits, first-command jump ≤ 0.15 rad;
  - the Cartesian box at every replan;
  - per-tick limiter, the same semantics as the lab's (tested);
  - the stop flag is checked right before every command.
- **Settings, fixed for the session:** 24 ticks per chunk, joint step 0.03 rad unless replay says
  0.05, gripper step 0.1. Inference is synchronous.
- **Replay tolerances (set before testing):**
  - unshifted per-joint RMSE ≤ 0.02 rad, max ≤ 0.05 rad;
  - command→qpos lag within ±2 ticks of the recording's own;
  - ≤ 1% of ticks joint-clipped.
- **Checked on this machine without the robot:**
  - unit tests (codec, image path, limiter equivalence, upsampling, prefix checks);
  - a loopback of server + runner with a mock robot and a held-out episode as cameras, covering
    probe, replay and rollouts;
  - the safety checks stopped an untrained checkpoint, as they should.

## 4. Single-task red and the offline open-loop check

- **Training:** single-task red (`single__dobot__puzzle_red_retrieve__seed0`, 110 demos, 30k
  steps, AMP) ran 22:21–23:18 (56 min). The loss was 0.01–0.03 at the end.
- **Offline check** (`scripts/hw/offline_check.py`, output in
  `results/dobot_serve/offline_red/offline_check.json`). The server ran the checkpoint; recorded
  frames and states were sent every 24 ticks, the size of one executed chunk. The predicted
  24-tick prefix was compared with the recorded actions, as mean absolute joint error, with
  fixed sampling noise.

  | Episodes | Chunks | Live path (video frame → JPEG q50) | Cache frames | Path difference (median / max) |
  |---|---:|---:|---:|---:|
  | Held out (110–119) | 292 | **0.0289 rad** | 0.0288 rad | 0.013 / 0.073 rad |
  | Training (0–9) | | **0.0049 rad** | 0.0042 rad | 0.011 / 0.047 rad |

  - The live image path costs nothing measurable on held-out episodes.
  - The gap that matters is held-out vs training, about 6×. That is memorisation with about 110
    demos.
  - Held-out episodes range from 0.023 to 0.036 rad, with the worst joint per episode at
    0.042–0.059 rad.
  - Open-loop error is not closed-loop success; the robot decides.
- **Server latency** (1,240 requests, while seq_ft trained on the same GPU): p50 54 ms, p95 59 ms,
  max 67 ms. ZeroTier adds the network round trip.
- **CPU test suite:** passes on the main checkout, including the stats-floor, `TaskRef` and
  gate0 changes and the new adapter and deploy tests. In the worktree, the tests that need the
  LIBERO submodule or the main repository's queue layout fail, as expected there.

## 5. Overnight queue (`scripts/queue_dobot.sh`, from `67f02aa`)

1. Single-task red (the proof policy).
2. seq_ft on red → green → yellow.
3. Adaptive GPM (`gpm_projected_adam_ne90`) on the same curriculum.
4. Single-task green, then yellow.

All runs use `--no-eval` (there is no simulator) and 30k steps per task. Results are in
`results/`, logs in `results/dobot_queue/`.

- **GPM smoke test** (50 steps per stage, separate results root): stages 0 and 1 completed.
  - At stage 1 the memory projection was active on all 91 layers; the largest residual was
    0.025 of its bound.
  - The Gram capture runs on the CPU and scales with dataset size: about 12 min for red's 30k
    samples and 24 min for green's ~50k. The real GPM run gains about 1 h of capture over three
    stages.
- **Expected end:** about 08:00 Wed 7 Oct (seq_ft about 2.6 h, GPM about 3.9 h, two single
  tasks about 1 h each).
- Timings and losses go into the robot-session record.
