# Deploying flowcl policies on the Piper from PiperStudio

**For:** the agent working in `~/piper_studio`. This page says what PiperStudio must build to run a
trained flowcl policy closed-loop on the follower arm, and the contract with the flowcl policy
server it talks to. Written Mon 2 Nov 2026 from flowcl `da0196a` and PiperStudio `c595d8c`.
Everything here is built new; nothing is ported from older robot code.

**Who builds what:**
- **flowcl** (`~/flowcl`, its own uv env, the GPU): the policy server (§2). It owns everything that
  must match training exactly: the image mask and resize, normalisation, sampling. **Built:**
  `flowcl/deploy/piper_server.py`, `scripts/hw/serve_piper.py`, tests in
  `tests/test_piper_server.py`.
- **PiperStudio** (`~/piper_studio`, `piper` conda env): everything on the robot side (§3–§10):
  observations, executing chunks, safety, the operator UI, logs. It never imports flowcl or torch.

## 1. What is deployed

- **Task:** put the sausage in the pot, then the lid on the pot. Demos start and end at the home
  pose. Recorded 30 Oct as `~/piper_datasets/pick_sausage` and `pick_sausage_v3`.
- **Robot:** the follower was **arm2** (Second arm, USB port `3-1:1.0`, `can1`); arm1 led. The
  policy drives arm2 only. arm1 is not needed and can stay parked.
- **Policy:** flow-matching, 15 Hz, predicts 16 future absolute joint and gripper targets (1.07 s)
  from two camera images and the measured joints. The robot executes 8 of them (0.53 s), then asks
  again. One server process serves one checkpoint. The first one is
  `results/piper_aug/single__piper__sausage_pot_lid__seed0/checkpoints/final.pt`.
- **Expectation:** offline, no policy beats "hold the last command" on unseen demos. It may fail on
  the robot, so the safety checks and the logs matter as much as the motion.
- **The thesis gate this serves (F1):** replay 3 recorded demos open-loop and check tracking (§6.3).
  Then run 10 closed-loop rollouts from fixed initial conditions (§6.4). Success means ≥ 6/10, with
  an episode cycle (run, reset, annotate) of ≤ 4 min.

## 2. The policy server (flowcl side): the contract

```
same PC (RTX 4090)
flowcl:  scripts/hw/serve_piper.py --checkpoint ... --port 8765     (uv env, GPU)
            ^  HTTP + JSON on 127.0.0.1:8765
            v
PiperStudio:  backend policy client + runner  (piper env; PiperStudio itself stays on :8000)
```

Plain HTTP with JSON bodies, so both ends need only the standard library (the `piper` env has no
msgpack; that is fine). Units are always rad for joints and m for the gripper. Vectors are ordered
`[joint_1 … joint_6, gripper]`, as in the datasets.

**Starting it** (on the flowcl side; pause GPU training first):

```bash
cd ~/flowcl && uv run python scripts/hw/serve_piper.py --checkpoint results/piper_aug/single__piper__sausage_pot_lid__seed0/checkpoints/final.pt
```

It prints `listening on http://127.0.0.1:8765/` once the checkpoint is loaded and warmed up,
about 20 s. Every request is logged to `results/piper_serve/<run>__<checkpoint>/requests.jsonl`.

### `GET /metadata`

Fetch this once when connecting, show it in the UI, and store it in every log. For the first
checkpoint it returns (abridged):

```json
{
  "protocol": 1,
  "checkpoint": "results/piper_aug/single__piper__sausage_pot_lid__seed0/checkpoints/final.pt",
  "checkpoint_sha256": "bec304e37196c9fa…",
  "run_id": "single__piper__sausage_pot_lid__seed0", "task_key": "piper/sausage_pot_lid",
  "instruction": "put the sausage in the pot and cover it with the lid",
  "cameras": {"front": "observation.images.image", "wrist": "observation.images.wrist_image"},
  "frame_hw": [480, 640], "policy_image_hw": [128, 128],
  "image_masks": {"front": [[[0,0],[260,0],[260,115],[150,250],[0,260]]]},
  "state_names": ["joint_1", "…", "joint_6", "gripper"], "units": {"joints": "rad", "gripper": "m"},
  "control_rate_hz": 15.0, "chunk_horizon": 16, "execute_k": 8,
  "start_pose": [0.0, 0.5115, -0.5039, 0.0, 0.4451, 0.0, 0.0204],
  "envelope": {"lo": [-1.675, 0.3604, -2.1108, -0.5518, -0.0423, -1.8618, -0.005],
               "hi": [1.422, 2.4769, -0.0676, 1.0217, 1.3717, 2.036, 0.08],
               "margin_rad": 0.15, "margin_gripper_m": 0.005},
  "held_out": [{"dataset": "pick_sausage_v3", "episode_index": 10}, "… 11-15"],
  "n_demos_trained": 30, "euler_steps": 10, "server_seed": 0
}
```

- `start_pose` is the median first recorded command of the training demos. That is 0, 29.31,
  −28.87, 0, 25.50, 0° and a 20.4 mm gripper: the home pose as arm2 reads it.
- `envelope` holds the per-dimension min and max of the training actions, widened by the server's
  margin (0.15 rad on the joints, 5 mm on the gripper). The raw ranges are J1 −1.53…1.27,
  J2 0.51…2.33, J3 −1.96…−0.22, J4 −0.40…0.87, J5 0.11…1.22, J6 −1.71…1.89 rad, and the gripper
  0…0.075 m.
- `held_out` lists the episodes the policy never trained on, as raw LeRobot episode indices.
  They are the rollouts' initial conditions.

### `POST /infer`

```json
{
  "seq": 17,
  "instruction": "<metadata.instruction, verbatim>",
  "state": [7 floats],
  "images": {"front": "<base64 JPEG>", "wrist": "<base64 JPEG>"},
  "seed": 123456,
  "return_inputs": false
}
```
returns
```json
{"seq": 17, "actions": [[7 floats] x 16],
 "timing_ms": {"decode": 3.1, "policy": 41.2, "total": 45.0},
 "inputs": {"front": "<base64 PNG 128x128>", "wrist": "…"}}
```
- `images` are the camera's JPEG bytes exactly as `Camera.latest()` returns them: 640×480, not
  decoded, resized, cropped, masked or colour-converted.
  - The server decodes them as the recorder does (`cv2.imdecode`, BGR→RGB).
  - It refuses any other size.
  - It applies the checkpoint's mask, its crop (a working-area polygon, when it has one; listed
    in `metadata.image_crops`) and the resize to 128×128.
  - Doing any image processing on the client would break the match with training.
- `seed` is optional. Without it the server draws noise from its own generator. Sending it, and
  logging it, makes a request reproducible offline.
- `inputs` are present only when `return_inputs` is true (for the probe, §6.1): the two 128×128
  images the policy actually saw.
- On a bad request (wrong instruction, state not 7 finite numbers, a missing camera, a frame not
  640×480) or a server failure, the reply is a non-200 status with `{"error": "…"}`. The client
  treats any non-200 reply as a stop.
- **Measured** on the first checkpoint (44 requests from held-out episode `pick_sausage_v3` 10,
  frames re-encoded as JPEG):
  - the round trip from a client is p50 50 ms, p95 53 ms, max 84 ms (the policy alone takes
    41 ms);
  - a request carries about 115 kB of base64 JPEG;
  - the 128×128 inputs the server builds differ from the training cache's by 0.4/255 on average;
  - the chunks' joint error over the 8 executed ticks is 0.052 rad (offline check, same episode:
    0.051).

  Inference is sequential on the server, so only one client should drive it. A request from the
  `piper` env with `urllib` alone works.

`GET /` is a small status page (checkpoint, recent requests, latency, errors).

## 3. Conventions that must match the recordings

| What | Must be | Where it comes from in PiperStudio |
|---|---|---|
| Arm | arm2, the recording follower. Its actions are in arm2's own joint readings (teleop already applied `joint_offset_deg`), so send them as they are. No offset conversion, never arm1 | config `policy.arm` |
| State | arm2's **measured** J1–J6 in rad, plus its measured gripper opening in m. Exactly `observation.state` of `recorder.build_frame`: `radians(fb.joints)`, `fb.gripper_mm / 1000`. Never the last command | `arm.snapshot()` |
| Action | Absolute targets: J1–J6 rad and gripper m. Send `degrees(a[:6])` and `a[6] * 1000` mm through `ArmController.send_follow`, the same high follow path teleop recorded with | §4 |
| front camera | arm2's `top` camera: `CV2T6610002D` on 30 Oct, the "first position" mount | `camera_keys_for(cfg, "arm2", present)` key `image` |
| wrist camera | arm2's `gripper` camera: `CV275610001S` | same, key `wrist_image` |
| Image timing | The latest frame of each camera at the moment the state is read, as `Recorder._capture` does. That gives the policy the same image/state timing it trained on | `Camera.latest()` |
| Camera pose | The front camera must not have moved since 30 Oct. Check it with the overlay (§6.2) before every session | |
| Instruction | `metadata.instruction` verbatim | |
| Rate | 15 Hz policy ticks, chunk of 16, replan every 8 | `metadata` |
| Start pose | `metadata.start_pose` | §6.4 |

- **Refuse to start** unless `camera_keys_for` maps exactly one camera to `image` and one to
  `wrist_image`. A second streaming `top` camera on arm2 would come out as `image_<serial>`;
  unplug it or unlabel it.
- **Refuse to start** if the camera stream is not 640×480.
- The operator stood in the top-left corner of the front view during recording, and that corner is
  masked. During rollouts, stay there or out of the view.

## 4. The executor (the core of the robot side)

One executor runs every chunked motion: policy rollouts, and recorded demos replayed as chunks
(§6.3). Running replay through the same code path tests its timing, limiter and checks before any
policy moves the arm.

**Two threads:**
- A **control thread** at `teleop.rate_hz` (200 Hz).
- An **inference thread** that does the HTTP calls, so a slow reply never blocks control.

**The current chunk** is `C` (16×7, rad / m) with `t_obs`, the `time.monotonic()` instant its
observation was taken. The chunk is played on the timeline it was predicted for: tick `i` is due at
`t_obs + i / 15`. In training, the first action of a chunk is the command in force when its
observation is taken (the measured state trails it by 1–2 ticks), so this is the timing the policy
learned.

**Every control tick, at time `now`:**
1. Compute `p = (now − t_obs) × 15`, clamped to `[0, 15]`.
2. Interpolate the target linearly between `C[floor p]` and `C[floor p + 1]`. This is the same
   linear interpolation `ReplaySession._sample` does between recorded frames.
3. **Rate-limit** the target against the last command sent:
   - each joint by at most `policy.max_joint_speed_rad_s × dt`;
   - the gripper by at most `policy.max_gripper_speed_m_s × dt`;
   - `dt` is the real time since the last send, capped at two periods.

   Count every tick where the limiter changed the target.
4. Run the per-tick checks (§5). Then call `arm.send_follow(degrees(joints), gripper_m * 1000)`,
   which also clamps to the joint limits.
5. Log the raw target, the sent target and the measured feedback (§7).

**Replanning:**
- When `p ≥ execute_k` (8) and no request is in flight, read an observation: the arm2 snapshot and
  the latest JPEG of each camera, together. That sets `t_obs_new`. Hand it to the inference thread.
- Until the reply arrives, keep following the old chunk. Its ticks 8–15 are the bridge across the
  latency, so the arm never pauses.
- When the reply arrives, run the chunk checks (§5). If they pass, swap: `C, t_obs = new, t_obs_new`.
  At about 50 ms latency, the new chunk is then entered at `p ≈ 0.75`.
- **Deadline:** if no reply has passed the checks by `t_obs_new + policy.request_timeout_s` (0.4 s),
  stop.
  - The bridge lasts `(15 − 8) / 15 = 0.47 s`, so the old chunk can never run out before the
    deadline.
  - Validate at config load that `request_timeout_s < (horizon − 1 − execute_k) / rate_hz`.

**Start:** before Go, the arm holds the start pose, which is then the last command sent. On Go,
read the first observation and request a chunk. During that first request the arm keeps holding.
On the reply, start following at `p = (now − t_obs) × 15`.

**Stop**, for any reason:
- stop sending, call `arm.end_follow()` (the arm holds its last target) and release the arm;
- discard any chunk still in flight;
- log the reason;
- show the reason in the UI.

**Why these values:**
- The demos' largest per-tick joint step is at p99 0.04–0.09 rad (1.3 rad/s at most), with a
  maximum of 0.16 rad. `max_joint_speed_rad_s: 1.5` sits above every joint's p99, so it smooths
  the step at a chunk swap without slowing normal motion.
- The gripper's per-tick step is at p99 7.7 mm (0.12 m/s), with a maximum of 22 mm; hence 0.25 m/s.
- If more than 1 % of the ticks of a replay are limited, raise the limit and keep the new value for
  everything after.

## 5. Safety checks

**Before a chunk is used** (any failure stops the run before the chunk moves the arm):
- the reply is a 200 with `actions` of shape 16×7, all finite;
- every value of the chunk lies within `metadata.envelope`;
- **first-command jump:** at the entry point `p`, the interpolated target is within
  `policy.max_first_jump_rad` (0.15 rad; J6, the wrist roll, `policy.max_first_jump_j6_rad`, 0.5 rad)
  of the span between the last sent command and the arm's measured joints, on every joint. Offline, the policy's first command is typically 0.03 rad off.
  (Until 2 Nov the reference was the last command alone. On the 53-demo policy's first 29 rollouts,
  10 of 20 stops were a new chunk that started where the arm was, while the arm lagged its command
  by 0.09–0.15 rad.)

**Every control tick** (any failure stops the run):
- arm2 is `ENABLED`, not E-stopped, and `fb.alive`. This is the same test as
  `ReplaySession._problem`;
- tracking: `|sent − measured|` exceeds `teleop.max_tracking_error_deg` on some joint for longer
  than `teleop.tracking_error_grace_s`. These are the same limits teleop and replay use;
- **table floor:** the measured end pose `fb.end_pose[2]` (mm) is below `policy.floor_z_mm` (75).
  The demos never went below 95 mm;
- the time limit `policy.max_episode_s` (60 s; demos are 22–47 s) is reached. This is a normal end,
  not a fault;
- the operator stops: Stop, E-STOP, the stop button on the controller, or the last PiperStudio page
  closing (as for teleop).

**At every observation** (any failure stops the run):
- either camera's latest frame is older than `policy.max_frame_age_s` (0.25 s). Track the arrival
  time of each new `seq`.

**Ownership:**
- The runner claims arm2 (`arm.claim("policy")`) for the whole run, so the arm panel, teleop,
  replay and recording all refuse it meanwhile.
- The runner refuses to start while teleop, replay or recording use arm2.

The power switch stays the real hardware stop. Someone stands at the arm for every run.

## 6. Modes (a new **Policy** tab)

**Connection:** the server URL, connected or not, and the metadata summary: checkpoint and sha256
prefix, run id, instruction. Also the latency of recent requests (p50 / max).

### 6.1 Probe (no motion)

With arm2 enabled and still, one button sends one request with `return_inputs: true`. Nothing is
sent to the arm. The probe shows:
- the latency;
- whether each check of §5 passes, including the first-command jump from the current pose;
- the predicted 16-tick chunk per joint, against the current state;
- the two 128×128 inputs the policy saw, next to the policy inputs of a recorded frame. Fetch those
  by sending a recorded frame through the same request with `return_inputs`. The two inputs should
  match in colour, framing and mask.

Run it at the start pose before the first rollout of every session.

### 6.2 Camera and scene check

- Show the live front frame blended 50/50 with frame 0 of a chosen held-out episode (from the
  dataset videos), with the mask polygon drawn.
- The static scene (table edges, pot base, fixtures) must line up. If it does not, the camera has
  moved: stop and fix the mount before any rollout.
- The same overlay is how the operator places the objects for an initial condition.

### 6.3 Replay through the executor (F1 open-loop check)

**Source:** a recorded episode's actions `A` (T×7), with no server involved.
- Chunk `n` is `A[8n : 8n + 16]`, padded with the last action, and `t_obs_n = t_go + 8n / 15`.
- Executed as in §4, with the same limiter and checks. The envelope and first-jump checks apply
  too: a recorded episode passes them by construction.

**Procedure:**
- Move to the episode's first action as for a rollout start, set the scene from its first frame
  (§6.2), then Go.

**Metrics,** shown and logged at each recorded tick (the 15 Hz grid), measured state against the
recording's `observation.state`:
- per joint: RMSE and max |error| at lag 0;
- the best-fitting lag (−3…+3 ticks);
- the gripper's error (mm);
- the share of limited ticks;
- the operator's outcome (did the sausage and the lid end up in place).

**Tolerances,** proposed; the user fixes them before the first replay, then they stay:
- per joint, RMSE ≤ 0.02 rad and max ≤ 0.05 rad;
- best lag within ±1 tick;
- ≤ 1 % of ticks limited.

Do 3 episodes: `pick_sausage_v3` 10, 11 and 12. They are held out, so their scenes are the
rollouts' too. The existing Replay feature stays as it is.

### 6.4 Rollouts

**The operator loop**, one rollout at a time:
1. **Pick the initial condition** from `metadata.held_out` and a free-text label. Set the scene
   with the overlay.
2. **Move to start.**
   - The first time in a session, use `teleop_align(start_pose, …, teleop.align_speed)` until every
     joint is within `teleop.align_tolerance_deg`, as replay does.
   - After a rollout the arm is in high follow mode, so return to start in that mode instead: a
     straight joint-space line at `policy.return_speed_dps` (20°/s) through `send_follow`, as
     teleop's go-home does. This avoids switching back from high follow to position control,
     which can need a reset.
3. **Go**, from the page or the controller's A button. The executor runs (§4).
4. **The end:** Stop (page or controller B), a safety stop, or the time limit. The arm holds.
5. **Annotate:**
   - sausage in the pot: yes / no;
   - lid on the pot: yes / no;
   - success = both;
   - a note.

   The annotation is saved into that rollout's log.
6. Reset the scene; **return to start**; next.

**Cycle time:** measure Go to Go and show it. The gate's limit is 4 min. Also show the running
tally (successes so far, rollouts done).

## 7. Logs

Each probe, replay and rollout writes `~/piper_rollouts/<YYYY-MM-DD>/<HHMMSS>_<mode>/`:

- **`meta.json`:**
  - the server's full metadata;
  - the PiperStudio git commit and whether the tree was dirty;
  - the `policy`, `teleop` and joint-limit config values in force;
  - the arm, the camera serials, the initial condition and the label;
  - the start and end times, the stop reason;
  - the annotation and the cycle time.
- **`inferences.jsonl`,** one line per request:
  - `seq`, `seed`, `t_obs`, the send and reply times;
  - each camera's frame age, the state;
  - the returned chunk;
  - the server timing;
  - each check's result, and whether the chunk was used.
- **`frames/`:** the exact JPEGs sent, `<seq>_front.jpg` and `<seq>_wrist.jpg`. With these, flowcl
  can re-run any request offline.
- **`ticks.npz`,** at the control rate: time, `p`, the raw target, the sent target, the measured
  joints and gripper, `end_pose`, and the limited and check flags.

A rollout that stops on a fault keeps its log. Logs are never deleted by the UI.

## 8. Configuration (`config.yaml`, new `policy:` section)

```yaml
policy:
  server_url: http://127.0.0.1:8765
  arm: arm2                     # the recording follower
  request_timeout_s: 0.4        # inference deadline; must stay below the 0.47 s bridge
  max_frame_age_s: 0.25
  max_joint_speed_rad_s: 1.5    # demos' p99 per joint is at most 1.3 rad/s
  max_gripper_speed_m_s: 0.25
  max_first_jump_rad: 0.15
  max_first_jump_j6_rad: 0.5
  floor_z_mm: 75                # demos' lowest end pose: 95 mm
  max_episode_s: 60
  return_speed_dps: 20
  log_root: ~/piper_rollouts
```

Tracking limits, rates and alignment come from the existing `teleop:` section. Joint limits come
from `joint_limits_deg`.

## 9. Tests without hardware (`piper_studio/tests`, fake SDK)

Add a fake policy server: an in-process `http.server` that follows §2 with scripted replies.

- **Timing:**
  - the target at time `t` equals the interpolation of the chunk at `(t − t_obs) × 15`;
  - replans happen at `p = 8`;
  - a swap enters the new chunk at its own `p`;
  - the old chunk keeps playing while a reply is late.
- **Units and mapping:**
  - the state sent is the measured state in rad and m, not the command;
  - commands go out in degrees and mm;
  - `front` is arm2's top camera;
  - the JPEG bytes sent equal `Camera.latest()`'s byte for byte.
- **Each stop,** with the arm left holding and released:
  - a non-200 reply;
  - NaN in a chunk, the wrong shape;
  - a value outside the envelope;
  - a first jump above the limit;
  - a reply after the deadline;
  - a stale frame;
  - the floor;
  - the tracking error;
  - an E-STOP mid-run;
  - Stop;
  - the last page closing.
- **The limiter:**
  - it never exceeds the speed limits;
  - it is the identity when the targets are slow;
  - it counts the ticks it limited.
- **Replay as chunks:** with no limiting, the executed targets equal the recorded actions at the
  recorded ticks.
- **Exclusion:** the runner refuses while teleop, replay or recording use the arm, and they refuse
  while it runs.

## 10. The first session on the robot, in order

1. **On the flowcl side,** pause any GPU training (`kill -STOP <pid>`). Start the server, then open
   `http://127.0.0.1:8765/`.
2. **In PiperStudio:**
   - Start arm2;
   - check that the front camera is `CV2T6610002D` and the wrist camera `CV275610001S`, both
     streaming at 640×480;
   - connect on the Policy tab.
3. **Camera check** (§6.2) against `pick_sausage_v3` episode 10.
4. **Probe** ×3 at the start pose (§6.1):
   - the latency is under 150 ms;
   - every check passes;
   - the first-command jump is small (about 0.03 rad);
   - the inputs match the recorded ones.
5. **Stop test:**
   - replay an episode through the executor in free space (no objects) at the start;
   - press Stop mid-motion. The arm must hold where it is.
   - Repeat with E-STOP.
6. **Replay** ×3 through the executor (§6.3), against the tolerances fixed beforehand.
7. **Rollouts** (§6.4). Copy `~/piper_rollouts/` back into the flowcl record afterwards.

## 11. Not now

- Recording rollouts as LeRobot datasets, for training on them later.
- Cross-fading between chunks, or asynchronous chunk blending: start with the limiter.
- Serving more than one checkpoint per server, or switching checkpoints from the UI. Restart the
  server with another `--checkpoint` instead.
- Cartesian workspace limits beyond the table floor.
