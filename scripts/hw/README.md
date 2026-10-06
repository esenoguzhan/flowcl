# Running flowcl policies on the Dobot X-Trainer (branch `dobot-hw`)

A try-out on the lab's robot. AgileX remains the thesis plan's Stage B robot.

```
4090 PC (10.147.20.55, ZeroTier)                Robot PC (lab, xtrainer Python env)
scripts/serve_policy.py   <--- ws:8000 --->     scripts/hw/dobot_run.py {probe,stoptest,replay,rollouts}
  checkpoint -> 10 Hz chunk -> 30 Hz             cameras (RealSense threads) + ZMQ robot (launch_nodes.py)
```

The robot side uses `flowcl/deploy/{images,wire,safety}.py` (numpy, OpenCV, msgpack,
websockets; no torch). It imports the lab's robot and camera helpers from
`~/dobot_xtrainer_ACT` **read-only**, and nothing in that repository changes.

## 0. Once: get the code onto the robot PC
```bash
git clone -b dobot-hw --depth 1 git@github.com:esenoguzhan/flowcl.git ~/flowcl
# no GitHub access there: from the 4090 PC
#   scp -r flowcl/__init__.py flowcl/deploy scripts/hw <robot-pc>:~/flowcl/...
cd ~/flowcl && python -c "import numpy, cv2, websockets, msgpack; print('ok')"   # in the xtrainer env
```
If `websockets` or `msgpack` are missing, run `pip install websockets msgpack`. openpi-client
normally brings both.

## 1. 4090 PC: serve a checkpoint
```bash
cd ~/flowcl
# Pause the training queue first, so it does not compete for the GPU:
#   kill -STOP <pid of the running python>     (later: kill -CONT <pid>)
uv run python scripts/serve_policy.py \
  --checkpoint results/single__dobot__puzzle_red_retrieve__seed0/checkpoints/final.pt
```
The server prints `listening on ws://0.0.0.0:8000`. It logs every request to
`results/dobot_serve/<run>__<ckpt>/requests.jsonl` and dumps the first 5 inputs as npz.

Other checkpoints:
- `results/dobot_puzzle__seq_ft__seed0/checkpoints/stage{0,1,2}.pt`
- `results/dobot_puzzle__gpm_projected_adam_ne90__seed0/checkpoints/stage{0,1,2}.pt`
- `results/single__dobot__puzzle_{green,yellow}_retrieve__seed0/checkpoints/final.pt`

## 2. Robot PC: bring the robot up (the lab's terminal)
```bash
cd ~/dobot_xtrainer_ACT && python -m experiments.run_robot_terminal
> power-on-enable-both
> home            # starts launch_nodes.py (ZMQ robot server, port 6001)
```
Do **not** start teleop: it holds the cameras. Keep the E-stop within reach throughout.

## 3. Robot PC: the checks, in order (same settings throughout)
```bash
cd ~/flowcl
R="python scripts/hw/dobot_run.py --server ws://10.147.20.55:8000"

$R probe --task puzzle_red_retrieve        # no motion
$R stoptest                                # free space; press Ctrl-C mid-motion
$R replay --task puzzle_red_retrieve --episode 110   # then 111, 112
$R rollouts --task puzzle_red_retrieve --episodes 110-119
```
- **probe:** checks latency (expect < 150 ms), that the chunk checks pass, and that the first
  command is close to the current pose. Then compare the saved `live_*.png` with the dataset
  photos (`datasets/dobot/photos/puzzle_red_retrieve/ep11x_*.png` on the 4090 PC): crop, flip,
  colour, camera order.
- **stoptest:** both joint-1s move ±0.15 rad slowly. After Ctrl-C the arm must hold, and the
  printed drift should be ≈ 0.
- **replay:** place the objects as in photo `ep110`. It moves slowly to the start, then
  streams the recorded actions. It prints tracking error vs. the recording and asks for the
  task outcome.
  - Pass: RMSE ≤ 0.02 rad, max ≤ 0.05 rad, lag within ±2 ticks of the recording, ≤ 1% of
    ticks joint-clipped.
  - If clipping fails at the default `--max-joint-step 0.03`, rerun with 0.05 and keep that
    value for everything after.
- **rollouts:** the held-out episodes 110–119 are the initial conditions. For each one:
  set the scene from the photo, press Enter, and the arm moves slowly to the task's start pose.
  The policy then runs until you press Ctrl-C (it holds) or 60 s pass. Annotate `s`/`f`/`p`
  plus a note. Cycle time is measured start to start.

Safety stops (the arm holds, the red light comes on, and the rollout ends):
- an inference deadline of 1 s;
- a chunk whose first command is more than 0.15 rad from the pose;
- a chunk that leaves the training envelope (±0.15 rad) or the lab's J3/J4 limits;
- a gripper tip outside the lab's Cartesian box.

Logs (`results/dobot_hw/<time>_<mode>/`) hold `events.jsonl` plus one npz per
rollout/replay. Copy them back to the 4090 PC for the record.

Execution settings:
- 24 ticks executed per chunk (0.8 s at 30 Hz), then replan;
- joint step 0.03 rad/tick unless replay says 0.05;
- gripper step 0.1/tick.

Inference is synchronous, so the arm pauses for the inference latency between chunks.
