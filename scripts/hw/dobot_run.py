"""Robot-side runner for flowcl policies on the Dobot X-Trainer (branch dobot-hw).

Runs on the robot PC in the lab's xtrainer Python environment (numpy, OpenCV, websockets,
msgpack; no torch). From a flowcl checkout it imports only ``flowcl.deploy.{images, wire,
safety}``; from the xtrainer checkout it imports the robot/camera helpers read-only. The
xtrainer repository is never modified.

Before any mode: ``python -m experiments.run_robot_terminal`` in the xtrainer checkout,
then ``power-on-enable-both`` and ``home`` (this starts launch_nodes.py, the ZMQ robot
server). Do not start teleop: it holds the cameras.

Modes:
  probe     one inference on live cameras + state, **no motion**: latency, chunk checks,
            saved images for the live-vs-dataset comparison.
  stoptest  a small slow joint wiggle in free space; press Ctrl-C mid-motion to check the
            arm holds when commands stop.
  replay    stream a recorded (held-out) episode's 30 Hz actions through the same loop and
            limiter, then report tracking error against the recorded joint positions.
  rollouts  operator loop: reset scene -> Enter -> policy runs -> annotate (s/f/p).

Every mode writes JSONL + npz logs under ``--log-dir``. Examples:

  python scripts/hw/dobot_run.py --server ws://10.147.20.55:8000 probe --task puzzle_red_retrieve
  python scripts/hw/dobot_run.py --server ws://10.147.20.55:8000 replay --task puzzle_red_retrieve --episode 110
  python scripts/hw/dobot_run.py --server ws://10.147.20.55:8000 rollouts --task puzzle_red_retrieve --episodes 110-119
"""

from __future__ import annotations

import argparse
import json
import signal
import sys
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

FLOWCL_ROOT = Path(__file__).resolve().parents[2]
# First, so this checkout's flowcl.deploy is used even where another flowcl is installed. The
# xtrainer root is inserted later; its ``scripts`` is a regular package, so it still wins over
# flowcl's (namespace) ``scripts`` directory.
sys.path.insert(0, str(FLOWCL_ROOT))

from flowcl.deploy import wire  # noqa: E402
from flowcl.deploy.images import CAMERAS, decode_jpeg, encode_jpeg, recorder_bgr  # noqa: E402
from flowcl.deploy.safety import (  # noqa: E402
    GRIPPER_DIMS,
    JOINT_DIMS,
    Envelope,
    cartesian_violations,
    check_prefix,
    limit_action_step,
)

GRIP = list(GRIPPER_DIMS)
STOP = threading.Event()


def _on_sigint(signum, frame) -> None:
    if STOP.is_set():
        raise KeyboardInterrupt
    STOP.set()
    print("\n[run] stop requested: holding (Ctrl-C again to abort)", flush=True)


# ---- robot and cameras ------------------------------------------------------------------


class RealRobot:
    """The lab's ZMQ robot server through xtrainer's RobotEnv (read-only import)."""

    def __init__(self, xtrainer_root: Path, host: str, port: int) -> None:
        sys.path.insert(0, str(xtrainer_root))
        from dobot_control.env import RobotEnv
        from dobot_control.robots.robot_node import ZMQClientRobot
        from scripts.manipulate_utils import set_light

        self._client = ZMQClientRobot(port=port, host=host)
        self._env = RobotEnv(self._client)
        self._set_light = set_light

    def joints(self) -> np.ndarray:
        return np.asarray(self._env.get_obs()["joint_positions"], np.float32).reshape(-1).copy()

    def step(self, command: np.ndarray) -> None:
        self._env.step(np.asarray(command, np.float64), np.array([1, 1]))

    def pose(self) -> np.ndarray:
        return np.asarray(self._env.get_XYZrxryrz_state(), np.float64).reshape(-1)

    def light(self, color: str) -> None:
        try:
            self._set_light(self._env, color, 1)
        except Exception as exc:  # the light is a convenience, never a reason to fail
            print(f"[run] light {color} failed: {exc}", flush=True)

    def close(self) -> None:
        self._client.close()


class MockRobot:
    """First-order lag follower for offline tests (about 3 ticks)."""

    def __init__(self, start: np.ndarray, alpha: float = 0.3) -> None:
        self.q = np.asarray(start, np.float32).copy()
        self.alpha = alpha

    def joints(self) -> np.ndarray:
        return self.q.copy()

    def step(self, command: np.ndarray) -> None:
        self.q += self.alpha * (np.asarray(command, np.float32) - self.q)

    def pose(self) -> np.ndarray:
        return np.array([0, -400, 200, 0, 0, 0, 0, -400, 200, 0, 0, 0], np.float64)

    def light(self, color: str) -> None:
        pass

    def close(self) -> None:
        pass


class RealCameras:
    """One thread per RealSense camera keeping the latest frame (as teleop recording does)."""

    def __init__(self, xtrainer_root: Path) -> None:
        sys.path.insert(0, str(xtrainer_root))
        from dobot_control.cameras.realsense_camera import RealSenseCamera
        from scripts.manipulate_utils import load_ini_data_camera

        serials = load_ini_data_camera()
        # Same flips as run_control_terminal.py: top and right are rotated 180 degrees.
        self._cams = {
            "top": RealSenseCamera(flip=True, device_id=serials["top"]),
            "left_wrist": RealSenseCamera(flip=False, device_id=serials["left"]),
            "right_wrist": RealSenseCamera(flip=True, device_id=serials["right"]),
        }
        self._latest: dict[str, tuple[np.ndarray, float]] = {}
        self._lock = threading.Lock()
        for name, cam in self._cams.items():
            threading.Thread(target=self._loop, args=(name, cam), daemon=True).start()
        deadline = time.monotonic() + 10.0
        while len(self._latest) < len(self._cams):
            if time.monotonic() > deadline:
                raise RuntimeError(f"cameras not streaming: have {sorted(self._latest)}")
            time.sleep(0.05)

    def _loop(self, name: str, cam) -> None:
        while True:
            rgb, _ = cam.read()
            bgr = recorder_bgr(np.asarray(rgb), name)
            with self._lock:
                self._latest[name] = (bgr, time.monotonic())

    def read(self) -> tuple[dict[str, np.ndarray], float]:
        """Recorder-format BGR frames and the age (s) of the oldest one."""
        now = time.monotonic()
        with self._lock:
            frames = {k: v[0] for k, v in self._latest.items()}
            age = max(now - v[1] for v in self._latest.values())
        return frames, age


class DatasetCameras:
    """Replays a recorded episode's frames (recorder-format BGR) in real time, for offline tests."""

    def __init__(self, raw_root: Path, episode_index: int) -> None:
        import cv2

        info = json.loads((raw_root / "meta" / "info.json").read_text())
        keys = {"top": "obs.camera.top", "left_wrist": "obs.wrist_camera.left", "right_wrist": "obs.wrist_camera.right"}
        self.frames = {}
        for cam, key in keys.items():
            path = raw_root / info["video_path"].format(video_key=key, episode_index=episode_index, episode_chunk=0)
            cap, frames = cv2.VideoCapture(str(path)), []
            while True:
                ok, bgr = cap.read()
                if not ok:
                    break
                frames.append(bgr)
            self.frames[cam] = frames
        self.t0 = time.monotonic()

    def read(self) -> tuple[dict[str, np.ndarray], float]:
        i = int((time.monotonic() - self.t0) * 30)
        return {c: f[min(i, len(f) - 1)] for c, f in self.frames.items()}, 0.0


# ---- server client ----------------------------------------------------------------------


class PolicyClient:
    def __init__(self, url: str, timeout_s: float) -> None:
        from websockets.sync.client import connect

        self.timeout_s = timeout_s
        self.ws = connect(url, compression=None, max_size=None, open_timeout=timeout_s)
        self.meta = self._recv()
        self.seq = 0

    def _recv(self):
        try:
            message = self.ws.recv(timeout=self.timeout_s)
        except TypeError:  # websockets < 12 has no recv timeout
            message = self.ws.recv()
        if isinstance(message, str):
            raise RuntimeError(f"policy server error:\n{message}")
        return wire.unpackb(message)

    def call(self, request: dict) -> dict:
        self.ws.send(wire.packb(request))
        return self._recv()

    def infer(self, frames: dict[str, np.ndarray], state: np.ndarray, prompt: str, seed: int | None = None) -> tuple[dict, float]:
        self.seq += 1
        request = {"type": "infer", "seq": self.seq, "prompt": prompt, "t_obs": time.time(),
                   "state": np.asarray(state, np.float32),
                   "images": {c: encode_jpeg(frames[c]) for c in CAMERAS}}
        if seed is not None:
            request["seed"] = seed
        t = time.perf_counter()
        reply = self.call(request)
        return reply, 1e3 * (time.perf_counter() - t)

    def close(self) -> None:
        self.ws.close()


# ---- logging ----------------------------------------------------------------------------


@dataclass
class TickLog:
    t: list = field(default_factory=list)
    command: list = field(default_factory=list)
    measured: list = field(default_factory=list)
    clipped: list = field(default_factory=list)
    gripper_clipped: list = field(default_factory=list)
    chunk: list = field(default_factory=list)

    def add(self, t, command, measured, clipped, chunk, gripper_clipped=False) -> None:
        self.t.append(t)
        self.command.append(np.asarray(command, np.float32))
        self.measured.append(np.asarray(measured, np.float32))
        self.clipped.append(bool(clipped))
        self.gripper_clipped.append(bool(gripper_clipped))
        self.chunk.append(int(chunk))

    def save(self, path: Path, **extra) -> None:
        arrays = {k: np.asarray(v) for k, v in vars(self).items()}
        np.savez_compressed(path, **arrays, **extra)


class Session:
    def __init__(self, log_dir: Path, mode: str) -> None:
        self.dir = log_dir / f"{time.strftime('%Y%m%d_%H%M%S')}_{mode}"
        self.dir.mkdir(parents=True, exist_ok=True)
        self.jsonl = self.dir / "events.jsonl"

    def event(self, **record) -> None:
        record["t"] = time.time()
        with open(self.jsonl, "a") as f:
            f.write(json.dumps(record, default=lambda o: np.asarray(o).tolist()) + "\n")


# ---- motion primitives ------------------------------------------------------------------


class Ticker:
    def __init__(self, hz: float) -> None:
        self.period = 1.0 / hz
        self.next = time.monotonic()

    def wait(self) -> float:
        """Sleep to the next tick; returns how late this tick is (s)."""
        self.next += self.period
        late = time.monotonic() - self.next
        if late < 0:
            time.sleep(-late)
            return 0.0
        if late > self.period:  # fell behind (inference pause): restart the grid
            self.next = time.monotonic()
        return late


def move_slowly(robot, target: np.ndarray, max_step_rad: float, hz: float, session: Session, label: str) -> np.ndarray:
    """Interpolate from the current joints to ``target``; returns the last command."""
    current = robot.joints()
    target = np.asarray(target, np.float32)
    span = float(np.max(np.abs(target[JOINT_DIMS] - current[JOINT_DIMS])))
    n = max(1, int(np.ceil(span / max_step_rad)))
    session.event(event="move", label=label, span_rad=span, ticks=n)
    ticker = Ticker(hz)
    command = current
    for k in range(1, n + 1):
        if STOP.is_set():
            break
        ticker.wait()
        command = current + (target - current) * (k / n)
        robot.step(command)
    return command


def confirm(prompt: str) -> bool:
    signal.signal(signal.SIGINT, signal.default_int_handler)
    try:
        return input(f"{prompt} [y/N] ").strip().lower() == "y"
    finally:
        signal.signal(signal.SIGINT, _on_sigint)


def state_from(joints: np.ndarray, last_command: np.ndarray) -> np.ndarray:
    """Training convention: measured joints, grippers = the last commanded gripper."""
    state = np.asarray(joints, np.float32).copy()
    state[GRIP] = np.asarray(last_command, np.float32)[GRIP]
    return state


# ---- policy episode ---------------------------------------------------------------------


def run_policy(robot, cameras, client: PolicyClient, args, session: Session, envelope: Envelope,
               tag: str) -> dict:
    meta = client.meta
    prompt = meta["languages_by_task"][args.task]
    execute = args.execute_ticks or int(meta["execute_ticks"])
    log = TickLog()
    latencies, ages, problems, n_chunks = [], [], [], 0
    last_command = robot.joints()  # as run_policy_control: start from the measured pose
    ticker = Ticker(args.rate)
    t_start = time.time()
    outcome = "time_limit"
    while time.time() - t_start < args.max_seconds:
        if STOP.is_set():
            outcome = "operator_stop"
            break
        pose = robot.pose()
        bad = cartesian_violations(pose)
        if bad:
            problems, outcome = bad, "cartesian_limit"
            break
        joints = robot.joints()
        frames, age = cameras.read()
        try:
            reply, latency = client.infer(frames, state_from(joints, last_command), prompt)
        except Exception as exc:
            problems, outcome = [f"inference failed: {exc!r}"], "inference_error"
            break
        latencies.append(latency)
        ages.append(age)
        prefix = np.asarray(reply["actions_30hz"], np.float32)[:execute]
        bad = check_prefix(prefix, last_command, envelope, args.max_first_jump)
        session.event(event="chunk", tag=tag, chunk=n_chunks, latency_ms=latency, obs_age_s=age,
                      server_timing=reply.get("timing"), problems=bad, first=prefix[0], last=prefix[-1])
        if bad:
            problems, outcome = bad, "chunk_rejected"
            break
        for k in range(prefix.shape[0]):
            if STOP.is_set():
                break
            ticker.wait()
            if STOP.is_set():  # checked again right before motion
                break
            command, clipped, grip_clipped = limit_action_step(prefix[k], last_command, args.max_joint_step,
                                                               args.max_gripper_step)
            robot.step(command)
            last_command = command
            log.add(time.time(), command, robot.joints() if args.log_measured else command, clipped, n_chunks,
                    grip_clipped)
        n_chunks += 1
    if problems:
        robot.light("red")
        print(f"[run] stopped: {outcome}: {problems}", flush=True)
    summary = {
        "outcome": outcome, "problems": problems, "seconds": time.time() - t_start, "chunks": n_chunks,
        "ticks": len(log.t), "clipped_frac": float(np.mean(log.clipped)) if log.t else 0.0,
        "gripper_clipped_frac": float(np.mean(log.gripper_clipped)) if log.t else 0.0,
        "latency_ms_p50": float(np.percentile(latencies, 50)) if latencies else None,
        "latency_ms_p95": float(np.percentile(latencies, 95)) if latencies else None,
        "obs_age_s_max": float(max(ages)) if ages else None,
        "tick_interval_ms_p95": float(np.percentile(np.diff(log.t) * 1e3, 95)) if len(log.t) > 2 else None,
        "tick_interval_ms_max": float(np.max(np.diff(log.t)) * 1e3) if len(log.t) > 2 else None,
    }
    log.save(session.dir / f"{tag}.npz", latency_ms=np.asarray(latencies), obs_age_s=np.asarray(ages))
    return summary


# ---- modes ------------------------------------------------------------------------------


def mode_probe(robot, cameras, client, args, session, envelope) -> None:
    import cv2

    meta = client.meta
    prompt = meta["languages_by_task"][args.task]
    joints = robot.joints()
    frames, age = cameras.read()
    for cam, bgr in frames.items():
        cv2.imwrite(str(session.dir / f"live_{cam}.png"), bgr)
    reply, latency = client.infer(frames, state_from(joints, joints), prompt)
    prefix = np.asarray(reply["actions_30hz"], np.float32)[: int(meta["execute_ticks"])]
    bad = check_prefix(prefix, joints, envelope, args.max_first_jump)
    jump = np.abs(prefix[0, JOINT_DIMS] - joints[JOINT_DIMS])
    start = np.asarray(meta["start_poses"][args.task], np.float32)
    session.event(event="probe", joints=joints, latency_ms=latency, obs_age_s=age, problems=bad,
                  first_jump_rad=jump, server_timing=reply.get("timing"), prefix=prefix)
    print(f"[probe] latency {latency:.0f} ms (server {reply['timing']}), camera age {age * 1e3:.0f} ms")
    print(f"[probe] joints now      {np.round(joints, 3).tolist()}")
    print(f"[probe] task start pose {np.round(start, 3).tolist()}")
    print(f"[probe] first command   {np.round(prefix[0], 3).tolist()}  (max jump {jump.max():.3f} rad)")
    print(f"[probe] chunk checks: {'OK' if not bad else bad}")
    timings = []
    for _ in range(args.repeat):
        frames, _ = cameras.read()
        timings.append(client.infer(frames, state_from(joints, joints), prompt)[1])
    if timings:
        print(f"[probe] {args.repeat} more calls: p50 {np.percentile(timings, 50):.0f} ms, max {max(timings):.0f} ms")
    print(f"[probe] live images saved in {session.dir}; compare them with the dataset photos")


def mode_stoptest(robot, args, session) -> None:
    start = robot.joints()
    if not confirm("Free-space stop test: joint 1 of both arms moves +-0.15 rad slowly. Area clear?"):
        return
    print("[stoptest] press Ctrl-C while it moves; the arm should stop and hold", flush=True)
    ticker = Ticker(args.rate)
    t0 = time.time()
    log = TickLog()
    command = start.copy()
    while time.time() - t0 < 12 and not STOP.is_set():
        ticker.wait()
        offset = 0.15 * np.sin(2 * np.pi * (time.time() - t0) / 6.0)
        target = start.copy()
        target[[0, 7]] += offset
        command, _, _ = limit_action_step(target, command, args.max_joint_step, args.max_gripper_step)
        robot.step(command)
        log.add(time.time(), command, robot.joints(), False, 0)
    held = []
    for _ in range(30):  # 1 s after the last command
        time.sleep(1 / 30)
        held.append(robot.joints())
    drift = float(np.max(np.abs(np.asarray(held)[:, JOINT_DIMS] - command[JOINT_DIMS])))
    log.save(session.dir / "stoptest.npz", held=np.asarray(held))
    session.event(event="stoptest", stopped_by_operator=STOP.is_set(), drift_after_stop_rad=drift)
    print(f"[stoptest] max joint distance from the last command over 1 s: {drift:.4f} rad")
    STOP.clear()
    if confirm("Return slowly to the start joints?"):
        move_slowly(robot, start, args.approach_step, args.rate, session, "stoptest_return")


def replay_metrics(commands: np.ndarray, measured: np.ndarray, recorded_q: np.ndarray, recorded_a: np.ndarray) -> dict:
    n = min(len(measured), len(recorded_q))
    m, q = measured[:n][:, JOINT_DIMS], recorded_q[:n][:, JOINT_DIMS]
    err = m - q
    lags = range(0, 11)

    def best_lag(cmd, meas):
        scores = [np.mean(np.abs(cmd[: len(cmd) - L] - meas[L:])) for L in lags]
        return int(np.argmin(scores))

    return {
        "ticks": int(n),
        "rmse_per_joint_rad": np.sqrt(np.mean(err ** 2, axis=0)).round(4).tolist(),
        "rmse_max_rad": float(np.sqrt(np.mean(err ** 2, axis=0)).max()),
        "abs_err_max_rad": float(np.abs(err).max()),
        "lag_ticks_replay": best_lag(commands[:n][:, JOINT_DIMS], m),
        "lag_ticks_recording": best_lag(recorded_a[:n][:, JOINT_DIMS], q),
    }


def mode_replay(robot, client, args, session) -> None:
    episode = client.call({"type": "episode", "task": args.task, "episode_index": args.episode})
    actions, qpos = np.asarray(episode["actions"], np.float32), np.asarray(episode["qpos"], np.float32)
    print(f"[replay] {args.task} episode {args.episode} (held out: {episode['held_out']}), "
          f"{len(actions)} ticks = {len(actions) / 30:.1f} s. Place the objects as in photo ep{args.episode:03d}.")
    if not confirm(f"Move slowly to the episode's start pose, then replay at {args.rate:.0f} Hz?"):
        return
    last = move_slowly(robot, actions[0], args.approach_step, args.rate, session, "replay_start")
    time.sleep(0.5)
    log = TickLog()
    ticker = Ticker(args.rate)
    for k in range(len(actions)):
        if STOP.is_set():
            break
        ticker.wait()
        if STOP.is_set():
            break
        command, clipped, grip_clipped = limit_action_step(actions[k], last, args.max_joint_step, args.max_gripper_step)
        robot.step(command)
        last = command
        log.add(time.time(), command, robot.joints(), clipped, 0, grip_clipped)
    measured = np.asarray(log.measured)
    metrics = replay_metrics(np.asarray(log.command), measured, qpos, actions)
    metrics["clipped_frac"] = float(np.mean(log.clipped)) if log.clipped else 0.0
    metrics["gripper_clipped_frac"] = float(np.mean(log.gripper_clipped)) if log.clipped else 0.0
    metrics["completed"] = not STOP.is_set()
    tol = {"rmse_max_rad": 0.02, "abs_err_max_rad": 0.05, "lag_delta_ticks": 2, "clipped_frac": 0.01}
    metrics["pass"] = bool(metrics["rmse_max_rad"] <= tol["rmse_max_rad"]
                           and metrics["abs_err_max_rad"] <= tol["abs_err_max_rad"]
                           and abs(metrics["lag_ticks_replay"] - metrics["lag_ticks_recording"]) <= tol["lag_delta_ticks"]
                           and metrics["clipped_frac"] <= tol["clipped_frac"])
    log.save(session.dir / f"replay_{args.task}_ep{args.episode:03d}.npz", recorded_qpos=qpos, recorded_actions=actions)
    if not STOP.is_set():
        print("[replay] task outcome? ", end="")
        metrics["task_outcome"] = annotate()
    session.event(event="replay", task=args.task, episode=args.episode, tolerances=tol,
                  max_joint_step=args.max_joint_step, **metrics)
    print(f"[replay] {json.dumps(metrics)}")


def annotate() -> dict:
    signal.signal(signal.SIGINT, signal.default_int_handler)
    try:
        while True:
            answer = input("success/fail/partial [s/f/p] + optional note: ").strip()
            if answer[:1].lower() in ("s", "f", "p"):
                return {"label": {"s": "success", "f": "fail", "p": "partial"}[answer[0].lower()],
                        "note": answer[1:].strip(), "t_annotated": time.time()}
    finally:
        signal.signal(signal.SIGINT, _on_sigint)


def parse_episodes(text: str) -> list[int]:
    out = []
    for part in text.split(","):
        if "-" in part:
            a, b = part.split("-")
            out.extend(range(int(a), int(b) + 1))
        elif part:
            out.append(int(part))
    return out


def mode_rollouts(robot, cameras, client, args, session, envelope) -> None:
    meta = client.meta
    start = np.asarray(meta["start_poses"][args.task], np.float32)
    episodes = parse_episodes(args.episodes)
    session.event(event="session", task=args.task, episodes=episodes, checkpoint=meta["checkpoint"],
                  checkpoint_sha256=meta["checkpoint_sha256"], trained=meta["trained"],
                  execute_ticks=args.execute_ticks or meta["execute_ticks"], rate=args.rate,
                  max_joint_step=args.max_joint_step, max_gripper_step=args.max_gripper_step,
                  max_first_jump=args.max_first_jump, max_seconds=args.max_seconds)
    print(f"[rollouts] {meta['checkpoint']} on {args.task}; {len(episodes)} initial conditions")
    results = []
    previous_start = None
    for n, ep in enumerate(episodes):
        STOP.clear()
        signal.signal(signal.SIGINT, signal.default_int_handler)
        try:
            answer = input(f"\n[{n + 1}/{len(episodes)}] Set the scene as photo ep{ep:03d}, then Enter "
                           "(s = skip, q = quit): ").strip().lower()
        finally:
            signal.signal(signal.SIGINT, _on_sigint)
        if answer == "q":
            break
        if answer == "s":
            continue
        t_ready = time.time()
        move_slowly(robot, start, args.approach_step, args.rate, session, f"start_ep{ep}")
        if STOP.is_set():
            print("[rollouts] approach interrupted; quitting")
            break
        robot.light("green")
        summary = run_policy(robot, cameras, client, args, session, envelope, tag=f"rollout_{n:02d}_ep{ep:03d}")
        robot.light("yellow")
        outcome = annotate()
        record = {"event": "rollout", "index": n, "initial_condition": ep, "t_ready": t_ready, **summary,
                  **outcome, "cycle_s_from_previous": (t_ready - previous_start) if previous_start else None}
        previous_start = t_ready
        session.event(**record)
        results.append(record)
        print(f"[rollouts] {outcome['label']} after {summary['seconds']:.1f} s; "
              f"latency p50 {summary['latency_ms_p50']} ms; clipped {summary['clipped_frac']:.3f}")
        if summary["outcome"] in ("chunk_rejected", "cartesian_limit", "inference_error"):
            if not confirm("The run stopped on a safety check. Continue with the next rollout?"):
                break
    successes = sum(r["label"] == "success" for r in results)
    cycles = [r["cycle_s_from_previous"] for r in results if r["cycle_s_from_previous"]]
    session.event(event="summary", n=len(results), successes=successes,
                  cycle_s_median=float(np.median(cycles)) if cycles else None)
    print(f"\n[rollouts] {successes}/{len(results)} successes; median cycle "
          f"{np.median(cycles) if cycles else float('nan'):.0f} s; logs in {session.dir}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--server", default="ws://10.147.20.55:8000")
    parser.add_argument("--xtrainer-root", type=Path, default=Path.home() / "dobot_xtrainer_ACT")
    parser.add_argument("--robot", choices=("real", "mock"), default="real")
    parser.add_argument("--robot-host", default="127.0.0.1")
    parser.add_argument("--robot-port", type=int, default=6001)
    parser.add_argument("--cameras", default="real", help="'real' or 'dataset:<raw dataset dir>:<episode>' (offline tests)")
    parser.add_argument("--log-dir", type=Path, default=FLOWCL_ROOT / "results" / "dobot_hw")
    parser.add_argument("--rate", type=float, default=30.0)
    parser.add_argument("--execute-ticks", type=int, default=0, help="0 = the server's execute_k * stride (24)")
    parser.add_argument("--max-joint-step", type=float, default=0.03, help="rad per tick (choose from replay)")
    parser.add_argument("--max-gripper-step", type=float, default=0.1,
                        help="per tick; the recordings exceed 0.1 on 0.3-0.5%% of ticks (0.05 on ~2%%)")
    parser.add_argument("--max-first-jump", type=float, default=0.15, help="rad from current pose to a chunk's first command")
    parser.add_argument("--approach-step", type=float, default=0.01, help="rad per tick for slow moves")
    parser.add_argument("--max-seconds", type=float, default=60.0)
    parser.add_argument("--timeout", type=float, default=1.0, help="inference deadline (s)")
    parser.add_argument("--log-measured", action=argparse.BooleanOptionalAction, default=True)
    sub = parser.add_subparsers(dest="mode", required=True)
    p = sub.add_parser("probe")
    p.add_argument("--task", required=True)
    p.add_argument("--repeat", type=int, default=10)
    sub.add_parser("stoptest")
    p = sub.add_parser("replay")
    p.add_argument("--task", required=True)
    p.add_argument("--episode", type=int, required=True)
    p = sub.add_parser("rollouts")
    p.add_argument("--task", required=True)
    p.add_argument("--episodes", required=True, help="initial-condition photos, e.g. 110-119")
    args = parser.parse_args()

    signal.signal(signal.SIGINT, _on_sigint)
    session = Session(args.log_dir, args.mode)
    session.event(event="start", argv=sys.argv, args={k: str(v) for k, v in vars(args).items()})
    client = None
    if args.mode != "stoptest":
        client = PolicyClient(args.server, timeout_s=max(args.timeout, 5.0))
        client.timeout_s = args.timeout
        meta = client.meta
        meta["languages_by_task"] = {v: k for k, v in meta["languages"].items()}
        session.event(event="server", metadata={k: v for k, v in meta.items() if k != "envelope"})
        print(f"[run] server: {meta['checkpoint']} trained {meta['trained']}")
        if getattr(args, "task", None) and args.task not in meta["languages_by_task"]:
            raise SystemExit(f"unknown task {args.task}; server knows {list(meta['languages_by_task'])}")
    envelope = Envelope.from_dict(client.meta["envelope"]) if client else None

    if args.robot == "real":
        robot = RealRobot(args.xtrainer_root, args.robot_host, args.robot_port)
    else:
        start = client.meta["start_poses"][args.task] if client and getattr(args, "task", None) else np.zeros(14)
        robot = MockRobot(np.asarray(start, np.float32))
    cameras = None
    if args.mode in ("probe", "rollouts"):
        if args.cameras == "real":
            cameras = RealCameras(args.xtrainer_root)
        else:
            _, raw, ep = args.cameras.split(":")
            cameras = DatasetCameras(Path(raw), int(ep))
    try:
        if args.mode == "probe":
            mode_probe(robot, cameras, client, args, session, envelope)
        elif args.mode == "stoptest":
            mode_stoptest(robot, args, session)
        elif args.mode == "replay":
            mode_replay(robot, client, args, session)
        else:
            mode_rollouts(robot, cameras, client, args, session, envelope)
    except KeyboardInterrupt:
        print("\n[run] aborted by operator; the arm holds its last command", flush=True)
        session.event(event="aborted")
    finally:
        if client:
            client.close()
        robot.close()
        print(f"[run] logs: {session.dir}")


if __name__ == "__main__":
    main()
