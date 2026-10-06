"""Websocket policy server for the Dobot X-Trainer runner (branch dobot-hw).

One process serves one checkpoint. ``scripts/serve_policy.py`` is the entry point; the
robot-side client is ``scripts/hw/dobot_run.py``. Message format: :mod:`flowcl.deploy.wire`.

Requests (dicts):

* ``{"type": "infer", "images": {camera: JPEG bytes}, "state": f32[14], "prompt": str,
  "seq": int, optional "seed": int}`` -> ``{"actions_10hz": f32[H, 14],
  "actions_30hz": f32[3(H-1)+1, 14], "timing": {...}, "seq": int}``. The JPEG bytes are
  the recorder-format frames (:mod:`flowcl.deploy.images`); ``policy_images`` (already
  reduced RGB frames) may be sent instead, for offline checks.
* ``{"type": "episode", "task": name, "episode_index": i}`` -> the recorded 30 Hz
  ``actions`` / ``qpos`` of one cached episode, for the replay check on the robot.

Errors are returned as a string (the traceback); the runner stops on a string reply.
"""

from __future__ import annotations

import asyncio
import hashlib
import itertools
import json
import threading
import time
import traceback
from pathlib import Path

import numpy as np
import torch

from flowcl.data.dobot_adapter import stride_for
from flowcl.data.dobot_tasks import TASKS, cache_path
from flowcl.deploy import wire
from flowcl.deploy.images import decode_jpeg, to_policy_frame
from flowcl.deploy.safety import Envelope, upsample_chunk
from flowcl.train.checkpoint import load_checkpoint

GRIPPER_DIMS = (6, 13)


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def training_actions(holdout: int, dataset_dir: Path | None = None) -> dict[str, np.ndarray]:
    """30 Hz actions of every cached task's training episodes (all but the last ``holdout``)."""
    import h5py

    out = {}
    for name, task in TASKS.items():
        path = cache_path(name, dataset_dir)
        if not path.is_file():
            continue
        with h5py.File(path, "r") as f:
            names = sorted(f["episodes"])[: task.n_episodes - holdout]
            out[name] = np.concatenate([np.asarray(f["episodes"][n]["actions"]) for n in names])
    return out


def start_poses(holdout: int, dataset_dir: Path | None = None) -> dict[str, list[float]]:
    """Per task: the median first command of its training episodes (the rollout start pose)."""
    import h5py

    out = {}
    for name, task in TASKS.items():
        path = cache_path(name, dataset_dir)
        if not path.is_file():
            continue
        with h5py.File(path, "r") as f:
            names = sorted(f["episodes"])[: task.n_episodes - holdout]
            firsts = np.stack([np.asarray(f["episodes"][n]["actions"][0]) for n in names])
        out[name] = np.median(firsts, axis=0).round(5).tolist()
    return out


class PolicyService:
    """A loaded checkpoint plus everything the runner needs to execute it safely."""

    def __init__(
        self,
        checkpoint: Path,
        device: str = "cuda",
        euler_steps: int = 10,
        seed: int = 0,
        envelope_margin_rad: float = 0.15,
        holdout: int = 10,
        dataset_dir: Path | None = None,
        log_dir: Path | None = None,
        dump_requests: int = 0,
    ) -> None:
        self.checkpoint = Path(checkpoint)
        self.device = torch.device(device)
        loaded = load_checkpoint(self.checkpoint, device=self.device)
        self.policy, self.spec, self.stats, payload = loaded.policy, loaded.spec, loaded.stats, loaded.payload
        if self.spec.name != "dobot_xtrainer":
            raise ValueError(f"{checkpoint} is a {self.spec.name!r} checkpoint, not dobot_xtrainer")
        self.stride = stride_for(self.spec)
        self.euler_steps = int(euler_steps)
        self.generator = torch.Generator(device="cpu").manual_seed(int(seed))
        self.holdout = int(holdout)
        self.dataset_dir = dataset_dir
        self.languages = {t.language: name for name, t in TASKS.items()}
        actions = training_actions(self.holdout, dataset_dir)
        if not actions:
            raise FileNotFoundError("no dobot caches found; the envelope needs the training actions")
        self.envelope = Envelope.from_actions(np.concatenate(list(actions.values())), envelope_margin_rad)
        self.envelope_tasks = sorted(actions)
        self.start_poses = start_poses(self.holdout, dataset_dir)
        self.trained = {
            "run_id": payload.get("run_id"),
            "stage": payload.get("stage"),
            "task_key": payload.get("task_key"),
            "task_keys": (payload.get("extra") or {}).get("task_keys"),
        }
        self.sha256 = _sha256(self.checkpoint)
        self.lock = threading.Lock()
        self.log_dir = Path(log_dir) if log_dir else None
        if self.log_dir:
            self.log_dir.mkdir(parents=True, exist_ok=True)
        self.dump_left = int(dump_requests)
        self._warm_up()

    def _warm_up(self) -> None:
        h, w = self.spec.observation.image_size
        request = {
            "policy_images": {c: np.zeros((h, w, 3), np.uint8) for c in self.spec.cameras},
            "state": np.asarray(self.start_poses.get(next(iter(self.start_poses)), np.zeros(14)), np.float32),
            "prompt": next(iter(self.languages)),
            "seed": 0,
        }
        for _ in range(2):
            self.infer(request, log=False)

    def metadata(self) -> dict:
        h = self.spec.action.chunk_horizon
        return {
            "server": "flowcl",
            "checkpoint": str(self.checkpoint),
            "checkpoint_sha256": self.sha256,
            "trained": self.trained,
            "embodiment": self.spec.to_dict(),
            "stride": self.stride,
            "policy_rate_hz": self.spec.action.control_rate_hz,
            "robot_rate_hz": self.spec.action.control_rate_hz * self.stride,
            "chunk_10hz": h,
            "chunk_30hz": self.stride * (h - 1) + 1,
            "execute_k": self.spec.action.execute_k,
            "execute_ticks": self.spec.action.execute_k * self.stride,
            "euler_steps": self.euler_steps,
            "envelope": self.envelope.to_dict(),
            "envelope_tasks": self.envelope_tasks,
            "start_poses": self.start_poses,
            "languages": self.languages,
            "holdout": self.holdout,
        }

    # ---- requests ---------------------------------------------------------------------

    def _frames(self, request: dict) -> dict[str, np.ndarray]:
        if "policy_images" in request:
            frames = {c: np.asarray(request["policy_images"][c], np.uint8) for c in self.spec.cameras}
        else:
            frames = {c: to_policy_frame(decode_jpeg(request["images"][c])) for c in self.spec.cameras}
        want = (*self.spec.observation.image_size, 3)
        for c, f in frames.items():
            if f.shape != want:
                raise ValueError(f"camera {c}: frame {f.shape}, expected {want}")
        return frames

    def infer(self, request: dict, log: bool = True) -> dict:
        t0 = time.perf_counter()
        prompt = request["prompt"]
        if prompt not in self.languages:
            raise ValueError(f"unknown prompt {prompt!r}; trained instructions: {list(self.languages)}")
        state = np.asarray(request["state"], dtype=np.float32).reshape(-1)
        if state.shape != (self.spec.d_state,) or not np.isfinite(state).all():
            raise ValueError(f"state must be {self.spec.d_state} finite values, got {state}")
        frames = self._frames(request)
        t1 = time.perf_counter()
        batch = {
            "images": {c: torch.from_numpy(f[None]).to(self.device) for c, f in frames.items()},
            "state": torch.from_numpy(self.stats.normalize_state(state[None])).to(self.device),
            "language": [prompt],
        }
        generator = (
            torch.Generator(device="cpu").manual_seed(int(request["seed"]))
            if request.get("seed") is not None else self.generator
        )
        with self.lock, torch.no_grad():
            chunk = self.policy.sample(batch, n_steps=self.euler_steps, generator=generator)
            if self.device.type == "cuda":
                torch.cuda.synchronize()
        a10 = self.stats.denormalize_action(chunk[0].float().cpu().numpy())
        a30 = upsample_chunk(a10, self.stride)
        t2 = time.perf_counter()
        reply = {
            "seq": request.get("seq"),
            "actions_10hz": a10.astype(np.float32),
            "actions_30hz": a30.astype(np.float32),
            "timing": {"decode_ms": 1e3 * (t1 - t0), "policy_ms": 1e3 * (t2 - t1)},
        }
        if log and self.dump_left > 0 and self.log_dir:
            self.dump_left -= 1
            np.savez_compressed(
                self.log_dir / f"request_{int(time.time() * 1e3)}.npz",
                state=state, prompt=prompt, actions_10hz=a10, **{f"img_{c}": f for c, f in frames.items()},
            )
        return reply

    def episode(self, request: dict) -> dict:
        import h5py

        name = request["task"]
        if name not in TASKS:
            raise ValueError(f"unknown task {name!r}; have {list(TASKS)}")
        index = int(request["episode_index"])
        with h5py.File(cache_path(name, self.dataset_dir), "r") as f:
            g = f["episodes"][f"{index:06d}"]
            return {
                "task": name,
                "episode_index": index,
                "held_out": index >= TASKS[name].n_episodes - self.holdout,
                "language": TASKS[name].language,
                "actions": np.asarray(g["actions"], np.float32),
                "qpos": np.asarray(g["qpos"], np.float32),
            }

    def handle(self, request: dict) -> dict:
        kind = request.get("type", "infer")
        if kind == "infer":
            return self.infer(request)
        if kind == "episode":
            return self.episode(request)
        raise ValueError(f"unknown request type {kind!r}")


class RequestLog:
    """Append-only JSONL: one line per request plus connection open/close events."""

    def __init__(self, path: Path | None) -> None:
        self.path = Path(path) if path else None
        if self.path:
            self.path.parent.mkdir(parents=True, exist_ok=True)

    def write(self, **record) -> None:
        if not self.path:
            return
        record["t"] = time.time()
        with open(self.path, "a") as f:
            f.write(json.dumps(record, default=lambda o: np.asarray(o).tolist()) + "\n")


async def serve(service: PolicyService, host: str, port: int, log: RequestLog) -> None:
    from websockets.asyncio.server import serve as ws_serve

    ids = itertools.count()

    async def handler(ws) -> None:
        conn = next(ids)
        peer = str(getattr(ws, "remote_address", ""))
        log.write(event="open", conn=conn, peer=peer, checkpoint=str(service.checkpoint))
        print(f"[serve] connection {conn} from {peer}", flush=True)
        await ws.send(wire.packb(service.metadata()))
        n = 0
        latencies = []
        try:
            async for message in ws:
                t_recv = time.perf_counter()
                try:
                    request = wire.unpackb(message)
                    reply = await asyncio.to_thread(service.handle, request)
                    total = 1e3 * (time.perf_counter() - t_recv)
                    if "timing" in reply:
                        reply["timing"]["server_ms"] = total
                        latencies.append(total)
                        log.write(event="infer", conn=conn, seq=request.get("seq"), prompt=request.get("prompt"),
                                  state=request.get("state"), first_action=reply["actions_30hz"][0],
                                  timing=reply["timing"], t_obs=request.get("t_obs"))
                    else:
                        log.write(event=request.get("type"), conn=conn, task=request.get("task"),
                                  episode_index=request.get("episode_index"))
                    await ws.send(wire.packb(reply))
                    n += 1
                except Exception:
                    tb = traceback.format_exc()
                    log.write(event="error", conn=conn, traceback=tb)
                    print(f"[serve] error on connection {conn}:\n{tb}", flush=True)
                    await ws.send(tb)
        finally:
            summary = {}
            if latencies:
                summary = {"p50_ms": float(np.percentile(latencies, 50)),
                           "p95_ms": float(np.percentile(latencies, 95)),
                           "max_ms": float(np.max(latencies))}
            log.write(event="close", conn=conn, n_requests=n, **summary)
            print(f"[serve] connection {conn} closed after {n} requests {summary}", flush=True)

    async with ws_serve(handler, host, port, compression=None, max_size=None):
        print(f"[serve] listening on ws://{host}:{port} ({service.checkpoint})", flush=True)
        await asyncio.Future()
