"""HTTP policy server for the AgileX Piper, driven by PiperStudio (thesis plan F1).

One process serves one checkpoint to one robot client. The contract is
``docs/piper_deploy_spec.md`` §2: ``GET /metadata``, ``POST /infer``, and a status page
at ``GET /`` (the same data at ``/status.json``). It uses plain HTTP and JSON, so the
robot side needs only the standard library. The entry point is
``scripts/hw/serve_piper.py``.

The server owns everything that must match training exactly, so the robot side does no
image processing at all:

* frames arrive as the cameras' own JPEG bytes (640x480). They are decoded the way
  PiperStudio's recorder decodes them (``cv2.imdecode``, then BGR to RGB), then reduced
  by :func:`flowcl.data.images.policy_frame` with the checkpoint's masks, the function
  the training cache used;
* the state is normalised and the chunk denormalised with the checkpoint's frozen stats;
* the instruction must be the registry task's, verbatim.

The metadata carries what the robot side needs to execute chunks safely, all taken from
the checkpoint's training episodes in the cache: the start pose (the median first
command), the action envelope (min and max, widened), and the held-out episodes that
serve as the rollouts' initial conditions.
"""

from __future__ import annotations

import base64
import binascii
import collections
import hashlib
import html
import json
import threading
import time
import traceback
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import numpy as np
import torch

from flowcl.data.images import policy_frame
from flowcl.data.piper_adapter import CAMERA_KEYS, RECORDED_HW, verify_cache
from flowcl.data.piper_tasks import SUITE, PiperTask, cache_path, get_task
from flowcl.train.checkpoint import load_checkpoint

PROTOCOL = 1
EMBODIMENT = "agilex_piper"
DEFAULT_PORT = 8765  # PiperStudio itself serves on 8000
GRIPPER = 6
COMPONENTS = (("joints", 6), ("gripper", 1))
ENVELOPE_MARGIN_RAD = 0.15
ENVELOPE_MARGIN_GRIPPER_M = 0.005
MAX_BODY_BYTES = 16 << 20


class RequestError(ValueError):
    """A malformed request: answered with HTTP 400, and the server keeps serving."""


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def task_of(task_key: str | None) -> PiperTask:
    """The registry task a checkpoint trained on, from its ``piper/<name>`` key."""
    suite, _, name = (task_key or "").partition("/")
    if suite != SUITE or not name:
        raise ValueError(f"checkpoint task key {task_key!r} is not a {SUITE} task")
    return get_task(name)


def training_summary(
    task: PiperTask,
    n_demos: int | None = None,
    dataset_dir: Path | None = None,
    margin_rad: float = ENVELOPE_MARGIN_RAD,
    margin_gripper_m: float = ENVELOPE_MARGIN_GRIPPER_M,
) -> dict:
    """Start pose, action envelope and held-out episodes from a task's cache.

    The training episodes are cache episodes ``0..n-1``, with ``n = n_demos`` or the
    registry's ``n_train`` (what training took for ``n_demos=None``). The held-out
    episodes are the registry's last ``holdout``, given as their LeRobot folder and raw
    episode index, which is how PiperStudio names them. Only the actions are read; the
    frames stay on disk.
    """
    import h5py

    n = task.n_train if n_demos is None else int(n_demos)
    path = cache_path(task.name, dataset_dir)
    verify_cache(path, task, n_demos=n)
    with h5py.File(path, "r") as f:
        groups = [f["episodes"][f"{i:06d}"] for i in range(task.n_episodes)]
        actions = [np.asarray(g["action"], np.float64) for g in groups[:n]]
        held_out = [{"dataset": str(g.attrs["source"]), "episode_index": int(g.attrs["raw_episode_index"])}
                    for g in groups[task.n_train:]]
    stacked = np.concatenate(actions)
    margin = np.full(stacked.shape[1], float(margin_rad))
    margin[GRIPPER] = float(margin_gripper_m)
    return {
        "n_demos": n,
        "start_pose": np.median(np.stack([a[0] for a in actions]), axis=0).round(4).tolist(),
        "envelope": {"lo": (stacked.min(axis=0) - margin).round(4).tolist(),
                     "hi": (stacked.max(axis=0) + margin).round(4).tolist(),
                     "margin_rad": float(margin_rad), "margin_gripper_m": float(margin_gripper_m)},
        "held_out": held_out,
    }


def decode_frame(data: str, camera: str) -> np.ndarray:
    """A base64 JPEG from the robot -> the 480x640 RGB frame the recorder stored."""
    import cv2

    try:
        raw = base64.b64decode(data, validate=True)
    except (binascii.Error, TypeError, ValueError) as e:
        raise RequestError(f"camera {camera}: not base64 ({e})") from None
    bgr = cv2.imdecode(np.frombuffer(raw, dtype=np.uint8), cv2.IMREAD_COLOR)
    if bgr is None:
        raise RequestError(f"camera {camera}: the bytes are not a JPEG image")
    if bgr.shape[:2] != RECORDED_HW:
        raise RequestError(f"camera {camera}: frame is {bgr.shape[1]}x{bgr.shape[0]}, the policy "
                           f"was trained on {RECORDED_HW[1]}x{RECORDED_HW[0]}")
    return np.ascontiguousarray(bgr[:, :, ::-1])


def encode_png(rgb: np.ndarray) -> str:
    import cv2

    ok, buf = cv2.imencode(".png", np.ascontiguousarray(rgb[:, :, ::-1]))
    if not ok:
        raise RuntimeError("cv2.imencode failed")
    return base64.b64encode(buf.tobytes()).decode("ascii")


class PiperPolicyService:
    """A loaded Piper checkpoint, answering metadata and inference requests.

    Inference is serialised by a lock, so concurrent requests queue rather than share
    the GPU. Without a ``seed`` in the request, the noise comes from one generator
    seeded at start, so a server restart replays the same noise sequence.
    """

    def __init__(
        self,
        checkpoint: Path,
        device: str = "cuda",
        n_demos: int | None = None,
        dataset_dir: Path | None = None,
        seed: int = 0,
        log_path: Path | None = None,
    ) -> None:
        self.checkpoint = Path(checkpoint)
        self.device = torch.device(device)
        loaded = load_checkpoint(self.checkpoint, device=self.device)
        self.policy, self.spec, self.stats, self.payload = loaded.policy, loaded.spec, loaded.stats, loaded.payload
        if self.spec.name != EMBODIMENT:
            raise ValueError(f"{checkpoint} is a {self.spec.name!r} checkpoint, not {EMBODIMENT}")
        if tuple(self.spec.action.component_keys) != COMPONENTS:
            raise ValueError(f"{checkpoint}: action components {self.spec.action.component_keys}, "
                             f"expected {COMPONENTS}")
        self.task = task_of(loaded.task_key)
        self.summary = training_summary(self.task, n_demos, dataset_dir)
        self.sha256 = _sha256(self.checkpoint)
        self.seed = int(seed)
        self.generator = torch.Generator().manual_seed(self.seed)
        self.lock = threading.Lock()          # one inference at a time
        self._status_lock = threading.Lock()  # bookkeeping never waits for the GPU
        self.started = time.time()
        self.n_requests = 0
        self.recent: collections.deque = collections.deque(maxlen=30)
        self.errors: collections.deque = collections.deque(maxlen=5)
        self.log_path = Path(log_path) if log_path else None
        if self.log_path:
            self.log_path.parent.mkdir(parents=True, exist_ok=True)
        self._warm_up()
        self.log(kind="start", metadata=self.metadata())

    def _warm_up(self) -> None:
        h, w = self.spec.observation.image_size
        frames = {c: np.zeros((h, w, 3), np.uint8) for c in self.spec.cameras}
        state = np.asarray(self.summary["start_pose"], np.float32)
        for _ in range(2):
            self.sample(frames, state, torch.Generator().manual_seed(0))

    # ---- what the robot side reads once ----------------------------------------------

    def metadata(self) -> dict:
        obs, act = self.spec.observation, self.spec.action
        return {
            "protocol": PROTOCOL,
            "checkpoint": str(self.checkpoint),
            "checkpoint_sha256": self.sha256,
            "run_id": self.payload.get("run_id"),
            "task_key": self.payload.get("task_key"),
            "instruction": self.task.language,
            "cameras": {c: CAMERA_KEYS[c] for c in self.spec.cameras},
            "frame_hw": list(RECORDED_HW),
            "image_masks": self.spec.to_dict()["observation"].get("image_masks", {}),
            "policy_image_hw": list(obs.image_size),
            "state_names": [f"joint_{i}" for i in range(1, 7)] + ["gripper"],
            "units": {"joints": "rad", "gripper": "m"},
            "control_rate_hz": act.control_rate_hz,
            "chunk_horizon": act.chunk_horizon,
            "execute_k": act.execute_k,
            "start_pose": self.summary["start_pose"],
            "envelope": self.summary["envelope"],
            "held_out": self.summary["held_out"],
            "n_demos_trained": self.summary["n_demos"],
            "euler_steps": self.policy.euler_steps,
            "server_seed": self.seed,
        }

    # ---- inference -------------------------------------------------------------------

    def policy_inputs(self, images: dict) -> dict[str, np.ndarray]:
        """Base64 JPEGs per camera -> the policy's 128x128 RGB inputs (masked)."""
        if not isinstance(images, dict):
            raise RequestError("images must be an object {camera: base64 JPEG}")
        missing = [c for c in self.spec.cameras if c not in images]
        if missing:
            raise RequestError(f"missing cameras {missing}; the policy needs {list(self.spec.cameras)}")
        return {c: policy_frame(decode_frame(images[c], c), c, self.spec) for c in self.spec.cameras}

    def sample(self, frames: dict[str, np.ndarray], state: np.ndarray, generator: torch.Generator) -> np.ndarray:
        """One chunk, ``(H, 7)`` in rad / m, from policy inputs and the measured state.

        The batch is assembled as :class:`flowcl.data.dataset.ChunkedActionDataset`
        assembles a training sample: uint8 frames, the state normalised by the frozen
        stats, the instruction as a string.
        """
        batch = {
            "images": {c: torch.from_numpy(np.ascontiguousarray(frames[c])[None]).to(self.device)
                       for c in self.spec.cameras},
            "state": torch.from_numpy(self.stats.normalize_state(state[None])).to(self.device),
            "language": [self.task.language],
        }
        with self.lock, torch.no_grad():
            chunk = self.policy.sample(batch, generator=generator)
            if self.device.type == "cuda":
                torch.cuda.synchronize(self.device)
        return self.stats.denormalize_action(chunk[0].float().cpu().numpy())

    def infer(self, request: dict) -> dict:
        t0 = time.perf_counter()
        if not isinstance(request, dict):
            raise RequestError("the request body must be a JSON object")
        if request.get("instruction") != self.task.language:
            raise RequestError(f"instruction {request.get('instruction')!r} is not the trained one, "
                               f"{self.task.language!r}")
        try:
            state = np.asarray(request.get("state"), dtype=np.float32).reshape(-1)
        except (TypeError, ValueError):
            raise RequestError("state must be a list of numbers") from None
        if state.shape != (self.spec.d_state,) or not np.isfinite(state).all():
            raise RequestError(f"state must be {self.spec.d_state} finite numbers, got {request.get('state')}")
        seed = request.get("seed")
        if seed is not None and (isinstance(seed, bool) or not isinstance(seed, int) or not 0 <= seed < 2**63):
            raise RequestError(f"seed must be a non-negative integer, got {seed!r}")
        frames = self.policy_inputs(request.get("images"))
        t1 = time.perf_counter()
        generator = torch.Generator().manual_seed(seed) if seed is not None else self.generator
        actions = self.sample(frames, state, generator)
        t2 = time.perf_counter()
        if not np.isfinite(actions).all():
            raise RuntimeError("the policy returned non-finite actions")
        reply = {
            "seq": request.get("seq"),
            "actions": actions.astype(np.float64).round(6).tolist(),
            "timing_ms": {"decode": round(1e3 * (t1 - t0), 2), "policy": round(1e3 * (t2 - t1), 2),
                          "total": round(1e3 * (time.perf_counter() - t0), 2)},
        }
        if request.get("return_inputs"):
            reply["inputs"] = {c: encode_png(f) for c, f in frames.items()}
        return reply

    # ---- bookkeeping -------------------------------------------------------------------

    def log(self, **record) -> None:
        if not self.log_path:
            return
        record["t"] = time.time()
        with open(self.log_path, "a") as f:
            f.write(json.dumps(record) + "\n")

    def record(self, request: dict | None, reply: dict | None, error: str | None, ms: float, peer: str) -> None:
        request = request if isinstance(request, dict) else {}
        seq = request.get("seq")
        with self._status_lock:
            self.n_requests += 1
            self.recent.append({"time": time.strftime("%H:%M:%S"), "seq": seq, "ms": round(ms, 1),
                                "error": error.splitlines()[-1] if error else ""})
            if error:
                self.errors.append({"time": time.strftime("%Y-%m-%d %H:%M:%S"), "peer": peer,
                                    "error": error[-2000:]})
        self.log(kind="infer", peer=peer, seq=seq, seed=request.get("seed"), state=request.get("state"),
                 ms=round(ms, 2), timing_ms=(reply or {}).get("timing_ms"),
                 actions=(reply or {}).get("actions"), error=error)

    def status(self) -> dict:
        with self._status_lock:
            recent = list(self.recent)[::-1]
            errors = list(self.errors)[::-1]
            n = self.n_requests
        ok = [r["ms"] for r in recent if not r["error"]]
        return {
            "now": time.strftime("%Y-%m-%d %H:%M:%S"),
            "uptime_s": round(time.time() - self.started),
            "checkpoint": str(self.checkpoint),
            "checkpoint_sha256": self.sha256[:16],
            "run_id": self.payload.get("run_id"),
            "instruction": self.task.language,
            "requests_total": n,
            "recent_ms_p50": float(np.median(ok)) if ok else None,
            "recent_ms_max": max(ok) if ok else None,
            "recent": recent,
            "errors": errors,
        }


def status_html(s: dict) -> str:
    esc = html.escape
    rows = "".join(
        f"<tr><td>{esc(r['time'])}</td><td>{esc(str(r['seq']))}</td><td>{r['ms']}</td>"
        f"<td>{esc(r['error'])}</td></tr>" for r in s["recent"]
    ) or "<tr><td colspan=4>no requests yet</td></tr>"
    errors = "".join(f"<pre>{esc(e['time'])} from {esc(e['peer'])}\n{esc(e['error'])}</pre>"
                     for e in s["errors"]) or "<p>none</p>"
    latency = ("no successful requests yet" if s["recent_ms_p50"] is None
               else f"p50 {s['recent_ms_p50']:.1f} ms, max {s['recent_ms_max']:.1f} ms over the recent ones")
    return f"""<!doctype html><html><head><meta charset="utf-8"><meta http-equiv="refresh" content="5">
<meta name="viewport" content="width=device-width, initial-scale=1"><title>Piper policy server</title>
<style>body{{font-family:system-ui,sans-serif;margin:16px;max-width:960px;background:#fff;color:#111}}
table{{border-collapse:collapse;width:100%}}td,th{{border-bottom:1px solid #ddd;padding:3px 6px;text-align:left;font-size:13px}}
code,pre{{background:#f4f4f4;padding:2px 4px;white-space:pre-wrap}}</style></head><body>
<h2>Piper policy server</h2>
<p>Updated {esc(s['now'])} (refreshes every 5 s); up {s['uptime_s'] // 60} min.</p>
<p><b>Checkpoint:</b> <code>{esc(s['checkpoint'])}</code> (sha256 {esc(s['checkpoint_sha256'])}…)<br>
<b>Run:</b> <code>{esc(str(s['run_id']))}</code><br>
<b>Instruction:</b> <i>{esc(s['instruction'])}</i><br>
<b>Requests:</b> {s['requests_total']}; {latency}</p>
<h3>Recent requests</h3><table><tr><th>time</th><th>seq</th><th>ms</th><th>error</th></tr>{rows}</table>
<h3>Recent errors</h3>{errors}
<p>The same as JSON: <a href="/status.json">/status.json</a>; the contract: <a href="/metadata">/metadata</a></p>
</body></html>"""


def make_handler(service: PiperPolicyService) -> type[BaseHTTPRequestHandler]:
    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"  # keep-alive, so a client can reuse one connection
        server_version = "flowcl-piper/1"

        def _send(self, code: int, body: bytes, content_type: str) -> None:
            self.send_response(code)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def _json(self, code: int, payload: dict) -> None:
            self._send(code, json.dumps(payload).encode(), "application/json")

        def do_GET(self) -> None:  # noqa: N802 (http.server's naming)
            path = self.path.split("?", 1)[0]
            if path == "/metadata":
                self._json(200, service.metadata())
            elif path == "/status.json":
                self._json(200, service.status())
            elif path == "/":
                self._send(200, status_html(service.status()).encode(), "text/html; charset=utf-8")
            else:
                self._json(404, {"error": f"no such path {path}"})

        def do_POST(self) -> None:  # noqa: N802
            path = self.path.split("?", 1)[0]
            if path != "/infer":
                self._json(404, {"error": f"no such path {path}"})
                return
            t0 = time.perf_counter()
            request = reply = error = None
            code = 200
            try:
                length = int(self.headers.get("Content-Length") or 0)
                if not 0 < length <= MAX_BODY_BYTES:
                    raise RequestError(f"body length {length} outside 1..{MAX_BODY_BYTES} bytes")
                try:
                    request = json.loads(self.rfile.read(length))
                except (json.JSONDecodeError, UnicodeDecodeError) as e:
                    raise RequestError(f"the body is not JSON ({e})") from None
                reply = service.infer(request)
            except RequestError as e:
                code, error = 400, str(e)
            except Exception:
                code, error = 500, traceback.format_exc()
            service.record(request, reply, error, 1e3 * (time.perf_counter() - t0), self.client_address[0])
            if error is None:
                self._json(200, reply)
            else:
                self._json(code, {"seq": request.get("seq") if isinstance(request, dict) else None,
                                  "error": error.strip().splitlines()[-1] if code == 500 else error})

        def log_message(self, format, *args) -> None:  # noqa: A002
            pass  # every request goes to the JSONL log instead

    return Handler


def make_server(service: PiperPolicyService, host: str = "127.0.0.1", port: int = DEFAULT_PORT) -> ThreadingHTTPServer:
    server = ThreadingHTTPServer((host, port), make_handler(service))
    server.daemon_threads = True
    return server
