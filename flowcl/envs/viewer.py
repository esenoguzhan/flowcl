"""Localhost watcher for qualitative rollout videos.

Stdlib HTTP only — no extra web framework. Binds loopback by default. Serves JPEG
frame sequences written by :mod:`flowcl.envs.video`; recording a missing episode
is a POST that blocks until the rollout finishes.
"""

from __future__ import annotations

import json
import mimetypes
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse

from flowcl.envs.video import (
    VideoArtifact,
    discover_artifacts,
    discover_runs,
    load_video_config,
    record_episode,
    resolve_out_dir,
)
from flowcl.utils.libero_paths import repo_root

_PAGE = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1"/>
<title>flowcl watcher</title>
<style>
  :root { color-scheme: dark; }
  body { font: 14px/1.45 ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
         margin: 0; background: #111; color: #e8e8e8; }
  header { padding: 16px 20px 8px; border-bottom: 1px solid #333; }
  h1 { font-size: 16px; margin: 0 0 4px; }
  .sub { color: #999; }
  main { display: grid; grid-template-columns: 320px 1fr; min-height: calc(100vh - 72px); }
  aside { padding: 16px 20px; border-right: 1px solid #333; }
  section { padding: 16px 20px; }
  label { display: block; margin: 12px 0 4px; color: #bbb; }
  select, input, button { font: inherit; background: #1c1c1c; color: inherit;
                          border: 1px solid #444; padding: 6px 8px; width: 100%;
                          box-sizing: border-box; }
  button { cursor: pointer; margin-top: 10px; background: #2a2a2a; }
  button:disabled { opacity: 0.5; cursor: wait; }
  button.primary { background: #2d4a2d; border-color: #3d6a3d; }
  .row { display: flex; gap: 8px; }
  .row > * { flex: 1; }
  .badge { display: inline-block; padding: 1px 6px; border-radius: 3px; font-size: 12px; }
  .ok { background: #1e4d2b; color: #b6f0c4; }
  .fail { background: #4d1e1e; color: #f0b6b6; }
  .hint { color: #888; font-size: 12px; margin-top: 8px; }
  .videos { list-style: none; padding: 0; margin: 8px 0 0; max-height: 280px; overflow: auto; }
  .videos li { padding: 6px 4px; cursor: pointer; border-bottom: 1px solid #222; }
  .videos li:hover, .videos li.active { background: #222; }
  .stage { display: flex; gap: 16px; flex-wrap: wrap; align-items: flex-start; }
  figure { margin: 0; }
  figcaption { color: #aaa; margin-bottom: 6px; }
  canvas { image-rendering: pixelated; background: #000; display: block; }
  .controls { margin-top: 14px; max-width: 720px; }
  .status { margin-top: 10px; color: #ccc; min-height: 1.4em; }
  .err { color: #f0b6b6; }
  .plots { margin-top: 18px; display: flex; gap: 20px; flex-wrap: wrap; }
  .plots figure canvas { background: #181818; image-rendering: auto; }
  .metrics { margin-top: 10px; color: #bbb; font-size: 12px; }
  .metrics span { margin-right: 14px; }
</style>
</head>
<body>
<header>
  <h1>flowcl watcher</h1>
  <div class="sub">Qualitative replay only — not a Gate / retention eval. Same
  <code>run_id</code> + episode index ⇒ same initial state as the recorded numbers.</div>
</header>
<main>
<aside>
  <label for="run">Checkpoint run</label>
  <select id="run"></select>
  <label for="task">Task</label>
  <select id="task"></select>
  <label for="episode">Episode (0-based)</label>
  <div class="row">
    <input id="episode" type="number" min="0" max="49" value="0"/>
    <button id="record" class="primary" type="button">Record</button>
  </div>
  <div class="hint" id="hints"></div>
  <label>Recorded episodes</label>
  <ul class="videos" id="videos"></ul>
</aside>
<section>
  <div id="meta"></div>
  <div class="stage" id="stage"></div>
  <div class="controls">
    <div class="row">
      <button id="play" type="button">Play</button>
      <button id="pause" type="button">Pause</button>
      <select id="rate">
        <option value="0.5">0.5×</option>
        <option value="1" selected>1× (control rate)</option>
        <option value="2">2×</option>
      </select>
    </div>
    <label for="scrub">Frame</label>
    <input id="scrub" type="range" min="0" max="0" value="0"/>
  </div>
  <div class="metrics" id="metrics"></div>
  <div class="plots" id="plots" hidden>
    <figure>
      <figcaption>EE path (ee_pos_0 vs ee_pos_1) — cursor is the current frame</figcaption>
      <canvas id="path" width="360" height="360"></canvas>
    </figure>
    <figure>
      <figcaption>Executed action (OSC Δ + gripper). Dashed = replan (every k=8)</figcaption>
      <canvas id="actions" width="520" height="360"></canvas>
    </figure>
  </div>
  <div class="status" id="status"></div>
</section>
</main>
<script>
const SCALE = __DISPLAY_SCALE__;
let state = {runs: [], videos: []};
let current = null;
let frames = {};
let trace = null;
let timer = null;
let idx = 0;
const ACTION_COLORS = ["#6ea8fe","#75b798","#ffc107","#fd7e14","#d63384","#6f42c1","#adb5bd"];

function $(id) { return document.getElementById(id); }

async function refresh() {
  const r = await fetch("/api/state");
  state = await r.json();
  const runSel = $("run");
  const prev = runSel.value;
  runSel.innerHTML = state.runs.map(run =>
    `<option value="${run.run_id}">${run.run_id}</option>`).join("");
  if ([...runSel.options].some(o => o.value === prev)) runSel.value = prev;
  fillTasks();
  renderVideoList();
}

function selectedRun() {
  return state.runs.find(r => r.run_id === $("run").value) || null;
}

function fillTasks() {
  const run = selectedRun();
  const taskSel = $("task");
  const prev = taskSel.value;
  const keys = run ? run.task_keys : [];
  taskSel.innerHTML = keys.map(k => `<option value="${k}">${k}</option>`).join("");
  if ([...taskSel.options].some(o => o.value === prev)) taskSel.value = prev;
  updateHints();
}

function updateHints() {
  const run = selectedRun();
  const task = $("task").value;
  const box = $("hints");
  if (!run || !task || !run.eval_successes[task]) {
    box.textContent = "No eval.json on this run — episode outcomes unknown until you record.";
    return;
  }
  const succ = run.eval_successes[task];
  const fails = succ.map((ok, i) => ok ? null : i).filter(x => x !== null);
  box.innerHTML = `${succ.filter(Boolean).length}/${succ.length} successes in stored eval.` +
    (fails.length ? ` Failures: ${fails.join(", ")}.` : " No failures.");
}

function videoKey(v) {
  return `${v.run_id}::${v.task_key}::${v.episode_idx}`;
}

function renderVideoList() {
  const task = $("task").value;
  const run = $("run").value;
  const items = state.videos.filter(v =>
    (!run || v.run_id === run) && (!task || v.task_key === task));
  $("videos").innerHTML = items.map(v => {
    const cls = v.success ? "ok" : "fail";
    const label = v.success ? "success" : "fail";
    return `<li data-key="${videoKey(v)}">ep ${v.episode_idx}
      <span class="badge ${cls}">${label}</span>
      · ${v.n_frames} frames</li>`;
  }).join("") || "<li>None yet. Record an episode.</li>";
  for (const li of $("videos").querySelectorAll("li[data-key]")) {
    li.onclick = () => openVideo(li.dataset.key);
  }
}

async function openVideo(key) {
  const v = state.videos.find(x => videoKey(x) === key);
  if (!v) return;
  current = v;
  $("episode").value = v.episode_idx;
  for (const li of $("videos").querySelectorAll("li")) {
    li.classList.toggle("active", li.dataset.key === key);
  }
  $("meta").innerHTML = `<strong>${v.task_key}</strong> · ep ${v.episode_idx} ·
    <span class="badge ${v.success ? "ok" : "fail"}">${v.success ? "success" : "fail"}</span>
    · ${v.n_steps} env steps · ${v.n_frames} frames · seed ${v.seed} · ${v.fps} Hz`;
  $("scrub").max = Math.max(0, v.n_frames - 1);
  $("scrub").value = 0;
  idx = 0;
  $("stage").innerHTML = "";
  frames = {};
  for (const cam of v.cameras) {
    const fig = document.createElement("figure");
    fig.innerHTML = `<figcaption>${cam}</figcaption><canvas data-cam="${cam}"></canvas>`;
    $("stage").appendChild(fig);
    const urls = [];
    for (let i = 0; i < v.n_frames; i++) {
      const name = String(i).padStart(5, "0") + ".jpg";
      urls.push(`/media/${encodeURI(v.rel_dir + "/" + cam + "/" + name)}`);
    }
    frames[cam] = urls;
    const img = new Image();
    img.onload = () => {
      const c = fig.querySelector("canvas");
      c.width = img.naturalWidth;
      c.height = img.naturalHeight;
      c.style.width = (img.naturalWidth * SCALE) + "px";
      c.style.height = (img.naturalHeight * SCALE) + "px";
      c.getContext("2d").drawImage(img, 0, 0);
    };
    img.src = urls[0];
  }
  $("status").textContent = "";
  trace = null;
  $("plots").hidden = true;
  $("metrics").textContent = "";
  showFrame(0);
  if (v.has_trace) loadTrace(v);
}

async function loadTrace(v) {
  const r = await fetch(`/media/${encodeURI(v.rel_dir + "/trace.json")}`);
  if (!r.ok) {
    $("metrics").textContent = "No trace on this recording (re-record to get actions / EE path).";
    return;
  }
  trace = await r.json();
  $("plots").hidden = false;
  const s = trace.smoothness || {};
  const jerk = s.rms_ee_jerk == null ? "n/a" : Number(s.rms_ee_jerk).toExponential(2);
  const step = s.max_ee_step == null ? "n/a" : Number(s.max_ee_step).toFixed(4);
  const clip = s.action_clip_fraction == null ? "n/a" : (100 * s.action_clip_fraction).toFixed(1) + "%";
  $("metrics").innerHTML =
    `<span>max |Δee_pos| ${step}</span>` +
    `<span>RMS ee jerk ${jerk}</span>` +
    `<span>action at clip rail ${clip}</span>` +
    `<span>replans ${s.n_replans ?? "?"}</span>` +
    (s.ee_pos_columns && s.ee_pos_columns.length
      ? ""
      : `<span class="err">no ee_pos columns in spec names</span>`);
  drawPlots();
}

function showFrame(i) {
  if (!current) return;
  idx = i;
  $("scrub").value = i;
  for (const cam of current.cameras) {
    const canvas = document.querySelector(`canvas[data-cam="${cam}"]`);
    if (!canvas || !frames[cam]) continue;
    const img = new Image();
    img.onload = () => canvas.getContext("2d").drawImage(img, 0, 0);
    img.src = frames[cam][i];
  }
  if (trace) drawPlots();
}

function namedCols(names, prefix) {
  return names.map((n, i) => (n === prefix || n.startsWith(prefix + "_")) ? i : -1)
              .filter(i => i >= 0);
}

function drawPlots() {
  if (!trace) return;
  drawPath();
  drawActions();
}

function drawPath() {
  const canvas = $("path");
  const ctx = canvas.getContext("2d");
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  const cols = namedCols(trace.state_names, "ee_pos");
  if (cols.length < 2) {
    ctx.fillStyle = "#888";
    ctx.fillText("need ee_pos_0 and ee_pos_1", 12, 20);
    return;
  }
  const xs = trace.states.map(s => s[cols[0]]);
  const ys = trace.states.map(s => s[cols[1]]);
  const pad = 24;
  const minX = Math.min(...xs), maxX = Math.max(...xs);
  const minY = Math.min(...ys), maxY = Math.max(...ys);
  const spanX = Math.max(maxX - minX, 1e-6);
  const spanY = Math.max(maxY - minY, 1e-6);
  const span = Math.max(spanX, spanY);
  const cx = (minX + maxX) / 2, cy = (minY + maxY) / 2;
  const sx = (canvas.width - 2 * pad) / span;
  const sy = (canvas.height - 2 * pad) / span;
  const mapX = x => pad + (x - (cx - span / 2)) * sx;
  const mapY = y => canvas.height - (pad + (y - (cy - span / 2)) * sy);
  ctx.strokeStyle = "#555";
  ctx.beginPath();
  ctx.moveTo(mapX(xs[0]), mapY(ys[0]));
  for (let i = 1; i < xs.length; i++) ctx.lineTo(mapX(xs[i]), mapY(ys[i]));
  ctx.stroke();
  const k = Math.max(0, Math.min(idx, xs.length - 1));
  ctx.fillStyle = "#6ea8fe";
  ctx.beginPath();
  ctx.arc(mapX(xs[k]), mapY(ys[k]), 4, 0, Math.PI * 2);
  ctx.fill();
  ctx.fillStyle = "#888";
  ctx.fillText("start", mapX(xs[0]) + 6, mapY(ys[0]) - 6);
}

function drawActions() {
  const canvas = $("actions");
  const ctx = canvas.getContext("2d");
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  const acts = trace.actions;
  if (!acts.length) {
    ctx.fillStyle = "#888";
    ctx.fillText("no executed actions", 12, 20);
    return;
  }
  const dim = acts[0].length;
  const padL = 36, padR = 10, padT = 10, padB = 22;
  const w = canvas.width - padL - padR;
  const h = canvas.height - padT - padB;
  const n = acts.length;
  const xAt = t => padL + (n === 1 ? w / 2 : (t / (n - 1)) * w);
  ctx.strokeStyle = "#333";
  ctx.beginPath();
  ctx.moveTo(padL, padT + h / 2);
  ctx.lineTo(padL + w, padT + h / 2);
  ctx.stroke();
  for (let t = 0; t < n; t++) {
    if (!trace.replan[t]) continue;
    ctx.strokeStyle = "#664";
    ctx.setLineDash([3, 3]);
    ctx.beginPath();
    ctx.moveTo(xAt(t), padT);
    ctx.lineTo(xAt(t), padT + h);
    ctx.stroke();
    ctx.setLineDash([]);
  }
  for (let d = 0; d < dim; d++) {
    ctx.strokeStyle = ACTION_COLORS[d % ACTION_COLORS.length];
    ctx.beginPath();
    for (let t = 0; t < n; t++) {
      const y = padT + h * (0.5 - acts[t][d] / 2);
      if (t === 0) ctx.moveTo(xAt(t), y);
      else ctx.lineTo(xAt(t), y);
    }
    ctx.stroke();
  }
  const actionIdx = Math.max(0, Math.min(idx - 1, n - 1));
  if (idx > 0) {
    ctx.strokeStyle = "#eee";
    ctx.beginPath();
    ctx.moveTo(xAt(actionIdx), padT);
    ctx.lineTo(xAt(actionIdx), padT + h);
    ctx.stroke();
  }
  ctx.fillStyle = "#888";
  ctx.fillText("t=0", padL, canvas.height - 6);
  ctx.fillText("t=" + (n - 1), padL + w - 28, canvas.height - 6);
  const names = trace.action_names || [];
  names.forEach((name, d) => {
    ctx.fillStyle = ACTION_COLORS[d % ACTION_COLORS.length];
    ctx.fillText(name.replace("osc_pose_delta", "d"), 4, 12 + d * 12);
  });
}

function stopTimer() {
  if (timer !== null) { clearInterval(timer); timer = null; }
}

function play() {
  if (!current) return;
  stopTimer();
  const fps = current.fps * parseFloat($("rate").value);
  const dt = 1000 / fps;
  timer = setInterval(() => {
    const next = idx + 1;
    if (next >= current.n_frames) { stopTimer(); return; }
    showFrame(next);
  }, dt);
}

async function record() {
  const run = selectedRun();
  if (!run) { $("status").innerHTML = '<span class="err">No checkpoint run found under results/.</span>'; return; }
  const body = {
    checkpoint: run.checkpoint,
    task_key: $("task").value,
    episode_idx: parseInt($("episode").value, 10),
    run_id: run.run_id,
  };
  $("record").disabled = true;
  $("status").textContent = "Recording… this is a real rollout (EGL + CUDA). Wait.";
  try {
    const r = await fetch("/api/record", {
      method: "POST",
      headers: {"Content-Type": "application/json"},
      body: JSON.stringify(body),
    });
    const payload = await r.json();
    if (!r.ok) throw new Error(payload.error || r.statusText);
    await refresh();
    openVideo(videoKey(payload));
    $("status").textContent = payload.success ? "Recorded a success." : "Recorded a failure (timeout or drop).";
  } catch (err) {
    $("status").innerHTML = `<span class="err">${err}</span>`;
  } finally {
    $("record").disabled = false;
  }
}

$("run").onchange = () => { fillTasks(); renderVideoList(); };
$("task").onchange = () => { updateHints(); renderVideoList(); };
$("record").onclick = record;
$("play").onclick = play;
$("pause").onclick = stopTimer;
$("rate").onchange = () => { if (timer !== null) play(); };
$("scrub").oninput = (e) => { stopTimer(); showFrame(parseInt(e.target.value, 10)); };
refresh();
</script>
</body>
</html>
"""


class _WatchState:
    def __init__(
        self,
        results_root: Path,
        out_dir: Path,
        video_cfg: dict,
        device: str,
    ) -> None:
        self.results_root = results_root
        self.out_dir = out_dir
        self.video_cfg = video_cfg
        self.device = device
        self.record_lock = threading.Lock()


def _artifact_payload(art: VideoArtifact, out_dir: Path) -> dict:
    rel = art.directory.resolve().relative_to(out_dir.resolve()).as_posix()
    return {
        "run_id": art.meta.get("run_id"),
        "task_key": art.meta["task_key"],
        "episode_idx": art.meta["episode_idx"],
        "success": art.meta["success"],
        "n_steps": art.meta["n_steps"],
        "n_frames": art.meta["n_frames"],
        "seed": art.meta["seed"],
        "fps": art.meta["fps"],
        "cameras": list(art.cameras),
        "rel_dir": rel,
        "directory": str(art.directory),
        "has_trace": bool(
            art.meta.get("has_trace") or (art.directory / "trace.json").is_file()
        ),
    }


class WatchHandler(BaseHTTPRequestHandler):
    """HTTP handler. ``server.watch`` holds the :class:`_WatchState`."""

    server_version = "flowcl-watch/0.1"

    def log_message(self, fmt: str, *args) -> None:
        print(f"[watch] {self.address_string()} {fmt % args}", flush=True)

    def _state(self) -> _WatchState:
        return self.server.watch  # type: ignore[attr-defined]

    def _send(self, code: int, body: bytes, content_type: str) -> None:
        self.send_response(code)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-cache")
        self.end_headers()
        self.wfile.write(body)

    def _send_json(self, code: int, payload: dict | list) -> None:
        raw = json.dumps(payload).encode("utf-8")
        self._send(code, raw, "application/json")

    def do_GET(self) -> None:  # noqa: N802 — stdlib signature
        parsed = urlparse(self.path)
        if parsed.path == "/":
            page = _PAGE.replace(
                "__DISPLAY_SCALE__", str(int(self._state().video_cfg["display_scale"]))
            )
            self._send(200, page.encode("utf-8"), "text/html; charset=utf-8")
            return
        if parsed.path == "/api/state":
            st = self._state()
            self._send_json(
                200,
                {
                    "runs": [r.as_dict() for r in discover_runs(st.results_root)],
                    "videos": [
                        _artifact_payload(a, st.out_dir)
                        for a in discover_artifacts(st.out_dir)
                    ],
                },
            )
            return
        if parsed.path.startswith("/media/"):
            self._serve_media(parsed.path[len("/media/") :])
            return
        self._send_json(404, {"error": f"unknown path {parsed.path}"})

    def do_POST(self) -> None:  # noqa: N802
        parsed = urlparse(self.path)
        if parsed.path != "/api/record":
            self._send_json(404, {"error": f"unknown path {parsed.path}"})
            return
        length = int(self.headers.get("Content-Length", "0"))
        try:
            body = json.loads(self.rfile.read(length) or b"{}")
        except json.JSONDecodeError as exc:
            self._send_json(400, {"error": f"invalid JSON: {exc}"})
            return
        required = ("checkpoint", "episode_idx")
        missing = [k for k in required if k not in body]
        if missing:
            self._send_json(400, {"error": f"missing fields: {missing}"})
            return
        st = self._state()
        checkpoint = Path(body["checkpoint"])
        try:
            checkpoint = checkpoint.resolve()
            results = st.results_root.resolve()
            if results not in checkpoint.parents:
                raise ValueError(f"checkpoint {checkpoint} is outside {results}")
            if not st.record_lock.acquire(blocking=False):
                self._send_json(409, {"error": "a recording is already running"})
                return
            try:
                art = record_episode(
                    checkpoint,
                    int(body["episode_idx"]),
                    task_key=body.get("task_key"),
                    run_id=body.get("run_id"),
                    device=st.device,
                    out_dir=st.out_dir,
                    video_cfg=st.video_cfg,
                )
            finally:
                st.record_lock.release()
        except Exception as exc:  # noqa: BLE001 — surface the real failure in the UI
            self._send_json(500, {"error": f"{type(exc).__name__}: {exc}"})
            return
        self._send_json(200, _artifact_payload(art, st.out_dir))

    def _serve_media(self, rel: str) -> None:
        st = self._state()
        # The browser encodes spaces; episode dirs should not have them, but
        # refuse ``..`` regardless.
        raw = Path(rel)
        if ".." in raw.parts:
            self._send_json(400, {"error": "path traversal"})
            return
        path = (st.out_dir / raw).resolve()
        try:
            if st.out_dir.resolve() not in path.parents and path != st.out_dir.resolve():
                raise ValueError("outside video root")
        except ValueError as exc:
            self._send_json(400, {"error": str(exc)})
            return
        if not path.is_file():
            self._send_json(404, {"error": f"missing {rel}"})
            return
        ctype = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
        self._send(200, path.read_bytes(), ctype)


def serve_viewer(
    *,
    host: str | None = None,
    port: int | None = None,
    results_root: Path | None = None,
    out_dir: Path | None = None,
    device: str = "cuda",
    video_cfg: dict | None = None,
) -> None:
    """Block serving the watcher. Ctrl-C to stop."""
    cfg = video_cfg if video_cfg is not None else load_video_config()
    results = (results_root or (repo_root() / "results")).resolve()
    dest = resolve_out_dir(out_dir, cfg)
    dest.mkdir(parents=True, exist_ok=True)
    bind_host = host if host is not None else str(cfg["host"])
    bind_port = int(port if port is not None else cfg["port"])

    state = _WatchState(
        results_root=results,
        out_dir=dest,
        video_cfg=cfg,
        device=device,
    )
    handler = WatchHandler
    server = ThreadingHTTPServer((bind_host, bind_port), handler)
    server.watch = state  # type: ignore[attr-defined]
    n_runs = len(discover_runs(results))
    print(
        f"[flowcl] watcher at http://{bind_host}:{bind_port}/  "
        f"({n_runs} checkpoint run(s) under {results}, videos -> {dest})",
        flush=True,
    )
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n[flowcl] watcher stopped", flush=True)
    finally:
        server.server_close()
