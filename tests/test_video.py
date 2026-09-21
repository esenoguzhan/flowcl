"""Qualitative video artifacts: paths, writers, discovery. No env / GPU."""

from __future__ import annotations

import json
import threading
from http.client import HTTPConnection
from http.server import ThreadingHTTPServer
from pathlib import Path

import numpy as np
import pytest
from omegaconf import OmegaConf

from flowcl.envs.libero_env import RolloutResult, RolloutTrace
from flowcl.envs.video import (
    assert_under,
    discover_artifacts,
    discover_runs,
    episode_dir,
    load_artifact,
    load_video_config,
    persist_rollout,
    persist_trace,
    resolve_out_dir,
    smoothness_summary,
    task_slug,
    write_jpeg_sequence,
    write_mp4,
)
from flowcl.utils.libero_paths import repo_root


def _rgb(n: int = 4, h: int = 8, w: int = 10, seed: int = 0) -> list[np.ndarray]:
    rng = np.random.default_rng(seed)
    return [rng.integers(0, 255, size=(h, w, 3), dtype=np.uint8) for _ in range(n)]


def test_shipped_video_config_is_loopback_and_not_the_eval_protocol():
    cfg = load_video_config()
    assert cfg["host"] == "127.0.0.1"
    assert cfg["port"] == 8765
    assert 1 <= cfg["jpeg_quality"] <= 100
    assert cfg["clip_abs"] == 1.0
    assert cfg["clip_rail_tol"] > 0
    eval_payload = OmegaConf.to_container(
        OmegaConf.load(repo_root() / "configs" / "eval" / "libero_eval.yaml"),
        resolve=True,
    )
    assert eval_payload["record_video"] is False


def test_video_config_rejects_unknown_keys(tmp_path):
    path = tmp_path / "video.yaml"
    path.write_text(
        "out_dir: x\nhost: 127.0.0.1\nport: 1\njpeg_quality: 80\n"
        "display_scale: 2\nmp4_codec: mp4v\nclip_abs: 1.0\nclip_rail_tol: 1e-6\n"
        "extra_knob: 1\n"
    )
    with pytest.raises(ValueError, match="unknown keys"):
        load_video_config(path)


def test_task_slug_and_episode_dir(tmp_path):
    key = "libero_object/pick_up_the_milk_and_place_it_in_the_basket"
    assert task_slug(key) == "libero_object__pick_up_the_milk_and_place_it_in_the_basket"
    dest = episode_dir(tmp_path, "runA", key, 6)
    assert dest == tmp_path / "runA" / task_slug(key) / "ep0006"
    with pytest.raises(ValueError, match="non-negative"):
        episode_dir(tmp_path, "runA", key, -1)
    with pytest.raises(ValueError, match="suite"):
        task_slug("not-a-key")


def test_assert_under_rejects_escape(tmp_path):
    root = tmp_path / "videos"
    root.mkdir()
    inside = root / "a" / "b.jpg"
    inside.parent.mkdir()
    inside.write_bytes(b"x")
    assert assert_under(root, inside) == inside.resolve()
    with pytest.raises(ValueError, match="outside"):
        assert_under(root, tmp_path / "secret")


def test_write_jpeg_sequence_round_trip(tmp_path):
    frames = _rgb()
    paths = write_jpeg_sequence(frames, tmp_path / "agentview", quality=80)
    assert [p.name for p in paths] == ["00000.jpg", "00001.jpg", "00002.jpg", "00003.jpg"]
    assert all(p.is_file() and p.stat().st_size > 0 for p in paths)


def test_write_jpeg_rejects_wrong_shape(tmp_path):
    with pytest.raises(ValueError, match="expected"):
        write_jpeg_sequence(
            [np.zeros((8, 10), dtype=np.uint8)], tmp_path / "cam", quality=80
        )


def test_write_mp4_synthetic(tmp_path):
    cv2 = pytest.importorskip("cv2")
    path = tmp_path / "agentview.mp4"
    write_mp4(_rgb(n=6, h=16, w=16), path, fps=20.0, codec="mp4v")
    assert path.is_file() and path.stat().st_size > 0
    cap = cv2.VideoCapture(str(path))
    try:
        ok, frame = cap.read()
        assert ok and frame is not None
    finally:
        cap.release()


def test_persist_and_discover(tmp_path):
    frames_a = _rgb(seed=1)
    frames_w = _rgb(seed=2)
    result = RolloutResult(
        success=True,
        n_steps=3,
        task_key="libero_object/pick_up_the_milk_and_place_it_in_the_basket",
        episode_idx=0,
        seed=123,
        n_replans=1,
        frames=frames_a,
        camera_frames={"agentview": frames_a, "robot0_eye_in_hand": frames_w},
    )
    dest = episode_dir(tmp_path, "runA", result.task_key, 0)
    art = persist_rollout(
        result,
        dest,
        fps=20.0,
        jpeg_quality=80,
        mp4_codec="mp4v",
        checkpoint=Path("/tmp/ckpt.pt"),
        extra={"run_id": "runA"},
    )
    assert art.n_frames == 4
    assert art.meta["success"] is True
    assert art.meta["run_id"] == "runA"
    assert art.meta["has_trace"] is False
    assert not (dest / "trace.json").exists()
    restored = load_artifact(dest)
    assert restored.cameras == ("agentview", "robot0_eye_in_hand")
    found = discover_artifacts(tmp_path)
    assert len(found) == 1
    assert found[0].directory == dest


def test_discover_runs_reads_config_and_eval(tmp_path):
    run = tmp_path / "single__libero_object__milk__seed0"
    (run / "checkpoints").mkdir(parents=True)
    (run / "checkpoints" / "final.pt").write_bytes(b"not-a-real-checkpoint")
    (run / "config.yaml").write_text(
        "run_id: single__libero_object__milk__seed0\n"
        "tasks:\n"
        "  - libero_object/pick_up_the_milk_and_place_it_in_the_basket\n"
    )
    (run / "eval.json").write_text(
        json.dumps(
            {
                "run_id": "single__libero_object__milk__seed0",
                "stage": 0,
                "tasks": [
                    {
                        "task_key": "libero_object/pick_up_the_milk_and_place_it_in_the_basket",
                        "successes": [True, False, True],
                    }
                ],
            }
        )
    )
    listings = discover_runs(tmp_path)
    assert len(listings) == 1
    listing = listings[0]
    assert listing.run_id == "single__libero_object__milk__seed0"
    assert listing.task_keys == [
        "libero_object/pick_up_the_milk_and_place_it_in_the_basket"
    ]
    assert listing.eval_successes[
        "libero_object/pick_up_the_milk_and_place_it_in_the_basket"
    ] == [True, False, True]


def test_resolve_out_dir_is_under_repo_by_default():
    cfg = load_video_config()
    path = resolve_out_dir(video_cfg=cfg)
    assert path == (repo_root() / "results" / "videos").resolve()


def test_persist_rollout_requires_frames(tmp_path):
    result = RolloutResult(
        success=False,
        n_steps=0,
        task_key="libero_object/pick_up_the_milk_and_place_it_in_the_basket",
        episode_idx=0,
        seed=0,
        n_replans=0,
    )
    with pytest.raises(ValueError, match="no frames"):
        persist_rollout(
            result,
            tmp_path / "empty",
            fps=20.0,
            jpeg_quality=80,
            mp4_codec="mp4v",
            checkpoint=Path("/tmp/ckpt.pt"),
        )


def test_watcher_state_and_media(tmp_path):
    """The localhost API lists runs and only serves files under the video root."""
    from flowcl.envs.viewer import WatchHandler, _WatchState, _artifact_payload

    run = tmp_path / "results" / "single__demo"
    (run / "checkpoints").mkdir(parents=True)
    (run / "checkpoints" / "final.pt").write_bytes(b"x")
    (run / "config.yaml").write_text(
        "run_id: single__demo\ntasks:\n  - libero_object/pick_up_the_milk_and_place_it_in_the_basket\n"
    )
    frames = _rgb(n=2, h=8, w=8)
    result = RolloutResult(
        success=True,
        n_steps=1,
        task_key="libero_object/pick_up_the_milk_and_place_it_in_the_basket",
        episode_idx=0,
        seed=1,
        n_replans=1,
        frames=frames,
        camera_frames={"agentview": frames},
    )
    videos = tmp_path / "videos"
    dest = episode_dir(
        videos, "single__demo", result.task_key, 0
    )
    art = persist_rollout(
        result,
        dest,
        fps=20.0,
        jpeg_quality=80,
        mp4_codec="mp4v",
        checkpoint=run / "checkpoints" / "final.pt",
        extra={"run_id": "single__demo"},
    )

    cfg = load_video_config()
    state = _WatchState(
        results_root=tmp_path / "results",
        out_dir=videos,
        video_cfg=cfg,
        device="cpu",
    )
    server = ThreadingHTTPServer(("127.0.0.1", 0), WatchHandler)
    server.watch = state
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        host, port = server.server_address
        conn = HTTPConnection(host, port, timeout=5)
        conn.request("GET", "/api/state")
        payload = json.loads(conn.getresponse().read())
        assert payload["runs"][0]["run_id"] == "single__demo"
        assert payload["videos"][0]["n_frames"] == 2
        rel = _artifact_payload(art, videos)["rel_dir"]
        conn.request("GET", f"/media/{rel}/agentview/00000.jpg")
        frame_resp = conn.getresponse()
        assert frame_resp.status == 200
        assert frame_resp.getheader("Content-Type", "").startswith("image/jpeg")
        assert frame_resp.read()
        conn.request("GET", "/media/../secret.jpg")
        forbidden = conn.getresponse()
        assert forbidden.status == 400
        forbidden.read()
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)


def _line_trace(n_steps: int = 6) -> RolloutTrace:
    """Constant-velocity EE motion along x; one action rail at t=0."""
    n_states = n_steps + 1
    states = np.zeros((n_states, 8), dtype=np.float32)
    states[:, 0] = np.arange(n_states, dtype=np.float32) * 0.1
    actions = np.zeros((n_steps, 7), dtype=np.float32)
    actions[0, 0] = 1.0
    replan = np.zeros(n_steps, dtype=bool)
    replan[0] = True
    if n_steps > 4:
        replan[4] = True
    return RolloutTrace(
        states=states,
        actions=actions,
        replan=replan,
        chunks=np.zeros((0, 16, 7), dtype=np.float32),
        state_names=(
            "ee_pos_0",
            "ee_pos_1",
            "ee_pos_2",
            "ee_ori_0",
            "ee_ori_1",
            "ee_ori_2",
            "gripper_states_0",
            "gripper_states_1",
        ),
        action_names=tuple(f"action_{i}" for i in range(7)),
    )


def test_smoothness_zero_jerk_on_constant_velocity():
    trace = _line_trace()
    summary = smoothness_summary(
        trace, fps=10.0, clip_abs=1.0, clip_rail_tol=1e-6
    )
    assert summary["max_ee_step"] == pytest.approx(0.1, rel=0, abs=1e-6)
    # float32 0.1 is inexact; constant-velocity jerk should be ~0, not "large".
    assert summary["rms_ee_jerk"] == pytest.approx(0.0, rel=0, abs=1e-4)
    assert summary["n_replans"] == 2
    assert summary["action_clip_count"] == 1
    assert summary["action_clip_fraction"] == pytest.approx(1.0 / (6 * 7))
    assert summary["ee_pos_columns"] == ["ee_pos_0", "ee_pos_1", "ee_pos_2"]


def test_smoothness_does_not_guess_ee_pos_from_column_order():
    trace = _line_trace()
    object.__setattr__(
        trace,
        "state_names",
        tuple(f"state_{i}" for i in range(8)),
    )
    summary = smoothness_summary(
        trace, fps=10.0, clip_abs=1.0, clip_rail_tol=1e-6
    )
    assert summary["ee_pos_columns"] == []
    assert summary["max_ee_step"] is None
    assert summary["rms_ee_jerk"] is None


def test_persist_writes_trace_aligned_to_frames(tmp_path):
    n_steps = 3
    frames = _rgb(n=n_steps + 1)
    trace = _line_trace(n_steps)
    result = RolloutResult(
        success=True,
        n_steps=n_steps,
        task_key="libero_object/pick_up_the_milk_and_place_it_in_the_basket",
        episode_idx=0,
        seed=1,
        n_replans=1,
        frames=frames,
        camera_frames={"agentview": frames},
        trace=trace,
    )
    dest = episode_dir(tmp_path, "runA", result.task_key, 0)
    art = persist_rollout(
        result,
        dest,
        fps=20.0,
        jpeg_quality=80,
        mp4_codec="mp4v",
        checkpoint=Path("/tmp/ckpt.pt"),
        extra={"run_id": "runA"},
        clip_abs=1.0,
        clip_rail_tol=1e-6,
    )
    assert art.meta["has_trace"] is True
    payload = json.loads((dest / "trace.json").read_text())
    assert payload["n_states"] == n_steps + 1
    assert payload["n_steps"] == n_steps
    loaded = np.load(dest / "trace.npz")
    assert loaded["states"].shape == (n_steps + 1, 8)
    assert loaded["actions"].shape == (n_steps, 7)


def test_persist_rejects_trace_misaligned_with_video(tmp_path):
    frames = _rgb(n=2)
    result = RolloutResult(
        success=True,
        n_steps=3,
        task_key="libero_object/pick_up_the_milk_and_place_it_in_the_basket",
        episode_idx=0,
        seed=1,
        n_replans=1,
        frames=frames,
        camera_frames={"agentview": frames},
        trace=_line_trace(3),
    )
    with pytest.raises(ValueError, match="simultaneous"):
        persist_rollout(
            result,
            tmp_path / "bad",
            fps=20.0,
            jpeg_quality=80,
            mp4_codec="mp4v",
            checkpoint=Path("/tmp/ckpt.pt"),
            clip_abs=1.0,
            clip_rail_tol=1e-6,
        )


def test_persist_trace_round_trip(tmp_path):
    trace = _line_trace(4)
    payload = persist_trace(
        trace, tmp_path, fps=10.0, clip_abs=1.0, clip_rail_tol=1e-6
    )
    assert (tmp_path / "trace.json").is_file()
    assert (tmp_path / "trace.npz").is_file()
    assert payload["smoothness"]["rms_ee_jerk"] == pytest.approx(0.0, rel=0, abs=1e-4)
