"""Piper policy server: metadata from the training cache, the image path, the HTTP contract.

A synthetic LeRobot folder (``test_piper_adapter.write_dataset``) is cached as registry
task ``toy``, a small random policy is saved as a Piper checkpoint, and a real server
answers on a free local port.
"""

from __future__ import annotations

import base64
import json
import threading
import urllib.error
import urllib.request
from types import SimpleNamespace

import cv2
import numpy as np
import pytest
import torch
from torch import nn

from flowcl.analysis import open_loop
from flowcl.data import piper_tasks
from flowcl.data.config import load_embodiment_spec
from flowcl.data.images import policy_frame
from flowcl.data.piper_adapter import CAMERA_KEYS, build_cache, iter_cache_episodes
from flowcl.data.stats import compute_stats
from flowcl.deploy.piper_server import (
    ENVELOPE_MARGIN_GRIPPER_M,
    ENVELOPE_MARGIN_RAD,
    PiperPolicyService,
    RequestError,
    decode_frame,
    make_server,
    task_of,
    training_summary,
)
from flowcl.models.build import build_policy
from flowcl.train.checkpoint import save_checkpoint

from test_piper_adapter import toy_task, write_dataset

TINY_POLICY = {"d_model": 384, "n_trunk_layers": 6, "n_heads": 8, "n_decoder_layers": 2,
               "n_context_tokens": 4, "pretrained": False, "euler_steps": 2}
LANGUAGE = "put the toy away"


def pattern_frame(h: int = 480, w: int = 640, shift: int = 0) -> np.ndarray:
    """A smooth RGB test image (JPEG keeps it close), different in every channel and
    nowhere black, so a masked pixel stands out."""
    y, x = np.mgrid[0:h, 0:w]
    rgb = 40 + np.stack([(x + shift) * 200 // (w + shift), y * 200 // h, ((x + y) // 4 + shift) % 200], axis=-1)
    return rgb.astype(np.uint8)


def jpeg_b64(rgb: np.ndarray, quality: int = 90) -> str:
    ok, buf = cv2.imencode(".jpg", np.ascontiguousarray(rgb[:, :, ::-1]), [cv2.IMWRITE_JPEG_QUALITY, quality])
    assert ok
    return base64.b64encode(buf.tobytes()).decode("ascii")


@pytest.fixture(scope="module")
def world(tmp_path_factory):
    root = tmp_path_factory.mktemp("piper_server")
    spec = load_embodiment_spec("agilex_piper")
    write_dataset(root / "raw" / "toy")
    task = toy_task(root / "raw", holdout=1, language=LANGUAGE)  # trains on episodes 0-1, holds out 2
    with pytest.MonkeyPatch.context() as mp:
        mp.setitem(piper_tasks.TASKS, task.name, task)
        build_cache(task, root / "raw", root / "piper" / "cache" / "toy.h5", spec)
        episodes = list(iter_cache_episodes(root / "piper" / "cache" / "toy.h5", spec, n_demos=2,
                                            task_id="piper/toy"))
        stats = compute_stats(episodes, spec.name, "piper/toy", normalize_actions=True)
        torch.manual_seed(0)
        policy = build_policy(TINY_POLICY, spec, pretrained=False)
        for p in policy.parameters():
            if p.requires_grad:
                nn.init.normal_(p, std=0.05)
        checkpoint = save_checkpoint(root / "run" / "checkpoints" / "final.pt", policy=policy,
                                     policy_config=TINY_POLICY, spec=spec, stats=stats, run_id="toy_run",
                                     task_key="piper/toy")
        log_path = root / "serve" / "requests.jsonl"
        service = PiperPolicyService(checkpoint, device="cpu", dataset_dir=root / "piper", log_path=log_path)
        server = make_server(service, "127.0.0.1", 0)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            yield SimpleNamespace(root=root, spec=spec, task=task, episodes=episodes, service=service,
                                  url=f"http://127.0.0.1:{server.server_address[1]}", log_path=log_path)
        finally:
            server.shutdown()
            server.server_close()


def get(url: str) -> tuple[int, bytes]:
    try:
        with urllib.request.urlopen(url, timeout=60) as r:
            return r.status, r.read()
    except urllib.error.HTTPError as e:
        return e.code, e.read()


def post(url: str, body: bytes) -> tuple[int, dict]:
    request = urllib.request.Request(url + "/infer", data=body, method="POST",
                                     headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(request, timeout=60) as r:
            return r.status, json.loads(r.read())
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read())


def good_request(world, **overrides) -> dict:
    request = {"seq": 1, "instruction": LANGUAGE, "state": [0.5] * 7,
               "images": {"front": jpeg_b64(pattern_frame()), "wrist": jpeg_b64(pattern_frame(shift=40))}}
    request.update(overrides)
    return request


# ---- metadata -------------------------------------------------------------------------


def test_task_of_reads_the_piper_task_key(world):
    assert task_of("piper/toy") is world.task
    for bad in ("libero_spatial/toy", "piper/", None):
        with pytest.raises(ValueError):
            task_of(bad)


def test_metadata_comes_from_the_training_episodes(world):
    code, body = get(world.url + "/metadata")
    assert code == 200
    meta = json.loads(body)
    assert meta["protocol"] == 1 and meta["instruction"] == LANGUAGE
    assert meta["run_id"] == "toy_run" and meta["task_key"] == "piper/toy"
    assert meta["cameras"] == {"front": CAMERA_KEYS["front"], "wrist": CAMERA_KEYS["wrist"]}
    assert meta["frame_hw"] == [480, 640] and meta["policy_image_hw"] == [128, 128]
    assert meta["image_masks"]["front"] == [[[0, 0], [260, 0], [260, 115], [150, 250], [0, 260]]]
    assert (meta["control_rate_hz"], meta["chunk_horizon"], meta["execute_k"]) == (15.0, 16, 8)
    assert meta["n_demos_trained"] == 2 and meta["euler_steps"] == 2
    # actions are row index + 0.5: episode 0 holds rows 0-3, episode 1 rows 4-6
    assert meta["start_pose"] == [2.5] * 7  # median of the first commands 0.5 and 4.5
    margin = [ENVELOPE_MARGIN_RAD] * 6 + [ENVELOPE_MARGIN_GRIPPER_M]
    np.testing.assert_allclose(meta["envelope"]["lo"], 0.5 - np.array(margin), atol=1e-4)
    np.testing.assert_allclose(meta["envelope"]["hi"], 6.5 + np.array(margin), atol=1e-4)
    assert meta["held_out"] == [{"dataset": "toy", "episode_index": 2}]
    assert meta["checkpoint_sha256"] == world.service.sha256


def test_fewer_training_demos_narrow_the_summary(world):
    summary = training_summary(world.task, n_demos=1, dataset_dir=world.root / "piper")
    assert summary["n_demos"] == 1 and summary["start_pose"] == [0.5] * 7
    np.testing.assert_allclose(summary["envelope"]["hi"][0], 3.5 + ENVELOPE_MARGIN_RAD, atol=1e-4)
    assert summary["held_out"] == [{"dataset": "toy", "episode_index": 2}]  # the registry's hold-out
    with pytest.raises(ValueError, match="held out"):
        training_summary(world.task, n_demos=3, dataset_dir=world.root / "piper")


# ---- the image path -------------------------------------------------------------------


def test_frames_are_decoded_as_the_recorder_decodes_them(world):
    rgb = pattern_frame()
    data = jpeg_b64(rgb)
    recorder = cv2.cvtColor(cv2.imdecode(np.frombuffer(base64.b64decode(data), np.uint8), cv2.IMREAD_COLOR),
                            cv2.COLOR_BGR2RGB)
    decoded = decode_frame(data, "front")
    np.testing.assert_array_equal(decoded, recorder)
    assert np.abs(decoded.astype(int) - rgb).mean() < 2  # RGB order kept through the JPEG

    inputs = world.service.policy_inputs({"front": data, "wrist": data})
    np.testing.assert_array_equal(inputs["front"], policy_frame(recorder, "front", world.spec))
    np.testing.assert_array_equal(inputs["wrist"], policy_frame(recorder, "wrist", world.spec))
    assert inputs["front"].shape == (128, 128, 3) and inputs["front"].dtype == np.uint8
    assert inputs["front"][0, 0].max() == 0 and inputs["wrist"][0, 0].max() > 0  # only front is masked


def test_bad_frames_are_refused(world):
    with pytest.raises(RequestError, match="320x240"):
        decode_frame(jpeg_b64(pattern_frame(240, 320)), "front")
    with pytest.raises(RequestError, match="base64"):
        decode_frame("not base64!", "front")
    with pytest.raises(RequestError, match="not a JPEG"):
        decode_frame(base64.b64encode(b"hello").decode(), "front")
    with pytest.raises(RequestError, match="missing cameras"):
        world.service.policy_inputs({"front": jpeg_b64(pattern_frame())})


def test_a_chunk_matches_the_offline_check_on_the_same_inputs(world):
    """The server assembles its batch as training does: the offline check's tick-0 chunk."""
    episode = world.episodes[0]
    service = world.service
    offline = open_loop.predict_chunks(service.policy, service.spec, service.stats, episode,
                                       torch.device("cpu"), seed=7, batch_size=64)
    frames = {c: episode.images[c][0] for c in service.spec.cameras}
    served = service.sample(frames, episode.state[0], torch.Generator().manual_seed(7))
    assert served.shape == (16, 7)
    np.testing.assert_allclose(served, offline.predicted[0], atol=1e-5)


# ---- HTTP -------------------------------------------------------------------------------


def test_infer_over_http(world):
    code, reply = post(world.url, json.dumps(good_request(world, seq=41, seed=3, return_inputs=True)).encode())
    assert code == 200, reply
    actions = np.asarray(reply["actions"])
    assert reply["seq"] == 41 and actions.shape == (16, 7) and np.isfinite(actions).all()
    assert set(reply["timing_ms"]) == {"decode", "policy", "total"}
    expected = world.service.policy_inputs(good_request(world)["images"])
    for camera, png in reply["inputs"].items():
        img = cv2.imdecode(np.frombuffer(base64.b64decode(png), np.uint8), cv2.IMREAD_COLOR)[:, :, ::-1]
        np.testing.assert_array_equal(img, expected[camera])

    _, again = post(world.url, json.dumps(good_request(world, seed=3)).encode())
    assert again["actions"] == reply["actions"]  # a seed reproduces the chunk
    assert "inputs" not in again
    _, first = post(world.url, json.dumps(good_request(world)).encode())
    _, second = post(world.url, json.dumps(good_request(world)).encode())
    assert first["actions"] != second["actions"]  # without one, the server's generator moves on


@pytest.mark.parametrize("change, message", [
    ({"instruction": "pick the toy"}, "not the trained one"),
    ({"state": [0.0] * 6}, "7 finite numbers"),
    ({"state": [0.0] * 6 + [float("nan")]}, "7 finite numbers"),
    ({"state": "zero"}, "list of numbers"),
    ({"images": {"front": jpeg_b64(pattern_frame())}}, "missing cameras"),
    ({"images": {"front": jpeg_b64(pattern_frame(240, 320)), "wrist": jpeg_b64(pattern_frame(240, 320))}},
     "320x240"),
    ({"seed": -1}, "seed"),
    ({"seed": True}, "seed"),
])
def test_bad_requests_get_400_and_the_server_keeps_serving(world, change, message):
    code, reply = post(world.url, json.dumps(good_request(world, **change)).encode())
    assert code == 400 and message in reply["error"]
    assert reply["seq"] == 1
    code, _ = post(world.url, json.dumps(good_request(world)).encode())
    assert code == 200


def test_malformed_bodies_and_paths(world):
    assert post(world.url, b"{not json")[0] == 400
    assert post(world.url, b"")[0] == 400
    assert post(world.url, b"[1, 2]")[0] == 400
    assert get(world.url + "/nothing")[0] == 404
    request = urllib.request.Request(world.url + "/metadata", data=b"{}", method="POST")
    with pytest.raises(urllib.error.HTTPError) as e:
        urllib.request.urlopen(request, timeout=60)
    assert e.value.code == 404


def test_status_and_request_log(world):
    post(world.url, json.dumps(good_request(world, seq=99)).encode())
    post(world.url, json.dumps(good_request(world, seq=100, instruction="no")).encode())
    code, page = get(world.url + "/")
    assert code == 200 and LANGUAGE in page.decode()
    status = json.loads(get(world.url + "/status.json")[1])
    assert status["requests_total"] >= 2 and status["errors"]
    assert {r["seq"] for r in status["recent"]} >= {99, 100}
    lines = [json.loads(line) for line in world.log_path.read_text().splitlines()]
    assert lines[0]["kind"] == "start" and lines[0]["metadata"]["instruction"] == LANGUAGE
    by_seq = {line.get("seq"): line for line in lines if line["kind"] == "infer"}
    assert by_seq[99]["error"] is None and len(by_seq[99]["actions"]) == 16
    assert "not the trained one" in by_seq[100]["error"] and by_seq[100]["actions"] is None
