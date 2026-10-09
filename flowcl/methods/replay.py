"""Experience replay with a fixed demo buffer (spec §6 item 2; docs/thesis_plan.md, Workstream B).

After each task the first ``buffer_demos`` demonstrations of that task (demo indices
``0..buffer_demos-1``) are kept, under the frozen T1 normalization stats. From the second task
on, every training batch is built here: ``(1 - replay_fraction)`` of it from the current task,
uniform over its ``(episode, t)`` chunks, and ``replay_fraction`` from the buffer, **split
equally across the finished tasks** (``n // k`` each, the first ``n % k`` tasks in curriculum
order taking one more) and uniform within each task's buffered demos. Fixed a priori; no bracket.

At T1 the buffer is empty and :meth:`build_batch` returns ``None``, so stage 0 is seq_ft's code
path, bitwise. From T2 on the batches come from replay's own generators
(``derive_seed(ns, "replay::current::<task>", τ)`` and ``"replay::buffer::<task>"``), and the
runner's DataLoader no longer consumes the shared stage generator: after T1, the ``s``/``A_0``
stream is **not** paired with seq_ft's (the rollout episodes still are).

The buffer violates the exemplar-free setting (``is_exemplar_free = False``). Each task's
buffered demos are an artifact (``method/replay_task{τ}.pt``, torch tensors) that every later
stage checkpoint names by SHA-256, so a resumed run restores exactly the same buffer.
"""

from __future__ import annotations

import hashlib
import os
from pathlib import Path

import numpy as np
import torch

from flowcl.data.episode import Episode
from flowcl.methods.base import BaseMethod, register_method
from flowcl.utils.run import file_sha256

ARTIFACT_KIND = "replay_buffer"


def equal_split(n: int, k: int) -> list[int]:
    """``n`` samples over ``k`` tasks: ``n // k`` each, the first ``n % k`` take one more."""
    if k < 1 or n < 0:
        raise ValueError(f"cannot split {n} samples over {k} tasks")
    return [n // k + (1 if j < n % k else 0) for j in range(k)]


def episodes_fingerprint(episodes: list[Episode]) -> str:
    """SHA-256 over every buffered array (cameras in name order) and demo index."""
    h = hashlib.sha256()
    for ep in episodes:
        h.update(str(ep.meta.get("demo_idx")).encode())
        h.update(ep.task_id.encode())
        for cam in sorted(ep.images):
            h.update(np.ascontiguousarray(ep.images[cam]).tobytes())
        h.update(np.ascontiguousarray(ep.state).tobytes())
        h.update(np.ascontiguousarray(ep.action).tobytes())
    return h.hexdigest()


def _episode_payload(ep: Episode) -> dict:
    return {"images": {c: torch.from_numpy(np.ascontiguousarray(v)) for c, v in ep.images.items()},
            "state": torch.from_numpy(np.ascontiguousarray(ep.state)),
            "action": torch.from_numpy(np.ascontiguousarray(ep.action)),
            "language": ep.language, "task_id": ep.task_id, "embodiment": ep.embodiment,
            "meta": dict(ep.meta)}


def _episode_from_payload(d: dict) -> Episode:
    return Episode(images={c: v.numpy() for c, v in d["images"].items()}, state=d["state"].numpy(),
                   action=d["action"].numpy(), language=d["language"], task_id=d["task_id"],
                   embodiment=d["embodiment"], meta=dict(d["meta"]))


@register_method
class Replay(BaseMethod):
    """Fixed-buffer experience replay; see the module docstring."""

    name = "replay"

    def __init__(self, buffer_demos: int = 10, replay_fraction: float = 0.5) -> None:
        super().__init__()
        if int(buffer_demos) < 1:
            raise ValueError(f"buffer_demos must be >= 1, got {buffer_demos}")
        if not 0.0 < float(replay_fraction) < 1.0:
            raise ValueError(f"replay_fraction must lie in (0, 1), got {replay_fraction}")
        self.buffer_demos = int(buffer_demos)
        self.replay_fraction = float(replay_fraction)
        # task key -> (task index, buffered episodes, fingerprint), in curriculum order
        self._buffer: dict[str, dict] = {}
        self._datasets: dict[str, object] = {}
        self._gen_current: torch.Generator | None = None
        self._gen_buffer: torch.Generator | None = None
        self._batch_size: int | None = None

    def config(self) -> dict:
        return {"buffer_demos": self.buffer_demos, "replay_fraction": self.replay_fraction}

    @property
    def is_exemplar_free(self) -> bool:
        return False

    def buffer_tasks(self) -> list[str]:
        return list(self._buffer)

    # ---- §6 hooks --------------------------------------------------------------

    def on_task_start(self, policy, task_idx, *, context) -> None:
        from flowcl.data.dataset import ChunkedActionDataset
        from flowcl.utils.seeding import derive_seed

        if not self._buffer:
            self._gen_current = self._gen_buffer = None
            return
        ns, key, batch_size = context.seed_namespace_run_id, context.task_key, context.batch_size
        if ns is None or key is None or batch_size is None or context.dataset is None:
            raise RuntimeError("replay needs the seed namespace, the task key, the dataset and the "
                               "batch size to build its batches")
        self._batch_size = int(batch_size)
        n_buffer = int(round(self._batch_size * self.replay_fraction))
        if not 1 <= n_buffer < self._batch_size:
            raise ValueError(f"batch size {self._batch_size} with replay_fraction {self.replay_fraction} "
                             "leaves no room for both halves")
        self._gen_current = torch.Generator().manual_seed(derive_seed(ns, f"replay::current::{key}", task_idx))
        self._gen_buffer = torch.Generator().manual_seed(derive_seed(ns, f"replay::buffer::{key}", task_idx))
        spec, stats = context.dataset.spec, context.dataset.stats
        self._datasets = {k: ChunkedActionDataset(b["episodes"], spec, stats) for k, b in self._buffer.items()}

    def build_batch(self, dataset, task_idx) -> dict | None:
        from flowcl.data.dataset import collate_chunks

        if not self._buffer:
            return None  # T1: the runner's DataLoader (seq_ft's path)
        n_buffer = int(round(self._batch_size * self.replay_fraction))
        n_current = self._batch_size - n_buffer
        samples = [dataset[i] for i in
                   torch.randint(len(dataset), (n_current,), generator=self._gen_current).tolist()]
        for key, count in zip(self._buffer, equal_split(n_buffer, len(self._buffer))):
            ds = self._datasets[key]
            samples += [ds[i] for i in torch.randint(len(ds), (count,), generator=self._gen_buffer).tolist()]
        return collate_chunks(samples)

    def on_task_end(self, policy, task_idx, *, context) -> None:
        key = context.task_key
        if key is None or context.dataset is None:
            raise RuntimeError("replay buffers one task's demos; it needs the task key and dataset")
        own = sorted((ep for ep in context.dataset.episodes if ep.task_id == key),
                     key=lambda ep: ep.meta.get("demo_idx", -1))
        kept = own[: self.buffer_demos]
        demo_idx = [ep.meta.get("demo_idx") for ep in kept]
        if demo_idx != list(range(self.buffer_demos)):
            raise ValueError(f"{key}: expected demos 0..{self.buffer_demos - 1}, found {demo_idx}")
        self._buffer[key] = {"task_idx": task_idx, "episodes": kept,
                             "fingerprint": episodes_fingerprint(kept)}
        self._datasets = {}

    # ---- artifacts -------------------------------------------------------------

    def save_artifacts(self, directory, task_idx, *, context) -> list[Path]:
        """Every buffered task's file (written if missing, e.g. after a resume), in order."""
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        paths = []
        for key, b in self._buffer.items():
            path = directory / f"replay_task{b['task_idx']}.pt"
            if not path.exists():
                tmp = path.with_name(path.name + ".tmp")
                torch.save({"kind": ARTIFACT_KIND, "task_key": key, "task_idx": b["task_idx"],
                            "config": self.config(), "fingerprint": b["fingerprint"],
                            "episodes": [_episode_payload(ep) for ep in b["episodes"]]}, tmp)
                os.replace(tmp, path)
            paths.append(path)
        return paths

    def restore_state(self, checkpoint_extra: dict, source_run: Path, task_idx: int) -> dict:
        artifacts = {a["path"]: a["sha256"] for a in checkpoint_extra.get("method_artifacts") or []}
        expected = (checkpoint_extra.get("method_state") or {}).get("buffer") or {}
        if not expected:
            raise ValueError(f"the stage-{task_idx} checkpoint records no replay buffer")
        restored = {}
        for key, info in sorted(expected.items(), key=lambda kv: kv[1]["task_idx"]):
            rel = f"method/replay_task{info['task_idx']}.pt"
            if rel not in artifacts:
                raise ValueError(f"the stage-{task_idx} checkpoint names no {rel}")
            path = Path(source_run) / rel
            actual = file_sha256(path)
            if actual != artifacts[rel]:
                raise ValueError(f"{path}: SHA-256 {actual} does not match the checkpoint's {artifacts[rel]}")
            payload = torch.load(path, map_location="cpu", weights_only=False)
            episodes = [_episode_from_payload(d) for d in payload["episodes"]]
            fingerprint = episodes_fingerprint(episodes)
            problems = []
            if payload.get("kind") != ARTIFACT_KIND or payload.get("task_key") != key:
                problems.append(f"kind/task {payload.get('kind')!r}/{payload.get('task_key')!r}")
            if payload.get("config") != self.config():
                problems.append(f"config {payload.get('config')} != {self.config()}")
            if not fingerprint == payload.get("fingerprint") == info["fingerprint"]:
                problems.append("buffer fingerprint differs")
            if problems:
                raise ValueError(f"{path}: replay buffer does not match: {problems}")
            restored[key] = {"task_idx": info["task_idx"], "episodes": episodes, "fingerprint": fingerprint}
        self._buffer = restored
        self._datasets = {}
        return {"tasks": list(restored), "fingerprints": {k: v["fingerprint"] for k, v in restored.items()}}

    # ---- reporting -------------------------------------------------------------

    def stored_bytes(self) -> int:
        total = 0
        for b in self._buffer.values():
            for ep in b["episodes"]:
                total += sum(v.nbytes for v in ep.images.values()) + ep.state.nbytes + ep.action.nbytes
        return total

    def state_dict(self) -> dict:
        return {"name": self.name, "config": self.config(), "stored_mb": self.stored_bytes() / 1e6,
                "buffer": {k: {"task_idx": b["task_idx"], "fingerprint": b["fingerprint"],
                               "demo_idx": [ep.meta.get("demo_idx") for ep in b["episodes"]]}
                           for k, b in self._buffer.items()}}

    def describe(self) -> str:
        return f"replay ({self.buffer_demos} demos/task, stores {self.stored_bytes() / 1e6:.1f} MB)"
