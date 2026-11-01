"""train_on_tasks' periodic checkpoints: written at multiples of ``checkpoint_every`` before
the last step, each the exact state a shorter run ends in."""

from __future__ import annotations

import torch

import flowcl.data.piper_tasks as piper_tasks
from flowcl.data.config import load_embodiment_spec
from flowcl.data.piper_adapter import build_cache
from flowcl.data.tasks import TaskRef
from flowcl.train.pipeline import train_on_tasks
from flowcl.train.trainer import TrainConfig
from test_low_update import TINY_POLICY
from test_piper_adapter import toy_task, write_dataset


def test_step_checkpoints_match_a_run_stopped_there(tmp_path, monkeypatch):
    spec = load_embodiment_spec("agilex_piper")
    write_dataset(tmp_path / "raw" / "toy")
    task = toy_task(tmp_path / "raw", holdout=1)
    monkeypatch.setitem(piper_tasks.TASKS, "toy", task)
    build_cache(task, tmp_path / "raw", piper_tasks.cache_path("toy", tmp_path), spec)
    ref = TaskRef.from_key("piper/toy")
    base = dict(steps=4, batch_size=2, warmup_steps=1, device="cpu", log_every=0, num_workers=0)

    def train(root, **extra):
        return train_on_tasks([ref], spec=spec, policy_config=TINY_POLICY, train_cfg=TrainConfig(**base, **extra),
                              run_id="toy_run", seed=0, dataset_dir=tmp_path, results_root=tmp_path / root,
                              pretrained=False)

    full = train("full", checkpoint_every=2)
    written = sorted(p.name for p in (full.run.path / "checkpoints").iterdir())
    assert written == ["final.pt", "step_000002.pt"]
    short = train("short", stop_after=2)
    step2 = torch.load(full.run.path / "checkpoints" / "step_000002.pt", weights_only=False)
    final2 = torch.load(short.checkpoint, weights_only=False)
    assert step2["extra"]["step"] == 2
    assert step2["state_dict"].keys() == final2["state_dict"].keys()
    assert all(torch.equal(v, final2["state_dict"][k]) for k, v in step2["state_dict"].items())
