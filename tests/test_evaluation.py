"""Evaluation protocol plumbing (§8.1): config parsing and report persistence."""

from __future__ import annotations

import pytest
from omegaconf import OmegaConf

from flowcl.analysis.metrics import success_estimate
from flowcl.envs.evaluation import (
    EvaluationReport,
    TaskEvaluation,
    eval_config_from_dict,
)
from flowcl.utils.libero_paths import repo_root


def test_shipped_eval_config_matches_the_protocol():
    """§8.1: 50 rollouts on the shared fixed set; §4.4: ensembling off by default."""
    payload = OmegaConf.to_container(
        OmegaConf.load(repo_root() / "configs" / "eval" / "libero_eval.yaml"),
        resolve=True,
    )
    cfg = eval_config_from_dict(payload)
    assert cfg.n_episodes == 50
    assert cfg.temporal_ensembling is False
    assert cfg.execute_k is None, "null means 'use the embodiment spec's k'"
    assert cfg.euler_steps == 10
    assert cfg.record_video is False
    assert payload["bootstrap"]["confidence"] == 0.95


def test_eval_config_rejects_unknown_keys():
    """A typo'd protocol key must not be silently ignored."""
    with pytest.raises(ValueError, match="unknown keys"):
        eval_config_from_dict({"n_epsiodes": 50})


def test_eval_config_ignores_the_bootstrap_block():
    cfg = eval_config_from_dict({"n_episodes": 5, "bootstrap": {"seed": 3}})
    assert cfg.n_episodes == 5


def make_evaluation(task_key: str, successes: list[bool]) -> TaskEvaluation:
    return TaskEvaluation(
        task_key=task_key,
        successes=successes,
        n_steps=[100] * len(successes),
        seeds=list(range(len(successes))),
        estimate=success_estimate(successes, seed=0),
    )


def test_report_round_trips(tmp_path):
    report = EvaluationReport(
        run_id="r",
        stage=1,
        tasks=[make_evaluation("s/a", [True] * 8 + [False] * 2)],
    )
    path = report.save(tmp_path / "eval.json")
    restored = EvaluationReport.load(path)

    assert restored.run_id == "r"
    assert restored.stage == 1
    assert restored.by_task()["s/a"].estimate.value == pytest.approx(0.8)


def test_report_stores_per_rollout_successes_not_just_the_rate(tmp_path):
    """So a CI can be recomputed later without re-running 50 rollouts."""
    successes = [True, False, True, True]
    report = EvaluationReport(run_id="r", stage=0, tasks=[make_evaluation("s/a", successes)])
    payload = report.as_dict()
    assert payload["tasks"][0]["successes"] == successes


def test_report_carries_a_ci_for_every_rate(tmp_path):
    """§11: no bare percentages."""
    report = EvaluationReport(
        run_id="r", stage=0, tasks=[make_evaluation("s/a", [True] * 25 + [False] * 25)]
    )
    entry = report.as_dict()["tasks"][0]
    assert entry["ci_low"] < entry["success_rate"] < entry["ci_high"]
    assert entry["confidence"] == 0.95
