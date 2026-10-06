"""The sequential continual-learning runner (§6, §8.1, §8.2).

One policy is carried across the curriculum's stages. After *every* stage the policy is
evaluated on *every* task — not only the ones seen so far — because §8.2's FWT needs the
above-diagonal entries ``R[j-1][j]``, performance on a task before it was ever trained.
Evaluating only the seen tasks is the cheaper thing to do and it silently makes FWT
uncomputable, which is why :meth:`RetentionMatrix.assert_complete_through` exists.

The runner is method-agnostic: it never branches on which method is active. It calls
the five §6 hooks in a fixed order via :func:`flowcl.train.trainer.train_one_task`, so
two methods' results differ only by their mechanism.

The §3.3 invariant is enforced at every stage boundary, as the spec demands: statistics
come from task 1 only and :func:`flowcl.data.stats.assert_frozen` is called before each
stage against the fingerprint recorded at stage 1. "Recomputing stats per task silently
changes the target distribution and corrupts forgetting measurements" — so the check is
a call site, not a comment.
"""

from __future__ import annotations

import dataclasses
import json
import shutil
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch

from flowcl.analysis.metrics import CLSummary, Estimate, RetentionMatrix, summarize
from flowcl.data.curriculum import Curriculum
from flowcl.data.spec import EmbodimentSpec
from flowcl.data.stats import NormalizationStats, assert_frozen
from flowcl.envs.evaluation import EvaluationReport, evaluate_tasks
from flowcl.envs.libero_env import EvalConfig
from flowcl.models.build import build_policy, load_policy_config
from flowcl.train.checkpoint import save_checkpoint
from flowcl.train.pipeline import build_dataset, fit_stats
from flowcl.train.trainer import TrainConfig, TrainLog, train_one_task
from flowcl.utils.run import RunHandle, atomic_write_text, create_run, file_sha256, git_sha
from flowcl.utils.seeding import derive_seed

# Default fail-fast bound for the T1 pairing check (see :func:`t1_pairing_check`).
T1_PAIRING_MAX_REL_DIFF = 0.05


@dataclass
class StageRecord:
    """What one curriculum stage produced."""

    stage: int
    task_key: str
    n_demos: int
    dataset_size: int
    train_log: TrainLog
    checkpoint: Path
    evaluation: EvaluationReport
    method_artifacts: list[dict] = field(default_factory=list)
    # A joint stage (co-trained tasks): its train tasks, per-task samples and text-cache warm-up.
    joint: dict | None = None

    def as_dict(self) -> dict:
        out = {
            "stage": self.stage,
            "task_key": self.task_key,
            "n_demos": self.n_demos,
            "dataset_size": self.dataset_size,
            "final_loss": self.train_log.final_loss,
            "mean_last_50_loss": self.train_log.mean_last(50),
            "steps": self.train_log.steps,
            "train_wall_clock_s": self.train_log.wall_clock_s,
            "checkpoint": str(self.checkpoint),
            "method_artifacts": self.method_artifacts,
            "evaluation": self.evaluation.as_dict(),
        }
        if self.joint is not None:
            out["joint"] = self.joint
        return out


@dataclass
class ContinualResult:
    """A complete sequential run: the retention matrix plus the §8.2 systems numbers."""

    run_id: str
    method: str
    curriculum: str
    seed: int
    task_keys: tuple[str, ...]
    matrix: RetentionMatrix
    stages: list[StageRecord] = field(default_factory=list)
    systems: dict = field(default_factory=dict)
    run: RunHandle | None = None
    seed_namespace_run_id: str | None = None
    method_registry_name: str | None = None
    t1_pairing: dict | None = None
    identity_checks: dict[str, dict] = field(default_factory=dict)
    # A resumed run: the stages before ``resume["start_stage"]`` come from the source run.
    resume: dict | None = None
    imported_stages: list[dict] = field(default_factory=list)

    def estimates(self) -> dict[tuple[int, str], Estimate]:
        """``(stage, task_key) -> Estimate``, so no rate travels without its CI."""
        out = {}
        for record in self.stages:
            for evaluation in record.evaluation.tasks:
                out[(record.stage, evaluation.task_key)] = evaluation.estimate
        return out

    def summary(self, baseline: dict[str, float] | None = None) -> CLSummary:
        return summarize(self.matrix, baseline=baseline)

    def as_dict(self, baseline: dict[str, float] | None = None) -> dict:
        return {
            "run_id": self.run_id,
            "method_run_id": self.run_id,
            "seed_namespace_run_id": self.seed_namespace_run_id,
            "method": self.method,
            "method_registry_name": self.method_registry_name,
            "curriculum": self.curriculum,
            "seed": self.seed,
            "task_keys": list(self.task_keys),
            "retention_matrix": {
                "task_keys": list(self.matrix.task_keys),
                # NaN is not valid JSON, so unevaluated cells are explicit nulls.
                "values": [
                    [None if np.isnan(v) else float(v) for v in row]
                    for row in self.matrix.values
                ],
                "n_rollouts": self.matrix.n_rollouts.tolist(),
            },
            "metrics": self.summary(baseline).as_dict(),
            "stages": self.imported_stages + [record.as_dict() for record in self.stages],
            "systems": self.systems,
            "t1_pairing": self.t1_pairing,
            "identity_checks": self.identity_checks,
            "resume": self.resume,
        }

    def save(self, path: str | Path, baseline: dict[str, float] | None = None) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.as_dict(baseline), indent=2) + "\n")
        return path


def continual_run_id(method: str, curriculum: str, seed: int) -> str:
    """Stable run id. Every field that changes the result is in the name."""
    return f"{curriculum}__{method}__seed{seed}"


def seed_namespace_run_id(curriculum: str, seed: int) -> str:
    """The run id every method's seeded streams derive from (§8.3).

    Training data order, flow times and noise, rollout seeds, and GPM's memory capture
    all derive from this id, never from a method's own run id, so every method in a
    comparison is paired with every other. It is the B1 baseline's id: for ``seq_ft``
    nothing changes, so the Gate 1 run stays reproducible and every later method is
    paired with it.
    """
    return continual_run_id("seq_ft", curriculum, seed)


# Train-config fields that may differ between a source run and its resumption (where and how
# fast it runs, not what it computes).
RESUME_TRAIN_EXEMPT = ("device", "num_workers", "log_every")


@dataclass
class ResumeSource:
    """A verified stage boundary of an earlier run (see :func:`load_resume_source`)."""

    run: Path
    start_stage: int
    checkpoint: Path
    sha256: str
    loaded: object        # LoadedCheckpoint of stage{start_stage - 1}.pt
    namespace: str
    evaluations: dict     # stage -> EvaluationReport, for stages < start_stage
    stage_dicts: list     # result.json-style records for stages < start_stage

    def record(self) -> dict:
        return {"run": str(self.run), "start_stage": self.start_stage,
                "checkpoint": str(self.checkpoint), "sha256": self.sha256,
                "imported_stages": list(range(self.start_stage))}


def load_resume_source(
    resume_run: Path,
    start_stage: int,
    curriculum: Curriculum,
    method_name: str,
    method_kwargs: dict | None,
    raw_policy_cfg: dict,
    train_cfg: TrainConfig,
    seed: int,
    expected_sha256: str | None = None,
    device: str = "cpu",
) -> ResumeSource:
    """Load and verify ``checkpoints/stage{K-1}.pt`` of ``resume_run`` (``K = start_stage``).

    Refuses (``ValueError``, every problem listed) unless the source's first ``K`` curriculum
    stages equal this curriculum's, its method and kwargs, recipe (except
    :data:`RESUME_TRAIN_EXEMPT`), policy config and seed equal this run's, the checkpoint is the
    end of stage ``K-1`` of that task (and has ``expected_sha256`` if given), its stats match the
    source's ``stats.json``, every method artifact it names is on disk with its recorded
    SHA-256, and every stage before ``K`` has its evaluation file. The seed namespace is the
    source's, so a resumed run keeps the source's pairing even under an extended curriculum.
    """
    from omegaconf import OmegaConf

    from flowcl.train.checkpoint import load_checkpoint

    resume_run = Path(resume_run)
    K = int(start_stage)
    if not 1 <= K < len(curriculum.stages):
        raise ValueError(f"start_stage must lie in 1..{len(curriculum.stages) - 1}, got {K}")
    conf = OmegaConf.to_container(OmegaConf.load(resume_run / "config.yaml"), resolve=True)
    checkpoint = resume_run / "checkpoints" / f"stage{K - 1}.pt"
    sha = file_sha256(checkpoint)
    problems: dict = {}
    if expected_sha256 is not None and sha != expected_sha256:
        problems["checkpoint_sha256"] = (sha, expected_sha256)

    src_tasks = [_stage_identity(t) for t in conf["curriculum"]["tasks"]]
    own_tasks = [_stage_identity(st.record()) for st in curriculum.stages]
    if len(src_tasks) < K or src_tasks[:K] != own_tasks[:K]:
        problems["curriculum"] = {"source": src_tasks[:K], "this": own_tasks[:K]}
    src_method = {k: v for k, v in conf["method"].items() if k != "display_name"}
    if src_method != {"name": method_name, **(method_kwargs or {})}:
        problems["method"] = {"source": src_method, "this": {"name": method_name, **(method_kwargs or {})}}
    if conf.get("seed") != seed:
        problems["seed"] = (conf.get("seed"), seed)
    defaults = {f.name: f.default for f in dataclasses.fields(TrainConfig)}
    own_train = vars(train_cfg)
    train_diff = {
        k: (conf["train"].get(k, defaults.get(k)), own_train.get(k))
        for k in set(conf["train"]) | set(own_train)
        if k not in RESUME_TRAIN_EXEMPT and conf["train"].get(k, defaults.get(k)) != own_train.get(k)
    }
    if train_diff:
        problems["train"] = train_diff

    loaded = load_checkpoint(checkpoint, device=device)
    payload, extra = loaded.payload, loaded.payload.get("extra", {})
    if payload.get("stage") != K - 1 or payload.get("task_key") != curriculum.stages[K - 1].task_key:
        problems["checkpoint_stage"] = (payload.get("stage"), payload.get("task_key"))
    if dict(payload["policy_config"]) != dict(raw_policy_cfg):
        problems["policy_config"] = "the checkpoint's policy config differs from this run's"
    namespace = conf.get("seed_namespace_run_id")
    if extra.get("seed_namespace_run_id") != namespace:
        problems["namespace"] = (extra.get("seed_namespace_run_id"), namespace)
    src_stats = NormalizationStats.load(resume_run / "stats.json")
    if src_stats.fingerprint() != loaded.stats.fingerprint():
        problems["stats"] = (src_stats.fingerprint(), loaded.stats.fingerprint())
    for art in extra.get("method_artifacts", []):
        path = resume_run / art["path"]
        if not path.is_file() or file_sha256(path) != art["sha256"]:
            problems.setdefault("method_artifacts", []).append(art["path"])

    result_path = resume_run / "result.json"
    recorded = json.loads(result_path.read_text())["stages"] if result_path.is_file() else None
    evaluations, stage_dicts = {}, []
    for k in range(K):
        path = resume_run / "eval" / f"stage{k}.json"
        if not path.is_file():
            problems.setdefault("evaluations", []).append(str(path))
            continue
        report = EvaluationReport.load(path)
        if report.stage != k or (report.seed_namespace_run_id not in (None, namespace)):
            problems.setdefault("evaluations", []).append(f"{path}: stage {report.stage}, "
                                                         f"namespace {report.seed_namespace_run_id}")
        evaluations[k] = report
        if recorded is not None and k < len(recorded):
            entry = dict(recorded[k])
            if entry.get("task_key") != curriculum.stages[k].task_key:
                problems.setdefault("result_stages", []).append(k)
        else:  # a run that stopped before writing result.json
            entry = {"stage": k, "task_key": curriculum.stages[k].task_key,
                     "n_demos": curriculum.stages[k].n_demos,
                     "checkpoint": str(resume_run / "checkpoints" / f"stage{k}.pt"),
                     "evaluation": report.as_dict()}
        entry["imported_from"] = str(resume_run)
        stage_dicts.append(entry)
    if problems:
        raise ValueError(f"cannot resume {resume_run} at stage {K}: {problems}")
    return ResumeSource(run=resume_run, start_stage=K, checkpoint=checkpoint, sha256=sha,
                        loaded=loaded, namespace=namespace, evaluations=evaluations,
                        stage_dicts=stage_dicts)


def _stage_identity(record: dict) -> tuple:
    """A recorded stage as resume compares it; configs written before co-training default."""
    return (record["task_key"], record["n_demos"], tuple(record.get("co_train") or ()),
            int(record.get("steps_factor", 1)))


def warm_runner_text_cache(policy, curriculum: Curriculum, dataset_dir, device, amp: bool) -> dict:
    """Recreate the runner's text-encoder cache at a stage boundary (for a resumed run).

    The frozen text encoder caches each instruction in the precision active when it is first
    encoded (:class:`flowcl.models.encoders.CachedTextEncoder`). In the runner, T1's instruction
    is first encoded inside T1's (AMP, if ``amp``) training step, and a joint stage 0's every
    train instruction by :func:`warm_joint_stage_cache` before it trains (AMP too); every other
    task's is first encoded in fp32 by the evaluation after stage 0, which covers every task. The
    instructions are the demo files' own (:func:`flowcl.data.libero_adapter.read_task_metadata`),
    one string per call, as :func:`flowcl.experiments.low_update.warm_runner_cache` does.
    """
    from flowcl.data.tasks import recorded_language

    device_type = torch.device(device).type
    first = set(r.task_key for r in curriculum.stages[0].train_refs)
    order = [*curriculum.stages[0].train_refs,
             *(r for r in curriculum.eval_refs if r.task_key not in first)]
    state = {}
    for ref in order:
        text = recorded_language(ref, dataset_dir)
        task_amp = bool(amp and ref.task_key in first and device_type == "cuda")
        with torch.no_grad(), torch.autocast(device_type=device_type, enabled=task_amp):
            policy.text_encoder([text])
        state[ref.task_key] = {"text": text, "precision": "amp" if task_amp else "fp32"}
    return state


def warm_joint_stage_cache(policy, stage, dataset_dir, device, amp: bool) -> dict:
    """Encode a joint stage's train instructions before it trains, one string per call.

    Without this the first training step would encode the not-yet-cached instructions of a
    joint stage in one batched call, in the order the first batch happens to hold them, under
    AMP: a cache state no later process could rebuild exactly. Instructions already cached
    (encoded by an earlier evaluation) are left as they are. Returns ``{task_key: {"text",
    "precision"}}`` for the instructions this call encoded.
    """
    from flowcl.data.tasks import recorded_language

    device_type = torch.device(device).type
    task_amp = bool(amp and device_type == "cuda")
    encoded = {}
    for ref in stage.train_refs:
        text = recorded_language(ref, dataset_dir)
        if text in policy.text_encoder._cache:
            continue
        with torch.no_grad(), torch.autocast(device_type=device_type, enabled=task_amp):
            policy.text_encoder([text])
        encoded[ref.task_key] = {"text": text, "precision": "amp" if task_amp else "fp32"}
    return encoded


def stage_train_config(train_cfg: TrainConfig, stage) -> TrainConfig:
    """The stage's recipe: the run's, with ``steps`` multiplied by the stage's ``steps_factor``."""
    if stage.steps_factor == 1:
        return train_cfg
    return dataclasses.replace(train_cfg, steps=int(train_cfg.steps) * int(stage.steps_factor))


def task_samples(dataset) -> dict[str, int]:
    """Training samples (timesteps) per task in a dataset: its exposure under uniform sampling."""
    out: dict[str, int] = {}
    for episode in dataset.episodes:
        out[episode.task_id] = out.get(episode.task_id, 0) + int(episode.length)
    return out


def method_label(method, method_name: str) -> str:
    """The variant-specific name used in run ids and reports (e.g. ``gpm_projected_adam``)."""
    return getattr(method, "display_name", None) or method_name


def stage_identity_check(checkpoint: Path, reference_run: Path, stage: int) -> dict:
    """Every state-dict tensor of this stage must equal the reference run's, bit for bit.

    For a variant that is identical to a reference run up to some stage by construction
    (e.g. the adaptive GPM target equals eps at T1, so stages 0-1 match the plain GPM
    run). Unlike :func:`t1_pairing_check` this is exact: training here is deterministic
    (a crashed and rerun GPM run reproduced its checkpoints bitwise). Method-agnostic.
    """
    ref_path = Path(reference_run) / "checkpoints" / f"stage{stage}.pt"
    if not ref_path.is_file():
        raise FileNotFoundError(f"identity reference checkpoint missing: {ref_path}")
    mine = torch.load(checkpoint, map_location="cpu", weights_only=False)["state_dict"]
    ref = torch.load(ref_path, map_location="cpu", weights_only=False)["state_dict"]
    shared = sorted(set(mine) & set(ref))
    different = [n for n in shared if not torch.equal(mine[n], ref[n])]
    missing, extra = sorted(set(ref) - set(mine)), sorted(set(mine) - set(ref))
    return {
        "stage": stage,
        "reference_run": str(reference_run),
        "reference_checkpoint": str(ref_path),
        "reference_sha256": file_sha256(ref_path),
        "n_tensors": len(shared),
        "n_different": len(different),
        "different": different,
        "missing": missing,
        "extra": extra,
        "passed": not (different or missing or extra),
    }


def t1_pairing_check(
    checkpoint: Path,
    reference_run: Path,
    train_log: TrainLog,
    policy,
    max_rel_diff: float = T1_PAIRING_MAX_REL_DIFF,
) -> dict:
    """Compare this run's stage-0 model with the reference run's (fail-fast pairing guard).

    Methods whose Task 1 is unconstrained train it on the reference's exact stream from
    the same initialisation, so the two stage-0 models should nearly coincide. Bitwise
    equality is not expected (GPU nondeterminism over 30 000 steps), and the reference's
    per-step losses were never stored, so this compares final trainable weights and the
    final / last-50 losses. ``max_rel_diff`` is a judgement that catches a broken pairing
    (wrong stream or init diverges by O(0.1-1)), not a calibrated bound; the measured
    numbers are recorded whatever they are.
    """
    reference_run = Path(reference_run)
    ref_payload = torch.load(
        reference_run / "checkpoints" / "stage0.pt", map_location="cpu", weights_only=False
    )
    mine = torch.load(checkpoint, map_location="cpu", weights_only=False)["state_dict"]
    ref = ref_payload["state_dict"]
    groups = {f"{e.name}.weight": e.group for e in policy.projectable_layers()}
    trainable = [n for n, p in policy.named_parameters() if p.requires_grad]
    num: dict[str, float] = {}
    den: dict[str, float] = {}
    for name in trainable:
        if name not in ref:
            raise KeyError(f"reference stage-0 checkpoint lacks {name}")
        a, b = mine[name].to(torch.float64), ref[name].to(torch.float64)
        group = groups.get(name, "other")
        num[group] = num.get(group, 0.0) + float(((a - b) ** 2).sum())
        den[group] = den.get(group, 0.0) + float((b**2).sum())
    if sum(den.values()) == 0.0:
        raise ValueError("reference stage-0 trainable weights are all zero; nothing to compare")
    overall = (sum(num.values()) / sum(den.values())) ** 0.5
    ref_stage = json.loads((reference_run / "result.json").read_text())["stages"][0]
    return {
        "reference_run": str(reference_run),
        "rel_weight_diff": overall,
        # A group whose reference norm is zero has no relative difference; its absolute
        # difference is reported instead of a made-up ratio.
        "rel_weight_diff_by_group": {
            g: ((num[g] / den[g]) ** 0.5 if den[g] > 0 else None) for g in num
        },
        "abs_weight_diff_by_group": {g: num[g] ** 0.5 for g in num},
        "final_loss": {"this": train_log.final_loss, "reference": ref_stage["final_loss"]},
        "mean_last_50_loss": {
            "this": train_log.mean_last(50),
            "reference": ref_stage["mean_last_50_loss"],
        },
        "max_rel_diff": max_rel_diff,
        "passed": overall <= max_rel_diff,
    }


def run_continual(
    curriculum: Curriculum,
    method_name: str,
    spec: EmbodimentSpec,
    policy_config: str | Path | dict,
    train_cfg: TrainConfig,
    eval_cfg: EvalConfig,
    method_kwargs: dict | None = None,
    seed: int = 0,
    bootstrap: dict | None = None,
    dataset_dir: Path | None = None,
    results_root: Path | None = None,
    pretrained: bool = True,
    exist_ok: bool = True,
    evaluate: bool = True,
    single_task_baseline: bool = False,
    require_clean_tree: bool = False,
    t1_reference_run: Path | None = None,
    t1_pairing_max_rel_diff: float = T1_PAIRING_MAX_REL_DIFF,
    identity_reference_run: Path | None = None,
    identity_stages: tuple[int, ...] = (),
    resume_run: Path | None = None,
    start_stage: int = 0,
    resume_sha256: str | None = None,
) -> ContinualResult:
    """Train one policy through ``curriculum`` under ``method_name``.

    Args:
        curriculum: Ordered task sequence (§5).
        method_name: Key into :data:`flowcl.methods.base.METHOD_REGISTRY`.
        method_kwargs: Method hyperparameters, from ``configs/method/<name>.yaml``.
        seed: Resolved seed. Note it does **not** feed the evaluation initial states,
            which derive from ``(run_id, task_key, episode_idx)`` (§8.3).
        evaluate: Set False to exercise the training path without a GL context. The
            retention matrix is then empty and no metric can be computed, which is
            correct: there is no such thing as a rollout-free success rate.
        single_task_baseline: Load the Gate 0 single-task references and write FWT
            (raises if they are missing).
        require_clean_tree: Refuse to start on a ``-dirty`` git tree (final Stage A
            runs need clean, committed provenance).
        t1_reference_run: Run directory to compare the stage-0 model against
            (:func:`t1_pairing_check`); the run stops there if the check fails.
        identity_reference_run: Run directory whose checkpoints this run must equal
            exactly at ``identity_stages`` (:func:`stage_identity_check`); checked right
            after each listed stage's checkpoint, before evaluation, and the run stops
            on a mismatch.
        resume_run, start_stage: Resume at a stage boundary: start from
            ``resume_run/checkpoints/stage{start_stage-1}.pt`` (verified by
            :func:`load_resume_source`, against ``resume_sha256`` if given), restore the
            method's state from that checkpoint's artifacts
            (``method.restore_state``), recreate the runner's text cache
            (:func:`warm_runner_text_cache`), import the earlier stages' evaluations, and
            train stages ``start_stage..``. The run keeps the source's seed namespace and
            frozen stats. It recovers stage boundaries only: a stage that crashed is rerun.

    Seeding: every stream derives from :func:`seed_namespace_run_id`, not from this
    run's id; both ids are recorded in the config, checkpoints, eval files and result.

    Per-stage order (a checkpoint never references an artifact that does not exist)::

        train (ends with on_task_end) -> save_artifacts -> sha256 -> save_checkpoint
        -> [stage 0: T1 pairing check] -> [identity check] -> evaluate

    Returns:
        A :class:`ContinualResult` whose artifacts are already on disk.
    """
    from flowcl.methods.base import build_method

    from flowcl.methods.base import TaskContext

    if require_clean_tree:
        sha = git_sha()
        if sha.endswith("-dirty"):
            raise RuntimeError(
                f"working tree is dirty ({sha}); final runs need clean, committed "
                "provenance. Commit first, or pass --allow-dirty (recorded)."
            )

    method = build_method(method_name, **(method_kwargs or {}))
    label = method_label(method, method_name)
    run_id = continual_run_id(label, curriculum.name, seed)
    seed_ns = seed_namespace_run_id(curriculum.name, seed)
    raw_policy_cfg = load_policy_config(policy_config)

    source = None
    if resume_run is not None:
        from flowcl.utils.libero_paths import repo_root

        target = (Path(results_root) if results_root else repo_root() / "results") / run_id
        if target.resolve() == Path(resume_run).resolve():
            raise ValueError(f"a resumed run must not write into its source ({resume_run})")
        source = load_resume_source(resume_run, start_stage, curriculum, method_name, method_kwargs,
                                    raw_policy_cfg, train_cfg, seed, expected_sha256=resume_sha256,
                                    device=train_cfg.device)
        seed_ns = source.namespace  # the source's pairing, also under an extended curriculum
    elif start_stage:
        raise ValueError("start_stage needs resume_run")

    run = create_run(
        run_id=run_id,
        cfg={
            "run_id": run_id,
            "method_run_id": run_id,
            "seed_namespace_run_id": seed_ns,
            "seed": seed,
            "method": {"name": method_name, "display_name": label, **(method_kwargs or {})},
            "provenance": {
                "require_clean_tree": require_clean_tree,
                "t1_reference_run": str(t1_reference_run) if t1_reference_run else None,
                "t1_pairing_max_rel_diff": t1_pairing_max_rel_diff,
                "single_task_baseline": single_task_baseline,
                "identity_reference_run": (
                    str(identity_reference_run) if identity_reference_run else None
                ),
                "identity_stages": list(identity_stages),
                "resume": source.record() if source else None,
            },
            "curriculum": {
                "name": curriculum.name,
                "tasks": [s.record() for s in curriculum.stages],
                "description": curriculum.description,
                "expectation": curriculum.expectation,
            },
            "embodiment": spec.to_dict(),
            "policy": raw_policy_cfg,
            "train": {k: v for k, v in vars(train_cfg).items()},
            "eval": {k: v for k, v in vars(eval_cfg).items()},
        },
        seed=seed,
        results_root=results_root,
        exist_ok=exist_ok,
    )

    curriculum.assert_consistent(dataset_dir)
    torch.manual_seed(seed)

    if source is None:
        # §3.3: fit once on task 1, then freeze for the whole curriculum.
        first = curriculum.stages[0]
        stats = fit_stats(
            first.ref, spec, n_demos=first.n_demos, dataset_dir=dataset_dir
        )
    else:
        stats = source.loaded.stats  # the source's frozen task-1 stats, verified above
    stats.save(run.artifact("stats.json"))
    stats_fingerprint = stats.fingerprint()
    print(
        f"[flowcl] {run_id}: stats fitted on {stats.fitted_on_task_id} "
        f"(fingerprint {stats_fingerprint})",
        flush=True,
    )

    resume_record = None
    if source is None:
        policy = build_policy(raw_policy_cfg, spec, pretrained=pretrained)
    else:
        policy = source.loaded.policy
        restored = method.restore_state(source.loaded.payload.get("extra", {}), source.run,
                                        source.start_stage - 1)
        cache = warm_runner_text_cache(policy, curriculum, dataset_dir, train_cfg.device,
                                       amp=train_cfg.amp)
        resume_record = {**source.record(), "method_state": restored, "text_cache": cache}
        atomic_write_text(run.artifact("resume.json"), json.dumps(resume_record, indent=2) + "\n")
        print(f"[flowcl] {run_id}: resumed from {source.checkpoint} (sha256 {source.sha256[:12]}), "
              f"training stages {source.start_stage}..{len(curriculum.stages) - 1}", flush=True)
    policy.to(torch.device(train_cfg.device))
    parameter_report = policy.parameter_report()

    matrix = RetentionMatrix.empty(curriculum.task_keys)
    result = ContinualResult(
        run_id=run_id,
        method=label,
        method_registry_name=method_name,
        seed_namespace_run_id=seed_ns,
        curriculum=curriculum.name,
        seed=seed,
        task_keys=curriculum.task_keys,
        matrix=matrix,
        run=run,
        resume=resume_record,
    )

    if source is not None:
        for k, report in sorted(source.evaluations.items()):
            shutil.copy2(source.run / "eval" / f"stage{k}.json", run.subdir("eval") / f"stage{k}.json")
            for task_position, task_key in enumerate(curriculum.task_keys):
                entry = report.by_task().get(task_key)
                if entry is not None:
                    matrix.set(k, task_position, entry.estimate.value, entry.n_rollouts)
        result.imported_stages = source.stage_dicts

    first_stage = source.start_stage if source is not None else 0
    total_started = time.perf_counter()
    for stage_idx, stage in enumerate(curriculum.stages):
        if stage_idx < first_stage:
            continue
        # The §3.3 stage-boundary assertion. Deliberately before the dataset is built,
        # so a method that refitted stats cannot get as far as training on them.
        assert_frozen(
            stats,
            embodiment=spec.name,
            first_task_id=curriculum.first_task_key,
            expected_fingerprint=stats_fingerprint,
        )

        print(
            f"\n[flowcl] === {run_id} stage {stage_idx}: {stage.task_key} ===",
            flush=True,
        )
        dataset = build_dataset(
            list(stage.train_refs),
            spec,
            stats,
            n_demos=stage.n_demos,
            dataset_dir=dataset_dir,
        )
        joint = None
        if stage.is_joint:
            joint = {
                "train_task_keys": [r.task_key for r in stage.train_refs],
                "steps_factor": stage.steps_factor,
                "task_samples": task_samples(dataset),
                "text_cache": warm_joint_stage_cache(
                    policy, stage, dataset_dir, train_cfg.device, amp=train_cfg.amp
                ),
            }
            print(f"[flowcl] joint stage {stage_idx}: {joint}", flush=True)

        context = TaskContext(
            task_key=stage.task_key,
            dataset=dataset,
            device=train_cfg.device,
            seed_namespace_run_id=seed_ns,
            method_run_id=run_id,
        )
        generator = torch.Generator(device="cpu").manual_seed(
            derive_seed(seed_ns, stage.task_key, stage_idx)
        )
        train_log = train_one_task(
            policy,
            dataset,
            stage_train_config(train_cfg, stage),
            method=method,
            task_idx=stage_idx,
            generator=generator,
            context=context,
        )

        # Artifacts first (written atomically), hashed from disk, then referenced by the
        # checkpoint: a checkpoint never names an artifact that does not yet exist.
        artifact_paths = method.save_artifacts(
            run.subdir("method"), stage_idx, context=context
        )
        method_artifacts = [
            {"path": str(Path(p).relative_to(run.path)), "sha256": file_sha256(p)}
            for p in artifact_paths
        ]

        checkpoint = save_checkpoint(
            run.subdir("checkpoints") / f"stage{stage_idx}.pt",
            policy=policy,
            policy_config=raw_policy_cfg,
            spec=spec,
            stats=stats,
            run_id=run_id,
            stage=stage_idx,
            task_key=stage.task_key,
            extra={
                "method": label,
                "method_registry_name": method_name,
                "method_run_id": run_id,
                "seed_namespace_run_id": seed_ns,
                "curriculum": curriculum.name,
                "method_state": method.state_dict(),
                "method_artifacts": method_artifacts,
                "final_loss": train_log.final_loss,
                **({"joint": joint} if joint is not None else {}),
            },
        )

        if stage_idx == 0 and t1_reference_run is not None:
            check = t1_pairing_check(
                checkpoint, t1_reference_run, train_log, policy, t1_pairing_max_rel_diff
            )
            result.t1_pairing = check
            run.artifact("t1_pairing.json").write_text(json.dumps(check, indent=2) + "\n")
            print(
                f"[flowcl] T1 pairing vs {t1_reference_run}: relative weight difference "
                f"{check['rel_weight_diff']:.4f} (max {t1_pairing_max_rel_diff})",
                flush=True,
            )
            if not check["passed"]:
                raise RuntimeError(
                    f"T1 pairing check failed: relative weight difference "
                    f"{check['rel_weight_diff']:.4f} > {t1_pairing_max_rel_diff}. The run's "
                    "Task-1 stream or initialisation does not match the reference; stopping "
                    "before 5 h of dependent work (details in t1_pairing.json)."
                )

        if identity_reference_run is not None and stage_idx in identity_stages:
            check = stage_identity_check(checkpoint, identity_reference_run, stage_idx)
            result.identity_checks[str(stage_idx)] = check
            atomic_write_text(
                run.artifact(f"identity_stage{stage_idx}.json"), json.dumps(check, indent=2) + "\n"
            )
            print(
                f"[flowcl] identity vs {identity_reference_run} stage {stage_idx}: "
                f"{check['n_different']} of {check['n_tensors']} tensors differ",
                flush=True,
            )
            if not check["passed"]:
                raise RuntimeError(
                    f"identity check failed at stage {stage_idx}: {check['n_different']} "
                    f"tensors differ (missing {len(check['missing'])}, extra "
                    f"{len(check['extra'])}) from {identity_reference_run}. The run is not "
                    "the paired variant it claims to be; stopping before evaluation "
                    f"(details in identity_stage{stage_idx}.json)."
                )

        # §8.2: evaluate on every task, including the unseen ones, or FWT is lost.
        evaluation = (
            evaluate_tasks(
                policy,
                curriculum.eval_refs,
                spec,
                stats,
                run_id=seed_ns,  # the rollout seed namespace, shared by every method
                cfg=eval_cfg,
                bootstrap=bootstrap,
                stage=stage_idx,
            )
            if evaluate
            else EvaluationReport(run_id=seed_ns, stage=stage_idx)
        )
        evaluation.method_run_id = run_id
        evaluation.seed_namespace_run_id = seed_ns
        evaluation.save(run.subdir("eval") / f"stage{stage_idx}.json")

        for task_position, task_key in enumerate(curriculum.task_keys):
            entry = evaluation.by_task().get(task_key)
            if entry is not None:
                matrix.set(
                    stage_idx, task_position, entry.estimate.value, entry.n_rollouts
                )

        result.stages.append(
            StageRecord(
                stage=stage_idx,
                task_key=stage.task_key,
                n_demos=stage.n_demos,
                dataset_size=len(dataset),
                train_log=train_log,
                checkpoint=checkpoint,
                evaluation=evaluation,
                method_artifacts=method_artifacts,
                joint=joint,
            )
        )
        # Episodes hold every demo's pixels; releasing the stage's dataset keeps peak
        # host memory proportional to one task rather than the whole curriculum.
        del dataset

    result.systems = {
        "trainable_params": parameter_report["trainable"],
        "frozen_params": parameter_report["frozen"],
        "registry_layers": parameter_report["registry_layers"],
        "method_stored_mb": getattr(method, "stored_bytes", lambda: 0)() / 1e6,
        "is_exemplar_free": getattr(method, "is_exemplar_free", True),
        "total_wall_clock_s": time.perf_counter() - total_started,
        "train_wall_clock_s": sum(r.train_log.wall_clock_s for r in result.stages),
        "resumed_at_stage": first_stage if source is not None else None,
    }

    if evaluate:
        baseline = (
            baseline_from_single_task_runs(curriculum.task_keys, seed, results_root)
            if single_task_baseline
            else None
        )
        result.save(run.artifact("result.json"), baseline=baseline)
        print(
            f"\n[flowcl] {run_id} F_1 = "
            f"{result.summary().final_average_success:.3f}",
            flush=True,
        )
    return result


def baseline_from_single_task_runs(
    task_keys: tuple[str, ...],
    seed: int = 0,
    results_root: Path | None = None,
) -> dict[str, float]:
    """Collect the §10.4 independent single-task success rates, for FWT.

    Reads the ``eval.json`` that :func:`flowcl.experiments.gate0.run_gate0` wrote, so
    FWT reuses Gate 0's runs instead of retraining them.
    """
    from flowcl.experiments.gate0 import single_task_run_id
    from flowcl.utils.libero_paths import repo_root

    root = Path(results_root) if results_root else (repo_root() / "results")
    baseline = {}
    for task_key in task_keys:
        path = root / single_task_run_id(task_key, seed) / "eval.json"
        if not path.is_file():
            raise FileNotFoundError(
                f"No single-task reference for {task_key} at {path}. FWT needs the "
                "§10.4 independent references; run scripts/gate0.py first."
            )
        report = EvaluationReport.load(path)
        baseline[task_key] = report.by_task()[task_key].estimate.value
    return baseline
