"""GPM feasibility pilot: does hard projection let the policy learn T2 while keeping T1?

Gate 3 (``docs/runs/2026-09-22_gate3.md``) passed but showed that hard projection at
eps = 0.95 would remove ~87% of Object's gradient energy. Whether the policy can still learn
Object is a training question. This pilot answers it for one pair of tasks before the
rest of the build order (§10 step 7) — it is a pilot, not the Stage A comparison.

Starting from the Gate 1 ``seq_ft`` stage-0 checkpoint (Spatial), each arm trains on
Object for the Gate 1 recipe (read from that run's own ``config.yaml``):

* ``gpm_projected_adam`` — registry weights only, gradient *and* AdamW's realised update
  projected against the fixed Task-1 memory (:mod:`flowcl.methods.gpm`);
* ``freeze_only`` — the same trainable set, no projection, to separate the effect of the
  §7.4 freezing from the effect of projection.

Pairing: both arms train on seq_ft stage 1's exact data/``s``/``A_0`` stream
(``derive_seed(seq_ft_run_id, object_key, 1)``), a fresh AdamW per stage as in Gate 1,
and are rolled out under **seq_ft's run_id as the seed namespace** — every episode has
the same initial state and sampling seed as seq_ft's own stage evaluations and as the
other arm. Provenance keeps ``method_run_id`` and ``evaluation_seed_run_id`` apart.

Pre-registered outcome (:func:`classify`): Object >= R[1][1] − 15 pp and Spatial >=
R[0][0] − 15 pp on the point estimate, references read from the Gate 1 artifacts.
"""

from __future__ import annotations

import dataclasses
import json
import math
import statistics
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import torch
from omegaconf import OmegaConf

from flowcl.analysis.metrics import Estimate, paired_difference_ci
from flowcl.analysis.subspace import load_bases
from flowcl.data.curriculum import Curriculum
from flowcl.envs.evaluation import EvaluationReport, eval_config_from_dict, evaluate_tasks
from flowcl.experiments.gate3 import check_provenance, update_interference
from flowcl.methods.gpm import GPM, allowlist, freeze_to_allowlist
from flowcl.methods.seq_ft import SeqFT
from flowcl.models.flow_head import draw_with_generator
from flowcl.models.losses import valid_element_count
from flowcl.train.checkpoint import load_checkpoint, save_checkpoint
from flowcl.train.trainer import TrainConfig, build_dataloader, move_batch, train_one_task
from flowcl.utils.libero_paths import repo_root
from flowcl.utils.run import create_run, git_sha
from flowcl.utils.seeding import derive_seed

ARMS = ("gpm_projected_adam", "freeze_only")


# ---- configuration -------------------------------------------------------------


@dataclass
class PilotConfig:
    """``configs/analysis/gpm_pilot.yaml``; see the comments there."""

    arms: tuple[str, ...]
    sanity_steps: int
    freeze_only_timing_steps: int
    timing_warmup_steps: int
    criteria: dict
    probe: dict
    eval_tasks: tuple[int, ...]

    def __post_init__(self) -> None:
        self.arms = tuple(self.arms)
        unknown = sorted(set(self.arms) - set(ARMS))
        if unknown:
            raise ValueError(f"unknown arms {unknown}; available {ARMS}")
        self.eval_tasks = tuple(self.eval_tasks)
        if set(self.criteria) != {"margin_pp", "expected_object_min", "expected_spatial_min"}:
            raise ValueError(f"criteria keys wrong: {sorted(self.criteria)}")
        if set(self.probe) != {"batch_size", "n_batches", "seed_tags"}:
            raise ValueError(f"probe keys wrong: {sorted(self.probe)}")
        if set(self.probe["seed_tags"]) != {"shuffle", "flow_time", "noise"}:
            raise ValueError(f"probe seed_tags wrong: {sorted(self.probe['seed_tags'])}")
        for steps in (self.sanity_steps, self.freeze_only_timing_steps):
            if steps <= self.timing_warmup_steps:
                raise ValueError(
                    f"timed runs need more than timing_warmup_steps={self.timing_warmup_steps}"
                )

    @classmethod
    def from_dict(cls, payload: dict) -> "PilotConfig":
        known = set(cls.__dataclass_fields__)
        unknown, missing = sorted(set(payload) - known), sorted(known - set(payload))
        if unknown or missing:
            raise ValueError(f"pilot config: unknown keys {unknown}, missing keys {missing}")
        return cls(**payload)


def load_pilot_config(path: str | Path | None = None) -> PilotConfig:
    path = Path(path) if path else repo_root() / "configs" / "analysis" / "gpm_pilot.yaml"
    return PilotConfig.from_dict(OmegaConf.to_container(OmegaConf.load(path), resolve=True))


def load_method_config(path: str | Path | None = None) -> dict:
    path = Path(path) if path else repo_root() / "configs" / "method" / "gpm.yaml"
    payload = OmegaConf.to_container(OmegaConf.load(path), resolve=True)
    if payload.pop("name", None) != "gpm":
        raise ValueError(f"{path} is not a gpm method config")
    # The pilot ran with a fixed Task-1 memory. configs/method/gpm.yaml now describes the
    # four-task sequence (update_memory: true); forcing false keeps the pilot equivalent
    # to what ran, and the override is recorded in the pilot's method_config.
    payload["update_memory"] = False
    return payload


@dataclass(frozen=True)
class PilotInputs:
    curriculum: Curriculum
    seq_run_dir: Path
    stage0: Path
    stage1: Path
    t1_bases: Path
    seed: int

    @property
    def seq_run_id(self) -> str:
        return self.seq_run_dir.name


def default_inputs(
    curriculum: Curriculum, seed: int = 0, results_root: Path | None = None
) -> PilotInputs:
    from flowcl.experiments.gate2 import bases_path
    from flowcl.train.continual import continual_run_id

    root = Path(results_root) if results_root else repo_root() / "results"
    run_id = continual_run_id("seq_ft", curriculum.name, seed)
    run_dir = root / run_id
    return PilotInputs(
        curriculum=curriculum,
        seq_run_dir=run_dir,
        stage0=run_dir / "checkpoints" / "stage0.pt",
        stage1=run_dir / "checkpoints" / "stage1.pt",
        t1_bases=bases_path(run_id, 0, root),
        seed=seed,
    )


def load_recipe(seq_run_dir: Path) -> tuple[TrainConfig, dict]:
    """The Gate 1 train/eval recipe, read from that run's own ``config.yaml``."""
    cfg = OmegaConf.to_container(OmegaConf.load(Path(seq_run_dir) / "config.yaml"), resolve=True)
    train = TrainConfig(**cfg["train"])
    return train, dict(cfg["eval"])


def pilot_run_id(arm: str, curriculum: str, seed: int, sanity: bool = False) -> str:
    return f"{curriculum}__{arm}_pilot{'_sanity' if sanity else ''}__seed{seed}"


# ---- pre-registered criteria ---------------------------------------------------


def criteria_thresholds(
    object_reference: float, spatial_reference: float, criteria: dict
) -> dict:
    """``reference − margin`` for each task, asserted against the pre-registered values."""
    margin = criteria["margin_pp"] / 100.0
    object_min = round(object_reference - margin, 6)
    spatial_min = round(spatial_reference - margin, 6)
    expected = (criteria["expected_object_min"], criteria["expected_spatial_min"])
    if not (math.isclose(object_min, expected[0]) and math.isclose(spatial_min, expected[1])):
        raise ValueError(
            f"thresholds from the Gate 1 artifacts ({object_min}, {spatial_min}) differ from "
            f"the pre-registered ({expected[0]}, {expected[1]}); the references changed"
        )
    return {
        "margin_pp": criteria["margin_pp"],
        "object_reference": object_reference,
        "spatial_reference": spatial_reference,
        "object_min": object_min,
        "spatial_min": spatial_min,
    }


OUTCOMES = {
    (True, True): "hard projection is viable",
    (False, True): "protection works, plasticity fails",
    (True, False): "plasticity works, protection fails",
    (False, False): "method or optimization failure",
}


def classify(object_estimate: Estimate, spatial_estimate: Estimate, thresholds: dict) -> dict:
    """The pre-registered four-way outcome. Point estimates decide; CIs flag borderline."""
    obj_ok = object_estimate.value >= thresholds["object_min"]
    spa_ok = spatial_estimate.value >= thresholds["spatial_min"]

    def straddles(est: Estimate, cut: float) -> bool:
        return est.low < cut <= est.high

    return {
        "outcome": OUTCOMES[(obj_ok, spa_ok)],
        "object_acceptable": obj_ok,
        "spatial_acceptable": spa_ok,
        "object_borderline": straddles(object_estimate, thresholds["object_min"]),
        "spatial_borderline": straddles(spatial_estimate, thresholds["spatial_min"]),
    }


def _estimate_dict(est: Estimate) -> dict:
    return {"value": est.value, "low": est.low, "high": est.high, "n": est.n,
            "formatted": est.format_pp()}


def paired(a: list, b: list, bootstrap: dict) -> dict:
    est = paired_difference_ci(
        a, b,
        seed=bootstrap["seed"],
        n_bootstrap=bootstrap["n_resamples"],
        confidence=bootstrap["confidence"],
    )
    return {"diff": est.value, "low": est.low, "high": est.high, "n": est.n}


# ---- measurements --------------------------------------------------------------


def _generators(seed_tags: dict, task_key: str) -> dict[str, torch.Generator]:
    return {
        role: torch.Generator(device="cpu").manual_seed(derive_seed(tag, task_key, 0))
        for role, tag in seed_tags.items()
    }


@torch.no_grad()
def probe_loss(policy, dataset, probe: dict, device) -> float:
    """Masked flow-matching loss over fixed batches, weighted by valid elements.

    Identical batches, ``s`` and ``A_0`` for every checkpoint (streams keyed on the probe
    tags and the data task), fp32, eval mode — a like-for-like stability/plasticity read.
    """
    if len(dataset.task_ids) != 1:
        raise ValueError(f"probe needs a single-task dataset, got {dataset.task_ids}")
    device = torch.device(device)
    gens = _generators(probe["seed_tags"], dataset.task_ids[0])
    loader = build_dataloader(
        dataset, batch_size=probe["batch_size"], num_workers=0, shuffle=True,
        generator=gens["shuffle"],
    )
    policy.eval()
    total, weight = 0.0, 0.0
    for idx, batch in enumerate(loader):
        if idx >= probe["n_batches"]:
            break
        batch = move_batch(batch, device)
        size = batch["actions"].shape[0]
        s = policy.s_sampler.sample(size, device, generator=gens["flow_time"])
        noise = draw_with_generator(
            tuple(batch["actions"].shape), device=device, generator=gens["noise"],
            dtype=torch.float32, normal=True,
        )
        with torch.autocast(device_type=device.type, enabled=False):
            loss = float(policy(batch, s=s, noise=noise)["loss"])
        n_b = float(valid_element_count(batch["action_mask"], policy.d_action))
        total += loss * n_b
        weight += n_b
    if weight == 0:
        raise ValueError("probe saw no valid elements")
    return total / weight


class StepTimer:
    """Steps/s over a synchronised window that excludes the first ``warmup`` steps."""

    def __init__(self, warmup: int, device) -> None:
        self.warmup = warmup
        self.device = torch.device(device)
        self.t0: float | None = None
        self.first = 0
        self.last = 0

    def _sync(self) -> None:
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)

    def __call__(self, step: int, outputs) -> None:
        if step + 1 == self.warmup:
            self._sync()
            self.t0 = time.perf_counter()
            self.first = step + 1
        self.last = step + 1

    def steps_per_second(self) -> float:
        if self.t0 is None or self.last <= self.first:
            raise RuntimeError("timed window is empty; run more steps than the warmup")
        self._sync()
        return (self.last - self.first) / (time.perf_counter() - self.t0)


def _snapshot_frozen(policy) -> dict[str, torch.Tensor]:
    keep = set(allowlist(policy))
    return {n: p.detach().clone() for n, p in policy.named_parameters() if n not in keep}


def _frozen_changes(policy, snapshot: dict[str, torch.Tensor]) -> list[str]:
    params = dict(policy.named_parameters())
    return [n for n, before in snapshot.items() if not torch.equal(before, params[n])]


# ---- one arm -------------------------------------------------------------------


@dataclass
class ArmResult:
    arm: str
    method_run_id: str
    evaluation_seed_run_id: str
    steps: int
    losses: list[float]
    train_wall_clock_s: float
    steps_per_second: float
    freeze_report: dict
    frozen_changed: list[str]
    probes: dict[str, float]
    update: dict
    displacement_norm: dict[str, float]
    method_state: dict
    evaluation: EvaluationReport | None = None
    checkpoint: str | None = None

    def mean_first(self, n: int = 50) -> float:
        return statistics.fmean(self.losses[:n])

    def mean_last(self, n: int = 50) -> float:
        return statistics.fmean(self.losses[-n:])

    def summary(self, eps: float) -> dict:
        out = {
            "arm": self.arm,
            "method_run_id": self.method_run_id,
            "evaluation_seed_run_id": self.evaluation_seed_run_id,
            "checkpoint": self.checkpoint,
            "train": {
                "steps": self.steps,
                "final_loss": self.losses[-1],
                "mean_first_50_loss": self.mean_first(),
                "mean_last_50_loss": self.mean_last(),
                "wall_clock_s": self.train_wall_clock_s,
                "steps_per_second": self.steps_per_second,
                "extrapolated_30000_steps_min": 30000 / self.steps_per_second / 60,
            },
            "freeze": self.freeze_report,
            "frozen_changed": self.frozen_changed,
            "probes": self.probes,
            "update_c_global": self.update["c_global"][str(eps)],
            "action_in_displacement": self.displacement_norm["flow_head.action_in"],
            "registry_layers_not_moved": sorted(
                n for n, v in self.displacement_norm.items() if v == 0.0
            ),
            "method": _method_summary(self.method_state),
        }
        if self.evaluation is not None:
            out["evaluation"] = {
                t.task_key: {**_estimate_dict(t.estimate), "successes": t.successes}
                for t in self.evaluation.tasks
            }
        return out


def _method_summary(state: dict) -> dict:
    """Compact view of the GPM logs (full logs stay in the checkpoint and gpm_logs.json)."""
    if state.get("name") != "gpm":
        return {"name": state.get("name")}

    def half_medians(log: dict) -> dict:
        out = {}
        for step, per_layer in log.items():
            row = {}
            for half, prefix in (("trunk", "trunk."), ("decoder", "flow_head.")):
                vals = [v for n, v in per_layer.items() if n.startswith(prefix) and not math.isnan(v)]
                row[half] = statistics.median(vals) if vals else None
            out[step] = row
        return out

    residuals = state["residuals"]
    worst = max(residuals.items(), key=lambda kv: kv[1]["max_residual_over_bound"]) if residuals else None
    return {
        "display_name": state["display_name"],
        "config": state["config"],
        "stored_mb": state["stored_mb"],
        "memory_extended": state["memory_extended"],
        "gradient_c_median_by_half": half_medians(state["gradient_c"]),
        "update_c_median_by_half": half_medians(state["update_c"]),
        "worst_residual": {"layer": worst[0], **worst[1]} if worst else None,
    }


def run_arm(
    arm: str,
    inputs: PilotInputs,
    datasets: dict,
    method_cfg: dict,
    train_cfg: TrainConfig,
    eval_cfg,
    bootstrap: dict,
    pilot: PilotConfig,
    device: str,
    steps: int,
    evaluate: bool,
    sanity: bool,
    results_root: Path | None = None,
) -> ArmResult:
    """Train one arm from the T1 checkpoint on T2 and measure it."""
    if arm not in ARMS:
        raise ValueError(f"unknown arm {arm!r}")
    curriculum = inputs.curriculum
    spatial_key, object_key = curriculum.task_keys[0], curriculum.task_keys[1]

    loaded = load_checkpoint(inputs.stage0, device=device)
    bases, meta = load_bases(inputs.t1_bases)
    eps = method_cfg["eps"]
    check_provenance(loaded, bases, meta, [eps])
    policy = loaded.policy
    stage0_state = loaded.payload["state_dict"]

    frozen_before = _snapshot_frozen(policy)
    freeze_report = freeze_to_allowlist(policy)
    print(
        f"[flowcl] {arm}: trainable {freeze_report['trainable_params']:,} params in "
        f"{freeze_report['trainable_tensors']} registry weights; froze "
        f"{freeze_report['newly_frozen_params']:,} params "
        f"({len(freeze_report['newly_frozen'])} tensors)",
        flush=True,
    )

    method_run_id = pilot_run_id(arm, curriculum.name, inputs.seed, sanity)
    run = create_run(
        run_id=method_run_id,
        cfg={
            "run_id": method_run_id,
            "arm": arm,
            "pilot": "gpm_feasibility",
            "evaluation_seed_run_id": inputs.seq_run_id,
            "start_checkpoint": str(inputs.stage0),
            "t1_bases": str(inputs.t1_bases),
            "task_key": object_key,
            "method": {"name": "gpm", **method_cfg} if arm == "gpm_projected_adam" else {"name": "seq_ft"},
            "train": {**vars(train_cfg), "steps": steps},
            "freeze": {k: v for k, v in freeze_report.items() if k != "newly_frozen"},
        },
        seed=inputs.seed,
        results_root=results_root,
        exist_ok=sanity,
    )

    if arm == "gpm_projected_adam":
        method = GPM(**method_cfg)
        method.set_memory(bases)
    else:
        method = SeqFT()

    # seq_ft stage 1's exact stream: same data order, s and A_0 (up to GPU nondeterminism).
    generator = torch.Generator(device="cpu").manual_seed(
        derive_seed(inputs.seq_run_id, object_key, 1)
    )
    timer = StepTimer(pilot.timing_warmup_steps, device)
    cfg = dataclasses.replace(train_cfg, steps=steps, device=device)
    log = train_one_task(
        policy, datasets[object_key], cfg, method=method, task_idx=1,
        generator=generator, on_step=timer,
    )
    rate = timer.steps_per_second()
    bad = [i for i, v in enumerate(log.losses) if not math.isfinite(v)]
    if bad:
        raise RuntimeError(f"{arm}: non-finite training loss at steps {bad[:10]}")
    (run.path / "losses.json").write_text(json.dumps(log.losses) + "\n")

    frozen_changed = _frozen_changes(policy, frozen_before)
    del frozen_before
    state = {k: v.detach().cpu() for k, v in policy.state_dict().items()}
    update = update_interference(stage0_state, state, bases, [eps])
    displacement = {n: v["delta_norm"] for n, v in update["per_layer"].items()}

    method_state = method.state_dict()
    if arm == "gpm_projected_adam":
        (run.path / "gpm_logs.json").write_text(json.dumps(method_state, indent=2) + "\n")
    checkpoint = save_checkpoint(
        run.subdir("checkpoints") / "stage1.pt",
        policy=policy,
        policy_config=loaded.payload["policy_config"],
        spec=loaded.spec,
        stats=loaded.stats,
        run_id=method_run_id,
        stage=1,
        task_key=object_key,
        extra={
            "arm": arm,
            "method_run_id": method_run_id,
            "evaluation_seed_run_id": inputs.seq_run_id,
            "start_checkpoint": str(inputs.stage0),
            "method_state": method_state,
            "freeze_report": freeze_report,
            "final_loss": log.final_loss,
        },
    )

    probes = {
        key: probe_loss(policy, datasets[key], pilot.probe, device)
        for key in (spatial_key, object_key)
    }

    evaluation = None
    if evaluate:
        refs = [curriculum.refs[i] for i in pilot.eval_tasks]
        evaluation = evaluate_tasks(
            policy, refs, loaded.spec, loaded.stats,
            run_id=inputs.seq_run_id,  # the seed namespace, see module docstring
            cfg=eval_cfg, bootstrap=bootstrap, stage=1,
        )
        payload = {
            "method_run_id": method_run_id,
            "evaluation_seed_run_id": inputs.seq_run_id,
            "note": "run_id below is the rollout seed namespace (seq_ft's), not this run",
            **evaluation.as_dict(),
        }
        (run.subdir("eval") / "stage1.json").write_text(json.dumps(payload, indent=2) + "\n")

    result = ArmResult(
        arm=arm,
        method_run_id=method_run_id,
        evaluation_seed_run_id=inputs.seq_run_id,
        steps=steps,
        losses=log.losses,
        train_wall_clock_s=log.wall_clock_s,
        steps_per_second=rate,
        freeze_report=freeze_report,
        frozen_changed=frozen_changed,
        probes=probes,
        update=update,
        displacement_norm=displacement,
        method_state=method_state,
        evaluation=evaluation,
        checkpoint=str(checkpoint),
    )
    (run.path / "pilot.json").write_text(json.dumps(result.summary(eps), indent=2) + "\n")
    del policy, loaded
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return result


def time_arm(arm, inputs, datasets, method_cfg, train_cfg, pilot, device, steps) -> float:
    """Steps/s for a short run of ``arm`` (no registry, no artifacts): overhead timing."""
    loaded = load_checkpoint(inputs.stage0, device=device)
    bases, meta = load_bases(inputs.t1_bases)
    check_provenance(loaded, bases, meta, [method_cfg["eps"]])
    freeze_to_allowlist(loaded.policy)
    if arm == "gpm_projected_adam":
        method = GPM(**method_cfg)
        method.set_memory(bases)
    else:
        method = SeqFT()
    object_key = inputs.curriculum.task_keys[1]
    timer = StepTimer(pilot.timing_warmup_steps, device)
    train_one_task(
        loaded.policy, datasets[object_key],
        dataclasses.replace(train_cfg, steps=steps, device=device, log_every=0),
        method=method, task_idx=1,
        generator=torch.Generator(device="cpu").manual_seed(
            derive_seed(inputs.seq_run_id, object_key, 1)
        ),
        on_step=timer,
    )
    rate = timer.steps_per_second()
    del loaded
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return rate


# ---- the pilot -----------------------------------------------------------------


def _reference_measurements(inputs, datasets, pilot, device, eps, bases) -> dict:
    spatial_key, object_key = inputs.curriculum.task_keys[:2]
    out = {}
    for label, path in (("stage0", inputs.stage0), ("seq_ft_stage1", inputs.stage1)):
        loaded = load_checkpoint(path, device=device)
        out[label] = {
            "checkpoint": str(path),
            "probes": {
                k: probe_loss(loaded.policy, datasets[k], pilot.probe, device)
                for k in (spatial_key, object_key)
            },
        }
        if label == "seq_ft_stage1":
            stage0 = torch.load(inputs.stage0, map_location="cpu", weights_only=False)
            state = {k: v.detach().cpu() for k, v in loaded.policy.state_dict().items()}
            out[label]["update_c_global"] = update_interference(
                stage0["state_dict"], state, bases, [eps]
            )["c_global"][str(eps)]
            del stage0
        del loaded
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    return out


def run_gpm_pilot(
    inputs: PilotInputs,
    pilot: PilotConfig,
    method_cfg: dict,
    device: str = "cuda",
    sanity: bool = False,
    arms: tuple[str, ...] | None = None,
    dataset_dir: Path | None = None,
    results_root: Path | None = None,
    out_dir: Path | None = None,
) -> dict:
    """Run the pilot (or its sanity check) and write the combined report."""
    from flowcl.data.tasks import TaskRef
    from flowcl.train.pipeline import build_dataset

    for path in (inputs.stage0, inputs.stage1, inputs.t1_bases, inputs.seq_run_dir / "config.yaml"):
        if not Path(path).is_file():
            raise FileNotFoundError(f"pilot input missing: {path} (run Gates 1 and 2 first)")

    train_cfg, eval_payload = load_recipe(inputs.seq_run_dir)
    eval_cfg = eval_config_from_dict(eval_payload)
    bootstrap = OmegaConf.to_container(
        OmegaConf.load(repo_root() / "configs" / "eval" / "libero_eval.yaml"), resolve=True
    )["bootstrap"]
    eps = method_cfg["eps"]
    spatial_key, object_key = inputs.curriculum.task_keys[:2]

    eval0 = EvaluationReport.load(inputs.seq_run_dir / "eval" / "stage0.json").by_task()
    eval1 = EvaluationReport.load(inputs.seq_run_dir / "eval" / "stage1.json").by_task()
    thresholds = criteria_thresholds(
        eval1[object_key].estimate.value, eval0[spatial_key].estimate.value, pilot.criteria
    )

    stats_ckpt = load_checkpoint(inputs.stage0, device="cpu")
    datasets = {
        key: build_dataset([TaskRef.from_key(key)], stats_ckpt.spec, stats_ckpt.stats,
                           dataset_dir=dataset_dir)
        for key in (spatial_key, object_key)
    }
    del stats_ckpt
    bases, _ = load_bases(inputs.t1_bases)
    references = _reference_measurements(inputs, datasets, pilot, device, eps, bases)
    ref_log = json.loads((inputs.seq_run_dir / "result.json").read_text())["stages"][1]
    references["seq_ft_stage1"]["train"] = {
        "final_loss": ref_log["final_loss"],
        "mean_last_50_loss": ref_log["mean_last_50_loss"],
        "wall_clock_s": ref_log["train_wall_clock_s"],
        "steps_per_second": ref_log["steps"] / ref_log["train_wall_clock_s"],
    }
    references["seq_ft_stage1"]["evaluation"] = {
        spatial_key: _estimate_dict(eval1[spatial_key].estimate),
        object_key: _estimate_dict(eval1[object_key].estimate),
    }
    references["stage0"]["evaluation"] = {spatial_key: _estimate_dict(eval0[spatial_key].estimate)}

    report: dict = {
        "pilot": "gpm_feasibility" + ("_sanity" if sanity else ""),
        "recorded": datetime.now(timezone.utc).isoformat(),
        "analysis_git_sha": git_sha(),
        "evaluation_seed_run_id": inputs.seq_run_id,
        "inputs": {k: str(getattr(inputs, k)) for k in ("stage0", "stage1", "t1_bases")},
        "recipe": {**vars(train_cfg)},
        "method_config": method_cfg,
        "criteria": thresholds,
        "references": references,
        "arms": {},
    }

    results: dict[str, ArmResult] = {}
    if sanity:
        res = run_arm("gpm_projected_adam", inputs, datasets, method_cfg, train_cfg, eval_cfg,
                      bootstrap, pilot, device, pilot.sanity_steps, evaluate=False, sanity=True,
                      results_root=results_root)
        results[res.arm] = res
        freeze_rate = time_arm("freeze_only", inputs, datasets, method_cfg, train_cfg, pilot,
                               device, pilot.freeze_only_timing_steps)
        report["timing"] = {
            "gpm_projected_adam_steps_per_second": res.steps_per_second,
            "freeze_only_steps_per_second": freeze_rate,
            "gate1_seq_ft_steps_per_second": references["seq_ft_stage1"]["train"]["steps_per_second"],
            "extrapolated_30000_steps_min": {
                "gpm_projected_adam": 30000 / res.steps_per_second / 60,
                "freeze_only": 30000 / freeze_rate / 60,
            },
            "projection_overhead": freeze_rate / res.steps_per_second - 1,
        }
        report["sanity_checks"] = sanity_checks(res, references, object_key)
    else:
        for arm in arms or pilot.arms:
            results[arm] = run_arm(arm, inputs, datasets, method_cfg, train_cfg, eval_cfg,
                                   bootstrap, pilot, device, train_cfg.steps, evaluate=True,
                                   sanity=False, results_root=results_root)
        report.update(compare_arms(results, eval0, eval1, spatial_key, object_key,
                                   thresholds, bootstrap))

    report["arms"] = {arm: r.summary(eps) for arm, r in results.items()}
    out_dir = Path(out_dir) if out_dir else repo_root() / "results" / "gpm_pilot"
    out = out_dir / ("sanity.json" if sanity else "pilot.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2) + "\n")
    print_report(report, sanity)
    print(f"[flowcl] wrote {out}", flush=True)
    if sanity:
        failed = [k for k, v in report["sanity_checks"].items() if not v["passed"]]
        if failed:
            raise RuntimeError(f"sanity checks failed: {failed} (details in {out})")
    return report


def sanity_checks(res: ArmResult, references: dict, object_key: str) -> dict:
    """The asserted sanity conditions (the residual bound is enforced inside GPM itself)."""
    full_rank = {n: r["full_rank"] for n, r in res.method_state["residuals"].items()}
    # A layer must move if and only if its basis leaves free directions.
    wrong = sorted(
        n for n, v in res.displacement_norm.items() if (v > 0) == full_rank[n]
    )
    worst = _method_summary(res.method_state)["worst_residual"]
    return {
        "object_loss_decreases": {
            "passed": res.mean_last() < res.mean_first()
            and res.probes[object_key] < references["stage0"]["probes"][object_key],
            "first_50": res.mean_first(),
            "last_50": res.mean_last(),
            "probe_object": res.probes[object_key],
            "probe_object_stage0": references["stage0"]["probes"][object_key],
        },
        "updates_orthogonal": {
            "passed": worst is not None and worst["max_residual_over_bound"] <= 1.0,
            "worst": worst,
        },
        "parameters_move": {
            "passed": not wrong,
            "moved_iff_not_full_rank_violations": wrong,
            "full_rank_layers": sorted(n for n, f in full_rank.items() if f),
        },
        "action_in_frozen": {
            "passed": full_rank["flow_head.action_in"]
            and res.displacement_norm["flow_head.action_in"] == 0.0,
            "displacement": res.displacement_norm["flow_head.action_in"],
        },
        "frozen_untouched": {"passed": not res.frozen_changed, "changed": res.frozen_changed},
        "finite": {"passed": all(math.isfinite(v) for v in res.losses)},
    }


def compare_arms(results, eval0, eval1, spatial_key, object_key, thresholds, bootstrap) -> dict:
    """Pre-registered outcome plus the paired differences, projection effect first."""
    out: dict = {"paired_differences": {}, "outcomes": {}}
    succ = {
        arm: {t.task_key: t.successes for t in r.evaluation.tasks} for arm, r in results.items()
    }
    est = {
        arm: {t.task_key: t.estimate for t in r.evaluation.tasks} for arm, r in results.items()
    }
    if {"gpm_projected_adam", "freeze_only"} <= set(results):
        out["paired_differences"]["gpm_projected_adam_minus_freeze_only"] = {
            key: paired(succ["gpm_projected_adam"][key], succ["freeze_only"][key], bootstrap)
            for key in (spatial_key, object_key)
        }
    for arm in results:
        out["paired_differences"][f"{arm}_minus_seq_ft_stage1"] = {
            key: paired(succ[arm][key], eval1[key].successes, bootstrap)
            for key in (spatial_key, object_key)
        }
        out["paired_differences"][f"{arm}_minus_stage0"] = {
            spatial_key: paired(succ[arm][spatial_key], eval0[spatial_key].successes, bootstrap)
        }
        out["outcomes"][arm] = {
            "decides": arm == "gpm_projected_adam",
            **classify(est[arm][object_key], est[arm][spatial_key], thresholds),
        }
    return out


def print_report(report: dict, sanity: bool) -> None:
    refs = report["references"]
    print("\n[flowcl] GPM pilot" + (" (sanity)" if sanity else ""), flush=True)
    for arm, s in report["arms"].items():
        t = s["train"]
        print(
            f"  {arm}: steps {t['steps']}, loss first-50 {t['mean_first_50_loss']:.5f} -> "
            f"last-50 {t['mean_last_50_loss']:.5f}, {t['steps_per_second']:.2f} steps/s "
            f"(30k ≈ {t['extrapolated_30000_steps_min']:.0f} min)",
            flush=True,
        )
        print(
            "    probes " + ", ".join(f"{k.split('/')[0]} {v:.5f}" for k, v in s["probes"].items())
            + f"; dW c_global {s['update_c_global']}",
            flush=True,
        )
        worst = s["method"].get("worst_residual")
        if worst:
            print(f"    worst residual/bound {worst['max_residual_over_bound']:.3e} ({worst['layer']})")
        if "evaluation" in s:
            print("    rollouts " + ", ".join(
                f"{k.split('/')[0]} {v['formatted']}" for k, v in s["evaluation"].items()))
    for label in ("stage0", "seq_ft_stage1"):
        print(
            f"  reference {label} probes "
            + ", ".join(f"{k.split('/')[0]} {v:.5f}" for k, v in refs[label]["probes"].items()),
            flush=True,
        )
    if sanity:
        print(f"  timing: {report['timing']}", flush=True)
        for name, check in report["sanity_checks"].items():
            print(f"  [{'PASS' if check['passed'] else 'FAIL'}] {name}", flush=True)
    else:
        for arm, o in report["outcomes"].items():
            print(f"  outcome {arm}{' (decides)' if o['decides'] else ''}: {o['outcome']} "
                  f"(borderline object {o['object_borderline']}, spatial {o['spatial_borderline']})")
        for name, diffs in report["paired_differences"].items():
            print(f"  {name}: " + ", ".join(
                f"{k.split('/')[0]} {100 * d['diff']:+.1f} pp [{100 * d['low']:+.1f}, {100 * d['high']:+.1f}]"
                for k, d in diffs.items()))
