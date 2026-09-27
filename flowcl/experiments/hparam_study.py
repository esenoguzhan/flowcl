"""Exploratory training-budget and learning-rate study (single seed).

Every run so far trains 30 000 steps per task (the Gate 0 recipe, never compared with
anything shorter), and the training loss never plateaus — but it is measured on the
training demos, so it cannot show overfitting. Three parts, fixed in
``configs/analysis/hparam_study.yaml`` before anything runs:

1. **trajectory** — one Gate-0-recipe run per task with the last demos held out.
   Checkpoints at registered steps (``train_one_task``'s ``on_step``), each evaluated with
   50 rollouts under Gate 0's episode seeds (paired with Gate 0's 30k result) and two fixed
   :class:`ProbeBank` losses: held-out chunks and an equal-size training subset, each with
   K fixed ``(s, A_0)`` draws. A sustained rise in held-out loss while the training loss
   falls is evidence *consistent with* overfitting; rollout success is the behavioural
   read. Mid-run checkpoints are mid-schedule (the learning rate is still high), so the
   curve is a lower bound on what a dedicated shorter recipe achieves.
2. **cl_budget** — from seq_ft's stage 0, the next task trained seq_ft-style for shorter
   budgets (each with its own warmup + cosine schedule), on seq_ft's stage data/``s``/``A_0``
   stream and under seq_ft's rollout namespace: paired, episode for episode, with seq_ft's
   own 30k stage. Plasticity is the new task's success, forgetting the retained task's.
3. **cl_lr** — the same at the budget ``B*`` of the registered rule
   (:func:`choose_budget`), with other learning rates.

Exploratory: one seed; nothing here changes any registered comparison.
"""

from __future__ import annotations

import dataclasses
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import torch
from omegaconf import OmegaConf

from flowcl.utils.libero_paths import repo_root
from flowcl.utils.run import atomic_write_text, git_sha
from flowcl.utils.seeding import derive_seed

PARTS = ("trajectory", "cl_budget", "cl_lr")


def load_study_config(path: str | Path | None = None) -> dict:
    path = Path(path) if path else repo_root() / "configs" / "analysis" / "hparam_study.yaml"
    return OmegaConf.to_container(OmegaConf.load(path), resolve=True)


def suite(task_key: str) -> str:
    return task_key.split("/")[0]


def trajectory_run_id(task_key: str, seed: int) -> str:
    return f"traj__{suite(task_key)}__seed{seed}"


def cl_run_id(task_key: str, budget: int, lr: float, seed: int) -> str:
    return f"cl__{suite(task_key)}__b{budget}__lr{lr:g}__seed{seed}"


def stage_seed(seed_namespace: str, task_key: str, stage_idx: int) -> int:
    """The data/``s``/``A_0`` stream seed :func:`~flowcl.train.continual.run_continual` uses."""
    return derive_seed(seed_namespace, task_key, stage_idx)


def _refuse_dirty(allow_dirty: bool) -> str:
    sha = git_sha()
    if sha.endswith("-dirty") and not allow_dirty:
        raise RuntimeError(f"working tree is dirty ({sha}); commit first, or pass allow_dirty "
                           "(recorded)")
    return sha


# ---- pure pieces ------------------------------------------------------------------------


def split_episodes(episodes: list, n_heldout: int) -> tuple[list, list]:
    """``(training, held-out)``: the last ``n_heldout`` demos are held out."""
    if not 0 < n_heldout < len(episodes):
        raise ValueError(f"n_heldout must lie in (0, {len(episodes)}), got {n_heldout}")
    return list(episodes[:-n_heldout]), list(episodes[-n_heldout:])


def choose_budget(rows: list[dict], reference_new: float, tolerance: float, fallback: int) -> dict:
    """``B*``: the smallest budget whose new-task success is within ``tolerance`` of the
    reference's (point estimates, inclusive); ``fallback`` if none qualifies."""
    floor = reference_new - tolerance
    qualifying = sorted(r["budget"] for r in rows if r["new_task"]["success"] >= floor - 1e-9)
    return {"b_star": qualifying[0] if qualifying else fallback, "floor": floor,
            "reference_new": reference_new, "tolerance": tolerance,
            "qualifying": qualifying, "fallback_used": not qualifying}


class CheckpointAt:
    """``on_step`` callable: saves a checkpoint after each registered number of steps."""

    def __init__(self, steps, directory: Path, policy, policy_config: dict, spec, stats,
                 run_id: str, task_key: str) -> None:
        self.steps = sorted({int(s) for s in steps})
        if not self.steps or self.steps[0] < 1:
            raise ValueError(f"checkpoint steps must be positive, got {steps}")
        self.directory = Path(directory)
        self.policy, self.policy_config, self.spec, self.stats = policy, policy_config, spec, stats
        self.run_id, self.task_key = run_id, task_key
        self.saved: dict[int, Path] = {}

    def __call__(self, step: int, outputs: dict) -> None:
        from flowcl.train.checkpoint import save_checkpoint

        done = step + 1  # on_step runs after optimiser step index ``step``
        if done in self.steps:
            self.saved[done] = save_checkpoint(
                self.directory / f"step{done}.pt", policy=self.policy,
                policy_config=self.policy_config, spec=self.spec, stats=self.stats,
                run_id=self.run_id, stage=0, task_key=self.task_key,
                extra={"steps_done": done},
            )


@dataclass
class ProbeBank:
    """Fixed chunks × K fixed ``(s, A_0)`` draws, reused unchanged at every checkpoint."""

    name: str
    indices: torch.Tensor  # (N,) chunk indices into the bank's dataset
    s: torch.Tensor        # (N, K)
    noise: torch.Tensor    # (N, K, H, D)

    @classmethod
    def build(cls, name: str, dataset, indices, k_draws: int, s_sampler, seed: int) -> "ProbeBank":
        from flowcl.models.flow_head import draw_with_generator

        idx = torch.as_tensor(list(indices), dtype=torch.long)
        if idx.numel() == 0 or k_draws < 1:
            raise ValueError(f"{name}: a bank needs chunks and k_draws >= 1")
        g = torch.Generator(device="cpu").manual_seed(seed)
        n = idx.numel()
        s = s_sampler.sample(n * k_draws, torch.device("cpu"), generator=g).reshape(n, k_draws)
        shape = tuple(dataset[int(idx[0])]["actions"].shape)
        noise = draw_with_generator((n, k_draws, *shape), device="cpu", generator=g,
                                    dtype=torch.float32, normal=True)
        return cls(name, idx, s.to(torch.float32), noise)

    @property
    def k_draws(self) -> int:
        return self.s.shape[1]

    def payload(self) -> dict:
        return {"name": self.name, "indices": self.indices, "s": self.s, "noise": self.noise}

    @torch.no_grad()
    def loss(self, policy, dataset, batch_size: int, device) -> dict:
        """Element-weighted mean flow-matching loss over every chunk and draw (fp32, eval)."""
        from flowcl.data.dataset import collate_chunks
        from flowcl.models.losses import valid_element_count
        from flowcl.train.trainer import move_batch

        device = torch.device(device)
        policy.eval()
        total = weight = 0.0
        per_draw = [[0.0, 0.0] for _ in range(self.k_draws)]
        for start in range(0, self.indices.numel(), batch_size):
            rows = slice(start, start + batch_size)
            batch = move_batch(collate_chunks([dataset[int(i)] for i in self.indices[rows]]), device)
            n_b = float(valid_element_count(batch["action_mask"], policy.d_action))
            for k in range(self.k_draws):
                with torch.autocast(device_type=device.type, enabled=False):
                    loss = float(policy(batch, s=self.s[rows, k].to(device),
                                        noise=self.noise[rows, k].to(device))["loss"])
                total += loss * n_b
                weight += n_b
                per_draw[k][0] += loss * n_b
                per_draw[k][1] += n_b
        if weight == 0:
            raise ValueError(f"{self.name}: the bank has no valid elements")
        return {"loss": total / weight, "per_draw": [a / b for a, b in per_draw],
                "n_chunks": int(self.indices.numel()), "k_draws": self.k_draws}


# ---- evaluation --------------------------------------------------------------------------

Evaluator = Callable  # (policy, refs, spec, stats, run_id, stage) -> EvaluationReport


def default_evaluator() -> tuple[Evaluator, dict]:
    """§8.1 rollouts with the registered eval config, and its bootstrap settings."""
    from flowcl.envs.evaluation import eval_config_from_dict, evaluate_tasks

    payload = OmegaConf.to_container(
        OmegaConf.load(repo_root() / "configs" / "eval" / "libero_eval.yaml"), resolve=True)
    eval_cfg = eval_config_from_dict(payload)
    bootstrap = payload["bootstrap"]

    def evaluate(policy, refs, spec, stats, run_id, stage):
        return evaluate_tasks(policy, refs, spec, stats, run_id=run_id, cfg=eval_cfg,
                              bootstrap=bootstrap, stage=stage)

    return evaluate, bootstrap


def paired_against(task_eval, reference_eval, bootstrap: dict) -> dict:
    """Paired difference in success; raises unless the episode seeds match."""
    from flowcl.analysis.metrics import paired_difference_ci

    if list(task_eval.seeds) != list(reference_eval.seeds):
        raise ValueError(f"{task_eval.task_key}: rollout seeds differ from the reference's; "
                         "the evaluations are not paired")
    est = paired_difference_ci(task_eval.successes, reference_eval.successes,
                               seed=bootstrap["seed"], n_bootstrap=bootstrap["n_resamples"],
                               confidence=bootstrap["confidence"])
    return {"diff": est.value, "low": est.low, "high": est.high}


def _success(task_eval) -> dict:
    e = task_eval.estimate
    return {"success": e.value, "ci": [e.low, e.high], "n": len(task_eval.successes)}


def _free(device) -> None:
    if torch.device(device).type == "cuda" and torch.cuda.is_available():
        torch.cuda.empty_cache()


# ---- Part 1: trajectory --------------------------------------------------------------------


def run_trajectory(
    cfg: dict,
    task_key: str,
    results_root: Path | None = None,
    dataset_dir: Path | None = None,
    device: str = "cuda",
    allow_dirty: bool = False,
    evaluator: Evaluator | None = None,
    bootstrap: dict | None = None,
    load_episodes: Callable | None = None,
    pretrained: bool = True,
) -> dict:
    from flowcl.data.config import load_embodiment_spec
    from flowcl.data.dataset import ChunkedActionDataset
    from flowcl.data.stats import compute_stats
    from flowcl.data.tasks import load_task_episodes, resolve_tasks
    from flowcl.envs.evaluation import EvaluationReport
    from flowcl.experiments.gate0 import single_task_run_id
    from flowcl.models.build import build_policy
    from flowcl.train.checkpoint import load_checkpoint
    from flowcl.train.trainer import TrainConfig, train_one_task
    from flowcl.utils.run import create_run

    sha = _refuse_dirty(allow_dirty)
    tcfg = cfg["trajectory"]
    if task_key not in tcfg["tasks"]:
        raise ValueError(f"{task_key} is not a registered trajectory task ({tcfg['tasks']})")
    seed = int(cfg["seed"])
    root = Path(results_root) if results_root else repo_root() / "results"
    gate0_dir = root / single_task_run_id(task_key, seed)
    gate0_cfg = OmegaConf.to_container(OmegaConf.load(gate0_dir / "config.yaml"), resolve=True)
    if int(gate0_cfg["seed"]) != seed:
        raise ValueError(f"{gate0_dir.name} used seed {gate0_cfg['seed']}, the study {seed}")
    gate0_eval = EvaluationReport.load(gate0_dir / "eval.json").by_task()[task_key]
    if evaluator is None:
        evaluator, bootstrap = default_evaluator()

    spec = load_embodiment_spec(gate0_cfg["embodiment"])
    policy_config = gate0_cfg["policy"]
    train_cfg = TrainConfig(**{**gate0_cfg["train"], "device": device})
    ref = resolve_tasks([task_key])[0]
    loader = load_episodes or (lambda r, s: load_task_episodes(r, s, dataset_dir=dataset_dir))
    episodes = loader(ref, spec)
    train_eps, held_eps = split_episodes(episodes, int(tcfg["n_heldout"]))

    run_id = trajectory_run_id(task_key, seed)
    run = create_run(
        run_id=run_id,
        cfg=OmegaConf.create({
            "run_id": run_id, "seed": seed, "part": "trajectory", "task_key": task_key,
            "recipe_from": gate0_dir.name, "policy": policy_config,
            "train": dataclasses.asdict(train_cfg), "study": tcfg,
            "demos": {"train": len(train_eps), "heldout": len(held_eps)},
        }),
        seed=seed, results_root=root / cfg["out"] / "trajectory",
    )

    # Seeding mirrors flowcl.train.pipeline.train_on_tasks (Gate 0): manual_seed, stats,
    # policy build, then a CPU generator seeded with the run seed.
    torch.manual_seed(seed)
    stats = compute_stats(train_eps, embodiment=spec.name, task_id=task_key,
                          normalize_actions=not spec.action.already_normalized)
    stats.save(run.artifact("stats.json"))
    train_ds = ChunkedActionDataset(train_eps, spec, stats)
    held_ds = ChunkedActionDataset(held_eps, spec, stats)
    policy = build_policy(policy_config, spec, pretrained=pretrained)

    bank_cfg = tcfg["bank"]
    tags = bank_cfg["seed_tags"]
    subset = torch.Generator(device="cpu").manual_seed(derive_seed(tags["subset"], task_key, seed))
    train_idx = torch.randperm(len(train_ds), generator=subset)[: len(held_ds)].sort().values
    banks = {
        "heldout": ProbeBank.build("heldout", held_ds, range(len(held_ds)), bank_cfg["k_draws"],
                                   policy.s_sampler, derive_seed(tags["heldout"], task_key, seed)),
        "train": ProbeBank.build("train", train_ds, train_idx.tolist(), bank_cfg["k_draws"],
                                 policy.s_sampler, derive_seed(tags["train"], task_key, seed)),
    }
    torch.save({k: b.payload() for k, b in banks.items()}, run.artifact("banks.pt"))
    bank_data = {"heldout": held_ds, "train": train_ds}

    checkpointer = CheckpointAt(tcfg["checkpoint_steps"], run.subdir("checkpoints"), policy,
                                policy_config, spec, stats, run_id, task_key)
    if checkpointer.steps[-1] > train_cfg.steps:
        raise ValueError(f"checkpoint at {checkpointer.steps[-1]} beyond {train_cfg.steps} steps")
    generator = torch.Generator(device="cpu").manual_seed(seed)
    started = time.perf_counter()
    log = train_one_task(policy, train_ds, train_cfg, generator=generator, on_step=checkpointer)
    wall = time.perf_counter() - started
    atomic_write_text(run.artifact("losses.json"), json.dumps(log.losses) + "\n")
    del policy
    _free(device)

    rows = []
    for step in checkpointer.steps:
        loaded = load_checkpoint(checkpointer.saved[step], device=device)
        losses = {name: bank.loss(loaded.policy, bank_data[name], bank_cfg["batch_size"], device)
                  for name, bank in banks.items()}
        report = evaluator(loaded.policy, [ref], loaded.spec, loaded.stats,
                           run_id=gate0_dir.name, stage=0)
        report.save(run.subdir("eval") / f"step{step}.json")
        task_eval = report.by_task()[task_key]
        rows.append({"step": step, **_success(task_eval),
                     "paired_vs_gate0_30k": paired_against(task_eval, gate0_eval, bootstrap),
                     "train_bank": losses["train"], "heldout_bank": losses["heldout"]})
        del loaded
        _free(device)
        print(f"[flowcl] {run_id} step {step}: success {rows[-1]['success']:.2f}, train bank "
              f"{losses['train']['loss']:.5f}, held-out bank {losses['heldout']['loss']:.5f}",
              flush=True)

    out = {
        "git_sha": sha, "allow_dirty": allow_dirty, "part": "trajectory", "run_id": run_id,
        "task_key": task_key, "seed": seed, "config": cfg,
        "gate0": {"run": gate0_dir.name, **_success(gate0_eval), "steps": gate0_cfg["train"]["steps"],
                  "demos": gate0_cfg.get("data", {}).get("n_demos")},
        "demos": {"train": len(train_eps), "heldout": len(held_eps)},
        "banks": {k: {"n_chunks": int(b.indices.numel()), "k_draws": b.k_draws} for k, b in banks.items()},
        "training": {"final_loss": log.final_loss, "mean_last_50_loss": log.mean_last(50),
                     "wall_clock_s": wall},
        "rows": rows,
    }
    atomic_write_text(run.artifact("report.json"), json.dumps(out, indent=2) + "\n")
    return out


# ---- Parts 2 and 3: budget and learning rate vs forgetting -----------------------------------


@dataclass
class _Reference:
    run_dir: Path
    config: dict
    curriculum: object
    seed_namespace: str
    new_eval: object
    retained_eval: object
    probe: dict
    probe_reference: dict


def _reference(cfg: dict, root: Path) -> _Reference:
    from flowcl.data.curriculum import load_curriculum
    from flowcl.envs.evaluation import EvaluationReport

    from flowcl.train.continual import seed_namespace_run_id

    c = cfg["cl"]
    run_dir = root / c["reference_run"]
    config = OmegaConf.to_container(OmegaConf.load(run_dir / "config.yaml"), resolve=True)
    result = json.loads((run_dir / "result.json").read_text())
    curriculum = load_curriculum(config["curriculum"])
    ti, ri = c["task_index"], c["retained_index"]
    if not 0 <= ri < ti < len(curriculum.stages):
        raise ValueError(f"need 0 <= retained_index < task_index, got {ri}, {ti}")
    # The runner's rule, not a recorded field: Gate 1's seq_ft result.json predates
    # ``seed_namespace_run_id``. Cross-checked against the field (when present) and the
    # namespace the reference stage's rollouts were seeded under.
    namespace = seed_namespace_run_id(curriculum.name, int(config["seed"]))
    report = EvaluationReport.load(run_dir / "eval" / f"stage{ti}.json")
    recorded = {"result.json": result.get("seed_namespace_run_id", namespace),
                f"eval/stage{ti}.json": report.run_id}
    if any(v != namespace for v in recorded.values()):
        raise ValueError(f"{run_dir.name}: seed namespace {namespace!r} (runner rule) disagrees "
                         f"with {recorded}")
    evals = report.by_task()
    diag = json.loads((root / c["probe_from"]).read_text())
    if diag["reference_run_id"] != run_dir.name:
        raise ValueError(f"{c['probe_from']} pairs with {diag['reference_run_id']!r}, "
                         f"not {run_dir.name!r}")
    L = diag["loss_matrix"]["reference"]
    return _Reference(
        run_dir=run_dir, config=config, curriculum=curriculum,
        seed_namespace=namespace,
        new_eval=evals[curriculum.stages[ti].task_key],
        retained_eval=evals[curriculum.stages[ri].task_key],
        probe=diag["config"]["probe"],
        probe_reference={"retained": L[ti][ri], "new": L[ti][ti]},
    )


def run_cl_variant(
    cfg: dict,
    part: str,
    budget: int,
    lr: float,
    results_root: Path | None = None,
    dataset_dir: Path | None = None,
    device: str = "cuda",
    evaluator: Evaluator | None = None,
    bootstrap: dict | None = None,
    build_datasets: Callable | None = None,
) -> dict:
    """One variant: the new task for ``budget`` steps at ``lr`` from the reference's previous
    stage, paired with the reference's own stage (data stream and rollout episodes)."""
    from flowcl.analysis.probes import probe_loss
    from flowcl.train.checkpoint import load_checkpoint, save_checkpoint
    from flowcl.train.pipeline import build_dataset
    from flowcl.train.trainer import TrainConfig, train_one_task
    from flowcl.utils.run import create_run

    if part not in ("cl_budget", "cl_lr"):
        raise ValueError(f"unknown CL part {part!r}")
    root = Path(results_root) if results_root else repo_root() / "results"
    ref = _reference(cfg, root)
    if evaluator is None:
        evaluator, bootstrap = default_evaluator()
    c = cfg["cl"]
    ti, ri = c["task_index"], c["retained_index"]
    new_stage, retained_stage = ref.curriculum.stages[ti], ref.curriculum.stages[ri]
    seed = int(cfg["seed"])

    start = load_checkpoint(ref.run_dir / "checkpoints" / f"stage{ti - 1}.pt", device=device)
    policy, spec, stats = start.policy, start.spec, start.stats
    make = build_datasets or (lambda stage: build_dataset(
        [stage.ref], spec, stats, n_demos=stage.n_demos, dataset_dir=dataset_dir))
    datasets = {"new": make(new_stage), "retained": make(retained_stage)}
    train_cfg = TrainConfig(**{**ref.config["train"], "steps": int(budget), "lr": float(lr),
                               "device": device})
    run_id = cl_run_id(new_stage.task_key, budget, lr, seed)
    run = create_run(
        run_id=run_id,
        cfg=OmegaConf.create({
            "run_id": run_id, "seed": seed, "part": part, "reference_run": ref.run_dir.name,
            "start_checkpoint": f"stage{ti - 1}.pt", "task_key": new_stage.task_key,
            "retained_task_key": retained_stage.task_key,
            "train": dataclasses.asdict(train_cfg),
            "stream_seed": {"namespace": ref.seed_namespace, "stage": ti},
        }),
        seed=seed, results_root=root / cfg["out"] / part,
    )
    generator = torch.Generator(device="cpu").manual_seed(
        stage_seed(ref.seed_namespace, new_stage.task_key, ti))
    started = time.perf_counter()
    log = train_one_task(policy, datasets["new"], train_cfg, task_idx=ti, generator=generator)
    wall = time.perf_counter() - started
    atomic_write_text(run.artifact("losses.json"), json.dumps(log.losses) + "\n")
    save_checkpoint(run.subdir("checkpoints") / "final.pt", policy=policy,
                    policy_config=start.payload["policy_config"], spec=spec, stats=stats,
                    run_id=run_id, stage=ti, task_key=new_stage.task_key,
                    extra={"budget": budget, "lr": lr})

    probes = {k: probe_loss(policy, datasets[k], ref.probe, device) for k in ("retained", "new")}
    report = evaluator(policy, [retained_stage.ref, new_stage.ref], spec, stats,
                       run_id=ref.seed_namespace, stage=ti)
    report.save(run.artifact("eval.json"))
    by_task = report.by_task()
    new_eval, retained_eval = by_task[new_stage.task_key], by_task[retained_stage.task_key]
    del policy, start
    _free(device)
    row = {
        "run_id": run_id, "budget": int(budget), "lr": float(lr),
        "new_task": {**_success(new_eval),
                     "paired_vs_reference": paired_against(new_eval, ref.new_eval, bootstrap)},
        "retained": {**_success(retained_eval),
                     "paired_vs_reference": paired_against(retained_eval, ref.retained_eval, bootstrap)},
        "probe_loss": {"retained": probes["retained"], "new": probes["new"]},
        "training": {"final_loss": log.final_loss, "mean_last_50_loss": log.mean_last(50),
                     "wall_clock_s": wall},
    }
    atomic_write_text(run.artifact("row.json"), json.dumps(row, indent=2) + "\n")
    print(f"[flowcl] {run_id}: new {row['new_task']['success']:.2f}, retained "
          f"{row['retained']['success']:.2f}, probe retained {probes['retained']:.4f}", flush=True)
    return row


def _reference_row(ref: _Reference) -> dict:
    steps = ref.config["train"]["steps"]
    return {"run_id": ref.run_dir.name, "budget": steps, "lr": ref.config["train"]["lr"],
            "new_task": _success(ref.new_eval), "retained": _success(ref.retained_eval),
            "probe_loss": ref.probe_reference}


def run_cl_part(
    cfg: dict,
    part: str,
    results_root: Path | None = None,
    dataset_dir: Path | None = None,
    device: str = "cuda",
    allow_dirty: bool = False,
    evaluator: Evaluator | None = None,
    bootstrap: dict | None = None,
    build_datasets: Callable | None = None,
) -> dict:
    """``cl_budget``: every registered budget at the reference learning rate. ``cl_lr``: every
    registered learning rate at ``B*``, read from the written ``cl_budget`` report."""
    sha = _refuse_dirty(allow_dirty)
    root = Path(results_root) if results_root else repo_root() / "results"
    ref = _reference(cfg, root)
    reference = _reference_row(ref)
    c = cfg["cl"]
    extra: dict = {}
    if part == "cl_budget":
        variants = [(int(b), float(ref.config["train"]["lr"])) for b in c["budgets"]]
    elif part == "cl_lr":
        budget_report = json.loads((root / cfg["out"] / "cl_budget" / "report.json").read_text())
        rule = c["budget_rule"]
        choice = choose_budget(budget_report["rows"], reference["new_task"]["success"],
                               rule["tolerance"], rule["fallback"])
        extra["budget_choice"] = choice
        variants = [(choice["b_star"], float(lr)) for lr in c["learning_rates"]]
        base = [r for r in budget_report["rows"] if r["budget"] == choice["b_star"]]
        extra["base_lr_row"] = base[0] if base else reference
    else:
        raise ValueError(f"unknown CL part {part!r}")
    rows = [run_cl_variant(cfg, part, b, lr, results_root=root, dataset_dir=dataset_dir,
                           device=device, evaluator=evaluator, bootstrap=bootstrap,
                           build_datasets=build_datasets) for b, lr in variants]
    out = {"git_sha": sha, "allow_dirty": allow_dirty, "part": part, "seed": cfg["seed"],
           "config": cfg, "reference": reference, **extra, "rows": rows}
    path = root / cfg["out"] / part / "report.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_text(path, json.dumps(out, indent=2) + "\n")
    print(f"[flowcl] wrote {path}", flush=True)
    return out
