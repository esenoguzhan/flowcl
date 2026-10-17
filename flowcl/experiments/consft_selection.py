"""ConSFT's τ_start × T bracket on the development seed (docs/thesis_plan.md, Workstream B; docs/consft_notes.md).

The rule is ``configs/analysis/baseline_selection_consft.yaml`` (committed before the calibration and before
any arm). :func:`calibrate` measures the per-sample losses of seq_ft seed 0's ``stage0.pt`` on T2's data (no
arm outcome) and fixes the grid ``τ_start(z) = κ·q50/z``; the rule then registers the calibration's values and
SHA-256. One arm per grid cell, plus the paper's Table 5 arm, trains T2 (Object) from that checkpoint through
:func:`flowcl.experiments.low_update.train_stage` with the runner's text-cache state, the method being
:class:`flowcl.methods.consft.ConSFT`. :func:`select_consft` (pure) applies the rule, including the edge
extension and the cap; :func:`run_select` freezes the choice in an immutable bundle bound to the rule's and
the calibration's SHA-256, carrying Track 2's exclusion keys for every arm; :func:`verify` re-validates every
row and recomputes the selection and the keys (fingerprints from the checkpoint tensors).

:func:`smoke_arm` is the bracket path against the runner, independent of the calibration: an arm with explicit
parameters from a scratch seq_ft run's ``stage0.pt`` must reproduce a scratch ConSFT run's ``stage1.pt``
tensor for tensor under the same TrainConfig, with the same weighting logs.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import math
from pathlib import Path

import torch
from omegaconf import OmegaConf

from flowcl.experiments.low_update import _root, load_low_update_config, load_published, refuse_dirty
from flowcl.utils.libero_paths import repo_root
from flowcl.utils.run import atomic_write_text, file_sha256, git_sha

TOL = 1e-9                # rates are k/50: allow float rounding only
EXTENSION_NEEDED = 10     # the CLI's exit code when --select owes the edge extension first
FAMILY = "consft"
EVAL_CONFIG = Path("configs") / "eval" / "libero_eval.yaml"
QUANTILES = (0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95)
CAL_S_BINS = 10
CONSTANTS = ("kappa", "decay_rate", "curve", "omega_min", "tau_end")


def rule_path(path: str | Path | None = None) -> Path:
    return Path(path) if path else repo_root() / "configs" / "analysis" / "baseline_selection_consft.yaml"


def load_config(path: str | Path | None = None) -> dict:
    return OmegaConf.to_container(OmegaConf.load(rule_path(path)), resolve=True)


def constants(cfg: dict) -> dict:
    return {k: float(cfg["consft"]["constants"][k]) for k in CONSTANTS}


def z_tag(z: float) -> str:
    return f"{float(z):g}"


def arm_key(arm: dict) -> str:
    if arm["arm_type"] == "table5":
        return "table5"
    if arm["arm_type"] == "smoke":
        return "smoke"
    return f"z{z_tag(arm['z'])}__T{int(arm['decay_steps'])}"


def arm_run_id(key: str, seed: int) -> str:
    return f"consftb__{key}__seed{seed}"


def _out(cfg: dict, out_root: Path) -> Path:
    return Path(out_root) / cfg["out"]


def calibration_path(cfg: dict, out_root: Path) -> Path:
    return _out(cfg, out_root) / "consft_calibration.json"


def arm_dir(cfg: dict, out_root: Path, key: str, seed: int) -> Path:
    return _out(cfg, out_root) / "consft" / f"seed{seed}" / arm_run_id(key, seed)


def _json_sha(payload) -> str:
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


def _file(path: Path) -> dict:
    return {"path": str(path), "sha256": file_sha256(path)}


# ---- the grid -------------------------------------------------------------------------------------


def plan_arms(cfg: dict, q50: float) -> dict:
    """The arms the rule implies at median initial loss ``q50``: ``{"grid": [...], "table5": {...},
    "extensions": [...]}``, each arm ``{key, arm_type, z, tau_start, decay_steps, capped}`` (extensions also
    carry ``side``). Refuses a non-positive or non-finite ``q50``."""
    if not (isinstance(q50, (int, float)) and math.isfinite(q50) and q50 > 0):
        raise ValueError(f"the calibration's median loss must be positive and finite, got {q50!r}")
    c = cfg["consft"]
    kappa, tau_end = float(c["constants"]["kappa"]), float(c["constants"]["tau_end"])

    def arm(arm_type, z, tau_start, steps, **extra):
        a = {"arm_type": arm_type, "z": float(z), "tau_start": float(tau_start), "decay_steps": int(steps),
             "capped": float(tau_start) >= tau_end, **extra}
        return {"key": arm_key(a), **a}

    grid = [arm("grid", z, kappa * q50 / float(z), t) for z in c["zs"] for t in c["decay_steps"]]
    t5 = c["table5"]
    table5 = arm("table5", kappa * q50 / float(t5["tau_start"]), t5["tau_start"], t5["decay_steps"])
    extensions = [arm("extension", z, kappa * q50 / float(z), t, side=side)
                  for side, z in sorted(c["extension"]["z"].items()) for t in c["decay_steps"]]
    return {"grid": grid, "table5": table5, "extensions": extensions}


def check_caps(plan: dict) -> None:
    capped = [a["key"] for a in plan["grid"] + [plan["table5"]] if a["capped"]]
    if capped:
        raise ValueError(f"arms {capped} have tau_start >= tau_end and would run at the cap throughout; "
                         "amend the rule before any arm")


def base_arms(plan: dict) -> dict[str, dict]:
    return {a["key"]: a for a in plan["grid"] + [plan["table5"]]}


def runnable_extensions(plan: dict) -> dict[str, dict]:
    return {a["key"]: a for a in plan["extensions"] if not a["capped"]}


# ---- the calibration ------------------------------------------------------------------------------


def _quantiles(x: torch.Tensor) -> dict:
    q = torch.quantile(x.double(), torch.tensor(QUANTILES, dtype=torch.float64))
    return {f"{p:g}": float(v) for p, v in zip(QUANTILES, q)}


def _bins(s: torch.Tensor) -> torch.Tensor:
    return torch.clamp((s.double() * CAL_S_BINS).long(), 0, CAL_S_BINS - 1)


def summarize_losses(losses: torch.Tensor, s: torch.Tensor) -> dict:
    bins = _bins(s)
    return {"n": int(losses.numel()), "mean": float(losses.double().mean()), "quantiles": _quantiles(losses),
            "by_s_bin": {str(b): {"n": int((bins == b).sum()),
                                  "median": float(losses[bins == b].double().median()) if (bins == b).any() else None,
                                  "mean": float(losses[bins == b].double().mean()) if (bins == b).any() else None}
                         for b in range(CAL_S_BINS)}}


def omega(losses: torch.Tensor, tau: float, consts: dict) -> torch.Tensor:
    """The method's weight for fp32 losses at temperature ``tau`` (as :meth:`ConSFT.weights`)."""
    return torch.exp(-consts["kappa"] * losses.float() / float(tau)).clamp(min=consts["omega_min"])


def summarize_omega(losses: torch.Tensor, s: torch.Tensor, tau: float, consts: dict) -> dict:
    w = omega(losses, tau, consts)
    floor = w <= consts["omega_min"]
    bins = _bins(s)
    return {"tau": float(tau), "mean": float(w.double().mean()), "floor_fraction": float(floor.double().mean()),
            "quantiles": _quantiles(w),
            "by_s_bin": {str(b): {"n": int((bins == b).sum()),
                                  "omega_mean": float(w[bins == b].double().mean()) if (bins == b).any() else None,
                                  "floor_fraction": float(floor[bins == b].double().mean()) if (bins == b).any()
                                  else None}
                         for b in range(CAL_S_BINS)}}


def sample_losses(policy, dataset, n: int, batch_size: int, seed: int, device: str,
                  amp: bool) -> tuple[torch.Tensor, torch.Tensor]:
    """Per-sample losses and flow times of ``n`` draws (indices, s and noise from one generator seeded
    ``seed``), in batches, train mode and AMP as in training, without gradient; the mode is restored."""
    from flowcl.data.dataset import collate_chunks
    from flowcl.methods.consft import per_sample_terms
    from flowcl.train.trainer import move_batch

    dev = torch.device(device)
    generator = torch.Generator(device="cpu").manual_seed(int(seed))
    indices = torch.randint(len(dataset), (int(n),), generator=generator).tolist()
    losses, flow_times = [], []
    was_training = policy.training
    policy.train()
    try:
        with torch.no_grad():
            for start in range(0, len(indices), int(batch_size)):
                batch = move_batch(collate_chunks([dataset[i] for i in indices[start:start + int(batch_size)]]), dev)
                with torch.autocast(device_type=dev.type, enabled=bool(amp) and dev.type == "cuda"):
                    out = policy(batch, generator=generator)
                summed, count = per_sample_terms(out["velocity"], out["target_velocity"], batch["action_mask"])
                losses.append((summed / count).float().cpu())
                flow_times.append(out["s"].float().cpu())
    finally:
        policy.train(was_training)
    out_l, out_s = torch.cat(losses), torch.cat(flow_times)
    if not torch.isfinite(out_l).all():
        raise RuntimeError("non-finite per-sample losses in the calibration")
    return out_l, out_s


def calibrate(cfg: dict, *, results_root=None, out_root=None, device: str = "cuda", dataset_dir=None,
              build_datasets=None, low_update_cfg: dict | None = None, rule: Path | None = None) -> dict:
    """Measure the initial loss scale and write the calibration file (no arm outcome is used)."""
    from flowcl.experiments.hparam_study import _free
    from flowcl.experiments.low_update import seed_context, warm_runner_cache
    from flowcl.train.checkpoint import load_checkpoint
    from flowcl.train.pipeline import build_dataset
    from flowcl.utils.seeding import derive_seed

    root = _root(results_root)
    out_root = Path(out_root) if out_root else root
    c = cfg["consft"]
    if c.get("calibration") is not None:
        raise RuntimeError("the rule is already calibrated; a new calibration needs a new rule")
    seed = int(cfg["bracket_seed"])
    ctx = seed_context(low_update_cfg or load_low_update_config(), seed, root)
    stage0, stage1 = ctx.ref_dir / "checkpoints" / "stage0.pt", ctx.ref_dir / "checkpoints" / "stage1.pt"
    amp = bool(ctx.ref_config["train"].get("amp"))
    stages = ctx.curriculum.stages
    task_key = stages[1].task_key
    draw_seed = derive_seed(ctx.namespace, f"consft_calibration::{task_key}", 1)
    n, bs = int(c["calibration_samples"]), int(c["calibration_batch"])
    measured, caches = {}, {}
    for name, path in (("stage0", stage0), ("stage1", stage1)):
        loaded = load_checkpoint(path, device=device)
        make = build_datasets or (lambda st: build_dataset([st.ref], loaded.spec, loaded.stats, n_demos=st.n_demos,
                                                           dataset_dir=dataset_dir))
        datasets = {j: make(stages[j]) for j in (0, 1)}   # what train_stage builds for stage 1
        caches[name] = warm_runner_cache(loaded.policy, datasets, device, amp=amp)
        measured[name] = sample_losses(loaded.policy, datasets[1], n, bs, draw_seed, device, amp)
        del loaded, datasets
        _free(device)
    (l0, s0), (l1, s1) = measured["stage0"], measured["stage1"]
    if not torch.equal(s0, s1):
        raise RuntimeError("the two checkpoints did not see the same draws")
    q50 = float(torch.quantile(l0.double(), 0.5))
    plan = plan_arms(cfg, q50)
    check_caps(plan)
    consts = constants(cfg)
    every = plan["grid"] + [plan["table5"]] + plan["extensions"]
    payload = {
        "git_sha": git_sha(), "seed": seed, "namespace": ctx.namespace, "task_key": task_key,
        "rule": {"path": str(rule_path(rule)), "sha256": file_sha256(rule_path(rule))},
        "stage0": _file(stage0), "stage1": _file(stage1),
        "draws": {"n": n, "batch": bs, "seed": draw_seed, "tag": f"consft_calibration::{task_key}", "amp": amp,
                  "mode": "train"},
        "constants": consts, "zs": [float(z) for z in c["zs"]], "decay_steps": [int(t) for t in c["decay_steps"]],
        "table5": {"tau_start": float(c["table5"]["tau_start"]), "decay_steps": int(c["table5"]["decay_steps"])},
        "extension_z": {k: float(v) for k, v in c["extension"]["z"].items()},
        "q50": q50,
        "arms": {a["key"]: {k: v for k, v in a.items() if k != "key"} for a in every},
        "losses": {"stage0": summarize_losses(l0, s0), "stage1": summarize_losses(l1, s1)},
        "omega_initial": {a["key"]: summarize_omega(l0, s0, a["tau_start"], consts) for a in every},
        "omega_stage1": {**{a["key"]: summarize_omega(l1, s1, a["tau_start"], consts) for a in every},
                         "tau_end": summarize_omega(l1, s1, consts["tau_end"], consts)},
        "text_cache": caches,
    }
    path = calibration_path(cfg, out_root)
    path.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_text(path, json.dumps(payload, indent=2, default=str) + "\n")
    print(f"[flowcl] ConSFT calibration: q50 {q50!r}; tau_start "
          f"{ {a['key']: a['tau_start'] for a in plan['grid']} } -> {path} (sha256 {file_sha256(path)})", flush=True)
    return payload


def validate_calibration(cfg: dict, out_root: Path) -> tuple[dict, dict, str]:
    """The rule's registered calibration: ``(plan, payload, sha256)``; raises on any mismatch."""
    reg = cfg["consft"].get("calibration")
    if not reg:
        raise RuntimeError("the rule is not calibrated yet (consft.calibration is null)")
    path = Path(out_root) / reg["file"]
    sha = file_sha256(path)
    if sha != reg["sha256"]:
        raise ValueError(f"{path}: SHA-256 {sha} differs from the rule's {reg['sha256']}")
    payload = json.loads(path.read_text())
    if float(reg["q50"]) != payload["q50"]:
        raise ValueError(f"registered q50 {reg['q50']!r} differs from the calibration's {payload['q50']!r}")
    plan = plan_arms(cfg, payload["q50"])
    check_caps(plan)
    every = plan["grid"] + [plan["table5"]] + plan["extensions"]
    recomputed = {a["key"]: a["tau_start"] for a in every}
    registered = {k: float(v) for k, v in reg["tau_starts"].items()}
    if registered != recomputed or {k: v["tau_start"] for k, v in payload["arms"].items()} != recomputed:
        raise ValueError(f"registered tau_starts {registered} differ from the calibration's {recomputed}")
    capped = sorted(a["key"] for a in plan["extensions"] if a["capped"])
    if sorted(reg.get("capped_extensions") or []) != capped:
        raise ValueError(f"registered capped extensions {reg.get('capped_extensions')} differ from {capped}")
    c = cfg["consft"]
    problems = []
    if payload.get("seed") != int(cfg["bracket_seed"]):
        problems.append(f"seed {payload.get('seed')} != {cfg['bracket_seed']}")
    if payload.get("constants") != constants(cfg):
        problems.append(f"constants {payload.get('constants')} != {constants(cfg)}")
    if payload["draws"]["n"] != int(c["calibration_samples"]) or payload["draws"]["batch"] != int(c["calibration_batch"]):
        problems.append(f"draws {payload['draws']} are not the rule's")
    ref = Path(out_root) / payload.get("namespace", "")
    for key, want in (("stage0", ref / "checkpoints" / "stage0.pt"), ("stage1", ref / "checkpoints" / "stage1.pt")):
        rec = payload.get(key) or {}
        if Path(rec.get("path", "")).resolve() != want.resolve():
            problems.append(f"{key} path {rec.get('path')} is not {want}")
        elif not want.is_file() or file_sha256(want) != rec.get("sha256"):
            problems.append(f"{key} {want} no longer has the calibrated SHA-256")
    if problems:
        raise ValueError(f"the calibration was made from other inputs: {problems}")
    return plan, payload, sha


# ---- arms -----------------------------------------------------------------------------------------


def method_spec(cfg: dict, arm: dict, rule_sha: str, calibration_sha: str | None) -> dict:
    return {"family": FAMILY, "arm_type": arm["arm_type"], "z": arm.get("z"), "tau_start": float(arm["tau_start"]),
            "decay_steps": int(arm["decay_steps"]), **constants(cfg),
            "rule_sha256": rule_sha, "calibration_sha256": calibration_sha}


def expected_train_config(ctx, lr: float, steps: int, device: str) -> dict:
    """The TrainConfig :func:`low_update.stage_setup` builds for an arm (the reference's recipe)."""
    from flowcl.train.trainer import TrainConfig

    return dataclasses.asdict(TrainConfig(**{**ctx.ref_config["train"], "steps": int(steps), "lr": float(lr),
                                             "device": device}))


def shared_inputs(cfg: dict, ctx, start_checkpoint: Path, rule_sha: str, calibration_sha: str | None) -> dict:
    """What every arm must share, from the current files."""
    eval_path = repo_root() / EVAL_CONFIG
    root = ctx.ref_dir.parent
    return {
        "start_checkpoint": _file(Path(start_checkpoint)),
        "reference_result": _file(ctx.ref_dir / "result.json"),
        "reference_evals_stage1": {role: _file(root / run / "eval" / "stage1.json")
                                   for role, run in sorted(ctx.runs.items())},
        "evaluation_config": {**_file(eval_path),
                              "content": OmegaConf.to_container(OmegaConf.load(eval_path), resolve=True)},
        "probe": ctx.probe,
        "namespace": ctx.namespace,
        "constants": constants(cfg),
        "recipe": {"steps": int(cfg["consft"]["recipe"]["steps"]), "lr": float(cfg["consft"]["recipe"]["lr"])},
        "rule_sha256": rule_sha,
        "calibration_sha256": calibration_sha,
    }


def state_fingerprint(state_dict: dict) -> str:
    from flowcl.methods.ewc import tensor_fingerprint

    return tensor_fingerprint(state_dict)


def checkpoint_fingerprint(path: Path) -> str:
    return state_fingerprint(torch.load(path, map_location="cpu", weights_only=False)["state_dict"])


def run_arm(cfg: dict, key: str, *, results_root=None, out_root=None, device: str = "cuda", evaluator=None,
            bootstrap=None, dataset_dir=None, build_datasets=None, low_update_cfg: dict | None = None,
            allow_dirty: bool = False, extension: bool = False, rule: Path | None = None,
            smoke_params: dict | None = None, smoke_start: Path | None = None, smoke_steps: int | None = None,
            smoke_episodes: int | None = None) -> dict:
    """Train and publish one arm (reused if already published under this rule and calibration).

    ``smoke_params`` (``{"tau_start", "decay_steps"}``) trains a smoke arm with explicit parameters,
    independent of the calibration (scratch output only)."""
    from flowcl.experiments.hparam_study import default_evaluator
    from flowcl.experiments.low_update import seed_context, train_stage
    from flowcl.methods.consft import ConSFT

    smoke = any(x is not None for x in (smoke_params, smoke_start, smoke_steps, smoke_episodes))
    refuse_dirty(allow_dirty or smoke)
    root = _root(results_root)
    out_root = Path(out_root) if out_root else root
    if smoke_params is not None:
        arm = {"arm_type": "smoke", "z": None, "tau_start": float(smoke_params["tau_start"]),
               "decay_steps": int(smoke_params["decay_steps"])}
        key, cal_sha = "smoke", None
    else:
        plan, _, cal_sha = validate_calibration(cfg, root)
        base, ext = base_arms(plan), runnable_extensions(plan)
        if key in base and not extension:
            arm = base[key]
        elif key in ext and extension:
            arm = ext[key]
        elif key in ext or key in base:
            raise ValueError(f"{key} is {'an extension' if key in ext else 'a base'} arm: "
                             f"{'use run_extension' if key in ext else 'not an extension'}")
        else:
            raise ValueError(f"{key} is not an arm of this rule ({sorted(base)}, extensions {sorted(ext)})")
    seed = int(cfg["bracket_seed"])
    ctx = seed_context(low_update_cfg or load_low_update_config(), seed, root)
    if smoke:  # 1-episode cells cannot be paired with the 50-episode references
        ctx = dataclasses.replace(ctx, evals={role: {} for role in ctx.evals})
    if evaluator is None:
        evaluator, bootstrap = default_evaluator(smoke_episodes)
    rule_sha = file_sha256(rule_path(rule))
    start = Path(smoke_start) if smoke_start is not None else ctx.ref_dir / "checkpoints" / "stage0.pt"
    steps = int(smoke_steps) if smoke_steps is not None else int(cfg["consft"]["recipe"]["steps"])
    lr = float(cfg["consft"]["recipe"]["lr"])
    inputs = shared_inputs(cfg, ctx, start, rule_sha, cal_sha)
    train_config = expected_train_config(ctx, lr, steps, device)
    params = {k: arm[k] for k in ("arm_type", "z", "tau_start", "decay_steps")}

    def setup(policy):
        return ConSFT(tau_start=arm["tau_start"], decay_steps=arm["decay_steps"], **constants(cfg))

    def extra_row(policy, method, start_state):
        return {"consft": {"key": key, **params, "logs": method.logs["1"], "inputs": inputs,
                           "train_config": train_config, "train_config_sha256": _json_sha(train_config),
                           "state_fingerprint": state_fingerprint(policy.state_dict())}}

    out_dir = arm_dir(cfg, out_root, key, seed)
    row = train_stage(
        ctx, 1, steps, lr, start, out_dir, arm_run_id(key, seed),
        method_spec=method_spec(cfg, arm, rule_sha, cal_sha), setup_method=setup,
        extra_row=extra_row, evaluator=evaluator, bootstrap=bootstrap, device=device,
        dataset_dir=dataset_dir, build_datasets=build_datasets, warm_cache="runner",
    )
    actual = _arm_train_config(out_dir)
    if actual != train_config:
        raise RuntimeError(f"{out_dir}: the arm trained under {actual}, not the rule's {train_config}")
    return row


def _arm_train_config(out_dir: Path) -> dict:
    return OmegaConf.to_container(OmegaConf.load(Path(out_dir) / "config.yaml"), resolve=True)["train"]


def published_rows(cfg: dict, out_root: Path, plan: dict, rule_sha: str, cal_sha: str) -> dict[str, dict]:
    """Every published base or extension arm, re-validated by :func:`load_published`."""
    seed = int(cfg["bracket_seed"])
    steps, lr = int(cfg["consft"]["recipe"]["steps"]), float(cfg["consft"]["recipe"]["lr"])
    rows = {}
    for key, arm in {**base_arms(plan), **runnable_extensions(plan)}.items():
        row = load_published(arm_dir(cfg, out_root, key, seed),
                             {"run_id": arm_run_id(key, seed), "method": method_spec(cfg, arm, rule_sha, cal_sha),
                              "lr": lr, "steps": steps})
        if row is not None:
            rows[key] = row
    return rows


def validate_rows(cfg: dict, rows: dict, plan: dict, ctx, out_root: Path, rule_sha: str, cal_sha: str, *,
                  allow_dirty: bool = False, fingerprints: bool = False) -> list[str]:
    """Problems with the rows: shared inputs not the current files', arm parameters not the plan's, a
    TrainConfig not the rule's, dirty provenance, or (``fingerprints``) a checkpoint whose tensors no
    longer match the recorded fingerprint."""
    problems = []
    steps, lr = int(cfg["consft"]["recipe"]["steps"]), float(cfg["consft"]["recipe"]["lr"])
    want_inputs = shared_inputs(cfg, ctx, ctx.ref_dir / "checkpoints" / "stage0.pt", rule_sha, cal_sha)
    arms = {**base_arms(plan), **runnable_extensions(plan)}
    for key, row in sorted(rows.items()):
        rec, name, arm = row["consft"], row["run_id"], arms[key]
        if rec["inputs"] != want_inputs:
            diff = sorted(k for k in set(want_inputs) | set(rec["inputs"])
                          if want_inputs.get(k) != rec["inputs"].get(k))
            problems.append(f"{name}: shared inputs differ from the current files in {diff}")
        if {k: rec[k] for k in ("arm_type", "z", "tau_start", "decay_steps")} != \
                {k: arm[k] for k in ("arm_type", "z", "tau_start", "decay_steps")}:
            problems.append(f"{name}: its parameters are not the plan's for {key}")
        want = expected_train_config(ctx, lr, steps, rec["train_config"]["device"])
        if rec["train_config"] != want or rec["train_config_sha256"] != _json_sha(want):
            problems.append(f"{name}: recorded TrainConfig is not the rule's")
        directory = arm_dir(cfg, out_root, key, int(cfg["bracket_seed"]))
        if _arm_train_config(directory) != want:
            problems.append(f"{name}: config.yaml's TrainConfig is not the rule's")
        if str(row.get("git_sha", "")).endswith("-dirty") and not allow_dirty:
            problems.append(f"{name}: trained on a dirty tree ({row.get('git_sha')})")
        if fingerprints and checkpoint_fingerprint(directory / row["checkpoint"]) != rec["state_fingerprint"]:
            problems.append(f"{name}: checkpoint tensors no longer match the recorded fingerprint")
    return problems


# ---- the rule (pure) ------------------------------------------------------------------------------


def summary(key: str, row: dict) -> dict:
    """The rule's view of a row."""
    c, rec = row["cells"], row["consft"]
    return {"key": key, "arm_type": rec["arm_type"], "z": float(rec["z"]), "tau_start": float(rec["tau_start"]),
            "decay_steps": int(rec["decay_steps"]), "run_id": row["run_id"],
            "spatial": c["0"]["success"], "object": c["1"]["success"],
            "spatial_probe": c["0"]["probe_loss"], "object_probe": c["1"]["probe_loss"]}


def _order(a: dict) -> tuple:
    return (-a["spatial"], a["spatial_probe"], a["z"], a["decay_steps"], a["key"])


def _fallback_order(a: dict) -> tuple:
    return (-a["object"], a["object_probe"], a["z"], a["decay_steps"], a["key"])


def _apply(arms: list[dict], floor: float) -> tuple[dict, list[str], list[dict], bool]:
    eligible = [a for a in arms if a["object"] >= floor - TOL]
    if eligible:
        chosen = min(eligible, key=_order)
        return chosen, (["all_eligible_spatial_zero"] if chosen["spatial"] <= TOL else []), eligible, False
    return min(arms, key=_fallback_order), ["no_qualifying_setting"], [], True


def owed_extension(chosen: dict, zs: list, extensions: list[dict]) -> tuple[dict | None, list[str]]:
    """The extension a selection at ``chosen`` owes (``None`` for a table5 or extension arm, a z inside the
    grid, or a capped extension, which is flagged)."""
    if chosen["arm_type"] != "grid":
        return None, []
    zs = [float(z) for z in zs]
    side = "low" if chosen["z"] == min(zs) else "high" if chosen["z"] == max(zs) else None
    if side is None:
        return None, []
    match = [e for e in extensions if e["side"] == side and int(e["decay_steps"]) == chosen["decay_steps"]]
    if len(match) != 1:
        raise ValueError(f"expected one {side} extension at T {chosen['decay_steps']}, found {len(match)}")
    if match[0]["capped"]:
        return None, [f"{'lower' if side == 'low' else 'upper'}_extension_capped"]
    return match[0], []


def select_consft(arms: list[dict], floor: float, expected_base: set, zs: list, extensions: list[dict]) -> dict:
    """The rule over arm summaries (:func:`summary`). ``expected_base`` is the grid's and the Table 5 arm's
    keys; ``extensions`` the plan's extension arms (with ``side`` and ``capped``). While the owed extension
    is missing, ``status`` is ``"needs_extension"``; otherwise ``"selected"``."""
    base = [a for a in arms if a["arm_type"] in ("grid", "table5")]
    extra = {a["key"]: a for a in arms if a["arm_type"] == "extension"}
    have = {a["key"] for a in base}
    if have != set(expected_base) or len(base) != len(expected_base):
        raise ValueError(f"base arms {sorted(have)} are not the rule's {sorted(expected_base)}")
    first, first_flags, first_eligible, fallback = _apply(base, floor)
    owed, cap_flags = (None, []) if fallback else owed_extension(first, zs, extensions)
    unowed = set(extra) - ({owed["key"]} if owed else set())
    if unowed:
        raise ValueError(f"extension arms {sorted(unowed)} are published but not owed by the grid's selection")
    common = {"grid_selection": first["key"], "owed_extension": owed["key"] if owed else None}
    if owed and owed["key"] not in extra:
        return {"status": "needs_extension", "selected": None, "missing": owed["key"],
                "flags": first_flags + ["edge_extension_owed"], **common}
    if owed:
        chosen, flags, eligible, _ = _apply(base + [extra[owed["key"]]], floor)
        flags = flags + ["edge_extension"] + (["optimum_at_boundary"] if chosen["arm_type"] == "extension" else [])
    else:
        chosen, flags, eligible = first, first_flags + cap_flags, first_eligible
    return {"status": "selected", "selected": chosen, "missing": None, "flags": flags,
            "eligible": sorted(a["key"] for a in eligible), **common}


def exclusion_keys(rows: dict[str, dict], fingerprint_of=None) -> list[dict]:
    """Track 2's exclusion keys: ``(fingerprint, task_key)`` for both trained tasks of every arm, sorted.
    ``fingerprint_of(key)`` recomputes the fingerprint (from the checkpoint); otherwise the recorded one."""
    out = []
    for key, row in rows.items():
        fp = fingerprint_of(key) if fingerprint_of else row["consft"]["state_fingerprint"]
        for j in ("0", "1"):
            out.append({"fingerprint": fp, "task_key": row["cells"][j]["task_key"], "run_id": row["run_id"]})
    return sorted(out, key=lambda k: (k["run_id"], k["task_key"]))


# ---- selection, extension, verification -----------------------------------------------------------


def _floor(rows: dict, tolerance: float) -> tuple[float, float]:
    refs = {r["cells"]["1"]["reference_success"].get("seq_ft") for r in rows.values()}
    if len(refs) != 1 or None in refs:
        raise ValueError(f"the arms disagree on seq_ft's stage-1 Object: {refs}")
    ref = refs.pop()
    return ref, ref - tolerance


def _context(cfg: dict, root: Path, low_update_cfg: dict | None):
    from flowcl.experiments.low_update import seed_context

    return seed_context(low_update_cfg or load_low_update_config(), int(cfg["bracket_seed"]), root)


def _state(cfg: dict, root: Path, out_root: Path, rule: Path | None, low_update_cfg, allow_dirty: bool,
           fingerprints: bool = False) -> dict:
    plan, payload, cal_sha = validate_calibration(cfg, root)
    rule_sha = file_sha256(rule_path(rule))
    ctx = _context(cfg, root, low_update_cfg)
    rows = published_rows(cfg, out_root, plan, rule_sha, cal_sha)
    problems = validate_rows(cfg, rows, plan, ctx, out_root, rule_sha, cal_sha, allow_dirty=allow_dirty,
                             fingerprints=fingerprints)
    return {"plan": plan, "payload": payload, "cal_sha": cal_sha, "rule_sha": rule_sha, "ctx": ctx, "rows": rows,
            "problems": problems}


def _evaluate_rule(cfg: dict, st: dict) -> tuple[dict, float, float]:
    expected = set(base_arms(st["plan"]))
    missing = sorted(expected - set(st["rows"]))
    if missing:
        raise FileNotFoundError(f"bracket arms not published: {missing}")
    seq_ft_object, floor = _floor(st["rows"], float(cfg["consft"]["object_tolerance"]))
    arms = [summary(k, r) for k, r in st["rows"].items()]
    choice = select_consft(arms, floor, expected, cfg["consft"]["zs"], st["plan"]["extensions"])
    return choice, seq_ft_object, floor


def _bundle_arm(key: str, row: dict) -> dict:
    return {**summary(key, row), "checkpoint_sha256": row["checkpoint_sha256"],
            "state_fingerprint": row["consft"]["state_fingerprint"], "git_sha": row.get("git_sha"),
            "wall_clock_s": row["training"]["wall_clock_s"],
            "paired_vs_seq_ft": {j: row["cells"][j].get("paired", {}).get("seq_ft") for j in ("0", "1")}}


def check_bundle(bundle: dict, rule_sha: str, cal_sha: str) -> None:
    if (bundle.get("rule_sha256") != rule_sha or bundle.get("calibration_sha256") != cal_sha
            or bundle.get("family") != FAMILY):
        raise ValueError(f"the selection bundle was made under another rule or calibration "
                         f"(bundle {bundle.get('rule_sha256')} / {bundle.get('calibration_sha256')})")


def run_select(cfg: dict, *, results_root=None, out_root=None, rule: Path | None = None,
               low_update_cfg: dict | None = None, allow_dirty: bool = False) -> dict:
    """Apply the rule. Returns ``{"status": "needs_extension"}`` (nothing published) or the bundle."""
    from flowcl.experiments.sgp_sensitivity import publish_selection

    root = _root(results_root)
    out_root = Path(out_root) if out_root else root
    st = _state(cfg, root, out_root, rule, low_update_cfg, allow_dirty)
    if st["problems"]:
        raise ValueError(f"published arms fail validation: {st['problems']}")
    choice, seq_ft_object, floor = _evaluate_rule(cfg, st)
    if choice["status"] == "needs_extension":
        print(f"[flowcl] ConSFT selection: the grid selects {choice['grid_selection']}; the edge extension "
              f"{choice['missing']} runs before anything is published", flush=True)
        return {"status": "needs_extension", "missing": choice["missing"]}
    chosen, rows, seed = choice["selected"], st["rows"], int(cfg["bracket_seed"])
    key = chosen["key"]
    selection = {
        "git_sha": git_sha(), "seed": seed, "family": FAMILY, "rule_sha256": st["rule_sha"],
        "calibration_sha256": st["cal_sha"], "q50": st["payload"]["q50"],
        "inputs": shared_inputs(cfg, st["ctx"], st["ctx"].ref_dir / "checkpoints" / "stage0.pt", st["rule_sha"],
                                st["cal_sha"]),
        "seq_ft_object": seq_ft_object, "object_floor": floor,
        "arms": {k: _bundle_arm(k, rows[k]) for k in sorted(rows)},
        "grid_selection": choice["grid_selection"], "owed_extension": choice["owed_extension"],
        "eligible": choice["eligible"],
        "selected": {"key": key, "arm_type": chosen["arm_type"], "z": chosen["z"], "tau_start": chosen["tau_start"],
                     "decay_steps": chosen["decay_steps"], "constants": constants(cfg), "run_id": chosen["run_id"],
                     "checkpoint": str(arm_dir(cfg, out_root, key, seed) / "checkpoints" / "final.pt"),
                     "checkpoint_sha256": rows[key]["checkpoint_sha256"],
                     "state_fingerprint": rows[key]["consft"]["state_fingerprint"]},
        "flags": choice["flags"],
        "track2_exclusion_keys": exclusion_keys(rows),
    }
    published, sha = publish_selection(_out(cfg, out_root) / "consft", selection)
    check_bundle(published, st["rule_sha"], st["cal_sha"])
    print(f"[flowcl] ConSFT selection: {key} (tau_start {published['selected']['tau_start']:.6g}, T "
          f"{published['selected']['decay_steps']}) {published['flags']} (bundle {sha[:12]})", flush=True)
    return {"status": "selected", **published, "bundle_sha256": sha}


def run_extension(cfg: dict, *, results_root=None, out_root=None, rule: Path | None = None,
                  low_update_cfg: dict | None = None, allow_dirty: bool = False, **kw) -> dict:
    """The owed edge-extension arm; refuses unless the grid's selection owes one that is not published."""
    root = _root(results_root)
    out_root_p = Path(out_root) if out_root else root
    st = _state(cfg, root, out_root_p, rule, low_update_cfg, allow_dirty)
    if st["problems"]:
        raise ValueError(f"published arms fail validation: {st['problems']}")
    choice, _, _ = _evaluate_rule(cfg, st)
    if choice["status"] != "needs_extension":
        raise RuntimeError("no edge extension is owed: the selection lies inside the grid, is not a grid arm, "
                           "none qualifies, the extension is capped, or it is already published")
    return run_arm(cfg, choice["missing"], extension=True, results_root=results_root, out_root=out_root, rule=rule,
                   low_update_cfg=low_update_cfg, allow_dirty=allow_dirty, **kw)


def verify(cfg: dict, *, results_root=None, out_root=None, rule: Path | None = None,
           low_update_cfg: dict | None = None, allow_dirty: bool = False) -> dict:
    """Every arm, its inputs and its checkpoint tensors; the bundle; the selection and Track 2's exclusion
    keys recomputed (fingerprints from the checkpoints)."""
    from flowcl.experiments.sgp_sensitivity import load_selection

    root = _root(results_root)
    out_root = Path(out_root) if out_root else root
    problems, sha = [], None
    try:
        st = _state(cfg, root, out_root, rule, low_update_cfg, allow_dirty, fingerprints=True)
        problems += st["problems"]
        bundle, sha = load_selection(_out(cfg, out_root) / "consft")
        check_bundle(bundle, st["rule_sha"], st["cal_sha"])
        choice, seq_ft_object, floor = _evaluate_rule(cfg, st)
        if choice["status"] != "selected":
            problems.append(f"the rule still owes the extension {choice['missing']}")
        else:
            got = {"selected": choice["selected"]["key"], "flags": choice["flags"], "eligible": choice["eligible"],
                   "grid_selection": choice["grid_selection"], "owed_extension": choice["owed_extension"],
                   "object_floor": floor, "seq_ft_object": seq_ft_object}
            want = {"selected": bundle["selected"]["key"], "flags": bundle["flags"], "eligible": bundle["eligible"],
                    "grid_selection": bundle["grid_selection"], "owed_extension": bundle["owed_extension"],
                    "object_floor": bundle["object_floor"], "seq_ft_object": bundle["seq_ft_object"]}
            problems += [f"recomputed {k} {got[k]} != the bundle's {want[k]}" for k in got if got[k] != want[k]]
        seed = int(cfg["bracket_seed"])
        recomputed = exclusion_keys(st["rows"], lambda k: checkpoint_fingerprint(
            arm_dir(cfg, out_root, k, seed) / st["rows"][k]["checkpoint"]))
        if bundle.get("track2_exclusion_keys") != recomputed:
            problems.append("the bundle's Track 2 exclusion keys differ from the checkpoints'")
        if {k: (v["checkpoint_sha256"], v["state_fingerprint"]) for k, v in bundle["arms"].items()} != \
                {k: (r["checkpoint_sha256"], r["consft"]["state_fingerprint"]) for k, r in st["rows"].items()}:
            problems.append("the bundle's arms differ from the published rows")
        current = shared_inputs(cfg, st["ctx"], st["ctx"].ref_dir / "checkpoints" / "stage0.pt", st["rule_sha"],
                                st["cal_sha"])
        if bundle.get("inputs") != current:
            problems.append("the bundle's inputs differ from the current files")
    except (OSError, ValueError, RuntimeError, KeyError) as exc:
        problems.append(f"{type(exc).__name__}: {exc}")
    return {"passed": not problems, "problems": problems, "bundle_sha256": sha}


def status(cfg: dict, *, results_root=None, out_root=None, rule: Path | None = None,
           low_update_cfg: dict | None = None, allow_dirty: bool = False) -> dict:
    """Published and missing arms (no training)."""
    root = _root(results_root)
    out_root = Path(out_root) if out_root else root
    st = _state(cfg, root, out_root, rule, low_update_cfg, allow_dirty)
    base = set(base_arms(st["plan"]))
    return {"rule_sha256": st["rule_sha"], "calibration_sha256": st["cal_sha"], "published": sorted(st["rows"]),
            "missing": sorted(base - set(st["rows"])), "row_problems": st["problems"],
            "ready": not st["problems"]}


# ---- smoke ----------------------------------------------------------------------------------------


def smoke_arm(cfg: dict, *, tau_start: float, decay_steps: int, reference_run: Path, identity_run: Path,
              steps: int = 200, episodes: int = 1, out_root: Path, **kw) -> dict:
    """The bracket path against the runner, before the queue: an arm from ``reference_run``'s (a scratch
    seq_ft run's) ``stage0.pt`` at ``steps`` steps must train under the same TrainConfig as
    ``identity_run``'s (a scratch ConSFT run's) stage 1, reproduce its ``stage1.pt`` tensor for tensor, and
    log the same weighting; the logged τ must follow eq. 11–12. The per-step time is reported against the
    scratch seq_ft run's stage 1."""
    from flowcl.methods.consft import temperature
    from flowcl.train.continual import stage_identity_check

    reference_run, identity_run = Path(reference_run), Path(identity_run)
    row = run_arm(cfg, "smoke", out_root=out_root, smoke_params={"tau_start": tau_start, "decay_steps": decay_steps},
                  smoke_start=reference_run / "checkpoints" / "stage0.pt", smoke_steps=steps,
                  smoke_episodes=episodes, **kw)
    directory = arm_dir(cfg, out_root, "smoke", int(cfg["bracket_seed"]))
    runner = json.loads((identity_run / "result.json").read_text())["stages"][1]
    arm_config, runner_config = _arm_train_config(directory), runner.get("train_config")
    if arm_config != runner_config:
        return {"passed": False, "reason": "the arm's TrainConfig differs from the runner's stage-1 TrainConfig",
                "arm": arm_config, "runner": runner_config}
    check = stage_identity_check(directory / "checkpoints" / "final.pt", None, 1,
                                 reference_checkpoint=identity_run / "checkpoints" / "stage1.pt")
    payload = torch.load(identity_run / "checkpoints" / "stage1.pt", map_location="cpu", weights_only=False)
    runner_logs = payload["extra"]["method_state"]["logs"]["1"]
    logs_equal = row["consft"]["logs"] == runner_logs
    c = constants(cfg)
    tau_ok = all(abs(rec["tau"] - temperature(int(t), tau_start, decay_steps, c["decay_rate"], c["curve"],
                                              c["tau_end"])) <= 1e-12 * max(1.0, rec["tau"])
                 for t, rec in row["consft"]["logs"]["steps"].items())
    omega_ok = all(c["omega_min"] - 1e-9 <= rec["omega_mean"] <= 1.0 for rec in row["consft"]["logs"]["steps"].values())
    seq_ft_stage1 = json.loads((reference_run / "result.json").read_text())["stages"][1]
    overhead = runner["train_wall_clock_s"] / seq_ft_stage1["train_wall_clock_s"] - 1.0
    return {"passed": bool(check["passed"] and logs_equal and tau_ok and omega_ok), "train_config_equal": True,
            "identity": {k: check[k] for k in ("n_tensors", "n_different", "different", "missing", "extra")},
            "logs_equal": logs_equal, "tau_follows_eq12": tau_ok, "omega_in_range": omega_ok,
            "stage1_wall_clock_s": {"consft": runner["train_wall_clock_s"],
                                    "seq_ft": seq_ft_stage1["train_wall_clock_s"]},
            "overhead_vs_seq_ft": overhead, "run_id": row["run_id"]}


def write_smoke_method(path: Path, tau_start: float, decay_steps: int, consts: dict) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    OmegaConf.save(OmegaConf.create({"name": "consft", "tau_start": float(tau_start), "decay_steps": int(decay_steps),
                                     **consts}), path)
    return path
