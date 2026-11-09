"""The 4-task baseline queue's checks (``scripts/queue_baselines.sh``; Workstream B, Track 2's test runs).

Six methods on ``seq_hetero``, each paired with seq_ft of the same seed: replay, EWC, LoRA, ConSFT and the
two projected-LoRA variants. Their settings are frozen in selection bundles from seed 0's T1 -> T2
brackets. This module holds what the queue checks outside the runner:

* :func:`preflight`, before any GPU work: a clean tree, an idle GPU, no target run directory yet, the
  seed's reference files by SHA-256, every bundle and its selected arm's checkpoint by SHA-256, the arm's
  evaluation on seed 0, and each production method config against the arm's recorded ``method`` block
  (every hyperparameter equal, every other production key a runner-only key with its fixed value). On
  seed 0 the runner also checks stage 1 against the arm bitwise; seeds 1-2 have only this config check.
* :func:`check_run`, after each run: the identity stages the queue expects ({0, 1} for a bracketed method
  on seed 0, else {0}) are exactly the configured ones and passed, and for the projected variants the
  Task-1 memory equals GPM's of the same seed, basis for basis.
* :func:`check_report`, on ``--from-step``: a prior report passes strict provenance again.
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

from omegaconf import OmegaConf

from flowcl.data.config import load_method_config
from flowcl.utils.libero_paths import repo_root
from flowcl.utils.run import file_sha256

CURRICULUM = "seq_hetero"
METHODS = ("ewc", "lora", "replay", "consft", "inflora", "projected_lora")   # the queue's order
PROJECTED = ("inflora", "projected_lora")

# Seed-specific references: seq_ft's stage 0 (T1 pairing and stage-0 identity) and gpm_projected_adam's
# Task-1 memory (the projected variants' memory must equal it).
SEQ_FT_STAGE0 = {
    0: "21a3e422e9ca3fa468a61ce8d9d48f71592acafa2d3d975827b3ef28c2e8e506",
    1: "d4b1c555d26160cb6fd38778c957ddb41aaaa5b74820608f7fcd3c4447d08b4f",
    2: "3e0eacd0b799ceef156a0df8e2850c6f09339e6a516dc256704ebebe67959605",
}
GPM_MEMORY0 = {
    0: "93b699356e97441d5e207d9654b6360fa361470371b65c2e6ad6be26d97fac45",
    1: "aa1f6e7af92657373e05805fcd46ead0d8901db719ed8eafe4c09a75d49bb5bf",
    2: "b2a122320a8da8d68b3520d7d551f6e6fa3a41e0dfd34c1152207b7e854afbc9",
}

# The frozen selection bundles (seed 0's brackets) and their selected arms. `compare` maps a production
# key to its path in the arm's recorded config.yaml `method` block; `runner_only` are production keys the
# arm path does not carry, with the only value allowed.
BRACKETS = {
    "ewc": {
        "bundle": "results/baseline_selection_v2/ewc/selection/selection.json",
        "bundle_sha256": "f09bc3ee99a334b261958f5742f5f870f04c27c46a996e086b886f36b1f2396e",
        "arm": "results/baseline_selection_v2/ewc/seed0/ewcb__r100__seed0",
        "final_sha256": "8af2f1e342d00cf63f5c203be473ab110fa8fc8dc42cbb476084ca5b0a138abc",
        "compare": {k: (k,) for k in ("lam", "fisher_samples", "fisher_estimator", "fisher_normalization")},
        "runner_only": {"log_interval": 100},
    },
    "lora": {
        "bundle": "results/baseline_selection_lora_v2/lora/selection/selection.json",
        "bundle_sha256": "455edc428c1352e90daae00556a512d7beeb6c1adc29e874eca0240944b8cf3b",
        "arm": "results/baseline_selection_lora_v2/lora/seed0/lorab__r16__lr3e-05__seed0",
        "final_sha256": "d9d1fd21fdef458d378fb55634325e9bac80d1ae395e63c0b050151e801ab215",
        "compare": {k: (k,) for k in ("rank", "alpha", "lr")},
        "runner_only": {},
    },
    "consft": {
        "bundle": "results/baseline_selection_consft/consft/selection/selection.json",
        "bundle_sha256": "b0f20f54b7f549150443efc9cd47ac118b30e6e4e628b0e2c19267a43bf0cad1",
        "arm": "results/baseline_selection_consft/consft/seed0/consftb__z1000__T30000__seed0",
        "final_sha256": "226da77eefbcd6be7444deae24b6240a0a2313afbe3c966d2be4a9da68b34d0e",
        "compare": {k: (k,) for k in ("tau_start", "decay_steps", "kappa", "decay_rate", "curve", "omega_min",
                                      "tau_end")},
        "runner_only": {"log_interval": 100},
    },
    "projected_lora": {
        "bundle": "results/baseline_selection_projected_lora/lora/selection/selection.json",
        "bundle_sha256": "e60d637ca09d3f22a26583632c91a084b172c5b2d0533bec1a59f3d744841375",
        "arm": "results/baseline_selection_projected_lora/lora/seed0/plorab__r64__lr0.0003__seed0",
        "final_sha256": "756e4a89db81d7d99cc369fefb85b92cbffdd85bfd04c313729b8e7a857e7376",
        "compare": {"rank": ("rank",), "alpha": ("alpha",), "lr": ("lr",), "variant": ("projection", "variant"),
                    "eps": ("projection", "eps")},
        "runner_only": {"update_memory": True, "capture_config": "subspace"},
    },
    "inflora": {
        "bundle": "results/baseline_selection_inflora/lora/selection/selection.json",
        "bundle_sha256": "dda181afc612c765feb03914f6afb3e4aa6b43c297d8a90f2ca70ad6be16ddf9",
        "arm": "results/baseline_selection_inflora/lora/seed0/inflb__r16__lr0.0001__seed0",
        "final_sha256": "473fc338c0f136252a74becd5e4e62c656f4cc9807e3820d4ca46720640c1f12",
        "compare": {"rank": ("rank",), "alpha": ("alpha",), "lr": ("lr",), "variant": ("projection", "variant"),
                    "eps": ("projection", "eps")},
        "runner_only": {"update_memory": True, "capture_config": "subspace"},
    },
}
# Replay has no bracket: its spec values are fixed a priori (configs/method/replay.yaml).
FIXED = {"replay": {"buffer_demos": 10, "replay_fraction": 0.5}}
# The bundle's `selected` hyperparameters, checked against the production config as well.
SELECTED_KEYS = {"ewc": ("lam",), "lora": ("rank", "lr", "alpha"), "consft": ("tau_start", "decay_steps"),
                 "projected_lora": ("rank", "lr", "alpha"), "inflora": ("rank", "lr", "alpha")}


def run_dir(results: Path, method: str, seed: int) -> Path:
    """The run's directory. Its method field is the config's name: both projected variants have registry
    name projected_lora and display names `projected_lora` and `inflora`."""
    return Path(results) / f"{CURRICULUM}__{method}__seed{seed}"


def expected_identity_stages(method: str, seed: int) -> list[int]:
    return [0, 1] if seed == 0 and method in BRACKETS else [0]


def arm_checkpoint(method: str, root: Path | None = None) -> Path:
    return (Path(root) if root else repo_root()) / BRACKETS[method]["arm"] / "checkpoints" / "final.pt"


def _equal(a, b) -> bool:
    if isinstance(a, bool) or isinstance(b, bool):
        return a is b
    if isinstance(a, (int, float)) and isinstance(b, (int, float)):
        return float(a) == float(b)
    return a == b


def _get(block: dict, path: tuple[str, ...]):
    for key in path:
        if not isinstance(block, dict) or key not in block:
            return KeyError
        block = block[key]
    return block


def config_check(method: str, root: Path | None = None) -> dict:
    """The production config (configs/method/<method>.yaml) against its bracket arm's recorded method
    block, its bundle's `selected` row and rule hash (replay: its fixed spec values)."""
    root = Path(root) if root else repo_root()
    _, production = load_method_config(method)
    problems = []
    if method in FIXED:
        expected = FIXED[method]
        for key in sorted(set(production) | set(expected)):
            if key not in production or key not in expected or not _equal(production[key], expected[key]):
                problems.append(f"{key}: production {production.get(key)!r}, fixed {expected.get(key)!r}")
        return {"passed": not problems, "problems": problems}

    spec = BRACKETS[method]
    bundle = json.loads((root / spec["bundle"]).read_text())
    arm_dir = root / spec["arm"]
    arm_cfg = OmegaConf.to_container(OmegaConf.load(arm_dir / "config.yaml"), resolve=True)
    arm_method = arm_cfg["method"]
    selected = bundle["selected"]
    if selected.get("run_id") != arm_dir.name:
        problems.append(f"bundle selects {selected.get('run_id')!r}, not {arm_dir.name!r}")
    if selected.get("checkpoint_sha256") != spec["final_sha256"]:
        problems.append("bundle's selected checkpoint_sha256 differs from the queue's")
    if bundle.get("rule_sha256") != arm_method.get("rule_sha256"):
        problems.append("bundle and arm record different rule_sha256")
    for key, path in spec["compare"].items():
        value = _get(arm_method, path)
        if value is KeyError:
            problems.append(f"{key}: the arm records no {'.'.join(path)}")
        elif key not in production or not _equal(production[key], value):
            problems.append(f"{key}: production {production.get(key)!r}, arm {value!r}")
    for key in SELECTED_KEYS[method]:
        if key not in selected or key not in production or not _equal(production[key], selected[key]):
            problems.append(f"{key}: production {production.get(key)!r}, bundle selected {selected.get(key)!r}")
    for key, value in production.items():
        if key in spec["compare"]:
            continue
        if key not in spec["runner_only"]:
            problems.append(f"{key}: a production key the arm does not record and not runner-only")
        elif not _equal(value, spec["runner_only"][key]):
            problems.append(f"{key}: production {value!r}, runner-only value {spec['runner_only'][key]!r}")
    if method in PROJECTED:
        memory = _get(arm_method, ("projection", "memory_sha256"))
        if memory != GPM_MEMORY0[0]:
            problems.append(f"the arm's memory_sha256 {memory!r} is not GPM seed 0's")
    return {"passed": not problems, "problems": problems}


def gpu_compute_pids() -> list[str]:
    out = subprocess.run(["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader"],
                         capture_output=True, text=True, check=True).stdout
    return [line.strip() for line in out.splitlines() if line.strip()]


def preflight(seed: int, methods: list[str], *, results: Path | None = None, smoke: bool = False,
              root: Path | None = None, gpu_pids=gpu_compute_pids) -> dict:
    """Every check before the queue's first GPU step; ``passed`` false refuses the launch.

    ``smoke`` skips the clean tree and the seed's reference hashes (the smoke's references are its own
    runs); everything else, the GPU, the targets, the bundles, the arms and the configs, is checked."""
    root = Path(root) if root else repo_root()
    results = Path(results) if results else root / "results"
    checks: dict = {}
    if not smoke:
        status = subprocess.run(["git", "status", "--porcelain"], cwd=root, capture_output=True,
                                text=True, check=True).stdout
        checks["clean_tree"] = {"passed": not status.strip(), "changes": status.splitlines()}
    pids = gpu_pids()
    checks["gpu_idle"] = {"passed": not pids, "compute_pids": pids}
    existing = [str(run_dir(results, m, seed)) for m in methods if run_dir(results, m, seed).exists()]
    checks["targets_absent"] = {"passed": not existing, "existing": existing}
    if not smoke:
        refs = {
            "seq_ft_stage0": (root / "results" / f"{CURRICULUM}__seq_ft__seed{seed}" / "checkpoints" / "stage0.pt",
                              SEQ_FT_STAGE0[seed]),
            "gpm_memory0": (root / "results" / f"{CURRICULUM}__gpm_projected_adam__seed{seed}" / "method"
                            / "memory_task0.pt", GPM_MEMORY0[seed]),
        }
        checks["references"] = _hashes(refs)
    bracket_files = {}
    for method in BRACKETS:
        spec = BRACKETS[method]
        bracket_files[f"{method}_bundle"] = (root / spec["bundle"], spec["bundle_sha256"])
        bracket_files[f"{method}_final"] = (arm_checkpoint(method, root), spec["final_sha256"])
    checks["brackets"] = _hashes(bracket_files)
    if seed == 0:
        missing = [str(root / BRACKETS[m]["arm"] / "eval.json") for m in BRACKETS
                   if not (root / BRACKETS[m]["arm"] / "eval.json").is_file()]
        checks["arm_evals"] = {"passed": not missing, "missing": missing}
    configs = {}
    for method in METHODS:
        try:
            configs[method] = config_check(method, root)
        except (OSError, KeyError, ValueError) as err:
            configs[method] = {"passed": False, "problems": [f"{type(err).__name__}: {err}"]}
    checks["configs"] = {"passed": all(c["passed"] for c in configs.values()), "methods": configs}
    return {"passed": all(c["passed"] for c in checks.values()), "seed": seed, "smoke": smoke,
            "methods": methods, "checks": checks}


def _hashes(files: dict[str, tuple[Path, str]]) -> dict:
    out, ok = {}, True
    for name, (path, expected) in files.items():
        got = file_sha256(path) if Path(path).is_file() else None
        out[name] = {"path": str(path), "sha256": got, "expected": expected, "passed": got == expected}
        ok = ok and got == expected
    return {"passed": ok, "files": out}


def check_run(run: Path, method: str, seed: int, gpm_memory: Path | None = None) -> dict:
    """A finished run: its configured identity stages are exactly the queue's, recorded and passed; for a
    projected variant, its Task-1 memory equals ``gpm_memory`` basis for basis."""
    run = Path(run)
    problems = []
    result_path = run / "result.json"
    if not result_path.is_file():
        return {"passed": False, "problems": [f"no result.json in {run}"]}
    result = json.loads(result_path.read_text())
    prov = OmegaConf.to_container(OmegaConf.load(run / "config.yaml"), resolve=True).get("provenance") or {}
    configured = sorted(set(prov.get("identity_stages") or [])
                        | {int(k) for k in prov.get("identity_reference_checkpoints") or {}})
    expected = expected_identity_stages(method, seed)
    if configured != expected:
        problems.append(f"identity stages configured {configured}, the queue expects {expected}")
    recorded = result.get("identity_checks") or {}
    for stage in expected:
        entry = recorded.get(str(stage))
        if entry is None or entry.get("passed") is not True:
            problems.append(f"stage {stage}: identity {'missing' if entry is None else 'failed'}")
    if (result.get("t1_pairing") or {}).get("passed") is not True:
        problems.append("T1 pairing missing or failed")
    out = {"identity_stages": configured}
    if method in PROJECTED:
        from flowcl.experiments.lora_selection import compare_memories

        if gpm_memory is None:
            problems.append("a projected variant needs GPM's Task-1 memory to compare with")
        else:
            memory = compare_memories(Path(gpm_memory), run / "method" / "memory_task0.pt")
            out["memory"] = memory
            if not memory["passed"]:
                problems.append(f"Task-1 memory differs from {gpm_memory} on {len(memory['different'])} layers")
    return {"passed": not problems, "problems": problems, **out}


def check_report(path: Path, allow_dirty: bool = False) -> dict:
    """A prior report on disk passes strict provenance again."""
    from flowcl.experiments.sequence_report import strict_failures

    path = Path(path)
    if not path.is_file():
        return {"passed": False, "problems": [f"missing {path}"]}
    failures = strict_failures(json.loads(path.read_text()), allow_dirty=allow_dirty)
    return {"passed": not failures, "problems": failures}
