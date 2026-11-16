"""The f = 0.7 adaptive GPM queue's checks (``scripts/queue_ne70.sh``; pre-E3 plan, A3).

E3 (docs/thesis_plan.md) puts adaptive GPM at ``new_energy_fraction`` 0.7 next to plain GPM and f = 0.9.
A from-scratch run on ``seq_hetero_t8`` would get the seed namespace ``seq_hetero_t8__seq_ft__seed<S>``
(the runner derives it from the curriculum name) and pair with none of the arms E3 resumes from the
4-task runs. So f = 0.7 first runs as a 4-task ``seq_hetero`` run, paired with the existing runs of the
same seed, and E3 resumes it at stage 4 like the others.

By construction its stage 0, Task-1 memory and stage 1 equal plain GPM's (at T1 the memory target is
eps): the runner checks stages 0-1 bitwise against plain GPM, and :func:`check_run` the Task-1 memory
basis for basis.

* :func:`preflight`, before any GPU work: a clean tree, an idle GPU, enough free disk, no target run
  directory yet, the method config, and the seed's references by SHA-256 (seq_ft's stage 0, plain GPM's
  stages 0-1 and Task-1 memory). A smoke checks its own references exist instead of their hashes.
* :func:`check_run`, after each run: identity at exactly stages {0, 1} against plain GPM, recorded and
  passed against the reference's current SHA-256; T1 pairing passed; Task-1 memory equal to plain GPM's.
"""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

from omegaconf import OmegaConf

from flowcl.data.config import load_method_config
from flowcl.experiments.baseline_queue import GPM_MEMORY0, SEQ_FT_STAGE0, _hashes, gpu_compute_pids
from flowcl.utils.libero_paths import repo_root
from flowcl.utils.run import file_sha256

CURRICULUM = "seq_hetero"
METHOD = "gpm_ne70"
LABEL = "gpm_projected_adam_ne70"
NEW_ENERGY_FRACTION = 0.70
IDENTITY_STAGES = [0, 1]
# Plain GPM's stage 0 and stage 1 per seed: the references the f = 0.9 and e99 runs were identity-checked
# against (their identity_stage{0,1}.json reference_sha256).
GPM_STAGE = {
    0: {0: "aaa4ea42edd3977ea660a3846e463e07ffec67013ac9f035396292353480f0c0",
        1: "97fb549183527b5645265f388cb16c1dca8b0468d90bee71fd81b5268c0c0053"},
    1: {0: "ecb40b2830ce4742ddc00d2339a03e853c892a4fa0afbd5d1051c2b86e3bde6d",
        1: "9a500ad870033bf635bce82c7171d6e06c4e10cfccb13ac6ac8848875e89fadf"},
    2: {0: "fb94b01d358723f029375e1abeae9518727867dd91cd74535e2cac21e2a7305d",
        1: "0ca9e5f0746e3931d68b386f31a3db9a7f1ccd535d07cef21e6aa0500540aa1f"},
}
# One 4-task GPM-family run is about 3.5 GB (checkpoints and memories); a 200-step smoke about 2.5 GB.
MIN_FREE_GB = {False: 15.0, True: 4.0}


def run_name(method_label: str, seed: int) -> str:
    return f"{CURRICULUM}__{method_label}__seed{seed}"


def free_gb(path: Path) -> float:
    return shutil.disk_usage(path).free / 1e9


def config_check() -> dict:
    """gpm_ne70.yaml is gpm.yaml plus new_energy_fraction 0.7, nothing else."""
    name, kwargs = load_method_config(METHOD)
    plain_name, plain = load_method_config("gpm")
    kwargs = dict(kwargs)
    f = kwargs.pop("new_energy_fraction", None)
    problems = []
    if name != plain_name:
        problems.append(f"registry name {name}, gpm's is {plain_name}")
    if f != NEW_ENERGY_FRACTION:
        problems.append(f"new_energy_fraction {f}, expected {NEW_ENERGY_FRACTION}")
    if kwargs != plain:
        problems.append(f"differs from gpm.yaml beyond new_energy_fraction: {kwargs} vs {plain}")
    return {"passed": not problems, "problems": problems}


def preflight(seeds: list[int], *, results: Path | None = None, smoke: bool = False,
              refs: Path | None = None, root: Path | None = None, gpu_pids=gpu_compute_pids,
              disk=free_gb) -> dict:
    """Every check before the queue's first GPU step; ``passed`` false refuses the launch."""
    root = Path(root) if root else repo_root()
    results = Path(results) if results else root / "results"
    refs = Path(refs) if refs else results
    checks: dict = {}
    if not smoke:
        status = subprocess.run(["git", "status", "--porcelain"], cwd=root, capture_output=True,
                                text=True, check=True).stdout
        checks["clean_tree"] = {"passed": not status.strip(), "changes": status.splitlines()}
    pids = gpu_pids()
    checks["gpu_idle"] = {"passed": not pids, "compute_pids": pids}
    free = disk(results if results.exists() else results.parent)
    checks["disk"] = {"passed": free >= MIN_FREE_GB[smoke], "free_gb": round(free, 1),
                      "needed_gb": MIN_FREE_GB[smoke]}
    existing = [str(results / run_name(LABEL, s)) for s in seeds if (results / run_name(LABEL, s)).exists()]
    checks["targets_absent"] = {"passed": not existing, "existing": existing}
    checks["config"] = config_check()
    files = {}
    for s in seeds:
        seq_ft, gpm = refs / run_name("seq_ft", s), refs / run_name("gpm_projected_adam", s)
        wanted = {f"seq_ft_stage0_s{s}": (seq_ft / "checkpoints" / "stage0.pt", SEQ_FT_STAGE0.get(s)),
                  f"gpm_stage0_s{s}": (gpm / "checkpoints" / "stage0.pt", GPM_STAGE.get(s, {}).get(0)),
                  f"gpm_stage1_s{s}": (gpm / "checkpoints" / "stage1.pt", GPM_STAGE.get(s, {}).get(1)),
                  f"gpm_memory0_s{s}": (gpm / "method" / "memory_task0.pt", GPM_MEMORY0.get(s))}
        files.update(wanted)
    if smoke:  # the smoke's references are smoke runs: they must exist, their hashes are their own
        missing = [str(p) for p, _ in files.values() if not p.is_file()]
        checks["references"] = {"passed": not missing, "missing": missing}
    else:
        checks["references"] = _hashes(files)
    return {"passed": all(c["passed"] for c in checks.values()), "seeds": seeds, "smoke": smoke,
            "checks": checks}


def check_run(run: Path, gpm_run: Path) -> dict:
    """A finished f = 0.7 run against plain GPM ``gpm_run`` of the same seed."""
    from flowcl.experiments.lora_selection import compare_memories

    run, gpm_run = Path(run), Path(gpm_run)
    result_path = run / "result.json"
    if not result_path.is_file():
        return {"passed": False, "problems": [f"no result.json in {run}"]}
    result = json.loads(result_path.read_text())
    prov = OmegaConf.to_container(OmegaConf.load(run / "config.yaml"), resolve=True).get("provenance") or {}
    problems = []
    configured = sorted(set(prov.get("identity_stages") or [])
                        | {int(k) for k in prov.get("identity_reference_checkpoints") or {}})
    if configured != IDENTITY_STAGES:
        problems.append(f"identity stages configured {configured}, the queue expects {IDENTITY_STAGES}")
    if Path(prov.get("identity_reference_run") or "") != gpm_run:
        problems.append(f"identity reference {prov.get('identity_reference_run')}, expected {gpm_run}")
    recorded = result.get("identity_checks") or {}
    for stage in IDENTITY_STAGES:
        entry = recorded.get(str(stage))
        if entry is None or entry.get("passed") is not True:
            problems.append(f"stage {stage}: identity {'missing' if entry is None else 'failed'}")
            continue
        reference = gpm_run / "checkpoints" / f"stage{stage}.pt"
        if entry.get("reference_sha256") != file_sha256(reference):
            problems.append(f"stage {stage}: checked against sha256 {entry.get('reference_sha256')}, "
                            f"{reference} has another")
    if (result.get("t1_pairing") or {}).get("passed") is not True:
        problems.append("T1 pairing missing or failed")
    memory = compare_memories(gpm_run / "method" / "memory_task0.pt", run / "method" / "memory_task0.pt")
    if not memory["passed"]:
        problems.append(f"Task-1 memory differs from plain GPM's on {len(memory['different'])} layers")
    return {"passed": not problems, "problems": problems, "identity_stages": configured, "memory": memory}
