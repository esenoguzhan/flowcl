"""Thesis figures, one per claim (status.md §5.2), into docs/assets/figures/.

    uv run python scripts/make_figures.py [--only NAME ...] [--out-dir DIR] [--n-boot N]

Every plot shows each seed separately (three seeds give no seed-level interval), marks seed 0 as
the development seed, and states the rollouts per cell. Intervals are rollout-level 95% bootstraps
(A5: flowcl/analysis/rollout_bootstrap.py, flowcl/analysis/metrics.py), paired where two runs share
a seed namespace. The script writes ``figures.json`` with every figure's state: ``made``,
``waiting for data`` (the E3 and hardware-success figures) or ``manual`` (the policy diagram).
"""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from flowcl.analysis.metrics import paired_difference_ci, success_estimate  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
SEEDS = (0, 1, 2)
TASKS4 = ("Spatial", "Object", "Goal", "LIBERO-10")
METHODS = {  # label -> run-name part on seq_hetero
    "seq_ft": "seq_ft", "EWC": "ewc", "LoRA": "lora", "ConSFT": "consft", "replay": "replay",
    "InfLoRA": "inflora", "projected LoRA": "projected_lora", "GPM plain": "gpm_projected_adam",
    "GPM adaptive f0.9": "gpm_projected_adam_ne90", "GPM e99": "gpm_projected_adam_e99",
}
COLORS = dict(zip(METHODS, plt.get_cmap("tab10").colors))
MARKERS = {0: "o", 1: "s", 2: "^"}
N_BOOT = 2000


def seed_label(s: int) -> str:
    return f"seed {s} (dev)" if s == 0 else f"seed {s}"


def result(run: str) -> dict:
    return json.loads((RESULTS / run / "result.json").read_text())


def run_name(method: str, seed: int, curriculum: str = "seq_hetero") -> str:
    return f"{curriculum}__{METHODS.get(method, method)}__seed{seed}"


def cell(payload: dict, stage: int, task: int) -> tuple[list[bool], list]:
    key = payload["task_keys"][task]
    for record in payload["stages"]:
        if record["stage"] == stage:
            for entry in record["evaluation"]["tasks"]:
                if entry["task_key"] == key:
                    return entry["successes"], entry.get("seeds") or []
    raise KeyError(f"{payload['run_id']}: no cell R[{stage}][{task}]")


def estimate(payload: dict, stage: int, task: int):
    return success_estimate(cell(payload, stage, task)[0], n_bootstrap=N_BOOT)


def paired(a: dict, b: dict, stage: int, task: int):
    (sa, ea), (sb, eb) = cell(a, stage, task), cell(b, stage, task)
    if ea != eb:
        raise ValueError(f"{a['run_id']} and {b['run_id']}: R[{stage}][{task}] not paired (episode seeds differ)")
    return paired_difference_ci(np.asarray(sa, float), np.asarray(sb, float), n_bootstrap=N_BOOT)


def save(fig, out: Path, name: str) -> str:
    out.mkdir(parents=True, exist_ok=True)
    path = out / f"{name}.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return str(path.relative_to(ROOT)) if path.resolve().is_relative_to(ROOT) else str(path)


def errbar(ax, x, est, **kw):
    ax.errorbar([x], [100 * est.value], yerr=[[100 * (est.value - est.low)], [100 * (est.high - est.value)]],
                capsize=3, **kw)


# ---- characterization and baselines ---------------------------------------------------------------


def fig_retention_grids(out: Path) -> dict:
    files = []
    for s in SEEDS:
        fig, axes = plt.subplots(2, 5, figsize=(17, 7.2))
        for ax, method in zip(axes.flat, METHODS):
            p = result(run_name(method, s))
            values = np.array([[np.nan if v is None else 100 * v for v in row]
                               for row in p["retention_matrix"]["values"]])
            ax.imshow(values, vmin=0, vmax=100, cmap="viridis")
            for i in range(4):
                for j in range(4):
                    v = values[i, j]
                    ax.text(j, i, "-" if np.isnan(v) else f"{v:.0f}", ha="center", va="center", fontsize=8,
                            color="white" if np.isnan(v) or v < 60 else "black")
            ax.set_title(method, fontsize=10)
            ax.set_xticks(range(4), [t[:4] for t in TASKS4], fontsize=7)
            ax.set_yticks(range(4), [f"after T{i + 1}" for i in range(4)], fontsize=7)
        fig.suptitle(f"Retention R[i][j], seq_hetero, {seed_label(s)}; 50 rollouts per cell", fontsize=12)
        files.append(save(fig, out, f"retention_grid_seed{s}"))
    return {"claim": "sequential fine-tuning forgets everything; which methods keep what", "files": files}


def fig_forgetting_curves(out: Path) -> dict:
    shown = ["seq_ft", "EWC", "LoRA", "ConSFT", "replay", "InfLoRA", "GPM plain", "GPM adaptive f0.9"]
    fig, axes = plt.subplots(3, 3, figsize=(14, 10), sharey=True)
    for col, s in enumerate(SEEDS):
        runs = {m: result(run_name(m, s)) for m in shown}
        for row in range(3):
            ax = axes[row, col]
            for m in shown:
                xs = list(range(row, 4))
                ests = [estimate(runs[m], i, row) for i in xs]
                ax.errorbar([x + 1 for x in xs], [100 * e.value for e in ests],
                            yerr=[[100 * (e.value - e.low) for e in ests], [100 * (e.high - e.value) for e in ests]],
                            marker="o", ms=3, capsize=2, color=COLORS[m], label=m, lw=1)
            ax.set_xticks(range(1, 5), [f"T{i}" for i in range(1, 5)])
            ax.set_ylim(-3, 103)
            ax.grid(alpha=0.3)
            if col == 0:
                ax.set_ylabel(f"success on {TASKS4[row]} (%)")
            if row == 0:
                ax.set_title(seed_label(s))
            if row == 2:
                ax.set_xlabel("after training task")
    axes[0, 2].legend(fontsize=7, loc="lower left")
    fig.suptitle("Forgetting curves per task, each seed separately; 50 rollouts per point, rollout 95% CI")
    return {"claim": "when each task is lost, by method", "files": [save(fig, out, "forgetting_curves")]}


def fig_baselines(out: Path) -> dict:
    shown = ["seq_ft", "EWC", "LoRA", "ConSFT", "InfLoRA", "projected LoRA", "replay"]
    fig, axes = plt.subplots(1, 4, figsize=(15, 4.2), sharey=True)
    for j, ax in enumerate(axes):
        for y, m in enumerate(shown):
            for s in SEEDS:
                e = estimate(result(run_name(m, s)), 3, j)
                ax.errorbar([100 * e.value], [y + (s - 1) * 0.18],
                            xerr=[[100 * (e.value - e.low)], [100 * (e.high - e.value)]], capsize=2,
                            marker=MARKERS[s], ms=5, color=COLORS[m],
                            mfc="white" if m == "replay" else COLORS[m])
        ax.set_title(f"final success on {TASKS4[j]}")
        ax.set_xlim(-3, 103)
        ax.grid(alpha=0.3, axis="x")
        ax.set_yticks(range(len(shown)), [m + (" (not exemplar-free)" if m == "replay" else "") for m in shown])
    handles = [plt.Line2D([], [], marker=MARKERS[s], ls="", color="gray", label=seed_label(s)) for s in SEEDS]
    axes[-1].legend(handles=handles, fontsize=8, loc="lower right")
    fig.suptitle("4-task baselines, final row after T4: each seed a point, rollout 95% CI, 50 rollouts per cell")
    return {"claim": "the baselines; replay keeps most and is not exemplar-free",
            "files": [save(fig, out, "baselines_final")]}


# ---- adaptive GPM and its controls ----------------------------------------------------------------


def fig_adaptive_vs_plain(out: Path) -> dict:
    cells = {"Object after Goal, R[2][1]": (2, 1), "final Object, R[3][1]": (3, 1)}
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2), sharey=True)
    notes = {}
    for ax, (title, (i, j)) in zip(axes, cells.items()):
        for s in SEEDS:
            plain, adaptive = result(run_name("GPM plain", s)), result(run_name("GPM adaptive f0.9", s))
            ep, ea = estimate(plain, i, j), estimate(adaptive, i, j)
            ax.plot([s - 0.15, s + 0.15], [100 * ep.value, 100 * ea.value], color="gray", lw=0.8)
            errbar(ax, s - 0.15, ep, marker="o", color=COLORS["GPM plain"])
            errbar(ax, s + 0.15, ea, marker="o", color=COLORS["GPM adaptive f0.9"])
            d = paired(adaptive, plain, i, j)
            ax.annotate(f"{100 * d.value:+.0f} [{100 * d.low:+.0f}, {100 * d.high:+.0f}]",
                        (s, 102), ha="center", fontsize=8)
            notes[f"{title} seed {s}"] = round(100 * d.value)
        ax.set_xticks(SEEDS, [seed_label(s) for s in SEEDS])
        ax.set_title(title)
        ax.set_ylim(-3, 110)
        ax.grid(alpha=0.3, axis="y")
    axes[0].set_ylabel("success (%)")
    axes[0].legend(handles=[plt.Line2D([], [], marker="o", ls="", color=COLORS[m], label=m)
                            for m in ("GPM plain", "GPM adaptive f0.9")], fontsize=8, loc="center left")
    fig.suptitle("Object retention, plain vs adaptive GPM, per seed; adaptive - plain paired (pp, 95% CI); "
                 "50 rollouts per cell")
    return {"claim": "adaptive GPM keeps Object", "files": [save(fig, out, "adaptive_vs_plain_object")],
            "values_pp": notes}


def fig_controls(out: Path) -> dict:
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    ax = axes[0]
    for s in SEEDS:
        plain, e99, adaptive = (result(run_name(m, s)) for m in ("GPM plain", "GPM e99", "GPM adaptive f0.9"))
        for k, (label, a, b, color) in enumerate((("e99 - plain (protection)", e99, plain, "tab:green"),
                                                  ("adaptive - e99 (allocation)", adaptive, e99, "tab:purple"))):
            errbar(ax, s + (k - 0.5) * 0.25, paired(a, b, 2, 1), marker="o", color=color,
                   label=label if s == 0 else None)
    ax.axhline(0, color="black", lw=0.8)
    ax.set_xticks(SEEDS, [seed_label(s) for s in SEEDS])
    ax.set_ylabel("paired difference, Object after Goal R[2][1] (pp)")
    ax.set_title("C1: more protection, or the allocation?")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3, axis="y")

    ax = axes[1]
    for s in SEEDS:
        summary = json.loads((RESULTS / "reverse_order" / f"seed{s}" / "summary.json").read_text())
        cfg_cells = summary["config"]["cells"]
        plain = json.loads((Path(summary["runs"]["plain"]) / "result.json").read_text())
        adaptive = json.loads((Path(summary["runs"]["adaptive"]) / "result.json").read_text())
        for k, (name, c) in enumerate(sorted(cfg_cells.items())):
            d = paired(adaptive, plain, c["stage"], c["task"])
            task = plain["task_keys"][c["task"]].split("/")[0]
            errbar(ax, s + (k - 0.5) * 0.25, d, marker="s", color=("tab:blue", "tab:orange")[k],
                   label=f"{name} cell: {task} R[{c['stage']}][{c['task']}]" if s == 0 else None)
    ax.axhline(0, color="black", lw=0.8)
    ax.set_xticks(SEEDS, [seed_label(s) for s in SEEDS])
    ax.set_ylabel("adaptive - plain (pp)")
    ax.set_title("C2: reverse task order")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3, axis="y")
    fig.suptitle("Controls as per-seed paired differences, rollout 95% CI; 50 rollouts per cell")
    return {"claim": "allocation, or only more protection; does the pattern survive a reversed order",
            "files": [save(fig, out, "controls")]}


def fig_protected_fraction(out: Path) -> dict:
    from flowcl.analysis.subspace import load_bases
    from flowcl.experiments.high_protection import layer_dims, protected_fractions

    shown = ["GPM plain", "GPM adaptive f0.9", "GPM e99"]
    fig, ax = plt.subplots(figsize=(7, 4.2))
    values = {}
    for m in shown:
        for s in SEEDS:
            run = RESULTS / run_name(m, s)
            _, meta = load_bases(run / "method" / "memory_task3.pt")
            history = {str(k): v for k, v in meta["memory_history"].items()}
            dims = layer_dims(run / "checkpoints" / "stage3.pt", sorted(history["0"]))
            frac = protected_fractions(history, dims)
            ys = [100 * frac[str(t)]["trunk"] for t in range(4)]
            values[f"{m} seed {s}"] = [round(y, 1) for y in ys]
            ax.plot(range(1, 5), ys, marker=MARKERS[s], color=COLORS[m], lw=1,
                    label=m if s == 0 else None)
    ax.set_xticks(range(1, 5), [f"after T{t}" for t in range(1, 5)])
    ax.set_ylabel("protected fraction of the trunk (%, parameter-weighted)")
    ax.set_ylim(0, 102)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    ax.set_title("Protection uses up capacity, T1-T4: the three seeds agree within 0.5 pp (lines overlap)")
    return {"claim": "protection uses up capacity (first half; T5-T8 wait for E3)",
            "files": [save(fig, out, "protected_fraction_t1_t4")], "values_pct": values}


# ---- Track 2 and the negative results -------------------------------------------------------------


def fig_track2(out: Path) -> dict:
    measures = json.loads((RESULTS / "track2" / "measures" / "measures.json").read_text())["cells"]
    manifest = {c["id"]: c for c in json.loads((RESULTS / "track2" / "manifest" / "manifest.json").read_text())["cells"]}
    fig, ax = plt.subplots(figsize=(7.5, 5))
    by_task: dict[str, list] = {}
    for cid, rec in measures.items():
        m = manifest[cid]
        by_task.setdefault(rec["task_key"], []).append(
            (rec["measures"]["elements"]["values"]["m0"], 100 * sum(m["successes"]) / len(m["successes"])))
    curriculum = json.loads((RESULTS / run_name("seq_ft", 0) / "result.json").read_text())["task_keys"]
    groups: dict[str, list] = {name: [] for name in (*TASKS4, "other tasks")}
    for task, pts in by_task.items():
        name = TASKS4[curriculum.index(task)] if task in curriculum else "other tasks"
        groups[name].extend(pts)
    for k, (name, pts) in enumerate(groups.items()):
        if pts:
            xs, ys = zip(*pts)
            ax.scatter(xs, ys, s=10, alpha=0.6, label=f"{name} ({len(pts)} cells)",
                       color="lightgray" if name == "other tasks" else plt.get_cmap("tab10").colors[k])
    ax.set_xscale("log")
    ax.set_xlabel("probe flow loss M0 (log)")
    ax.set_ylabel("closed-loop success (%, 50 rollouts)")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=7)
    ax.set_title(f"Track 2 development cells ({len(measures)}): the same loss, very different success")
    return {"claim": "probe loss does not determine closed-loop success across tasks",
            "files": [save(fig, out, "track2_loss_vs_success")]}


def fig_twosided(out: Path) -> dict:
    s = json.loads((RESULTS / "twosided_gonogo" / "summary.json").read_text())
    table = s["tables"]["adaptive"]["0.95"]
    tasks = sorted(table, key=int)
    gpm = [table[t]["fractions"]["trunk"]["gpm"] for t in tasks]
    two = [table[t]["fractions"]["trunk"]["two_sided"] for t in tasks]
    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    x = np.arange(len(tasks))
    ax.bar(x - 0.18, gpm, 0.36, label="GPM (input side only)")
    ax.bar(x + 0.18, two, 0.36, label="two-sided (output x input)")
    ax.axhline(s["verdict"]["threshold"], color="black", ls="--", lw=1, label="go threshold 0.50")
    ax.set_xticks(x, [f"after T{int(t) + 1}" for t in tasks])
    ax.set_ylabel("protected fraction of the trunk")
    ax.set_ylim(0, 1)
    ax.legend(fontsize=8)
    ax.set_title(f"E1: two-sided frees too little (decisive {s['verdict']['decisive']:.3f} vs GPM "
                 f"{s['verdict']['gpm_at_decisive']:.3f}); seed 0")
    return {"claim": "why the two-sided line stopped", "files": [save(fig, out, "twosided_fraction")]}


def fig_recovery(out: Path) -> dict:
    from flowcl.analysis import recovery as rc

    analysis = json.loads((RESULTS / "recovery_study" / "analysis" / "analysis.json").read_text())
    tasks = list(analysis["tasks"])
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.4))
    ax = axes[0]
    groups = list(rc.ALL_GROUPS)
    width = 0.8 / len(tasks)
    for k, task in enumerate(tasks):
        shares = analysis["tasks"][task]["descriptive"]["group_injury"]
        rates = [shares[g]["injury_rate"] or 0.0 for g in groups]
        ax.bar(np.arange(len(groups)) + (k - (len(tasks) - 1) / 2) * width, rates, width,
               label=task.split("/")[0])
    ax.set_xticks(range(len(groups)), groups)
    ax.set_ylabel("injury rate (at-risk, effective trials)")
    ax.set_title("by offset type")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3, axis="y")

    ax = axes[1]
    work = RESULTS / "recovery_study" / ".work"
    for k, task in enumerate(tasks):
        rows = []
        for f in sorted((work / task.replace("/", "__")).glob("rollout_*.json")):
            rows.extend(json.loads(f.read_text())["rows"])
        p = rc.perturbed(rc.Table.from_rows(rows))
        keep = (~p["noop"]) & p["nominal_success"]
        for gname, ls in (("position", "-"), ("gripper", "--")):
            gi = groups.index(gname)
            qs = sorted(set(p["query"][keep & (p["group"] == gi)].tolist()))
            ys = [float(p["injury"][keep & (p["group"] == gi) & (p["query"] == q)].mean()) for q in qs]
            if qs:
                ax.plot([2 * q + 1 for q in qs], ys, ls=ls, marker="o", ms=3,
                        color=plt.get_cmap("tab10").colors[k], label=f"{task.split('/')[0]} {gname}")
    ax.set_xlabel("query phase (twelfths of the rollout)")
    ax.set_ylabel("injury rate")
    ax.set_title("by phase: position (solid) and gripper (dashed)")
    ax.legend(fontsize=7, ncol=2)
    ax.grid(alpha=0.3)
    fig.suptitle("Recovery study: injury = the nominal continuation succeeds and the perturbed one fails "
                 "(seq_ft seed 0 policies)")
    return {"claim": "recovery injury rate by offset type and phase", "files": [save(fig, out, "recovery_injury")]}


# ---- hardware -------------------------------------------------------------------------------------


def _resolution(policy: str) -> int:
    for token in policy.replace("+", " ").split():
        for res in ("448", "336", "224"):
            if token.endswith(f"crop{res}"):
                return int(res)
    return 128


def fig_piper_miss(out: Path) -> dict:
    sources = sorted((RESULTS / "piper_rollouts").glob("*/diag/aim.json"))
    by_res: dict[int, list[float]] = {}
    for src in sources:
        for r in json.loads(src.read_text())["runs"].values():
            # the "+blend" runs are an executor variant (record of 6 Nov), not a resolution
            if r.get("miss_mm") is not None and "grape" in r.get("policy", "") and "+blend" not in r["policy"]:
                by_res.setdefault(_resolution(r["policy"]), []).append(r["miss_mm"])
    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    rng = np.random.default_rng(0)
    for res, misses in sorted(by_res.items()):
        ax.scatter(res + rng.uniform(-8, 8, len(misses)), misses, s=14, alpha=0.6)
        med = statistics.median(misses)
        ax.plot([res - 20, res + 20], [med, med], color="black", lw=2)
        ax.annotate(f"median {med:.0f} mm, n={len(misses)}", (res, med), xytext=(6, 6),
                    textcoords="offset points", fontsize=8)
    ax.set_xticks(sorted(by_res), [f"{r} px" for r in sorted(by_res)])
    ax.set_ylabel("miss distance at the end (mm)")
    ax.set_title("Piper grape: finer input aims better (each try a point)")
    ax.grid(alpha=0.3, axis="y")
    note = ("each run's miss from scripts/hw/piper_resolution_diag.py aim; the 3 Nov 128 px file was made on "
            "15 Nov with --checkpoint-has grape_plate_crop__seed0 (23 runs, 21 measured, p50 54 mm as recorded); "
            "the '+blend' executor-variant runs are excluded")
    return {"claim": "miss distance by resolution", "files": [save(fig, out, "piper_miss_by_resolution")],
            "sources": [str(s.relative_to(ROOT)) for s in sources], "note": note}


def fig_piper_zones(out: Path) -> dict:
    src = RESULTS / "piper_rollouts" / "2026-11-06" / "diag" / "zones.json"
    zones = json.loads(src.read_text())["zones"]
    names = list(zones)
    fig, ax = plt.subplots(figsize=(7, 4.2))
    for x, name in enumerate(names):
        z = zones[name]
        est = success_estimate([True] * z["placed"] + [False] * (z["tries"] - z["placed"]), n_bootstrap=N_BOOT)
        errbar(ax, x, est, marker="o", color="tab:blue")
        ax.annotate(f"{z['placed']}/{z['tries']} placed\n{z['demos']} demos", (x, 100 * est.high),
                    xytext=(0, 6), textcoords="offset points", ha="center", fontsize=8)
    ax.set_xticks(range(len(names)), [f"{n} mm" for n in names])
    ax.set_ylim(-3, 115)
    ax.set_ylabel("grape placed (%, 95% CI over tries)")
    ax.set_title("Piper grape at 336 px: placement by grape position zone (6 Nov)")
    ax.grid(alpha=0.3, axis="y")
    return {"claim": "grasp positions by zone", "files": [save(fig, out, "piper_zones")],
            "sources": [str(src.relative_to(ROOT))]}


# ---- setup ----------------------------------------------------------------------------------------


def fig_task_images(out: Path) -> dict:
    import h5py

    from flowcl.data.curriculum import load_curriculum
    from flowcl.data.libero_setup import task_file_pattern
    from flowcl.utils.libero_paths import default_dataset_dir

    curriculum = load_curriculum(ROOT / "configs" / "curriculum" / "seq_hetero_t8.yaml")
    keys = list(curriculum.task_keys)
    fig, axes = plt.subplots(2, 4, figsize=(14, 7.4))
    for k, (ax, key) in enumerate(zip(axes.flat, keys)):
        path = default_dataset_dir() / task_file_pattern(key)
        with h5py.File(path, "r") as f:
            frame = f["data"]["demo_0"]["obs"]["agentview_rgb"][0]
        ax.imshow(frame)
        ax.set_title(f"T{k + 1} {key.split('/')[0]}\n{key.split('/')[1][:44]}", fontsize=7)
        ax.axis("off")
    fig.suptitle("The curriculum: first frame of demo 0 (agentview); T1-T4 = seq_hetero, T5-T8 = E3's extension")
    return {"claim": "what is learned, in which order", "files": [save(fig, out, "task_images")]}


FIGURES = {
    "retention_grids": fig_retention_grids,
    "forgetting_curves": fig_forgetting_curves,
    "baselines": fig_baselines,
    "adaptive_vs_plain": fig_adaptive_vs_plain,
    "controls": fig_controls,
    "protected_fraction": fig_protected_fraction,
    "track2": fig_track2,
    "twosided": fig_twosided,
    "recovery": fig_recovery,
    "piper_miss": fig_piper_miss,
    "piper_zones": fig_piper_zones,
    "task_images": fig_task_images,
}

NOT_MADE = {
    "piper_setup_photo": {"state": "made (existing asset)", "claim": "the hardware setup, mask and crop",
                          "files": ["docs/assets/piper_mask_review.jpg", "docs/assets/piper_grape_crop_closes.jpg"]},
    "e3_frontier": {"state": "waiting for data", "claim": "overall success against capacity, plain / f 0.7 / f 0.9",
                    "needs": "E3 (8 tasks, 3 seeds)"},
    "protected_fraction_t5_t8": {"state": "waiting for data", "claim": "protection after T5-T8", "needs": "E3"},
    "hardware_success_counts": {"state": "waiting for data", "claim": "hardware success counts with intervals",
                                "needs": "F3/F4 rollouts with rollout_record/v1 (flowcl/deploy/rollout_records.py)"},
    "policy_diagram": {"state": "manual", "claim": "where the projection sits in the policy",
                       "needs": "drawn by hand"},
}


def main() -> None:
    global N_BOOT
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--only", nargs="*", choices=sorted(FIGURES))
    parser.add_argument("--out-dir", type=Path, default=ROOT / "docs" / "assets" / "figures")
    parser.add_argument("--n-boot", type=int, default=N_BOOT)
    args = parser.parse_args()
    N_BOOT = args.n_boot
    manifest_path = args.out_dir / "figures.json"
    manifest = json.loads(manifest_path.read_text()) if manifest_path.is_file() else {}
    for name in args.only or FIGURES:
        try:
            manifest[name] = {"state": "made", **FIGURES[name](args.out_dir)}
            print(f"[figures] {name}: {manifest[name]['files']}", flush=True)
        except (FileNotFoundError, KeyError, ValueError) as err:
            manifest[name] = {"state": "failed", "error": f"{type(err).__name__}: {err}"}
            print(f"[figures] {name}: FAILED {err}", flush=True)
    manifest.update(NOT_MADE)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"[figures] manifest {manifest_path}")


if __name__ == "__main__":
    main()
