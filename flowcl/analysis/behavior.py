"""Offline measures of a (checkpoint, task) cell for Track 2 (docs/thesis_plan.md, Workstream D).

On a fixed bank (the forgetting diagnostics' probe: the same batches, ``s`` and ``A_0`` for every
checkpoint; fp32, eval mode) one forward pass per batch gives the per-element squared velocity errors
``e = (v - (A_1 - A_0))^2``, masked, from which

* **M0** is the masked flow-matching loss (equal to ``probes.probe_batch_losses``),
* **M1** the same loss per flow-time bin (``flow_head.S_BINS``),
* **M2** the loss per action component and on the executed first ``execute_k`` chunk steps against
  the rest;

and **M3** samples ``K`` action chunks per observation with the policy's Euler integrator from a fixed
noise bank and scores them against the demonstration chunk: the mean over draws (the expected
single-draw error on this offline bank) and best-of-K (an oracle diagnostic: a deployed policy draws
once). Every measure is stored as additive (sum, count) pairs per batch and per demonstration, the
resampling units of later bootstraps (chunks overlap within a demonstration, so batches are not
independent units).
"""

from __future__ import annotations

import torch

from flowcl.models.flow_head import S_BINS, draw_with_generator
from flowcl.train.trainer import move_batch
from flowcl.utils.seeding import derive_seed


def _gens(seed_tags: dict, task_key: str) -> dict[str, torch.Generator]:
    return {role: torch.Generator(device="cpu").manual_seed(derive_seed(tag, task_key, 0))
            for role, tag in seed_tags.items()}


def bank_indices(n: int, probe: dict, task_key: str) -> list[list[int]]:
    """The probe's batches as dataset indices: the exact order ``build_dataloader(shuffle=True)`` draws
    with the probe's shuffle stream (an index-only loader through the same sampler code)."""
    gens = _gens({"shuffle": probe["seed_tags"]["shuffle"]}, task_key)
    # build_dataloader's sampling arguments (its collate_fn only shapes samples, not their order)
    loader = torch.utils.data.DataLoader(_Indices(n), batch_size=probe["batch_size"], shuffle=True,
                                         num_workers=0, drop_last=False, generator=gens["shuffle"])
    out = []
    for idx, batch in enumerate(loader):
        if idx >= probe["n_batches"]:
            break
        out.append([int(i) for i in batch])
    return out


class _Indices(torch.utils.data.Dataset):
    def __init__(self, n: int):
        self.n = n

    def __len__(self) -> int:
        return self.n

    def __getitem__(self, i: int) -> int:
        return i


class _Acc:
    """Additive (sum, count) per key, per batch and per demonstration."""

    def __init__(self, keys: list[str]):
        self.keys = keys
        self.batches: list[dict] = []
        self.demos: dict[int, dict] = {}

    def new_batch(self) -> dict:
        b = {k: [0.0, 0.0] for k in self.keys}
        self.batches.append(b)
        return b

    def add(self, batch_acc: dict, demo: int, key: str, total: float, count: float) -> None:
        batch_acc[key][0] += total
        batch_acc[key][1] += count
        d = self.demos.setdefault(int(demo), {k: [0.0, 0.0] for k in self.keys})
        d[key][0] += total
        d[key][1] += count

    def value(self, key: str) -> float | None:
        total = sum(b[key][0] for b in self.batches)
        count = sum(b[key][1] for b in self.batches)
        return total / count if count > 0 else None

    def payload(self) -> dict:
        return {"values": {k: self.value(k) for k in self.keys},
                "batches": self.batches,
                "demos": {str(k): v for k, v in sorted(self.demos.items())}}


def element_keys(components: dict, executed_steps: int) -> list[str]:
    return (["m0"] + [f"m1_bin{i}" for i in range(len(S_BINS))] + [f"m2_{c}" for c in components]
            + ["m2_executed", "m2_rest"])


@torch.no_grad()
def cell_measures(policy, dataset, task_key: str, cfg: dict, device) -> dict:
    """M0-M3 for one checkpoint on one task's bank. ``cfg`` is the rule's ``measures`` section."""
    if dataset.task_ids != (task_key,):
        raise ValueError(f"the bank needs {task_key}'s single-task dataset, got {dataset.task_ids}")
    from flowcl.data.dataset import collate_chunks

    device = torch.device(device)
    probe = cfg["probe"]
    components = {name: tuple(span) for name, span in cfg["m2"]["components"].items()}
    k_exec = int(cfg["m2"]["executed_steps"])
    gens = _gens(probe["seed_tags"], task_key)
    m3_gen = torch.Generator(device="cpu").manual_seed(derive_seed(cfg["m3"]["noise_tag"], task_key, 0))
    euler = cfg["m3"].get("euler_steps") or policy.euler_steps
    elem = _Acc(element_keys(components, k_exec))
    m3 = _Acc(["m3_mean", "m3_best", "m3_mean_executed", "m3_best_executed"])
    batches = bank_indices(len(dataset), probe, task_key)
    demo_of = [ix.episode_idx for ix in dataset._index]
    policy.eval()
    checks = []
    for b_idx, idx in enumerate(batches):
        batch = move_batch(collate_chunks([dataset[i] for i in idx]), device)
        size = batch["actions"].shape[0]
        s = policy.s_sampler.sample(size, device, generator=gens["flow_time"])
        noise = draw_with_generator(tuple(batch["actions"].shape), device=device, generator=gens["noise"],
                                    dtype=torch.float32, normal=True)
        with torch.autocast(device_type=device.type, enabled=False):
            out = policy(batch, s=s, noise=noise)
        mask = batch["action_mask"].to(torch.float32)
        err = (out["velocity"].float() - out["target_velocity"].float()).pow(2) * mask[..., None]
        d = err.shape[-1]
        checks.append((float(out["loss"]), float(err.sum()) / float(mask.sum() * d)))
        acc = elem.new_batch()
        s_cpu = s.detach().float().cpu()
        per_sample = err.sum(dim=(1, 2)).cpu()
        n_valid = (mask.sum(dim=1) * d).cpu()
        comp = {c: err[..., a:b].sum(dim=(1, 2)).cpu() for c, (a, b) in components.items()}
        comp_n = {c: (mask.sum(dim=1) * (b - a)).cpu() for c, (a, b) in components.items()}
        ex = err[:, :k_exec].sum(dim=(1, 2)).cpu()
        ex_n = (mask[:, :k_exec].sum(dim=1) * d).cpu()
        for j, i in enumerate(idx):
            demo = demo_of[i]
            elem.add(acc, demo, "m0", float(per_sample[j]), float(n_valid[j]))
            for bi, (lo, hi) in enumerate(S_BINS):
                last = bi == len(S_BINS) - 1
                if lo <= float(s_cpu[j]) < hi or (last and float(s_cpu[j]) == hi):
                    elem.add(acc, demo, f"m1_bin{bi}", float(per_sample[j]), float(n_valid[j]))
            for c in components:
                elem.add(acc, demo, f"m2_{c}", float(comp[c][j]), float(comp_n[c][j]))
            elem.add(acc, demo, "m2_executed", float(ex[j]), float(ex_n[j]))
            elem.add(acc, demo, "m2_rest", float(per_sample[j] - ex[j]), float(n_valid[j] - ex_n[j]))
        if b_idx < int(cfg["m3"]["n_batches"]):
            _m3_batch(policy, batch, mask, idx, demo_of, m3, m3_gen, int(cfg["m3"]["k"]), euler, k_exec, device)
    rel = max(abs(a - b) / max(abs(a), 1e-30) for a, b in checks)
    if rel > 1e-5:
        raise RuntimeError(f"per-element errors do not reproduce the forward loss (max relative {rel:.2e})")
    return {"elements": elem.payload(), "m3": m3.payload(), "n_batches": len(batches),
            "m3_batches": min(len(batches), int(cfg["m3"]["n_batches"])), "euler_steps": int(euler),
            "loss_reproduction_max_rel": rel}


def _m3_batch(policy, batch, mask, idx, demo_of, acc_m3, gen, k, euler, k_exec, device) -> None:
    with torch.autocast(device_type=device.type, enabled=False):
        context = policy.encode_observation(batch)
        actions = batch["actions"].float()
        per_draw, per_draw_ex = [], []
        for _ in range(k):
            noise = draw_with_generator(tuple(actions.shape), device=device, generator=gen,
                                        dtype=torch.float32, normal=True)
            chunk = policy.flow_head.sample(context, n_steps=euler, noise=noise).float()
            e = (chunk - actions).pow(2) * mask[..., None]
            n = mask.sum(dim=1) * e.shape[-1]
            n_ex = mask[:, :k_exec].sum(dim=1) * e.shape[-1]
            per_draw.append((e.sum(dim=(1, 2)) / n).cpu())
            per_draw_ex.append((e[:, :k_exec].sum(dim=(1, 2)) / n_ex.clamp(min=1)).cpu())
    draws, draws_ex = torch.stack(per_draw), torch.stack(per_draw_ex)   # (K, B): per-sample MSE
    acc = acc_m3.new_batch()
    for j, i in enumerate(idx):
        demo = demo_of[i]
        acc_m3.add(acc, demo, "m3_mean", float(draws[:, j].mean()), 1.0)
        acc_m3.add(acc, demo, "m3_best", float(draws[:, j].min()), 1.0)
        acc_m3.add(acc, demo, "m3_mean_executed", float(draws_ex[:, j].mean()), 1.0)
        acc_m3.add(acc, demo, "m3_best_executed", float(draws_ex[:, j].min()), 1.0)
