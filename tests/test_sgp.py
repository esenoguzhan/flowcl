"""SGP (AdamW adaptation): Eq. 2/6-10 importance, the two projectors, the scaled-update
invariant, bitwise equality with GPM at Λ = I, artifacts, and an end-to-end identity run."""

from __future__ import annotations

import math

import pytest
import torch
import torch.nn as nn

from flowcl.analysis.importance import (
    accumulate_importance,
    importance_update,
    protected_energy_fraction,
    sgp_importance,
    surrogate_sigma,
)
from flowcl.analysis.subspace import basis_from_gram, extend_basis
from flowcl.methods.base import TaskContext
from flowcl.methods.gpm import GPM
from flowcl.methods.sgp import SGP
from flowcl.models.policy import RegistryEntry

NEG_TOL = 1e-8
CTX = TaskContext(task_key=None, dataset=None, device="cpu")


class Toy(nn.Module):
    """A one-layer 'policy' exposing the registry interface GPM and SGP rely on."""

    def __init__(self, d_in: int = 6, d_out: int = 4, seed: int = 0) -> None:
        super().__init__()
        torch.manual_seed(seed)
        self.a = nn.Linear(d_in, d_out, bias=False)

    def projectable_layers(self):
        return (RegistryEntry("a", self.a, "trunk_mlp"),)

    def projectable_parameters(self):
        return ("a.weight",)


def basis_for(d_in: int, k: int, seed: int = 0):
    g = torch.Generator().manual_seed(seed)
    q, _ = torch.linalg.qr(torch.randn(d_in, d_in, generator=g, dtype=torch.float64))
    sigmas = torch.cat([torch.linspace(10.0, 5.0, k), torch.full((d_in - k,), 1e-4)])
    R = q @ torch.diag(sigmas.to(torch.float64))
    basis = basis_from_gram(R @ R.T, "a", n_samples=d_in, thresholds=[0.95], neg_tol=NEG_TOL)
    assert basis.ranks[0.95] == k
    return basis, q


def sgp_for(model, basis, lam=None, **kwargs):
    method = SGP(**kwargs)
    method.set_memory({"a": basis}, None if lam is None else {"a": torch.tensor(lam, dtype=torch.float64)})
    method.on_task_start(model, 1, context=CTX)
    return method


def gpm_for(model, basis, **kwargs):
    method = GPM(eps=0.95, **kwargs)
    method.set_memory({"a": basis})
    method.on_task_start(model, 1, context=CTX)
    return method


# ---- Eq. 2, 6-10 --------------------------------------------------------------------


def test_eq2_hand_values_and_exact_one_at_the_max():
    sigma = torch.tensor([4.0, 2.0, 1.0], dtype=torch.float64)
    torch.testing.assert_close(sgp_importance(sigma, 1.0),
                               torch.tensor([1.0, 2 / 3, 0.4], dtype=torch.float64))
    torch.testing.assert_close(sgp_importance(sigma, 25.0),
                               torch.tensor([1.0, 13 / 13.5, 6.5 / 7.25], dtype=torch.float64))
    g = torch.Generator().manual_seed(0)
    for alpha in (0.5, 1.0, 3.0, 10.0, 25.0, 1e3):
        s = torch.rand(50, generator=g, dtype=torch.float64) * 7.3
        lam = sgp_importance(s, alpha)
        assert lam[int(s.argmax())].item() == 1.0  # exactly, not approximately
        assert bool(((lam > 0) & (lam <= 1)).all())
        order = torch.argsort(s)
        assert bool((lam[order].diff() >= 0).all())  # monotone in sigma
    s = torch.tensor([1.0, 0.1, 0.01], dtype=torch.float64)
    assert sgp_importance(s, 1e9).min().item() > 0.999  # alpha -> inf approaches GPM
    assert sgp_importance(torch.tensor([3.0, 0.0], dtype=torch.float64), 25.0)[1].item() == 0.0


@pytest.mark.parametrize("sigma, alpha, message", [
    ([], 25.0, "non-empty"),
    ([1.0, -0.1], 25.0, "negative"),
    ([1.0, math.nan], 25.0, "non-finite"),
    ([0.0, 0.0], 25.0, "max sigma is 0"),
    ([1.0], 0.0, "alpha"),
    ([1.0], -1.0, "alpha"),
    ([1.0], math.inf, "alpha"),
])
def test_eq2_fails_loudly(sigma, alpha, message):
    with pytest.raises(ValueError, match=message):
        sgp_importance(torch.tensor(sigma, dtype=torch.float64), alpha)


def test_surrogate_equals_the_papers_projected_svd_formula():
    """Eq. 6-8: sigma'^2 = (C ⊙ C) sigma^2 over the SVD of M M^T R equals m^T (R R^T) m."""
    g = torch.Generator().manual_seed(1)
    d, n, k = 12, 40, 5
    R = torch.randn(d, n, generator=g, dtype=torch.float64) * torch.linspace(3, 0.1, d)[:, None]
    M, _ = torch.linalg.qr(torch.randn(d, k, generator=g, dtype=torch.float64))
    U, S, _ = torch.linalg.svd(M @ M.T @ R, full_matrices=False)
    C = M.T @ U
    paper = ((C * C) @ (S**2)).sqrt()
    torch.testing.assert_close(surrogate_sigma(M, R @ R.T), paper, rtol=1e-10, atol=1e-12)


def test_accumulation_caps_at_one_exactly_and_appends_new_bases():
    lam = accumulate_importance(torch.tensor([0.9, 0.3], dtype=torch.float64),
                                torch.tensor([0.5, 0.2], dtype=torch.float64),
                                torch.tensor([0.7], dtype=torch.float64))
    assert lam[0].item() == 1.0
    torch.testing.assert_close(lam, torch.tensor([1.0, 0.5, 0.7], dtype=torch.float64))
    with pytest.raises(ValueError, match="entries"):
        accumulate_importance(torch.zeros(2), torch.zeros(3), torch.zeros(0))


def test_importance_update_on_a_real_extension():
    g = torch.Generator().manual_seed(2)
    d = 10
    R1 = torch.randn(d, 60, generator=g, dtype=torch.float64) * torch.linspace(4, 0.05, d)[:, None]
    K1 = R1 @ R1.T
    M1, info1 = extend_basis(None, K1, 0.95, "x", neg_tol=NEG_TOL)
    first = importance_update(None, K1, info1["residual_spectrum"], info1["k_added"],
                              torch.zeros(0, dtype=torch.float64), 25.0)
    # T1: sigma = sqrt of the kept eigenvalues; no old bases.
    assert first["sigma_old"].numel() == 0
    torch.testing.assert_close(first["sigma_new"], info1["residual_spectrum"][: M1.shape[1]].sqrt())
    torch.testing.assert_close(first["lambda"], sgp_importance(first["sigma_new"], 25.0))
    assert first["lambda"][0].item() == 1.0

    R2 = torch.randn(d, 60, generator=g, dtype=torch.float64) * torch.linspace(0.05, 4, d)[:, None]
    K2 = R2 @ R2.T
    M2, info2 = extend_basis(M1, K2, 0.95, "x", neg_tol=NEG_TOL)
    second = importance_update(M1, K2, info2["residual_spectrum"], info2["k_added"],
                               first["lambda"], 25.0)
    k1 = M1.shape[1]
    assert second["lambda"].numel() == M2.shape[1]
    torch.testing.assert_close(second["sigma_old"], surrogate_sigma(M1, K2))
    lam_task = sgp_importance(torch.cat([second["sigma_old"], second["sigma_new"]]), 25.0)
    torch.testing.assert_close(second["lambda_task"], lam_task)
    torch.testing.assert_close(second["lambda"][:k1],
                               torch.clamp(first["lambda"] + lam_task[:k1], max=1.0))
    assert bool((second["lambda"][:k1] >= first["lambda"]).all())  # never decreases

    # No basis added: the max runs over the old bases' sigma' only (inferred detail).
    none = importance_update(M1, K2, torch.zeros(0, dtype=torch.float64), 0, first["lambda"], 25.0)
    assert none["sigma_new"].numel() == 0 and none["lambda_task"].max().item() == 1.0
    with pytest.raises(ValueError, match="entries"):
        importance_update(M1, K2, info2["residual_spectrum"], 1, torch.zeros(0), 25.0)


def test_protected_energy_fraction():
    K = torch.diag(torch.tensor([4.0, 3.0, 2.0, 1.0], dtype=torch.float64))
    M = torch.eye(4, dtype=torch.float64)[:, :2]
    lam = torch.tensor([1.0, 0.5], dtype=torch.float64)
    assert protected_energy_fraction(M, lam, K) == pytest.approx((4.0 + 1.5) / 10.0)


# ---- projectors and the scaled-update invariant ---------------------------------------


def test_gradient_projector_holds_only_the_hard_bases_and_update_projector_scales():
    basis, _ = basis_for(6, 2)
    M = basis.basis(0.95).to(torch.float64)
    method = sgp_for(Toy(), basis, lam=[1.0, 0.25])
    layer = method._layers[0]
    eye = torch.eye(6, dtype=torch.float64)
    m0, m1 = M[:, :1], M[:, 1:]
    torch.testing.assert_close(layer.P_grad.double(), eye - m0 @ m0.T, atol=1e-6, rtol=0)
    torch.testing.assert_close(layer.P_update.double(), eye - m0 @ m0.T - 0.25 * m1 @ m1.T,
                               atol=1e-6, rtol=0)
    assert layer.n_hard == 1
    torch.testing.assert_close(layer.keep, torch.tensor([0.0, 0.75]))


def test_full_rank_hard_memory_builds_exact_zero_projectors():
    basis, _ = basis_for(4, 4)
    method = sgp_for(Toy(d_in=4), basis, lam=[1.0, 1.0, 1.0, 1.0])
    layer = method._layers[0]
    assert torch.count_nonzero(layer.P_grad) == 0 and layer.P_update is layer.P_grad
    soft = sgp_for(Toy(d_in=4), basis, lam=[1.0, 1.0, 1.0, 0.5])._layers[0]
    assert torch.count_nonzero(soft.P_update) > 0


def test_scaled_projection_acts_on_the_input_dimension_of_a_square_layer():
    """dW = u m1^T with m1 soft (lambda 0.25): the update keeps 0.75 of it; the gradient
    projector leaves it alone. A transposed projection would not."""
    basis, q = basis_for(6, 2)
    M = basis.basis(0.95).float()
    model = Toy(d_in=6, d_out=6)
    method = sgp_for(model, basis, lam=[1.0, 0.25])
    u = q[:, 4].float()
    G = torch.outer(u, M[:, 1])
    model.a.weight.grad = G.clone()
    method.modify_gradients(model, {"step": 1, "task_idx": 1})
    torch.testing.assert_close(model.a.weight.grad, G)  # soft: Adam sees it in full
    model.a.weight.grad = torch.outer(u, M[:, 0])
    method.modify_gradients(model, {"step": 1, "task_idx": 1})
    assert float(model.a.weight.grad.norm()) < 1e-6  # hard: removed before Adam
    w0 = model.a.weight.detach().clone()
    with torch.no_grad():
        model.a.weight.add_(G)
    method.after_step(model, {"step": 1, "task_idx": 1})
    torch.testing.assert_close(model.a.weight.detach() - w0, 0.75 * G, atol=1e-6, rtol=0)


def adamw_steps(model, method, n=5, wd=0.1, seed=0):
    g = torch.Generator().manual_seed(seed)
    opt = torch.optim.AdamW(model.parameters(), lr=0.05, weight_decay=wd)
    for step in range(n):
        x, y = torch.randn(32, model.a.in_features, generator=g), torch.randn(32, model.a.out_features, generator=g)
        opt.zero_grad()
        ((model.a(x) - y) ** 2).mean().backward()
        method.modify_gradients(model, {"step": step, "task_idx": 1})
        opt.step()
        method.after_step(model, {"step": step, "task_idx": 1})


def test_sgp_with_all_importances_one_is_bitwise_gpm():
    basis, _ = basis_for(6, 3)
    gpm_model, sgp_model = Toy(), Toy()
    gpm = gpm_for(gpm_model, basis)
    sgp = sgp_for(sgp_model, basis, lam=[1.0, 1.0, 1.0])
    assert torch.equal(gpm._layers[0].P, sgp._layers[0].P_grad)
    assert torch.equal(gpm._layers[0].P, sgp._layers[0].P_update)
    adamw_steps(gpm_model, gpm)
    adamw_steps(sgp_model, sgp)
    assert torch.equal(gpm_model.a.weight, sgp_model.a.weight)


def test_adamw_steps_keep_exactly_one_minus_lambda_along_each_basis():
    basis, _ = basis_for(6, 3)
    model = Toy()
    method = sgp_for(model, basis, lam=[1.0, 0.6, 0.1])
    adamw_steps(model, method, n=4)  # after_step asserts the invariant every step
    assert 0.0 < method._layers[0].max_residual_ratio <= 1.0
    # Soft directions moved, the hard one did not.
    M = basis.basis(0.95).float()
    moved = (method._layers[0].displacement @ M).norm(dim=0)
    assert moved[0] < 1e-5 and moved[1] > 1e-4 and moved[2] > 1e-4


def one_step(model, method, opt, clip=None, seed=1):
    """One trainer-ordered step (gradient hook, clip, optimizer, post-step hook): the raw
    gradient and the applied update."""
    g = torch.Generator().manual_seed(seed)
    x, y = torch.randn(32, 6, generator=g), torch.randn(32, 4, generator=g)
    opt.zero_grad()
    ((model.a(x) - y) ** 2).mean().backward()
    raw = model.a.weight.grad.detach().clone().double()
    method.modify_gradients(model, {"step": 0, "task_idx": 1})
    if clip is not None:
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
    before = model.a.weight.detach().clone()
    opt.step()
    method.after_step(model, {"step": 0, "task_idx": 1})
    return raw, (model.a.weight.detach() - before).double()


def test_under_vanilla_sgd_the_adaptation_is_exactly_the_scaled_projection():
    """The coupling test's premise: hard removal before the step plus scaling after it equals
    ``-eta G (I - M Lambda M^T)`` for vanilla SGD, unclipped. Clipping (on the hard-projected
    gradient) rescales by a norm that includes the soft components; AdamW is not a scaled step."""
    basis, _ = basis_for(6, 3)
    lam = [1.0, 0.6, 0.1]
    M = basis.basis(0.95).to(torch.float64)
    P = torch.eye(6, dtype=torch.float64) - M @ torch.diag(torch.tensor(lam, dtype=torch.float64)) @ M.T
    P_grad = torch.eye(6, dtype=torch.float64) - M[:, :1] @ M[:, :1].T
    lr = 0.05

    model = Toy()
    raw, applied = one_step(model, sgp_for(model, basis, lam=lam),
                            torch.optim.SGD(model.parameters(), lr=lr, momentum=0.0))
    torch.testing.assert_close(applied, -lr * raw @ P, atol=1e-6, rtol=0)

    model = Toy()
    raw, applied = one_step(model, sgp_for(model, basis, lam=lam),
                            torch.optim.SGD(model.parameters(), lr=lr, momentum=0.0), clip=1e-3)
    clipped_by = 1e-3 / float((raw @ P_grad).norm())   # the norm the trainer's clip sees
    exact = 1e-3 / float((raw @ P).norm())              # the scaled gradient's own norm
    torch.testing.assert_close(applied, -lr * clipped_by * raw @ P, atol=1e-7, rtol=1e-4)
    assert clipped_by < 0.9 * exact

    model = Toy()
    raw, applied = one_step(model, sgp_for(model, basis, lam=lam),
                            torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.0))
    target = -lr * raw @ P
    scale = float((applied * target).sum() / (target * target).sum())
    assert float((applied - scale * target).norm()) > 0.1 * float(applied.norm())


def test_residual_check_catches_an_unscaled_update():
    basis, _ = basis_for(6, 2)
    model = Toy()
    method = sgp_for(model, basis, lam=[1.0, 0.5])
    method.after_step(model, {"step": 0, "task_idx": 1})  # skipped AMP step: no change
    method._layers[0].P = torch.eye(6)  # sabotage: nothing gets scaled
    with torch.no_grad():
        model.a.weight.add_(0.1 * torch.randn_like(model.a.weight))
    with pytest.raises(RuntimeError, match="not scaled by 1 - lambda"):
        method.after_step(model, {"step": 1, "task_idx": 1})


def test_importances_are_validated_when_the_projector_is_built():
    basis, _ = basis_for(6, 2)
    with pytest.raises(ValueError, match=r"\(0, 1\]"):
        sgp_for(Toy(), basis, lam=[1.0, 0.0])
    with pytest.raises(RuntimeError, match="1 importances for a 2-column memory"):
        sgp_for(Toy(), basis, lam=[1.0])
    from flowcl.methods.gpm import _ProjectedLayer

    with pytest.raises(ValueError, match="1 importances for 2 basis vectors"):
        _ProjectedLayer("a", Toy().a.weight, basis.basis(0.95), 1e-6,
                        lam=torch.tensor([1.0], dtype=torch.float64))


# ---- names, configs ---------------------------------------------------------------


def test_names_configs_and_method_yamls():
    from flowcl.data.config import load_method_config
    from flowcl.methods.base import build_method
    from flowcl.train.continual import continual_run_id

    assert SGP().display_name == "sgp_projected_adam"
    assert SGP(new_energy_fraction=0.9).display_name == "sgp_projected_adam_ne90"
    assert SGP().config()["alpha"] == 25.0
    with pytest.raises(ValueError, match="not implemented"):
        SGP(projection="gradient_and_update")
    with pytest.raises(ValueError, match="not implemented"):
        GPM(projection="hard_gradient_scaled_update")
    for bad in (0, -1.0, math.nan, "25"):
        with pytest.raises(ValueError, match="alpha"):
            SGP(alpha=bad)

    name, kwargs = load_method_config("sgp")
    ne_name, ne_kwargs = load_method_config("sgp_ne90")
    _, gpm_kwargs = load_method_config("gpm")
    assert name == ne_name == "sgp" and kwargs["alpha"] == 25
    assert ne_kwargs.pop("new_energy_fraction") == 0.9 and ne_kwargs == kwargs
    # Otherwise identical to plain GPM, so the memory stays paired.
    assert {k: v for k, v in kwargs.items() if k not in ("alpha", "projection")} == {
        k: v for k, v in gpm_kwargs.items() if k != "projection"}
    baseline = build_method(name, **kwargs)
    adaptive = build_method(ne_name, **load_method_config("sgp_ne90")[1])
    assert (continual_run_id(baseline.display_name, "seq_hetero", 0)
            == "seq_hetero__sgp_projected_adam__seed0")
    assert adaptive.display_name == "sgp_projected_adam_ne90"


# ---- the tiny real policy, end to end ----------------------------------------------

MILK = "libero_object/pick_up_the_milk_and_place_it_in_the_basket"
SAUCE = "libero_object/pick_up_the_tomato_sauce_and_place_it_in_the_basket"
BBQ = "libero_object/pick_up_the_bbq_sauce_and_place_it_in_the_basket"
TINY_POLICY = {"d_model": 384, "n_trunk_layers": 6, "n_heads": 8, "n_decoder_layers": 2,
               "n_context_tokens": 4, "pretrained": False, "euler_steps": 2}


def test_sgp_runs_end_to_end_paired_with_gpm(dataset_dir, tmp_path):
    """Three tasks: SGP's stage 0 and T1 memory equal GPM's, its T1 importance is Eq. 2 of
    GPM's stored spectrum, every stored importance update recomputes, the provenance checks
    pass, and adaptive SGP equals SGP at stages 0-1 (p = 0 at T1)."""
    from omegaconf import OmegaConf

    from flowcl.analysis.subspace import load_bases
    from flowcl.data.config import load_embodiment_spec
    from flowcl.data.curriculum import load_curriculum
    from flowcl.envs.libero_env import EvalConfig
    from flowcl.experiments.sequence_report import RunView, provenance_checks
    from flowcl.train.continual import run_continual, stage_identity_check
    from flowcl.train.trainer import TrainConfig
    from flowcl.utils.libero_paths import repo_root
    from flowcl.utils.run import file_sha256

    cfg = OmegaConf.load(repo_root() / "configs" / "analysis" / "subspace.yaml")
    cfg.min_samples_per_dim = 0.01
    cfg.num_workers = 0
    capture = tmp_path / "capture.yaml"
    OmegaConf.save(cfg, capture)
    trio = load_curriculum({"name": "test_trio", "tasks": [
        {"task_key": MILK, "n_demos": 1}, {"task_key": SAUCE, "n_demos": 1},
        {"task_key": BBQ, "n_demos": 1}]})
    common = dict(
        spec=load_embodiment_spec("libero_franka"), policy_config=TINY_POLICY,
        train_cfg=TrainConfig(steps=3, batch_size=2, num_workers=0, device="cpu", log_every=0,
                              warmup_steps=1),
        eval_cfg=EvalConfig(n_episodes=1), seed=0, dataset_dir=dataset_dir,
        results_root=tmp_path, pretrained=False, evaluate=False,
    )
    base_kwargs = {"update_memory": True, "capture_config": str(capture), "log_interval": 1}
    gpm = run_continual(trio, method_name="gpm", method_kwargs=base_kwargs, **common)
    sgp = run_continual(trio, method_name="sgp", method_kwargs={**base_kwargs, "alpha": 25},
                        identity_reference_run=tmp_path / gpm.run_id, identity_stages=(0,),
                        **common)
    assert sgp.run_id == "test_trio__sgp_projected_adam__seed0"
    ada = run_continual(trio, method_name="sgp",
                        method_kwargs={**base_kwargs, "alpha": 25, "new_energy_fraction": 0.9},
                        identity_reference_run=tmp_path / sgp.run_id, identity_stages=(0, 1),
                        **common)
    assert ada.run_id == "test_trio__sgp_projected_adam_ne90__seed0"
    gpm_dir, sgp_dir, ada_dir = (tmp_path / r.run_id for r in (gpm, sgp, ada))
    assert stage_identity_check(sgp_dir / "checkpoints" / "stage0.pt", gpm_dir, 0)["passed"]
    assert not stage_identity_check(sgp_dir / "checkpoints" / "stage1.pt", gpm_dir, 1)["passed"]

    g0, _ = load_bases(gpm_dir / "method" / "memory_task0.pt")
    s0, meta0 = load_bases(sgp_dir / "method" / "memory_task0.pt")
    a0, _ = load_bases(ada_dir / "method" / "memory_task0.pt")
    assert meta0["config"]["alpha"] == 25.0 and meta0["kind"] == "accumulated_memory"
    for name, b in g0.items():
        assert torch.equal(b.vectors, s0[name].vectors), name
        k = b.vectors.shape[1]
        expected = sgp_importance(b.singular_values[:k], 25.0)
        assert torch.equal(s0[name].meta["importance"], expected), name  # bitwise
        assert torch.equal(s0[name].meta["importance"], a0[name].meta["importance"]), name

    for run_dir in (sgp_dir, ada_dir):
        prev = {}
        for task in range(3):
            bases, meta = load_bases(run_dir / "method" / f"memory_task{task}.pt")
            for name, b in bases.items():
                upd, lam = b.meta["importance_update"], b.meta["importance"]
                assert b.meta["importance_update_task"] == task
                assert torch.equal(upd["lambda"], lam) and lam.numel() == b.vectors.shape[1]
                assert torch.equal(upd["lambda_prev"], prev.get(name, torch.zeros(0, dtype=torch.float64)))
                lam_task = sgp_importance(torch.cat([upd["sigma_old"], upd["sigma_new"]]), 25.0)
                assert torch.equal(lam_task, upd["lambda_task"])
                k_old = upd["lambda_prev"].numel()
                assert torch.equal(lam, accumulate_importance(upd["lambda_prev"], lam_task[:k_old],
                                                              lam_task[k_old:]))
                assert bool(((lam > 0) & (lam <= 1)).all())
                assert meta["memory_history"][str(task)][name]["importance"]["n_hard"] >= 1
                prev[name] = lam
        restored = SGP(update_memory=True)
        if run_dir == ada_dir:
            restored = SGP(update_memory=True, new_energy_fraction=0.9)
        path = run_dir / "method" / "memory_task2.pt"
        restored.restore_memory(path, file_sha256(path), task_idx=2)
        assert all(torch.equal(restored._importance[n], prev[n]) for n in prev)
        with pytest.raises(ValueError, match="alpha"):
            SGP(alpha=10, new_energy_fraction=restored.new_energy_fraction).restore_memory(
                path, file_sha256(path))

        view = RunView(run_dir, {"task_keys": list(trio.task_keys),
                                 "seed_namespace_run_id": "test_trio__seq_ft__seed0"}, {})
        checks = provenance_checks(view, RunView(tmp_path / "test_trio__seq_ft__seed0", {}, {}))
        for name in ("seed_namespace", "occupancy_non_decreasing", "residuals_within_bound",
                     "artifact_hashes_match", "frozen_from_stage1", "memory_chained"):
            assert checks[name]["passed"], (run_dir.name, name, checks[name])


def test_non_registered_alpha_is_named_and_configured_separately():
    from flowcl.data.config import load_method_config
    from flowcl.methods.base import build_method
    from flowcl.train.continual import continual_run_id

    # The registered alpha keeps its names (and so every existing run id).
    assert SGP(alpha=25).display_name == "sgp_projected_adam"
    assert SGP(alpha=25, new_energy_fraction=0.9).display_name == "sgp_projected_adam_ne90"
    assert SGP(alpha=800).display_name == "sgp_projected_adam_a800"
    assert SGP(alpha=800, new_energy_fraction=0.9).display_name == "sgp_projected_adam_a800_ne90"
    for name, registered in (("sgp_a800", "sgp"), ("sgp_a800_ne90", "sgp_ne90")):
        n, kwargs = load_method_config(name)
        _, base = load_method_config(registered)
        assert n == "sgp" and kwargs["alpha"] == 800
        assert {k: v for k, v in kwargs.items() if k != "alpha"} == {
            k: v for k, v in base.items() if k != "alpha"}
    assert continual_run_id(build_method("sgp", **load_method_config("sgp_a800")[1]).display_name,
                            "seq_hetero", 0) == "seq_hetero__sgp_projected_adam_a800__seed0"
    assert continual_run_id(build_method("sgp", **load_method_config("sgp_a800_ne90")[1]).display_name,
                            "seq_hetero", 0) == "seq_hetero__sgp_projected_adam_a800_ne90__seed0"
