"""FlowPolicy: the §4.5 registry, the §7.4 allowlist, and §8.2 parameter budget.

Encoders are built with ``pretrained=False`` so the suite needs no network access; the
architecture, registry and allowlist are all weight-independent.
"""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn

from flowcl.data.config import load_embodiment_spec
from flowcl.models.build import build_policy
from flowcl.models.policy import FlowPolicy


@pytest.fixture(scope="module")
def spec():
    return load_embodiment_spec("libero_franka")


@pytest.fixture(scope="module")
def policy(spec):
    torch.manual_seed(0)
    return build_policy("flowpolicy_small", spec, pretrained=False)


def make_batch(spec, batch_size: int = 2, language=None) -> dict:
    h, w = spec.observation.image_size
    return {
        "images": {
            camera: torch.randint(0, 255, (batch_size, h, w, 3), dtype=torch.uint8)
            for camera in spec.cameras
        },
        "state": torch.randn(batch_size, spec.d_state),
        "actions": torch.randn(batch_size, spec.action.chunk_horizon, spec.d_action),
        "action_mask": torch.ones(batch_size, spec.action.chunk_horizon),
        "language": language or ["pick up the milk"] * batch_size,
    }


# ---- §4.5 layer registry ------------------------------------------------------


def test_registry_is_complete_and_exclusive(policy):
    """Covers every eligible nn.Linear and nothing else."""
    policy.assert_registry_complete()


def test_registry_order_is_deterministic(spec):
    """§4.5: 'Order must be deterministic.'"""
    a = build_policy("flowpolicy_small", spec, pretrained=False)
    b = build_policy("flowpolicy_small", spec, pretrained=False)
    assert a.registry_names() == b.registry_names()
    assert policy_names_sorted_by_depth(a.registry_names())


def policy_names_sorted_by_depth(names) -> bool:
    """Trunk entries precede decoder entries, and block indices ascend."""
    trunk = [n for n in names if n.startswith("trunk.")]
    decoder = [n for n in names if n.startswith("flow_head.")]
    if names != tuple(trunk) + tuple(decoder):
        return False
    indices = [
        int(n.split(".")[2]) for n in decoder if n.startswith("flow_head.blocks.")
    ]
    return indices == sorted(indices)


def test_registry_entries_are_linear_with_sane_widths(policy):
    for entry in policy.projectable_layers():
        assert isinstance(entry.module, nn.Linear)
        assert entry.d_in > 0 and entry.d_out > 0
        assert entry.group


def test_registry_excludes_frozen_encoders(policy):
    """§0 freezes the encoders, so they must not be projectable."""
    for name in policy.registry_names():
        assert not name.startswith("vision_encoder")
        assert not name.startswith("text_encoder")


def test_registry_excludes_adaln_and_s_embedding(policy):
    """§7.4 excludes AdaLN/FiLM modulation and the s embedding from projection."""
    for name in policy.registry_names():
        assert "modulation" not in name, name
        assert "s_embedding" not in name, name


def test_registry_includes_every_required_group(policy):
    """§4.5 names the groups explicitly; all must be present."""
    groups = {entry.group for entry in policy.projectable_layers()}
    assert groups == {
        "trunk_input",
        "trunk_attn",
        "trunk_mlp",
        "decoder_input",
        "decoder_self_attn",
        "decoder_cross_attn",
        "decoder_mlp",
        "decoder_output",
    }


def test_registry_separates_qkv(policy):
    """Fused QKV would make per-projection subspaces impossible to separate (§7.1)."""
    block0 = [n for n in policy.registry_names() if n.startswith("trunk.blocks.0.attn")]
    assert block0 == [
        "trunk.blocks.0.attn.q_proj",
        "trunk.blocks.0.attn.k_proj",
        "trunk.blocks.0.attn.v_proj",
        "trunk.blocks.0.attn.out_proj",
    ]


def test_completeness_assertion_fires_on_an_unregistered_layer(policy):
    """A new linear layer that nobody registered must be caught, not ignored."""
    policy.trunk.sneaky = nn.Linear(8, 8)
    try:
        with pytest.raises(RuntimeError, match="eligible but unregistered"):
            policy.assert_registry_complete()
    finally:
        del policy.trunk.sneaky


def test_completeness_assertion_fires_on_a_stale_entry(policy):
    """A registry entry whose module is gone must be caught too."""
    original = policy._registry
    from flowcl.models.policy import RegistryEntry

    policy._registry = original + (
        RegistryEntry(name="trunk.ghost", module=nn.Linear(4, 4), group="trunk_mlp"),
    )
    try:
        with pytest.raises(RuntimeError, match="registered but not eligible"):
            policy.assert_registry_complete()
    finally:
        policy._registry = original


def test_projectable_parameters_are_registry_weights(policy):
    names = set(policy.projectable_parameters())
    actual = dict(policy.named_parameters())
    assert names
    for name in names:
        assert name in actual, name
        assert actual[name].ndim == 2, f"{name} should be a weight matrix"


# ---- §7.4 allowlist -----------------------------------------------------------


def test_adaln_and_frozen_params_is_recorded_and_non_empty(policy):
    names = policy.adaln_and_frozen_params()
    assert len(names) > 0
    assert names == tuple(sorted(names))


def test_allowlist_covers_layernorms_and_biases_and_adaln(policy):
    """§7.4: LayerNorm weights/biases, all linear biases, AdaLN params, s embeddings."""
    names = set(policy.adaln_and_frozen_params())

    layernorm_params = {
        f"{module_name}.{param_name}"
        for module_name, module in policy.named_modules()
        if isinstance(module, nn.LayerNorm)
        for param_name, _ in module.named_parameters(recurse=False)
    }
    assert layernorm_params <= names, sorted(layernorm_params - names)

    linear_biases = {
        f"{module_name}.bias"
        for module_name, module in policy.named_modules()
        if isinstance(module, nn.Linear) and module.bias is not None
    }
    assert linear_biases <= names, sorted(linear_biases - names)

    assert any("modulation" in n for n in names)
    assert any("s_embedding" in n for n in names)


def test_allowlist_and_projectable_params_are_disjoint(policy):
    """A parameter cannot be both frozen by §7.4 and projected by §7.2."""
    frozen = set(policy.adaln_and_frozen_params())
    projected = set(policy.projectable_parameters())
    assert not (frozen & projected), sorted(frozen & projected)


def test_allowlist_names_all_resolve(policy):
    actual = dict(policy.named_parameters())
    for name in policy.adaln_and_frozen_params():
        assert name in actual, name


# ---- §8.2 parameter budget ----------------------------------------------------


@pytest.mark.parametrize("config_name", ["flowpolicy_small", "flowpolicy_base"])
def test_trainable_parameter_count_in_20_to_60M(spec, config_name):
    """§8.2 reports trainable params; the plan requires 20-60M."""
    built = build_policy(config_name, spec, pretrained=False)
    n = built.trainable_parameter_count()
    assert 20e6 <= n <= 60e6, f"{config_name} has {n / 1e6:.1f}M trainable parameters"


def test_frozen_encoder_parameters_are_not_trainable(spec, policy):
    """§0: the visual encoder is frozen during all continual learning."""
    for name, param in policy.vision_encoder.backbone.named_parameters():
        assert not param.requires_grad, name
    for name, param in policy.text_encoder.backbone.named_parameters():
        assert not param.requires_grad, name

    # The patch projection is the trainable interface (§4.1).
    assert policy.vision_encoder.patch_projection.weight.requires_grad


def test_frozen_backbone_stays_in_eval_mode(policy):
    """Calling .train() must not put the frozen backbone into training mode.

    Otherwise dropout/norm statistics drift with the task distribution and the
    'frozen encoder' claim is false in a way the parameter count cannot reveal.
    """
    policy.train()
    assert not policy.vision_encoder.backbone.training
    assert not policy.text_encoder.backbone.training


def test_parameter_report_structure(policy):
    report = policy.parameter_report()
    assert report["trainable"] > 0
    assert report["frozen"] > 0
    assert report["registry_layers"] == len(policy.projectable_layers())
    assert report["by_group"]


# ---- forward / sample ---------------------------------------------------------


def test_forward_shapes(policy, spec):
    out = policy(make_batch(spec))
    assert out["loss"].ndim == 0
    assert out["velocity"].shape == (2, spec.action.chunk_horizon, spec.d_action)
    assert out["s"].shape == (2,)
    assert out["context"].shape[0] == 2
    assert out["context"].shape[2] == policy.d_model


def test_forward_returns_s_for_hook_tagging(policy, spec):
    """§7.1 tags each captured activation with the s that produced it."""
    s = torch.tensor([0.1, 0.9])
    out = policy(make_batch(spec), s=s)
    torch.testing.assert_close(out["s"], s)


def test_loss_is_finite_and_positive(policy, spec):
    out = policy(make_batch(spec))
    assert torch.isfinite(out["loss"])
    assert float(out["loss"]) > 0


def test_sample_shape_and_determinism(policy, spec):
    batch = make_batch(spec)
    g1 = torch.Generator().manual_seed(99)
    g2 = torch.Generator().manual_seed(99)
    a = policy.sample(batch, generator=g1)
    b = policy.sample(batch, generator=g2)
    assert a.shape == (2, spec.action.chunk_horizon, spec.d_action)
    assert torch.equal(a, b)


def test_default_euler_steps_is_ten(policy):
    """§4.4: N = 10."""
    assert policy.euler_steps == 10


def test_missing_camera_raises(policy, spec):
    batch = make_batch(spec)
    del batch["images"]["robot0_eye_in_hand"]
    with pytest.raises(KeyError, match="missing cameras"):
        policy(batch)


def test_language_cache_is_used_across_calls(policy, spec):
    policy.text_encoder.clear_cache()
    batch = make_batch(spec, language=["pick up the milk", "pick up the ketchup"])
    policy(batch)
    assert policy.text_encoder.cache_stats["size"] == 2
    misses_after_first = policy.text_encoder.cache_stats["misses"]

    policy(batch)
    # No new strings, so no new text-tower calls (§4.1).
    assert policy.text_encoder.cache_stats["misses"] == misses_after_first


def test_language_tokens_are_per_token_not_pooled(policy, spec):
    """Planned deviation from §4.1: keep per-token embeddings.

    seq_correlated tasks differ only by the object noun, and pooled sentence vectors
    for those are nearly collinear.
    """
    tokens = policy.text_encoder(["pick up the milk"])
    assert tokens.ndim == 3
    assert tokens.shape[1] > 1, "language must contribute more than one token"


def test_distinct_instructions_give_distinct_embeddings(policy):
    """If instructions are not discriminable, the study measures under-conditioning."""
    tokens = policy.text_encoder(
        ["pick up the milk and place it in the basket",
         "pick up the ketchup and place it in the basket"]
    )
    assert not torch.allclose(tokens[0], tokens[1], atol=1e-4)


def test_task_id_is_not_a_policy_input(policy, spec):
    """§0: task identity is never a policy input.

    The forward path must work with no task_id present at all.
    """
    batch = make_batch(spec)
    assert "task_id" not in batch
    out = policy(batch)
    assert torch.isfinite(out["loss"])


def test_gradients_reach_trunk_and_decoder_but_not_backbone(policy, spec):
    policy.zero_grad(set_to_none=True)
    policy(make_batch(spec))["loss"].backward()

    assert policy.trunk.blocks[0].attn.q_proj.weight.grad is not None
    assert policy.flow_head.blocks[0].cross_attn.q_proj.weight.grad is not None
    for param in policy.vision_encoder.backbone.parameters():
        assert param.grad is None


def test_build_rejects_out_of_spec_trunk_depth(spec):
    with pytest.raises(ValueError, match="§4.2 specifies 6-8 trunk layers"):
        FlowPolicy(spec, n_trunk_layers=4, pretrained=False)


def test_build_rejects_out_of_spec_d_model(spec):
    with pytest.raises(ValueError, match="§4.2 specifies d_model 384-512"):
        FlowPolicy(spec, d_model=768, pretrained=False)
