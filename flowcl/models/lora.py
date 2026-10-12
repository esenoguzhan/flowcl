"""Low-rank adapters on ``nn.Linear`` layers (Hu et al., ICLR 2022, arXiv 2106.09685v2).

A rank-``r`` adapter adds ``(alpha / r) * B A x`` to a frozen layer's output ``W x + b`` (§4.1,
Eq. 3), as the authors' ``loralib`` computes it (``x @ A^T @ B^T * scaling``). ``B`` starts at zero,
so the adapted layer equals the base at the first step. ``A`` is initialised exactly as the released
implementation does, ``kaiming_uniform_(A, a=sqrt(5))`` (U(+-1/sqrt(d_in))); the paper's text says
only "random Gaussian" without a scale, and the released code's comment calls the difference
immaterial, so the code's choice removes one free parameter. ``A`` is drawn on the CPU from the
caller's generator, so it does not depend on the device or touch the global RNG. Merging adds
``(alpha / r) * B A`` to ``W`` in fp32 (§4.1, "No Additional Inference Latency"; ``loralib``'s
``merge_weights``); detaching removes the hook and both parameters, so the layer's state dict is
the base layer's again.

The adapters are registered as parameters *on the layer* (``lora_A``, ``lora_B``), so the
optimiser and gradient clipping, which collect ``policy.parameters()``, see them without any
special case. No adapter dropout (``loralib``'s default ``lora_dropout = 0``).
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import nn

INIT_SPEC = ("A: kaiming_uniform_(a=sqrt(5)) = U(-1/sqrt(d_in), 1/sqrt(d_in)) on the CPU from the "
             "adapter generator (the released loralib initialisation); B: zeros")
_HOOK = "_lora_hook"
_SCALE = "_lora_scale"


def is_adapted(module: nn.Linear) -> bool:
    return hasattr(module, _HOOK)


def _forward_hook(module: nn.Linear, args, output):
    x = args[0]
    return output + getattr(module, _SCALE) * F.linear(F.linear(x, module.lora_A), module.lora_B)


def attach(module: nn.Linear, rank: int, alpha: float, generator: torch.Generator) -> int:
    """Attach a fresh rank-``rank`` adapter to ``module``; return its parameter count."""
    if not isinstance(module, nn.Linear):
        raise TypeError(f"LoRA adapts nn.Linear layers, got {type(module).__name__}")
    if is_adapted(module):
        raise RuntimeError("this layer already carries an adapter")
    if int(rank) < 1:
        raise ValueError(f"rank must be >= 1, got {rank}")
    weight = module.weight
    a = torch.empty(int(rank), module.in_features, dtype=torch.float32)
    nn.init.kaiming_uniform_(a, a=math.sqrt(5), generator=generator)
    module.register_parameter("lora_A", nn.Parameter(a.to(device=weight.device, dtype=weight.dtype)))
    module.register_parameter("lora_B", nn.Parameter(
        torch.zeros(module.out_features, int(rank), device=weight.device, dtype=weight.dtype)))
    setattr(module, _SCALE, float(alpha) / int(rank))
    setattr(module, _HOOK, module.register_forward_hook(_forward_hook))
    return module.lora_A.numel() + module.lora_B.numel()


def delta(module: nn.Linear) -> torch.Tensor:
    """``(alpha / r) * B A`` in fp32, shaped like ``module.weight``."""
    return getattr(module, _SCALE) * (module.lora_B.detach().float() @ module.lora_A.detach().float())


def merge_and_detach(module: nn.Linear) -> dict:
    """Merge the adapter into ``W`` (fp32), then remove it; return the update's norms."""
    if not is_adapted(module):
        raise RuntimeError("this layer carries no adapter")
    with torch.no_grad():
        update = delta(module)
        base_norm = float(module.weight.detach().float().norm())
        module.weight.add_(update.to(module.weight.dtype))
    norm = float(update.norm())
    detach(module)
    return {"delta_fro": norm, "weight_fro_before": base_norm,
            "relative": norm / base_norm if base_norm > 0 else math.inf}


def detach(module: nn.Linear) -> None:
    """Remove the hook and both adapter parameters without merging."""
    getattr(module, _HOOK).remove()
    delattr(module, _HOOK)
    delattr(module, _SCALE)
    del module._parameters["lora_A"]
    del module._parameters["lora_B"]


# ---- projected adapters (projected LoRA, an InfLoRA adaptation): the update confined to P⊥ -----------
#
# A projected adapter keeps a protected input basis M (d_in x k, orthonormal columns) and uses the
# effective down-projection A_eff = A - (A M) M^T, formed differentiably at every forward pass, so the
# update (scale) B A_eff satisfies Delta W M = 0 in exact arithmetic for any A and B, whatever the
# optimiser does to A. That equals projecting every input (x P⊥), at r d_in k instead of a pass over
# every activation. With M = None the adapter is unprojected. ``train_A=False`` freezes A (InfLoRA's
# fixed down-projection); its rows must already lie in the complement.

_PSCALE = "_plora_scale"
_PHOOK = "_plora_hook"
_PM = "_plora_M"


def is_projected(module: nn.Linear) -> bool:
    return hasattr(module, _PHOOK)


def effective_down(module: nn.Linear) -> torch.Tensor:
    a = module.lora_A
    m = getattr(module, _PM)
    return a if m is None else a - (a @ m) @ m.T


def _projected_hook(module: nn.Linear, args, output):
    x = args[0]
    return output + getattr(module, _PSCALE) * F.linear(F.linear(x, effective_down(module)), module.lora_B)


def attach_projected(module: nn.Linear, a_init: torch.Tensor, memory: torch.Tensor | None, scale: float,
                     train_a: bool) -> int:
    """Attach a projected adapter with down-projection ``a_init`` (r x d_in) and B = 0; return the
    number of trainable parameters it adds."""
    if not isinstance(module, nn.Linear):
        raise TypeError(f"projected adapters adapt nn.Linear layers, got {type(module).__name__}")
    if is_projected(module) or is_adapted(module):
        raise RuntimeError("this layer already carries an adapter")
    weight = module.weight
    r = int(a_init.shape[0])
    if a_init.shape != (r, module.in_features) or r < 1:
        raise ValueError(f"a_init must be (r >= 1, {module.in_features}), got {tuple(a_init.shape)}")
    module.register_parameter("lora_A", nn.Parameter(a_init.to(device=weight.device, dtype=weight.dtype),
                                                     requires_grad=bool(train_a)))
    module.register_parameter("lora_B", nn.Parameter(
        torch.zeros(module.out_features, r, device=weight.device, dtype=weight.dtype)))
    m = None if memory is None else memory.to(device=weight.device, dtype=weight.dtype)
    setattr(module, _PM, m)
    setattr(module, _PSCALE, float(scale))
    setattr(module, _PHOOK, module.register_forward_hook(_projected_hook))
    return module.lora_B.numel() + (module.lora_A.numel() if train_a else 0)


def projected_delta(module: nn.Linear) -> torch.Tensor:
    """The intended update ``scale * B A_eff`` in fp32, shaped like ``module.weight``."""
    with torch.no_grad():
        return getattr(module, _PSCALE) * (module.lora_B.detach().float() @ effective_down(module).detach().float())


def merge_projected(module: nn.Linear) -> tuple[torch.Tensor, torch.Tensor]:
    """Merge the intended update into W (fp32) and detach; return ``(intended, realized)`` where
    realized = W_after - W_before."""
    with torch.no_grad():
        intended = projected_delta(module)
        before = module.weight.detach().clone()
        module.weight.add_(intended.to(module.weight.dtype))
        realized = module.weight.detach().float() - before.float()
    detach_projected(module)
    return intended, realized


def detach_projected(module: nn.Linear) -> None:
    getattr(module, _PHOOK).remove()
    for attr in (_PHOOK, _PSCALE, _PM):
        delattr(module, attr)
    del module._parameters["lora_A"]
    del module._parameters["lora_B"]
