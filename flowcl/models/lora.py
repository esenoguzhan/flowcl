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
