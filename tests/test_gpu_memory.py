"""Peak GPU memory windows (flowcl/utils/gpu_memory.py): None without CUDA; an inner reset does
not cut an outer window short."""

from __future__ import annotations

import pytest
import torch

from flowcl.utils import gpu_memory
from flowcl.utils.gpu_memory import PeakWindow


def test_cpu_windows_report_not_measured():
    with PeakWindow("cpu") as w:
        torch.zeros(10)
    assert (w.allocated, w.reserved) == (None, None)


def test_without_cuda_reset_returns_none(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    assert gpu_memory.reset_peak_memory_stats() is None
    with PeakWindow("cuda") as w:
        pass
    assert w.allocated is None


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_an_inner_reset_is_folded_into_the_outer_window():
    size = 64 * 2**20
    with PeakWindow("cuda") as outer:
        big = torch.empty(size, dtype=torch.uint8, device="cuda")
        del big
        gpu_memory.reset_peak_memory_stats()  # what projected LoRA does for its own timing
        small = torch.empty(1024, dtype=torch.uint8, device="cuda")
        del small
    assert outer.allocated >= size
    assert outer.reserved >= outer.allocated


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_nested_windows_each_see_their_own_peak():
    size = 32 * 2**20
    with PeakWindow("cuda") as outer:
        x = torch.empty(2 * size, dtype=torch.uint8, device="cuda")
        del x
        with PeakWindow("cuda") as inner:
            y = torch.empty(size, dtype=torch.uint8, device="cuda")
            del y
    assert outer.allocated >= 2 * size
    assert size <= inner.allocated < 2 * size + 2**20
