"""Peak GPU memory over a measurement window, robust to resets inside it (status.md §5 item 2).

``torch.cuda.max_memory_allocated`` reports the peak since the last
``reset_peak_memory_stats``. A component that resets the counter for its own measurement
(projected LoRA times its training phase) would cut any outer window short. Code inside a
window therefore resets through :func:`reset_peak_memory_stats` here, which first folds the
current peaks into every open window, so each window still reports its true maximum.

Only the PyTorch CUDA caching allocator is measured: ``allocated`` is memory held by tensors,
``reserved`` what the allocator holds from the driver. EGL rendering and the CUDA context are
not included. On a machine without CUDA every value is ``None`` ("not measured"), never 0.
"""

from __future__ import annotations

import torch

_open: list["PeakWindow"] = []


def _cuda(device) -> bool:
    device = torch.device(device) if device is not None else None
    return torch.cuda.is_available() and (device is None or device.type == "cuda")


def _current_peaks() -> tuple[int, int]:
    return int(torch.cuda.max_memory_allocated()), int(torch.cuda.max_memory_reserved())


def reset_peak_memory_stats() -> tuple[int, int] | None:
    """Fold the current peaks into every open window, then reset the counter.

    Returns the peaks that were current before the reset (``None`` without CUDA)."""
    if not torch.cuda.is_available():
        return None
    peaks = _current_peaks()
    for window in _open:
        window._fold(*peaks)
    torch.cuda.reset_peak_memory_stats()
    return peaks


class PeakWindow:
    """``with PeakWindow(device) as w: ...`` then ``w.allocated`` / ``w.reserved`` (bytes or None)."""

    def __init__(self, device=None):
        self.enabled = _cuda(device)
        self.allocated: int | None = None
        self.reserved: int | None = None
        self._max = (0, 0)

    def _fold(self, allocated: int, reserved: int) -> None:
        self._max = (max(self._max[0], allocated), max(self._max[1], reserved))

    def start(self) -> "PeakWindow":
        if self.enabled:
            reset_peak_memory_stats()
            self._max = (0, 0)
            _open.append(self)
        return self

    def stop(self) -> tuple[int | None, int | None]:
        if self.enabled:
            self._fold(*_current_peaks())
            if self in _open:
                _open.remove(self)
            self.allocated, self.reserved = self._max
        return self.allocated, self.reserved

    def __enter__(self) -> "PeakWindow":
        return self.start()

    def __exit__(self, *exc) -> None:
        self.stop()
