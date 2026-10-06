"""Message codec between the robot-side runner and the policy server.

msgpack, with numpy arrays packed in openpi-client's ``__ndarray__`` convention
(``openpi_client/msgpack_numpy.py``), so a message stays readable by openpi tooling.
Both ends of this link are ours, so the format is pinned here and tested as a pair.

Protocol: on connect the server sends one metadata dict. Each request is a dict; the
reply is a dict, or a ``str`` carrying the server-side traceback (the runner treats a
string reply as a fatal error and stops).
"""

from __future__ import annotations

from typing import Any

import msgpack
import numpy as np


def _pack_default(obj: Any):
    if isinstance(obj, np.ndarray):
        if obj.dtype.kind in ("V", "O", "c"):
            raise ValueError(f"unsupported array dtype {obj.dtype}")
        return {
            b"__ndarray__": True,
            b"data": np.ascontiguousarray(obj).tobytes(),
            b"dtype": obj.dtype.str,
            b"shape": list(obj.shape),
        }
    if isinstance(obj, np.generic):
        return {b"__npgeneric__": True, b"data": obj.item(), b"dtype": obj.dtype.str}
    raise TypeError(f"cannot pack {type(obj)}")


def _unpack_hook(obj: dict):
    if b"__ndarray__" in obj:
        return np.ndarray(
            buffer=obj[b"data"], dtype=np.dtype(obj[b"dtype"]), shape=tuple(obj[b"shape"])
        ).copy()
    if b"__npgeneric__" in obj:
        return np.dtype(obj[b"dtype"]).type(obj[b"data"])
    return obj


def packb(obj: Any) -> bytes:
    return msgpack.packb(obj, default=_pack_default, use_bin_type=True)


def unpackb(data: bytes) -> Any:
    return msgpack.unpackb(data, object_hook=_unpack_hook, raw=False)
