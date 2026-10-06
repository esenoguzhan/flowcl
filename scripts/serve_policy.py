"""Serve one flowcl checkpoint to the Dobot X-Trainer runner over a websocket (branch dobot-hw).

    uv run python scripts/serve_policy.py --checkpoint results/<run>/checkpoints/final.pt
    # robot PC: python scripts/hw/dobot_run.py --server ws://10.147.20.55:8000 probe ...

Logs every request to ``results/dobot_serve/<checkpoint run>/requests.jsonl`` and, with
``--dump-requests N``, saves the first N policy inputs for the live-vs-dataset frame check.
"""

from __future__ import annotations

import argparse
import asyncio
import time
from pathlib import Path

from flowcl.deploy.server import PolicyService, RequestLog, serve
from flowcl.utils.libero_paths import repo_root


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--euler-steps", type=int, default=10)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--envelope-margin", type=float, default=0.15, help="rad beyond the training actions")
    parser.add_argument("--holdout", type=int, default=10)
    parser.add_argument("--dump-requests", type=int, default=5)
    parser.add_argument("--log-dir", type=Path, default=None)
    args = parser.parse_args()

    ckpt = args.checkpoint.resolve()
    run_name = ckpt.parent.parent.name if ckpt.parent.name == "checkpoints" else ckpt.stem
    log_dir = args.log_dir or repo_root() / "results" / "dobot_serve" / f"{run_name}__{ckpt.stem}"
    print(f"[serve] loading {ckpt}", flush=True)
    t = time.perf_counter()
    service = PolicyService(ckpt, device=args.device, euler_steps=args.euler_steps, seed=args.seed,
                            envelope_margin_rad=args.envelope_margin, holdout=args.holdout,
                            log_dir=log_dir, dump_requests=args.dump_requests)
    meta = service.metadata()
    print(f"[serve] ready in {time.perf_counter() - t:.1f}s: trained {meta['trained']}, "
          f"execute {meta['execute_ticks']} of {meta['chunk_30hz']} ticks, logs in {log_dir}", flush=True)
    asyncio.run(serve(service, args.host, args.port, RequestLog(log_dir / "requests.jsonl")))


if __name__ == "__main__":
    main()
