"""Serve one Piper checkpoint to PiperStudio over HTTP (contract: docs/piper_deploy_spec.md §2).

    uv run python scripts/hw/serve_piper.py \
        --checkpoint results/piper_aug/single__piper__sausage_pot_lid__seed0/checkpoints/final.pt

The server listens on 127.0.0.1:8765 (PiperStudio itself is on 8000). Open
http://127.0.0.1:8765/ for the status page and /metadata for what the robot side reads.
Every request is logged to ``results/piper_serve/<run>__<checkpoint>/requests.jsonl``
unless ``--log-dir`` says otherwise.

The number of training demos comes from the run's ``config.yaml`` (``data.n_demos``),
two levels above the checkpoint; ``--n-demos`` overrides it. Pause any training on the
GPU first (``kill -STOP <pid>``), so inference latency stays steady.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from flowcl.deploy.piper_server import DEFAULT_PORT, PiperPolicyService, make_server  # noqa: E402
from flowcl.utils.libero_paths import repo_root  # noqa: E402


def trained_n_demos(checkpoint: Path) -> int | None:
    """``data.n_demos`` of the run that wrote the checkpoint, or None without a config."""
    from omegaconf import OmegaConf

    config = checkpoint.resolve().parent.parent / "config.yaml"
    if not config.is_file():
        return None
    return OmegaConf.load(config).data.get("n_demos")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--host", default="127.0.0.1",
                        help="0.0.0.0 to accept other machines; the default serves this PC only.")
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=0, help="Noise for requests that send no seed.")
    parser.add_argument("--n-demos", type=int, default=None,
                        help="Training demos (cache episodes 0..n-1); default: the run's config.yaml.")
    parser.add_argument("--log-dir", type=Path, default=None)
    args = parser.parse_args()

    n_demos = args.n_demos if args.n_demos is not None else trained_n_demos(args.checkpoint)
    run = args.checkpoint.resolve().parent.parent.name
    log_dir = args.log_dir or repo_root() / "results" / "piper_serve" / f"{run}__{args.checkpoint.stem}"
    service = PiperPolicyService(args.checkpoint, device=args.device, n_demos=n_demos, seed=args.seed,
                                 log_path=log_dir / "requests.jsonl")
    meta = service.metadata()
    print(f"[serve_piper] {meta['checkpoint']} (sha256 {meta['checkpoint_sha256'][:16]}…), "
          f"run {meta['run_id']}, {meta['n_demos_trained']} training demos", flush=True)
    print(f"[serve_piper] instruction: {meta['instruction']!r}", flush=True)
    print(f"[serve_piper] start pose {meta['start_pose']}; held out "
          f"{[(h['dataset'], h['episode_index']) for h in meta['held_out']]}", flush=True)
    print(f"[serve_piper] logging to {log_dir / 'requests.jsonl'}", flush=True)
    server = make_server(service, args.host, args.port)
    print(f"[serve_piper] listening on http://{args.host}:{server.server_address[1]}/", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
