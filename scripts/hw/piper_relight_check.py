"""A Piper checkpoint on held-out demos under real sun shadows: opening, approach and accuracy.

The held-out episodes come from ``--eval-task``'s cache (default ``new_grape_trim2_ho10_336``: the
10 held-out demos cut to their motion, so plain and trimmed checkpoints are scored on the same
ticks). Each is predicted open-loop at ticks 0, 8, 16, ... (flowcl.analysis.open_loop), with the
same noise for every checkpoint, under three conditions:

* ``clean``: the recorded frames;
* ``front_maps`` (primary): the front camera relit by each map of ``--maps`` (the sun session's
  ratio maps, never trained on), averaged over maps within an episode;
* ``front_maps_wrist``: as front_maps, the wrist camera also shaded by a synthetic pattern
  (flowcl.data.augment.synthetic_light, fixed seeds).

Measures (executed ticks 1-8 of each chunk only; every episode weighted equally):

* ``opening_recall``: in the chunk containing the recorded first crossing of 20 mm by the gripper
  command, the prediction opens past 20 mm within its first 8 ticks; ``opening_deferred``: it opens
  only in ticks 9-16; ``premature``: an earlier chunk already opens;
* ``approach_mae``: joint MAE over the chunks starting in the 2 s before the recorded close (the
  close onset of piper_session_diag.py, on the command);
* ``mae``: joint MAE over every chunk;
* ``plausible_reach``: the first chunk the recording moves (> 0.05 rad) has a predicted joint
  displacement at cosine >= 0.5 with the recorded one.

    uv run python scripts/hw/piper_relight_check.py --checkpoint results/.../checkpoints/final.pt
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from flowcl.analysis.open_loop import episode_seed, predict_chunks  # noqa: E402
from flowcl.data.augment import synthetic_light  # noqa: E402
from flowcl.data.piper_adapter import iter_cache_episodes, verify_cache  # noqa: E402
from flowcl.data.piper_tasks import cache_path, get_task  # noqa: E402
from flowcl.data.shadow_maps import load_bank  # noqa: E402
from flowcl.train.checkpoint import load_checkpoint  # noqa: E402

OPEN_M = 0.02
K = 8
RATE_HZ = 15.0


def close_onset(gripper_mm: np.ndarray) -> int | None:
    """As piper_session_diag.first_close: after the gripper is wider than 60 mm, the first tick 10 mm
    below its widest opening so far."""
    wide = np.where(gripper_mm > 60.0)[0]
    if not len(wide):
        return None
    g = gripper_mm[wide[0]:]
    below = np.where(g < np.maximum.accumulate(g) - 10.0)[0]
    return int(wide[0] + below[0]) if len(below) else None


def measures(pred, rec, valid, ticks, state, t_open, t_close) -> dict:
    ex = valid.copy()
    ex[:, K:] = False
    err = np.abs(pred[..., :6] - rec[..., :6]).mean(-1)
    out = {"mae": float(err[ex].mean())}
    if t_close is not None:
        sel = (ticks >= t_close - 2 * RATE_HZ) & (ticks < t_close)
        out["approach_mae"] = float(err[sel][ex[sel]].mean()) if sel.any() else None
    if t_open is not None:
        ci = t_open // K
        g_ex = np.where(ex[ci], pred[ci, :, 6], -np.inf)
        g_late = np.where(valid[ci] & ~ex[ci], pred[ci, :, 6], -np.inf)
        opens = bool(g_ex.max() > OPEN_M)
        out["opening_recall"] = float(opens)
        out["opening_deferred"] = float(not opens and g_late.max() > OPEN_M)
        out["premature"] = float(any(np.where(ex[j], pred[j, :, 6], -np.inf).max() > OPEN_M for j in range(ci)))
    moving = [j for j in range(len(ticks)) if np.abs(rec[j, K - 1, :6] - state[ticks[j], :6]).max() > 0.05]
    if moving:
        j = moving[0]
        d_rec = rec[j, K - 1, :6] - state[ticks[j], :6]
        d_pred = pred[j, K - 1, :6] - state[ticks[j], :6]
        cos = float(d_rec @ d_pred / (np.linalg.norm(d_rec) * np.linalg.norm(d_pred) + 1e-9))
        out["plausible_reach"] = float(cos >= 0.5)
        out["reach_cosine"] = cos
    return out


def relit(ep, front_map=None, wrist_light=None):
    images = dict(ep.images)
    if front_map is not None:
        images["front"] = np.clip(images["front"].astype(np.float32) * front_map, 0, 255).astype(np.uint8)
    if wrist_light is not None:
        images["wrist"] = np.clip(images["wrist"].astype(np.float32) * wrist_light, 0, 255).astype(np.uint8)
    return dataclasses.replace(ep, images=images)


def mean_of(rows: list[dict], key: str):
    vals = [r[key] for r in rows if r.get(key) is not None]
    return float(np.mean(vals)) if vals else None


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--eval-task", default="new_grape_trim2_ho10_336")
    p.add_argument("--maps", type=Path, default=Path("datasets/piper/shadow_maps/table2_eval.npz"))
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--device", default="cuda")
    p.add_argument("--out", type=Path, default=None, help="default: relight_check.json in the run directory")
    args = p.parse_args()

    task = get_task(args.eval_task)
    path = cache_path(task.name)
    verify_cache(path, task)
    loaded = load_checkpoint(args.checkpoint, device=args.device)
    policy, spec, stats = loaded.policy, loaded.spec, loaded.stats
    device = torch.device(args.device)
    maps, provenance = load_bank(args.maps)
    maps = maps.astype(np.float32)
    held = list(range(task.n_train, task.n_episodes))
    episodes = list(iter_cache_episodes(path, spec, episode_indices=held))
    h, w = episodes[0].images["wrist"].shape[1:3]
    rows = {c: [] for c in ("clean", "front_maps", "front_maps_wrist")}
    per_episode = []
    for n, ep in enumerate(episodes):
        raw = ep.meta.get("raw_episode_index", held[n])
        seed = episode_seed(args.seed, f"{task.sources[0].folder}/{raw}")
        g_mm = ep.action[:, 6] * 1000.0
        t_open = int(np.argmax(ep.action[:, 6] > OPEN_M)) if (ep.action[:, 6] > OPEN_M).any() and ep.action[0, 6] <= OPEN_M else None
        t_close = close_onset(g_mm)

        def run(e):
            c = predict_chunks(policy, spec, stats, e, device, seed, args.batch_size)
            return measures(c.predicted, c.recorded, c.valid, c.ticks, ep.state, t_open, t_close)

        clean = run(ep)
        gen = torch.Generator().manual_seed(seed)
        lights = synthetic_light(len(maps), h, w, gen, "cpu").numpy()
        fm = [run(relit(ep, m)) for m in maps]
        fmw = [run(relit(ep, m, lights[i])) for i, m in enumerate(maps)]
        avg = lambda rs: {k: mean_of(rs, k) for k in rs[0]}
        rows["clean"].append(clean)
        rows["front_maps"].append(avg(fm))
        rows["front_maps_wrist"].append(avg(fmw))
        per_episode.append({"raw_episode_index": int(raw), "t_open": t_open, "t_close": t_close, "clean": clean,
                            "front_maps": avg(fm), "front_maps_wrist": avg(fmw)})
        print(f"[relight] episode {raw}: clean mae {clean['mae']:.4f}, maps mae {avg(fm)['mae']:.4f}, "
              f"opening clean/maps {clean.get('opening_recall')}/{avg(fm).get('opening_recall')}", flush=True)
    keys = ("mae", "approach_mae", "opening_recall", "opening_deferred", "premature", "plausible_reach", "reach_cosine")
    summary = {c: {k: mean_of(r, k) for k in keys} for c, r in rows.items()}
    for c, s in summary.items():
        print(f"[relight] {c:16s} " + ", ".join(f"{k} {v:.4f}" if v is not None else f"{k} -" for k, v in s.items()))
    out = args.out or args.checkpoint.parent.parent / "relight_check.json"
    out.write_text(json.dumps({"checkpoint": str(args.checkpoint), "eval_task": task.name, "maps": str(args.maps),
                               "maps_sha256": provenance.get("maps_sha256"), "seed": args.seed,
                               "summary": summary, "per_episode": per_episode}, indent=1) + "\n")
    print(f"[relight] wrote {out}")


if __name__ == "__main__":
    main()
