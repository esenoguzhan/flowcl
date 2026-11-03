# Piper: the milk session (pick_milk, 40 demos) and six policies on it, the front-camera mask crossed with no augmentation, daylight 0.3 and colour 0.3. All six fit their demos (training-episode error 0.0029–0.0043 rad); the three unmasked ones read the leader arm, so the masked ones are the ones to try on the robot

**Status: thesis plan F1, policies for the robot.** On Mon 2 Nov the user recorded a new task. They
asked for a proper instruction and four policies:
1. the data as recorded;
2. the parts that do not belong to the task masked (the operator, the leader arm);
3. as 1 with daylight augmentation;
4. as 2 with daylight augmentation.

They then asked for two more: as 1 and as 2, with only the old brightness, contrast and saturation
augmentation.
- **Code:**
  - Run 1 was trained at `85f5d09`; runs 2–6 at `0626c4a`, which has the same Python code plus
    the second queue script. All six from clean trees.
  - The corner probe (§7), `scripts/hw/piper_corner_probe.py`, is committed with this record.
- **Data:** 40 demos of one task. Nothing is held out, so there is no offline generalisation number:
  the open-loop check measures fitting only (§6).
- **Not done:** robot rollouts. They decide between the variants; the offline numbers cannot.

## 1. The session (checked Mon 2 Nov)

- **Source:** `~/piper_datasets/pick_milk`, recorded 16:53–17:22 in PiperStudio.
  - Fingerprint `b198239c…`, pinned in `flowcl/data/piper_tasks.py` (source `PICK_MILK`).
  - 40 episodes, 11,757 frames, 199–416 ticks (13.3–27.7 s). LeRobot v3.0, 15 fps, the same features
    as the sausage sessions.
- **The task:** every demo picks a small milk carton off the table, puts it on the plate and returns
  home. The carton and plate positions vary. The label is "pick up the milk" on every episode.
- **Nothing is excluded.**
- **Speed:** largest joint step per tick, p99 0.083 rad (max 0.147).
  - That is under PiperStudio's clamp of 0.1 rad per tick at p99.
  - The sausage sessions were at 0.090–0.156.

## 2. The instruction

- **The registry's instruction:** "pick up the milk carton and put it on the blue plate"
  (`MILK_PLATE`). It replaces the recorded label in the cache.
- **Blue, not purple:** the user suggested "purple plate". In both cameras the plate is
  turquoise-blue; this was checked against skin tones and the arm's red LED, so the white balance is
  not off.
  - The word can be changed. That needs both caches rebuilt and the policies retrained, since the
    instruction is stored in the cache and the checkpoint.
- **For a single-task policy the instruction barely matters** (status §2): it has no other task to
  tell apart.
  - What it does fix: the server refuses a request whose instruction is not this one.
  - It also keeps the data usable for multi-task training later.

## 3. Masked and unmasked

- **`milk_plate`** uses `agilex_piper`'s front mask unchanged: `[[0,0],[290,0],[290,180],[255,300],[0,310]]`,
  the polygon widened on 2 Nov for the v4 sausage view.
  - Checked on 92 review frames (`datasets/piper/review/milk_plate/mask_review_*.jpg`): the operator
    and the leader arm stay inside it on every one, and the carton and the plate stay visible.
  - The follower's shoulder is often blacked out; the joint state carries it.
  - The wrist camera is never masked.
- **`milk_plate_unmasked`** uses the same recordings with `configs/embodiment/agilex_piper_unmasked.yaml`.
  That file is `agilex_piper` without the mask, identical otherwise.
- **The plumbing (`85f5d09`):**
  - Each registry task names the embodiment its cache is built and trained with
    (`PiperTask.embodiment`), and `prepare_piper` builds with it.
  - A cache refuses to load under the other embodiment's mask.
  - The server accepts a checkpoint only if its embodiment is its task's.
  - Caches: `datasets/piper/cache/milk_plate.h5` and `milk_plate_unmasked.h5`. Input sheets:
    `datasets/piper/review/<task>/cache_inputs.jpg`.

## 4. Augmentation

Both kinds are drawn per sample and per camera from the augmentation generator, and on both cameras.
They never change the run's batch order or its flow-time and noise draws. Neither kind shifts the
images.
- **Daylight** (new in `85f5d09`; `flowcl.data.augment.daylight`, `--augment-daylight S`, S = 0.3).
  Three steps, in order:
  1. **A colour temperature:** red × (1 + t), blue × (1 − t), with t ∈ [−S/2, S/2].
  2. **A linear illumination ramp:** a random direction, ± up to S at the frame's corners.
  3. **The colour jitter below at S.**
  - S = 0.3 was chosen from a sheet of augmented frames. It gives warm, cool, dim and washed-out
    looks with the carton and the plate still clear.
  - Contrast jitter sometimes lifts the masked corner from black to dark grey, as in the 1 Nov
    `piper_aug` run.
- **Colour** (`--augment-color 0.3`): brightness, contrast and saturation factors in [0.7, 1.3].
  - This is the colour half of the 1 Nov augmentation. That run used 0.2, together with 8 px shifts.
  - 0.3 is the jitter daylight 0.3 applies inside. So runs 5 and 6 differ from 3 and 4 only by the
    colour temperature and the ramp.

## 5. Training

**Recipe:** Gate 0's single-task recipe: `flowpolicy_base`, 30k steps, batch 64, lr 1e-4, AMP,
seed 0, a checkpoint every 5k steps. That is about 163 epochs (the 30-demo sausage policy: 140).

**Queues:**
- `scripts/queue_piper_milk.sh` (runs 1–4; tmux `piper_milk`) ran 22:30 to 01:08.
- `scripts/queue_piper_milk_color.sh` (runs 5–6; tmux `piper_milk_color`) was chained after it and
  ran 01:08 to 02:26.
- Logs: `results/piper_queue/milk/` and `results/piper_queue/milk_color/`.
- The summary lines at the end of each `queue.log` are in file-name order: masked + daylight,
  masked, raw + daylight, raw. Read the run's `offline_check.json` for which is which.

| # | Front camera | Augmentation | Run | Time | Last 1k steps' loss |
|---|---|---|---|---:|---:|
| 1 | as recorded | none | `results/single__piper__milk_plate_unmasked__seed0` | 38.9 min | 0.0179 |
| 2 | masked | none | `results/single__piper__milk_plate__seed0` | 39.0 min | 0.0185 |
| 3 | as recorded | daylight 0.3 | `results/piper_daylight/single__piper__milk_plate_unmasked__seed0` | 39.7 min | 0.0235 |
| 4 | masked | daylight 0.3 | `results/piper_daylight/single__piper__milk_plate__seed0` | 39.5 min | 0.0244 |
| 5 | as recorded | colour 0.3 | `results/piper_color/single__piper__milk_plate_unmasked__seed0` | 39.2 min | 0.0218 |
| 6 | masked | colour 0.3 | `results/piper_color/single__piper__milk_plate__seed0` | 39.2 min | 0.0223 |

**Mean logged loss per 5k steps:**

| # | 0–5k | 5–10k | 10–15k | 15–20k | 20–25k | 25–30k |
|---|---:|---:|---:|---:|---:|---:|
| 1 | 0.181 | 0.053 | 0.038 | 0.026 | 0.020 | 0.015 |
| 2 | 0.193 | 0.054 | 0.039 | 0.027 | 0.020 | 0.016 |
| 3 | 0.189 | 0.058 | 0.043 | 0.031 | 0.024 | 0.021 |
| 4 | 0.195 | 0.061 | 0.045 | 0.032 | 0.025 | 0.021 |
| 5 | 0.190 | 0.056 | 0.041 | 0.029 | 0.022 | 0.019 |
| 6 | 0.191 | 0.058 | 0.042 | 0.030 | 0.023 | 0.020 |

- Augmentation ends 20–30 % higher, daylight more than colour; the mask changes little.
- Every run has checkpoints at 5k, 10k, …, 25k and `final.pt`.

## 6. Offline open-loop check (training episodes 0–5 only)

**Method:** `scripts/hw/piper_offline_check.py` on `final.pt`, with 223 chunks. Output:
`<run>/offline_check.json`. Baselines on the same ticks: holding the state 0.0379 rad, holding the
last command 0.0313.

| # | Front camera | Augmentation | Executed 8 ticks | 16 ticks | Worst joint | Gripper |
|---|---|---|---:|---:|---:|---:|
| 1 | as recorded | none | **0.0029 rad** (0.0027–0.0032) | 0.0030 | 0.0039 | 0.23 mm |
| 2 | masked | none | **0.0032** (0.0028–0.0036) | 0.0032 | 0.0043 | 0.23 mm |
| 3 | as recorded | daylight 0.3 | **0.0040** (0.0034–0.0044) | 0.0043 | 0.0053 | 0.33 mm |
| 4 | masked | daylight 0.3 | **0.0043** (0.0035–0.0052) | 0.0048 | 0.0058 | 0.35 mm |
| 5 | as recorded | colour 0.3 | **0.0034** (0.0029–0.0039) | 0.0036 | 0.0046 | 0.26 mm |
| 6 | masked | colour 0.3 | **0.0037** (0.0029–0.0043) | 0.0040 | 0.0050 | 0.31 mm |

**Readings:**
- **All six fit their demos:** 7–11× better than holding the last command. This is the same range
  as the 30-demo sausage policy on its training episodes (0.0036).
- **Augmentation costs a little fit,** as it should: daylight more than colour.
- **This is not generalisation.** Nothing is held out, so this check cannot rank the variants for the
  robot.
  - On the sausage task, a policy this tight on its training episodes was no better than holding
    still on unseen ones (1 Nov record §5).
  - Augmentation lowered that held-out error there by 6 % (§10).

## 7. Do the unmasked policies read the leader arm? The corner probe

**Why it matters:** in the recordings the leader arm, in the masked corner, moves exactly as the
follower is commanded. An unmasked policy can read the next command off it. On the robot the leader
arm stands still.

**The probe** (`scripts/hw/piper_corner_probe.py`, `final.pt`, no retraining):
- It runs the offline check on training episodes 0–5 twice, with the same noise:
  - **recorded:** the frames as recorded;
  - **corner still:** the front camera's masked corner (4,708 of 16,384 policy-frame pixels,
    `agilex_piper`'s polygon) held at the episode's first frame. That is the leader at rest, as in a
    rollout.
- It also reports how far the 16-tick plans move between the two inputs.
- **For the masked policies the corner is black either way.** Only the border pixels that the mask
  touches in part, which include some of the follower's shoulder, change. Their numbers are the
  floor.
- Output: `<run>/corner_probe.json`.

| # | Front camera | Augmentation | Recorded | Corner still | Plans move (mean, per-episode range) | Plans move, p95 (worst episode) |
|---|---|---|---:|---:|---:|---:|
| 1 | as recorded | none | 0.0029 | **0.0193** (0.018–0.024) | 0.026 (0.022–0.032) | 0.125 |
| 3 | as recorded | daylight 0.3 | 0.0040 | **0.0150** (0.013–0.018) | 0.021 (0.019–0.025) | 0.102 |
| 5 | as recorded | colour 0.3 | 0.0034 | **0.0151** (0.013–0.018) | 0.021 (0.018–0.029) | 0.101 |
| 2 | masked | none | 0.0032 | 0.0047 (0.004–0.005) | 0.004 (0.003–0.005) | 0.015 |
| 4 | masked | daylight 0.3 | 0.0043 | 0.0044 (0.004–0.005) | 0.001 (0.001–0.002) | 0.005 |
| 6 | masked | colour 0.3 | 0.0037 | 0.0041 (0.003–0.005) | 0.002 (0.001–0.002) | 0.007 |

Errors are joint MAE over the executed 8 ticks in rad; plan moves are in rad over the 16 ticks.

**Readings:**
- **The unmasked policies use the leader arm.** With it standing still, their error rises 4–7×, to
  about half of holding the last command. Their plans move 0.02–0.03 rad on average, and their
  worst 5 % by 0.10–0.13 rad.
  - On the robot every input will look like "corner still". Part of what these policies learned
    will be missing there.
  - Augmentation lowers the dependence somewhat (0.026 to 0.021) but does not remove it.
- **The masked policies are unaffected:** their plans move 0.001–0.004 rad.
  - The plain masked policy moves most of the three, probably through the half-masked border
    pixels.
- **Caveats:**
  - Holding the corner also holds the operator's body and part of the follower's shoulder.
  - The follower's shoulder is also in the joint state, so it is unlikely to explain a 4–7× rise.
  - Single seed, training episodes only.

## 8. Next

- **For the robot: the masked policies, 2, 4 and 6.** Serve with:

  ```bash
  uv run python scripts/hw/serve_piper.py --checkpoint results/single__piper__milk_plate__seed0/checkpoints/final.pt
  ```

  (`results/piper_daylight/…` and `results/piper_color/…` for 4 and 6). Pause any training while
  serving.
- **Comparing no augmentation, colour and daylight needs rollouts:**
  - in the same scenes;
  - ideally also under lighting different from 2 Nov afternoon's, which is what daylight is for.
- **If the unmasked policies are tried, expect plans that depend on where the leader arm stands.**
  The probe above is the offline evidence for that.
- **A held-out check would need a split,** for example the last 6 demos held out, and retraining.
  Nothing was held out, at the user's choice for these policies.
