# Piper: a third session (pick_sausage_v4, 17 demos), a wider front-camera mask, and one policy on every demo (53, nothing held out)

**Status: thesis plan F1, the policy for the robot.** The user recorded a third session on Mon 2 Nov
and asked for a policy trained on all the data, with nothing kept for testing.
- **Code:** `230a52c`. The run `results/single__piper__sausage_pot_lid_all__seed0` was trained at that
  commit, from a clean tree.
- **Data:** 53 demos of the same task (sausage into the pot, lid on the pot): `pick_sausage` 20,
  `pick_sausage_v3` 16, `pick_sausage_v4` 17. 22,545 frames.
- **Nothing is held out.** The offline open-loop check therefore measures fitting only; there is no
  generalisation number for this policy. The earlier held-out results (record
  `2026-11-01_piper_f1_prep.md` §5–§10) are the only offline evidence on unseen episodes.

## 1. The new session (checked Mon 2 Nov)

- **Source:** `~/piper_datasets/pick_sausage_v4`, recorded 13:29–13:57 in PiperStudio. 17 episodes,
  20.5–32.0 s (308–480 ticks). LeRobot v3.0, 15 fps, the same features as on 30 Oct.
  - The first look found 14 episodes: the recording session was still open, the last parquet had no
    footer and the episode metadata for 14–16 was not written. The user ended the session at 13:57:36;
    the folder was fingerprinted after that.
  - Integrity as before: rows, frame indices and timestamps contiguous, one video frame per row.
- **Labels:** episode 0 "pick the sausage", 1–16 "pick up the sausage". The task is the same in every
  episode (contact sheets of 8 frames per episode): the sausage into the pot, then the lid on the pot.
  The lid now starts off the pot, and the object layout varies.
  - The cache keeps both labels (`raw_labels` is now a list per folder); the registry's instruction
    replaces them.
- **Episode 0** ends with the arm over the pot, not at home (1.43 rad from its start pose). Kept, like
  `pick_sausage` raw 21.
- **The front camera's view changed.** The background (monitor, PC tower) is where it was; the robot,
  the leader arm and the table edge sit about 50–60 px lower in the frame. The rollouts must use this
  view (the one the camera has now), not the 30 Oct one.
- **Fast wrist rolls:** joint 6 steps up to 0.43 rad per tick (v4 ep13 t70; five episodes above 0.26),
  with the follower about 0.4 rad behind. Largest joint step per tick, p99: 0.090 (`pick_sausage`), 0.105
  (`v3`), 0.156 (`v4`). The follower itself turns joint 6 at up to 0.385 rad per tick (5.8 rad/s).
  PiperStudio clamps every joint to `max_joint_speed_rad_s` = 1.5 rad/s (0.1 rad per tick). In the
  follower's measured motion some joint is faster than that on 4.6 % of `v4` ticks (0.35 % on
  `pick_sausage`, 1.1 % on `v3`), so in rollouts these rolls would run up to 4× slower than
  demonstrated.
- **Exclusions unchanged:** `pick_sausage` 0 (front camera in another pose) and 15 (a 3.7 s arm return
  with no task). 22 + 16 + 17 = 55 recorded, 53 used.

## 2. The front-camera mask, widened

- **Why:** in the v4 view the leader arm and the operator's hands reach past the old polygon's
  diagonal edge, by up to about 90 px (e.g. v4 ep2 t87, ep11 t107). A policy trained on that would see
  the leader arm, whose pose is the action.
- **New polygon:** `[[0,0],[290,0],[290,180],[255,300],[0,310]]` (old:
  `[[0,0],[260,0],[260,115],[150,250],[0,260]]`). It is one mask for all three sessions, since serving
  applies one.
  - Checked on 102 v4 frames, 6 per episode at the ticks where the leader arm is farthest from its
    start pose: the leader arm and the hands are inside on every one.
  - Checked on 32 `pick_sausage` / `v3` frames: the pot, the lid and the sausage stay visible.
  - **The cost:** in the v4 view the follower's elbow passes through the same image region as the
    leader arm, so it is often blacked out. The joint state carries it.
- **Earlier policies are unaffected:** every checkpoint carries its own mask, and the server applies
  the checkpoint's. The old caches load only under their own checkpoints' specs; rebuilding an old
  task now gives it the new mask.

## 3. Setup

- **Task** `sausage_pot_lid_all` in `flowcl/data/piper_tasks.py`: the three sources in order, 53
  episodes, `holdout=0`.
- **Recipe:** §10 B of the 1 Nov record, the best one offline: `flowpolicy_base`, 30k steps, batch 64,
  lr 1e-4, AMP, seed 0, image augmentation (shifts 8 px, colour 0.2), a checkpoint every 5k.
  About 85 epochs (140 for the 30-demo policy).
- **Queue:** `bash scripts/queue_piper_single.sh sausage_pot_lid_all 53 --augment-shift-px 8
  --augment-color 0.2 --checkpoint-every 5000` (tmux `piper_all`), then the offline check.

## 4. Training

RESULTS_PENDING

## 5. Offline check (training episodes only)

RESULTS_PENDING

## 6. Next

- Serve `final.pt` with `scripts/hw/serve_piper.py`; PiperStudio's rollouts use free scenes, since
  the metadata now lists no held-out episodes.
- Whether to raise PiperStudio's joint speed limit for joint 6 (1.5 rad/s against wrist rolls of up
  to 5.8 rad/s in the demos, §1) is the user's call.
