# Piper second table, held-out arm session (Fri 27 Nov): with the start trim, both trimmed policies left home in 12 of 12 runs and each placed the grape in 4 of 6 unseen scenes; the untrimmed policy stayed home in 4 of 6 and placed 1. The light had no sun stripes, so the shadow augmentation was not tested

**Status: thesis plan F1, the first arm test of the 24–26 Nov queue's recipes on held-out scenes** (queue
`scripts/queue_piper_shadow.sh`, record of the first session `runs/2026-11-24_piper_new_grape_rollouts.md`). The user
ran the session; Claude checked the logs and the recordings afterwards.
- **Zero-shot:** every rollout replanned every 8 ticks with blending off (`execute_k` 8, `blend_s` 0 in every
  `meta.json`), so the commands are the policy's own.
- **Code:** flowcl `f6e23cb` on `piper-table2` (server `serve_piper.py`, live frames through the training video at
  crf 30). PiperStudio `ecc2894` on `policy-picker-groups`, dirty (the 24 Nov dropdown filter and the shadow label).
- **Analysis:** `scripts/hw/piper_session_diag.py` `outcomes`, `scenes` and the new `phases`; `scenes` now also names
  the recorded episode each scene matches. Outputs in `results/piper_rollouts/2026-11-27/diag/`, with front-camera
  strips of every rollout (`front_strips_{0,1,2}.jpg`) and the ep 64 / ep 70 frames (`ep64_end_ep70_holds.jpg`).
- **Logs:** `results/piper_rollouts/2026-11-27/`, copied from `~/piper_rollouts/2026-11-27`: 18 rollouts and 11
  probes, 13:51–14:50.

## 1. The design

Fixed before the session (this conversation, 27 Nov ~13:45):
- **Three seed-0 policies trained on 80 demos, the same 10 held out** (`NEW_GRAPE_HO10`, k-means medoids of the
  grape and plate start pixels). All three: embodiment `agilex_piper_table2_336`, 30k steps, daylight 0.3.

  | | Run (`results/piper_shadow/`) | Recipe | sha256 |
  |---|---|---|---|
  | **D** | `v0/single__piper__new_grape_ho10_336__seed0` | no trim, no shadows | `dc49b9008fbf711a…` |
  | **S** | `s3_mix/single__piper__new_grape_trim_ho10_336__seed0` | start trim + shadows (the rule's recipe) | `a7393786beee8735…` |
  | **B** | `b3_mix/single__piper__new_grape_trim2_ho10_336__seed0` | both trims + shadows | `fee9471867ead720…` |

- **Six held-out scenes**, each set up from the recorded episode's first frame with PiperStudio's overlay:
  episodes 3, 16, 28, 34, 64, 70. One rollout per policy per scene.
- **Order reversed at each scene** (D S B, then B S D, …), so the server switches twice per scene.
- **Annotation** on the Policy tab's form, which still carries the sausage task's questions: "sausage in the pot" =
  the grape was lifted, "lid on the pot" = the grape ended on the plate, the note = what happened and the light.

First plan, dropped before any rollout: the three all-90 policies on these episodes. The user pointed out that they
trained on them; the 80-demo versions make the scenes unseen and keep them reproducible.

## 2. What was checked, and the data's faults

**Scenes, by image rather than by the log.** `scenes` finds the grape and the plate in each rollout's first front
frame and matches the pair to every demo's frame 0:
- every rollout matches its intended episode within 1–7 px (grape and plate together); the next-nearest episode is
  37–73 px away;
- **three rollouts are logged as episode 3** but were set up as their scene's episode: S at 14:20 (episode 28),
  D at 14:31 (34), S at 14:36 (64). Their partners in the same scene are logged correctly, and the grape and plate
  moved under 15 px between them. The record uses the matched episode.
- Cause: PiperStudio's scene picker resets to the first held-out episode whenever the server's metadata arrives
  (`PolicyPanel.tsx`, the `firstHeldOut` effect), which happens at every checkpoint switch; the probes after each
  switch are logged as episode 3 too.

**Outcomes, from the recordings, against the annotations.**
- 15 of 18 rollouts are annotated; the three of the first scene (13:55–13:59) are not. Their outcomes come from the
  logs and frames.
- Each annotation agrees with the gripper trace and the last front frame. The notes' retry counts are looser than
  the traces (S ep 28 "tried 3 times": one open-and-close and one partial opening; B ep 28 "2–3 times": one
  open-and-close, after 50 s with the gripper shut).
- Every label is the task instruction (typed into the label box, which the policy never reads); the light is in
  the notes only.

**Probes:** D's first four (13:51–13:55) were taken with the arm parked and refused on the first jump (J2 0.52–0.55
rad); every probe at the start pose passed.

**Unchanged from 24 Nov:** round trip p50 59–61 ms (max 118); limiter 0.3–5.6 % of ticks for D, 6–14 % for S and
B (24 Nov: 0–12 %); no run reached the floor stop.

## 3. Light

Diffuse, no sun on the table. Over the whole visible table (`scenes`):

| | Demos (23 Nov) | Today | 24 Nov afternoon (sun) |
|---|---|---|---|
| sharp-edge share | 0.000–0.001 | 0.000 | 0.006–0.025 |
| residual after a linear ramp | 0.045–0.052 | 0.057–0.061 | 0.068–0.091 |
| mean grey | 148–157 | 154–160 | |

No stripes, so **this session tests the start trim, not the shadow augmentation.** The ramp residual still sits
just above the demos' range.

## 4. Outcomes

| Episode | D (no trim) | S (start trim + shadows) | B (both trims + shadows) |
|---|---|---|---|
| 3 | stayed home | **placed** 16.5 s | **placed** 12.1 s |
| 16 | stayed home | **placed** 58.7 s, after 32 s shut and a miss; not back home at 60 s | **placed** 11.0 s |
| 28 | stayed home | miss: closed from above 140 mm, then 52 s shut | 50 s shut, opened, never below z 125, closed empty |
| 34 | stayed home | **placed** 13.7 s | **placed** 11.3 s |
| 64 | **placed** 18.0 s | 3 empty closes at grape height (6.0, 10.2, 13.5 s), grape pushed; then 46 s shut | **placed** 13.3 s |
| 70 | lifted; hovered beside the plate at z 160–190 for 52 s | **placed** 12.7 s | lifted; held at the plate's rim, z 106–133, 48 s, never opened |
| **Placed** | **1 of 6** | **4 of 6** (3 first try) | **4 of 6** (all first try) |
| Lifted | 2 of 6 | 4 of 6 | 5 of 6 |
| Left home | 2 of 6 (1.7, 1.1 s) | 6 of 6 (0.2–0.4 s) | 6 of 6 (0.2–0.4 s) |

Times are when the release started. The demos close at z p50 102 mm; every grasp today closed at 98–109 mm and held
the grape at 43–51 mm, except S ep 16 (34 mm, then released from z 148 and still landed on the plate).

## 5. The start

- **D stayed at the start pose in 4 of 6 scenes for the full 60 s**, the failure of the 24 Nov evening session.
  When it moved (ep 64, 70) it left after 1.1–1.7 s and closed 7.4–9.0 s in.
- **S and B left home in every run, 0.2–0.4 s after Go**, and B closed 5.3–6.4 s in.
- The offline start check predicted this: departure within 2 s on the logged demo-light inputs of 24 Nov is 0.06
  for D and 1.00 for S and B; on the arm today, 2 of 6 for D and 6 of 6 for each.

## 6. The failures that remain

1. **Episode 28: grape and plate nearly touching.** 39 px apart in the front frame, against 55 px at the demos' 5th
   percentile (p50 122). Neither trimmed policy came down to the grape: both moved for about 50 s with the gripper
   shut, and their one close was from 125 mm or higher. This is the one place where the offline per-episode numbers
   agree with the arm: the held-out approach error on ep 28 is 0.051–0.078 rad, against 0.017–0.027 on the other five.
2. **Aim at the grasp (S, ep 64).** Three closes at the right height (z 105–110), each on nothing; the fingers pushed
   the grape about 25 px down the image.
3. **No release (ep 70, plate next to the base).** S released at x 225, y 79 mm and placed. B carried the grape to
   x 188–205 (20–37 mm nearer the base), lowered it to the plate's rim by 12 s (the wrist camera shows the rim beside
   the left finger) and held it there, z 106–133, for 48 s. D carried it beside the plate at carry height and hovered.
4. **Long stretches with the gripper shut, away from home**: S ep 16 (32 s), S ep 28 (52 s), B ep 28 (50 s), S ep 64
   (46 s). The 24 Nov "hover" is still there in the scenes the policies get wrong.

## 7. Reading

- **The start trim fixes the stay-at-home failure on the arm.** It is the one result that does not depend on a single
  run per scene: 12 of 12 against 2 of 6, as the offline start check said beforehand.
- **With the start fixed, both trimmed policies place in most unseen scenes** (4 of 6 each). 24 Nov's 3 of 18 is not
  a baseline for this: other policies (all 90 demos, no trim), sun, free scenes.
- **S and B are not separated.** Both place 4 of 6; B's four were all first tries (11–13 s), one of S's came at 58.7 s
  after a stall. That is one run's difference with one run per scene. The held-out offline ranking (B: lower approach
  error, no premature opening, on two seeds) points the same way, but the arm does not show it.
- **The end trim's own effect is not visible here:** B differs from S by trimming the end hold, and both left home
  equally fast.
- **Per-episode offline opening recall does not predict the arm:** S and B score 0 on episode 3 and placed there.

## 8. Next (the user's choice)

1. **The shadows under sun:** the same six scenes with stripes on the table, S and B, to test what this session could
   not.
2. **Separate S and B:** more runs per scene, or the four held-out episodes not used today (7, 23, 69, 73).
3. **PiperStudio:** keep the chosen episode across a server switch, so the log records the scene that was set up.
4. **The failures are training-side:** ep 28's adjacent grape and plate and ep 70's plate by the base are thin in the
   demos; more demos there would be allowed under the zero-shot rule.
