# Status: where the thesis stands (Sun 18 Oct 2026, week W2 of `docs/thesis_plan.md`)

This is the handoff page: what is done, what is parked, what comes next. The plan itself is
`docs/thesis_plan.md`; every result has a record in `docs/runs/`. Update this page when the state
changes.

**Deadline:** submission ~1 Mar 2027. Experiments freeze at W17 (25 Jan).

**Right now:** back on the plan.
- **The Dobot demo was cancelled on Wed 7 Oct:** the robot is not available.
- **AgileX stays the Stage B robot.** The user is setting up the arms in the lab; hardware
  deployment comes later.
- **Piper F1, data and training half, done Sun 1 Nov** (`runs/2026-11-01_piper_f1_prep.md`).
  - 36 demos of one task: sausage into the pot, then the lid on the pot.
  - The front camera is masked where the operator and the leader arm are.
  - A 30-demo single-task policy is trained.
  - Offline, it memorises: held-out open-loop error 0.054 rad, no better than holding the last
    command (0.044); training 0.0036.
  - Follow-ups (record §8–§10): held-out error orders by demo count, not by session. Earlier
    stopping does not help. Image augmentation (`41cc359`) lowers it to 0.051, level with holding
    the state and still above holding the last command.
  - LIBERO calibration of that check (Mon 2 Nov, `runs/2026-11-02_libero_offline_calibration.md`).
    On the moka-pot task, 20, 30 and 40 demos reach 94, 86 and 94 % closed-loop. On held-out
    demos their error is 0.48 of standing still; Piper's is 1.15–1.22 of holding the last command.
    So Piper's shortfall is real by this check. It sits at tick 0 of each chunk, which may be the
    absolute-target offset (record reading 4).
  - **The policy server is built** (Mon 2 Nov): `flowcl/deploy/piper_server.py`, HTTP + JSON on
    port 8765. The contract with PiperStudio, and what PiperStudio must build, is in
    `docs/piper_deploy_spec.md`. Smoke run on held-out `v3` 10: round trip p50 50 ms; the served
    inputs are within 0.4/255 of the training cache's.
  - Next: the PiperStudio side (runner, replay through it, rollouts), built from that spec.
- **Piper milk task: six policies trained Mon 2 – Tue 3 Nov** (`runs/2026-11-02_piper_milk.md`).
  - **Data:** 40 demos (`pick_milk`), instruction "pick up the milk carton and put it on the blue
    plate", nothing held out.
  - **The six:** front camera masked or as recorded, crossed with no augmentation, daylight 0.3 (new:
    a colour temperature, a light ramp, then the colour jitter) or colour 0.3.
  - All six fit their demos: training-episode error 0.0029–0.0043 rad (holding the last command:
    0.031).
  - **The unmasked ones read the leader arm.** With the corner held still, their error rises 4–7×
    and their plans move 0.02–0.03 rad; the masked ones' plans move ≤ 0.004.
  - **Robot candidates: the masked three,** `results/{,piper_daylight/,piper_color/}single__piper__milk_plate__seed0`.
  - **On the robot, Tue 3 Nov, none picks up the milk** (`runs/2026-11-03_piper_milk_rollouts.md`;
    17 rollouts and 1 replay).
    - **The unmasked three never open the gripper.** The leader arm stands still on the robot.
    - **The masked three reach, open, descend, and close beside the carton or over the plate.**
    - **On a training layout (episode 20), the replayed demo picks and places;** two masked policies
      fail there.
    - **Ruled out:** the camera, the executor, the timing and the background.
    - **What happens:** the policy leaves the demo's path within 1.6–3.2 s and follows other demos.
      It then stalls in states no demo visited.
    - **The images move the plan:** at the start pose, today's frame gives 5–12× the error of the
      recorded frame of the same layout. Most likely the light: lamp at 17:00 against daylight.
      Augmentation cuts this about 2.5×.
    - **Next:** the same layout under the recording light decides it. Then demos in daylight, with
      more carton positions.
- **Piper grape task, Tue 3 Nov** (`runs/2026-11-03_piper_grape.md`). 40 demos recorded at midday,
  in the light the robot is tested in. Two policies: masked, and masked + the front camera cropped
  to the table the user outlined (new `image_crops` in the spec, carried into the checkpoint).
  - **On a training layout the masked policy follows its demo** (0.05–0.18 rad off its path for
    10 s), unlike the milk policies under other light. It grasped, then dropped the grape just
    short of the plate.
  - **Cropped, on free scenes: 4 of 21 put the grape on the plate,** and a fifth carried it there
    before being stopped. The masked policy managed 0 of 4.
  - **The misses:**
    - **aim:** 12 of 21 closed with the grape ahead of, beside or out of the fingers' view;
    - **grip:** after closing, the command goes back to about the grape's width (p50 48 mm),
      while the demos squeeze 12 mm past it (38.9 against 50.8 mm).
  - **Next, zero-shot (the user's choice):** more demos at more positions, with the leader gripper
    closed fully while carrying; then the cropped policy on all of them.
- **Piper resolution, Fri 6 Nov** (`runs/2026-11-06_piper_resolution.md`). The table crop at 224 and
  336 px on grape and milk; seven of eight policies trained, the last paused. 54 rollouts.
  - **Finer input aims better:** median miss 54 mm at 128 px, 31 at 224, 27 at 336. The grape reached
    the plate in 12 of 30 tries at 336 px (128 px: 4 of 23). No grape was placed from a miss above
    31 mm.
  - **Grip came right from the same demos:** the command after the close is p50 41 mm (128 px: 48;
    demos: 39), and all 14 placed runs squeezed by 9–16 mm.
  - **Serving costs almost nothing:** 55 ms round trip at 336 px. Training costs 4.2×.
  - **Two PiperStudio changes that keep runs zero-shot:** J5's own first-jump limit, 0.35 rad, without
    which 7 of 8 336 px runs were refused; and a rollout ends back at the start pose.
  - **The first-chunk jump and the restart at home came from compression** (below), not memorising.
  - **The motion is rougher than the demos at each replan** (a backward target step, not sampling
    noise, not compression). A blend at replans was smoother but not better. It breaks the zero-shot
    rule, so it stays off, as a tick box reported apart.
  - **Milk has not had a fair test:** it ran at midday, and it was recorded under the lamp.
  - **Replan interval** (`runs/2026-11-06_piper_replan.md`): 4 and 12 ticks are no better than 8, which
    stays. Execution tuning is finished.
  - **The scene decides:** the close follows the grape at slope 0.86 along the table's y, so aim is
    pulled sideways, about 27 mm short on one side. One zone places 10 of 14, the rest 9 of 34.
  - **Compression** (`runs/2026-11-06_piper_compression.md`): training frames are H.264 at crf 30, live
    frames sharper JPEGs. `serve_piper.py` now passes live frames through the same video by default
    (`--video-crf none` turns it off).
    - **On the arm, 13 runs:** first-chunk J5 +0.106 → +0.016 rad. Placed 5 of 13, miss 31 mm, roughness
      1.40: no change. In replay the restart at home nearly goes (11 of 12 plans → 2).
    - **Not settled:** only 4 runs were outside the working zone, and the error may have moved into x
      (+28 mm).
  - **Next:** runs in the weak zones, to decide whether new demos are needed at all; milk with the
    step, which may also explain 3 Nov; the paused queue (milk 336 daylight,
    `pkill -CONT -f 'scripts/[g]ate0.py'`).
- **Piper second table, Mon 23 – Tue 24 Nov** (`runs/2026-11-24_piper_new_grape_rollouts.md`; registry,
  embodiment and queue `7d7884a`, branch `piper-table2`).
  - **The data:** `new_grape`, 90 demos on a white table with the front camera in a new pose, recorded
    15:43–18:21.
  - **The policies:** embodiment `agilex_piper_table2_336` (mask, no crop, 336 px), every demo
    trained on, daylight 0.3. Plain seeds 0 and 1 trained; the trim run died at 11.7k (GPU timeout).
  - **First session on the arm, Tue 24 Nov afternoon: 3 of 18 placed** (all seed 1, 3 of its 15). The
    replay passes and every scene is near a demo's.
    - **Hover (9):** the arm stops part-way down with the gripper shut, at the demos' carry height,
      mostly over the plate.
    - **Miss (5):** the fingers close from higher up (106.7–120.4 mm against 99.5–105.4).
  - **Every rollout ran in afternoon sun with hard window stripes; no demo has them,** and the daylight
    augmentation only makes ramps. Over the whole table, the stripe measures do not overlap the
    demos'. Whether the light causes the failures is not shown.
  - **Second session, Tue 24 Nov evening (31 runs; not yet in a record):** in light like the demos', seed 1
    stayed at the start pose in 13 of 14 runs. Offline, the plan at home moves only when the input changes (sun
    stripes, a person in view, the arm creeping). The demos' 1.5 s hold at home before saving (2,088 frames, none
    with motion ahead) is the likely source.
  - **Queue `scripts/queue_piper_shadow.sh`, tmux `piper_shadow`, from Tue 24 Nov evening (~53 h):**
    - block A: 10 demos held out (`new_grape_*_ho10_336`); v0, trimmed at both ends, end only, start only, and
      both ends + shadow augmentation (`--augment-shadow`, real sun maps from `scripts/hw/piper_shadow_maps.py`);
    - checks: `piper_start_check.py` (departure on the logged at-home inputs, with PiperStudio's entry check)
      and `piper_relight_check.py` (opening, approach, MAE under the sun maps);
    - block B: `piper_choose_recipe.py` picks by the rule fixed beforehand, trained on all 90, seeds 0-1;
    - block C: ablations. Logs `results/piper_queue/shadow/`. Stop it before Friday's arm session.
    - Ended Thu 26 Nov 15:53, every run but v3_mix (Xid 8). Overnight `scripts/queue_piper_shadow_extra.sh` (b3_mix on
      all 90, seed 1 of b3_mix and s3_mix) ended Fri 27 Nov 09:40, "QUEUE DONE ok".
  - **Held-out arm session, Fri 27 Nov 13:51–14:50** (`runs/2026-11-27_piper_held_out_rollouts.md`). Three seed-0
    80-demo policies on six held-out scenes set up from the recorded first frames, one run each, diffuse light
    (no stripes, so the shadows were not tested):
    - **Start trim works on the arm:** S (start trim + shadows) and B (both trims + shadows) left home in 12 of 12
      runs; D (neither) stayed home in 4 of 6. The offline start check predicted it (1.00 against 0.06).
    - **Placed: S 4 of 6, B 4 of 6 (all first try), D 1 of 6.** S and B are not separated by one run per scene.
    - **b0 (B's recipe without shadows), added at 15:09 in the same light: placed 3 of 6, left home 6 of 6.** It differs
      from B on ep 64 only (five closes, never released); one run, so the shadows' effect without sun is not shown.
    - **Left:** ep 28 (grape and plate nearly touching; offline approach error 2–4× the other scenes'), one aim miss,
      ep 70's plate by the base (B held the grape at the rim without releasing), long stalls with the gripper shut.
    - Three runs are logged under episode 3: PiperStudio resets the scene picker at every server switch. The scenes
      were checked by image (`piper_session_diag.py scenes`).
- **The Dobot try-out is parked on branch `dobot-hw`** (not merged, never run on the robot).
  Its reusable pieces are listed in §3.
- **C1 and its follow-up are finished** (records `runs/2026-10-08_high_protection_three_seeds.md`,
  `runs/2026-10-09_high_protection_followup.md`).
- **Replay and EWC are implemented, and EWC's λ is decided.**
  - v1 bracket: `runs/2026-10-10_ewc_bracket.md`; Spatial 0/50 at every λ.
  - v2 follow-up (sampled Fisher, per-task normalization, wider grid, probe-loss tie-break),
    finished Sun 11 Oct 00:39: `runs/2026-10-11_ewc_bracket_v2.md`. Spatial is still 0/50 at every
    eligible λ.
  - It selects r = 100 (λ = 0.001227), now in `configs/method/ewc.yaml`.
- **LoRA is implemented and its lr is decided.** Merged LoRA keeps Spatial where EWC kept nothing.
  - v1 bracket: `runs/2026-10-11_lora_bracket.md`; rank 4, lr 3e-5, at the lr boundary.
  - v2 follow-up (post hoc; lr 1e-5 and 3e-5 at ranks 4–64), finished Mon 12 Oct 03:10:
    `runs/2026-10-12_lora_bracket_v2.md`.
  - **Selected: rank 16, lr 3e-5,** inside the grid, with Spatial 0.52 and Object 0.70 (GPM: 0.94
    and 0.78). It is in `configs/method/lora.yaml`.
  - At lr 1e-5, Object falls below the floor at every rank. Retention falls steeply with lr.
- **ConSFT is implemented and its τ_start and T are decided** (Sun 18 Oct 09:40;
  `runs/2026-10-18_consft_bracket.md`; notes `consft_notes.md`).
  - **The method:** Table 5's constants (the user's choice). T1 is seq_ft, and the schedule
    restarts at every task.
  - **The bracket:** 9 arms plus one edge extension. The z ladder was amended before any arm (to
    {10, 30, 100, 300}), because τ_end = 5 caps z = 1 and z = 3.
  - **Spatial is 0/50 at every arm, and no arm's Spatial probe loss is below seq_ft's (0.725).**
  - **Selected: z = 1000 at T = 30000** (τ_start 0.0163). It is the upper edge extension, chosen by
    the probe-loss tie-break: 0.730, Object 0.92. Flags: `all_eligible_spatial_zero`,
    `edge_extension`, `optimum_at_boundary`. It is in `configs/method/consft.yaml`.
  - **Readings, not demonstrations:**
    - under AdamW, a uniform floor trains like SFT (the Table 5 arm's first 1,700 steps);
    - the T = 2000 arms' abrupt weight rise coincides with a loss jump and a higher Spatial probe
      loss (note (c));
    - much of the weighting runs along flow time.
  - **The bundle carries Track 2 exclusion keys** for all 10 arms × 2 tasks. Step 4 must use them
    (§4, item 1).
- **Order decided on Sun 11 Oct:** the 4-task replay, EWC and LoRA runs are Track 2's held-out
  test, so they run after Track 2's freeze.
- **Track 2 is FROZEN (Thu 15 Oct 21:59; `runs/2026-10-15_track2_selection.md`).** Steps 1–3 are done:
  the manifest (449 cells), the measures M0–M3, and the selection (rule `1a6b6de`, written before any
  fitting; bundle `b397bc0a…`, verified).
  - M0 already ranks within tasks at C_a 0.82 and is miscalibrated across tasks by 8 pp.
  - Carried to the held-out test: `m2_position` (a: +0.008) and `m3_mean` (b: +0.016). Both
    development intervals include 0, and `m3_mean` fails root weighting, so "not shown" is the likely
    held-out result.
  - **The test set is complete** (Sun 15 Nov): the six 4-task baselines on seeds 0–2
    (`runs/2026-11-15_baselines_seeds1-2.md`).
  - **Next:** step 4, the held-out test. It must also exclude the bracket arms that came after the
    freeze (§4, item 1).
- **The projected-LoRA comparison is done** (Wed 14 Oct 07:44; `runs/2026-10-14_projected_lora_comparison.md`).
  No new mechanism is claimed. T1 -> T2 on seed 0, low-rank adapters confined to the complement of
  GPM's Task-1 memory:
  - all three families keep Spatial at 0.90–0.98 (plain LoRA: 0.52);
  - selected: `inflora` (16, 1e-4) 0.90 / 0.82 with 1.0M trainable parameters and about 35% less peak
    GPU memory; `pca_trained` (16, 1e-4) 0.98 / 0.82; `projected_lora` (64, 3e-4) 0.94 / 0.70;
    `gpm_projected_adam` 0.94 / 0.78;
  - unfreezing InfLoRA's A helped at every lr (8 of 8 comparisons equal or higher); every variant is
    10–25% slower per step.
  - The 4-task runs are reserved for Track 2's test set, so they wait for the freeze.
- **The recovery measurement study: GO** (Sat 17 Oct 04:34; `runs/2026-10-17_recovery_study.md`).
  An exploratory study outside the plan, with no training. The rule `88a8a72b…` was committed before
  any perturbed rollout and validated beforehand (false GO 0/200 under every null).
  - **Design:** 18,768 closed-loop continuations on the seq_ft seed-0 policies, one task each.
    Constant 8-step action offsets (position, rotation, gripper, isotropic) are applied at 6 phases.
  - **Result:** all 4 tasks pass Q1 (cell S20 0.65–0.94) and Q2 (beyond-magnitude gain 0.04–0.13
    nats) at 95%; 3 of 4 also pass at 99%. The negative control is clean (0 / 3,246 no-op
    mismatches or injuries).
  - **The structure:** position offsets harm, while rotation is nearly inert per demo SD (a
    scaling-dependent contrast). A gripper flip is fatal only while an object is carried (0.79–0.91).
  - **GO authorizes only the integrated-action Jacobian and capacity feasibility study,** the input to
    D2's angle B (Uᵀ ΔW M = 0). It does not authorize a method.
- **E1-B, that feasibility study: `fail`** (Sat 17 Oct 18:48; `runs/2026-10-17_twosided_angleb.md`).
  It reran E1 with the integrated executed prefix, weighted by the recovery study's injury rates. There
  was no training; the rule `17663228…` was committed before production.
  - **Result:** the trunk two-sided fraction at T4's start is 0.861 (GPM 0.933, threshold ≤ 0.50).
    That is E1's number to three decimals. Uniform, shuffled, phase-pooled and gripper-free weightings
    all give 0.850–0.870; the label is `near_full_rank`.
  - **Reading:** the trunk's output-side occupancy comes from state variety, not from which outputs
    are measured.
  - **Pre-written consequence:** the recovery line stops (a scheduling decision), and **D2 = the
    minimum thesis**. The recovery GO stands as a measured finding.
  - **Disk:** `results/twosided_angleb/` holds 48 GB. About 42 GB are Grams kept for E2-B, which will
    not run; deleting them is the user's call.

## 1. Plan progress

| Item | State | Record / note |
|---|---|---|
| **A1** stage-boundary resume | done (confirmatory test passes) | `runs/2026-10-02_resume_check.md` |
| **A2** `seq_hetero_t8`, Gate 0 T6–T8 | done (T6 microwave, T7, T8 pudding → drawer) | `runs/2026-10-02_gate0_t8*.md` |
| **A3** seen-task-only evaluation (`eval_unseen: false`) | done (`f2a5119`, Mon 16 Nov): `--no-eval-unseen`; the GPU smoke matches the default mode in every checkpoint and shared rollout | §5; `runs/2026-11-16_measurements_verification.md` |
| **A4** ε schedule for GPM (`eps_later`, `gpm_projected_adam_e99`) | done | `configs/method/gpm_e99.yaml`; CPU test: stages 0–1 and T1 memory bitwise equal to plain |
| **A5** rollout-level paired bootstrap for ACC / NBT / AUC, per-seed tables | done (`22bd039`, Sun 15 Nov) | §5 item 3 |
| **A6** metric naming (ACC, F_first) in tables and README | done (`5b870d4`, Sun 15 Nov) | §5 item 4 |
| **A7** close SGP | done | `runs/2026-10-01_t5_sweep.md` |
| **B** baselines: replay, EWC, LoRA, ConSFT | **replay and EWC implemented** (`2c2b827`, `aaedf24`, `5f1d124`); smoke passed (stage-0 identity, resume equality). **EWC v1 bracket** (empirical Fisher, r ≤ 100): Spatial 0.00 at every λ. **EWC v2 bracket** (sampled Fisher, normalized, r ≤ 10000): Spatial 0.00 at every eligible λ (r ≤ 100); 2/50 only at r ≥ 1000, where Object falls to 0.44 / 0.36. At equal r, v1 and v2 trace the same trade-off. **Selected: r = 100, λ = 0.001227** (probe-loss tie-break, `all_eligible_spatial_zero`); Object 0.72, Spatial probe loss 0.165 (GPM 0.042). **LoRA** (`75fb0e1`) **v1 bracket:** rank {4,16,64} × lr {1e-4,3e-4,1e-3} plus (1, 1e-4) and (4, 3e-5); Spatial falls with lr at every rank; rank 4, lr 3e-5 at the boundary. **v2** (`620727b`, post hoc): lr 1e-5 and 3e-5 at ranks 4–64; at 1e-5 Object < floor at every rank. **Selected rank 16, lr 3e-5:** Spatial 0.52, Object 0.70; the top eligible arms are within noise (0.52 / 0.48 / 0.46 / 0.44). **ConSFT** (`a8250ce`; Table 5 constants) **bracket:** z ∈ {10, 30, 100, 300} × T ∈ {2000, 30000} plus Table 5's point (amended before any arm, `288c343`), calibrated at q50 0.653 (`9a90b6f`): Spatial 0.00 at every arm, Spatial probe loss never below seq_ft's (best 0.730). **Selected z = 1000, T = 30000** (τ_start 0.0163; upper edge extension, `all_eligible_spatial_zero`, `optimum_at_boundary`); Object 0.92; in `configs/method/consft.yaml` (`5b4ae22`). **4-task baselines, seed 0, done** (`27c7e88`, Thu 12 Nov): every identity check passed bitwise (stage 0 against seq_ft, stage 1 against each selected arm), and all six strict reports passed. Final rows: replay 96 / 42 / 90 / 96 (ACC 0.81, not exemplar-free); projected_lora 90 / 4 / 88 / 96; inflora 92 / 28 / 78 / 70; EWC, LoRA and ConSFT 0 on every earlier task. **Seeds 1–2 done** (`39ce581`, Sun 15 Nov): all 12 strict reports and stage-0 identities passed. Mean final rows over seeds 0–2: replay 95 / 69 / 95 / 95 (seed 0's Object 42 was its low seed; 82 on seeds 1–2); inflora 96 / 21 / 72 / 73; projected_lora 87 / 7 / 71 / 87; EWC, LoRA and ConSFT keep no earlier task on any seed. The Track 2 test set is complete | `runs/2026-10-10_ewc_bracket.md`, `runs/2026-10-11_ewc_bracket_v2.md`, `runs/2026-10-11_lora_bracket.md`, `runs/2026-10-12_lora_bracket_v2.md`, `runs/2026-10-18_consft_bracket.md`, `runs/2026-11-12_baselines_seed0.md`, `runs/2026-11-15_baselines_seeds1-2.md` |
| **C1** high-protection control (e99 × 3 seeds) | done: **`unresolved`** ×3 (replication label `mixed`, a rule gap). At matched T2 protection (as predicted, 91/91 ranks) a flat 0.99 reproduces the Object gain over plain on every seed (G +0.70 / +0.62 / +0.48), indistinguishable from adaptive (D +0.04 / −0.02 / −0.08), non-inferiority at 10 pp not shown at 50 rollouts. Descriptive: adaptive keeps more final Object (−0.14 / −0.16 / −0.22 for e99) at ~15 pp more capacity. **Follow-up** (3 repetitions of the 50 initial states, second-stage, decided after C1): `protection_suffices` / `unresolved` / `unresolved` + significant `adaptive_advantage` (D +0.03 / −0.05 / −0.10), replication `mixed`: the allocation's effect at this cell is small and seed-dependent | `runs/2026-10-08_high_protection_seed0.md`, `runs/2026-10-08_high_protection_three_seeds.md`, `runs/2026-10-09_high_protection_followup.md` |
| **C1-b** allocation rules at matched capacity (seed 0, env B, pre-E3) | done: room **`allocation_hurts`** (ΔACC −4.0 pp [−7.5, −0.7]); novelty −1.0 [−4.0, +2.0] and size +0.8 [−2.3, +4.0] `unresolved`. No arm goes to seeds 1–2; E3 keeps its flat-0.99 arm and takes no new one. T4 rerun check amended to a McNemar tolerance (env-B render noise) before the outcome was read | `runs/2026-11-22_allocation_rules_envb_seed0.md` |
| **C2** reverse order × 3 seeds | done: `inconclusive` / `task` / `both` (no three-seed recurrence; Object lost across Spatial every seed; adaptive trade replicates) | `runs/2026-10-0{3,4}_reverse_order_seed*.md` |
| **D** Track 2 (offline predictors) | **step 2 done: the measures M0–M3** (`d213ddb`/`99e8083`): 449 cells, bundle `5be958cc…`, instrument checks 281/281 within 2.1e-5; descriptive only (`runs/2026-10-13_track2_measures.md`). **Step 1 done: the development manifest** (`2ec4c03`, merged): 449 cells (checkpoint × trained task, 50 rollouts) under 24 roots; seed 0's shared stage 0 holds 214; 63 bitwise duplicates collapsed (one disagrees by 2/50 rollouts); 114 depth-1 / 147 first-divergence branches (branch definition changed in `thesis_plan.md` §D, the literal one kept as sensitivity); held-out key set published. **Next: step 3's rule**, then validation and the freeze | freeze target Wed 28 Oct, **hard deadline Fri 30 Oct**; `runs/2026-10-12_track2_manifest.md` |
| **E1** two-sided go/no-go | done: **`fail`** (`near_full_rank`; trunk 0.861 vs GPM 0.933) | `runs/2026-10-02_twosided_gonogo.md` |
| **E1-B** angle-B capacity gate (recovery-weighted executed prefix) | done: **`fail`** (`near_full_rank`; trunk 0.861 vs GPM 0.933, the same as E1 to three decimals; every weighting 0.850–0.870) | `runs/2026-10-17_twosided_angleb.md` |
| **E2** pilots | **not run:** angle A is off (E1), angle B is off (E1-B). **D2 resolves to the minimum thesis** (pre-written in E1-B's rule) | `runs/2026-10-02_twosided_gonogo.md`, `runs/2026-10-17_twosided_angleb.md` |
| **F0** hardware readiness | **log format inspected on AgileX Piper** (LeRobot v3.0, 15 fps, absolute joint targets, measured gripper); access, reset time and cycle time not measured yet. The Dobot try-out was cancelled before any robot time | `runs/2026-11-01_piper_f1_prep.md`; branch `dobot-hw`: `runs/2026-10-06_dobot_hw_prep.md` |
| **F1** engineering + proof | **data and training half done:** adapter, masked front camera, 30-demo single-task policy (`9ec29d9`); offline held-out error no better than holding still; augmentation (`41cc359`) brings it level with holding the state, not below holding the last command; LIBERO policies that succeed closed-loop are at 0.48 of standing still on the same check, so the shortfall is real (`b89fd2e`). **Policy server built** (`flowcl/deploy/piper_server.py`; spec `docs/piper_deploy_spec.md`). **Not started:** replay of 3 demos, 10 rollouts. **Milk task** (2 Nov, 40 demos): six policies, mask × {none, daylight, colour}; the unmasked ones read the leader arm (corner probe), so the masked three go to the robot. **PiperStudio runner built** (PiperStudio `a9e82e7`). **Milk rollouts (3 Nov): 0 picked.** The unmasked never open the gripper. The masked follow other demos' paths and stall, even on a training layout; the same demo replayed through the executor succeeds (all replay tolerances pass). The images, most likely the light, move the plan 5–12×. Next: the recording-light test, then daylight demos. **Grape (3 Nov, recorded in the test light):** masked and masked + table crop; the cropped policy placed the grape in 4 of 21 free-scene tries. The misses are aim and a loose grip. Next: more demos with a firm grip | `runs/2026-11-01_piper_f1_prep.md`; `runs/2026-11-02_libero_offline_calibration.md`; `runs/2026-11-02_piper_milk.md`; `runs/2026-11-03_piper_milk_rollouts.md`; `runs/2026-11-03_piper_grape.md` |
| **D1** supervisor: minimum thesis, claim form, hardware scope, proof-gate thresholds | not recorded here | |

## 2. Side line, outside the plan: similar tasks and language

Three studies, 4–6 Oct, and a pilot on Tue 17 Nov with follow-ups Wed 18 – Fri 20 Nov (item 4):
1. **`runs/2026-10-04_similar_tasks.md`:** similar tasks overlap more with A's GPM memory, by
   modest margins. The instruction barely reaches a single-task policy's output.
2. **`runs/2026-10-05_similar_cl.md`:** A → similar B. Every arm, GPM included, learns B and
   loses A completely (0.94 → 0.00). On a diverse B, GPM keeps A at 0.92.
3. **`runs/2026-10-06_language_cl.md`:** a base trained jointly on A + cabinet top **does select
   by instruction**.
   - **Swap drops:** 0.90 and 1.00. **Requested goals reached:** 1.00 and 0.94.
   - So the language pathway works; the earlier blindness came from single-task stages.
   - From that base, GPM keeps the cabinet task (0.98) but still loses A (plate) to stove
     (0.90 → 0.00). The base never learned the plate/stove contrast. Motion similarity is
     confounded with it.
4. **`runs/2026-11-17_instruction_negatives.md` (exploratory, seed 0, summary `valid`):**
   instruction negatives at stage A, meaning wrong instructions from a fixed bank trained to a
   hold target. With them, **GPM keeps plate through stove at 0.70** [0.56, 0.82]; historical GPM
   kept 0.00.
   - **Projection is needed.** The same negatives under allowlist_ft keep 0.00, and stove is
     learned equally well (0.96 / 0.96).
   - **Labels:** `effect_present` and `projection_contributes` (+0.70 each, the first against a
     historical control). `retains` is not shown (forgetting 0.24).
   - **The base abstains under untrained instructions.** Plate's goal is reached 0/50 under
     "stove" and "cabinet top".
   - **The plate/stove contrast is large and mostly inside GPM's memory.** Its chunk divergence
     is 0.85; the A-only base's was 0.0004.
   - **After stove, GPM still selects by instruction both ways** (plate 0.78, stove 0.98).
   - **Cost:** +57% at stage A.
   - **The follow-up on cabinet top and milk does not generalise the result**
     (`runs/2026-11-18_instruction_negatives_curricula.md`, Wed 18 Nov 01:52–06:03, seed 0, summary
     `valid`). It was resumed at stage 1 from the same base.
     - **Cabinet top:** GPM + negatives keeps plate at **0.04** (historical 0.00, +0.04 [0.00, +0.10]).
       Every retention label is `not_shown`; cabinet top is learned at 0.96.
       - **What survives:** the policy still avoids the cabinet under the plate instruction (0 of
         50), and its chunk contrast (0.31) matches stove's.
       - **What is lost:** the plate behaviour itself.
     - **Milk:** neutral. Plate is 0.92, as historical GPM (`no_cost_beyond_10pp`); milk is 0.82
       against 0.76. `milk_no_harm` is `not_shown`: milk's CI low is −0.1005, at the −0.10 bound.
     - **So** the stove effect depended on the B task: 1 of 2 similar B tasks.
     - **The base is not why** (record §3.7, a probe added afterwards). It separates plate from
       cabinet top as strongly as from stove (chunk contrast 0.91 against 0.85), and as far inside
       GPM's memory. The loss arises during stage B.
   - **Seeds 1–2: stove replicates, cabinet top is `mixed`**
     (`runs/2026-11-19_instruction_negatives_seeds.md`, Thu 19 Nov 01:38–18:50 with a reboot,
     summary `valid`). Each seed has a new negatives base, and every control is fresh plain GPM.
     - **Stove:** plate after stove 0.92 and 0.88, against 0.00; `effect_present` and `retains` at
       both seeds. Over seeds 0–2: 0.70 / 0.92 / 0.88, so seed 0 was the low seed.
     - **Cabinet top:** 0.10 (+0.10 [+0.02, +0.18], `not_shown`) and 0.28 (+0.28 [+0.16, +0.42],
       `effect_present`), against 0.00. With seed 0's 0.04, the gain is positive at every seed and no
       rollout favours the control, but it stays far from retention.
     - **Seed 0's loss gap does not replicate:** A's probe loss after cabinet top equals stove's at
       seeds 1–2. What differs is the release: under the plate instruction in B's scenes the arm
       reopens the gripper in 38–49 of 50 rollouts after stove, and 13–31 after cabinet top.
   - **A second pair, wine bottle → cabinet top then → rack (seeds 0–2): `not_assessed` by the rule,
     but GPM + negatives keeps A at 0.88 / 0.94 / 0.16 against 0.02–0.04 for plain GPM**
     (`runs/2026-11-20_instruction_negatives_wine.md`, Thu 19 Nov 19:14 → Fri 20 Nov 10:35, summary
     `valid`). Paired neg − ctrl: +0.86, +0.90, +0.14, each with its CI low above 0.
     - **Why the gate failed:** under "put the wine bottle on the rack" the base still does A (0.90 /
       0.92 / 0.68), and R0[A] is 0.78 and 0.52 at seeds 0 and 2. It holds under every other wrong
       instruction.
     - **The contrast grows during B,** from 0.03–0.10 to 0.11–0.28. On the bowl pair it started at
       0.73–0.85 and was worn down. So the abstention the gate tests was not needed here.
     - **Costs:** A is learned less well with the negatives at seeds 0 and 2 (−0.20, −0.42), and B is
       a little lower at every seed (−0.04 to −0.14).
   - Follow-ups are parked in the records' §6: a gate on instruction sensitivity for the next pair,
     the text-encoder distances, A's acquisition under the negatives, videos.

**Parked follow-ups** (not decided; none is in the plan):
- allowlist_ft from the joint base (~2.7 h);
- a broader LIBERO-Goal base;
- the diverse-base control;
- a joint A + S2 + S control.

**The in-plan remedy is replay:** add A → stove to replay's runs once replay exists (~2 h).

**Lesson for hardware task choice (F2):** if two hardware tasks share a setup and differ only by
instruction, single-task stages give the policy no reason to read it, and every method will appear
to forget. Choose tasks that differ visually, or plan for joint/replay training.

## 3. The hardware gate (F0/F1 on AgileX): what it must establish

The plan's gate (`thesis_plan.md`, Workstream F, F1):
1. **The data format:**
   - cameras, proprioception;
   - action space (joint or end-effector, absolute or delta, both grippers);
   - control rate, timestamps.

   This defines `configs/embodiment/<robot>.yaml` (D_action not padded) and the dataset adapter.
2. **Open-loop replay of 3 recorded demos** reproduces their trajectories (action convention,
   frame, timing).
3. **The single-task proof:**
   - ~15 demos of one task, trained from scratch (the LIBERO policy does not transfer to another
     embodiment);
   - 10 closed-loop rollouts on fixed initial conditions;
   - **≥ 6/10 successes**, episode cycle (execution, reset, annotation) **≤ 4 min**.
4. **Record:** what was learned (format, rates, measured cycle time, robot access including
   W12–W13) goes in `docs/runs/` and back into this page.

**Reusable from the parked Dobot try-out** (branch `dobot-hw`; record
`docs/runs/2026-10-06_dobot_hw_prep.md` on that branch):
- **Code:**
  - `TaskRef` suite dispatch for non-LIBERO tasks;
  - a LeRobot v2.1 adapter with an HDF5 cache;
  - `stats_std_floor` in the embodiment spec;
  - `gate0 --no-eval`;
  - a websocket policy server with a status page;
  - a robot-side runner (probe without motion, stop test, replay with tracking metrics, operator
    rollout loop with cycle time; safety checks on every chunk);
  - an offline open-loop check.
- **Lessons for AgileX F1:**
  - Make the training state match exactly what the robot can send live (the Dobot's gripper state
    was the last *command*).
  - Frozen Task-1 stats can put later tasks' joints many stds out on a joint-position embodiment.
  - A `--no-eval` `run_continual` writes no `result.json`, so non-seq_ft methods need
    `--no-t1-check`.
  - h5py step-sliced reads (`[::k]`) are very slow; read contiguously, then subsample.
  - Measure demo joint speed against the robot's per-tick clip before choosing it.
- **Disk:** `datasets/dobot/` (69 GB) and the Dobot runs in `results/` (about 6.7 GB) can be
  deleted if space is needed.

## 4. Next, when back on the plan (priority order)

0. **Done: the 4-task baseline queue, seeds 0–2** (Track 2's held-out test set).
   - Seed 0 Thu 12 Nov (`runs/2026-11-12_baselines_seed0.md`); seeds 1–2 Sun 15 Nov
     (`runs/2026-11-15_baselines_seeds1-2.md`). Every identity check and strict report passed.
   - **Watch for reboots** on any later queue: gamma reboots itself after 30 min offline
     (`net-watchdog`). A killed queue resumes with `--from-step K` once the partial run directory has
     been moved aside.
   - The recovery line is closed (E1-B `fail`), and D2 resolves to the minimum thesis.
1. **Track 2 (D), step 4: post-freeze bracket arms must be excluded.**
   - **The gap:** the frozen rule excludes test cells whose (fingerprint, task) is in the published
     `development_keys`. Those come from the 12 Oct manifest, so they cover the EWC and LoRA bracket
     arms but not the brackets that ran after it.
   - **ConSFT:** step 4 must exclude the union of the development keys and ConSFT's bundle's
     `track2_exclusion_keys`. That is 20 keys, recomputed by `verify` from the checkpoints.
   - **`projected_lora` (13 arms) and `inflora` (12 arms):** their 14 Oct bundles carry no keys.
     Step 4 must derive them from those bundles' arm checkpoints (`ewc.tensor_fingerprint`, both
     trained tasks).
   - **The acceptance check, for each method:** a production stage-1 cell identity-checked against
     its selected arm is actually excluded by the union.
   - The frozen rule and its bundle are not modified.
2. **Thesis wording from C1:** raising T2 protection removes the Object forgetting; the new-energy
   allocation adds at most a small, seed-dependent amount at that cell (significant on one seed of
   three). Its consistent edge is final Object retention at about 15 pp more capacity (E3's frontier).
   - **E3 gets a flat-0.99 arm** (decided Mon 16 Nov; `thesis_plan.md`, E3). A literature check
     found no paper with the new-energy rule itself, but raising ε for later tasks is prior
     practice (GPM's CIFAR-100 schedule, InfLoRA's linear one). So the rule's claim rests on
     beating a flat threshold, which C1 did not show.
   - The arm resumes C1's e99 runs to T8, about 6 h per seed. It goes into `main_eval.yaml`, which
     is not written yet.
   - **The matched-capacity test (allocation pilot, 22 Nov, seed 0, env B) did not show it
     either.**
     - The new-energy rule at e99's exact capacity: −1.0 pp [−4.0, +2.0].
     - The room-aware rule, which moves protection from the trunk to the decoder: −4.0 pp
       [−7.5, −0.7].
     - The wording that holds is "a flat 0.99 threshold is not beaten by the tested allocation
       rules at matched capacity, and moving protection off the trunk hurts" (exploratory, one
       seed).
3. **Track 2 (D):** frozen on Thu 15 Oct. Angle B no longer depends on it (E1-B closed angle B), so
   the held-out test is reported for its own sake. The 4-task baseline runs (item 0) come first.
4. **A3, A5, A6:** done (Sun 15 – Mon 16 Nov, §5), with the per-step loss, peak GPU memory and the
   optional probe loss that E3 needs.
5. **Hardware:** F1 engineering on AgileX continues in parallel to the GPU work.
   - **Done:** the Piper adapter, mask and first policy (`runs/2026-11-01_piper_f1_prep.md`).
   - **Next:** the policy server and robot runner, reusing the `dobot-hw` pieces (§3), then the
     replay of 3 demos and the rollouts.
   - **Keep the front camera's pose fixed** until the proof is done.

6. **Pre-E3 plan (Mon 16 – Tue 17 Nov): the three code problems E3 would hit are fixed.**
   - **A1** (`b997c6f`): seq_ft seed 0, whose config predates the namespace field, resumes; per-stage
     records survive an interruption; `run_continual.py --check-resume-only`.
   - **A2** (`4b78e16`): `sequence_report` reads resumed, seen-only and 8-task runs. 4-task reports
     are unchanged.
   - **A3** (`40366b3`): f = 0.7 runs as a 4-task `seq_hetero` run first (`scripts/queue_ne70.sh`), so
     it shares the seed namespace of the arms E3 resumes (`thesis_plan.md`, E3).
   - **B:** the 8-task resume smoke passed on Tue 17 Nov (`runs/2026-11-17_t8_resume_smoke.md`).
   - **The instruction-negatives pilot (§2, item 4)** used the GPU after it, 03:22–08:55 on Tue 17
     Nov. The GPU has been free since.
   - **The f = 0.7 queue is done** (Tue 17 Nov 10:56 → Wed 18 Nov 01:16,
     `runs/2026-11-18_ne70_4task.md`). Every identity check and strict report passed on seeds 0–2.
     - Mean ACC 77.5, between plain GPM (62.7) and f = 0.9 (86.7).
     - Below the flat 0.99 (84.2), at about the same trunk use (86% against 89%).
   - **The instruction-negatives follow-up (§2, item 4)** used the GPU Wed 18 Nov 01:52–06:03.
   - **Instruction negatives at seeds 1–2 (§2, item 4)** used the GPU Thu 19 Nov 01:38–14:52 and,
     after a reboot, 16:12–18:50.
   - **The wine pair (§2, item 4)** used the GPU Thu 19 Nov 18:59 (smoke) → Fri 20 Nov 10:35.
   - **Machine move, Fri 20 – Sat 21 Nov:** the SSD moved from gamma to ThinkStation-2 (env B, RTX
     PRO 4000 Blackwell, torch 2.9.1+cu128; `e40060e`).
     - Env-A runs are never paired with env-B runs: training is bitwise deterministic within each
       environment, not across them.
     - The 4-task seed-0 references (seq_ft, plain GPM, e99) were retrained in `results/envB/`
       (`7fe311d`).
   - **Allocation-rules pilot, seed 0, env B: done** (Sun 22 Nov 03:07 → 18:16;
     `runs/2026-11-22_allocation_rules_envb_seed0.md`). It tests three GPM allocation rules against
     flat 0.99 at e99's own capacity; the rule was amended for env B before training.
     - **Outcome** (4 tasks × 50 states × 3 repetitions; e99 ACC 90.2):
       - novelty −1.0 pp [−4.0, +2.0], `unresolved`;
       - size +0.8 [−2.3, +4.0], `unresolved`;
       - **room −4.0 [−7.5, −0.7], `allocation_hurts`.** It moves protection from the trunk to the
         decoder (trunk 67 % against 89 %) and loses on every task.
     - **Consequences, as registered:** no arm goes to seeds 1–2, and E3 takes no new arm (it keeps
       its flat-0.99 arm). The tested rules do not beat a flat 0.99 at matched capacity (seed 0,
       exploratory).
     - **Gates:** every training-side gate passes on all three arms. The bitwise rep-0 rerun of T4
       failed for every run, e99 included.
       - **Cause:** the wrist camera's multisampled rendering of the stove burner varies by 1–2
         levels on a few pixels between renders of the same state on this GPU (multisampling off:
         none in 2,665 states). Physics and the policy are deterministic.
       - **The fix to the gate:** the user chose to amend the T4 check (amendment 2, `132d153`: seeds
         equal plus an exact McNemar test). It was committed before any outcome was read, and all
         four reruns pass it.
     - **For E3 and later env-B work:** a bitwise rerun check fails on any task whose wrist camera
       sees such geometry. Deterministic rendering (multisampling off) changes the images, and so
       the protocol.
     - **A side finding, fixed** (`b7013ba`): a standalone evaluation encoded T1's instruction in
       fp32, while the run had encoded it under AMP, so T1 never reproduced a run's own cell, on
       gamma either. `evaluate.py --runner-text-cache` fixes it.
   - **Next:** Track 2 step 4 (item 1), then `main_eval.yaml` and E3.

## 5. To do: thesis figures and what they need measured (noted Thu 12 Nov)

Moved here from `.scratch/` on Sun 15 Nov, after the seed 1–2 queue. Items 1 and 2 below, and A3
(§4, item 4), had to land before the 8-task runs (E3). **State on Mon 16 Nov:** items 1–5 and A3
are done; item 6 waits on the PiperStudio form; item 7 is drafted, to send. None of them blocks
Track 2 step 4.

**What "done" means:** code, tests and a GPU smoke pass, committed. The default 4-task protocol is
unchanged: with every new switch off, a smoke of all 8 methods (seed 2, 200 steps, 1 episode)
matches the smoke made before these changes in every checkpoint tensor, method artifact and
rollout (`scripts/compare_smokes.py`, added in `bc57779`). Record:
`runs/2026-11-16_measurements_verification.md`.

### 5.1 Measurement gaps (priority order)

1. **Per-step training loss. State: done (`f2a5119`).**
   - Each stage writes `train/stage{K}_losses.json`, recorded as `loss_file` (path and SHA-256) in
     `result.json`. A resumed run copies the files of the stages it imports.
   - Three series per step, from the outputs the step already has:
     - `loss`, the optimised loss;
     - `batch_loss`, the flow loss before `modify_loss` (it differs for EWC and ConSFT);
     - `task_loss`, the flow loss of the stage's own samples. For replay this is the new-task
       curve; for the other methods it equals `batch_loss` up to rounding.
2. **Peak GPU memory. State: done (`f2a5119`).**
   - Allocated and reserved bytes, per stage, for two windows: training (from `on_task_start` to
     `on_task_end`) and evaluation. They are in `stages[k].peak_gpu`, and the maxima in
     `systems.peak_gpu`.
   - projected_lora and InfLoRA reset the peak counter when they compute their design; that peak is
     folded into the training window, and also kept in `timings` as `design_peak_gpu_bytes`.
   - It covers the PyTorch allocator only, not EGL rendering or the CUDA context. Stages that were
     not measured (imported, or older runs) are `null`, never 0.
   - `tables.systems_table` shows both windows as its last two columns.
3. **A5, the rollout-level paired bootstrap. State: done (`22bd039`).**
   - `flowcl/analysis/rollout_bootstrap.py` covers ACC, NBT, AUC and F_first. Each replicate
     resamples episodes once per task column and applies the draw to every stage of that task.
     Paired differences between runs use the same draw.
   - `tables.per_seed_table` (one row per method and seed) is the thesis default. `main_table`
     shows NBT and AUC by seed instead of a seed-level bootstrap.
4. **A6, metric naming. State: done (`5b870d4`).** Final average success is shown as **ACC**;
   `result.json` keeps the key `F_1`. **F_first** = R[0][0] − R[1][0]
   (`metrics.first_task_forgetting`). Both are defined in `docs/implementation_notes.md` §8.2.
5. **Optional: old-task probe loss during training. State: done (`f2a5119`), off by default.**
   - `run_continual.py --probe-every N` writes `train/stage{K}_probes.json`: the flow loss of every
     earlier task, at the stage start and every N steps.
   - Each task's probe batch is fixed (chunks, flow times and noise from
     `derive_seed(namespace, "probe::<task>", 0)`). A resumed run rebuilds it.
   - The guard smoke (probes every 50 steps, 5 methods) gives checkpoints and rollouts identical to
     the probes-off smoke.
6. **Hardware rollout record, before F3 / F4. State: spec and reader done (`5e22984`); the
   PiperStudio form waits on its uncommitted frontend edits.**
   - `rollout_record/v1` (`docs/piper_deploy_spec.md` §6.4 and §7): success, an outcome from a
     fixed failure taxonomy, miss distance, light, a note, and a start photo saved at Go. The cycle
     time is already logged.
   - `flowcl/deploy/rollout_records.py` reads and validates the logs. Logs from before 15 Nov are
     read as `legacy`, with nothing filled in.
   - Still to do: the form in `~/piper_studio` (`PolicyPanel.tsx`, `styles.css`, `types.ts` have
     uncommitted edits) and the start photo at Go.
7. **Ask the supervisor (D1):** the plan has no jointly trained multitask upper bound.
   **State: drafted (Sun 15 Nov), to send.** Draft:

   > The sequential results (4 tasks, 3 seeds; next, 8 tasks) are compared with sequential
   > fine-tuning and the baselines, but there is no jointly trained multitask reference: one
   > policy trained on all tasks at once and evaluated on each. It would show how far each method
   > is from the ceiling of this architecture and data. The code supports it already
   > (`run_continual.py --joint-reference`, the same steps per task as one sequential stage). The
   > cost is about 3 GPU-hours per seed for the 4 tasks and about 6 h per seed for the 8. Should
   > the thesis include it, for the 4-task sequence, the 8-task sequence, or neither?

**A3, seen-only evaluation (§4, item 4). State: done (`f2a5119`).**
- `run_continual.py --no-eval-unseen` evaluates, after each stage, only the tasks trained so far
  (co-trained tasks included). The default still evaluates every task.
- `config.yaml` and `result.json` record `eval_unseen: false` only in that mode.
- After the stage-0 evaluation it encodes the remaining instructions in fp32, as the default
  evaluation does, so training is unchanged. The GPU smoke (seq_ft, GPM, replay) matches the
  default mode in every checkpoint and in every rollout both modes ran.
- It refuses `--single-task-baseline`, since FWT needs the unseen tasks.

### 5.2 Figures, one per claim

`scripts/make_figures.py` writes them to `docs/assets/figures/` (Sun 15 Nov, `8be8f33`).
`figures.json` records each figure's state and sources.

| Chapter | Figure | What it shows | State |
|---|---|---|---|
| Setup | Task images of the curriculum | What is learned, in which order | made, `task_images.png` |
| Setup | Where the projection sits in the policy | | manual |
| Characterization | Retention grid per seed; forgetting curves per task and seed | Sequential fine-tuning forgets everything | made, `retention_grid_seed*.png`, `forgetting_curves.png` |
| Characterization / Track 2 | Probe loss against closed-loop success, by curriculum task | The same loss (0.09–0.10) gives 88–90 %, 56 % and 2 % | made, `track2_loss_vs_success.png` |
| Adaptive GPM | Object retention, plain against adaptive, per seed | +66 / +64 / +56 pp after Goal | made, `adaptive_vs_plain_object.png` |
| Controls | e99 and reverse order as per-seed paired differences | Allocation, or only more protection | made, `controls.png` |
| Capacity | Protected fraction of the trunk after T1–T4 | Protection uses up capacity | made, `protected_fraction_t1_t4.png` |
| Capacity | Protected fraction after T5–T8; overall success against capacity for plain, f 0.7, f 0.9 and flat 0.99 | | waiting for data (E3) |
| Baselines | Final success per task and method | Replay marked as not exemplar-free | made, `baselines_final.png` |
| Negative results | Two-sided fraction 0.861 against GPM 0.933 and the 0.50 threshold; recovery injury rate by offset type and phase | Why the two-sided line stopped | made, `twosided_fraction.png`, `recovery_injury.png` |
| Hardware | Setup photo with mask and crop | | made (existing `docs/assets/piper_*.jpg`) |
| Hardware | Miss distance by resolution (128 / 224 / 336 px); grasp positions by zone | Transfer evidence, not method ranking | made, `piper_miss_by_resolution.png`, `piper_zones.png` |
| Hardware | Success counts with intervals | | waiting for data (F3 / F4, `rollout_record/v1`) |

**On every plot:** each seed shown separately (three seeds give no seed-level interval), seed 0
marked as the development seed, and the number of rollouts stated.
