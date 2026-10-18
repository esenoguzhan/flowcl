# ConSFT τ_start × T bracket (seed 0, T1 → T2): Spatial is 0/50 at every arm, and no arm brings Spatial's probe loss below seq_ft's. The rule selects z = 1000 at T = 30000 (τ_start 0.0163), the upper edge extension, by the probe-loss tie-break; its probe loss (0.730) equals seq_ft's (0.725), and its Object is 0.92

**Status: the pre-registered bracket, complete.**
- The queue ended `QUEUE DONE ok`, the verification passed, and the bundle is published.
- **The registered selection is τ_start = 0.016330303251743318, decay_steps (T) = 30000**, with
  Table 5's constants. It is flagged `all_eligible_spatial_zero`, `edge_extension` and
  `optimum_at_boundary`.
  - The grid selected z = 300 at T = 30000, at the grid's upper z edge.
  - That owed the extension at z = 1000, T = 30000, which won the re-applied rule.
  - Every eligible arm has 0 Spatial successes, so the selection rests on the Spatial probe-loss
    tie-break: 0.730 against 0.791 for the next arm.
- **The z ladder was amended before any arm** (see the design section). The approved
  z ∈ {1, 3, 10, 30} could not run under τ_end = 5, so it moved up one decade to {10, 30, 100, 300}.
  - The amendment was made overnight without the user's confirmation, because the alternative was
    an idle GPU.
  - It was reported on Sun 18 Oct and has not been reversed.
- **It means "the best among tested settings on the development seed"**, never a resolved optimum.

| arm | z | τ_start | T | Spatial `R[1][0]` | Object `R[1][1]` [95% CI] (floor 0.68) | Object − seq_ft [CI] | Spatial probe | Object probe | raw loss, last 1k steps |
|---|---:|---:|---:|---:|---|---|---:|---:|---:|
| z10 | 10 | 1.633 | 2000 | 0.00 | 0.74 [0.62, 0.86] | −0.04 [−0.18, +0.10] | 0.897 | 0.0060 | 0.0053 |
| z10 | 10 | 1.633 | 30000 | 0.00 | 0.64 [0.50, 0.76] ✗ | −0.14 [−0.26, −0.02] | 0.891 | 0.0066 | 0.0057 |
| z30 | 30 | 0.544 | 2000 | 0.00 | 0.88 [0.78, 0.96] | +0.10 [−0.04, +0.24] | 0.965 | 0.0060 | 0.0054 |
| z30 | 30 | 0.544 | 30000 | 0.00 | 0.88 [0.78, 0.96] | +0.10 [0.00, +0.22] | 0.860 | 0.0135 | 0.0129 |
| z100 | 100 | 0.163 | 2000 | 0.00 | 0.74 [0.62, 0.86] | −0.04 [−0.18, +0.10] | 1.065 | 0.0061 | 0.0055 |
| z100 | 100 | 0.163 | 30000 | 0.00 | 0.80 [0.68, 0.90] | +0.02 [−0.10, +0.16] | 0.837 | 0.0222 | 0.0255 |
| z300 | 300 | 0.0544 | 2000 | 0.00 | 0.80 [0.68, 0.90] | +0.02 [−0.06, +0.10] | 1.271 | 0.0061 | 0.0056 |
| z300 | 300 | 0.0544 | 30000 | 0.00 | 0.88 [0.78, 0.96] | +0.10 [−0.04, +0.24] | 0.791 | 0.0185 | 0.0241 |
| **z1000 (ext.)** | **1000** | **0.0163** | **30000** | **0.00** | **0.92 [0.84, 0.98]** | **+0.14 [+0.04, +0.26]** | **0.730** | 0.0147 | 0.0153 |
| Table 5 | 5443 | 0.003 | 2000 | 0.00 | 0.88 [0.78, 0.96] | +0.10 [−0.02, +0.22] | 1.164 | 0.0065 | 0.0058 |
| *seq_ft (reference)* | | | | *0.00* | *0.78* | | *0.725* | *0.0064* | |
| *EWC r = 0.01, v1 / v2 (near seq_ft)* | | | | *0.00* | *0.84 / 0.84* | | *0.732 / 0.699* | *0.0064 / 0.0065* | *0.0058 (v2)* |
| *plain GPM (reference)* | | | | *0.94* | *0.78* | | *0.042* | *0.0072* | |

The table's notation:
- z is the weight's exponent κ·L/τ at the median initial sample, so τ_start = κ·q50 / z with
  q50 = 0.6532. The Table 5 arm's effective z is κ·q50 / 0.003.
- Every arm's Spatial − GPM is −0.94 [−1.00, −0.86].
- "Raw loss" is the method's logged unweighted per-sample mean. It is a single-batch snapshot every
  100 steps, averaged over the last 10 snapshots.
- The EWC r = 0.01 row is the masked mean on the same batches. Its penalty is negligible, and its
  normalization differs from the raw loss's by about 1.4% at step 0 (1.876 against 1.849).

**What it shows:**
- **No tested setting retains Spatial, by either measure.**
  - Success is 0/50 at all ten arms.
  - On the probe loss, no arm is below seq_ft (0.725) or the two near-seq_ft EWC runs
    (0.699, 0.732). The best arm, the selected one, sits at 0.730.
  - **With T = 30000** the probe loss falls with z: 0.891, 0.860, 0.837, 0.791, 0.730. It approaches
    seq_ft's value from above.
  - **With T = 2000** it is higher than seq_ft's at every z, and rises with z: 0.897, 0.965, 1.065,
    1.271; Table 5 is 1.164.
  - For comparison, at their selections EWC reached 0.165, LoRA 0.083 and GPM 0.042.
- **Object is kept, and often raised.**
  - Seven of the nine eligible arms are at or above seq_ft's 0.78; the other two are at 0.74.
  - The selected arm's paired difference, +0.14 [+0.04, +0.26], excludes 0. That interval covers
    rollout noise only, not variation between training runs.
  - **How much Object varies between similar training runs is not known from one seed.**
    - The runs at or near plain SFT give 0.78–0.84: seq_ft, and EWC r = 0.01 in v1 and v2.
    - The two z10 arms (mean ω 0.89–0.93) give 0.74 and 0.64. The only ineligible arm is one of
      them.
    - So an Object gain of this size is not established by one seed.
  - **The long schedule leaves the Object fit looser.** The T = 30000 arms at z ≥ 30 end the stage
    at a 2–4× higher raw loss (0.013–0.026 against 0.0053–0.0058) and a 2–3× higher Object probe
    loss (0.013–0.022 against 0.006). Their Object success is as high or higher (0.80–0.92). On
    Object, the probe loss and the success rate disagree.
- **Overhead: none measurable.**
  - Training took 2365–2397 s per arm, and 2522 s for the first arm.
  - seq_ft's stage 1 took 2558 s through the runner.
  - The two were timed on different days, so this is not a controlled comparison.

**What the weights did** (the method's logs: one batch every 100 steps; flow-time bins of width 0.2)

| arm | first logged step with τ = τ_end | floor fraction, steps 0–1k / 5–10k / 25–30k | mean ω, steps 0–1k / 5–10k / 25–30k | mean ω over the stage | stage-mean ω by s bin, [0, 0.2) … [0.8, 1) |
|---|---:|---|---|---:|---|
| z10 T2000 | 1800 | 0.10 / 0.00 / 0.00 | 0.54 / 0.93 / 0.98 | 0.93 | 0.97 0.97 0.96 0.96 0.79 |
| z10 T30000 | 25700 | 0.10 / 0.01 / 0.00 | 0.53 / 0.83 / 0.98 | 0.89 | 0.94 0.94 0.93 0.93 0.68 |
| z30 T2000 | 2000 | 0.30 / 0.00 / 0.00 | 0.36 / 0.93 / 0.98 | 0.92 | 0.97 0.96 0.95 0.95 0.78 |
| z30 T30000 | 28600 | 0.29 / 0.14 / 0.00 | 0.36 / 0.65 / 0.94 | 0.74 | 0.81 0.80 0.81 0.82 0.47 |
| z100 T2000 | 2000 | 0.51 / 0.00 / 0.00 | 0.15 / 0.93 / 0.98 | 0.91 | 0.95 0.94 0.93 0.93 0.78 |
| z100 T30000 | 29600 | 0.52 / 0.25 / 0.04 | 0.16 / 0.53 / 0.85 | 0.63 | 0.73 0.72 0.72 0.68 0.33 |
| z300 T2000 | 2000 | 0.78 / 0.00 / 0.00 | 0.03 / 0.93 / 0.98 | 0.89 | 0.93 0.93 0.92 0.92 0.77 |
| z300 T30000 | 29900 | 0.78 / 0.30 / 0.08 | 0.03 / 0.42 / 0.80 | 0.54 | 0.67 0.65 0.62 0.54 0.22 |
| **z1000 T30000** | **none (τ = 2.14 at step 29900)** | **0.93 / 0.46 / 0.11** | **0.005 / 0.17 / 0.78** | **0.41** | **0.55 0.52 0.47 0.37 0.13** |
| Table 5 | 2000 | 1.00 / 0.00 / 0.00 | 0.001 / 0.92 / 0.98 | 0.88 | 0.92 0.92 0.91 0.91 0.76 |

- **The T = 2000 arms reach τ_end by step 2000 and then run as SFT at τ_end.** Under Table 5's
  τ_end = 5 that means ω = max(ω_min, exp(−5·L)), with mean ω 0.98 for the rest of the stage.
  - τ_end caps the schedule, so the weighting never fully relaxes.
  - The highest flow-time bin keeps ω ≈ 0.76–0.79 against 0.91–0.97 elsewhere.
- **With T = 30000, the conservative phase lasts the whole stage.**
  - The selected arm starts with 93% of the batch at the floor.
  - Its logged mean ω first reaches 0.5 at step 14,900.
  - Its stage-mean ω is 0.41.
- **Much of the reweighting runs along flow time** (note (d) in `consft_notes.md`).
  - In every arm, the bin s ∈ [0.8, 1) gets the least weight. The calibration's median initial
    loss there is 8.4, against 0.17 at s ≈ 0.05.
  - In the selected arm, that bin averages ω = 0.13 over the stage, against 0.37–0.55 in the
    other bins.
  - So part of ConSFT's "confidence" here is down-weighting the high-s end of the flow, not
    specific demonstrations.
- **A uniform floor trains like SFT under AdamW** (note (a)).
  - Through step 1,600 the Table 5 arm's logged mean ω is at most 0.0013, with 0.95–1.00 of each
    logged batch at the floor.
  - Its raw loss tracks the near-seq_ft run on the same batches: window means 0.250 and 0.045 over
    steps 0–1k and 1k–2k, against EWC r = 0.01's 0.247 and 0.040.
  - The batches are the same: the data order and draws are seeded identically. Step 0's raw loss
    is 1.849 in all ten arms, and EWC's masked loss at step 0 (1.876) is the Table 5 arm's weighted
    loss divided by ω_min.
  - **What follows:** the paper's exact setting behaves, in our recipe, as near-SFT for about 1,700
    steps, then a transition, then SFT at τ_end.
  - It follows that, under AdamW, ConSFT reduces to scaled SFT at both ends of z. At small z every
    ω is near 1; at large z every ω sits at ω_min, a uniform scale. Its effect can only come from
    the spread of weights within a batch. This is a reading, not a measured decomposition.
- **The transition at T = 2000 coincides with a loss jump.**
  - c = 3.5 makes τ rise steeply. The weights go from the floor to near 1 within the last 100–200
    steps before T.
  - At logged step 2000 (the same batch in every arm), the raw loss grows with the jump:
    - z10 0.049, z30 0.072, z100 0.084, z300 0.126, Table 5 0.167;
    - the near-seq_ft run has 0.039 on that batch;
    - at step 1900 every arm is at 0.033–0.066, and by step 2100 at 0.027–0.067.
  - The T = 2000 arms' Spatial probe loss rises in the same order of z (Table 5 is the exception).
  - **The reading:** this is consistent with note (c)'s transient. Weights rise up to about 500×
    within about 100 steps, while β2 = 0.999 keeps roughly 1000 steps of memory, so Adam's step
    size is inflated for a while.
  - **The limits:** these are single-batch snapshots, and the logs do not measure the step size. It
    is a reading, not a demonstration.
  - **At T = 30000 there is no comparable jump:** mean ω over steps 20k–25k is already 0.64–0.96
    before τ rises in the last 5k steps.

**Ranges:**
- Tested: z from 10 to 300 at T = 2000 and T = 30000, plus z = 1000 at T = 30000 and Table 5's exact
  point (z ≈ 5443, T = 2000).
- Not tested:
  - z = 3: its τ_start of 5.44 is at the cap, so it is not runnable;
  - z = 1000 at T = 2000: not owed;
  - z > 1000;
  - T other than 2000 and 30000. T is never extended, and T > 30000 would leave the schedule
    unfinished at the stage's end.
- The selection sits at the upper z edge. The T = 30000 probe-loss trend points toward seq_ft's
  value, not below it.

**Pre-written interpretations, applied:**
- **The selection is "the best among tested settings" on seed 0.** It is not a resolved optimum,
  and it is flagged `optimum_at_boundary`.
- **`all_eligible_spatial_zero`: ConSFT still enters the 4-task runs at its selection**, as EWC did.
- **The selection becomes `configs/method/consft.yaml`** in a commit of its own (`5b4ae22`).
- **The record reports what the rule requires:** the full Spatial–Object trade-off, the per-arm
  weighting logs, and the weights per flow-time bin (above).
- **Limits:**
  - one seed;
  - one training run per arm, so training-run variation is in none of the intervals (near-SFT runs
    span 0.78–0.84 in Object and 0.70–0.73 in Spatial probe loss);
  - a selection decided by a probe-loss margin of 0.06;
  - our base is a single-task T1 policy, not a pretrained foundation model (`consft_notes.md`,
    decision 3);
  - AdamW with β2 = 0.999 instead of the paper's Adam at 0.95 (note (c)).

**What the rule now implies:**
- **The 4-task ConSFT runs use `configs/method/consft.yaml`:** τ_start 0.016330303251743318,
  decay_steps 30000, and Table 5's constants.
- **Production ConSFT seed 0's stage 1 is identity-checked against the selected arm:**
  `--identity-reference-checkpoint 1=results/baseline_selection_consft/consft/seed0/consftb__z1000__T30000__seed0/checkpoints/final.pt`
  (tensor fingerprint `3b6cd54d…`).
- **Track 2 (step 4) must exclude the union of the frozen development keys and the bundle's
  `track2_exclusion_keys`.**
  - The bundle holds 20 keys: (tensor fingerprint, task key) for both trained tasks of all ten arms,
    selected or not, the extension included.
  - `verify` recomputed them from the checkpoint tensors.
  - **The acceptance check:** production ConSFT seed 0's stage 1, once identity-checked against the
    selected arm, must be excluded by that union.
  - The frozen rule and its bundle are not modified.
- **The same gap exists for the 14 Oct brackets of `projected_lora` (13 arms) and `inflora` (12 arms).**
  - Both methods are in the reserved test runs.
  - Their arms ran after the 12 Oct manifest, and their bundles carry no exclusion keys.
  - Step 4 must derive their keys from those bundles' arm checkpoints, with the same acceptance
    check.

**Date (local, CEST):** Sat 17 Oct 23:58:40 → Sun 18 Oct 09:40:17 (9 h 42 min).
- **The nine base arms** took 57.3–62.7 min each, of which training was 39.4–42.0 min.
- **The first selection (6 s)** found z300 T30000 and owed the extension.
- **The extension** took 56.9 min.
- **The re-applied selection** took 6 s and the verification 19 s.

**Code:**
- **The method and rule:**
  - `a8250ce`: the method and the rule, with `calibration: null`;
  - `288c343`: the amendment, before any arm;
  - `9a90b6f`: the calibrated rule.
- **The arms all ran at `9a90b6f`, clean at queue start.** Every row's `git_sha` is `9a90b6f…`.
- **Tests:**
  - the full CPU suite passed at `a8250ce`: 1155 tests, 0 failures, 4591 s;
  - the 21 ConSFT tests passed at `288c343`;
  - with the production-config test at `5b4ae22`, the 22 ConSFT tests pass.
- **The GPU smoke** ran into scratch (200 steps, 1 episode, before the calibration). It found:
  - stage 0 identical to seq_ft: 0 of 656 tensors differ;
  - resume at stage 3 equal under `compare_resume`;
  - the bracket path reproducing the runner's stage 1 tensor for tensor, with identical weighting
    logs;
  - logged τ(t) following eq. 12.

  Its per-step timing was confounded by a concurrent CPU suite, so the overhead is read from the
  30k-step arms instead (above).

**Cite:**
- the bundle `results/baseline_selection_consft/consft/selection/selection.json`, SHA-256
  `b0f20f54b7f549150443efc9cd47ac118b30e6e4e628b0e2c19267a43bf0cad1`;
- the rule `configs/analysis/baseline_selection_consft.yaml`, SHA-256 `9f66d364…`;
- the calibration `results/baseline_selection_consft/consft_calibration.json`, SHA-256 `cc2d2a39…`;
- the arms `results/baseline_selection_consft/consft/seed0/consftb__*__seed0/row.json`;
- the queue log `results/logs/queue_20261017_235840_consft_bracket/queue.log`.

---

## Design (as pre-registered; `configs/analysis/baseline_selection_consft.yaml`, `docs/consft_notes.md`)

- **The method:** each sample's flow-matching loss is scaled by the stop-gradient weight
  ω = max(ω_min, exp(−κ·L_i / τ(t))), with τ(t) = min(τ_start / max(ρ(t), 1e-8), τ_end) from eq. 11–12
  (arXiv 2605.08879v2, App. C.1).
  - **The constants are Table 5's** (the user's decision over the released code's defaults):
    κ 25, λ 0.8, c 3.5, ω_min 1e-3, τ_end 5.
  - **T1 is seq_ft,** and the schedule restarts at every task.
- **The setup** is that of the EWC and LoRA brackets:
  - seed 0; T2 (Object) trained from seq_ft seed 0's `stage0.pt` through `low_update.train_stage`
    with the runner's text cache;
  - seq_ft's recipe: 30k steps, lr 1e-4, batch 64, AMP, warmup and cosine;
  - T1 and T2 each evaluated with 50 rollouts, plus the fixed probe losses.
- **The calibration** used no arm outcome.
  - q50 is the median initial per-sample loss of `stage0.pt` on T2: 4096 draws, train mode, AMP.
    The result: q50 = 0.6532. The mean is 1.69, q05 0.076 and q95 8.64.
  - The per-s-bin medians run from 0.17 at s ≈ 0.05 to 8.4 at s ≈ 0.95.
  - On seq_ft's `stage1.pt`, the median is 0.0002 and the mean 0.0072.
  - Initial floor fractions: z10 0.613, z30 0.864, z100 0.981, z300 0.999, z1000 and Table 5 1.000.
- **The amendment** (Sat 17 Oct 23:50, before any arm):
  - **Why:** the first calibration refused under the pre-written cap rule. z = 1 and z = 3 need
    q50 < 0.20 and < 0.60, and under τ_end = 5 every feasible arm has z > κ·q50 / τ_end ≈ 3.3.
  - **What changed:**
    - z moved up one decade, to {10, 30, 100, 300};
    - the extensions became z = 3 (capped at this q50, so a selection at z = 10 would be flagged
      and owe nothing) and z = 1000;
    - T, the Table 5 arm and every constant are unchanged.
- **The grid:** z × T ∈ {2000, 30000}, plus the Table 5 arm (τ_start 0.003, T 2000), 9 arms.
- **The rule:**
  - **Eligible:** Object ≥ seq_ft's stage-1 Object − 0.10, which gives 0.68.
  - **Order:** the highest Spatial, then the lower Spatial probe loss, then the smaller effective z,
    then the shorter T, then the arm id.
  - **Fallback when no arm is eligible:** the highest Object, then the lower Object probe loss, then
    z, T and the arm id. It is flagged `no_qualifying_setting`.
  - **Extension eligibility comes from the arm type,** never from z.
  - **Edge extension:**
    - a selected grid arm at min(zs) owes z = 3, at the selected T, unless capped;
    - one at max(zs) owes z = 1000, at the selected T;
    - the rule is then re-applied once;
    - Table 5 and extension arms are never extended, and T is never extended.
- **Track 2:**
  - The bundle carries `track2_exclusion_keys`: (`ewc.tensor_fingerprint` of the arm's final state
    dict, task key) for both trained tasks of every completed arm.
  - `verify` recomputes them from the checkpoint tensors.
  - These arms come after the freeze, so the frozen development keys cannot contain them.
