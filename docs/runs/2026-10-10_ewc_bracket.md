# EWC λ bracket (seed 0, T1 → T2): Spatial retention is 0/50 at every λ up to r = 100, so all five arms tie, and the rule selects the smallest λ (1.244, r = 0.01), which leaves EWC effectively seq_ft. The penalty cuts Spatial's probe loss fourfold but never reaches a level that rollouts reward

**Status: the pre-registered bracket, complete.**
- The queue ended `QUEUE DONE ok`, the verification passed, and the bundle is published.
- **The registered selection is λ = 1.2439615682657392 (r = 0.01).** It is a **degenerate tie**:
  Spatial success is 0.00 at every λ.
- The edge extension did not trigger, because the largest λ tied rather than won. The rule had no
  provision for "everything ties at zero". That gap is disclosed here, and the rule is not changed
  after the fact.

| r | λ | Spatial `R[1][0]` | Object `R[1][1]` (floor 0.68) | Object − seq_ft [95% CI] | Spatial probe loss | Object probe loss | mean loss, last 50 (with penalty) | weighted penalty, last log |
|---:|---:|---:|---:|---|---:|---:|---:|---:|
| 0.01 | 1.244 | 0.00 | 0.84 | +0.06 [−0.06, +0.20] | 0.732 | 0.0064 | 0.0076 | 0.0001 |
| 0.1 | 12.44 | 0.00 | 0.82 | +0.04 [−0.06, +0.16] | 0.682 | 0.0065 | 0.0081 | 0.0005 |
| 1 | 124.4 | 0.00 | 0.86 | +0.08 [−0.02, +0.20] | 0.559 | 0.0066 | 0.0098 | 0.0021 |
| 10 | 1244 | 0.00 | 0.80 | +0.02 [−0.08, +0.12] | 0.351 | 0.0075 | 0.0134 | 0.0047 |
| 100 | 12440 | 0.00 | 0.70 | −0.08 [−0.18, +0.02] | 0.170 | 0.0104 | 0.0196 | 0.0077 |
| *seq_ft (reference)* | | *0.00* | *0.78* | | *0.725* | *0.0064* | | |
| *plain GPM (reference)* | | *0.94* | *0.78* | | *0.042* | *0.0072* | | |

**What it shows:**
- **The penalty does what it is built to do, on the loss.** Spatial's probe loss falls
  monotonically with λ, from 0.73 (seq_ft's level) to 0.17 at r = 100. That is a fourfold reduction.
- **No closed-loop retention follows.** Spatial is 0 of 50 at every λ. The level that rollouts
  reward is plain GPM's: 0.042 probe loss with 94% success, still four times lower than EWC's best.
- **Object's learning holds up to r = 10** (0.80–0.86) and drops to 0.70 at r = 100. That is still
  above the floor, but its probe loss is 1.6 times seq_ft's there.
- **Inside this grid,** this EWC buys a smaller loss on the old task at a growing cost on the new
  one, and never reaches the regime where the old task succeeds.
  - This matches the threshold-like relation between loss and success that motivates Track 2. At a
    similar probe loss, Spatial's success varies widely.
- **Higher λ would most likely drop Object below the floor** (0.70 at r = 100, against 0.68). That
  is an extrapolation and was not tested.

**The method this applies to** (the rule's own wording): EWC with an **empirical-Fisher surrogate**.
That is the squared per-sample gradients of the masked flow-matching loss, not the Fisher
information of the action distribution. The result concerns this specified EWC baseline. Each
ratio r describes the penalty at the seq_ft endpoint, not throughout an EWC run.

**What the rule now implies, and the open decision.** By the registered rule, the 4-task EWC runs
use λ = 1.244. At that strength EWC is effectively seq_ft: 0.00 Spatial, and Object +0.06 with a CI
that includes 0. The rule is applied as written; nothing is changed after seeing the outcomes. Two
honest paths remain, and the user decides between them:
1. **Keep the registered selection.** Report EWC as "no λ in the bracket retained the first task at
   a strength that kept the second task learning (r ≤ 100); the rule's tie-break then selects the
   weakest λ".
2. **Pre-register a follow-up before any 4-task EWC run, labelled post hoc.**
   - For example, break ties on Spatial probe loss (that would choose r = 100), or extend the grid
     upward, r ∈ {1000, 10000}, under the same floor.
   - Either is decided after these outcomes, so it is disclosed as a revision of the selection rule,
     not as the registered result.

**Date (local, CEST):** Fri 9 Oct 23:19:24 → Sat 10 Oct 04:59:07 (5 h 40 min).
- Each arm took 66.6–70.2 min. Training was 48.5–48.9 min against plain T2's ~43, a ~13% EWC
  overhead that includes two 1024-sample Fishers.
- The selection took 4 s and the verification 3 s.

**Code:** `aaedf24`, clean at queue start. Every row's `git_sha` is `aaedf24…`.
- The rule `configs/analysis/baseline_selection.yaml` has SHA-256 `08a516db…`. Its calibrated grid
  was committed in `615c7b3`, before any arm, and the review fixes followed in `aaedf24`.
- The calibration `results/baseline_selection/ewc_calibration.json` has SHA-256 `b8e79779…`.
  - It ran at 22:27 on `2c2b827` and used no EWC-arm outcome.
  - Values: D = 1.2196e-4 (the T1-Fisher-weighted drift of seq_ft stage 0 → 1), and
    L = 0.0075858106359373775 (seq_ft's stage-1 `mean_last_50_loss`).
  - The T1 Fisher came from 1024 samples, seed 4030082706, with T1 cached under AMP and Object in
    fp32. Its fingerprint is `320120a7…`.
- **Every arm's T1 Fisher fingerprint is `320120a7…`,** equal to the calibration's. Since the review
  fixes, this is checked before training.
- **The full CPU suite passed (1022).** It ran on `615c7b3`; the review fixes changed only the
  bracket module, its CLI and its tests, whose 14 tests pass.

**The smoke** (scratch, 22:28–22:53, 200 steps, 1 episode; `QUEUE DONE ok`):
- replay and EWC are bitwise seq_ft at stage 0 (0 of 656 tensors), with T1 pairing at relative
  difference 0.0;
- **resume equality at stage 3** for both methods:
  - the tensors match (0 of 656);
  - every named artifact is equal (replay's 4 buffer files; EWC's cumulative state);
  - every evaluation episode is equal;
- the bracket arm's T1 Fisher equals the calibration's;
- a second single-arm smoke after the review fixes also passed.

**Cite:**
- `results/baseline_selection/ewc/selection/selection.json` (SHA-256 `5413c907…`, inside the
  immutable bundle);
- the arms `results/baseline_selection/ewc/seed0/ewcb__r{0.01,0.1,1,10,100}__seed0/row.json`;
- the calibration file above;
- the queue log `results/logs/queue_20261009_231924_ewc_bracket/queue.log`.

---

## Design (as pre-registered; `configs/analysis/baseline_selection.yaml`)

- **The method:** online EWC (γ = 1).
  - The Fisher is the mean over 1024 samples of squared per-sample gradients, in fp32, with its own
    seeds and in `train()` mode.
  - Fishers are summed across tasks, anchored at the post-task weights.
  - The penalty, (λ/2) Σ F (θ − θ\*)², is computed in fp32.
- **The arms:** seed 0, starting from seq_ft's `stage0.pt` (EWC's stage 0 is bitwise seq_ft's).
  - Each trains Object with the standard recipe (30k steps, lr 1e-4), through `train_stage` with
    the runner's text-cache state.
  - Each arm's setup computes the T1 Fisher with production seeds, through the same function the
    calibration uses.
  - Rollouts are paired with seq_ft's stage 1.
- **The grid:** λ_r = 2rL/D for r ∈ {0.01, 0.1, 1, 10, 100}.
- **The rule:**
  - Eligible arms keep Object ≥ seq_ft's Object − 0.10. Among them, the highest Spatial wins;
    ties go to the smaller λ.
  - If no arm is eligible, the smallest λ is selected and flagged.
  - If the largest λ wins outright, one extension arm runs at 10× it.
  - The selection bundle is bound to the rule's and the calibration's SHA-256.
  - **Disclosed:** one transition on the development seed, which also enters the final evaluation.
- **Track 2:** every bracket cell is excluded from the held-out set. So is EWC seed 0's stage 1 if
  it reproduces the selected arm (an identity check with `--identity-reference-checkpoint`), and
  so are the T1 cells shared with seq_ft.
