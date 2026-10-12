# LoRA bracket v2, the lower-lr band (seed 0, T1 → T2): the rule selects rank 16, lr 3e-5 (Spatial 0.52, Object 0.70), inside the grid. At lr 1e-5 Spatial rises further (up to 0.58) but Object falls below the floor at every rank, so the low-lr edge is now located

**Status: the pre-registered follow-up, complete.**
- The queue ended `QUEUE DONE ok`. The verification passed, including v1's own verification and the
  imported arms' match with v1's bundle.
- **The registered selection is rank 16, lr 3e-5 (alpha 16).** No flags: the cell is inside the
  grid, so no extension was owed.
  - The imported rank-1 row (1, 1e-4) was not owed. It is listed as `available_not_owed` and stayed
    out of the selection.
- **This follow-up was decided after v1's outcome** (v1 selected rank 4, lr 3e-5,
  `optimum_at_boundary`), and is labelled post hoc. v1's rule, rows, bundle and record
  (`runs/2026-10-11_lora_bracket.md`) stand unchanged. **v2's selection supersedes v1's** for the
  4-task runs.
- **It means "the best among tested settings on the development seed".** The selected arm's Object
  (0.70) is one rollout above the floor (0.68). Its Spatial lead over the next eligible arms is 2–4
  rollouts (0.52 against 0.48, 0.46, 0.44), within noise.

| rank | lr | origin | Spatial `R[1][0]` [95% CI] | Object `R[1][1]` [CI] (floor 0.68) | Object − seq_ft [CI] | Spatial − GPM | Spatial probe | Object probe | merge ‖ΔW‖/‖W‖, median (max) |
|---:|---:|---|---|---|---|---:|---:|---:|---|
| 1 | 1e-4 | v1 ext. | 0.02 [0.00, 0.06] | 0.66 ✗ | −0.12 [−0.32, +0.08] | −0.92 | 0.115 | 0.0363 | 0.34 (1.45) |
| 4 | **1e-5** | **v2** | 0.58 [0.44, 0.72] | 0.54 [0.40, 0.68] ✗ | −0.24 [−0.40, −0.06] | −0.36 | 0.070 | 0.0716 | 0.08 (0.33) |
| 4 | 3e-5 | v1 | 0.48 [0.34, 0.62] | 0.72 | −0.06 [−0.22, +0.10] | −0.46 | 0.087 | 0.0478 | 0.12 (0.46) |
| 4 | 1e-4 | v1 | 0.46 [0.32, 0.60] | 0.72 | −0.06 [−0.20, +0.10] | −0.48 | 0.113 | 0.0288 | 0.22 (0.71) |
| 4 | 3e-4 | v1 | 0.28 [0.16, 0.42] | 0.72 | −0.06 [−0.22, +0.10] | −0.66 | 0.133 | 0.0167 | 0.40 (1.33) |
| 4 | 1e-3 | v1 | 0.04 [0.00, 0.10] | 0.68 | −0.10 [−0.28, +0.06] | −0.90 | 0.349 | 0.0129 | 0.90 (2.86) |
| 16 | **1e-5** | **v2** | 0.48 [0.34, 0.62] | 0.50 [0.36, 0.64] ✗ | −0.28 [−0.44, −0.12] | −0.46 | 0.073 | 0.0702 | 0.07 (0.27) |
| **16** | **3e-5** | **v2** | **0.52 [0.38, 0.66]** | **0.70 [0.58, 0.82]** | −0.08 [−0.24, +0.08] | −0.42 | **0.083** | 0.0471 | 0.10 (0.38) |
| 16 | 1e-4 | v1 | 0.44 [0.30, 0.58] | 0.88 | +0.10 [−0.02, +0.22] | −0.50 | 0.116 | 0.0274 | 0.16 (0.61) |
| 16 | 3e-4 | v1 | 0.20 [0.10, 0.32] | 0.64 ✗ | −0.14 [−0.28, 0.00] | −0.74 | 0.157 | 0.0131 | 0.28 (1.17) |
| 16 | 1e-3 | v1 | 0.02 [0.00, 0.06] | 0.76 | −0.02 [−0.18, +0.14] | −0.92 | 0.388 | 0.0086 | 0.65 (2.16) |
| 64 | **1e-5** | **v2** | 0.44 [0.30, 0.58] | 0.58 [0.44, 0.72] ✗ | −0.20 [−0.38, 0.00] | −0.50 | 0.074 | 0.0695 | 0.06 (0.27) |
| 64 | **3e-5** | **v2** | 0.32 [0.20, 0.46] | 0.80 [0.68, 0.90] | +0.02 [−0.14, +0.18] | −0.62 | 0.085 | 0.0472 | 0.09 (0.33) |
| 64 | 1e-4 | v1 | 0.20 [0.10, 0.32] | 0.90 | +0.12 [+0.02, +0.24] | −0.74 | 0.119 | 0.0269 | 0.14 (0.53) |
| 64 | 3e-4 | v1 | 0.08 [0.02, 0.16] | 0.74 | −0.04 [−0.20, +0.12] | −0.86 | 0.156 | 0.0123 | 0.25 (1.11) |
| 64 | 1e-3 | v1 | 0.00 [0.00, 0.00] | 0.72 | −0.06 [−0.22, +0.12] | −0.94 | 0.393 | 0.0075 | 0.54 (1.84) |
| *seq_ft* | *1e-4* | | *0.00* | *0.78* | | | *0.725* | *0.0064* | |
| *plain GPM* | *1e-4* | | *0.94* | *0.78* | | | *0.042* | *0.0072* | |

**What it shows:**
- **Below lr 3e-5, Object stops learning enough.** At lr 1e-5 every rank falls below the floor
  (Object 0.50–0.58; CIs exclude seq_ft's 0.78 at ranks 4 and 16). Spatial keeps rising
  (0.44–0.58), and Spatial's probe loss falls to 0.070–0.074. The floor is what bounds the choice,
  and v2 has now found where it binds.
- **The trade-off frontier** (arms no other arm beats on both Spatial and Object):

  | arm | Spatial | Object |
  |---|---:|---:|
  | (4, 1e-5), ineligible | 0.58 | 0.54 |
  | **(16, 3e-5), selected** | **0.52** | **0.70** |
  | (4, 3e-5) | 0.48 | 0.72 |
  | (16, 1e-4) | 0.44 | 0.88 |
  | (64, 1e-4) | 0.20 | 0.90 |

  - Along it, Spatial trades against Object.
  - **(16, 1e-4) gives up 4 Spatial rollouts for 9 Object rollouts** against the selected arm. The
    rule ranks Spatial first among eligible arms, so the selection follows point estimates that
    differ within noise.
- **Rank matters less than lr, and less at low lr.**
  - At lr 3e-5: rank 16 keeps 0.52, rank 4 0.48, rank 64 0.32 (with more Object, 0.80).
  - At lr 1e-5, Spatial falls with rank (0.58, 0.48, 0.44).
  - The merged update is small at low lr: ‖ΔW‖/‖W‖ has a median of 0.06–0.12 at 1e-5 and 3e-5,
    against 0.54–0.90 at 1e-3.
- **GPM still keeps far more.** The paired difference from GPM is −0.42 for the selected arm, and
    −0.36 even for the ineligible (4, 1e-5). GPM keeps 0.94 at Object 0.78. LoRA's lowest Spatial
    probe loss (0.070) is still 1.7 times GPM's.

**Ranges, kept separate:**
- **v1:** lr 1e-4 to 1e-3 at ranks 4, 16 and 64; lr 3e-5 at rank 4; rank 1 at lr 1e-4.
- **v2:** adds lr 1e-5 and 3e-5 at ranks 4, 16 and 64.
- No arm below lr 1e-5 or above rank 64 was tested. The rule owed none, since the selection lies
  inside the grid.

**Pre-written interpretations, applied:**
- **The selection:** rank 16, lr 3e-5 is "the best among tested settings" on seed 0.
- **Spatial > 0 at eligible arms:** observed retention on the development seed. Broader claims wait
  for the replication seeds.
- **Comparison with GPM:** LoRA and GPM are two training procedures on the same weight support. The
  gap is not attributable to rank alone.
- **Limits:**
  - v1's limits;
  - the post-hoc decision;
  - arms from two commits with a byte-identical training path;
  - a selection decided by a one-rollout Object margin at the floor and a 2–4-rollout Spatial lead.

**What the rule now implies:**
- **The 4-task LoRA runs use rank 16, lr 3e-5, alpha 16** (in `configs/method/lora.yaml`, its own
  commit).
- **Production LoRA seed 0's stage 1 is identity-checked against this arm**
  (`--identity-reference-checkpoint 1=results/baseline_selection_lora_v2/lora/seed0/lorab__r16__lr3e-05__seed0/checkpoints/final.pt`;
  tensor fingerprint `e339c578…`). If it reproduces the arm, it is excluded from Track 2.
- The 4-task runs wait for Track 2's freeze.

**Date (local, CEST):** Sun 11 Oct 21:49:39 → Mon 12 Oct 03:10:13 (5 h 21 min).
- **The status step (16 s)** found:
  - 11 imported arms (10 grid cells plus the available (1, 1e-4));
  - 5 missing cells;
  - v1's verification passing.
- **The five new arms** took 61.7–67.6 min each, of which training was 46.8–50.5 min.
- The selection took 7 s and the verification 28 s.

**Code:**
- **The v2 arms ran at `620727b`, clean at queue start.** Every new row's `git_sha` is `620727b…`. The
  imported arms ran at `75fb0e1`.
- **The training path is unchanged:** `git diff 75fb0e1` over `models/lora.py`, `methods/lora.py`,
  `train/`, `low_update.py`, `hparam_study.py` and `methods/base.py` was empty before launch.
- **The rule:** `configs/analysis/baseline_selection_lora_v2.yaml`, SHA-256 `5b53ede2…`, committed
  before any v2 arm. It imports v1's rule (`a6582554…`) and bundle (`a1d62334…`).
- **Launch before the suite (disclosed):** the queue launched at 21:49, before the full CPU suite
  had run.
  - The bracket tests and a dry run had passed.
  - The suite then passed at `620727b`: 1067 tests, 0 failures, finished 22:55.
- **No GPU smoke:** v1's smoke validated the unchanged training path.

**Cite:**
- `results/baseline_selection_lora_v2/lora/selection/selection.json` (SHA-256 `455edc42…`, inside the
  immutable bundle);
- the new arms `results/baseline_selection_lora_v2/lora/seed0/lorab__r{4,16,64}__lr{1e-05,3e-05}__seed0/row.json`;
- the imported arms, cited in v1's record;
- the queue log `results/logs/queue_20261011_214939_lora_bracket_v2/queue.log`.

---

## Design (as pre-registered; `configs/analysis/baseline_selection_lora_v2.yaml`)

- **Unchanged from v1:**
  - the method (sequential merged LoRA, alpha 16, the released initialization);
  - seed 0 and the start checkpoint;
  - the recipe and the paired cells;
  - the floor, the ordering (Spatial, Spatial probe loss, smaller rank, smaller lr) and the
    no-qualifying fallback;
  - the extensions and the input binding;
  - `verify`'s recomputation.
- **The grid:** ranks {4, 16, 64} × lrs {1e-5, 3e-5, 1e-4, 3e-4, 1e-3}.
  - **10 grid cells are imported** from v1 (bound to v1's rule and bundle; each equal to v1's
    bundle arm).
  - **(1, 1e-4) is imported as an available extension**, entering the selection only if owed.
  - **5 new arms.**
- **Extensions:** lr 1e-5 → 3e-6, rank 4 → 1 or 64 → 256, re-applied once. None was owed.
- **The bundle** records the imports, each arm's origin and v2's rule SHA. `verify` also runs v1's
  verification.
- **Track 2:** as v1's. Every bracket cell is excluded from the held-out set, matched by tensor
  fingerprint.
