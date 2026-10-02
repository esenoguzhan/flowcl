# flowcl — Continual Learning for Flow-Matching Robot Policies

**Implementation spec for Cursor.** This is the authoritative description of what to build. Read it fully before generating code. When a request conflicts with this file, follow this file and say so.

M.Sc. thesis: *Continual Learning for Flow-Matching Robot Policies: Forgetting, Interference, and Mitigation*.

Two-stage plan:

1. **Stage A — LIBERO (simulation).** All method development, all mechanistic analysis, all ablations. Cheap rollouts, multiple seeds, multiple task orderings.
2. **Stage B — AgileX dual-arm (hardware).** Validation only: the winning method plus two reference baselines on a short task sequence. No hyperparameter search on hardware.

Everything in Stage A must run unchanged on Stage B except an embodiment config and a dataset adapter. That constraint drives most of the architecture below.

---



## 0. Non-negotiable scoping decisions

These are settled. Do not re-litigate them in code comments or propose alternatives unprompted.


| Decision                   | Value                                                                    | Why                                                                                                                          |
| -------------------------- | ------------------------------------------------------------------------ | ---------------------------------------------------------------------------------------------------------------------------- |
| Visual encoder             | **Frozen** during all continual learning                                 | Confines forgetting to trunk + flow decoder, so `ρ_l` / `c_l` measure the mechanism under study rather than perception drift |
| Policy size                | ~20–60M trainable params                                                 | Must be trainable end-to-end on one RTX 4090 in < 6h per task                                                                |
| Action representation      | Fixed dimensionality **per embodiment**, never padded across embodiments | Padding produces artifactual forgetting (switching which slice is active), not representational interference                 |
| Task identity at inference | **Not used**                                                             | Exemplar-free, task-ID-free evaluation; language conditioning only                                                           |
| Projection scope           | Linear layers in trunk + decoder only                                    | Norm params, biases, and timestep embeddings are frozen in projection experiments (§7.4)                                     |
| Novelty claim              | Flow-time-conditioned (`s`-binned) subspace protection                   | Gradient projection itself is not claimed as novel                                                                           |


**Known gap to handle explicitly:** LIBERO is single-arm Franka. The bimanual handover / coordinated-transport tasks from the proposal cannot be run in Stage A. Bimanual skills are a **Stage B–only** contribution. Stage A studies interference on single-arm heterogeneous skills. Do not fake bimanual behavior in LIBERO by splitting the action vector.

---



## 1. Repository layout

```
flowcl/
├── configs/                     # hydra configs, one concern per file
│   ├── config.yaml              # root, composes the rest
│   ├── embodiment/              # libero_franka.yaml, agilex_dual.yaml
│   ├── policy/                  # flowpolicy_small.yaml, flowpolicy_base.yaml
│   ├── curriculum/              # seq_hetero.yaml, seq_correlated.yaml, agilex_short.yaml
│   ├── method/                  # seq_ft, lora, replay, consft, ewc, gpm, sgp, sgp_sbin
│   └── eval/                    # libero_eval.yaml, hw_eval.yaml
├── flowcl/
│   ├── data/
│   │   ├── spec.py              # EmbodimentSpec, ObservationSpec, ActionSpec
│   │   ├── episode.py           # canonical episode container (see §3.1)
│   │   ├── libero_adapter.py    # LIBERO hdf5 -> canonical episodes
│   │   ├── agilex_adapter.py    # teleop logs -> canonical episodes (Stage B)
│   │   ├── dataset.py           # chunked torch Dataset, normalization
│   │   └── stats.py             # per-embodiment normalization stats, versioned
│   ├── models/
│   │   ├── encoders.py          # frozen vision backbones + language encoder
│   │   ├── trunk.py             # observation projections + transformer
│   │   ├── flow_head.py         # flow-matching action decoder
│   │   ├── policy.py            # FlowPolicy: forward(loss) + sample(actions)
│   │   └── lora.py              # LoRA injection into named linear layers
│   ├── train/
│   │   ├── trainer.py           # single-task training loop
│   │   ├── continual.py         # sequential runner over a curriculum
│   │   ├── optim.py             # ProjectedOptimizer wrapper (see §7.3)
│   │   └── losses.py
│   ├── methods/                 # one file per CL method, shared interface (§6)
│   │   ├── base.py              # ContinualMethod protocol
│   │   ├── seq_ft.py  lora.py  replay.py  consft.py  ewc.py
│   │   ├── gpm.py     sgp.py   sgp_sbinned.py
│   ├── analysis/
│   │   ├── hooks.py             # activation capture, s-tagged
│   │   ├── subspace.py          # SVD bases, ρ_l, energy thresholds
│   │   ├── interference.py      # c_l, per-layer gradient decomposition
│   │   ├── flowtime.py          # s-binned ρ_l(s), c_l(s)
│   │   └── metrics.py           # R_{i|j}, F_1, AUC, FWT, NBT
│   ├── envs/
│   │   ├── libero_env.py        # rollout wrapper, deterministic seeding
│   │   └── hw_env.py            # AgileX closed-loop wrapper (Stage B)
│   └── utils/                   # seeding, logging, checkpoint registry, wandb
├── scripts/                     # thin CLI entrypoints only, no logic
│   ├── prepare_libero.py  train_single.py  run_continual.py
│   ├── collect_basis.py   analyze_interference.py  evaluate.py
│   └── make_tables.py
├── tests/
└── results/                     # run registry, never committed
```

**Rules:** no logic in `scripts/`. No `sys.path` hacks. No notebooks in the critical path — notebooks may only read from `results/`.

---



## 2. Environment

- Python 3.10, PyTorch 2.4+, CUDA 12.x. Target: Ubuntu 24.04 + RTX 4090 (`gamma`), plus DGX for parallel seeds.
- LIBERO requires `robosuite` + MuJoCo. Pin exact versions in `pyproject.toml`; LIBERO's own repo is a **submodule or vendored dependency**, never edited in place.
- `uv` for dependency management. One lockfile, committed.
- Every run writes `results/<run_id>/config.yaml`, `git_sha`, `pip freeze`, and the resolved seed. A run without these is invalid.

---



## 3. Data layer



### 3.1 Canonical episode format

Both LIBERO and AgileX must be converted to one format. The policy code never sees a raw LIBERO HDF5 or a raw teleop log.

```python
@dataclass
class Episode:
    images: dict[str, np.ndarray]   # camera_name -> (T, H, W, 3) uint8
    state:  np.ndarray              # (T, D_state) float32, proprioception
    action: np.ndarray              # (T, D_action) float32
    language: str
    task_id: str                    # bookkeeping/analysis only, never a policy input
    embodiment: str                 # "libero_franka" | "agilex_dual"
    meta: dict                      # source file, demo index, success flag
```

`EmbodimentSpec` declares camera names, `D_state`, `D_action`, control mode, and control rate. The policy builds its input/output projections from the spec. Switching embodiment = new spec + re-initialized input/output projections, **trunk weights transfer**.

### 3.2 LIBERO adapter

- Suites: `libero_spatial`, `libero_object`, `libero_goal`, `libero_10`. 10 tasks each, 50 demos each.
- Use LIBERO's own HDF5 demos; do not regenerate. Verify demo count and action stats on load and fail loudly on mismatch.
- Cameras: `agentview` + `robot0_eye_in_hand`, resized to 128×128 (configurable).
- State: end-effector pose + gripper qpos. Actions: 7-dim OSC delta (6 pose + 1 gripper), already normalized to [-1, 1] by LIBERO — do not renormalize, just record the stats.



### 3.3 Chunking and normalization

- Action chunk horizon `H = 16` (config). Dataset item = `(obs_t, A_t ∈ R^{H × D_action})`, right-padded at episode end with a validity mask; masked steps contribute zero loss.
- Normalization stats are computed **once per embodiment over Task 1's data only** and frozen for the whole curriculum. Recomputing stats per task silently changes the target distribution and corrupts forgetting measurements. Assert this at every stage boundary.

---



## 4. Policy

```
images ──> frozen vision encoder ──┐
state  ──> linear proj ────────────┼──> transformer trunk ──> context tokens
language ─> frozen text encoder ───┘                              │
                                                                  ▼
                        (A_s, s) ──> flow-matching decoder ──> v_θ(A_s, o, s)
```



### 4.1 Encoders (frozen)

Default: DINOv2-S or SigLIP-B, frozen, with a trainable linear patch projection. Language: frozen CLIP/T5 text embedding, cached per task string (there are few unique instructions — cache them, do not run the text tower every step).

**Escape hatch, use only if needed:** if single-task LIBERO success is below ~70% with frozen features, switch to a ResNet-18 trained from scratch (robomimic-style) for the *single-task capability* study and document the change. Frozen large ViT features are often weaker than a small trained CNN at 50-demo scale. Decide this in Month 1, once, and record it. Do not silently vary the encoder between runs.

### 4.2 Trunk

Small transformer: 6–8 layers, `d_model` 384–512, 8 heads, pre-norm, learned positional embeddings over the observation token set. Outputs a fixed set of context tokens consumed by the decoder via cross-attention.

### 4.3 Flow-matching decoder

Conditional flow matching over action chunks:

- `A_1` = ground-truth chunk, `A_0 ~ N(0, I)`, `s ~ U(0,1)` (make the `s` sampler swappable; add logit-normal as an option).
- `A_s = (1-s) A_0 + s A_1`
- Loss: `|| v_θ(A_s, o, s) - (A_1 - A_0) ||²`, masked.
- `s` enters via sinusoidal embedding + AdaLN or FiLM on decoder blocks. **Record which parameters are AdaLN/timestep-related** — §7.4 needs to freeze exactly these.
- Inference: Euler integration, `N = 10` steps (config). Deterministic given a seed.



### 4.4 Execution

Open-loop execution of the first `k = 8` actions of each chunk, then replan. No temporal ensembling in the main experiments — it smooths over failures and confounds the forgetting signal. Implement it, keep it off by default.

### 4.5 Layer registry

`policy.projectable_layers()` returns an ordered dict of `name -> nn.Linear` covering every linear layer eligible for subspace analysis and projection (trunk attention Q/K/V/O, trunk MLPs, decoder attention/MLPs, action in/out projections). This registry is the single source of truth used by hooks, SVD, projection, and plotting. Every analysis and every method iterates it in the same order.

---



## 5. Curricula

Defined declaratively in `configs/curriculum/`. A curriculum is an ordered list of `(task_key, dataset_path, n_demos)`.

**Stage A runs two 4-task sequences**, because the correlated-vs-heterogeneous contrast is exactly what RQ3 and Outcome D are about, and in sim it costs almost nothing:

- `seq_hetero`: one task each from SPATIAL → OBJECT → GOAL → LONG. Expect large forgetting, low gradient overlap.
- `seq_correlated`: four tasks from within OBJECT (same motion, different objects). Expect forward transfer and high `c_l` — this is the regime where hard projection should fail.

Also run **reverse order** for `seq_hetero` (LIBERO showed ordering matters) and 3 seeds each. References: independent single-task policies, and joint multi-task training on the union.

Stage B (`agilex_short`): 3 tasks, single sequence, 2 seeds max. Include at least one bimanual-coordination task, since that is the part LIBERO cannot cover.

---



## 6. Continual-learning methods

Single interface. Every method is a `ContinualMethod` and the sequential runner is method-agnostic.

```python
class ContinualMethod(Protocol):
    def on_task_start(self, task_idx, policy, dataset) -> None: ...
    def modify_loss(self, loss, batch, policy) -> Tensor: ...        # EWC, ConSFT
    def modify_gradients(self, policy, batch_meta) -> None: ...      # GPM, SGP, s-binned
    def build_batch(self, dataset, task_idx) -> Batch: ...           # replay mixing
    def on_task_end(self, task_idx, policy, dataset) -> None: ...    # basis/Fisher update
    def state_dict(self) -> dict: ...
```

Implement in this order:

1. `seq_ft` — plain sequential fine-tuning. Baseline B1 and the reference for every forgetting number.
2. `replay` — keep a fixed budget (e.g. 10 demos/task) mixed at a configurable ratio. Strong reference, violates exemplar-free. B3.
3. `lora` — rank-`r` adapters on registry layers; option to keep per-task adapters or accumulate one. B2.
4. `ewc` — diagonal Fisher, two-task probe only.
5. `gpm` — hard projection, `G' = G(I - M M^T)`. B5 candidate.
6. `consft` — confidence-scaled updates for flow-matching. Reimplement from the paper; if details are ambiguous, write the assumption in the docstring and in `docs/consft_notes.md`. B4.
7. `sgp` — soft projection, `G' = G_⊥ + α_l G_∥`, per-layer `α_l`.
8. `sgp_sbinned` — the thesis extension (§7.5). Only after Gate 4 passes.

---



## 7. Analysis machinery — the technical core

This is the part where a careless implementation quietly invalidates the thesis. Treat these as the highest-risk modules and cover them with tests.

### 7.1 Activation capture

Forward hooks on every registry layer capture the layer **input** `x ∈ R^{d_l}`. Requirements:

- Tag every captured sample with the `s` value used in that forward pass, the task, and the token position type.
- Subsample tokens (e.g. random 10–20 per sample) to keep `R_l` manageable; fix the subsampling seed.
- Build `R_l = [x_1 ... x_N] ∈ R^{d_l × N}` with `N ≳ 10 · d_l` where memory allows; log the actual `N/d_l` ratio per layer — SVD bases from too few samples are garbage and this is easy to miss.
- Hooks must be removable and must not be active during evaluation rollouts.



### 7.2 Subspace bases and occupancy

- `R_l = U Σ V^T`; keep `k_l` = smallest k with `Σ_{i≤k} σ_i² ≥ ε · Σ_i σ_i²`, default `ε = 0.95` (sweep `ε ∈ {0.90, 0.95, 0.99}`).
- `M_l = U[:, :k_l]`, `ρ_l = k_l / d_l`.
- Multi-task accumulation uses GPM's incremental rule: project new activations onto the orthogonal complement of the existing basis, SVD the residual, append directions passing the residual-energy criterion. **Do not** concatenate raw activations and re-SVD from scratch.
- Persist bases to disk per `(run_id, task_idx, layer)` with the energy threshold, `k_l`, `d_l`, `N`, and singular-value spectrum. Bases are artifacts, recomputing them must be unnecessary.



### 7.3 Gradient interference and projected updates

For layer `l` with gradient `G_l`:

```
G_∥ = G_l M_l M_l^T
G_⊥ = G_l - G_∥
c_l = ||G_∥||_F / ||G_l||_F
```

Note this normalizes by the **full** gradient norm, so `c_l ∈ [0,1]`. Fix this convention everywhere; the proposal's §13 formula is ambiguous on the denominator.

`ProjectedOptimizer` wraps the real optimizer and applies `modify_gradients` after `backward()` and before `step()`. It must:

- handle weight matrices with the correct orientation (`nn.Linear.weight` is `(d_out, d_in)`; the projection acts on the **input** dimension — get this wrong and the whole method is silently broken; unit-test it);
- log `c_l` per layer per step at a configurable interval;
- assert that no parameter excluded by §7.4 receives a projected update.



### 7.4 Projection parameter protocol

In projection experiments (`gpm`, `sgp`, `sgp_sbinned`), freeze: LayerNorm weights and biases, all linear biases, AdaLN/FiLM modulation parameters, and `s` embeddings. Otherwise forgetting leaks through unprotected parameters and the comparison is meaningless. Implement as an explicit allowlist derived from the layer registry, asserted at train start, and logged. Other baselines train normally — record the difference in the results table.

### 7.5 Flow-time conditioning (`s`-binning) — the novel piece

Bins: `s ∈ [0,0.25), [0.25,0.5), [0.5,0.75), [0.75,1.0]`.

**Analysis:** per-bin bases `M_{l,b}` → `ρ_l(s)`, and per-bin gradient decomposition → `c_l(s)`. Also report pairwise principal angles between `M_{l,b}` across bins — if the bins span the same subspace, the whole hypothesis dies, and that is a clean, fast, publishable-either-way check. Run it early.

**Method:** the difficulty is that a normal batch mixes `s` values, so the gradient cannot be attributed to a bin. Implementation:

1. Split each batch into `B` microbatches, one per `s` bin, with `s` sampled inside the bin.
2. Backward each microbatch separately, capture per-bin gradients `G_{l,b}`.
3. Project each with its own basis: `G'_{l,b} = G_⊥{l,b} + α_{l,b} G_∥{l,b}`.
4. Sum over bins, then `step()`.

Cost: `B×` backward passes. Budget for it; use gradient accumulation and keep `B = 4`. Verify equivalence: with `α = 1` everywhere, the `s`-binned path must reproduce standard training loss curves to within noise. Write that as a test.

---



## 8. Evaluation



### 8.1 Protocol

- LIBERO: 50 rollouts per `(method, seed, stage, task)` cell, fixed initial-state set shared across all methods. The same 50 init states, every time — otherwise the variance swamps the effect.
- 4-task sequence → 10 cells per run (stages 1..4 evaluated on tasks seen so far); with 3 seeds and 2 orderings this is a few thousand sim rollouts per method, which is fine.
- Stage B hardware: 20 rollouts per cell, scripted reset procedure, logged initial-condition photos, and a human-annotated failure taxonomy.



### 8.2 Metrics

`metrics.py` implements, from a single retention matrix `R[i][j]` (success on task `j` after training stage `i`):

- `F_1 = R[1][1] - R[2][1]` (and the general per-task forgetting)
- Negative backward transfer (NBT), forward transfer (FWT), AUC — use CLARE's definitions and cite them in the docstring so the numbers are comparable.
- Bootstrap 95% CIs over rollouts. Every reported success rate carries a CI. No bare percentages anywhere, including in plots.
- System metrics: trainable params, stored memory (bases + buffers, in MB), wall-clock train time, inference latency, integration time.



### 8.3 Determinism

Rollout seeding is derived from `(run_id, task, episode_idx)` and independent of method — the same episode index means the same initial state across methods. Assert this in a test.

---



## 9. Decision gates

Implement each as a script that reads `results/` and prints a verdict. The thesis narrative depends on these being run and recorded, not on their outcome.


| Gate | When    | Question                                     | Threshold                                                                                                |
| ---- | ------- | -------------------------------------------- | -------------------------------------------------------------------------------------------------------- |
| 0    | Month 1 | Is single-task success adequate?             | ≥ 80% on each chosen LIBERO task, else fix the policy before anything else                               |
| 1    | Month 2 | Does forgetting exist?                       | `F_1 ≥ 15 pp` on at least one sequence, else change curriculum                                           |
| 2    | Month 2 | Is projection geometrically plausible?       | `ρ_l` after T1 not already ≈ 1 on most registry layers                                                   |
| 3    | Month 3 | Does the new task need protected directions? | `c_l` after T2; `c_l ≈ 1` ⇒ hard projection incompatible with plasticity                                 |
| 4    | Month 3 | Is `s`-conditioning justified?               | Reproducible variation in `ρ_l(s)`, `c_l(s)` and non-trivial principal angles across bins, over ≥3 seeds |


Gate 4 failing is an acceptable, reportable outcome. Do not implement `sgp_sbinned` for the main table if it fails — report the null result and ship the characterization study.

---



## 10. Build order

Work in this order and do not skip ahead. Each step ends with a green test and a logged run.

1. Data layer + LIBERO adapter + chunked dataset. Test: reconstruct a demo, replay its actions in the env, confirm success.
2. Policy + flow-matching training on one task. Test: overfit 5 demos to near-zero loss.
3. Rollout wrapper + `evaluate.py`. Gate 0.
4. Sequential runner + `seq_ft` + retention matrix + metrics + plots. Gate 1.
5. Hooks + SVD + `ρ_l`. Gate 2.
6. Gradient decomposition + `c_l`. Gate 3.
7. `replay`, `lora`, `ewc`.
8. `gpm`, `sgp` + `ProjectedOptimizer` (+ orientation test, §7.3).
9. `flowtime.py`: `ρ_l(s)`, `c_l(s)`, principal angles. Gate 4.
10. `consft`.
11. `sgp_sbinned` if Gate 4 passes.
12. Full Stage A matrix: 2 sequences × 3 seeds × 5 methods.
13. AgileX adapter + `hw_env` + Stage B validation.

---



## 11. Cursor working rules

Save as `.cursor/rules/project.mdc`:

```
- This repo is a thesis experiment codebase. Correctness of measurement beats
  feature count. A silently wrong SVD or a transposed projection invalidates months
  of work; a missing convenience flag does not.
- Never change normalization statistics, evaluation initial states, or random-seed
  derivation to make a number look better. If a result looks wrong, say so.
- Every new analysis quantity needs: a unit test on a synthetic case with a known
  answer, and a shape/orientation assertion.
- No silent fallbacks. No try/except that swallows an error and returns a default.
  Fail loudly with the offending values in the message.
- Configs are the only place where experiment parameters live. No magic numbers in
  flowcl/. No hardcoded paths.
- Do not add dependencies without asking. Do not vendor code from papers without a
  docstring citing the source and listing which details were inferred.
- Prefer small, reviewable changes over large rewrites. When touching methods/ or
  analysis/, state in the PR description which experiments would need rerunning.
- Ask before implementing anything from Section 7.5; it is the thesis novelty and
  its details are decided by experimental results, not by convenience.
```

---



## 12. Out of scope

No billion-parameter VLA. No RL on hardware. No force/tactile policies. No autonomous task discovery. No new continual-learning theory. No requirement to beat replay — characterizing *why* directional constraints succeed or fail relative to magnitude control is the result.