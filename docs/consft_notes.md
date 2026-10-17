# ConSFT: implementation notes (written before any calibration)

ConSFT is the fourth required baseline (`thesis_plan.md`, Workstream B). This page fixes how we
reimplement it, where we follow the paper and where the released code differs, and why. The method is
`flowcl/methods/consft.py`. Its bracket rule is `configs/analysis/baseline_selection_consft.yaml`.

**Sources** (checked on Sat 17 Oct 2026):
- Zhang, Zhai, Zhang, Huang, Zhang. "Preserving Foundational Capabilities in Flow-Matching VLAs
  through Conservative SFT", arXiv 2605.08879v2 (19 May 2026). Checked against both the PDF and the
  HTML.
- The released code: github.com/tyzhang2907/ConservativeSFT, `src/consft.py` (two commits).

## 1. What the paper specifies

- **Objective** (eq. 5, App. C.1): each sample's flow-matching loss is multiplied by a
  stop-gradient weight

  `ω = max(ω_min, exp(−κ · L_SFT / τ(t)))`, `J = sg[ω] · L_SFT`,

  "on a per-sample basis via a stop-gradient operator" (§5.1).
- **Temperature schedule** (eq. 11–12):

  `ρ(t) = (exp(−λ · min(t/T, 1)^c) − e^{−λ}) / (1 − e^{−λ})`,
  `τ(t) = min(τ_start / max(ρ(t), 1e-8), τ_end)`.

  τ rises from τ_start toward τ_end. With c > 1 it stays near τ_start for most of T and rises steeply
  near the end.
- **Constants** (Table 5): τ_start 0.003, τ_end 5.0, c 3.5, λ 0.8, κ 25, ω_min 0.001, T 2000.
- **What is tuned:** App. C.1 says λ, c, κ and ω_min "remain fixed" across every evaluation. Only
  τ_start and T "require task-specific tuning". The external review's description (per-sample
  weighting, stop-gradient, a schedule whose start and length need tuning) is therefore confirmed.
- **Training:** Adam at lr 2.5e-5, betas (0.9, 0.95), ε 1e-8, weight decay 1e-10, clip 1.0, batch 1024.
  T = 2000 steps is 2.05M samples.

## 2. Where the released code differs

The code implements the same formula: `_get_temperature` is eq. 11–12 exactly, and the weight is
computed under `torch.no_grad()` and reduced with `(weight * raw_loss).mean()`. Its defaults differ
from Table 5:

| Constant | Table 5 | Repo default |
|---|---|---|
| κ (`scale`) | 25 | 3 |
| τ_start | 0.003 | 0.2 |
| λ (`temperature_decay`) | 0.8 | 3 |
| c (`temperature_curve`) | 3.5 | 1.5 |
| ω_min | 0.001 | none (no floor) |
| τ_end | 5 | none: after T the weight is exactly 1 (plain SFT) |
| T | 2000 | 2000 |

Its docstring says the loss is "per token/sample"; the paper says per sample.

**Decision (user, 17 Oct): Table 5.** Those are the values behind the reported results. The repo's
defaults match no configuration the paper reports.

## 3. Decisions for our setting

1. **The per-sample loss.** `L_i = S_i / (N_i·D)`, where `S_i` is the sample's summed masked squared
   error against `A_1 − A_0` and `N_i` its valid steps. This is the paper's "per-sample" and the same
   quantity `masked_mse` averages.
2. **The normalization against our masked loss.** The stage objective is `Σ_i ω_i S_i / Σ_i N_i·D`.
   At ω ≡ 1 it equals seq_ft's `masked_mse`, our SFT loss, so "relaxing to unconstrained regression"
   means relaxing to exactly our seq_ft objective. It differs from the paper's plain sample mean
   `mean_i ω_i L_i` only through right-padded tail chunks (`N_i` < H). That is a small reweighting of
   the last chunks of each demo, and we accept it so that ω ≡ 1 reproduces seq_ft.
3. **T1 policy: ConSFT acts from T2 on.** At task 0, `modify_loss` returns the loss tensor
   untouched, so stage 0 is bitwise seq_ft's (EWC's pattern), and the calibration starts from seq_ft
   seed 0's `stage0.pt` (`thesis_plan.md` Workstream B). Reasons:
   - The paper fine-tunes a pretrained base, and our T1 is that base.
   - A from-scratch T1 has nothing to retain.
   - Its losses would put almost every weight at ω_min.
4. **The schedule restarts at every task.** Each task is a fine-tune of the previous policy. The
   step counter resets in `on_task_start` and advances in `after_step`, once per optimizer step;
   AMP-skipped steps are counted, like the code's `global_step`. τ_start is one fixed number for all
   tasks. It is calibrated once, on T1 → T2, and not recalibrated per task, which would be a new
   variant.
5. **Precision.** The weight is computed in fp32 with autocast off, under `no_grad` (the
   stop-gradient). The weighted loss is fp32 and backpropagates through the AMP activations.
6. **Our recipe stays shared.** ConSFT trains under seq_ft's recipe: AdamW at lr 1e-4, betas
   (0.9, 0.999), weight decay 1e-4, clip 1.0, batch 64, 30k steps, warmup and cosine, AMP. Every
   baseline shares the optimizer; the bracket tunes only τ_start and T, as the paper does. Our stage
   is 30k × 64 = 1.92M samples, close to the paper's 2000 × 1024 = 2.05M, so the bracket's T = 30000
   is the sample-matched counterpart of the paper's T.
7. **The calibration.** Our loss scale differs from π0's. The weight depends on κ·L/τ, and τ_start
   cannot transfer, so the bracket sets `τ_start = κ·q50/z`. Here q50 is the median initial
   per-sample loss on T2, measured before any arm, and z is the weight's exponent at that median.
   With τ_end fixed at 5, a small z gives τ_start ≥ τ_end and the arm runs at the cap throughout.
   The rule handles this explicitly (see its header).

## 4. Analysis notes for the thesis (no code changes)

These concern how to read the paper's argument and our results.

- **(a) Adam and loss scale.** AdamW cancels a loss scale only if that scale is constant across the
  gradient history, ε is negligible against `sqrt(v̂)`, and clipping does not interfere. ConSFT's
  scale changes with τ(t) and with the losses, so it enters through Adam's moment estimates. ε
  competes with `sqrt(v̂)` (the RMS of the weighted gradient history), not with the current ω·|g|.
  So under Adam the method acts mainly through the relative weights across samples and through how
  the overall scale changes over time. It does not act as a plain step-size reduction.
- **(b) Eq. 7 holds per sample, not per batch.** The FIM argument (eq. 6–7, App. B.2) treats one
  gradient `g` scaled by one scalar ω: "for any parameter gradient g and a scalar modulation weight
  ω". With unequal ω_i the batch gradient `Σ ω_i g_i` is not one scalar times `Σ g_i`. Reweighting
  can remove gradient cancellation between samples, so the ω² bound does not establish batch-level
  risk reduction, even under SGD.
- **(c) β2 = 0.999 (ours) against 0.95 (the paper's).** τ rises sharply in the last part of T
  (c = 3.5), so the weights can grow by up to about 1000× in a short window. A slow `v̂` (about 1000
  steps of memory at β2 = 0.999) can then inflate the effective step for a while; the paper's
  β2 = 0.95 adapts in about 20 steps. The method's logs (raw loss, mean ω, floor fraction)
  characterize the weighting transition. They do not directly measure optimizer-step amplification.
- **(d) Flow time.** The per-sample loss depends on the flow time `s` (uniform sampler), so part of
  the "confidence" is a reweighting across flow times rather than across demonstrations. The
  calibration and the logs report losses and weights per s-bin.
- **(e) Logged loss.** The trainer logs the weighted loss, as EWC's includes its penalty. The method
  also logs the raw loss, so training curves can be compared with seq_ft's.
- **(f) App. B.1.** It argues that a nonzero `L_behavior` only multiplies every weight by one
  constant and therefore "exerts no influence". That holds only without the ω_min clamp: a global
  factor changes which samples sit at the floor.
- **(g) The paper's limitation (§6)** that confidence weighting slows target convergence is what the
  selection rule's Object floor (seq_ft's stage-1 Object − 10 pp) guards against.

## 5. What is not implemented

- **Paper ablations:** the sparsity metric (App. A.3) and the PPO and LwF comparisons. The
  sparsity metric could be computed later from the checkpoints if wanted.
- **The repo's metrics:** they are logged, but the per-token variant is not offered.
