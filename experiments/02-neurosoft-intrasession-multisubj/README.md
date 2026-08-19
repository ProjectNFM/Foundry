# NeuroSoft Intrasession Multisubject Decoding

**Experiments:** 10
**Date range:** 2026-07-29 to 2026-08-18
**Contributors:** LS

## Overarching Question

What species-specific training recipes maximize POYO-EEG validation
metrics for NeuroSoft 8-band auditory decoding under **intrasession
evaluation with multisubject pooling**, as a foundation before
co-training and cross-species transfer?

## Summary of Findings

This thread built species-separate recipes under a shared evaluation
protocol (NeuroSoft 8-band, intrasession, eventually multi-subject). An
HP search established that **monkeys outperform minipigs in absolute F1**
and that **optima are not shared** (tokenizer fusion, dropout, lr, WD,
clip). With those HPs frozen, **multi-subject** training was the
strongest paradigm and became the backbone for later ablations.

On that backbone, imbalance remedies (**CW smoothing**, **focal loss**)
and **token-rate** tweaks gave small or null mean gains once a sensible
CE+CW / 100 Hz baseline was in place. **Causal** splits were clearly
harder for minipigs but nearly neutral for monkeys (fold 0).
**Pure-frequency** labels helped monkeys slightly and not minipigs. The
biggest lever was **reducing capacity**: very small models for minipigs,
mid-size for monkeys. Stacking focal on that small model did not yield a
clean combined win. Session-level **EEGNet and GRU** still beat or
match the best multi-subject POYO when that model is scored as a
**fold-0 session mean** of `val_session/` max F1 / AUROC (minipig
EEGNet leads; monkey GRU is tied). Summing session confusion matrices
gives the same F1 ranking at species level (EEGNet 0.57 / GRU 0.59 vs
multi-subject POYO 0.39 / 0.54). The capacity run’s **true pooled** AUROC
(0.80 / 0.89) remains higher than n-weighted session AUROCs. EEGNet/GRU
session HPs were already tuned; remaining F1 gaps are more likely a
threshold / calibration issue (and pooled vs session-mean AUROC) than
an untuned baseline.

## Key Takeaways

### Best strategies by species (working recipes)

**Minipigs — what helped most**

- **Small capacity** is the main win vs default 256/4/8/8:
  `embed_dim=32`, `depth=2`, heads `6×6`, concat tokenizer with
  `channel_emb_fraction=1/2` (fold-0 peak F1 ≈ **0.394**).
- Tokenizer: **resample CNN (concat)**; avoid CWT.
- Training HPs: `atn_dropout=0.2`, `lr=2.75e-5`, `WD≈0.08`,
  `grad_clip=0.5`.
- Pooling: **multi-subject** ≫ single-subject / single-session.
- Imbalance: **CW smoothing ≈ 0.75**; focal adds little on top
  (especially once capacity is small).
- Tokens: **100 Hz**; causal splits cost ~**7%** F1 — prefer block for
  primary metrics unless causal is the scientific target.
- Pure-freq labels: **no gain** (and fewer trials) — keep multi-freq
  8-band.

**Monkeys — what helped most**

- **Mid capacity** beats default large: `embed_dim=64`, `depth=4`, heads
  `6×8`, **add** tokenizer (fold-0 peak F1 ≈ **0.538**).
- Tokenizer: **`per_channel_resample_cnn_add`**; avoid CWT.
- Training HPs: `atn_dropout=0.4`, `lr=2.5e-5`, stronger
  **WD (0.08–0.3; peak capacity used 0.3)**, `grad_clip=1.0`.
- Pooling: **multi-subject** strongest; QC pathological single sessions
  before trusting session means.
- Imbalance: **CW smoothing ≈ 1.0** (fold-mean); focal only modest vs CW
  on default capacity, and **hurts** vs small-cap CE when WD isn’t
  matched.
- Tokens: **100 Hz**; causal ≈ block on fold 0 (provisional).
- Pure-freq: **small fold-mean F1 gain (~+3%)** — optional if label
  design allows; watch trial count.

### Implications for mixed-species / co-training

- Do **not** assume one shared architecture or HP set: fusion mode
  (concat vs add), capacity, dropout, WD, and CW already diverge.
- Co-training should track **per-species** metrics (and possibly
  per-subject); minipigs will likely remain the harder / lower-F1
  species and may need capacity- or sampling-aware treatment so monkey
  gradients don’t dominate.
- A practical baseline for transfer studies: train with each species’
  **small/mid-cap CE + CW + 100 Hz + multisubject block** recipe, then
  compare joint / sequential / frozen-encoder protocols against those
  separate ceilings.
- Expect **asymmetric transfer** a priori (monkey→minipig vs reverse);
  causal difficulty and pure-freq sensitivity already differ by species.
- Prefer **block** splits for co-training scoreboards unless the
  question is temporal leakage; report causal as a stress test,
  especially for minipigs.

## Experiment Index

| # | Experiment | Hypothesis Verdict | Key Metric |
|---|------------|-------------------|------------|
| 1 | [HP search](./20260717-LS-intrasession-multisubj-hp.md) | Exploratory | Best F1 by species (resample CNN) |
| 2 | [Opt-HP paradigms](./20260727-LS-intrasession-opt-baselines.md) | Confirmed (multisubj best) | Multisubj F1 ≈ 0.36 / 0.50 |
| 3 | [Class-weight smoothing](./20260729-LS-class-weight-smoothing.md) | Small effect | Prefer s=0.75 / 1.0 (fold-mean) |
| 4 | [Sampling rate](./20260803-LS-sampling-rate.md) | Partial (100 Hz best mean) | 100 Hz > 50; ≳ 200 on mean |
| 5 | [Causal split](./20260805-LS-causal-split.md) | Species-dependent | Minipigs −7.4%; monkeys ≈0 |
| 6 | [Pure-frequency labels](./20260805-LS-pure-frequency-labels.md) | Partial (monkeys only) | Monkey +3% fold-mean F1 |
| 7 | [Model capacity](./20260805-LS-model-capacity.md) | Confirmed (best-config) | Best F1 0.394 / 0.538 |
| 8 | [Focal loss](./20260807-LS-focal-loss.md) | Partial / small vs CW | ΔF1 vs CW ≈ +0.001 / +0.010 |
| 9 | [Capacity + focal](./20260811-LS-capacity-focal.md) | Refuted (no combined win) | vs small-cap CE +0.001 / −0.022 |
| 10 | [EEGNet / GRU session baselines](./20260818-LS-singlesess-eegnet-gru-baselines.md) | Not supported on fold-0 session-mean F1 or AUROC | Fold-0 session EEGNet/GRU F1 0.58 / 0.57 vs multi-subject POYO 0.43 / 0.57; AUROC 0.77 / 0.58 vs 0.68 / 0.58 |

## Open Questions

- Which **co-training** strategies improve over the species-separate
  multisubject ceilings above?
- Does representation learning **transfer across species**, and under
  which protocols (zero-shot, few-shot, joint training, shared vs
  species-specific heads)?
- Can **per-class F1 thresholds**, score calibration (Platt / isotonic),
  or PR-AUC close the F1 gap for pooled POYO without losing its AUROC
  ranking advantage?
- Can **EEGNet / GRU** at single-subject or multi-subject pooling match
  or beat their strong session-level F1?
- How to further **reduce overfitting** on this NeuroSoft setup—e.g.
  **channel dropout / channel dropping**, temporal or amplitude jitter,
  mixup / cutmix-style trial mixing, stronger weight decay or dropout
  schedules, early stopping on a held-out subject, or frozen / lightly
  updated tokenizer layers—without erasing the small-capacity gains?
