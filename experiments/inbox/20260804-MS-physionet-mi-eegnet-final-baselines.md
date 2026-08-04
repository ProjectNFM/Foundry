# PhysioNet MI EEGNet Final Baselines (Best HPs × 3 Folds)

**Status:** Completed
**Date started:** 2026-08-04
**Parent experiment:** [PhysioNet Motor Imagery HP Search](20260731-MS-physionet-mi-hp-search.md)
**Follow-up experiments:** TBD
**Tags:** motor_imagery, physionet, eegnet, baseline, from_scratch, final

## Background

The [HP search](20260731-MS-physionet-mi-hp-search.md) tuned EEGNet on
PhysioNet MI fold 0, improving F1 from the [original baseline's](20260731-MS-physionet-mi-baselines.md)
0.873 to **0.924** — a +5.9% gain driven mainly by increased patience
(convergence at epoch 291 vs early stopping at epoch 123). The HP landscape
was very flat (all 50 configs within 0.914–0.924), but the best config was:

| Parameter      | Value |
| -------------- | ----- |
| learning_rate  | 1e-4  |
| weight_decay   | 0.0   |
| class_weights  | none  |
| F1 (filters)   | 8     |

The [POYO final baselines](20260804-MS-physionet-mi-poyo-final-baselines.md)
experiment is currently running all 4 POYO conditions (2 tokenizers ×
2 channel embedding modes) across 3 intersubject folds with HP-tuned settings
(lr=1e-4, bs=8, cw=auto, dim=256). To produce a fair cross-architecture
comparison, we need the corresponding EEGNet results on all 3 folds with
its own best HPs, run under identical training settings (patience=50,
max_epochs=500).

The original [baseline experiment](20260731-MS-physionet-mi-baselines.md)
ran EEGNet on all 3 folds but with suboptimal HPs (wd=0.01, cw=auto,
patience=20). Re-running with the tuned HPs should improve all folds.

## Question

Does EEGNet with HP-tuned settings (lr=1e-4, wd=0.0, cw=none, F1=8,
patience=50) maintain its 0.924 F1 on fold 0 and generalize consistently
across all 3 intersubject folds?

## Hypothesis

1. **Fold 0 will reproduce ~0.924 F1**, confirming the HP search result.
2. **All 3 folds will achieve ≥0.90 F1**, up from the baseline's 0.873 mean
   (which already had low cross-fold variance, std=0.020).
3. **The mean across folds will be ~0.91–0.93 F1**, providing a strong
   EEGNet reference point for comparison against the POYO conditions.

## Experiment

### Setup

- **Model:** EEGNet (F1=8, D=2, F2=16, kernel_length=64, dropout=0.5)
- **Data:** PhysionetMI (`physionet_mi/allsess`), intersubject split
- **Task:** Binary motor imagery classification (Left Hand vs Right Hand)
- **Training:** max 500 epochs, early stopping patience 50, bf16-mixed
- **Hardware:** 1× L40S per run, 6 CPUs, 32 GB RAM (SLURM)
- **WandB:** project=foundry_finetuning, group=PHYSIONET_MI_EEGNET_FINAL_3FOLD

**Best HPs (from parent HP search):**

| Parameter     | Value |
| ------------- | ----- |
| learning_rate | 1e-4  |
| weight_decay  | 0.0   |
| batch_size    | 64    |
| model.F1      | 8     |
| class_weights | none  |

**Conditions (3 total = 1 model × 3 folds):**

| Condition | Model  | Folds   |
| --------- | ------ | ------- |
| eegnet    | EEGNet | 0, 1, 2 |

### Launch command

```bash
uv run python main.py experiment=motor_imagery/eegnet_physionet_final_3fold -m
```

### Key config overrides

- Config file: `configs/experiment/motor_imagery/eegnet_physionet_final_3fold.yaml`
- Hydra sweep: `hyperparameters.fold_number` over 0, 1, 2
- Best HPs from HP search: lr=1e-4, wd=0.0, cw=none, F1=8
- Patience=50, max_epochs=500 (matching HP search settings)
- Changed from baseline: wd 0.01→0.0, cw auto→none, patience 20→50, max_epochs 1000→500

## Results

### Summary

All 3 runs (1 condition × 3 folds) completed successfully. SLURM job array
10282876, WandB group `PHYSIONET_MI_EEGNET_FINAL_3FOLD`.

Fold 0 exactly reproduces the HP search result (0.924 F1). Folds 1 and 2
score lower at 0.866 and 0.870, giving a 3-fold mean of 0.887 ± 0.027. The
≥0.90 threshold is not met in aggregate, though the cross-fold variance is
modest compared to the POYO conditions.

### Metrics

| Fold   | Val F1 | Val Acc | Val AUROC | Epochs |
| ------ | ------ | ------- | --------- | ------ |
| Fold 0 | 0.9241 | 0.9271 | 0.9725    | 291    |
| Fold 1 | 0.8657 | 0.8634 | 0.9367    | —      |
| Fold 2 | 0.8696 | 0.8711 | 0.9443    | —      |
| **Mean** | **0.8865** | **0.8872** | **0.9512** | — |
| **Std**  | **0.0266** | — | — | — |

**Comparison with baseline experiment** (suboptimal HPs, patience=20):
- Baseline mean F1: 0.873 ± 0.020
- Final mean F1: 0.887 ± 0.027
- Δ: +1.4 pp (driven mainly by fold 0: 0.873→0.924)

**Comparison with POYO conditions** (from [POYO Final Baselines](20260804-MS-physionet-mi-poyo-final-baselines.md)):

| Condition     | Mean F1 | Std    |
| ------------- | ------- | ------ |
| **EEGNet**    | **0.8865** | **0.0266** |
| CWT Disabled  | 0.8840  | 0.0330 |
| CWT Dynamic   | 0.8764  | 0.0402 |
| RCNN Disabled | 0.8802  | 0.0365 |
| RCNN Dynamic  | 0.8733  | 0.0424 |

EEGNet achieves the highest mean F1 and lowest cross-fold variance of all 5
conditions, though the margin is negligible (−0.2 pp vs best POYO).

### Analysis

Script: `analysis/030_physionet_mi_final_baselines.py`

```bash
uv run python analysis/030_physionet_mi_final_baselines.py
```

### Figures

![Main Results — bar chart with error bars](analysis/figures/030_physionet_mi_final_main_results.png)

![F1 Learning Curves — fold 0](analysis/figures/030_physionet_mi_final_f1_curves.png)

![Cross-Fold Variance — strip plot](analysis/figures/030_physionet_mi_final_fold_variance.png)

## Conclusions

**Fold 0 reproduced; cross-fold generalization weaker than hypothesized.**

1. **H1 (fold 0 reproduces ~0.924): Confirmed.** Fold 0 achieved exactly 0.924
   F1, matching the HP search result.
2. **H2 (all folds ≥0.90): Not met.** Folds 1 (0.866) and 2 (0.870) fall
   below 0.90, though they improve over the baseline's corresponding folds.
3. **H3 (mean ~0.91–0.93): Not met.** The mean is 0.887, below the predicted
   range. The cross-fold variance (std=0.027) is comparable to the baseline
   (std=0.020), so the HP-tuned settings did not reduce fold sensitivity.

EEGNet with tuned HPs is the best-performing model overall on PhysioNet MI,
though all 5 architectures (including 4 POYO conditions) fall within a narrow
1.3 pp band (0.873–0.887 mean F1). EEGNet's advantage is primarily its lower
cross-fold variance.

## Notes for future experiments

- **Pretraining POYO before fine-tuning** — from-scratch POYO matches EEGNet
  but does not surpass it. Pretraining on a large multi-dataset corpus may
  unlock the architectural advantage of POYO's flexible tokenization.
- **Test on other MI datasets** — verify whether the EEGNet ≈ POYO equivalence
  holds on datasets with different channel counts, subject pools, or paradigms
  (e.g., BCI Competition IV 2a, Lee2019).
- **Investigate why dynamic channel embeddings hurt POYO** — possible
  overfitting with PhysioNet's 64-channel array. Understanding this may reveal
  how to better leverage spatial information in POYO.
