# PhysioNet MI EEGNet Final Baselines (Best HPs × 3 Folds)

**Status:** Draft
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

TBD

## Conclusions

TBD

## Notes for future experiments

TBD
