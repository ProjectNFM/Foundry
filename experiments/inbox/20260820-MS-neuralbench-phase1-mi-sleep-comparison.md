# NeuralBench Phase 1 — Motor Imagery & Sleep Stage EEGNet Comparison

**Status:** In progress (NeuralBench reference seed 33 complete; Foundry comparison pending)
**Date started:** 2026-08-20
**Parent experiment:** [P300 EEGNet Comparison](./20260820-MS-neuralbench-p300-eegnet-comparison.md)
**Follow-up experiments:** Phase 2 — Generic task onboarding
**Tags:** neuralbench, motor_imagery, sleep_stage, eegnet, comparison, phase1

## Background

The POC (P300 / Korczowski2014A) established the NeuralBench adapter
infrastructure and demonstrated a ~2.1 pp validation gap between Foundry
EEGNet and NeuralBench EEGNet, attributable to documented implementation
differences (dropout 0.5 vs 0.25, constant LR vs OneCycleLR).

Phase 1 expands to two additional NeuralBench tasks to validate that the
adapter generalizes beyond P300:

| Task | NeuralBench ID | Dataset | Classes | Epoch | Hz | Samples |
|------|---------------|---------|---------|-------|-----|---------|
| Motor Imagery | `motor_imagery` | Schalk2004Bci2000 (PhysioNet) | 4 | 4.0 s | 120 | 480 |
| Sleep Stage | `sleep_stage` | Kemp2000Analysis | 5 | 30.0 s | 120 | 3600 |

Both tasks use:
- Subject-based splits (60/20/20 with seed 33)
- CrossEntropyLoss with label_smoothing=0.1
- Compute class weights from training data
- Monitor: val/bal_acc (max), early stopping patience=5

## Question

Can the NeuralBench adapter infrastructure (designed for P300) generalize
to multi-class tasks with different epoch lengths and class structures — and
do Foundry EEGNet validation metrics remain within ~5 pp of NeuralBench's
reference EEGNet on these tasks?

## Hypothesis

1. The adapter will successfully load and train on both MI and Sleep Stage
   without task-specific code changes beyond configuration.
2. The validation balanced-accuracy gap between Foundry EEGNet and NeuralBench
   EEGNet will be ≤5 pp for each task, consistent with the P300 finding.
3. Training time differences will be primarily explained by dataset size and
   epoch length, not by adapter overhead.

## Experiment

### Setup — Motor Imagery

- **NeuralBench task:** `motor_imagery` / `Schalk2004Bci2000`
- **Classes:** 4 (imagery_bilateral_feet, imagery_bilateral_fist,
  imagery_left_fist, imagery_right_fist; verified from the LabelEncoder)
- **Model:** Foundry EEGNetEncoder (F1=8, D=2, F2=16, kernel=64, dropout=0.5,
  num_samples=480)
- **Training:** AdamW lr=1e-4, weight_decay=0.05, 40 epochs, patience=5
- **WandB:** project=foundry-neuralbench, group=NB_MI_EEGNET_COMPARISON
- **Seed:** 33 (single seed for initial comparison)

### Setup — Sleep Stage

- **NeuralBench task:** `sleep_stage` / `Kemp2000Analysis`
- **Classes:** 5 (Wake, N1, N2, N3, REM — from `SleepStage.stage` field)
- **Model:** Foundry EEGNetEncoder (F1=8, D=2, F2=16, kernel=64, dropout=0.5,
  num_samples=3600)
- **Training:** AdamW lr=1e-4, weight_decay=0.05, 40 epochs, patience=5
- **WandB:** project=foundry-neuralbench, group=NB_SLEEP_EEGNET_COMPARISON
- **Seed:** 33 (single seed for initial comparison)

### Launch commands

```bash
# Foundry EEGNet — Motor Imagery
uv run python main.py experiment=neuralbench/mi_eegnet_comparison

# Foundry EEGNet — Sleep Stage
uv run python main.py experiment=neuralbench/sleep_stage_eegnet_comparison

# NeuralBench reference — Motor Imagery
uv run neuralbench --grid --force --model eegnet --dataset schalk2004bci2000 eeg motor_imagery

# NeuralBench reference — Sleep Stage
uv run neuralbench --grid --force --model eegnet eeg sleep_stage
```

### Key config overrides (matching NeuralBench)

| Setting | Value | Rationale |
|---------|-------|-----------|
| `model.dropout_rate` | 0.5 | Foundry default; NeuralBench uses 0.25 |
| `hyperparameters.weight_decay` | 0.05 | Matches NeuralBench |
| `loss.label_smoothing` | 0.1 | Matches NeuralBench contract |
| `class_weights.mode` | auto | Matches `compute_class_weights=true` |
| `trainer.precision` | 32-true | Matches NeuralBench |
| `seed` | 33 | Single seed for this comparison |

### Known implementation differences (same as P300)

| Aspect | Foundry | NeuralBench | Impact |
|--------|---------|-------------|--------|
| Dropout rate | 0.5 | 0.25 | Higher regularization → slower convergence |
| LR schedule | Constant 1e-4 | OneCycleLR (cosine, pct_start=0.1) | Different optimization trajectory |
| BatchNorm momentum | PyTorch default (0.1) | braindecode default (0.01) | Slightly different running stats |
| Spatial max_norm | None | 1.0 (ConstrainedConv2d) | May limit spatial filter magnitude |

## Results

### Summary

NeuralBench EEGNet seed 33 completed locally on one L40S for both task--dataset
pairs. Both datasets passed the adapter/data verification before training:
Schalk2004Bci2000 loaded all 1,526 timelines with the expected four labels,
and Kemp2000Analysis loaded its 153 recordings with the expected five labels.
The scores are NeuralBench **test** metrics; Foundry validation results have
not yet been collected, so this is the baseline half of the comparison.

### Motor Imagery

| Side | Balanced Acc. | AUROC | F1 (macro) | Train time (s) | Epochs |
|------|-------------|-------|-----------|----------------|--------|
| Foundry EEGNet (val) | — | — | — | — | — |
| NeuralBench EEGNet (test, seed 33) | 0.5949 | 0.8134 | 0.5890 | 85.8 | best 29; stopped 34 |

### Sleep Stage

| Side | Balanced Acc. | AUROC | F1 (macro) | Train time (s) | Epochs |
|------|-------------|-------|-----------|----------------|--------|
| Foundry EEGNet (val) | — | — | — | — | — |
| NeuralBench EEGNet (test, seed 33) | 0.6849 | 0.9159 | 0.6179 | 869.1 | — |

### Timing Analysis

| Task | Foundry time | NeuralBench time | Ratio | Notes |
|------|-------------|-----------------|-------|-------|
| MI | — | 85.8 s | — | 3,092 parameters; early stopping selected epoch 29 |
| Sleep | — | 869.1 s | — | 10,101 parameters |

### Analysis

NeuralBench v0.2.3 local job artifacts (W&B logging was disabled) are stored
under `/network/scratch/s/sobralm/neuralbench-results/`:

- MI: `.../seed=33,task_name=motor_imagery,...-9e4eb200/job.pkl`
- Sleep: `.../seed=33,task_name=sleep_stage,...-3925ff96/job.pkl`

The direct comparison reference is the NeuralBench v0.2.3 three-seed
(33--35) distribution on each exact task--dataset pair. The published
NeuralBench headline MI protocol uses its default Stieger2021Continuous
dataset rather than Schalk2004Bci2000, so it is not a strict numeric reference
for the MI result above.

### Potential sources of major discrepancies

1. **Dropout rate (0.5 vs 0.25):** Foundry uses higher dropout which increases
   regularization, potentially hurting performance on smaller datasets but helping
   on larger ones.
2. **LR schedule:** OneCycleLR (NeuralBench) reaches higher peak LR then decays,
   often faster convergence. Foundry uses constant LR.
3. **Class weighting implementation:** Both compute weights from training labels
   but the exact formula and smoothing may differ.
4. **Epoch length / model capacity:** Sleep Stage has 3600 samples (30s @ 120 Hz)
   which is very large for EEGNet. The temporal conv kernel (64) spans only
   ~0.5s; this may work differently in both implementations.
5. **Data preprocessing:** Both use NeuralBench's preprocessing (0.1–75 Hz filter,
   50/60 Hz notch, RobustScaler, clamp ±20). Any difference would indicate an
   adapter bug, not an expected gap.
6. **Split assignment:** Both use NeuralBench's subject-split (seed=33). A
   difference here would be a critical bug.

## Conclusions

*To be filled after analysis.*

## Notes for future experiments

- Run 3-seed protocol (33, 34, 35) for statistical significance once
  single-seed parity is confirmed.
- Test with `dropout_rate=0.25` to quantify dropout contribution.
- Expand to Stieger2021 (MI default dataset with more subjects).
- Add POYO-EEG evaluation on both tasks.
