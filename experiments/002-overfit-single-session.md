# Single-Session Overfit

**Status:** Draft
**Date started:** 2026-07-08
**Parent experiment:** [001 - Single-Batch Overfit Sanity Check](001-overfit-single-batch.md)
**Follow-up experiments:** TBD

## Background

[Experiment 001](001-overfit-single-batch.md) confirmed that the MaskedPOYOEEG
model with ResampleCNN tokenizer can memorize a single batch. The train loss
dropped ~36% over 119 epochs, validating the architecture and training pipeline.

The next step is to verify the model can learn meaningful reconstruction from an
entire session's worth of data (many batches). This is a stronger test than
single-batch overfit: the model must generalize across different windows within
the same recording rather than memorizing one fixed input. If successful, it
gives us confidence that the model can actually learn the structure of EEG data
before we scale to multi-session pretraining.

## Question

Can the MaskedPOYOEEG model learn to reconstruct masked EEG tokens from a
single session, achieving steadily decreasing train loss and stable (not
diverging) validation loss on held-out windows from the same session?

## Hypothesis

With intrasession splitting on a single recording, the model should be able to
overfit the training windows while still achieving reasonable reconstruction on
validation windows from the same session. Unlike the single-batch experiment,
we expect validation loss to decrease or plateau (not diverge), since the model
will be exposed to enough data diversity within the session to learn local EEG
structure rather than memorizing specific mask patterns.

## Experiment

### Setup

- **Model:** `MaskedPOYOEEGModel`, embed_dim=256, depth=4, mask_ratio=0.5, tokenizer=`per_channel_resample_cnn`
- **Data:** `OpenNeuroMultiBrainset` with a single brainset (`klinzing_sleep_ds005555`), filtered to a single recording via `recording_ids`, intrasession split
- **Task:** `masked_reconstruction` (MAE, MSE loss)
- **Training:** max_epochs=200, LR=1e-3 (same elevated LR from experiment 001), bf16-mixed, no early stopping, no `limit_train_batches` (use all batches from the session)
- **Hardware:** 1x GPU (Mila cluster)
- **WandB:** project=`foundry_pretraining`, group=`DEBUGGING`, run name and ID TBD after launch



### Launch command

```bash
uv run python main.py \
  experiment=pretraining/poyo_multi_dataset_pretrain \
  logger=wandb \
  run.name=OVERFIT_SINGLE_SESSION \
  run.group=DEBUGGING \
  data.dataset_kwargs.brainsets='[klinzing_sleep_ds005555]' \
  hyperparameters.learning_rate=0.001 \
  ~trainer.callbacks.early_stopping
```



### Key config overrides


| Override                                                  | Purpose                                               |
| --------------------------------------------------------- | ----------------------------------------------------- |
| `data.dataset_kwargs.brainsets=[klinzing_sleep_ds005555]` | Use only one brainset                                 |
| `hyperparameters.learning_rate=0.001`                     | Same elevated LR from experiment 001                  |
| `~trainer.callbacks.early_stopping`                       | Disable early stopping to observe full training curve |




### Notes on session selection

The default `klinzing_sleep_ds005555` dataset includes multiple recordings. For
a true single-session test, we may additionally need to filter to one recording
via `data.recording_ids`. The exact recording ID should be confirmed by
inspecting the dataset, e.g.:

```python
from foundry.data.datasets.openneuro import OpenNeuroMultiBrainset
ds = OpenNeuroMultiBrainset(root="./data/processed/", brainsets=["klinzing_sleep_ds005555"], split_type="intrasession")
print(ds.recording_ids[:5])
```

If filtering to a single recording is needed, add:

```bash
  data.recording_ids.klinzing_sleep_ds005555='[<recording_id>]'
```



## Results

TBD — experiment not yet run.

### Analysis

**Analysis script:** `analysis/002_overfit_single_session.py` (to be created after run completes)

## Conclusions

TBD

## Notes for future experiments

- If this succeeds, the natural next step is multi-session pretraining on the
full dataset.
- Compare learning dynamics: how many epochs does it take to plateau on a full
session vs one batch? This tells us about the data efficiency of the model.
- Monitor reconstruction visualizations on WandB to qualitatively assess whether
the model is learning meaningful signal structure.
- If val loss diverges even on a single session, investigate whether the
mask_ratio (0.5) is too aggressive or the model capacity is too low for the
channel count.
- Consider trying the default LR (1e-4) as well to see if the elevated LR
causes instability with more data.

