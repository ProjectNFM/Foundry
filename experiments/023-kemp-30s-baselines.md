# KempSleep 30s-Epoch From-Scratch Baselines

**Status:** Draft
**Date started:** 2026-07-29
**Parent experiment:** [KempSleep Baselines and Finetuning: Dynamic Channel Embeddings × Tokenizer](../experiments/022-kemp-baselines-finetune-cwt-dynch.md)
**Follow-up experiments:** TBD

## Background

Previous Kemp sleep staging experiments (exp 006, 008, 020, 022) used 2s
context windows (`sequence_length: 2.0`), which is non-standard for sleep
staging. The clinical convention is to classify 30-second epochs. Using short
windows means each sample sees only a fraction of the sleep epoch, losing
important temporal context (e.g. spindle sequences, slow-wave buildup).

Additionally, prior experiments were run only on fold 0 of the intersubject
split. Multi-fold cross-validation (folds 0, 1, 2) is needed to establish
reliable performance estimates and confidence intervals.

This experiment also introduces a `session_pct` mechanism to create a 10%
training subset of KempSleep for fast iteration. The small config
(`kemp_sleep_edf/allsess_small`) keeps 10% of training recordings while
preserving all validation/test recordings, allowing quick debugging runs
before committing to full-dataset training.

The experiment establishes clean from-scratch baselines for all model
architectures, disentangling architectural effects from any pretraining
benefit.

By running each model on both the small (10% train) and full dataset, this
experiment also provides a coarse measure of how each architecture scales
with training data. Models that benefit most from additional data may be
better candidates for pretraining or for scaling to larger datasets, while
models that plateau early may have capacity or inductive bias limitations
that are worth understanding.

## Question

How do the different model architectures (EEGNet, POYO with CWT-CNN, POYO
with ResampleCNN) and channel embedding modes (disabled, dynamic) compare on
KempSleep 5-class sleep staging when trained from scratch with proper
30-second epochs and 3-fold intersubject cross-validation? Additionally,
which architectures benefit most from scaling from 10% to 100% of the
training data?

## Hypothesis

1. **30s epochs will improve F1** for all models compared to 2s windows,
   since the full epoch provides the temporal context needed for
   distinguishing sleep stages (especially N2 vs N3 and REM).
2. **EEGNet will be competitive** as a strong supervised baseline since it
   was designed specifically for EEG classification.
3. **Dynamic channel embeddings will outperform disabled** even from scratch,
   consistent with the +4.1 pp advantage seen in exp 020's random-init
   condition (albeit at 2s windows).
4. **CWT-CNN and ResampleCNN will perform similarly** from scratch since the
   architectural differences mainly manifest through pretraining benefit
   (exp 008 showed CWT-CNN's advantage was primarily in pretrained features).
5. **Performance variance across folds** will be moderate (1–3 pp F1),
   confirming that single-fold results from prior experiments were
   representative.
6. **POYO models will scale better** with training data than EEGNet. The
   transformer backbone has higher capacity and should benefit more from
   the 10× increase in training recordings. EEGNet, being a smaller
   convolutional model, may plateau earlier.

## Experiment

### Setup

- **Models:**
  - **EEGNet:** F1=8, D=2, F2=16, kernel_length=64, dropout=0.5, 2 channels,
    3000 samples (30s × 100 Hz)
  - **POYO (CWT-CNN):** POYOEEGModel, embed_dim=256, depth=4, 8 heads,
    dim_head=128, `per_channel_cwt_cnn` tokenizer
  - **POYO (ResampleCNN):** Same backbone, `per_channel_resample_cnn` tokenizer
- **Channel embeddings (POYO only):** `disabled` / `dynamic`
- **Session embeddings:** `disabled` for all POYO conditions
- **Data:** KempSleepEDF2013 (`kemp_sleep_edf/allsess`), intersubject split,
  all 3 folds (0, 1, 2)
- **Data (small):** `kemp_sleep_edf/allsess_small` — 10% of training
  recordings, 100% validation/test
- **Task:** 5-class sleep staging (W, N1, N2, N3, REM), class-weighted
  cross-entropy
- **Training:** sequence_length=30.0s, batch_size=32 (POYO) / 64 (EEGNet), lr=1e-4,
  weight_decay=0.01, max_epochs=1000, early stopping on val F1 (patience=20),
  bf16-mixed precision
- **Hardware:** 1× L40S per run, 6 CPUs, 32 GB RAM (SLURM)
- **WandB:** project=foundry_finetuning, group=KEMP_30S_BASELINES

**Conditions (30 total = 5 models × 3 folds × 2 dataset sizes):**

| Condition                           | Model      | Tokenizer      | channel_emb | Dataset | Folds |
| ----------------------------------- | ---------- | -------------- | ----------- | ------- | ----- |
| eegnet                              | EEGNet     | N/A            | N/A         | 10/100% | 0,1,2 |
| poyo-cwt-ch-disabled                | POYO       | CWT-CNN        | `disabled`  | 10/100% | 0,1,2 |
| poyo-cwt-ch-dynamic                 | POYO       | CWT-CNN        | `dynamic`   | 10/100% | 0,1,2 |
| poyo-rcnn-ch-disabled               | POYO       | ResampleCNN    | `disabled`  | 10/100% | 0,1,2 |
| poyo-rcnn-ch-dynamic                | POYO       | ResampleCNN    | `dynamic`   | 10/100% | 0,1,2 |

### Launch command

```bash
# --- Small dataset (10% train, fast iteration) ---

# POYO models (12 jobs: 2 tokenizers × 2 channel_emb × 3 folds)
uv run python main.py experiment=sleep_staging/kemp_30s_baselines \
    data=kemp_sleep_edf/allsess_small -m

# EEGNet (3 jobs: 3 folds)
uv run python main.py experiment=sleep_staging/eegnet_kemp_30s_baselines \
    data=kemp_sleep_edf/allsess_small -m

# --- Full dataset ---

# POYO models (12 jobs: 2 tokenizers × 2 channel_emb × 3 folds)
uv run python main.py experiment=sleep_staging/kemp_30s_baselines -m

# EEGNet (3 jobs: 3 folds)
uv run python main.py experiment=sleep_staging/eegnet_kemp_30s_baselines -m
```

### Key config overrides

- POYO experiment config:
  `configs/experiment/sleep_staging/kemp_30s_baselines.yaml`
- EEGNet experiment config:
  `configs/experiment/sleep_staging/eegnet_kemp_30s_baselines.yaml`
- Small data config:
  `configs/data/kemp_sleep_edf/allsess_small.yaml` —
  `session_pct.train: 0.1`, `session_pct.valid: 1.0`, `session_pct.test: 1.0`
- `data.split_type: intersubject` for all conditions (standard for sleep
  staging)
- `hyperparameters.sequence_length: 30.0` (proper 30s epochs)
- `model/session_emb: disabled` for all POYO conditions
- Hydra multirun sweeps `hyperparameters.fold_number` over 0, 1, 2

## Results

### Summary

TBD

### Metrics

TBD

### Analysis

TBD

**Analysis script:** `analysis/023_kemp_30s_baselines.py`

```bash
uv run python analysis/023_kemp_30s_baselines.py
```

### Figures

TBD

## Conclusions

TBD

## Notes for future experiments

- If 30s epochs show substantial improvement, revisit whether 2s results from
  prior experiments (exp 006, 008, 020, 022) need reinterpretation.
- Once baselines are established, compare against finetuned models (using
  pretrained checkpoints from exp 018/021) at 30s to quantify pretraining
  benefit on proper epochs.
- The `session_pct` mechanism can be reused for any dataset to create quick
  iteration subsets — consider creating small configs for OpenNeuro/Klinzing
  as well.
- If dynamic channel embedding advantage persists at 30s, it confirms the
  `RelativeChannelEncoder` provides a genuine architectural benefit beyond
  what longer context can provide.
- Consider adding DeepSleepNet or U-Sleep baselines for comparison against
  the sleep staging literature.
