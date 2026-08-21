# NeuralBench POYO-EEG Tokenizer Baselines

**Status:** Draft
**Date started:** 2026-08-21
**Parent experiment:** [NeuralBench Matched EEGNet — Three-Task Test Parity](20260821-MS-neuralbench-matched-test-parity.md)
**Follow-up experiments:** TBD
**Tags:** neuralbench, poyo_eeg, tokenizer, from_scratch, baseline, p300, motor_imagery, sleep_stage

## Background

The parent [matched EEGNet parity experiment](20260821-MS-neuralbench-matched-test-parity.md) fixes the NeuralBench v0.2.3 task, subject-split, seed, training, and best-checkpoint test-evaluation contracts for P300, Motor Imagery, and Sleep Stage. It is therefore the appropriate controlled comparator for an initial POYO-EEG baseline: POYO changes the model family, while the data and evaluation protocol remain fixed.

This is deliberately a from-scratch study. The older [Foundry downstream baseline group](../01-downstream-from-scratch-baselines/README.md) found that POYO generally matched rather than decisively exceeded EEGNet, and that tokenizer effects were task-dependent: CWT-CNN had only a small advantage over ResampleCNN on PhysioNet MI, while P300 transfer was especially sensitive to generalization. NeuralBench supplies a common, independently defined benchmark contract on which to measure whether that pattern persists.

The experiment varies only POYO's temporal tokenizer. It compares the parameter-matched per-channel CWT-CNN and ResampleCNN tokenizers while fixing the POYO backbone, channel/session embedding configuration, task data, splits, seeds, optimizer, schedule, stopping rule, and held-out test evaluation.

## Question

On the exact NeuralBench P300, Motor Imagery, and Sleep Stage tasks, subject splits, and seeds used by the matched EEGNet experiment, how do from-scratch POYO-EEG CWT-CNN and ResampleCNN tokenizers compare on held-out test performance, and how does each compare with the matched EEGNet baseline?

## Hypothesis

With all non-tokenizer choices fixed, CWT-CNN will achieve higher mean three-seed test balanced accuracy than ResampleCNN on at least two of the three tasks, with a practical advantage of at least 1 percentage point on one task. The tokenizer ranking may vary by task; this experiment establishes the from-scratch POYO baseline rather than assuming POYO will outperform matched EEGNet.

## Experiment

### Setup

- **Model:** From-scratch POYO-EEG, fixed at `embed_dim=256`, depth 4, 8 cross-/self-attention heads, dynamic channel embeddings, disabled session embeddings, and `channel_fusion=concat`.
- **Tokenizer conditions:** `per_channel_cwt_cnn` versus parameter-matched `per_channel_resample_cnn`; this is the sole independent variable.
- **Data and task contract:** NeuralBench v0.2.3 / NeuralSet subject splits, identical to the parent:
  - P300 / `Korczowski2014A`: 16 channels, 1.0 s epochs.
  - Motor Imagery / `Schalk2004Bci2000`: 64 channels, 4.0 s epochs.
  - Sleep Stage / `Kemp2000Analysis`: 2 channels, 30.0 s epochs.
- **Seeds:** 33, 34, and 35 for every task and tokenizer condition (18 runs total).
- **Training:** Mirror the parent’s non-architectural protocol: AdamW (`lr=1e-4`, `weight_decay=0.05`), OneCycleLR with cosine annealing and `pct_start=0.1` at step interval, batch size 64, FP32, gradient clipping 1.0, and a 40-epoch cap. Early-stopping patience is 10 for P300 and 5 for MI/Sleep.
- **Evaluation:** Evaluate the best-validation checkpoint on the NeuralBench held-out test split (`run.evaluate_test=true`).
- **WandB:** project `foundry-neuralbench`; groups `NB_P300_POYO_TOKENIZER_BASELINES`, `NB_MI_POYO_TOKENIZER_BASELINES`, and `NB_SLEEP_POYO_TOKENIZER_BASELINES`.

### Launch command

```bash
export FOUNDRY_SNAPSHOT_ROOT=/network/scratch/s/sobralm/foundry-launches

# Each command submits 2 tokenizer conditions × 3 seeds to the long partition.
uv run python main.py experiment=neuralbench/p300_poyo_tokenizer_baselines -m
uv run python main.py experiment=neuralbench/mi_poyo_tokenizer_baselines -m
uv run python main.py experiment=neuralbench/sleep_stage_poyo_tokenizer_baselines -m
```

### Key config overrides

The three POYO experiment YAMLs should compose the corresponding matched EEGNet task/data/trainer contract, then override only:

| Setting | Value |
|---|---|
| `model` | `poyo_eeg` |
| `model/tokenizer` | sweep: `per_channel_cwt_cnn`, `per_channel_resample_cnn` |
| `model.embed_dim`, `model.depth` | `256`, `4` |
| `model.channel_emb_mode`, session embedding | dynamic / disabled |
| `run.evaluate_test` | `true` |
| `seed` | sweep: `33,34,35` |
| `hydra.launcher.partition` | `long` |
| `hydra.launcher.gres` | RTX 8000 GPU, matching the parent’s compatible-GPU constraint |

## Results

### Summary

TBD

### Metrics

TBD

### Analysis

Use the WandB-backed analysis script after all 18 POYO runs and the parent EEGNet runs have completed:

```bash
uv run python analysis/20260821-MS-neuralbench-poyo-tokenizer-baselines_analysis.py
```

### Figures

TBD

## Conclusions

TBD

## Notes for future experiments

TBD
