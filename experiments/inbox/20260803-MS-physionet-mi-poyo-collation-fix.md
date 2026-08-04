# PhysioNet MI POYO Collation Fix + HP Tuning

**Status:** Completed
**Date started:** 2026-08-03
**Parent experiment:** [PhysioNet MI HP Search](20260731-MS-physionet-mi-hp-search.md)
**Follow-up experiments:** [PhysioNet MI POYO Final Baselines](20260804-MS-physionet-mi-poyo-final-baselines.md)
**Tags:** motor_imagery, physionet, poyo, cwt_cnn, bug_fix, hp_search

## Background

The [parent HP search](20260731-MS-physionet-mi-hp-search.md) found that all 12
POYO CWT-CNN runs crashed with `RuntimeError: Trying to resize storage that is
not resizable` before training began. EEGNet achieved 0.924 F1 on the same task.

Root cause analysis revealed that PhysioNet MI contains recordings at **two
different sampling rates** (128 Hz and 160 Hz). After length normalization,
`input_values` tensors have shapes `(65, 512)` vs `(65, 640)` depending on the
recording's native rate. The `POYOEEGModel.tokenize()` method returned these as
raw tensors, and PyTorch's default `torch.stack` collation failed because it
cannot stack tensors of different shapes.

The fix wraps `input_values` in `pad2d()` — the same pattern already used by
`MaskedPOYOEEGModel` and `BaselineEEGModel`. This tells the `torch_brain`
collation system to zero-pad the time dimension to `max_T` in each batch,
producing `(B, C, max_T)` tensors. The model's `input_seq_len` and
`input_sampling_rate` fields (already present per-sample) allow CWT to
correctly handle the padded signals.

The fix has been applied in `foundry/models/poyo_eeg.py` (import `pad2d`,
wrap `result["input_values"]` in `tokenize()`). All 19 existing tests pass.

## Question

With the `pad2d` collation fix applied, can POYO CWT-CNN train successfully on
PhysioNet MI and achieve competitive or superior F1 compared to tuned EEGNet
(0.924)?

## Hypothesis

1. **POYO will train without crashing** — the `pad2d` fix resolves the
   variable-length collation issue for all batch_size / embed_dim combinations.
2. **HP-tuned POYO CWT-CNN can reach ≥0.90 F1**, given its richer architecture
   (CWT time-frequency features + Perceiver IO cross-attention + dynamic
   channel embeddings).
3. **POYO may match or exceed EEGNet's 0.924 F1** at the best HP configuration,
   since CWT captures frequency-domain information that EEGNet's temporal
   convolutions may miss.

## Experiment

### Setup

- **Model:** POYO CWT-CNN (dynamic channel embedding only)
- **Data:** PhysionetMI (`physionet_mi/allsess`), intersubject split
- **Task:** Binary motor imagery classification (Left Hand vs Right Hand)
- **Fold:** 0 only (HP search phase; best configs re-run on all 3 folds later)
- **Hardware:** 1× L40S per run, 6 CPUs, 32 GB RAM (SLURM)
- **WandB:** project=foundry_finetuning, group=PHYSIONET_MI_HP_SEARCH_POYO
- **SLURM:** job array 10273554_[0-23] (L40S, 6 CPUs, 32 GB, 12h timeout)
- **Training:** max 500 epochs, early stopping patience 50

**Hyperparameter grid (same as parent, 24 jobs):**

| Parameter | Values |
|-----------|--------|
| learning_rate | 1e-3, 5e-4, 1e-4 |
| batch_size | 8, 16 |
| class_weights.mode | none, auto (smoothing=1.0 when auto) |
| model.embed_dim | 128, 256 |

Fixed: `model.depth=4`, `model.num_heads=8`, `model.channel_emb_mode=dynamic`,
`trainer.callbacks.early_stopping.patience=50`

### Launch command

```bash
# POYO CWT-CNN dynamic (24 jobs: 3 lr × 2 batch_size × 2 class_weights.mode × 2 embed_dim)
uv run python main.py experiment=motor_imagery/physionet_hp_search_poyo -m
```

### Key config overrides

- Config file: `configs/experiment/motor_imagery/physionet_hp_search_poyo.yaml`
- Code fix: `foundry/models/poyo_eeg.py` — `pad2d(input_values)` in `tokenize()`
- Same YAML as parent experiment (no config changes needed)

## Results

### Summary

All 24 POYO CWT-CNN runs trained successfully — the `pad2d` collation fix
completely resolves the variable-length tensor issue. No crashes or runtime
errors occurred across any batch_size / embed_dim combination.

11/24 configurations converged to meaningful F1 (>0.70). The remaining 13
got stuck at ~0.662 (majority-class baseline), all of which used lr=1e-3
or were dim=256 at lr=5e-4.

### Metrics (top 11 converged runs, sorted by val F1)

| Run | LR | BS | CW | Dim | Val F1 | Val AUROC | Val Acc | Epoch |
|-----|----|----|-----|-----|--------|-----------|---------|-------|
| lr0.0001_bs8_cw-auto_dim256 | 1e-4 | 8 | auto | 256 | **0.937** | 0.977 | 0.938 | 170 |
| lr0.0001_bs16_cw-auto_dim128 | 1e-4 | 16 | auto | 128 | 0.933 | 0.970 | 0.934 | 219 |
| lr0.0005_bs16_cw-none_dim128 | 5e-4 | 16 | none | 128 | 0.929 | 0.963 | 0.930 | 230 |
| lr0.0001_bs16_cw-none_dim128 | 1e-4 | 16 | none | 128 | 0.928 | 0.969 | 0.929 | 225 |
| lr0.0005_bs16_cw-auto_dim128 | 5e-4 | 16 | auto | 128 | 0.926 | 0.964 | 0.927 | 322 |
| lr0.0001_bs16_cw-none_dim256 | 1e-4 | 16 | none | 256 | 0.926 | 0.968 | 0.927 | 144 |
| lr0.0001_bs8_cw-none_dim128 | 1e-4 | 8 | none | 128 | 0.924 | 0.968 | 0.924 | 142 |
| lr0.0001_bs8_cw-none_dim256 | 1e-4 | 8 | none | 256 | 0.921 | 0.966 | 0.922 | 132 |
| lr0.0001_bs16_cw-auto_dim256 | 1e-4 | 16 | auto | 256 | 0.921 | 0.964 | 0.921 | 101 |
| lr0.0005_bs8_cw-auto_dim128 | 5e-4 | 8 | auto | 128 | 0.920 | 0.964 | 0.920 | 225 |
| lr0.0001_bs8_cw-auto_dim128 | 1e-4 | 8 | auto | 128 | 0.920 | 0.967 | 0.920 | 117 |

### Convergence analysis

| LR | Converged / Total |
|----|-------------------|
| 1e-4 | 8/8 (100%) |
| 5e-4 | 3/8 (38%) |
| 1e-3 | 0/8 (0%) |

| Embed Dim | Converged / Total |
|-----------|-------------------|
| 128 | 7/12 (58%) |
| 256 | 4/12 (33%) |

### Analysis

```bash
uv run python analysis/028_physionet_mi_hp_search_poyo.py
```

### Figures

![HP Heatmap](../analysis/figures/028_physionet_mi_poyo_hp_heatmap.png)
![LR Effect](../analysis/figures/028_physionet_mi_poyo_hp_lr_effect.png)

### WandB

- Project: [foundry_finetuning](https://wandb.ai/poyo-eeg/foundry_finetuning)
- Group: `PHYSIONET_MI_HP_SEARCH_POYO`
- Best run: [physionet_mi_hp_poyo_lr0.0001_bs8_cw-auto_dim256](https://wandb.ai/poyo-eeg/foundry_finetuning/runs/ogg6292o)

## Conclusions

All three hypotheses are **confirmed**:

1. **POYO trains without crashing** — the `pad2d` fix fully resolves the
   variable-length collation issue. All 24 configurations across different
   batch_size and embed_dim combinations ran to completion.

2. **HP-tuned POYO CWT-CNN reaches ≥0.90 F1** — 11/24 configs achieved
   val F1 between 0.920 and 0.937.

3. **POYO exceeds EEGNet's 0.924 F1** — the best configuration
   (lr=1e-4, bs=8, cw=auto, dim=256) achieved **0.937 F1**, surpassing
   tuned EEGNet by +1.4%.

Key findings:
- **Learning rate is critical**: lr=1e-4 gives 100% convergence; lr=1e-3
  gives 0%. The model requires a conservative LR.
- **dim=128 is more reliable** for convergence (58% vs 33%) but dim=256
  achieved the single best result.
- Among converged runs, performance is tightly clustered (0.920–0.937),
  suggesting the architecture is robust once training converges.

## Notes for future experiments

- **3-fold evaluation**: Re-run best config (lr=1e-4, bs=8, cw=auto, dim=256)
  on all 3 folds for proper comparison with EEGNet 3-fold results.
  Config: `experiment=motor_imagery/physionet_poyo_3fold`
