# PhysioNet MI POYO Collation Fix + HP Tuning

**Status:** Running
**Date started:** 2026-08-03
**Parent experiment:** [PhysioNet MI HP Search](20260731-MS-physionet-mi-hp-search.md)
**Follow-up experiments:** TBD
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

TBD

## Conclusions

TBD

## Notes for future experiments

TBD
