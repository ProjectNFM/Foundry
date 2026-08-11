# Multi-Length Pretraining: Do Varied Temporal Scales Produce More Versatile Representations?

**Status:** Draft
**Date started:** 2026-08-11
**Parent experiment:** [Data Scaling Group](../02-data-scaling/README.md) (builds on B2 sweet spot)
**Follow-up experiments:** TBD
**Tags:** pretraining, mae, masked, sequence_length, multi_scale, cwt_cnn, dynamic_ch

## Background

The [data scaling experiments](../02-data-scaling/README.md) established B2
(Klinzing + Shirazi + Pavlov, ~37k ch·h) as the pretraining sweet spot. All runs
used fixed 2s windows, but the downstream tasks span vastly different temporal
scales: P300 detection uses 1s epochs, motor imagery uses ~4s trials, and sleep
staging uses 30s epochs. The group's
[open questions](../02-data-scaling/README.md#open-questions) flag this mismatch
explicitly: "Representations learned on 2s windows may miss longer-range temporal
structure critical for sleep staging (spindles, K-complexes, slow waves)."

Sleep staging in particular requires recognizing features at multiple timescales:
sleep spindles (~0.5–2s), K-complexes (~1s), and slow-wave oscillations (0.5–4 Hz,
spanning seconds). A model trained only on 2s windows can capture spindles and
K-complexes but never sees full slow-wave cycles or the broader temporal context
that human scorers use. Conversely, P300 signals are sub-second, so 2s windows
include substantial irrelevant context.

This experiment tests whether training on **mixed-length windows**
(1s, 2s, 5s, 10s, 30s simultaneously) produces representations that transfer
better across the full downstream task spectrum compared to fixed 2s pretraining.
Each batch randomly selects one window length; all samples within that batch share
the same duration.

The same structural changes as the masking sweep apply: intersubject validation
and 400k max steps with patience=10 early stopping.

## Question

Does exposing the model to varied temporal scales during pretraining — from 1s
snippets up to 30s epochs — produce representations that transfer better than
fixed-2s pretraining across downstream tasks with different temporal requirements?

## Hypothesis

Multi-length pretraining (S1) will outperform the fixed-2s baseline (M0) on
sleep staging (where 30s context captures slow waves), perform comparably on
motor imagery (where 2s is adequate), and improve on P300 (where 1s windows
are a closer match to the downstream epoch length). The improvement on sleep
staging will be larger for linear probes (representation quality) than for
finetuning (where the model can adapt).

Expected: S1 Kemp Sleep LP > M0 Kemp Sleep LP by at least +0.03 F1.

## Experiment

### Setup

- **Model:** POYO CWT-CNN + dynamic channel embeddings, session_emb disabled
- **Data:** B2 = Klinzing + Shirazi + Pavlov (`three_dataset_pretrain.yaml`)
- **Task:** MAE pretraining (masked reconstruction)
- **Training:** 400k max steps, batch_size=64, lr=1e-4, warmup 2k + cosine decay
  over 398k steps, bf16-mixed, intersubject validation, early stopping patience=10
- **Window lengths:** [1.0, 2.0, 5.0, 10.0, 30.0] — per-batch random selection
- **WandB:** `foundry_pretraining`, group `MASKING_SEQLEN`

### Code changes required

This run requires modifications to the data pipeline before it can launch. See the
[plan](../../.cursor/plans/masking_and_seqlen_experiments_c02e47cd.plan.md) for
full details:

1. **`VariableLengthBatchSampler`** (`foundry/data/samplers.py`) — batch sampler
   that randomly selects a window length per batch from the configured list.
2. **Tokenization** (`foundry/models/poyo_eeg.py`) — derive actual window duration
   from the data sample instead of using the fixed `self.sequence_length`.
3. **Fixed-count dynamic latent grid** (`foundry/models/poyo_eeg.py`) — keep the
   number of latent time bins constant (20) and scale the step size proportionally
   to actual duration, always producing 320 latents regardless of window length.
4. **`NeuralDataModule` wiring** (`foundry/data/datamodules/base.py`) — use
   `VariableLengthBatchSampler` when `window_lengths` is provided.

### Pretraining run

| Run | window_lengths | mask_ratio | block_size | Notes |
|-----|:-:|:---:|:---:|-------|
| S1 (multi-length) | [1, 2, 5, 10, 30] | 0.5 | 10 | All 5 window sizes, per-batch random selection |

Compared against M0 (baseline, fixed 2s) from the
[masking parameter sweep](./20260811-MS-masking-parameter-sweep.md).

### Launch command — Pretraining

```bash
# S1: Multi-length pretraining (requires code changes first)
uv run python main.py experiment=pretraining/poyo_masking_seqlen_sweep \
  data=openneuro/three_dataset_pretrain \
  data.window_lengths=[1.0,2.0,5.0,10.0,30.0] \
  hyperparameters.sequence_length=30.0 \
  run.name=pretrain_S1_multilength run.group=MASKING_SEQLEN -m
```

### Launch commands — Downstream evaluation

After pretraining, evaluate on 3 tasks × 2 modes × 3 folds = 18 runs:

```bash
# Kemp Sleep
uv run python main.py experiment=sleep_staging/kemp_finetune_from_data_scaling \
  run.pretrain_run_name=pretrain_S1_multilength run.pretrain_group=MASKING_SEQLEN -m
uv run python main.py experiment=sleep_staging/kemp_linear_probe_from_data_scaling \
  run.pretrain_run_name=pretrain_S1_multilength run.pretrain_group=MASKING_SEQLEN -m

# PhysioNet MI
uv run python main.py experiment=motor_imagery/physionet_finetune_from_data_scaling \
  run.pretrain_run_name=pretrain_S1_multilength run.pretrain_group=MASKING_SEQLEN -m
uv run python main.py experiment=motor_imagery/physionet_linear_probe_from_data_scaling \
  run.pretrain_run_name=pretrain_S1_multilength run.pretrain_group=MASKING_SEQLEN -m

# Brain Invaders P300
uv run python main.py experiment=p300/brain_invaders_finetune_from_data_scaling \
  run.pretrain_run_name=pretrain_S1_multilength run.pretrain_group=MASKING_SEQLEN -m
uv run python main.py experiment=p300/brain_invaders_linear_probe_from_data_scaling \
  run.pretrain_run_name=pretrain_S1_multilength run.pretrain_group=MASKING_SEQLEN -m
```

### Key config overrides

| Config | Purpose |
|--------|---------|
| `configs/experiment/pretraining/poyo_masking_seqlen_sweep.yaml` | Base pretraining config (intersubject, patience=10, 400k steps) |
| `configs/data/openneuro/three_dataset_pretrain.yaml` | B2 data (3 brainsets) |

### Key comparisons

- **S1 vs M0:** Same data, same masking, but mixed-length vs fixed-2s. Isolates the effect of temporal scale diversity.
- **S1 on Kemp Sleep vs S1 on P300:** Tests whether multi-length pretraining helps tasks at both ends of the temporal spectrum or only one.
- **S1 LP vs S1 FT:** Tests whether the diversity benefit is more visible in frozen representations (LP) or also in finetuning.

### Risks and mitigations

- **30s batches may OOM.** With CWT-CNN producing ~480 tokens/channel at 30s and
  batch_size=64 on L40S (48GB), memory may be tight. Mitigation: reduce batch_size
  for 30s batches via length-to-batchsize mapping, or use gradient checkpointing.
- **Val loss is not directly comparable across lengths.** Batches at different
  lengths have different reconstruction difficulties. Consider logging per-length
  val loss for diagnostics.

## Results

TBD

## Conclusions

TBD

## Notes for future experiments

TBD
