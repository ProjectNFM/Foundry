# Volume Scaling: Does More Data from a Single Source Improve Transfer?

**Status:** Draft
**Date started:** 2026-08-07
**Parent experiment:** [01-downstream-from-scratch-baselines](../01-downstream-from-scratch-baselines/README.md)
**Follow-up experiments:** TBD
**Tags:** pretraining, mae, masked, data_scaling, volume, cwt_cnn, dynamic_ch

## Background

The [from-scratch baselines](../01-downstream-from-scratch-baselines/README.md) established
reference performance for POYO CWT-CNN across three downstream EEG tasks (Kemp Sleep F1=0.730,
PhysioNet MI F1=0.884, Brain Invaders P300 F1=0.364). The
[two-dataset pretrain experiment](../inbox/20260805-MS-two-dataset-pretrain-downstream-eval.md)
is testing whether pretraining on Klinzing + Shirazi can beat these baselines.

This experiment isolates the **volume scaling** axis: holding the model fixed (CWT-CNN +
dynamic channel embeddings), how does increasing data volume from a single source affect
downstream transfer? It also compares a high-channel-count source (Shirazi, 129ch) against
a lower-channel-count source (Klinzing, ~10ch avg) to test whether channel richness
provides additional benefit.

## Question

Does pretraining on more data from a single EEG source monotonically improve downstream
transfer, and does a higher-channel-density source provide more benefit per recording?

## Hypothesis

Downstream F1 will increase from A1 → A2 (small → full Klinzing, ~10x volume increase),
and A3 (Shirazi, 129ch, ~1300 recordings) will outperform A2 despite being a different
domain, because its higher channel density provides richer spatial representations per
recording. Expected ordering: A1 < A2 < A3 on at least 2 of 3 downstream tasks.

## Experiment

### Setup

- **Model:** POYO CWT-CNN + dynamic channel embeddings, session_emb disabled
- **Data:** Three single-source pretraining runs at increasing scale
- **Training:** 200k steps, batch_size=64, lr=1e-4, bf16-mixed
- **Evaluation:** Kemp Sleep 5-class, PhysioNet MI binary, Brain Invaders P300 binary
  (finetuning + linear probe, 3-fold intersubject CV each)
- **WandB:** pretraining in `foundry_pretraining`, downstream in `foundry_finetuning`

### Pretraining runs

| Run | Data config | Datasets | ~Effective data | Disk size |
|-----|-------------|----------|----------------|-----------|
| A1 | `openneuro/klinzing_small` | Klinzing 28 recordings (14 subjects) | ~2,338 ch·h | ~15G |
| A2 | `openneuro/sleep_brainset` | Klinzing full 256 recordings (128 subjects) | ~19,484 ch·h | ~134G |
| A3 | `openneuro/shirazi_only` | Shirazi HBN 1,342 recordings (136 subjects) | ~15,163 ch·h | ~204G |

### Launch commands — Pretraining

Each run uses the base config with data/name/group overrides:

```bash
# A1: Klinzing small (28 recordings)
uv run python main.py experiment=pretraining/poyo_data_scaling_base \
  data=openneuro/klinzing_small \
  run.name=pretrain_A1_klinzing_small \
  run.group=DATA_SCALING_VOLUME

# A2: Klinzing full (256 recordings)
uv run python main.py experiment=pretraining/poyo_data_scaling_base \
  data=openneuro/sleep_brainset \
  run.name=pretrain_A2_klinzing_full \
  run.group=DATA_SCALING_VOLUME

# A3: Shirazi only (1,342 recordings, 129ch)
uv run python main.py experiment=pretraining/poyo_data_scaling_base \
  data=openneuro/shirazi_only \
  run.name=pretrain_A3_shirazi_only \
  run.group=DATA_SCALING_VOLUME
```

### Launch commands — Downstream evaluation

After pretraining completes, evaluate each checkpoint on all 3 tasks.
Each command launches 3 folds:

```bash
# --- A1 downstream ---
# Kemp Sleep finetuning (3 folds)
uv run python main.py experiment=sleep_staging/kemp_finetune_from_data_scaling \
  run.pretrain_run_name=pretrain_A1_klinzing_small \
  run.pretrain_group=DATA_SCALING_VOLUME -m

# Kemp Sleep linear probe (3 folds)
uv run python main.py experiment=sleep_staging/kemp_linear_probe_from_data_scaling \
  run.pretrain_run_name=pretrain_A1_klinzing_small \
  run.pretrain_group=DATA_SCALING_VOLUME -m

# PhysioNet MI finetuning (3 folds)
uv run python main.py experiment=motor_imagery/physionet_finetune_from_data_scaling \
  run.pretrain_run_name=pretrain_A1_klinzing_small \
  run.pretrain_group=DATA_SCALING_VOLUME -m

# PhysioNet MI linear probe (3 folds)
uv run python main.py experiment=motor_imagery/physionet_linear_probe_from_data_scaling \
  run.pretrain_run_name=pretrain_A1_klinzing_small \
  run.pretrain_group=DATA_SCALING_VOLUME -m

# Brain Invaders P300 finetuning (3 folds)
uv run python main.py experiment=p300/brain_invaders_finetune_from_data_scaling \
  run.pretrain_run_name=pretrain_A1_klinzing_small \
  run.pretrain_group=DATA_SCALING_VOLUME -m

# Brain Invaders P300 linear probe (3 folds)
uv run python main.py experiment=p300/brain_invaders_linear_probe_from_data_scaling \
  run.pretrain_run_name=pretrain_A1_klinzing_small \
  run.pretrain_group=DATA_SCALING_VOLUME -m

# --- A2 downstream (replace pretrain_run_name) ---
# Kemp Sleep finetuning
uv run python main.py experiment=sleep_staging/kemp_finetune_from_data_scaling \
  run.pretrain_run_name=pretrain_A2_klinzing_full \
  run.pretrain_group=DATA_SCALING_VOLUME -m

# Kemp Sleep linear probe
uv run python main.py experiment=sleep_staging/kemp_linear_probe_from_data_scaling \
  run.pretrain_run_name=pretrain_A2_klinzing_full \
  run.pretrain_group=DATA_SCALING_VOLUME -m

# PhysioNet MI finetuning
uv run python main.py experiment=motor_imagery/physionet_finetune_from_data_scaling \
  run.pretrain_run_name=pretrain_A2_klinzing_full \
  run.pretrain_group=DATA_SCALING_VOLUME -m

# PhysioNet MI linear probe
uv run python main.py experiment=motor_imagery/physionet_linear_probe_from_data_scaling \
  run.pretrain_run_name=pretrain_A2_klinzing_full \
  run.pretrain_group=DATA_SCALING_VOLUME -m

# Brain Invaders P300 finetuning
uv run python main.py experiment=p300/brain_invaders_finetune_from_data_scaling \
  run.pretrain_run_name=pretrain_A2_klinzing_full \
  run.pretrain_group=DATA_SCALING_VOLUME -m

# Brain Invaders P300 linear probe
uv run python main.py experiment=p300/brain_invaders_linear_probe_from_data_scaling \
  run.pretrain_run_name=pretrain_A2_klinzing_full \
  run.pretrain_group=DATA_SCALING_VOLUME -m

# --- A3 downstream (replace pretrain_run_name) ---
# Kemp Sleep finetuning
uv run python main.py experiment=sleep_staging/kemp_finetune_from_data_scaling \
  run.pretrain_run_name=pretrain_A3_shirazi_only \
  run.pretrain_group=DATA_SCALING_VOLUME -m

# Kemp Sleep linear probe
uv run python main.py experiment=sleep_staging/kemp_linear_probe_from_data_scaling \
  run.pretrain_run_name=pretrain_A3_shirazi_only \
  run.pretrain_group=DATA_SCALING_VOLUME -m

# PhysioNet MI finetuning
uv run python main.py experiment=motor_imagery/physionet_finetune_from_data_scaling \
  run.pretrain_run_name=pretrain_A3_shirazi_only \
  run.pretrain_group=DATA_SCALING_VOLUME -m

# PhysioNet MI linear probe
uv run python main.py experiment=motor_imagery/physionet_linear_probe_from_data_scaling \
  run.pretrain_run_name=pretrain_A3_shirazi_only \
  run.pretrain_group=DATA_SCALING_VOLUME -m

# Brain Invaders P300 finetuning
uv run python main.py experiment=p300/brain_invaders_finetune_from_data_scaling \
  run.pretrain_run_name=pretrain_A3_shirazi_only \
  run.pretrain_group=DATA_SCALING_VOLUME -m

# Brain Invaders P300 linear probe
uv run python main.py experiment=p300/brain_invaders_linear_probe_from_data_scaling \
  run.pretrain_run_name=pretrain_A3_shirazi_only \
  run.pretrain_group=DATA_SCALING_VOLUME -m
```

### Key comparisons

- **A1 → A2:** ~10x volume increase (same source, same channel density). Isolates pure volume effect.
- **A2 vs A3:** Similar effective data (~19k vs ~15k ch·h) but Shirazi has 129ch/recording vs Klinzing ~10ch. Tests channel richness effect.
- **A3 vs A2:** Shirazi has fewer recording-hours but more channels. If A3 > A2, spatial richness matters more than temporal volume.

### Config files used

| Config | Purpose |
|--------|---------|
| `configs/experiment/pretraining/poyo_data_scaling_base.yaml` | Base pretraining config |
| `configs/data/openneuro/klinzing_small.yaml` | A1 data (28 recordings) |
| `configs/data/openneuro/sleep_brainset.yaml` | A2 data (256 recordings) |
| `configs/data/openneuro/shirazi_only.yaml` | A3 data (1,342 recordings) |
| `configs/experiment/sleep_staging/kemp_finetune_from_data_scaling.yaml` | Downstream finetuning (Kemp) |
| `configs/experiment/sleep_staging/kemp_linear_probe_from_data_scaling.yaml` | Downstream linear probe (Kemp) |
| `configs/experiment/motor_imagery/physionet_finetune_from_data_scaling.yaml` | Downstream finetuning (PhysioNet MI) |
| `configs/experiment/motor_imagery/physionet_linear_probe_from_data_scaling.yaml` | Downstream linear probe (PhysioNet MI) |
| `configs/experiment/p300/brain_invaders_finetune_from_data_scaling.yaml` | Downstream finetuning (P300) |
| `configs/experiment/p300/brain_invaders_linear_probe_from_data_scaling.yaml` | Downstream linear probe (P300) |

## Results

TBD

## Conclusions

TBD

## Notes for future experiments

TBD
