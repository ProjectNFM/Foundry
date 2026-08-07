# Diversity vs Volume vs Channel Density Controls

**Status:** Draft
**Date started:** 2026-08-07
**Parent experiment:** [02-volume-scaling](../02-volume-scaling/20260807-MS-volume-scaling-pretrain.md)
**Follow-up experiments:** TBD
**Tags:** pretraining, mae, masked, data_scaling, controls, channel_density, volume_matched, cwt_cnn, dynamic_ch

## Background

The [volume scaling](../02-volume-scaling/20260807-MS-volume-scaling-pretrain.md) and
[diversity scaling](../03-diversity-scaling/20260807-MS-diversity-scaling-pretrain.md)
experiments vary both data volume and diversity simultaneously, making it hard to
attribute improvements to one factor.

This experiment introduces **controlled comparisons** that isolate individual factors:
- **C1 (headband-only):** Same source as A2 (Klinzing) but restricted to low-density
  headband recordings (~6ch). If A2 > C1 at similar effective data, channel density
  per recording matters.
- **C2 (volume-matched 3-dataset):** Three datasets subsampled to match A2's effective
  data (~19,484 ch·h). If C2 > A2, diversity provides an independent benefit beyond
  total data volume.

## Question

Does dataset diversity provide an independent benefit beyond total effective data
(recordings × channels), and does channel density per recording matter at constant
effective data?

## Hypothesis

C2 (3-source, volume-matched) will outperform A2 (single-source, same effective data)
by 1-3 F1 points, demonstrating that diversity provides independent benefit. C1
(headband-only, ~6ch) will underperform A2 (PSG+headband, ~10-16ch) despite similar
data volume, confirming that spatial richness per recording matters.

## Experiment

### Setup

- **Model:** POYO CWT-CNN + dynamic channel embeddings, session_emb disabled
- **Data:** Two controlled pretraining configurations
- **Training:** 200k steps, batch_size=64, lr=1e-4, bf16-mixed
- **Evaluation:** Kemp Sleep 5-class, PhysioNet MI binary, Brain Invaders P300 binary
  (finetuning + linear probe, 3-fold intersubject CV each)

### Pretraining runs

| Run | Data config | Design | ~Effective data | Key control |
|-----|-------------|--------|----------------|-------------|
| C1 | `openneuro/klinzing_headband_only` | 128 headband recordings (~6ch each) | ~7,292 ch·h | Low channel density control |
| C2 | `openneuro/three_dataset_volume_matched` | 3 sources subsampled to ~19,580 ch·h | ~19,580 ch·h | Diversity at constant volume |

**Volume matching for C2:** Klinzing 82 PSG recs (~8,584 ch·h) + Shirazi 455 recs
(~8,509 ch·h) + Pavlov all 156 recs (~2,488 ch·h) = ~19,580 ch·h total (100.5% of
A2's effective_data).

### Launch commands — Pretraining

```bash
# C1: Klinzing headband only (128 recordings, ~6ch)
uv run python main.py experiment=pretraining/poyo_data_scaling_base \
  data=openneuro/klinzing_headband_only \
  run.name=pretrain_C1_headband_only \
  run.group=DATA_SCALING_CONTROLS

# C2: 3-dataset volume-matched to A2
uv run python main.py experiment=pretraining/poyo_data_scaling_base \
  data=openneuro/three_dataset_volume_matched \
  run.name=pretrain_C2_volume_matched \
  run.group=DATA_SCALING_CONTROLS
```

### Launch commands — Downstream evaluation

```bash
# --- C1 downstream ---
for TASK_CMD in \
  "experiment=sleep_staging/kemp_finetune_from_data_scaling" \
  "experiment=sleep_staging/kemp_linear_probe_from_data_scaling" \
  "experiment=motor_imagery/physionet_finetune_from_data_scaling" \
  "experiment=motor_imagery/physionet_linear_probe_from_data_scaling" \
  "experiment=p300/brain_invaders_finetune_from_data_scaling" \
  "experiment=p300/brain_invaders_linear_probe_from_data_scaling"; do
  uv run python main.py $TASK_CMD \
    run.pretrain_run_name=pretrain_C1_headband_only \
    run.pretrain_group=DATA_SCALING_CONTROLS -m
done

# --- C2 downstream ---
for TASK_CMD in \
  "experiment=sleep_staging/kemp_finetune_from_data_scaling" \
  "experiment=sleep_staging/kemp_linear_probe_from_data_scaling" \
  "experiment=motor_imagery/physionet_finetune_from_data_scaling" \
  "experiment=motor_imagery/physionet_linear_probe_from_data_scaling" \
  "experiment=p300/brain_invaders_finetune_from_data_scaling" \
  "experiment=p300/brain_invaders_linear_probe_from_data_scaling"; do
  uv run python main.py $TASK_CMD \
    run.pretrain_run_name=pretrain_C2_volume_matched \
    run.pretrain_group=DATA_SCALING_CONTROLS -m
done
```

### Critical comparisons

- **C1 vs A2:** Same source (Klinzing), but A2 includes PSG (~16ch) + headband (~6ch)
  while C1 is headband-only (~6ch). C1 has ~7,292 ch·h vs A2's ~19,484 ch·h. If A2
  wins, it could be volume or channel density; C1's lower effective data is a confound.
- **C2 vs A2:** Same effective data (~19,580 ch·h), but C2 draws from 3 sources vs 1.
  If C2 wins, diversity provides an independent benefit at constant effective data.
- **C2 vs B2:** Same 3 sources, but C2 is subsampled (~19,580 ch·h) while B2 uses all
  data (~37,134 ch·h). Quantifies how much of B2's gain over A2 is diversity vs volume.

## Results

TBD

## Conclusions

TBD

## Notes for future experiments

TBD
