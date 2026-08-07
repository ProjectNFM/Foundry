# Paradigm Diversity: Does Visual Naming EEG Help Transfer?

**Status:** Draft
**Date started:** 2026-08-07
**Parent experiment:** [01-downstream-from-scratch-baselines](../01-downstream-from-scratch-baselines/README.md)
**Follow-up experiments:** TBD
**Tags:** pretraining, mae, masked, data_scaling, paradigm_diversity, kochi, visual_naming, cwt_cnn, dynamic_ch

## Background

The [volume scaling](../02-volume-scaling/20260807-MS-volume-scaling-pretrain.md) and
[diversity scaling](../03-diversity-scaling/20260807-MS-diversity-scaling-pretrain.md)
experiments test within-modality diversity (all EEG, varying datasets). This experiment
introduces **paradigm diversity** by adding Kochi Visual Naming (ds006914) — an iEEG
dataset from a fundamentally different cognitive paradigm (visual object naming vs
sleep/resting-state/working memory).

Kochi visual naming is a scalp EEG dataset registered as "ieeg" in the system due to its
variable electrode configurations (2-66ch, mean 50ch). It provides ~2,565 ch·h from 353
recordings across 110 subjects performing picture naming tasks.

## Question

Does adding a paradigm-diverse EEG source (visual naming) during pretraining improve
downstream transfer on sleep, motor imagery, and P300 tasks, even though the pretraining
paradigm differs substantially from all downstream tasks?

## Hypothesis

Adding Kochi to the pretraining mix will provide modest improvement (+0.5-1.5 F1) on
tasks that benefit from general neural representations (PhysioNet MI, Brain Invaders P300)
but may not help sleep staging, since sleep-specific temporal dynamics are absent in
visual naming data. D3 (Klinzing + Shirazi + Kochi) should outperform B1 (Klinzing +
Shirazi) if paradigm diversity adds value on top of dataset diversity.

## Experiment

### Setup

- **Model:** POYO CWT-CNN + dynamic channel embeddings, session_emb disabled
- **Data:** Three configurations isolating the effect of paradigm-diverse Kochi data
- **Training:** 200k steps, batch_size=64, lr=1e-4, bf16-mixed
- **Evaluation:** Kemp Sleep 5-class, PhysioNet MI binary, Brain Invaders P300 binary
  (finetuning + linear probe, 3-fold intersubject CV each)

### Pretraining runs

| Run | Data config | Datasets | ~Effective data | Disk size |
|-----|-------------|----------|----------------|-----------|
| D1 | `openneuro/kochi_only` | Kochi visual naming (353 recordings) | ~2,565 ch·h | ~69G |
| D2 | `openneuro/klinzing_kochi` | Klinzing + Kochi | ~22,049 ch·h | ~203G |
| D3 | `openneuro/klinzing_shirazi_kochi` | Klinzing + Shirazi + Kochi | ~37,211 ch·h | ~407G |

### Launch commands — Pretraining

```bash
# D1: Kochi visual naming only
uv run python main.py experiment=pretraining/poyo_data_scaling_base \
  data=openneuro/kochi_only \
  run.name=pretrain_D1_kochi_only \
  run.group=DATA_SCALING_PARADIGM -m

# D2: Klinzing + Kochi
uv run python main.py experiment=pretraining/poyo_data_scaling_base \
  data=openneuro/klinzing_kochi \
  run.name=pretrain_D2_klinzing_kochi \
  run.group=DATA_SCALING_PARADIGM -m

# D3: Klinzing + Shirazi + Kochi
uv run python main.py experiment=pretraining/poyo_data_scaling_base \
  data=openneuro/klinzing_shirazi_kochi \
  run.name=pretrain_D3_klinzing_shirazi_kochi \
  run.group=DATA_SCALING_PARADIGM -m
```

### Launch commands — Downstream evaluation

```bash
# --- D1 downstream ---
for TASK_CMD in \
  "experiment=sleep_staging/kemp_finetune_from_data_scaling" \
  "experiment=sleep_staging/kemp_linear_probe_from_data_scaling" \
  "experiment=motor_imagery/physionet_finetune_from_data_scaling" \
  "experiment=motor_imagery/physionet_linear_probe_from_data_scaling" \
  "experiment=p300/brain_invaders_finetune_from_data_scaling" \
  "experiment=p300/brain_invaders_linear_probe_from_data_scaling"; do
  uv run python main.py $TASK_CMD \
    run.pretrain_run_name=pretrain_D1_kochi_only \
    run.pretrain_group=DATA_SCALING_PARADIGM -m
done

# --- D2 downstream ---
for TASK_CMD in \
  "experiment=sleep_staging/kemp_finetune_from_data_scaling" \
  "experiment=sleep_staging/kemp_linear_probe_from_data_scaling" \
  "experiment=motor_imagery/physionet_finetune_from_data_scaling" \
  "experiment=motor_imagery/physionet_linear_probe_from_data_scaling" \
  "experiment=p300/brain_invaders_finetune_from_data_scaling" \
  "experiment=p300/brain_invaders_linear_probe_from_data_scaling"; do
  uv run python main.py $TASK_CMD \
    run.pretrain_run_name=pretrain_D2_klinzing_kochi \
    run.pretrain_group=DATA_SCALING_PARADIGM -m
done

# --- D3 downstream ---
for TASK_CMD in \
  "experiment=sleep_staging/kemp_finetune_from_data_scaling" \
  "experiment=sleep_staging/kemp_linear_probe_from_data_scaling" \
  "experiment=motor_imagery/physionet_finetune_from_data_scaling" \
  "experiment=motor_imagery/physionet_linear_probe_from_data_scaling" \
  "experiment=p300/brain_invaders_finetune_from_data_scaling" \
  "experiment=p300/brain_invaders_linear_probe_from_data_scaling"; do
  uv run python main.py $TASK_CMD \
    run.pretrain_run_name=pretrain_D3_klinzing_shirazi_kochi \
    run.pretrain_group=DATA_SCALING_PARADIGM -m
done
```

### Key comparisons

- **D1 alone:** Kochi-only pretraining on a small, paradigm-diverse source. Baseline for Kochi's intrinsic value.
- **D2 vs A2:** Both include Klinzing; D2 adds Kochi. If D2 > A2, paradigm diversity helps on top of sleep data.
- **D3 vs B1:** Both include Klinzing + Shirazi; D3 adds Kochi. Direct test of Kochi's marginal value in a multi-source setting.
- **D3 vs B2:** B2 adds Pavlov (same-domain WM), D3 adds Kochi (cross-paradigm). Compares paradigm diversity vs within-domain diversity.

## Results

TBD

## Conclusions

TBD

## Notes for future experiments

TBD
