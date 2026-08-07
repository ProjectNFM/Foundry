# Diversity Scaling: Does Adding More EEG Datasets Improve Transfer?

**Status:** Draft
**Date started:** 2026-08-07
**Parent experiment:** [01-downstream-from-scratch-baselines](../01-downstream-from-scratch-baselines/README.md)
**Follow-up experiments:** TBD
**Tags:** pretraining, mae, masked, data_scaling, diversity, cwt_cnn, dynamic_ch

## Background

The [from-scratch baselines](../01-downstream-from-scratch-baselines/README.md) established
reference downstream performance. The
[two-dataset pretrain experiment](../inbox/20260805-MS-two-dataset-pretrain-downstream-eval.md)
tests Klinzing + Shirazi pretraining. The [volume scaling experiment](../02-volume-scaling/20260807-MS-volume-scaling-pretrain.md)
isolates single-source volume effects.

This experiment measures the **marginal value of dataset diversity**: holding the model
fixed (CWT-CNN + dynamic channel embeddings), does adding a 3rd and 4th EEG dataset to
the pretraining mix improve downstream transfer beyond what the additional data volume
alone would explain?

## Question

Does adding more diverse EEG datasets during pretraining improve downstream transfer
beyond what the additional data volume provides?

## Hypothesis

Each additional dataset will provide diminishing but positive marginal benefit.
B2 (3 datasets) will outperform B1 (2 datasets) by ~1-2 F1 points on at least 2 of 3
downstream tasks. B3 (4 datasets) will provide a smaller additional gain over B2.
The diversity benefit will be most visible on tasks structurally different from the
pretraining data (PhysioNet MI, Brain Invaders P300).

## Experiment

### Setup

- **Model:** POYO CWT-CNN + dynamic channel embeddings, session_emb disabled
- **Data:** 2, 3, and 4-dataset pretraining mixes (progressively adding sources)
- **Training:** 200k steps, batch_size=64, lr=1e-4, bf16-mixed
- **Evaluation:** Kemp Sleep 5-class, PhysioNet MI binary, Brain Invaders P300 binary
  (finetuning + linear probe, 3-fold intersubject CV each)
- **WandB:** pretraining in `foundry_pretraining`, downstream in `foundry_finetuning`

### Pretraining runs

| Run | Data config | Datasets | #Sources | ~Effective data | Disk size |
|-----|-------------|----------|----------|----------------|-----------|
| B1 | `openneuro/two_dataset_pretrain` | Klinzing + Shirazi | 2 | ~34,647 ch·h | ~338G |
| B2 | `openneuro/three_dataset_pretrain` | Klinzing + Shirazi + Pavlov | 3 | ~37,134 ch·h | ~372G |
| B3 | `openneuro/four_dataset_pretrain` | Klinzing + Shirazi + Pavlov + Getzmann | 4 | ~48,001 ch·h | ~664G |

**Note:** B1 overlaps with the existing
[two-dataset pretrain experiment](../inbox/20260805-MS-two-dataset-pretrain-downstream-eval.md).
If that experiment's CWT-CNN dynamic run completes, reuse its checkpoint for B1 downstream
evaluation. Otherwise, re-run with the standardized 200k-step budget.

### Launch commands — Pretraining

```bash
# B1: Klinzing + Shirazi (2 datasets)
uv run python main.py experiment=pretraining/poyo_data_scaling_base \
  data=openneuro/two_dataset_pretrain \
  run.name=pretrain_B1_two_dataset \
  run.group=DATA_SCALING_DIVERSITY

# B2: Klinzing + Shirazi + Pavlov (3 datasets)
uv run python main.py experiment=pretraining/poyo_data_scaling_base \
  data=openneuro/three_dataset_pretrain \
  run.name=pretrain_B2_three_dataset \
  run.group=DATA_SCALING_DIVERSITY

# B3: Klinzing + Shirazi + Pavlov + Getzmann (4 datasets)
uv run python main.py experiment=pretraining/poyo_data_scaling_base \
  data=openneuro/four_dataset_pretrain \
  run.name=pretrain_B3_four_dataset \
  run.group=DATA_SCALING_DIVERSITY
```

### Launch commands — Downstream evaluation

After pretraining, evaluate each checkpoint. Each command launches 3 folds:

```bash
# --- B1 downstream ---
for TASK_CMD in \
  "experiment=sleep_staging/kemp_finetune_from_data_scaling" \
  "experiment=sleep_staging/kemp_linear_probe_from_data_scaling" \
  "experiment=motor_imagery/physionet_finetune_from_data_scaling" \
  "experiment=motor_imagery/physionet_linear_probe_from_data_scaling" \
  "experiment=p300/brain_invaders_finetune_from_data_scaling" \
  "experiment=p300/brain_invaders_linear_probe_from_data_scaling"; do
  uv run python main.py $TASK_CMD \
    run.pretrain_run_name=pretrain_B1_two_dataset \
    run.pretrain_group=DATA_SCALING_DIVERSITY -m
done

# --- B2 downstream ---
for TASK_CMD in \
  "experiment=sleep_staging/kemp_finetune_from_data_scaling" \
  "experiment=sleep_staging/kemp_linear_probe_from_data_scaling" \
  "experiment=motor_imagery/physionet_finetune_from_data_scaling" \
  "experiment=motor_imagery/physionet_linear_probe_from_data_scaling" \
  "experiment=p300/brain_invaders_finetune_from_data_scaling" \
  "experiment=p300/brain_invaders_linear_probe_from_data_scaling"; do
  uv run python main.py $TASK_CMD \
    run.pretrain_run_name=pretrain_B2_three_dataset \
    run.pretrain_group=DATA_SCALING_DIVERSITY -m
done

# --- B3 downstream ---
for TASK_CMD in \
  "experiment=sleep_staging/kemp_finetune_from_data_scaling" \
  "experiment=sleep_staging/kemp_linear_probe_from_data_scaling" \
  "experiment=motor_imagery/physionet_finetune_from_data_scaling" \
  "experiment=motor_imagery/physionet_linear_probe_from_data_scaling" \
  "experiment=p300/brain_invaders_finetune_from_data_scaling" \
  "experiment=p300/brain_invaders_linear_probe_from_data_scaling"; do
  uv run python main.py $TASK_CMD \
    run.pretrain_run_name=pretrain_B3_four_dataset \
    run.pretrain_group=DATA_SCALING_DIVERSITY -m
done
```

### Key comparisons

- **B1 → B2:** Adding Pavlov (19ch verbal WM, 156 subjects) to the mix. +2,488 ch·h. Tests marginal value of a low-density diverse source.
- **B2 → B3:** Adding Getzmann (64ch resting, 608 subjects) to the mix. +10,867 ch·h. Largest single-step volume + diversity increase.
- **B1 vs A2:** B1 adds Shirazi to Klinzing; if B1 > A2, diversity beats single-source volume.
- **B3 vs A3:** B3 has 4 datasets (~48k ch·h) vs A3's single Shirazi (~15k ch·h). Tests whether diversity at scale dominates.

### Staging note

B3 requires staging ~664G of data. Ensure SLURM_TMPDIR has sufficient space.
If staging fails, reduce `num_workers` or request a node with larger local storage.

## Results

TBD

## Conclusions

TBD

## Notes for future experiments

TBD
