# Maximum Data: All Available EEG Sources Combined

**Status:** Draft
**Date started:** 2026-08-07
**Parent experiment:** [03-diversity-scaling](../03-diversity-scaling/20260807-MS-diversity-scaling-pretrain.md)
**Follow-up experiments:** TBD
**Tags:** pretraining, mae, masked, data_scaling, maximum_data, all_datasets, cwt_cnn, dynamic_ch

## Background

The [diversity scaling](../03-diversity-scaling/20260807-MS-diversity-scaling-pretrain.md)
experiment tests up to 4 EEG datasets (B3), and the
[paradigm diversity](../05-paradigm-diversity/20260807-MS-paradigm-diversity-pretrain.md)
experiment tests adding Kochi visual naming to various mixes. This experiment combines
**all 5 available datasets** to determine if maximum data yields the best downstream
transfer, and whether the marginal return from adding Kochi on top of 4 datasets is
positive.

The combined dataset spans 5 paradigms (sleep, resting-state, working memory,
cognitive tasks, and visual naming), 3 sampling rates (256/500/1000 Hz), channel counts
from 2 to 129, and ~50,566 ch·h of effective data from 5,371 recordings across 1,138
subjects.

## Question

Does combining all available EEG and iEEG data provide the best downstream transfer,
and what is the marginal return of adding the 5th dataset (Kochi) on top of the 4-dataset
mix?

## Hypothesis

E1 (all 5 datasets) will achieve the best overall downstream performance across the
3 tasks, outperforming B3 (4 datasets) by a small margin (+0.5-1 F1). The Kochi addition
provides paradigm diversity not present in the other 4 sources. If E1 does NOT
outperform B3, it suggests diminishing returns from dataset diversity or potential
interference from paradigm-mismatched data.

## Experiment

### Setup

- **Model:** POYO CWT-CNN + dynamic channel embeddings, session_emb disabled
- **Data:** All 5 datasets combined
- **Training:** 200k steps, batch_size=64, lr=1e-4, bf16-mixed
- **Evaluation:** Kemp Sleep 5-class, PhysioNet MI binary, Brain Invaders P300 binary
  (finetuning + linear probe, 3-fold intersubject CV each)

### Pretraining run

| Run | Data config | Datasets | ~Effective data | Disk size |
|-----|-------------|----------|----------------|-----------|
| E1 | `openneuro/all_datasets` | Klinzing + Shirazi + Pavlov + Getzmann + Kochi | ~50,566 ch·h | ~733G |

### Staging feasibility

**Total dataset size: ~733G.** This is the critical constraint:
- Klinzing: 134G
- Shirazi: 204G
- Pavlov: 34G
- Getzmann: 292G
- Kochi: 69G

SLURM_TMPDIR on Mila L40S nodes typically provides ~800G-1TB of local NVMe storage.
Staging 733G is tight but should be feasible on most nodes. If staging fails:
1. Request a node with `--tmp` flag for larger local storage
2. Consider running from network storage with `data.root=./data/processed/` and
   `stage.skip=true` (slower but no staging needed)
3. As a fallback, compress archives with `stage.compress=true` for smaller transfers

**Test staging first** with a dry-run before launching the full training job.

### Launch commands — Pretraining

```bash
# E1: All 5 datasets (~733G staging required)
uv run python main.py experiment=pretraining/poyo_data_scaling_base \
  data=openneuro/all_datasets \
  run.name=pretrain_E1_all_datasets \
  run.group=DATA_SCALING_MAXDATA -m
```

### Launch commands — Downstream evaluation

```bash
# --- E1 downstream ---
for TASK_CMD in \
  "experiment=sleep_staging/kemp_finetune_from_data_scaling" \
  "experiment=sleep_staging/kemp_linear_probe_from_data_scaling" \
  "experiment=motor_imagery/physionet_finetune_from_data_scaling" \
  "experiment=motor_imagery/physionet_linear_probe_from_data_scaling" \
  "experiment=p300/brain_invaders_finetune_from_data_scaling" \
  "experiment=p300/brain_invaders_linear_probe_from_data_scaling"; do
  uv run python main.py $TASK_CMD \
    run.pretrain_run_name=pretrain_E1_all_datasets \
    run.pretrain_group=DATA_SCALING_MAXDATA -m
done
```

### Key comparisons

- **E1 vs B3:** B3 has 4 datasets (~48,001 ch·h), E1 adds Kochi (~2,565 ch·h). Tests marginal value of the 5th paradigm-diverse source.
- **E1 vs D3:** D3 has Klinzing + Shirazi + Kochi (~37,211 ch·h), E1 adds Pavlov + Getzmann. Tests whether more EEG diversity helps on top of paradigm diversity.
- **E1 vs all others:** E1 is the "kitchen sink" — if it doesn't win, it suggests interference effects or diminishing returns from heterogeneous pretraining.

## Results

TBD

## Conclusions

TBD

## Notes for future experiments

TBD
