# Within-Subject Split Control

**Status:** Draft
**Date started:** 2026-07-24
**Parent experiment:** [Session Embedding Ablation](../experiments/011-session-embedding-ablation.md)
**Follow-up experiments:** [Intersubject Pretraining — Session Embedding Generalization](../experiments/013-pretrain-intersubject-session-embeddings.md)

## Background

Experiments 009–011 observed a persistent train-val loss gap of ~0.94 (train
loss ~0.32, val loss ~1.25) from the very first epoch in inter-subject sleep
staging. Experiment 011 ruled out session embeddings as the cause — disabling
them had negligible effect on the gap or val F1.

The gap has been interpreted as "overfitting" or "memorization," but there is
an alternative explanation: **it is the natural consequence of inter-subject
distribution shift**. In an inter-subject split, train and val subjects are
entirely disjoint populations. EEG signals vary substantially across
individuals (amplitude scales, spectral profiles, electrode impedances,
morphology of sleep spindles and K-complexes). If this variability is large
relative to within-subject variability, a perfectly calibrated model would
still show a train-val gap because the loss on unseen subjects is genuinely
higher — not because the model is overfitting.

This experiment provides the critical control: train the identical model with
a within-subject (intrasession) split. In this setting, epochs from every
subject are randomly divided into train/val, so both sets share the same
subject distribution. Any remaining gap reflects only temporal
autocorrelation within sessions, not subject-level shift.

## Question

Is the train-val loss gap (~0.94) observed in inter-subject sleep staging
primarily caused by model overfitting, or does it reflect the inherent
distribution shift between subjects?

## Hypothesis

The train-val gap will **drop substantially** (to <0.2) with a
within-subject split, demonstrating that the gap observed in inter-subject
evaluation is dominated by genuine subject-level variability rather than
model overfitting. Concretely:

- If the intrasession gap is near-zero (~0.05–0.15): the model generalizes
well within subjects, and the inter-subject gap is entirely due to
distribution shift. No amount of regularization will close it — more data
or domain adaptation is needed.
- If the intrasession gap remains large (>0.5): the model is truly
overfitting to temporal patterns within recordings, and regularization or
augmentation could help even in the inter-subject setting.



## Experiment



### Setup

- **Model:** POYOEEGModel with CWT-CNN tokenizer (per_channel_cwt_cnn),
embed_dim=256, depth=4 — identical to experiments 009–011
- **Data:** KempSleepEDF2013, **intrasession** split (random epoch-level
assignment within each recording), fold 0
- **Task:** 5-class sleep staging (sleep_stage_5class), auto class weights
(smoothing=1.0)
- **Training:** lr=1e-4, weight_decay=0.01, patience=50 — identical to best
scratch config from exp 009
- **Hardware:** 1× L40S, 6 CPUs, 32 GB RAM, 6h wall time (SLURM)
- **WandB:** project=foundry_finetuning, group=KEMP_INTRASUBJECT_SPLIT

**Conditions:**


| Condition                | Split Type   | Init       | Group                            | Runs | Purpose                                    |
| ------------------------ | ------------ | ---------- | -------------------------------- | ---- | ------------------------------------------ |
| Intrasession (primary)   | intrasession | scratch    | KEMP_INTRASUBJECT_SPLIT          | 1    | Measure gap without subject shift          |
| Intersubject (control)   | intersubject | scratch    | KEMP_INTRASUBJECT_SPLIT_CONTROLS | 1    | Direct comparison under same config        |
| Intrasession (pretrained)| intrasession | pretrained | KEMP_INTRASUBJECT_SPLIT          | 1    | Pretrained model gap on intrasession split |


The intersubject control is optional — it replicates the exp 009 best scratch
config (lr=1e-4, wu=0, fold 0) under this experiment's config file to ensure
any differences are not due to config drift.

The pretrained intrasession condition finetunes from the CWT-CNN SSL checkpoint
(exp 005) on the same intrasession split to measure whether pretraining
changes the train-val gap when subject shift is removed.

### Launch command

```bash
# Intrasession split (primary experiment, submits to SLURM via submitit):
uv run python main.py experiment=sleep_staging/poyo_kemp_intrasubject_split -m

# Intersubject control (same config, inter-subject split for comparison):
uv run python main.py experiment=sleep_staging/poyo_kemp_intrasubject_split \
    data.split_type=intersubject \
    'run.name=kemp_intersubject_control_scratch' \
    run.group=KEMP_INTRASUBJECT_SPLIT_CONTROLS \
    'run.tags=[sleep_staging,poyo,kemp,intersubject,control,exp012]' -m

# Pretrained intrasession (finetune from SSL checkpoint, intrasession split):
uv run python main.py experiment=sleep_staging/poyo_kemp_intrasubject_split \
    'run.pretrained_checkpoint=${pretrained_checkpoints.per_channel_cwt_cnn}' \
    run.init_mode=pretrained \
    'run.name=kemp_intrasession_split_pretrained' \
    'run.tags=[sleep_staging,poyo,kemp,intrasession,pretrained,exp012]' -m
```



### Key config overrides

Uses new config
`configs/experiment/sleep_staging/poyo_kemp_intrasubject_split.yaml`.

Key difference from exp 009 config (`poyo_kemp_finetune_hp_search.yaml`):

- `data.split_type: intrasession` — the only intervention. Changes from
inter-subject to within-subject epoch-level random split. All subjects
contribute to both train and val sets.
- **No LR/warmup sweep** — uses the best config from exp 009 (lr=1e-4, wu=0)
directly to minimise compute.
- **Pretrained condition added** — finetunes from the CWT-CNN SSL checkpoint
(exp 005) on the intrasession split to compare the train-val gap with vs
without pretraining when subject shift is removed.



## Results



### Summary

TBD

### Metrics

TBD

### Analysis

TBD

**Analysis script:** `analysis/012_within_subject_split.py`

```bash
uv run python analysis/012_within_subject_split.py
```



### Figures

TBD

## Conclusions

TBD

## Notes for future experiments

- If the intrasession gap is near-zero, the path forward is **not**
regularization but rather: more subjects, subject-adaptive normalization
(e.g., z-score per subject), or domain adaptation / test-time adaptation
techniques that align val subjects to the training distribution.
- If the gap persists within subjects, investigate temporal autocorrelation:
consecutive sleep epochs are highly correlated (30s windows in the same
stage). A **shuffled epoch** variant that breaks temporal order could
isolate whether the model exploits temporal structure vs. stage features.
- Compare the absolute val F1 between splits: intrasession should be
substantially higher since the model "knows" each subject's patterns.
The difference quantifies the subject-level difficulty.

