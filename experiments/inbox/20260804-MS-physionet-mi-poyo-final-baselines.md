# PhysioNet MI POYO Final Baselines (All Conditions × 3 Folds)

**Status:** Draft
**Date started:** 2026-08-04
**Parent experiment:** [PhysioNet MI POYO Collation Fix + HP Tuning](20260803-MS-physionet-mi-poyo-collation-fix.md)
**Follow-up experiments:** TBD
**Tags:** motor_imagery, physionet, poyo, cwt_cnn, resample_cnn, baseline, from_scratch, final

## Background

The [PhysioNet MI POYO Collation Fix](20260803-MS-physionet-mi-poyo-collation-fix.md)
experiment resolved the `pad2d` collation bug and HP-tuned POYO CWT-CNN dynamic on
fold 0, achieving **0.937 F1** — surpassing tuned EEGNet (0.924) by +1.4%.

However, that experiment only evaluated the CWT-CNN + dynamic channel embedding
condition. To produce a complete cross-architecture comparison mirroring the
[KempSleep 30s baselines](../_legacy/023-kemp-30s-baselines.md) (which ran all
4 POYO conditions × 3 folds), we need results for all combinations of:

- **Tokenizer:** CWT-CNN vs ResampleCNN
- **Channel embedding:** disabled vs dynamic

On Kemp sleep staging, CWT-CNN outperformed ResampleCNN by +3.1–3.5 pp F1, and
dynamic channel embeddings provided negligible benefit (+0.2–0.7 pp). Motor
imagery may show different patterns given its 64-channel high-density array
(vs 2 channels for Kemp) and the relevance of spatial ERD patterns.

The existing `[physionet_poyo_3fold.yaml](../../configs/experiment/motor_imagery/physionet_poyo_3fold.yaml)`
config already confirmed that CWT-CNN dynamic trains successfully on all 3 folds
with the best HPs (lr=1e-4, bs=8, cw=auto, dim=256). This experiment extends
coverage to the remaining 3 conditions using the same HPs.

## Question

How do the 4 POYO conditions (2 tokenizers × 2 channel embedding modes) compare
on PhysioNet Motor Imagery when evaluated with HP-tuned settings across all 3
intersubject folds?

## Hypothesis

1. **CWT-CNN will outperform ResampleCNN** on MI, consistent with the +3.1–3.5 pp
  advantage seen on KempSleep 30s. The wavelet decomposition captures mu (8–13 Hz)
   and beta (13–30 Hz) band ERD patterns more effectively.
2. **Dynamic channel embeddings may help more here than on Kemp**, given the
  64-channel array (vs 2 channels). The `RelativeChannelEncoder` can exploit
   topographic structure of sensorimotor ERD patterns. Expect +1–3 pp F1 over
   disabled (larger than Kemp's negligible +0.2 pp).
3. **All 4 conditions will achieve ≥0.90 F1**, given that even the HP search's
  weakest converged CWT-CNN config scored 0.920 on fold 0.
4. **CWT-CNN dynamic will reproduce ~0.937 F1** on fold 0, confirming the HP
  search result, with folds 1 and 2 expected within ±2 pp.



## Experiment



### Setup

- **Model:** POYO EEG (embed_dim=256, depth=4)
- **Data:** PhysionetMI (`physionet_mi/allsess`), intersubject split
- **Task:** Binary motor imagery classification (Left Hand vs Right Hand)
- **Training:** max 500 epochs, early stopping patience 50, bf16-mixed
- **Hardware:** 1× L40S per run, 6 CPUs, 32 GB RAM (SLURM)
- **WandB:** project=foundry_finetuning, group=PHYSIONET_MI_POYO_BASELINES_3FOLD

**Best HPs (from parent experiment, applied to all conditions):**


| Parameter          | Value                |
| ------------------ | -------------------- |
| learning_rate      | 1e-4                 |
| batch_size         | 8                    |
| class_weights.mode | auto (smoothing=1.0) |
| model.embed_dim    | 256                  |
| model.depth        | 4                    |
| weight_decay       | 0.01                 |


**Conditions (12 total = 4 models × 3 folds):**


| Condition     | Tokenizer                | channel_emb_mode | Folds   |
| ------------- | ------------------------ | ---------------- | ------- |
| cwt-disabled  | per_channel_cwt_cnn      | disabled         | 0, 1, 2 |
| cwt-dynamic   | per_channel_cwt_cnn      | dynamic          | 0, 1, 2 |
| rcnn-disabled | per_channel_resample_cnn | disabled         | 0, 1, 2 |
| rcnn-dynamic  | per_channel_resample_cnn | dynamic          | 0, 1, 2 |




### Launch command

```bash
uv run python main.py experiment=motor_imagery/physionet_poyo_baselines_3fold -m
```



### Key config overrides

- Config file: `configs/experiment/motor_imagery/physionet_poyo_baselines_3fold.yaml`
- Hydra sweep: `model/tokenizer` × `model.channel_emb_mode` × `hyperparameters.fold_number`
- Same HPs as parent experiment's best config (lr=1e-4, bs=8, cw=auto, dim=256)
- Patience=50, max_epochs=500 (matching HP search settings)



## Results

TBD

## Conclusions

TBD

## Notes for future experiments

TBD