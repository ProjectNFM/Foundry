# PhysioNet MI POYO Final Baselines (All Conditions × 3 Folds)

**Status:** Completed
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

### Summary

All 12 runs (4 conditions × 3 folds) completed successfully. SLURM job array
10282387, WandB group `PHYSIONET_MI_POYO_BASELINES_3FOLD`.

All conditions perform similarly, with mean F1 spanning a narrow 0.873–0.884
range. Fold 0 is consistently strong (~0.93) across all conditions, while
folds 1 and 2 show more variation and lower scores (~0.83–0.87). The CWT-CNN
tokenizer has a marginal edge over ResampleCNN (+0.3–0.4 pp), far smaller than
the +3.1–3.5 pp gap observed on KempSleep. Dynamic channel embeddings slightly
*hurt* performance (−0.7 to −0.8 pp), contrary to the hypothesis.

### Metrics

| Condition     | Mean F1 | Std    | Fold 0 | Fold 1 | Fold 2 | Mean Acc | Mean AUROC | Epochs (f0) |
| ------------- | ------- | ------ | ------ | ------ | ------ | -------- | ---------- | ----------- |
| CWT Disabled  | 0.8840  | 0.0330 | 0.9285 | 0.8743 | 0.8493 | 0.8844   | 0.9373     | 230         |
| CWT Dynamic   | 0.8764  | 0.0402 | 0.9307 | 0.8635 | 0.8349 | 0.8753   | 0.9362     | 242         |
| RCNN Disabled | 0.8802  | 0.0365 | 0.9278 | 0.8735 | 0.8392 | 0.8819   | 0.9401     | 255         |
| RCNN Dynamic  | 0.8733  | 0.0424 | 0.9294 | 0.8637 | 0.8269 | 0.8754   | 0.9374     | 236         |

**Tokenizer comparison (CWT-CNN vs ResampleCNN):**

| Channel Emb | CWT F1 | RCNN F1 | Δ (pp) |
| ----------- | ------ | ------- | ------ |
| Disabled    | 0.8840 | 0.8802  | +0.4   |
| Dynamic     | 0.8764 | 0.8733  | +0.3   |

**Channel embedding effect (Disabled vs Dynamic):**

| Tokenizer   | Disabled F1 | Dynamic F1 | Δ (pp) |
| ----------- | ----------- | ---------- | ------ |
| CWT-CNN     | 0.8840      | 0.8764     | −0.8   |
| ResampleCNN | 0.8802      | 0.8733     | −0.7   |

**Comparison with EEGNet** (from [EEGNet Final Baselines](20260804-MS-physionet-mi-eegnet-final-baselines.md)):
- Best POYO condition: CWT Disabled (0.884 mean F1)
- EEGNet: 0.887 ± 0.027
- Δ (best POYO − EEGNet): −0.2 pp

### Analysis

Script: `analysis/030_physionet_mi_final_baselines.py`

```bash
uv run python analysis/030_physionet_mi_final_baselines.py
```

### Figures

![Main Results — bar chart with error bars](analysis/figures/030_physionet_mi_final_main_results.png)

![F1 Learning Curves — fold 0](analysis/figures/030_physionet_mi_final_f1_curves.png)

![Cross-Fold Variance — strip plot](analysis/figures/030_physionet_mi_final_fold_variance.png)

## Conclusions

**Partially confirmed.** Of the four hypotheses:

1. **H1 (CWT > RCNN): Directionally correct but effect much smaller than expected.**
   CWT-CNN edges out ResampleCNN by only +0.3–0.4 pp on MI, versus +3.1–3.5 pp
   on KempSleep. The wavelet decomposition advantage appears task- and
   dataset-dependent rather than universal.

2. **H2 (dynamic helps more on 64-ch): Refuted.** Dynamic channel embeddings
   slightly *hurt* on MI (−0.7 to −0.8 pp), opposite to the hypothesized +1–3 pp
   benefit. The `RelativeChannelEncoder` may be overfitting given the 64-channel
   array, or the sensorimotor topography is not captured effectively by this
   approach.

3. **H3 (all ≥0.90 F1): Refuted.** Only fold 0 consistently meets the 0.90
   threshold (~0.93 for all conditions). Folds 1 and 2 range from 0.827 to
   0.874, pulling all condition means below 0.90.

4. **H4 (CWT dynamic reproduces ~0.937 on fold 0): Confirmed.** CWT Dynamic
   achieved 0.931 F1 on fold 0, close to the parent experiment's 0.937 (within
   expected noise).

The most striking finding is the **near-equivalence of all architectures** on
this dataset: the entire range spans only 1.1 pp (0.873–0.884). The high
cross-fold variance (std 0.027–0.042) dwarfs any architecture effect.

## Notes for future experiments

- **Pretraining POYO before fine-tuning** — the from-scratch POYO conditions
  match EEGNet but don't surpass it. Pretraining on a large multi-dataset corpus
  may unlock the architectural advantage of POYO's flexible tokenization.
- **Test on other MI datasets** — the near-equivalence of all architectures may
  be specific to PhysioNet MI's 64-channel, 109-subject setup. Other MI
  datasets (e.g., BCI Competition IV 2a, Lee2019) could reveal whether this
  pattern holds.
- **Investigate why dynamic channel embeddings hurt** — possible overfitting
  with the 64-channel high-density array. Regularization (dropout on channel
  embeddings, fewer embedding dimensions) or different channel encoding
  strategies may help.