# Data Scaling: Volume, Diversity, and Paradigm Mix for EEG Pretraining

**Experiments:** 6
**Date range:** 2026-08-05 to 2026-08-11
**Contributors:** MS

## Overarching Question

How does the composition of EEG pretraining data — its volume, source
diversity, channel density, and paradigm mix — affect downstream transfer
for sleep staging, motor imagery, and P300 detection?

## Summary of Findings

This group spans the first systematic investigation of data scaling for
MAE-style EEG pretraining. Starting from a proof-of-concept two-dataset
pretrain (Klinzing sleep + Shirazi resting-state), we ran 12 pretraining
configurations spanning 1–5 datasets and 2,338–50,566 ch·h of effective data,
each evaluated on 3 downstream tasks with both finetuning and linear probing
(216 downstream runs total). All pretraining used the same model (CWT-CNN +
dynamic channel embeddings) and compute budget (200k steps).

The central finding is that **more data is not always better**. Downstream
transfer peaks at 3 datasets (B2: Klinzing + Shirazi + Pavlov, ~37k ch·h) and
degrades when adding a 4th or 5th source. B2 is the only configuration across
all 12 that beats the EEGNet baseline on motor imagery (F1 = 0.891 vs 0.887).
Adding Getzmann (4th dataset, resting-state) or Kochi (5th dataset, visual
naming) introduces interference that outweighs any additional diversity
benefit. The "kitchen sink" run (E1, all 5 datasets, ~51k ch·h) performs
worse than B2 on every task despite having 36% more data.

Within single-source scaling, 10x volume increases (A1 → A2) yield negligible
finetuning gains (+0.001–0.003 F1) but meaningful representation improvements
visible in linear probes (+0.038 on Kemp Sleep). This reveals a recurring
theme: **diversity and volume help representations (linear probes) more than
they help finetuning**. The controlled comparison (C2 vs A2, same effective
data but 3 sources vs 1) confirms this directly — C2 has the best linear
probes across all 12 configs but mixed finetuning results. Finetuning appears
to compensate for suboptimal representations through adaptation, while linear
probes expose the raw quality of pretrained features.

Paradigm-diverse data (Kochi visual naming) consistently fails to help and
sometimes actively harms transfer. Kochi-only pretraining causes catastrophic
motor imagery failure (F1 = 0.725 ± 0.107), and adding Kochi to any multi-source
mix provides no benefit. The one exception is a modest P300 advantage for D1
(Kochi-only), possibly reflecting shared event-related processing structure.

P300 detection remains resistant to pretraining transfer across all
configurations, with every pretrained model falling 4–6 F1 points below the
EEGNet baseline. This likely reflects an architectural limitation rather than
a data scaling issue.

## Key Takeaways

- **3-dataset pretraining is the sweet spot.** B2 (Klinzing + Shirazi + Pavlov)
  achieves the best MI finetuning (0.891) and strong Sleep results (0.738).
  Adding more datasets hurts. *(Diversity Scaling)*

- **Diversity helps representations more than finetuning.** Volume-matched
  3-source pretraining (C2) produces the best linear probes on all tasks but
  does not consistently win finetuning. *(Controls)*

- **Volume scaling has sharply diminishing returns for finetuning.** 10x more
  data from the same source gives <0.5% F1 improvement on finetuning, though
  representations improve meaningfully. *(Volume Scaling)*

- **Paradigm-mismatched data hurts.** Kochi visual naming provides no benefit
  in any mix and causes catastrophic failure when used alone for MI.
  *(Paradigm Diversity)*

- **The initial two-dataset pretrain established that pretraining helps Sleep
  staging.** CWT-CNN/disabled achieved 0.740 F1 (best across all experiments),
  surpassing the from-scratch baseline by +1.0 pp. CWT-CNN also produces
  richer representations than ResampleCNN. *(Two-Dataset Pretrain)*

- **P300 remains unsolved by pretraining.** No configuration closes the gap
  with EEGNet (0.386). This is likely architectural, not data-related.
  *(All experiments)*

## Experiment Index

| # | Experiment | Hypothesis Verdict | Key Metric |
|---|-----------|-------------------|------------|
| 1 | [Two-Dataset Pretrain](./20260805-MS-two-dataset-pretrain-downstream-eval.md) | Partially refuted | Kemp Sleep FT: 0.740 (+0.010 vs baseline) |
| 2 | [Volume Scaling (A1-A3)](./20260807-MS-volume-scaling-pretrain.md) | Partially confirmed | A1→A2: +0.001 FT, +0.038 LP (Kemp) |
| 3 | [Diversity Scaling (B1-B3)](./20260807-MS-diversity-scaling-pretrain.md) | Partially confirmed | B2 MI FT: 0.891 (beats EEGNet 0.887) |
| 4 | [Controls (C1, C2)](./20260807-MS-diversity-volume-controls.md) | Partially confirmed | C2 best LP on all 3 tasks |
| 5 | [Paradigm Diversity (D1-D3)](./20260807-MS-paradigm-diversity-pretrain.md) | Mostly refuted | D1 MI: 0.725 (catastrophic) |
| 6 | [Maximum Data (E1)](./20260807-MS-maximum-data-pretrain.md) | Partially refuted | E1 < B2 on MI and Sleep |

## Open Questions

- **Why is B2 the sweet spot?** Pavlov's working memory paradigm (19ch, 156
  subjects) may share task-relevant temporal structure with downstream tasks
  that Getzmann's resting-state and Kochi's visual naming lack. Characterizing
  what makes a pretraining source "useful" for a given downstream task is a
  key open question.

- **Masking strategy.** All experiments used TemporalBlockMasking (block_size=10,
  mask_ratio=0.5). Alternative strategies — RandomTokenMasking (standard MAE),
  ChannelMasking (spatial), or hybrid approaches — could produce fundamentally
  different representations. The optimal masking strategy likely depends on
  which downstream task the representations need to support.
  → Follow-up: [Masking Parameter Sweep](../inbox/20260811-MS-masking-parameter-sweep.md)

- **Sequence length mismatch.** All pretraining used 2s windows, but downstream
  tasks range from 1s (P300) to 30s (Sleep). Representations learned on 2s
  windows may miss longer-range temporal structure critical for sleep staging
  (spindles, K-complexes, slow waves). Training on varied sequence lengths or
  longer windows could yield more versatile representations.
  → Follow-up: [Multi-Length Pretraining](../inbox/20260811-MS-multi-length-pretraining.md)

- **Data augmentation.** None of the pretraining runs used data augmentation.
  Standard EEG augmentations (time shift, amplitude scaling, channel dropout,
  noise injection) or more advanced approaches (mixing, spectral perturbation)
  could increase effective diversity without requiring additional datasets,
  potentially achieving the diversity benefit observed in C2 without the
  volume-matching constraint.

- **Representation vs finetuning gap.** Diversity consistently helps
  representations (linear probes) more than finetuning. Partial freezing or
  selective unfreezing strategies might better exploit the richer
  representations from diverse pretraining.
