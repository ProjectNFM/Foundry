# NeuroSoft Pretraining Evaluation

**Experiments:** 3
**Date range:** 2026-08-14 to 2026-08-17
**Contributors:** MS

## Overarching Question

Does leak-fixed MAE pretraining on iEEG/EEG data provide meaningful transfer to NeuroSoft 8-band acoustic-stimulus decoding, and does the benefit extend from intrasession to cross-subject (LOSO) evaluation?

## Summary of Findings

This group of experiments established architecture-matched from-scratch baselines and measured transfer benefit from two leak-fixed pretrained initializations (Kochi-only iEEG and Kochi + B2 EEG mixture) on NeuroSoft auditory decoding for minipigs and monkeys.

The intrasession baselines confirmed that the CWT-CNN transfer-compatible architecture reaches 0.270 F1 (minipigs) and 0.264 F1 (monkeys) when trained from scratch — stable across three block folds but well below the 0.394/0.538 ceiling of architecture-optimized Resample-CNN models. Both pretrained initializations substantially improved over these scratch baselines: Kochi-only achieved +20% for minipigs and +9% for monkeys, while Kochi + B2 achieved +13% for minipigs and +15% for monkeys. Importantly, these relative gains are measured against a deliberately constrained baseline chosen for transfer compatibility, not the best achievable POYO performance. Determining whether pretraining provides absolute improvement at higher baseline F1 requires finding better recipes for both the from-scratch control and the finetuning procedure.

It is encouraging that both pretraining data mixes consistently improve over scratch for both species. While the point estimates suggest Kochi-only favors minipigs and Kochi + B2 favors monkeys, this apparent species-specificity pattern should be interpreted cautiously — with only 2–3 folds per condition and sensitivity to the particular checkpoint selected, the difference between the two initializations in either direction may well be noise.

The LOSO scratch baselines revealed that cross-subject generalization is effectively at chance level (0.124 minipigs, 0.126 monkeys vs 0.125 theoretical chance) for both species, with several subjects falling below chance. The limited LOSO transfer data (3 Kochi-only minipig subjects completed before cancellation) showed negligible benefit (+0.001 F1 over scratch), confirming that pretraining does not meaningfully lift LOSO performance. The intrasession transfer benefit does not carry over to the leave-one-subject-out regime, where subject-specific channel layouts and physiology dominate.

## Key Takeaways

- Leak-fixed pretraining consistently improves intrasession NeuroSoft F1 by +9–20% over matched scratch baselines for both data mixes and both species — pretraining helps within-session decoding.
- These gains are relative to a transfer-compatible baseline that is lower than the best achievable POYO architecture; absolute improvement at higher baseline F1 remains to be tested with improved recipes for both scratch and finetuning.
- The ranking between Kochi-only and Kochi + B2 initializations appears species-dependent in these results, but with limited folds and checkpoint sensitivity, this difference may not be robust.
- Cross-subject (LOSO) generalization is at chance for both scratch and pretrained models — pretraining alone cannot bridge the subject gap.

## Experiment Index

| # | Experiment | Hypothesis Verdict | Key Metric |
|---|-----------|-------------------|------------|
| 1 | [Leak-Fixed iEEG Pretraining for Neurosoft Transfer](./20260814-MS-ieeg-leak-fixed-pretraining.md) | Partially confirmed | Intrasession: +9–20% F1 over scratch; LOSO: negligible |
| 2 | [NeuroSoft Intrasession Multisubject From-Scratch Baselines](./20260817-MS-neurosoft-intrasession-baselines.md) | Confirmed | Minipigs 0.270, Monkeys 0.264 (stable, below Resample-CNN ceiling) |
| 3 | [NeuroSoft Leave-One-Subject-Out From-Scratch Baselines](./20260817-MS-neurosoft-loso-baselines.md) | Partially confirmed | Minipigs 0.124, Monkeys 0.126 (at chance; some subjects below) |

## Open Questions

- Does the intrasession transfer benefit persist at higher baseline F1 with an optimized downstream recipe (Resample-CNN, higher capacity) for both scratch and finetuning?
- Can explicit channel-alignment strategies or substantially more training subjects lift LOSO above chance, where pretraining alone cannot?
- Would a longer pretraining schedule or larger-scale iEEG data improve the currently negligible LOSO transfer delta?
