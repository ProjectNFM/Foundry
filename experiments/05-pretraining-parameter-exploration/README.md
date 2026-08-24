# Pretraining Parameter Exploration

**Experiments:** 3
**Date range:** 2026-08-11 to 2026-08-12
**Contributors:** MS

## Overarching Question

After establishing the B2 data sweet spot (Klinzing + Shirazi + Pavlov,
~37k ch·h), which pretraining configuration choices — masking strategy,
information-leak corrections, tokenizer selection, and temporal scale
diversity — produce the best downstream representations for sleep staging,
motor imagery, and P300 detection?

## Summary of Findings

This group of experiments systematically explored the pretraining
configuration space around the B2 data scaling baseline. However, **the
lack of a consistent downstream evaluation across all experiments severely
muddies the conclusions.** Kemp Sleep finetuning failed in every
experiment, Brain Invaders P300 evaluation crashed entirely for the
multi-length condition, and several other task/mode combinations had
incomplete fold coverage. As a result, most comparisons rest on PhysioNet
MI alone — a single task that may not be representative of overall
representation quality.

The leak-fix ablation is the most complete experiment in the group. It
discovered two significant information leaks — channel-encoder masking and
temporal-embedding receptive-field bleed — which together reduced
reconstruction loss by nearly 5×. Fixing both leaks made the pretraining
objective honest, but downstream metrics were essentially unchanged on the
tasks that completed. This suggests the encoder backbone was learning
useful features despite the decoder exploiting shortcuts, though without
reliable sleep staging finetuning results, the picture remains incomplete.
The ablation also indicated CWT-CNN outperforms ResampleCNN at B2 scale,
though this too would benefit from confirmation with a full task battery.

The masking parameter sweep tested whether harder reconstruction (mask
ratios 0.7–0.9) would force richer representations. On the tasks with
completed folds, higher ratios were neutral to slightly worse, but the
incomplete Kemp Sleep data means the conclusion holds primarily for MI and
P300. The multi-length pretraining experiment was the most compromised:
all P300 evaluations crashed, Sleep finetuning failed for both conditions,
and the pretraining itself stopped suspiciously early (63k vs 458k steps).
The available MI data showed no benefit, but too little survived to draw
strong conclusions.

These experiments should be revisited once the downstream evaluation
infrastructure is stabilized — particularly Kemp Sleep finetuning and P300
compatibility with non-standard checkpoints — so that comparisons can be
made across a consistent set of tasks.

## Key Takeaways

- **Reconstruction loss may not predict downstream quality**, but the
  evidence is preliminary. A 5× increase in pretraining loss (from leak
  fixes) had no measurable downstream impact on the tasks that completed,
  but the missing Sleep FT data is a significant gap.
  *(Leak fix ablation)*

- **The 0.5 mask ratio appears reasonable.** Higher ratios (0.7–0.9) did
  not help on MI or P300, but the Sleep comparison is incomplete. This
  tentatively contradicts the vision MAE finding that ~0.75 is optimal.
  *(Masking parameter sweep)*

- **CWT-CNN likely outperforms ResampleCNN at B2 scale.** Consistent
  advantages across all tasks that completed, but confirmation with a full
  task battery would strengthen this.
  *(Leak fix ablation)*

- **Both leak fixes should remain enabled.** They make the pretraining
  objective honest without hurting downstream performance. All future runs
  use `disable_channel_encoder_token_mask=false` and
  `zero_masked_signal=true` as defaults.
  *(Leak fix ablation)*

- **Multi-length pretraining is inconclusive.** Too many downstream
  failures to draw any conclusion. This axis is paused until the evaluation
  pipeline is more reliable.
  *(Multi-length pretraining)*

- **The downstream evaluation pipeline is unreliable.** This is the
  strongest finding: across all three experiments, Sleep FT consistently
  failed, P300 was fragile, and partial fold coverage was the norm. These
  experiments need to be revisited once the infrastructure is fixed.

## Experiment Index

| # | Experiment | Hypothesis Verdict | Key Metric |
|---|---|---|---|
| 1 | [Information Leak Fixes + Tokenizer Comparison](./20260812-MS-channel-encoder-leak-fix-impact.md) | Partially confirmed (leaks real, downstream negligible) | Leak-fixed CWT: Sleep FT F1 = 0.738 |
| 2 | [Masking Parameter Sweep](./20260811-MS-masking-parameter-sweep.md) | Refuted (higher ratios not better) | M0 (0.5/10): MI FT F1 = 0.884 |
| 3 | [Multi-Length Pretraining](./20260811-MS-multi-length-pretraining.md) | Inconclusive (severe run failures) | S1: MI FT F1 = 0.879 (neutral vs M0) |

## Open Questions

- **Downstream infrastructure:** Why do Kemp Sleep finetuning runs
  consistently fail? Why did P300 crash with the multi-length checkpoint?
  Fixing these is a prerequisite for revisiting any pretraining config
  question.
- **Re-run with full task coverage:** Once the pipeline is stable, the
  masking sweep and multi-length experiments should be re-evaluated with
  all three tasks completing reliably across all folds.
- **Multi-length pretraining deserves a second look:** The 63k-step early
  stop may have been premature; longer training or a curriculum approach
  could change the outcome.
- **Masking strategy alternatives:** Beyond ratio and block-size, are there
  strategies (frequency-aware masking, curriculum masking) better suited to
  EEG temporal structure?
