# NeuralBench From-Scratch Baselines

**Experiments:** 4
**Date range:** 2026-08-20 to 2026-08-24
**Contributors:** MS

## Overarching Question

Can Foundry's EEGNet and POYO-EEG implementations match or exceed the
NeuralBench v0.2.3 reference EEGNet on P300, Motor Imagery, and Sleep Stage
when trained from scratch on the exact same data, splits, and evaluation
protocol — and what does the comparison reveal about POYO's architectural
scaling limits?

## Summary of Findings

This group traces a four-experiment progression from adapter proof-of-concept
to a systematic POYO tokenizer benchmark on NeuralBench's three core tasks.

The first experiment validated the NeuralBench integration adapter by training
Foundry's EEGNet on P300 (Korczowski2014A) and comparing best-validation
metrics against NeuralBench's reference EEGNet test metrics. The ~2.1 pp
balanced-accuracy gap was within the pre-specified 5 pp threshold and
attributable to documented implementation differences (dropout 0.5 vs 0.25,
constant LR vs OneCycleLR). Critically, this was a val-vs-test comparison —
not apples-to-apples.

The second experiment extended the adapter to Motor Imagery
(Schalk2004Bci2000, 64 channels, 4 classes) and Sleep Stage
(Kemp2000Analysis, 5 classes, 30 s epochs). Sleep validated easily (1.3 pp
gap), but MI showed a 6.6 pp gap that exceeded the target. The val-vs-test
confound was a leading explanation: one retained NeuralBench log showed seed
33 selecting val balanced accuracy of 0.509 but achieving 0.595 on the test
set.

The third experiment resolved all confounds by matching every exposed
NeuralBench hyperparameter (dropout 0.25, BN momentum 0.01/epsilon 1e-3,
spatial max-norm 1.0, OneCycleLR at step interval) and evaluating the
best-validation checkpoint on the exact NeuralBench test split. All three
tasks passed the tightened ±2 pp parity criterion: P300 (−1.14 pp), Motor
Imagery (−1.45 pp), and Sleep Stage (+0.54 pp). The MI discrepancy from
Phase 1 was entirely explained by the val-vs-test comparison and unmatched
hyperparameters. This experiment established the controlled EEGNet comparator
for the POYO baselines.

The fourth experiment benchmarked from-scratch POYO-EEG with CWT-CNN and
ResampleCNN tokenizers against the matched EEGNet baseline. POYO outperformed
EEGNet on P300 (+2.1–2.3 pp balanced accuracy, +10 pp F1) but collapsed on
Motor Imagery (−19 to −23 pp) and timed out on Sleep Stage (3–4 epochs before
the 12 h Slurm limit). The MI failure was traced to three compounding factors:
a 40:1 Perceiver compression ratio from 25,600 input tokens, FP16 softmax
underflow at long sequence lengths, and the lack of spatial inductive bias
that EEGNet's depthwise convolution provides for 64-channel data. The Sleep
timeout was caused by 4,800 latent tokens producing 900× the self-attention
cost of P300. Despite only 3–4 training epochs, POYO CWT-CNN's validation
balanced accuracy (0.653) trailed EEGNet's fully converged result (0.674) by
only 2.1 pp, suggesting competitiveness after reducing the latent sequence
length.

Across all experiments, CWT-CNN ≥ ResampleCNN on every task (by 0–3.5 pp),
but the differences were small and not statistically significant at n=3. The
dominant factor on MI and Sleep was architectural, not tokenizer-related.

## Key Takeaways

- **NeuralBench adapter is validated.** Foundry EEGNet achieves test balanced
  accuracy within ±1.5 pp of NeuralBench's reference EEGNet on all three
  tasks when hyperparameters are matched and the same test-evaluation protocol
  is used (Experiment 3).

- **Val-vs-test comparisons are unreliable.** The Phase 1 MI gap of 6.6 pp
  collapsed to 1.45 pp once both sides evaluated on the same test split. This
  underscores the need for controlled test-set evaluation in all future
  comparisons (Experiments 1–3).

- **POYO scales well on P300 but fails on MI and Sleep.** The flat Perceiver
  with independent per-channel tokenization works in the moderate-token regime
  (1,600 tokens, 10:1 compression) but cannot handle 25,600 tokens (MI) or
  4,800 latents (Sleep). This is an architecture-scale mismatch, not a
  tokenizer deficiency (Experiment 4).

- **Tokenizer effects are small relative to architectural bottlenecks.**
  CWT-CNN and ResampleCNN are nearly interchangeable on these tasks. Future
  work should prioritise architectural changes (hierarchical structure,
  reduced latent counts) over tokenizer refinements (Experiment 4).

- **FP16 precision is unsafe at long sequence lengths.** The `16-mixed` policy
  validated on P300 (1,600 tokens) produces dead attention patterns at MI's
  25,600 tokens. Precision must be validated per task, not assumed portable
  (Experiment 4, [MI gap analysis](../../docs/neuralbench-poyo-mi-performance-gap.md)).

## Experiment Index

| # | Experiment | Hypothesis Verdict | Key Metric |
|---|-----------|-------------------|------------|
| 1 | [NeuralBench P300 EEGNet Comparison](./20260820-MS-neuralbench-p300-eegnet-comparison.md) | Confirmed | P300 val gap: 2.1 pp (≤5 pp target) |
| 2 | [NeuralBench Phase 1 — MI & Sleep EEGNet Comparison](./20260820-MS-neuralbench-phase1-mi-sleep-comparison.md) | Partially confirmed | Sleep 1.3 pp ✓; MI 6.6 pp ✗ (val-vs-test confound) |
| 3 | [NeuralBench Matched EEGNet — Three-Task Test Parity](./20260821-MS-neuralbench-matched-test-parity.md) | Confirmed | P300 −1.14 pp, MI −1.45 pp, Sleep +0.54 pp (all ≤2 pp) |
| 4 | [NeuralBench POYO-EEG Tokenizer Baselines](./20260821-MS-neuralbench-poyo-tokenizer-baselines.md) | Partially confirmed | P300 +2.1 pp ✓; MI −19 pp ✗; Sleep inconclusive (timed out) |

## Open Questions

- **Hierarchical architecture feasibility.** Can a hierarchical Perceiver
  variant (e.g. per-channel temporal attention → cross-channel spatial
  attention) reduce the effective token count and unlock POYO on high-channel
  and long-duration tasks?
- **Pretrained POYO.** Do the from-scratch patterns hold after pretraining, or
  does transfer learning compensate for the architectural scaling limits?
