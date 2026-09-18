# NeuroSoft Transferability

**Experiments:** 9
**Date range:** 2026-09-15 to 2026-09-18
**Contributors:** MS

## Overarching Question

Does supervised NeuroSoft pretraining produce a reusable foundation that
improves downstream learning across recordings and subjects?

## Summary of Findings

The source task was learnable. All tested architectures improved source
supported macro-F1, including smaller, larger, bias-free, and shared-adapter
variants. However, stronger source performance did not translate into better
downstream transfer.

As source training continued, the models became increasingly dependent on
recording-specific adapter biases. At the same time, source-validation loss
improved while downstream performance and optimization efficiency generally
worsened. This indicates increasing specialization to the source recordings
rather than development of a broadly reusable representation.

Early checkpoints initially appeared to provide modest transfer gains. Those
gains were not durable: longer pretraining was neutral or harmful, and
changing model scale, removing adapter bias, or sharing the adapter did not
reliably rescue transfer. None of these conditions consistently improved
optimization efficiency over scratch.

The final recipe-matched controls removed the apparent early advantage.
Matching the scratch model's optimizer recipe improved scratch by 1.84 F1
points in minipigs and 4.14 points in monkeys. No pretrained checkpoint then
outperformed scratch under the preregistered criterion, while later
checkpoints were significantly worse. Within this dataset, model family,
supervised objective, and downstream protocol, the experiments provide no
evidence of foundation-like transfer behavior.

## Main Figures

Source-task performance improves consistently across all tested architectures:

![Source validation supported macro-F1 trajectories](../../analysis/figures/20260916-MS-pretraining-architecture-viability_val_supported_f1_trajectories.png)

At the same time, the models become increasingly dependent on correctly
matched recording-specific adapter biases:

![Adapter-bias derangement cross-entropy penalty](../../analysis/figures/20260915-MS-adapter-bias-perturbation_cross_entropy_penalty.png)

Improving source validation is accompanied by deteriorating downstream
performance and optimization efficiency:

![Source-validation and downstream trajectories](../../analysis/figures/20260915-MS-source-validation-downstream-trajectory_main_trajectories.png)

After matching the scratch training recipe, no checkpoint demonstrates
positive transfer in minipigs:

![Minipig transfer relative to recipe-matched scratch](../../analysis/figures/20260918-MS-recipe-matched-scratch-control_recentered_transfer_effects.png)

The monkey replication gives the same conclusion, with every point estimate
favoring scratch:

![Monkey transfer relative to recipe-matched scratch](../../analysis/figures/20260918-MS-recipe-matched-scratch-monkeys_recentered_transfer_effects.png)

## Key Takeaways

- Successful source-task learning is not evidence of transferable
  representation learning.
- Better same-recording source validation became increasingly misaligned with
  downstream performance.
- Model scale and adapter/interface changes did not produce durable transfer
  or faster downstream optimization.
- Apparent early transfer gains were explained by an unmatched
  scratch-training recipe.
- The result is specific to the tested NeuroSoft data, model family,
  objective, and full-data downstream setting; it does not rule out other
  pretraining approaches.

## Experiment Index

| # | Experiment | Hypothesis Verdict | Key Metric |
|---|---|---|---|
| 1 | [Adapter-bias reliance across source-pretraining age](./20260915-MS-adapter-bias-perturbation.md) | Confirmed | Bias-derangement loss slope: +0.133 minipigs, +0.228 monkeys |
| 2 | [Source-validation performance versus downstream transfer](./20260915-MS-source-validation-downstream-trajectory.md) | Confirmed for minipigs; partial for monkeys | Step 10,000 vs 100 F1: -2.24 pp minipigs, -2.40 pp monkeys |
| 3 | [Source-pretraining architecture viability](./20260916-MS-pretraining-architecture-viability.md) | Confirmed | Source F1 increased in all 10 architecture/species conditions |
| 4 | [Batch-128 reference transfer replication](./20260916-MS-batch128-reference-transfer-replication.md) | Initially confirmed; superseded by matched controls | Step-100 F1: +2.06 pp minipigs, +3.01 pp monkeys |
| 5 | [Adapter-bias transfer](./20260916-MS-bias-free-transfer.md) | No durable benefit | Late transfer neutral or harmful; steps saved remained negative |
| 6 | [Model-scale transfer](./20260916-MS-model-scale-transfer.md) | No durable benefit | Every scale worsened between steps 100 and 10,000 |
| 7 | [Shared-adapter transfer](./20260916-MS-shared-adapter-transfer.md) | No durable benefit | Retaining the shared interface provided no consistent advantage |
| 8 | [Recipe-matched scratch transfer control](./20260918-MS-recipe-matched-scratch-control.md) | Refuted | Minipig effects: +0.22 pp at step 100; -4.27 pp at step 10,000 |
| 9 | [Monkey recipe-matched scratch transfer control](./20260918-MS-recipe-matched-scratch-monkeys.md) | Refuted | Every checkpoint favored scratch; effects ranged from -1.13 to -5.16 pp |

## Open Questions

- Would an objective that explicitly rewards cross-recording invariance
  transfer better?
- Would checkpoint selection using unseen recordings avoid source-specialized
  solutions?
- Could transfer become useful in genuinely low-data downstream regimes?
