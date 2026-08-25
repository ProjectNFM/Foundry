# HERO relational-context sufficiency for Motor Imagery

**Status:** In Progress
**Date started:** 2026-08-25
**Parent experiment:** [HERO spatial-slot ablation: 1-factor vs 8-slot fusion in a flat temporal control](20260824-MS-hero-spatial-slots.md)
**Follow-up experiments:** TBD
**Tags:** neuralbench, hero, motor_imagery, relational_context, spatial_routing, absolute_position, ablation, from_scratch

## Background

The [HERO spatial-slot ablation](20260824-MS-hero-spatial-slots.md) tests
whether eight spatial slots improve on one-factor pooling before introducing
the temporal hierarchy. Motor Imagery (MI) is the key high-channel target: the
earlier [NeuralBench POYO tokenizer baselines](../04-neuralbench-from-scratch-baselines/20260821-MS-neuralbench-poyo-tokenizer-baselines.md)
found that flat POYO was 19--23 percentage points behind EEGNet on MI and
attributed the failure partly to the absence of an effective spatial inductive
bias across 64 channels. The matched [NeuralBench EEGNet test-parity
experiment](../04-neuralbench-from-scratch-baselines/20260821-MS-neuralbench-matched-test-parity.md)
provides the external absolute-performance comparator under the same dataset,
split, and best-validation-checkpoint test protocol.

This experiment implements Phase 4 of the [HERO sample-derived relational
context plan](../../docs/hero-measurement-context-plan.md). It asks whether a
low-bandwidth, permutation-equivariant context derived from the same
unnormalized sample can give spatial slots enough information to route MI
channel content without relying on absolute electrode coordinates. This
builds on the legacy [dynamic channel embedding
analysis](../_legacy/019-dynamic-channel-embedding-analysis.md), which found
that signal-conditioned cross-channel embeddings captured sample properties
rather than stable electrode identity, and its [linear-probe
follow-up](../_legacy/020-linear-probe-dynamic-channel-emb.md), which showed
that such cross-channel processing can nevertheless improve downstream
representations.

The parent experiment is still a draft. Per the experiment decision, Phase 4
is not launch-gated on its result: all HERO conditions use eight spatial slots
regardless. This preserves the planned architecture but means a null result
must be interpreted jointly with the parent's eventual 1-vs-8-slot finding.

## Question

On NeuralBench Motor Imagery, is same-sample relational channel context
sufficient to improve eight-slot HERO spatial routing without useful
incremental information from known absolute electrode position?

## Hypothesis

Relational context is sufficient if all of the following pre-registered
criteria hold for held-out test balanced accuracy:

1. relational-only exceeds both signal-only and local-context by at least
   0.02 in the three-seed mean;
2. relational-only is no more than 0.02 below relational + position;
3. shuffled relational is at least 0.02 below correctly bound
   relational-only; and
4. relational-only exceeds both signal-only and local-context for at least two
   of the three matched seeds, with no seed more than 0.02 below signal-only.

Failure of any criterion refutes the full sufficiency hypothesis. The
condition matrix remains useful for distinguishing whether any gain comes
from local raw-signal summaries, correct cross-channel binding, absolute
position, channel type, or added capacity.

## Experiment

### Setup

- **Model:** HERO from scratch with `temporal_mode=flat`, eight spatial slots,
  `embed_dim=64`, two local attention blocks, eight attention heads, context
  width 32, three shared local-context convolution blocks, and two
  four-head relational blocks. Context affects spatial-routing logits only;
  normalized channel content remains the value stream.
- **Data:** NeuralBench v0.2.3 / NeuralSet `Schalk2004Bci2000`, 64 channels,
  4.0-second trials, canonical NeuralBench subject split, and the same target
  contract as the matched EEGNet baseline.
- **Task:** Four-class Motor Imagery classification.
- **Independent variable:** Spatial-routing context source: signal-only,
  type-only, local-context, relational-only, position-only, relational +
  position, or shuffled relational.
- **Seeds:** 33, 34, and 35 for all seven conditions (21 HERO runs).
- **Training:** Match `mi_hero_spatial_slots`: AdamW (`lr=1e-4`,
  `weight_decay=0.05`), step-wise cosine OneCycleLR with `pct_start=0.1`, batch
  size 64, `16-mixed` precision, gradient clipping 1.0, `torch.compile`,
  40-epoch cap, and early-stopping patience 10.
- **Evaluation:** Evaluate the best-validation-balanced-accuracy checkpoint on
  the held-out test split. The primary metric is three-seed mean test balanced
  accuracy. Also report per-seed values, standard deviation, per-class recall,
  the left-fist/right-fist confusion, train/validation curves, selected epoch,
  context gates, relational and spatial-slot attention summaries, parameter
  count, peak memory, and wall-clock time.
- **Controlled baseline:** Eight-slot signal-only HERO using the same
  refactored one-resample path as every context condition.
- **External comparator:** Matched EEGNet group `NB_MI_EEGNET_MATCHED`. This
  measures absolute task performance but is not used to attribute the effect
  of relational context because architecture and parameter count differ.
- **WandB:** Project `foundry-neuralbench`; planned HERO group
  `NB_MI_HERO_RELATIONAL_CONTEXT`.

The seven conditions are:

| Condition | Local raw summary | Cross-channel relations | Type | Absolute position |
|---|:---:|:---:|:---:|:---:|
| Signal-only | No | No | No | No |
| Type-only | No | No | Yes | No |
| Local-context | Yes | No | Yes | No |
| Relational-only | Yes | Yes | Yes | No |
| Position-only | No | No | Yes | Yes |
| Relational + position | Yes | Yes | Yes | Yes |
| Shuffled relational | Yes | Yes, misassigned | Yes | No |

### Launch command

```bash
FOUNDRY_SNAPSHOT_ROOT=/network/scratch/s/sobralm/foundry-launches \
  uv run python main.py experiment=neuralbench/mi_hero_relational_context -m
```

Production launch requirements from `AGENTS.md`: the repository must be clean
and committed, `FOUNDRY_SNAPSHOT_ROOT` must be
`/network/scratch/s/sobralm/foundry-launches`, the Hydra launcher must use the
`long` partition, and the normal `python main.py ... -m` snapshot workflow must
be retained.

**Slurm job array:** `10499135_[0-20]` (21 jobs)
**Snapshot bundle:** `/network/scratch/s/sobralm/foundry-launches/20260825T211631_NB_MI_HERO_RELATIONAL_CONTEXT_f0e61ea2_8032ba63`
**Git SHA:** `f0e61ea268d6b3f3173a88222c164a270ad4e7c9`

### Key config overrides

Planned Hydra config:
`configs/experiment/neuralbench/mi_hero_relational_context.yaml`, derived from
`configs/experiment/neuralbench/mi_hero_spatial_slots.yaml`. Only the
condition-specific non-default model settings should vary:

| Condition | `model.channel_context_mode` | `model.channel_type_enabled` | `model.absolute_position_enabled` | Shuffle hook |
|---|---|:---:|:---:|:---:|
| Signal-only | `disabled` | `false` | `false` | off |
| Type-only | `disabled` | `true` | `false` | off |
| Local-context | `local` | `true` | `false` | off |
| Relational-only | `relational` | `true` | `false` | off |
| Position-only | `position` | `true` | `true` | off |
| Relational + position | `relational_position` | `true` | `true` | off |
| Shuffled relational | `relational` | `true` | `false` | on |

Fixed overrides relative to the model defaults:

| Setting | Value |
|---|---|
| `model.temporal_mode` | `flat` |
| `model.num_spatial_slots` | `8` |
| `seed` | sweep: `33,34,35` |
| `run.evaluate_test` | `true` |

Before launch, the experiment config must provide a stable condition label in
the W&B config, plumb the experimental relational permutation hook through the
training path, and log routing diagnostics plus test confusion counts. The
shuffle must be a non-identity channel permutation, fixed per seed and applied
only to relational vectors after their correct computation; signal, targets,
type, positions, masks, and content values must not be permuted.

## Results

### Summary

TBD

### Metrics

TBD

### Analysis

Use
[`analysis/20260825-MS-hero-relational-context-sufficiency_analysis.py`](../../analysis/20260825-MS-hero-relational-context-sufficiency_analysis.py)
to fetch the HERO and matched EEGNet groups through the W&B API, cache the
per-run table, print the pre-registered sufficiency checks, and generate the
balanced-accuracy comparison figure. Extend the script with attention and
confusion analyses once the final logging keys are fixed in the launch config.

### Figures

TBD

## Conclusions

TBD

## Notes for future experiments

If relational sufficiency is supported, proceed to Phase 5 on at least one
coordinate-poor ECoG or SEEG task and test channel-removal robustness. If it
is not supported, use the controls to decide between local raw-signal
information, absolute-position routing, optimization failure from zero-gated
sources, and a spatial-slot bottleneck before increasing context capacity.
