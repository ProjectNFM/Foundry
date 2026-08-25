# HERO spatial-slot ablation: 1-factor vs 8-slot fusion in a flat temporal control

**Status:** Draft
**Date started:** 2026-08-24
**Parent experiment:** [NeuralBench POYO-EEG Tokenizer Baselines](../04-neuralbench-from-scratch-baselines/20260821-MS-neuralbench-poyo-tokenizer-baselines.md)
**Follow-up experiments:** TBD
**Tags:** neuralbench, hero, spatial_slots, ablation, from_scratch, p300, motor_imagery, sleep_stage

## Background

The [NeuralBench POYO-EEG tokenizer baselines](../04-neuralbench-from-scratch-baselines/20260821-MS-neuralbench-poyo-tokenizer-baselines.md) demonstrated that the flat Perceiver architecture—not the temporal tokenizer—is the dominant bottleneck on MI (64 channels, 25,600 tokens) and Sleep (30 s, 4,800 latents). The [hierarchical EEG representation plan](../../docs/hierarchical-eeg-representation-plan.md) addresses this with a three-level temporal hierarchy (HERO), which passed no-training validation on 2026-08-24 (31 deterministic tests, GPU profiling with linear scaling confirmed).

Before testing the full hierarchy (Ladder Question 2), the plan's Stage 1 asks a prerequisite question: does the spatial-slot fusion mechanism—which fuses an unordered set of channel-local features at each time bin through learned queries—actually help compared to a trivial one-factor pooling? This is the first controlled supervised test of the HERO spatial mixer.

Both conditions share the same channel encoder, flat temporal path, optimizer, scheduler, and evaluation protocol. Only the number of spatial slots differs. A flat temporal mode (`temporal_mode=flat`) is used deliberately: the hierarchy is the subject of Question 2, and testing it simultaneously would confound the spatial-factor result.

The matched EEGNet baselines from the [three-task test parity experiment](../04-neuralbench-from-scratch-baselines/20260821-MS-neuralbench-matched-test-parity.md) serve as the absolute task-specific reference: P300 0.625, MI 0.571, Sleep 0.680 (test balanced accuracy).

## Question

On the three NeuralBench targets (P300/16ch/1s, MI/64ch/4s, Sleep/2ch/30s), does the 8-slot spatial mixer improve over 1-slot pooling when the temporal path is held flat?

## Hypothesis

Eight spatial slots will outperform one-slot pooling by at least 2 pp test balanced accuracy on Motor Imagery (64 channels), where the spatial mixer has the richest input set per time bin. The advantage will be smaller or absent on P300 (16 channels) and Sleep (2 channels). If 1-slot and 8-slot are indistinguishable on all three tasks, the spatial-slot mechanism can be simplified before testing the full hierarchy.

## Experiment

### Setup

- **Model:** HERO with `temporal_mode=flat`, `num_local_attn_blocks=2`, `embed_dim=64`, `canonical_rate=128`, `num_attn_heads=8`, `channel_encoder_layers=3`, `channel_encoder_kernel_size=7`.
- **Independent variable:** `num_spatial_slots`: 1 (one-factor pooling) vs 8 (reference spatial-slot mixer).
- **Data and task contract:** NeuralBench v0.2.3 / NeuralSet subject splits, identical to the matched EEGNet and POYO tokenizer baselines:
  - P300 / `Korczowski2014A`: 16 channels, 1.0 s epochs.
  - Motor Imagery / `Schalk2004Bci2000`: 64 channels, 4.0 s epochs.
  - Sleep Stage / `Kemp2000Analysis`: 2 channels, 30.0 s epochs.
- **Seeds:** 33, 34, and 35 for every task and slot condition (18 runs total).
- **Training:** AdamW (`lr=1e-4`, `weight_decay=0.05`), OneCycleLR with cosine annealing and `pct_start=0.1` at step interval, batch size 64, 16-mixed precision, `torch.compile` enabled, gradient clipping 1.0, 40-epoch cap. Early-stopping patience is 10 for all tasks.
- **Evaluation:** Best-validation checkpoint on the NeuralBench held-out test split (`run.evaluate_test=true`).
- **WandB:** project `foundry-neuralbench`; groups `NB_P300_HERO_SPATIAL_SLOTS`, `NB_MI_HERO_SPATIAL_SLOTS`, and `NB_SLEEP_HERO_SPATIAL_SLOTS`.
- **EEGNet comparator groups:** `NB_P300_EEGNET_MATCHED`, `NB_MI_EEGNET_MATCHED`, and `NB_SLEEP_EEGNET_MATCHED`.

### Launch command

```bash
export FOUNDRY_SNAPSHOT_ROOT=/network/scratch/s/sobralm/foundry-launches

uv run python main.py experiment=neuralbench/p300_hero_spatial_slots -m
uv run python main.py experiment=neuralbench/mi_hero_spatial_slots -m
uv run python main.py experiment=neuralbench/sleep_stage_hero_spatial_slots -m
```

### Key config overrides

| Setting | Value |
|---|---|
| `model` | `hero` |
| `model.temporal_mode` | `flat` |
| `model.num_spatial_slots` | sweep: `1, 8` |
| `model.num_local_attn_blocks` | `2` |
| `model.embed_dim` | `64` |
| `trainer.precision` | `16-mixed` |
| `run.compile` | `default` |
| `run.evaluate_test` | `true` |
| `seed` | sweep: `33, 34, 35` |

## Results

TBD

## Conclusions

TBD

## Notes for future experiments

TBD
