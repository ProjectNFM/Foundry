# Masking Parameter Sweep: Does Harder Masking Produce Better Downstream Features?

**Status:** On hold (2026-08-12) — awaiting [Channel Encoder Leak Fix](20260812-MS-channel-encoder-leak-fix-impact.md) results
**Date started:** 2026-08-11
**Parent experiment:** [Data Scaling Group](../02-data-scaling/README.md) (builds on B2 sweet spot)
**Follow-up experiments:** [Channel Encoder Leak Fix Impact](20260812-MS-channel-encoder-leak-fix-impact.md)
**Tags:** pretraining, mae, masked, masking_sweep, cwt_cnn, dynamic_ch

> **On hold (2026-08-12):** All runs in this sweep use
> `channel_emb_mode="dynamic"`, which was affected by an information leak in
> the `RelativeChannelEncoder` (the encoder pooled over masked tokens, giving
> the decoder a shortcut). A [leak fix ablation](20260812-MS-channel-encoder-leak-fix-impact.md)
> is now running to quantify the impact. Because the leak interacts with
> mask_ratio (higher ratios leak more information), the optimal masking
> configuration may shift after the fix. This sweep is paused until those
> results are in; it will be relaunched post-fix if the leak materially
> affects downstream transfer.

## Background

The [data scaling experiments](../02-data-scaling/README.md) established that B2
(Klinzing + Shirazi + Pavlov, ~37k ch·h) is the pretraining sweet spot — the only
configuration that beats the EEGNet baseline on motor imagery (FT F1 = 0.891 vs
0.887). All 12 data-scaling runs used identical masking parameters
(TemporalBlockMasking, mask_ratio=0.5, block_size=10) inherited from the initial
two-dataset pretrain.

The group's [open questions](../02-data-scaling/README.md#open-questions) explicitly
flag masking strategy as an unexplored axis: "Alternative strategies — or simply
different ratios — could produce fundamentally different representations." In the
vision MAE literature, higher mask ratios (e.g. 75%) are standard and often
outperform lower ratios. The current 50% ratio was chosen as a conservative default
but has never been validated against alternatives.

This experiment holds the data fixed at B2 and varies the masking parameters using
a one-factor-at-a-time star design around the baseline configuration.

Two structural changes are introduced for all runs (vs. previous B2):
1. **Intersubject validation** (`split_type=intersubject`) — uses held-out subjects
   instead of held-out time segments, preventing overfitting to training subjects.
2. **Doubled compute budget** (400k max steps) — with tighter early stopping
   (patience=10), most runs will stop well before 400k. The doubled budget ensures
   no run is artificially truncated.

## Question

Does increasing the mask ratio beyond 0.5 — forcing the model to reconstruct more
tokens from sparser context — produce better downstream representations for sleep
staging, motor imagery, and P300 detection? Does block size matter independently
of masking difficulty?

## Hypothesis

Higher mask ratios (0.7–0.8) will improve downstream F1 for finetuning and linear
probing, with an optimal point around 0.7–0.8 beyond which performance degrades.
Block size changes (10 → 20) will have a smaller effect than ratio changes, since
the overall fraction of masked tokens stays the same.

Expected ordering for finetuning: M1 (0.7) ≈ M2 (0.8) > M0 (0.5) > M3 (0.9).
M4 (block_size=20) will perform comparably to M0 (same ratio, different granularity).

## Experiment

### Setup

- **Model:** POYO CWT-CNN + dynamic channel embeddings, session_emb disabled
- **Data:** B2 = Klinzing + Shirazi + Pavlov (`three_dataset_pretrain.yaml`)
- **Task:** MAE pretraining (masked reconstruction)
- **Training:** 400k max steps, batch_size=64, lr=1e-4, warmup 2k + cosine decay
  over 398k steps, bf16-mixed, intersubject validation, early stopping patience=10
- **WandB:** `foundry_pretraining`, group `MASKING_SEQLEN`

### Pretraining runs

| Run | mask_ratio | block_size | Notes |
|-----|:---:|:---:|-------|
| M0 (baseline) | 0.5 | 10 | Shared control (same as original B2 but with intersubject val + 400k steps) |
| M1 | 0.7 | 10 | Harder reconstruction task |
| M2 | 0.8 | 10 | Much harder; forces longer-range prediction |
| M3 | 0.9 | 10 | Near-extreme; only 10% context remains |
| M4 | 0.5 | 20 | Coarser temporal blocks; larger masked spans |

### Launch commands — Pretraining

```bash
# M0: Baseline (mask_ratio=0.5, block_size=10)
uv run python main.py experiment=pretraining/poyo_masking_seqlen_sweep \
  data=openneuro/three_dataset_pretrain \
  run.name=pretrain_M0_baseline run.group=MASKING_SEQLEN -m

# M1: mask_ratio=0.7
uv run python main.py experiment=pretraining/poyo_masking_seqlen_sweep \
  data=openneuro/three_dataset_pretrain \
  model.masking.mask_ratio=0.7 \
  run.name=pretrain_M1_ratio70 run.group=MASKING_SEQLEN -m

# M2: mask_ratio=0.8
uv run python main.py experiment=pretraining/poyo_masking_seqlen_sweep \
  data=openneuro/three_dataset_pretrain \
  model.masking.mask_ratio=0.8 \
  run.name=pretrain_M2_ratio80 run.group=MASKING_SEQLEN -m

# M3: mask_ratio=0.9
uv run python main.py experiment=pretraining/poyo_masking_seqlen_sweep \
  data=openneuro/three_dataset_pretrain \
  model.masking.mask_ratio=0.9 \
  run.name=pretrain_M3_ratio90 run.group=MASKING_SEQLEN -m

# M4: block_size=20
uv run python main.py experiment=pretraining/poyo_masking_seqlen_sweep \
  data=openneuro/three_dataset_pretrain \
  model.masking.block_size=20 \
  run.name=pretrain_M4_block20 run.group=MASKING_SEQLEN -m
```

### Launch commands — Downstream evaluation

After pretraining, evaluate each checkpoint on 3 tasks × 2 modes × 3 folds = 18 runs per checkpoint:

```bash
# Template — replace $NAME with pretrain run name (e.g. pretrain_M1_ratio70)
for NAME in pretrain_M0_baseline pretrain_M1_ratio70 pretrain_M2_ratio80 pretrain_M3_ratio90 pretrain_M4_block20; do
  # Kemp Sleep
  uv run python main.py experiment=sleep_staging/kemp_finetune_from_data_scaling \
    run.pretrain_run_name=$NAME run.pretrain_group=MASKING_SEQLEN -m
  uv run python main.py experiment=sleep_staging/kemp_linear_probe_from_data_scaling \
    run.pretrain_run_name=$NAME run.pretrain_group=MASKING_SEQLEN -m
  # PhysioNet MI
  uv run python main.py experiment=motor_imagery/physionet_finetune_from_data_scaling \
    run.pretrain_run_name=$NAME run.pretrain_group=MASKING_SEQLEN -m
  uv run python main.py experiment=motor_imagery/physionet_linear_probe_from_data_scaling \
    run.pretrain_run_name=$NAME run.pretrain_group=MASKING_SEQLEN -m
  # Brain Invaders P300
  uv run python main.py experiment=p300/brain_invaders_finetune_from_data_scaling \
    run.pretrain_run_name=$NAME run.pretrain_group=MASKING_SEQLEN -m
  uv run python main.py experiment=p300/brain_invaders_linear_probe_from_data_scaling \
    run.pretrain_run_name=$NAME run.pretrain_group=MASKING_SEQLEN -m
done
```

### Key config overrides

| Config | Purpose |
|--------|---------|
| `configs/experiment/pretraining/poyo_masking_seqlen_sweep.yaml` | Base pretraining config (intersubject, patience=10, 400k steps) |
| `configs/data/openneuro/three_dataset_pretrain.yaml` | B2 data (3 brainsets) |

### Key comparisons

- **M0 → M1 → M2 → M3:** Monotonic mask_ratio increase (0.5, 0.7, 0.8, 0.9). Tests whether harder reconstruction improves or eventually degrades downstream transfer.
- **M0 vs M4:** Same mask_ratio (0.5) but different block granularity (10 vs 20 timesteps). Isolates the effect of temporal block structure.
- **M0 vs original B2:** Same data and masking but intersubject val + 400k steps. Tests whether the structural changes (longer training, tighter validation) affect the result.

## Results

TBD

## Conclusions

TBD

## Notes for future experiments

TBD
