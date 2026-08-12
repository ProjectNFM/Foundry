# Channel Encoder Leak Fix: Does Preventing Masked Token Leakage Improve Pretraining?

**Status:** Draft
**Date started:** 2026-08-12
**Parent experiment:** [Masking Parameter Sweep](20260811-MS-masking-parameter-sweep.md)
**Follow-up experiments:** [Masking Parameter Sweep](20260811-MS-masking-parameter-sweep.md) (on hold), [Multi-Length Pretraining](20260811-MS-multi-length-pretraining.md) (on hold) — both paused pending these results
**Tags:** pretraining, mae, masked, dynamic_ch, channel_encoder, information_leak, bugfix, ablation

## Background

The [masking parameter sweep](20260811-MS-masking-parameter-sweep.md) tests how
mask ratio and block size affect downstream transfer, using the standard B2 data
configuration with CWT-CNN + dynamic channel embeddings. All prior pretraining
runs with `channel_emb_mode="dynamic"` — including
[experiment 018](../_legacy/018-dynamic-channel-embeddings.md) (which showed a 72%
reconstruction loss reduction vs disabled) and the
[data scaling group](../02-data-scaling/README.md) — contained an information leak
in the `RelativeChannelEncoder`.

**The bug:** When `MaskedPOYOEEGModel` computes channel embeddings via the
`RelativeChannelEncoder`, the encoder's temporal pooling stage attends over ALL
token embeddings — including those that will be masked and used as reconstruction
targets. This means:

1. The channel embeddings encode information about the signal at masked positions.
2. Reconstruction queries use these leaked channel embeddings to reconstruct
   masked tokens, making the task artificially easier.
3. The encoder backbone doesn't need to learn as rich representations because
   the decoder already has partial answers via the channel embeddings.

**The fix** (implemented in this commit) threads a `token_mask` through the
pipeline so the `RelativeChannelEncoder` only pools over visible tokens when
computing channel embeddings. This eliminates the shortcut, forcing the model to
reconstruct from genuine context only.

All previous reconstruction loss numbers for dynamic channel embedding models
(e.g. 0.11 in exp 018) benefited from this leak. Post-fix, the reconstruction
task becomes strictly harder — the decoder loses its unfair channel embedding
signal. The key question is whether this harder task produces better
representations.

## Question

Does fixing the information leak in the RelativeChannelEncoder — forcing
reconstruction to rely solely on visible-token context — change pretraining
dynamics and ultimately improve downstream transfer quality?

## Hypothesis

1. **Reconstruction loss will increase** (post-fix val loss > pre-fix val loss)
   because the decoder can no longer exploit leaked masked-token information via
   channel embeddings. Expected magnitude: 20–50% relative increase in val loss,
   since the leak effectively gave the decoder partial access to reconstruction
   targets.

2. **Downstream transfer will improve** because the encoder backbone must
   compensate for the removed shortcut by learning more informative
   representations. The pre-fix model could "offload" reconstruction to the
   channel encoder leak; post-fix, the backbone must do the heavy lifting.

3. **Training will be slower to converge** but ultimately reach a better
   representation quality, as the model optimizes a genuinely harder objective.

## Experiment

### Setup

- **Model:** POYO CWT-CNN + dynamic channel embeddings, session_emb disabled
- **Data:** B2 = Klinzing + Shirazi + Pavlov (`three_dataset_pretrain.yaml`)
- **Task:** MAE pretraining (masked reconstruction), TemporalBlockMasking
  (mask_ratio=0.5, block_size=10)
- **Training:** 400k max steps, batch_size=64, lr=1e-4, warmup 2k + cosine
  decay, bf16-mixed, intersubject validation, early stopping patience=10
- **WandB:** `foundry_pretraining`, group `CHANNEL_LEAK_FIX`

### Pretraining runs

| Run | Fix applied | Notes |
|-----|:-----------:|-------|
| leak-baseline | No (`token_mask=None` forced) | Same as masking sweep M0; channel encoder sees all tokens |
| leak-fixed | Yes (default post-commit behavior) | Channel encoder only pools visible tokens |

Both runs use identical config, data, seeds, and compute budget. The only
difference is whether the `token_mask` is passed to the channel encoder.

### Launch commands — Pretraining

```bash
# leak-baseline: Force token_mask=None (pre-fix behavior)
uv run python main.py experiment=pretraining/poyo_masking_seqlen_sweep \
  data=openneuro/three_dataset_pretrain \
  +model.disable_channel_encoder_token_mask=true \
  run.name=pretrain_leak_baseline run.group=CHANNEL_LEAK_FIX -m

# leak-fixed: Default post-fix behavior (token_mask passed to encoder)
uv run python main.py experiment=pretraining/poyo_masking_seqlen_sweep \
  data=openneuro/three_dataset_pretrain \
  run.name=pretrain_leak_fixed run.group=CHANNEL_LEAK_FIX -m
```

**Note:** The `+model.disable_channel_encoder_token_mask=true` override needs a
small config hook to force `ch_token_mask = None` in `MaskedPOYOEEGModel.forward()`.
Alternatively, use git stash/checkout to run the baseline on the pre-fix commit.

### Launch commands — Downstream evaluation

```bash
for NAME in pretrain_leak_baseline pretrain_leak_fixed; do
  # Kemp Sleep
  uv run python main.py experiment=sleep_staging/kemp_finetune_from_data_scaling \
    run.pretrain_run_name=$NAME run.pretrain_group=CHANNEL_LEAK_FIX -m
  uv run python main.py experiment=sleep_staging/kemp_linear_probe_from_data_scaling \
    run.pretrain_run_name=$NAME run.pretrain_group=CHANNEL_LEAK_FIX -m
  # PhysioNet MI
  uv run python main.py experiment=motor_imagery/physionet_finetune_from_data_scaling \
    run.pretrain_run_name=$NAME run.pretrain_group=CHANNEL_LEAK_FIX -m
  uv run python main.py experiment=motor_imagery/physionet_linear_probe_from_data_scaling \
    run.pretrain_run_name=$NAME run.pretrain_group=CHANNEL_LEAK_FIX -m
  # Brain Invaders P300
  uv run python main.py experiment=p300/brain_invaders_finetune_from_data_scaling \
    run.pretrain_run_name=$NAME run.pretrain_group=CHANNEL_LEAK_FIX -m
  uv run python main.py experiment=p300/brain_invaders_linear_probe_from_data_scaling \
    run.pretrain_run_name=$NAME run.pretrain_group=CHANNEL_LEAK_FIX -m
done
```

### Key config overrides

| Config | Purpose |
|--------|---------|
| `configs/experiment/pretraining/poyo_masking_seqlen_sweep.yaml` | Base config (intersubject val, patience=10, 400k steps) |
| `configs/data/openneuro/three_dataset_pretrain.yaml` | B2 data (3 brainsets) |

### Key comparisons

- **leak-baseline vs leak-fixed (pretraining):** Val reconstruction loss curves.
  Expect leak-fixed to have higher loss (harder task) and possibly slower
  convergence.
- **leak-baseline vs leak-fixed (downstream):** F1 on 3 tasks × 2 modes.
  Expect leak-fixed to produce better linear probes (representation quality)
  and comparable or better finetuning.
- **leak-fixed vs masking sweep M0:** Should be identical if M0 is run after
  the fix is merged. If M0 was run before the fix, it matches leak-baseline.

## Results

TBD

## Conclusions

TBD

## Notes for future experiments

- If the fix significantly improves downstream transfer, all prior dynamic
  channel embedding results (exp 018, 020, 021, data scaling) should be
  considered as lower bounds on the method's true potential.
- The magnitude of the leak's impact may interact with mask_ratio: at higher
  ratios (M1–M3 from the masking sweep), the leak provides more information
  (more masked tokens → more signal leaking into channel embeddings). Re-running
  the full masking sweep post-fix could reveal a different optimal mask ratio.
- Consider whether the fix changes the relative value of dynamic vs disabled
  channel embeddings. If the fix narrows the gap (because part of the previous
  advantage was the leak), this would need to be accounted for in future
  architecture comparisons.
