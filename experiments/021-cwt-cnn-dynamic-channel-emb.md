# CWT CNN with Dynamic Channel Embeddings: Pretraining, Linear Probing, and Embedding Analysis

**Status:** In Progress
**Date started:** 2026-07-28
**Parent experiment:** [Dynamic Channel Embeddings via Relative Inter-Channel Attention](../experiments/018-dynamic-channel-embeddings.md)
**Follow-up experiments:** [KempSleep Baselines and Finetuning: CWT-CNN with Dynamic Channel Embeddings](../experiments/022-kemp-baselines-finetune-cwt-dynch.md)

## Background

Experiment 018 introduced `channel_emb_mode="dynamic"` via a `RelativeChannelEncoder`
and showed it massively outperforms the disabled baseline in reconstruction loss
(0.11 vs 0.40, a 72% relative reduction) using a ResampleCNN tokenizer. The
follow-up experiments (019, 020) analyzed the learned embeddings and evaluated
downstream performance via linear probing.

Separately, experiment 008 compared CWT-CNN vs ResampleCNN tokenizers in a linear
probing setting and found that the CWT-CNN pretrained backbone gives a much larger
advantage over random init (+15 pp F1) compared to ResampleCNN (+2.2 pp). This
suggests that CWT-CNN's frequency-domain decomposition (wavelet transform) better
captures the spectral features relevant for downstream EEG tasks like sleep staging.

However, all dynamic channel embedding experiments (018, 019, 020) used
ResampleCNN. This experiment repeats the full pipeline — pretraining, linear
probing, and embedding analysis — using the CWT-CNN tokenizer instead, to
determine whether the tokenizer choice interacts with the dynamic channel
embedding mechanism.

## Question

Does the CWT-CNN tokenizer produce the same dynamic channel embedding behaviour
as ResampleCNN (channel embeddings not clustering by electrode type), while
yielding better downstream linear probing performance?

## Hypothesis

1. **CWT-CNN dynamic channel embeddings will show the same structural patterns
   as ResampleCNN** — i.e., channel embeddings will *not* cluster by electrode
   type (as found in exp 019), because the RelativeChannelEncoder operates on
   the signal statistics rather than electrode identity regardless of tokenizer.
2. **CWT-CNN will outperform ResampleCNN in linear probing** for both the
   disabled and dynamic pretrained conditions, consistent with exp 008's finding
   that CWT-CNN pretraining captures more sleep-stage-relevant features (+15 pp
   F1 advantage for CWT-CNN vs +2.2 pp for ResampleCNN).
3. **The dynamic channel embedding will still outperform disabled for CWT-CNN**
   in reconstruction loss, though the gap may differ in magnitude from
   ResampleCNN since CWT-CNN may already capture some frequency-domain channel
   identity in its wavelet decomposition.

## Experiment

This experiment has three phases. Phase 1 (pretraining) is the priority and
should be launched immediately.

---

### Phase 1: Pretraining (CWT-CNN × disabled/dynamic)

#### Setup

- **Model:** MaskedPOYOEEGModel, embed_dim=256, depth=4, 8 cross/self heads,
  dim_head=128, TemporalBlockMasking (block_size=10, mask_ratio=0.5),
  `zero_output_timestamps: false`, `normalize_inputs: true`,
  **CWT-CNN tokenizer** (`per_channel_cwt_cnn`), channel_encoder_heads=4
- **Data:** Balanced Klinzing subset (`sleep_brainset_small`) — 14 subjects,
  28 recordings, **intersubject** split, fold 0, sequence_length=2.0s
- **Task:** Masked reconstruction (MSE loss), mask_ratio=0.5
- **Training:** batch_size=512, lr=1e-4, weight_decay=0.01, max_epochs=200,
  bf16-mixed precision, warmup_epochs=0
- **Hardware:** 1× L40S per run, 6 CPUs, 32 GB RAM (SLURM)
- **WandB:** project=foundry_pretraining,
  group=PRETRAIN_CWT_DYNAMIC_CHANNEL_EMB

**Conditions:**

| Condition    | channel_emb_mode | session_emb_mode | tokenizer           | Purpose                                |
| ------------ | ---------------- | ---------------- | ------------------- | -------------------------------------- |
| ch-disabled  | `disabled`       | `disabled`       | `per_channel_cwt_cnn` | CWT-CNN baseline (no channel identity) |
| ch-dynamic   | `dynamic`        | `disabled`       | `per_channel_cwt_cnn` | CWT-CNN + dynamic channel attention    |

#### Launch command

```bash
uv run python main.py experiment=pretraining/poyo_pretrain_dynamic_session_emb \
    model/tokenizer=per_channel_cwt_cnn \
    model/session_emb=disabled \
    'model.channel_emb_mode=disabled,dynamic' \
    run.group=PRETRAIN_CWT_DYNAMIC_CHANNEL_EMB \
    'run.name=pretrain_cwt_dynch_ch-${model.channel_emb_mode}' \
    'run.tags=[pretraining,mae,masked,cwt_cnn,dynamic_channel_emb,intersubject,exp021]' \
    hydra.launcher.timeout_min=1440 \
    -m
```

#### Key config overrides

Base config:
`configs/experiment/pretraining/poyo_pretrain_dynamic_session_emb.yaml`
(same as exp 018)

Overrides vs exp 018:

- `model/tokenizer=per_channel_cwt_cnn` (CWT-CNN instead of ResampleCNN)
- `run.group: PRETRAIN_CWT_DYNAMIC_CHANNEL_EMB`
- `run.name` includes `cwt` prefix
- `hydra.launcher.timeout_min=1440` (24h — exp 018 timed out at 3h/44 epochs)
- Tags include `cwt_cnn` and `exp021`

---

### Phase 2: Linear Probing (after pretraining completes)

#### Setup

Same as exp 020, but with CWT-CNN tokenizer and pointing to the exp 021
pretrained checkpoints. 4 conditions: 2 channel_emb_mode × {pretrained, random}.

A new config file `configs/experiment/sleep_staging/poyo_kemp_linear_probe_cwt_dynch.yaml`
will be created when Phase 1 completes, mirroring `poyo_kemp_linear_probe_dynch.yaml`
but with `per_channel_cwt_cnn` tokenizer and updated checkpoint paths.

#### Launch command (to be finalized after pretraining)

```bash
# Pretrained backbone conditions (2 jobs: ch-disabled, ch-dynamic):
uv run python main.py experiment=sleep_staging/poyo_kemp_linear_probe_cwt_dynch \
    run.init_mode=pretrained -m

# Random-init backbone conditions (2 jobs: ch-disabled, ch-dynamic):
uv run python main.py experiment=sleep_staging/poyo_kemp_linear_probe_cwt_dynch \
    run.init_mode=random run.pretrained_checkpoint=null -m
```

---

### Phase 3: Embedding Analysis (after pretraining completes)

#### Setup

Same as exp 019, but using CWT-CNN pretrained checkpoints from Phase 1.
Extract backbone and channel embeddings, visualize with t-SNE/PCA colored by
sleep stage, channel type, and session.

#### Launch command (to be finalized after pretraining)

```bash
# Disabled condition (backbone only):
uv run python scripts/extract_embeddings.py \
    experiment=sleep_staging/poyo_kemp_allsess \
    model/tokenizer=per_channel_cwt_cnn \
    model.channel_emb_mode=disabled \
    model/session_emb=disabled \
    '++run.pretrained_checkpoint=/network/scratch/s/sobralm/runs/PRETRAIN_CWT_DYNAMIC_CHANNEL_EMB/pretrain_cwt_dynch_ch-disabled/checkpoints/last.ckpt' \
    ++run.pretrained_transfer_mode=permissive \
    extract.output_dir=outputs/embeddings/021_disabled \
    extract.max_batches=200

# Dynamic condition (backbone + channel embeddings):
uv run python scripts/extract_embeddings.py \
    experiment=sleep_staging/poyo_kemp_allsess \
    model/tokenizer=per_channel_cwt_cnn \
    model.channel_emb_mode=dynamic \
    model/session_emb=disabled \
    '++run.pretrained_checkpoint=/network/scratch/s/sobralm/runs/PRETRAIN_CWT_DYNAMIC_CHANNEL_EMB/pretrain_cwt_dynch_ch-dynamic/checkpoints/last.ckpt' \
    ++run.pretrained_transfer_mode=permissive \
    extract.output_dir=outputs/embeddings/021_dynamic \
    +extract.extract_channel_emb=true \
    extract.max_batches=200

# Visualization:
uv run python analysis/021_cwt_dynamic_channel_emb_viz.py
```

## Results

### Summary

TBD

### Metrics

TBD

### Analysis

TBD

### Figures

TBD

## Conclusions

TBD

## Notes for future experiments

- Compare reconstruction loss magnitude between CWT-CNN and ResampleCNN
  for both disabled and dynamic conditions — the CWT-CNN disabled baseline
  from exp 005 had much lower reconstruction loss (0.036 vs 0.119), so the
  gap from dynamic may be different.
- If CWT-CNN + dynamic shows strong linear probe results, combine with
  full finetuning using gradual unfreezing (informed by exp 008 and 009
  catastrophic forgetting findings).
- Consider running a direct 4-way comparison: {CWT-CNN, ResampleCNN} ×
  {disabled, dynamic} in a single linear probe experiment.
