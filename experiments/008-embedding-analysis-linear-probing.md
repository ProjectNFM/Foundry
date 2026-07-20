# Embedding Analysis: t-SNE/PCA and Linear Probing

**Status:** Draft
**Date started:** 2026-07-20
**Parent experiment:** [Pretraining Loss vs Downstream Task Performance](../experiments/007-pretraining-loss-vs-downstream.md)
**Follow-up experiments:** TBD

## Background

Experiment 007 showed that finetuning from pretrained checkpoints produces
*worse* downstream sleep staging performance than training from scratch (−2 to
−5 pp F1), despite the pretraining achieving low reconstruction loss. This
negative transfer is consistent across both tokenizer architectures and all
folds.

Before investing further in alternative pretraining objectives or longer
training schedules, we need to understand *what* the pretrained backbone is
actually representing. Two complementary analyses will clarify this:

1. **Embedding visualization (t-SNE/PCA):** Extract backbone outputs for the
   Kemp Sleep validation set from both pretrained and randomly-initialized
   models, then visualize with t-SNE and PCA colored by sleep stage. If
   pretraining creates useful representations, sleep stages should cluster more
   clearly in the pretrained embeddings than in random ones.

2. **Linear probing:** Freeze the pretrained backbone entirely and train only a
   linear classification head on sleep staging. This directly measures how much
   discriminative information for sleep staging exists in the pretrained
   representations, without the confound of full finetuning potentially
   overwriting everything. Comparing linear probe accuracy (pretrained vs
   random init) gives a clean signal of representation quality.

These are lightweight diagnostics that avoid the cost of full finetuning runs
and will guide whether to pursue better pretraining objectives vs other
approaches entirely.

## Question

Do the pretrained backbone representations contain any discriminative structure
for sleep stage classification, as measured by embedding separability (t-SNE/PCA
visualization) and linear probing accuracy?

## Hypothesis

The pretrained representations will show *some* structure in t-SNE/PCA (since
the model does learn EEG features), but the clusters will not align well with
sleep stage boundaries. Linear probing from the pretrained backbone will perform
only marginally better than linear probing from a randomly-initialized backbone,
confirming that the reconstruction objective does not learn sleep-stage-relevant
features.

## Experiment

### Setup

- **Model:** POYOEEGModel backbone from experiment 005 pretrained checkpoints
  (same architecture as experiments 005–007)
- **Data:** KempSleepEDF2013, same inter-subject split as exp 006/007, fold 0
  (single fold sufficient for this diagnostic)
- **Task:**
  - Embedding extraction: forward pass through backbone, collect latent
    representations before the task head
  - Linear probing: freeze backbone, train a single linear layer for 5-class
    sleep staging
- **Conditions:**
  - Pretrained CWT-CNN backbone (from exp 005, run `wlmobz7y`)
  - Pretrained ResampleCNN backbone (from exp 005, run `vup5m7er`)
  - Randomly-initialized CWT-CNN backbone
  - Randomly-initialized ResampleCNN backbone
- **Training (linear probe):** lr=1e-3, batch_size=512, max_epochs=100,
  early stopping on val F1 (patience=10), only the linear head is trainable
- **Hardware:** 1× GPU (embedding extraction and linear probing are lightweight)
- **WandB:** project=foundry_finetuning, group=KEMP_EMBEDDING_ANALYSIS

### Launch command

```bash
# --- Linear Probing (4 SLURM jobs: 2 tokenizers × 2 init modes) ---

# Pretrained backbone conditions (2 jobs):
uv run python main.py experiment=sleep_staging/poyo_kemp_linear_probe \
    run.init_mode=pretrained -m

# Random-init backbone conditions (2 jobs):
uv run python main.py experiment=sleep_staging/poyo_kemp_linear_probe \
    run.init_mode=random run.pretrained_checkpoint=null -m

# --- Embedding Extraction (run after linear probe or independently) ---

# Pretrained CWT-CNN:
uv run python scripts/extract_embeddings.py \
    experiment=sleep_staging/poyo_kemp_linear_probe \
    model/tokenizer=per_channel_cwt_cnn \
    run.init_mode=pretrained \
    extract.output_dir=outputs/embeddings/008_pretrained_cwt_cnn \
    extract.max_batches=200

# Random CWT-CNN:
uv run python scripts/extract_embeddings.py \
    experiment=sleep_staging/poyo_kemp_linear_probe \
    model/tokenizer=per_channel_cwt_cnn \
    run.init_mode=random run.pretrained_checkpoint=null \
    extract.output_dir=outputs/embeddings/008_random_cwt_cnn \
    extract.max_batches=200

# Pretrained ResampleCNN:
uv run python scripts/extract_embeddings.py \
    experiment=sleep_staging/poyo_kemp_linear_probe \
    model/tokenizer=per_channel_resample_cnn \
    run.init_mode=pretrained \
    extract.output_dir=outputs/embeddings/008_pretrained_resample_cnn \
    extract.max_batches=200

# Random ResampleCNN:
uv run python scripts/extract_embeddings.py \
    experiment=sleep_staging/poyo_kemp_linear_probe \
    model/tokenizer=per_channel_resample_cnn \
    run.init_mode=random run.pretrained_checkpoint=null \
    extract.output_dir=outputs/embeddings/008_random_resample_cnn \
    extract.max_batches=200
```

### Key config overrides

- `run.freeze_pretrained: true` (freeze transferred backbone when checkpoint loaded)
- `run.freeze_backbone: true` (freeze backbone components for random-init condition)
- Single fold (fold 0) only
- Frozen components: tokenizer, backbone (PerceiverIO), rotary_emb, latent_emb
- Trainable components: channel_emb, session_emb, task_emb, readout heads

## Results

### Summary

TBD

### Metrics

TBD

### Analysis

TBD

**Analysis script:** `analysis/008_embedding_analysis.py`

```bash
uv run python analysis/008_embedding_analysis.py
```

### Figures

TBD

## Conclusions

TBD

## Notes for future experiments

- If linear probing confirms poor representations, pivot to alternative
  pretraining objectives (contrastive learning, temporal prediction, etc.).
- If t-SNE shows structure that doesn't align with sleep stages, the
  reconstruction objective may be optimizing for irrelevant features
  (high-frequency detail, noise patterns) — consider frequency-band-specific
  objectives.
- If linear probing from pretrained backbone does outperform random init
  (contrary to hypothesis), the problem may be in the full finetuning procedure
  (learning rate too high, catastrophic forgetting of useful features early in
  training) — investigate layerwise LR decay or gradual unfreezing.
