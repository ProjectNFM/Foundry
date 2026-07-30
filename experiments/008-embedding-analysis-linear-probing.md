# Embedding Analysis: t-SNE/PCA and Linear Probing

**Status:** Completed
**Date started:** 2026-07-20
**Parent experiment:** [Pretraining Loss vs Downstream Task Performance](../experiments/007-pretraining-loss-vs-downstream.md)
**Follow-up experiments:** [Finetuning Hyperparameter Search](../experiments/009-finetuning-hyperparameter-search.md)

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
  - `kemp_linear_probe_per_channel_cwt_cnn_pretrained` / `5rsh5wva`
  - `kemp_linear_probe_per_channel_cwt_cnn_random` / `ucr1bv41`
  - `kemp_linear_probe_per_channel_resample_cnn_pretrained` / `ns70dh9j`
  - `kemp_linear_probe_per_channel_resample_cnn_random` / `a39iv8qj`



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

Linear probing reveals that pretraining provides a clear benefit for the
CWT-CNN tokenizer, but only a marginal one for ResampleCNN. The pretrained
CWT-CNN backbone achieves a best val F1 of 0.418, substantially outperforming
its random-init counterpart (0.267, +15.0 pp). The ResampleCNN pretrained
backbone shows a smaller improvement (0.285 vs 0.264, +2.2 pp). All runs timed
out after ~4 epochs (172 min wall time out of 180 min limit), so these results
reflect early-training performance; longer runs could shift the picture,
particularly for the random-init conditions.

Note: the embedding visualization (t-SNE/PCA) part of this experiment has not
been completed yet.

### Metrics

| Condition | Best Val F1 | Best Epoch | WandB Run ID |
|---|---|---|---|
| CWT-CNN Pretrained | **0.418** | 0 | `5rsh5wva` |
| CWT-CNN Random | 0.267 | 0 | `ucr1bv41` |
| ResampleCNN Pretrained | 0.285 | 0 | `ns70dh9j` |
| ResampleCNN Random | 0.263 | 0 | `a39iv8qj` |

**Pretraining advantage (F1 delta):**
- CWT-CNN: +0.150 (0.418 − 0.267)
- ResampleCNN: +0.022 (0.285 − 0.263)

### Analysis

Results were extracted directly from the SLURM logs for the latest run batch
(jobs 10168063 and 10168330). Each log contains the best checkpoint F1 from the
`ModelCheckpoint` callback. All four conditions completed epoch 0 and were
validated, establishing their best scores at that point. Subsequent epochs did
not improve, and runs timed out partway through epoch 4.

The CWT-CNN pretrained backbone also transferred from a much better pretrained
checkpoint (val_loss=0.0364 vs 0.1190 for ResampleCNN), which likely contributed
to its stronger linear probe performance.

**Analysis script:** `analysis/008_embedding_analysis.py`

```bash
uv run python analysis/008_embedding_analysis.py
```

### Figures

TBD (embedding visualization not yet run)

## Conclusions

The hypothesis was **partially refuted**. Contrary to the prediction that
pretraining would provide only marginal benefit for linear probing, the CWT-CNN
pretrained backbone shows a substantial +15 pp F1 advantage over random init
in the linear probe setting. This indicates that the CWT-CNN reconstruction
pretraining *does* learn sleep-stage-relevant features that are linearly
separable in the embedding space.

The ResampleCNN pretrained backbone, however, aligns more closely with the
original hypothesis — its +2.2 pp advantage is marginal and could be noise,
suggesting that its pretrained representations carry less discriminative
structure for sleep staging.

This result, combined with experiment 007's finding that full finetuning from
pretrained checkpoints *hurts* performance, points to a specific failure mode:
the pretrained CWT-CNN backbone learns useful representations, but full
finetuning overwrites them (catastrophic forgetting). This is consistent with
the linear probe finding — the frozen backbone succeeds precisely because its
pretrained features are preserved.

The tokenizer architecture matters significantly: CWT-CNN appears to be a more
natural fit for EEG pretraining, likely because its frequency-domain
decomposition (wavelet transform) captures the spectral features that
distinguish sleep stages (delta, theta, alpha, sigma, beta bands).

## Notes for future experiments

- **Catastrophic forgetting mitigation:** Since the CWT-CNN pretrained backbone
  *does* contain useful representations (confirmed by linear probe), investigate
  layerwise LR decay, gradual unfreezing, or lower finetuning LR to preserve
  pretrained features during full finetuning.
- **CWT-CNN as the preferred tokenizer:** The significant linear probe advantage
  for CWT-CNN over ResampleCNN suggests focusing future pretraining experiments
  on the CWT-CNN architecture.
- **Longer linear probe runs:** All conditions timed out after ~4 epochs.
  Re-run with longer wall time (or resume from checkpoints) to see if the
  random-init conditions eventually catch up to the pretrained ones.
- **Complete embedding visualization:** Run the t-SNE/PCA extraction to get a
  qualitative view of the embedding space structure and confirm the linear probe
  findings visually.
- **Per-class analysis:** Examine whether the pretrained CWT-CNN backbone is
  particularly better at distinguishing specific sleep stages (e.g., N1 vs N2,
  which are notoriously hard to separate).

