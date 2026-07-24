# Intersubject Pretraining — Session Embedding Generalization

**Status:** In Progress
**Date started:** 2026-07-24
**Parent experiment:** [Session Embedding Ablation](../experiments/011-session-embedding-ablation.md), [Within-Subject Split Control](../experiments/012-within-subject-split-control.md)
**Follow-up experiments:** TBD

## Background

Experiments 009–011 on Kemp sleep staging finetuning revealed a large
train-val reconstruction/classification loss gap (~0.94) from the very first
epoch when using an inter-subject split. Validation subjects are entirely
unseen during training, and the model's `session_emb` (`InfiniteVocabEmbedding`)
provides a per-session learned vector used in both input tokenization and
downstream readout queries. Unseen sessions fall back to the padding embedding,
creating a systematic distribution shift between train and val.

Experiment 011 ablated session embeddings during finetuning and found they
were not the primary driver of the val F1 gap — but the architectural mechanism
remains: session embeddings can encode subject-specific shortcuts during
training that do not transfer to held-out subjects.

Experiment 005 pretrained tokenizer models with an **intrasession** split
(`data.split_type: intrasession`), where every subject contributes epochs to
both train and val. That setting masks whether the same subject-level
generalization failure appears during pretraining itself.

Experiment 012 is running the complementary finetuning control (intrasession
vs intersubject). This experiment runs the same tokenizer pretraining sweep as
experiment 005 but with an **intersubject** split, to test whether the
new-subject issues observed downstream are already present at the pretraining
stage — as expected from session embeddings in the architecture.

## Question

Does masked-reconstruction pretraining with an inter-subject split exhibit the
same large train-val loss gap seen in inter-subject finetuning, indicating that
new-subject generalization failure is intrinsic to the architecture (session
embeddings) rather than specific to the downstream task?

## Hypothesis

Yes — intersubject pretraining will show a substantial train-val reconstruction
loss gap from early epochs, analogous to the finetuning gap in experiments
009–011. This gap reflects the expected behaviour of session embeddings:
training sessions get learned identity vectors that reduce reconstruction loss,
while validation sessions rely on the padding embedding. The effect should
appear across both tokenizers in the sweep (ResampleCNN and CWT-CNN), since it
stems from the shared POYO backbone and session embedding mechanism, not from
tokenizer choice.

## Experiment

### Setup

- **Model:** MaskedPOYOEEGModel, embed_dim=256, depth=4, 8 cross/self heads,
  dim_head=128, TemporalBlockMasking (block_size=10, mask_ratio=0.5),
  `zero_output_timestamps: false`, `normalize_inputs: true`
- **Data:** OpenNeuro multi-brainset (`klinzing_sleep_ds005555` and related
  sessions), **intersubject** split, fold 0, sequence_length=2.0s
- **Task:** Masked reconstruction (MSE loss), mask_ratio=0.5
- **Training:** batch_size=100, lr=1e-4, weight_decay=0.01, max_epochs=200,
  bf16-mixed precision, warmup_epochs=0
- **Hardware:** 1× L40S per run, 6 CPUs, 32 GB RAM (SLURM)
- **WandB:** project=foundry_pretraining, group=PRETRAIN_TOKENIZER_SWEEP_INTERSUBJECT
  - Run name(s): `pretrain_tokenizer_<tokenizer>_intersubject`
  - Run ID(s): TBD

**Conditions:**

| Condition | Tokenizer              | Split Type   | Runs | Purpose                                      |
| --------- | ---------------------- | ------------ | ---- | -------------------------------------------- |
| ResampleCNN | per_channel_resample_cnn | intersubject | 1  | Baseline tokenizer, intersubject pretraining   |
| CWT-CNN   | per_channel_cwt_cnn    | intersubject | 1    | Best exp-005 tokenizer, intersubject setting |

### Launch command

```bash
# SLURM sweep (2 tokenizers in parallel, intersubject split):
uv run python main.py experiment=pretraining/poyo_pretrain_tokenizer_sweep \
    data.split_type=intersubject \
    run.group=PRETRAIN_TOKENIZER_SWEEP_INTERSUBJECT \
    'run.name=pretrain_tokenizer_${hydra:runtime.choices.model/tokenizer}_intersubject' \
    'run.tags=[pretraining,mae,masked,tokenizer_sweep,intersubject,exp013]' \
    -m
```

### Key config overrides

Base config: `configs/experiment/pretraining/poyo_pretrain_tokenizer_sweep.yaml`

Non-default overrides applied for this experiment:

- `data.split_type: intersubject` — subjects are disjoint between train and
  val (vs `intrasession` in experiment 005)
- `run.group: PRETRAIN_TOKENIZER_SWEEP_INTERSUBJECT` — separate WandB group
  from the intrasession sweep in exp 005
- `run.name` suffix `_intersubject` — distinguish runs from exp 005 checkpoints

Hydra sweeper varies `model/tokenizer` over:

- `per_channel_resample_cnn`
- `per_channel_cwt_cnn`

All other settings match experiment 005 (masking, hyperparameters, trainer).

## Results

### Summary

TBD

### Metrics

| Metric | ResampleCNN | CWT-CNN |
|--------|-------------|---------|
| Best val/loss | TBD | TBD |
| Train loss at best val | TBD | TBD |
| Train-val gap at best val | TBD | TBD |
| Epoch of best val | TBD | TBD |

### Analysis

TBD

**Analysis script:** TBD (not yet created)

### Figures

TBD

## Conclusions

TBD

## Notes for future experiments

- Compare train-val gap magnitude against experiment 005 (intrasession
  pretraining) and experiment 012 (intrasession finetuning) to quantify how
  much of the gap is attributable to subject shift vs temporal structure.
- If the intersubject pretraining gap is large, run a session-embedding
  ablation during pretraining (mirror of experiment 011) to confirm the
  architectural mechanism.
- If gap is small despite intersubject split, the finetuning-specific
  readout/query construction may amplify the effect beyond what pretraining
  reconstruction loss reveals.
