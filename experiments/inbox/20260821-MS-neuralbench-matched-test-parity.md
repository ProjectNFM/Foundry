# NeuralBench Matched EEGNet — Three-Task Test Parity

**Status:** In Progress
**Date started:** 2026-08-21
**Parent experiment:** [NeuralBench Phase 1 — Motor Imagery & Sleep Stage EEGNet Comparison](20260820-MS-neuralbench-phase1-mi-sleep-comparison.md)
**Follow-up experiments:** TBD
**Tags:** neuralbench, eegnet, parity, test_evaluation, p300, motor_imagery, sleep_stage

## Background

[Phase 1](20260820-MS-neuralbench-phase1-mi-sleep-comparison.md) established
that the NeuralBench adapter supports Motor Imagery and Sleep Stage, but its
Foundry-validation versus NeuralBench-test comparison was not apples-to-apples.
The one retained NeuralBench MI training log makes the confound concrete: seed
33 selected a validation balanced accuracy of 0.509 but achieved 0.595 on the
test set. Therefore, the original 6.6 pp MI gap cannot be interpreted as a
Foundry implementation deficit.

Foundry now evaluates its best-validation checkpoint on the exact NeuralBench
test split. The matched EEGNet configurations additionally align the exposed
NeuralBench v0.2.3 recipe for P300, Motor Imagery, and Sleep Stage: dropout
(0.25), BatchNorm momentum/epsilon (0.01/1e-3), spatial max-norm (1.0),
AdamW (lr=1e-4, weight decay=0.05), step-level cosine OneCycleLR
(`pct_start=0.1`), DataLoader settings, training cap, and early stopping.

The remaining structural difference is intentional and documented: Foundry
uses its multi-task ReadoutRouter while NeuralBench uses braindecode's single
classifier head. This experiment tests whether the matched exposed parameters
are sufficient for practical test-set parity despite that implementation detail.

## Question

When both systems use the exact NeuralBench data/split contract and evaluate
their best-validation checkpoint on the held-out test set, does matched Foundry
EEGNet achieve test balanced accuracy within 2 percentage points of NeuralBench
EEGNet on P300, Motor Imagery, and Sleep Stage?

## Hypothesis

For each task, the mean three-seed absolute difference in test balanced
accuracy between Foundry and NeuralBench will be **≤2 pp**. Matching the
optimizer schedule and EEGNet regularization/normalization parameters should
remove the apparent MI discrepancy caused by the val-vs-test comparison and
reduce remaining implementation-level gaps.

## Experiment

### Setup

- **Model:** Foundry EEGNetEncoder configured to NeuralBench's exposed EEGNet
  parameters (F1=8, D=2, F2=16, kernel=64, dropout=0.25, BN momentum=0.01,
  BN epsilon=1e-3, spatial max-norm=1.0).
- **Data:** NeuralBench v0.2.3 / NeuralSet subject splits, seed 33; the three
  training seeds are 33, 34, and 35.
- **Tasks:** P300 / Korczowski2014A; Motor Imagery /
  Schalk2004Bci2000; Sleep Stage / Kemp2000Analysis.
- **Training:** AdamW (lr=1e-4, weight_decay=0.05), OneCycleLR
  (`pct_start=0.1`, cosine, step interval), 40 epoch cap, gradient clip=1.0,
  FP32, and NeuralBench-matched early stopping (P300 patience 10; MI/Sleep
  patience 5).
- **Evaluation:** `trainer.test(ckpt_path="best")` on NeuralBench's held-out
  test split after each run.
- **WandB:** `foundry-neuralbench`, groups `NB_P300_EEGNET_MATCHED`,
  `NB_MI_EEGNET_MATCHED`, and `NB_SLEEP_EEGNET_MATCHED`.

### Launch commands

```bash
export FOUNDRY_SNAPSHOT_ROOT=/network/scratch/s/sobralm/foundry-launches

uv run python main.py experiment=neuralbench/p300_eegnet_matched -m
uv run python main.py experiment=neuralbench/mi_eegnet_matched -m
uv run python main.py experiment=neuralbench/sleep_stage_eegnet_matched -m
```

Each multirun submits three independent seed jobs (33, 34, 35) to the `long`
partition, for nine training-and-test jobs in total.

### Key config overrides

| Setting | Matched value |
|---|---:|
| Dropout | 0.25 |
| BatchNorm momentum / epsilon | 0.01 / 1e-3 |
| Spatial convolution max-norm | 1.0 |
| Optimizer | AdamW, lr=1e-4, wd=0.05 |
| LR schedule | OneCycleLR, cosine, `pct_start=0.1`, per step |
| Batch size / workers / pinned memory | 64 / 10 / true |
| Max epochs | 40 |
| Test evaluation | best validation checkpoint on exact test split |

## Results

### Submission

| Task | Slurm array | Snapshot bundle |
|---|---|---|
| P300 | `10438103_[0-2]` | `/network/scratch/s/sobralm/foundry-launches/20260821T174353_NB_P300_EEGNET_MATCHED_3edfa24f_aeea2fbc` |
| Motor Imagery | `10438105_[0-2]` | `/network/scratch/s/sobralm/foundry-launches/20260821T174411_NB_MI_EEGNET_MATCHED_3edfa24f_283905a4` |
| Sleep Stage | `10438107_[0-2]` | `/network/scratch/s/sobralm/foundry-launches/20260821T174428_NB_SLEEP_EEGNET_MATCHED_3edfa24f_a439c4c8` |

Nine jobs were submitted on 2026-08-21. Results pending.

Sleep Stage seed 34 (`10438107_1`; Slurm raw element ID `10438115`) failed
before training on `cn-e002`: it was assigned a V100 and the installed PyTorch
build has no compatible CUDA kernel. The current matched configs explicitly
request RTX 8000 GPUs. The interim L40S replacement (`10438514`) was cancelled
while pending; seed 34 was relaunched as `10439024` from snapshot
`/network/scratch/s/sobralm/foundry-launches/20260821T185904_NB_SLEEP_EEGNET_MATCHED_22c8c4a0_8379129b`.
It is pending on the `long` partition with `gres/gpu:rtx8000:1`.

## Conclusions

TBD

## Notes for future experiments

TBD
