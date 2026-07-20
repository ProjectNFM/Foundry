# Self-Supervised Pretraining Experiments

## Grand Hypothesis

A CWT+CNN tokenizer, defended by domain-specific augmentations (amplitude
scaling, pink/line noise) and pre-trained via a Continuous Contrastive Masked
Predictive Coding objective with spatial-temporal exclusion zones, learns
robust, generalizable neural manifolds across mixed (EEG/ECoG) datasets,
outperforming both reconstruction-based objectives and a ResampleCNN baseline
on downstream fine-tuning tasks.

**Tokenizer:** CWT+CNN with highpass conditioning (best architecture from
supervised Experiments 1–7).

**Sequencing rationale:** Each experiment gates the next. If the contrastive
objective cannot beat reconstruction on a single dataset (Exp 1), the
architecture comparison (Exp 2) is meaningless. If CWT+CNN does not
outperform ResampleCNN under InfoNCE (Exp 2), scaling to mixed modalities
(Exp 3) will not rescue it. If mixed pretraining works but ablations show
the defenses are inert (Exp 4), the result is fragile and non-reproducible.

---

## Experiment 1: The Objective Stress Test — Reconstruction vs. Contrastive — PENDING

**W&B group:** `SSL_OBJECTIVE_STRESS_TEST`

**Question:** Does Masked Signal Modeling (MSE reconstruction of CWT tokens)
fail due to the CWT's temporal smearing, and does Continuous Contrastive
(InfoNCE) solve this failure mode?

**Sub-Hypothesis:** Reconstructing masked CWT tokens allows trivial
interpolation from neighboring tokens, yielding excellent pretraining loss
but near-random downstream transfer. The contrastive objective prevents
this by requiring the model to discriminate targets from hard negatives
rather than merely interpolating smooth scalogram surfaces.

**Pretraining Data:** 1 EEG dataset.

**Downstream Task:** EEG Downstream (EEG-DS).

**Tokenizer:** CWT+CNN (with highpass conditioning).

### Test Arms

| Arm | Pretraining Objective | Description |
| --- | --- | --- |
| MAE Baseline | Masked Autoencoder (MSE) | Mask a span of CWT+CNN latent tokens, reconstruct via MSE loss |
| Contrastive | InfoNCE | Mask a span, predict correct latent against negative distractors (with spatial-temporal exclusion zones) |
| Train-from-Scratch | None (supervised only) | No pretraining; supervised learning directly on EEG-DS |

### Success Criteria

- The InfoNCE arm converges to a **higher downstream metric** than the
  train-from-scratch baseline, proving that contrastive pretraining provides
  positive transfer.
- The MAE arm exhibits a **massive disconnect**: low pretraining loss (easy
  interpolation), but fails to beat or only matches the train-from-scratch
  baseline downstream.
- If MAE actually transfers well, the temporal-smearing hypothesis is wrong
  and the contrastive objective's added complexity is unjustified — revisit
  assumptions before proceeding.

### Predicted Outcomes

| Outcome | Interpretation |
| --- | --- |
| InfoNCE >> Scratch > MAE | Temporal smearing confirmed; contrastive objective is essential |
| InfoNCE > Scratch ≈ MAE | Contrastive helps, but MAE is not catastrophically bad — smearing may be partial |
| InfoNCE ≈ MAE >> Scratch | Both objectives transfer — CWT smearing is not a problem; use simpler MAE |
| All arms ≈ Scratch | Pretraining provides no value at this data scale; revisit data or architecture |

---

## Experiment 2: The Architecture Face-Off — CWT+CNN vs. ResampleCNN under InfoNCE — PENDING

**W&B group:** `SSL_ARCHITECTURE_FACEOFF`

**Prerequisite:** Experiment 1 confirms InfoNCE > train-from-scratch.

**Question:** Does the CWT+CNN's structural inductive bias provide a superior
foundation for contrastive self-supervised learning compared to learning
directly from the raw waveform via ResampleCNN?

**Sub-Hypothesis:** The ResampleCNN will struggle to align contrastive
representations across varying sampling rates and noise profiles without
the CWT's built-in sampling-rate invariance and explicit frequency
decomposition. The CWT+CNN's spectral structure gives the contrastive
objective a more stable target space.

**Pretraining Data:** 2 EEG datasets (introducing cross-dataset variance in
sampling rate, referencing, and noise floor, while keeping the modality
constant).

**Downstream Task:** EEG-DS.

**Objective:** InfoNCE (validated in Experiment 1).

### Test Arms

| Arm | Tokenizer | Pretraining | Fine-Tune |
| --- | --- | --- | --- |
| ResampleCNN + InfoNCE | ResampleCNN | InfoNCE on 2 EEG datasets | EEG-DS |
| CWT+CNN + InfoNCE | CWT+CNN (highpass) | InfoNCE on 2 EEG datasets | EEG-DS |

### Success Criteria

- CWT+CNN demonstrates a **statistically significant improvement** over
  ResampleCNN on the downstream EEG-DS task (across multiple folds/seeds),
  validating that the CWT front-end is essential for self-supervised
  representation learning.
- If ResampleCNN matches or beats CWT+CNN, the CWT's inductive bias is not
  necessary for SSL and the simpler architecture should be preferred.

### Predicted Outcomes

| Outcome | Interpretation |
| --- | --- |
| CWT+CNN >> ResampleCNN | CWT's frequency decomposition is a critical inductive bias for SSL across heterogeneous datasets |
| CWT+CNN > ResampleCNN (small gap) | CWT helps but ResampleCNN can partially compensate with enough data — CWT advantage may shrink at scale |
| CWT+CNN ≈ ResampleCNN | CWT's inductive bias is irrelevant for SSL; prefer simpler ResampleCNN |
| ResampleCNN >> CWT+CNN | CWT's temporal smearing or fixed spectral structure actively hurts contrastive learning — reassess tokenizer |

---

## Experiment 3: The Cross-Modality Foundry Test — Mixed ECoG/EEG Pretraining — PENDING

**W&B group:** `SSL_CROSS_MODALITY`

**Prerequisite:** Experiment 2 confirms CWT+CNN > ResampleCNN under InfoNCE.

**Question:** Does mixed ECoG/EEG pretraining result in positive transfer
or catastrophic interference? Can a sufficiently large Perceiver backbone
project both low-amplitude scalp EEG and high-amplitude intracranial ECoG
into a shared temporal latent space?

**Sub-Hypothesis:** The amplitude, bandwidth, and SNR differences between
EEG and ECoG are large enough that naive co-training could collapse the
latent space (ECoG dominates) or fragment it (separate manifolds per
modality). However, the CWT+CNN with highpass conditioning normalizes
the spectral representation sufficiently, and the contrastive objective
with exclusion zones prevents cross-modality shortcut solutions.

**Pretraining Data:** All available datasets (2 EEG + 1 ECoG).

**Downstream Tasks:**
- **EEG-DS:** EEG downstream benchmark.
- **ECoG-DS:** ECoG downstream benchmark (e.g., AJILE12 5-class behavior
  classification).

### Test Arms

| Arm | Pretraining Data | Evaluate On |
| --- | --- | --- |
| EEG-Only Pretrain | 2 EEG datasets | EEG-DS and ECoG-DS |
| Mixed Pretrain (EEG+ECoG) | 2 EEG + 1 ECoG | EEG-DS and ECoG-DS |

### Success Criteria

- The Mixed pretrain arm must perform **at least as well** as EEG-Only on
  the EEG-DS (no catastrophic interference from ECoG data).
- The Mixed pretrain arm must perform **substantially better** than a
  train-from-scratch baseline on ECoG-DS (positive cross-modality transfer).
- If Mixed performs **worse** on EEG-DS than EEG-Only, the ECoG data caused
  catastrophic interference — investigate modality-specific heads, gradient
  balancing, or modality-aware sampling before proceeding.

### Predicted Outcomes

| Outcome | Interpretation |
| --- | --- |
| Mixed ≥ EEG-Only on EEG-DS, Mixed >> Scratch on ECoG-DS | Shared manifold hypothesis confirmed; CWT normalizes cross-modality features |
| Mixed ≈ EEG-Only on EEG-DS, Mixed ≈ Scratch on ECoG-DS | No interference but no positive transfer — ECoG and EEG live on separate manifolds |
| Mixed < EEG-Only on EEG-DS | Catastrophic interference — ECoG data corrupts EEG representations |
| Mixed >> EEG-Only on both tasks | Ideal outcome — cross-modality data acts as a strong regularizer |

---

## Experiment 4: Ablating the Defenses — Exclusion Zones and Augmentations — PENDING

**W&B group:** `SSL_DEFENSE_ABLATION`

**Prerequisite:** Experiment 3 produces a winning mixed-pretrain model.

**Question:** Are the spatial-temporal exclusion zones and domain-specific
augmentations (amplitude scaling, pink noise, line noise injection)
actually what prevent the continuous contrastive objective from collapsing
into noise-memorization? Or does InfoNCE work regardless?

**Sub-Hypothesis:** Without exclusion zones, the model can trivially solve
the InfoNCE task by phase-matching adjacent tokens (temporal shortcut).
Without targeted augmentations, the model memorizes recording-specific
noise signatures rather than neural dynamics. Both failure modes produce
artificially fast pretraining convergence but collapsed downstream
performance.

**Pretraining Data:** 1 ECoG + 1 EEG dataset (subset for speed).

**Downstream Task:** ECoG-DS.

**Reference Model:** The fully-defended model from Experiment 3.

### Test Arms

| Arm | Exclusion Zones | Augmentations | Description |
| --- | --- | --- | --- |
| Full Model (Exp 3) | ✓ | Full (amplitude scaling, pink noise, line noise) | Reference from Experiment 3 |
| No Exclusion Zones | ✗ | Full | Negatives sampled randomly, allowing trivial phase-matching from adjacent tokens |
| Naive Augmentation | ✓ | Gaussian noise only | Domain-specific augmentations removed; only standard Gaussian noise applied |

### Success Criteria

- Both ablation arms should show artificially **"good" contrastive loss
  curves** during pretraining (faster convergence, lower InfoNCE loss)
  but **fail spectacularly** on the downstream ECoG-DS compared to the
  fully-defended Experiment 3 model.
- If removing exclusion zones does not hurt: adjacent tokens are not
  trivially similar in the CWT+CNN space, and exclusion zones are
  unnecessary complexity — simplify.
- If removing targeted augmentations does not hurt: domain-specific noise
  profiles are not a shortcut risk, and Gaussian noise is sufficient —
  simplify.

### Predicted Outcomes

| Outcome | Interpretation |
| --- | --- |
| Both ablations: fast pretrain convergence, poor downstream | Defenses are essential — exclusion zones prevent temporal shortcuts, augmentations prevent noise memorization |
| No-exclusion-zones ablation hurts, naive-augmentation does not | Temporal shortcuts are the primary risk; augmentation choice is secondary |
| Naive-augmentation ablation hurts, no-exclusion-zones does not | Noise memorization is the primary risk; exclusion zones are over-engineered |
| Neither ablation hurts | Defenses are inert — InfoNCE + CWT is inherently robust; simplify the pipeline |

---

## Summary

| # | Experiment | Status | W&B Group | Isolates |
| --- | --- | --- | --- | --- |
| 1 | Objective Stress Test (Reconstruction vs. Contrastive) | Pending | `SSL_OBJECTIVE_STRESS_TEST` | Pretraining objective |
| 2 | Architecture Face-Off (CWT+CNN vs. ResampleCNN) | Pending | `SSL_ARCHITECTURE_FACEOFF` | Tokenizer architecture |
| 3 | Cross-Modality Foundry Test (Mixed ECoG/EEG) | Pending | `SSL_CROSS_MODALITY` | Data scaling / modality mixing |
| 4 | Ablating the Defenses (Exclusion Zones + Augmentations) | Pending | `SSL_DEFENSE_ABLATION` | Regularizers / anti-shortcut mechanisms |

**Dependency chain:** 1 → 2 → 3 → 4 (strictly sequential — each experiment
gates the next). If Experiment 1 fails to show Contrastive > Reconstruction,
stop and revisit the objective design before burning GPU hours on later
experiments.

---

## Follow-up (conditional on Experiment 4 results)

- **Experiment 5 — Learning rate schedule for fine-tuning:** Systematic
  sweep of fine-tuning LR, warmup, and layer-wise LR decay to ensure
  pretrained representations are not destroyed during downstream adaptation.
  This was deliberately deferred to avoid optimizing a hyperparameter
  before the foundational objective and architecture are validated.
- **Experiment 6 — Scaling laws:** Vary pretraining data volume (25%, 50%,
  100% of available data) to characterize the data-efficiency curve and
  determine whether more pretraining data yields diminishing returns.
- **Experiment 7 — Additional downstream tasks:** Evaluate the best
  pretrained model on held-out downstream tasks not used during any
  experiment design decisions, to test true generalization.
