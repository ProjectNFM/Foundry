# Foundry Task Refactor — Design Document

> **Status**: Draft v3 — decomposed architecture, Lightning-native
> **Replaces**: `torch_brain.registry`, `torch_brain.nn.loss`, `torch_brain.nn.MultitaskReadout`,
> `torch_brain.nn.multitask_readout.prepare_for_multitask_readout`

---

## 1. Problem statement

Foundry currently depends on four constructs that torchbrain removed in
`[2426bbb](https://github.com/neuro-galaxy/torch_brain/commit/2426bbb)`:


| Removed construct                                          | Where Foundry uses it                                                   |
| ---------------------------------------------------------- | ----------------------------------------------------------------------- |
| `register_modality` / `MODALITY_REGISTRY` / `ModalitySpec` | `modalities.py`, `mixins.py`, `utils.py`, every model, every datamodule |
| `Loss` / `CrossEntropyLoss` / `MSELoss`                    | `modalities.py`, loss computation in `task_modules.py`                  |
| `MultitaskReadout`                                         | Every model's readout head                                              |
| `prepare_for_multitask_readout`                            | Every model's `tokenize()` method                                       |


These constructs blended concerns that belong to different layers: data schema,
training loss, output projection, and global identity were all packed into a
single `ModalitySpec` managed by a mutable global registry with
auto-incrementing integer IDs. Label remapping was encoded as a special loss
subclass and later inspected via `isinstance` in the Lightning module.

The refactor must also accommodate **non-standard objectives** (MAE
self-supervised learning, contrastive losses, distillation) that don't fit the
"linear readout → target → standard loss" pattern.

### 1.1 Design principles

1. **Separation of concerns** — data transforms, model forward pass, loss
  computation, and training strategy are distinct layers with clean interfaces.
2. **Lightning-native** — use PyTorch Lightning idioms (callbacks, composable
  modules, `configure_optimizers`) rather than reinventing them.
3. **Hydra-composable** — every knob that an experimenter might tweak is
  overridable from YAML without touching Python.
4. **Single-task fast path** — most runs are single-objective; the common case
  should pay zero routing/indexing overhead.
5. **Models are pure `nn.Module`** — no loss, no metrics, no data transforms on
  the model. This keeps models exportable (ONNX, `torch.compile`, TorchScript).
6. **SSL is a training strategy** — modifying how the backbone executes is a
  training concern, not an output-head concern.

---

## 2. Vocabulary


| Term                  | Definition                                                                                                                      |
| --------------------- | ------------------------------------------------------------------------------------------------------------------------------- |
| **Task**              | Any training objective — supervised (classification, regression) or unsupervised (MAE reconstruction, contrastive).             |
| **Readout head**      | A small `nn.Module` that projects backbone embeddings to task-specific predictions. Pure forward pass — no loss, no data logic. |
| **Target extractor**  | A callable (not `nn.Module`) that pulls targets from a `temporaldata.Data` object during tokenization. A data concern.          |
| **Loss function**     | A composable callable that scores predictions against targets. Pluggable, overridable from config.                              |
| **Training strategy** | A Lightning Callback (or model wrapper) that modifies model execution for SSL objectives.                                       |
| **Task config**       | A Hydra-instantiable structured config that wires together a readout head, target extractor, loss, and metrics for one task.    |


Things we stop saying: "modality" (when meaning task), "decoder index"
(when meaning task index), "spec" as a god-object.

---

## 3. Architecture overview

The previous draft (v2) consolidated five responsibilities into a single
`ReadoutModule` class. This revision **decomposes** them into focused, single-
responsibility components that align with Lightning's layering:

```
┌────────────────────────────────────────────────────────────────────────┐
│  Data layer (CPU, per-sample)                                          │
│                                                                        │
│  TargetExtractor: Data → {timestamps, values, weights}                 │
│  (plain callable — no nn.Module, no embed_dim dependency)              │
└────────────────────────────────┬───────────────────────────────────────┘
                                 │ collated batch
                                 ▼
┌────────────────────────────────────────────────────────────────────────┐
│  Model layer (GPU, pure nn.Module)                                     │
│                                                                        │
│  ReadoutHead: embeddings → predictions (just a linear/MLP)             │
│  ReadoutRouter: dispatches embeddings to heads by task index            │
│  Model.forward: inputs → backbone → router → {task: predictions}       │
└────────────────────────────────┬───────────────────────────────────────┘
                                 │ outputs dict
                                 ▼
┌────────────────────────────────────────────────────────────────────────┐
│  Training layer (LightningModule)                                      │
│                                                                        │
│  TaskLoss: composable loss per task (CE, MSE, Focal, weighted, etc.)   │
│  Metrics: MetricCollection per task (from torchmetrics)                 │
│  FoundryModule._shared_step: unpack → forward → loss → metrics         │
│  SSLStrategy callback: modifies model execution for SSL objectives     │
└────────────────────────────────────────────────────────────────────────┘
```

### 3.1 Why decompose rather than consolidate


| Concern               | v2 (ReadoutModule)          | v3 (decomposed)                     | Benefit                                                   |
| --------------------- | --------------------------- | ----------------------------------- | --------------------------------------------------------- |
| Output projection     | `readout.forward()`         | `ReadoutHead.forward()`             | Model stays pure `nn.Module` — exportable, compilable     |
| Target extraction     | `readout.prepare_targets()` | `TargetExtractor(data)`             | CPU-only, testable without `embed_dim`, no gradient graph |
| Loss function         | `readout.compute_loss()`    | `TaskLoss(preds, targets, weights)` | Swappable from YAML (CE → Focal → label-smoothed)         |
| Metrics               | `readout.build_metrics()`   | Config-driven `MetricCollection`    | Overridable from experiment config                        |
| Backbone modification | `readout.wrap_backbone()`   | `SSLStrategy` callback              | Doesn't invert control hierarchy; composable              |


---

## 4. Component details

### 4.1 Target extraction (data layer)

```python
# foundry/tasks/targets.py
from dataclasses import dataclass
import numpy as np
from temporaldata import Data


@dataclass(frozen=True)
class TargetExtractor:
    """Extracts targets from a Data sample during tokenization.

    A pure data transform — no nn.Module, no embed_dim, no GPU tensors.
    Testable in isolation with just a Data object.
    """
    timestamp_key: str
    value_key: str
    label_map: dict[int, int] | None = None

    def __call__(self, data: Data) -> dict:
        timestamps = _get_nested(data, self.timestamp_key)
        values = _get_nested(data, self.value_key)

        if self.label_map is not None:
            mapped = np.empty_like(values)
            for src, dst in self.label_map.items():
                mapped[values == src] = dst
            values = mapped

        if values.dtype == np.float64:
            values = values.astype(np.float32)

        return {"timestamps": timestamps, "values": values}


def _get_nested(data: Data, key: str):
    """Navigate a dot-separated key path into a Data object."""
    obj = data
    for part in key.split("."):
        obj = getattr(obj, part)
    return obj
```

**Key properties:**

- Frozen dataclass — serializable, hashable, loggable.
- No `nn.Module` inheritance — doesn't pollute `model.parameters()`.
- Testable with just `Data()` — no `embed_dim`, no GPU needed.
- Label mapping happens here, once, as a data transform. No downstream code
ever sees unmapped labels.

### 4.2 Readout heads (model layer)

```python
# foundry/tasks/heads.py
import torch
import torch.nn as nn


class ReadoutHead(nn.Module):
    """Pure projection from backbone embeddings to task predictions.

    No loss, no metrics, no data logic. Just a forward pass.
    """
    def __init__(self, embed_dim: int, output_dim: int):
        super().__init__()
        self.output_dim = output_dim
        self.projection = nn.Linear(embed_dim, output_dim)

    def forward(self, embeddings: torch.Tensor) -> torch.Tensor:
        return self.projection(embeddings)


class MLPReadoutHead(nn.Module):
    """Multi-layer projection head (for SSL projection, deeper readouts)."""

    def __init__(
        self,
        embed_dim: int,
        output_dim: int,
        hidden_dim: int | None = None,
        num_layers: int = 2,
        activation: str = "gelu",
    ):
        super().__init__()
        hidden_dim = hidden_dim or embed_dim
        act_fn = {"gelu": nn.GELU, "relu": nn.ReLU}[activation]

        layers = []
        in_dim = embed_dim
        for _ in range(num_layers - 1):
            layers.extend([nn.Linear(in_dim, hidden_dim), act_fn()])
            in_dim = hidden_dim
        layers.append(nn.Linear(in_dim, output_dim))
        self.net = nn.Sequential(*layers)
        self.output_dim = output_dim

    def forward(self, embeddings: torch.Tensor) -> torch.Tensor:
        return self.net(embeddings)
```

**Key properties:**

- Pure `nn.Module` — `forward(embeddings) → predictions`. Nothing else.
- No `name`, `kind`, or metadata — those belong to the task config.
- `torch.compile`-friendly (static shapes if no routing needed).

### 4.3 Task loss functions (training layer)

```python
# foundry/tasks/losses.py
import torch
import torch.nn as nn
import torch.nn.functional as F


class CrossEntropyTaskLoss(nn.Module):
    """Cross-entropy loss with optional class weights and label smoothing."""

    def __init__(
        self,
        label_smoothing: float = 0.0,
        class_weights: list[float] | None = None,
    ):
        super().__init__()
        self.label_smoothing = label_smoothing
        if class_weights is not None:
            self.register_buffer(
                "class_weights", torch.tensor(class_weights, dtype=torch.float32)
            )
        else:
            self.class_weights = None

    def forward(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor,
        sample_weights: torch.Tensor | float = 1.0,
    ) -> torch.Tensor:
        loss = F.cross_entropy(
            predictions,
            targets.long(),
            weight=self.class_weights,
            label_smoothing=self.label_smoothing,
            reduction="none",
        )
        if isinstance(sample_weights, torch.Tensor):
            loss = loss * sample_weights
        return loss.mean()


class MSETaskLoss(nn.Module):
    """MSE loss with optional per-sample weighting."""

    def forward(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor,
        sample_weights: torch.Tensor | float = 1.0,
    ) -> torch.Tensor:
        loss = F.mse_loss(predictions, targets, reduction="none")
        if isinstance(sample_weights, torch.Tensor):
            loss = loss * sample_weights.unsqueeze(-1)
        return loss.mean()


class FocalTaskLoss(nn.Module):
    """Focal loss for class-imbalanced classification."""

    def __init__(self, gamma: float = 2.0, alpha: float | None = None):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha

    def forward(self, predictions, targets, sample_weights=1.0):
        ce = F.cross_entropy(predictions, targets.long(), reduction="none")
        pt = torch.exp(-ce)
        focal = ((1 - pt) ** self.gamma) * ce
        if self.alpha is not None:
            focal = self.alpha * focal
        if isinstance(sample_weights, torch.Tensor):
            focal = focal * sample_weights
        return focal.mean()
```

**Key properties:**

- Each loss is an `nn.Module` — can register buffers (class weights), be
instantiated by Hydra, and participate in `state_dict`.
- Composable from YAML: swap `CrossEntropyTaskLoss` → `FocalTaskLoss` without
changing any Python.
- Uniform signature: `(predictions, targets, sample_weights) → scalar`.
- No label remapping inside the loss — that happened in `TargetExtractor`.

### 4.4 Readout router (model layer)

```python
# foundry/models/readout.py
import torch
import torch.nn as nn


class ReadoutRouter(nn.Module):
    """Routes output embeddings to task-specific ReadoutHeads.

    Replaces MultitaskReadout. For single-task runs (the common case),
    use the fast path that skips index masking entirely.
    """

    def __init__(self, heads: dict[str, nn.Module]):
        super().__init__()
        self.heads = nn.ModuleDict(heads)
        self._task_names = sorted(heads.keys())
        self._name_to_idx = {n: i for i, n in enumerate(self._task_names)}
        self._single_task = len(heads) == 1

    def forward(
        self,
        output_embs: torch.Tensor,
        task_index: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        if self._single_task:
            name = self._task_names[0]
            return {name: self.heads[name](output_embs)}

        outputs = {}
        for idx, name in enumerate(self._task_names):
            mask = task_index == idx
            if not mask.any():
                continue
            outputs[name] = self.heads[name](output_embs[mask])
        return outputs

    def get_task_index_by_name(self, name: str) -> int:
        return self._name_to_idx[name]

    @property
    def num_tasks(self) -> int:
        return len(self._task_names)
```

**Key improvement over v2:** explicit single-task fast path. When
`len(heads) == 1` (the common case per the user's statement), no masking, no
indexing, no iteration — just call the one head directly.

### 4.5 Task config (Hydra-composable)

```python
# foundry/tasks/config.py
from dataclasses import dataclass, field
from typing import Any


@dataclass
class TaskConfig:
    """Hydra-instantiable task configuration.

    Contains everything needed to construct the task's components.
    Each field maps to a Hydra _target_ or literal value, making the
    full task overridable from the command line.
    """
    name: str
    head: dict[str, Any]           # Hydra config for ReadoutHead
    target_extractor: dict[str, Any]  # Hydra config for TargetExtractor
    loss: dict[str, Any]           # Hydra config for loss function
    metrics: dict[str, Any] | None = None  # Hydra config for MetricCollection
    class_names: list[str] | None = None
    metric_summary_modes: dict[str, str] = field(default_factory=dict)

    @property
    def output_dim(self) -> int:
        return self.head.get("output_dim", self.head.get("num_classes", 1))

    @property
    def kind(self) -> str:
        """Inferred from output_dim and loss type for callbacks."""
        if "CrossEntropy" in self.loss.get("_target_", ""):
            return "binary" if self.output_dim == 2 else "multiclass"
        return "continuous"
```

This is a plain dataclass that Hydra can compose and override. Example YAML:

```yaml
# configs/tasks/motor_imagery_5class.yaml
name: motor_imagery_5class
head:
  _target_: foundry.tasks.heads.ReadoutHead
  output_dim: 5
target_extractor:
  _target_: foundry.tasks.targets.TargetExtractor
  timestamp_key: motor_imagery_trials.timestamps
  value_key: motor_imagery_trials.movement_ids
loss:
  _target_: foundry.tasks.losses.CrossEntropyTaskLoss
  label_smoothing: 0.0
metrics:
  _target_: foundry.tasks.metrics.classification_metrics
  num_classes: 5
class_names: [Rest, Left hand, Right hand, Feet, Tongue]
metric_summary_modes:
  acc: max
  f1: max
  auroc: max
  loss: min
```

**Key advantage:** experiment overrides work naturally:

```bash
# Try focal loss for this run
python main.py experiment=physionet_mi \
    'tasks.motor_imagery_5class.loss._target_=foundry.tasks.losses.FocalTaskLoss' \
    'tasks.motor_imagery_5class.loss.gamma=3.0'

# Add label smoothing
python main.py experiment=physionet_mi \
    'tasks.motor_imagery_5class.loss.label_smoothing=0.1'
```

No new Python constants needed per experiment variant.

### 4.6 Metric factories

```python
# foundry/tasks/metrics.py
from torchmetrics import MetricCollection
from torchmetrics.classification import (
    AUROC, Accuracy, CohenKappa, F1Score, Precision, Recall,
)
from torchmetrics.regression import MeanAbsoluteError, MeanSquaredError, R2Score


def classification_metrics(num_classes: int) -> MetricCollection:
    task_type = "binary" if num_classes == 2 else "multiclass"
    return MetricCollection({
        "acc": Accuracy(task=task_type, num_classes=num_classes),
        "f1": F1Score(task=task_type, num_classes=num_classes, average="macro"),
        "auroc": AUROC(task=task_type, num_classes=num_classes),
        "precision": Precision(task=task_type, num_classes=num_classes, average="macro"),
        "recall": Recall(task=task_type, num_classes=num_classes, average="macro"),
        "balanced_acc": Accuracy(task=task_type, num_classes=num_classes, average="macro"),
        "cohen_kappa": CohenKappa(task=task_type, num_classes=num_classes),
    })


def regression_metrics() -> MetricCollection:
    return MetricCollection({
        "mse": MeanSquaredError(),
        "mae": MeanAbsoluteError(),
        "r2": R2Score(multioutput="uniform_average"),
    })


def ssl_metrics() -> MetricCollection:
    return MetricCollection({"recon_mse": MeanSquaredError()})
```

These are plain factory functions that Hydra can call via `_target_`. The
Lightning module calls them once per task at init to build train/val metric
collections.

---

## 5. SSL as a training strategy

### 5.1 Why SSL doesn't belong on the readout head

The v2 design placed `wrap_backbone()` on `ReadoutModule`, creating a
control inversion where a downstream component (output projection) drove
an upstream component (the backbone). Problems:

1. **Control inversion** — a readout is conceptually *downstream* of the
  backbone; having it *control* backbone execution violates the data flow
   direction.
2. **Fragile detection** — `type(readout).wrap_backbone is not
  ReadoutModule.wrap_backbone` breaks with decorators, multiple inheritance,
   mocking, or metaclasses.
3. **Single-SSL constraint** — silently picks first SSL readout by iteration
  order if multiple exist.
4. **Tight architecture coupling** — the `IntermediateMAEReadout` directly
  iterated `processor.layers`, coupling a "readout" to the Perceiver's
   internal structure.
5. **Breaks model exportability** — a model whose forward pass is conditionally
  delegated to an output head can't be cleanly compiled or exported.

### 5.2 The alternative: `SSLStrategy` as a Lightning Callback

SSL methods (MAE, contrastive, distillation) modify *how the model trains*,
not *how outputs are projected*. Lightning provides the infrastructure for
exactly this — callbacks that hook into the training loop at well-defined points.

```python
# foundry/training/strategies.py
import lightning as L
import torch
import torch.nn as nn


class SSLStrategy(L.Callback):
    """Base class for self-supervised training strategies.

    Subclass to implement MAE masking, contrastive dual-view,
    knowledge distillation, etc. The model stays a pure nn.Module
    with a standard forward pass.
    """

    def modify_forward_inputs(
        self,
        model: nn.Module,
        batch: dict,
    ) -> dict:
        """Transform batch/model inputs before the forward pass.

        Override to inject masking, create augmented views, or modify
        the batch structure. Called by FoundryModule before model.forward().

        Returns:
            Modified batch dict.
        """
        return batch

    def compute_ssl_loss(
        self,
        model: nn.Module,
        batch: dict,
        model_outputs: dict,
        backbone_intermediates: dict,
    ) -> torch.Tensor | None:
        """Compute the SSL loss from model outputs and/or intermediates.

        Called by FoundryModule after the forward pass. Returns a loss
        tensor to add to the total, or None if this strategy only modifies
        inputs (masking strategies handled purely via modify_forward_inputs).

        Returns:
            SSL loss tensor, or None.
        """
        return None

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        """Post-step hooks: EMA updates, momentum scheduling, etc."""
        pass
```

### 5.3 Intermediate-layer MAE as a strategy

```python
class IntermediateMAEStrategy(SSLStrategy):
    """Mask latents after layer K, reconstruct pre-mask embeddings.

    The model exposes intermediate states via a forward hook rather
    than delegating its execution to an external component.
    """

    def __init__(
        self,
        mask_after_layer: int = 1,
        mask_ratio: float = 0.5,
        decoder_dim: int = 256,
    ):
        super().__init__()
        self.mask_after_layer = mask_after_layer
        self.mask_ratio = mask_ratio
        self._decoder = None
        self._decoder_dim = decoder_dim
        self._intermediates = {}
        self._hook_handle = None

    def on_fit_start(self, trainer, pl_module):
        model = pl_module.model
        embed_dim = model.embed_dim

        self._decoder = nn.Sequential(
            nn.Linear(embed_dim, self._decoder_dim),
            nn.GELU(),
            nn.Linear(self._decoder_dim, embed_dim),
        ).to(pl_module.device)

        # Register a forward hook on the processor to capture intermediates
        processor = model.backbone.processor
        self._hook_handle = processor.register_forward_hook(
            self._capture_intermediates
        )

    def _capture_intermediates(self, module, input, output):
        """Forward hook: captures processor output for reconstruction target."""
        self._intermediates["pre_mask"] = output.detach().clone()

    def modify_forward_inputs(self, model, batch):
        """Inject a masking transform on the latents after layer K.

        Instead of reaching into the backbone, we register a hook that
        applies masking between processor iterations. The model's own
        forward pass handles the rest.
        """
        # Store mask info for loss computation
        batch["_ssl_mask_ratio"] = self.mask_ratio
        batch["_ssl_mask_layer"] = self.mask_after_layer
        return batch

    def compute_ssl_loss(self, model, batch, model_outputs, backbone_intermediates):
        pre_mask = backbone_intermediates.get("pre_mask_embs")
        post_mask_output = backbone_intermediates.get("post_mask_embs")

        if pre_mask is None or post_mask_output is None:
            return None

        reconstructed = self._decoder(post_mask_output)
        return F.mse_loss(reconstructed, pre_mask)

    def on_fit_end(self, trainer, pl_module):
        if self._hook_handle:
            self._hook_handle.remove()
```

### 5.4 Model support for SSL intermediates

Rather than delegating control to a readout, the model exposes intermediates
through a **protocol** that SSL strategies can consume:

```python
class POYOEEGModel(nn.Module):
    def forward(
        self,
        *,
        input_values,
        input_timestamps,
        # ... other inputs ...
        task_index=None,
        return_intermediates: bool = False,
    ):
        # Standard forward: encoder → processor → decoder → router
        inputs = self.tokenizer(...)
        latents = self.backbone.encoder(inputs, ...)
        processed = self.backbone.processor(latents, latent_timestamp_emb)
        output_latents = self.backbone.decoder(processed, ...)
        predictions = self.router(output_latents, task_index)

        if return_intermediates:
            return predictions, {
                "encoder_output": latents,
                "processor_output": processed,
                "decoder_output": output_latents,
            }
        return predictions
```

The model decides what to expose. SSL strategies consume the intermediates
without driving the model's execution. The model's forward pass stays
sequential, deterministic, and compilable.

### 5.5 Contrastive learning as a strategy

```python
class ContrastiveStrategy(SSLStrategy):
    """SimCLR/VICReg-style contrastive training.

    Runs two augmented views through the backbone and a shared projection
    head. Computes InfoNCE loss on the projected embeddings.
    """

    def __init__(
        self,
        proj_dim: int = 128,
        temperature: float = 0.1,
    ):
        super().__init__()
        self.proj_dim = proj_dim
        self.temperature = temperature
        self._projector = None

    def on_fit_start(self, trainer, pl_module):
        embed_dim = pl_module.model.embed_dim
        self._projector = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.BatchNorm1d(embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, self.proj_dim),
        ).to(pl_module.device)

    def modify_forward_inputs(self, model, batch):
        """The dataloader provides two views; we run the model twice."""
        # Augmentation is a data transform — the batch already contains
        # 'input_values_view1' and 'input_values_view2'
        return batch

    def compute_ssl_loss(self, model, batch, model_outputs, backbone_intermediates):
        # model was called twice (handled by FoundryModule for contrastive)
        z1 = self._projector(backbone_intermediates["view1_embeddings"])
        z2 = self._projector(backbone_intermediates["view2_embeddings"])

        z1 = F.normalize(z1, dim=-1)
        z2 = F.normalize(z2, dim=-1)

        similarity = torch.mm(z1, z2.t()) / self.temperature
        labels = torch.arange(z1.shape[0], device=z1.device)
        loss = (F.cross_entropy(similarity, labels)
                + F.cross_entropy(similarity.t(), labels)) / 2
        return loss
```

**Why this works better than v2:**

- The model runs its standard forward pass — no conditional `if ssl_readout`.
- The strategy owns its own parameters (`_projector`, `_decoder`) separately
from model parameters — enabling separate LR, no-grad teacher branches, etc.
- Multiple strategies can compose (intermediate MAE + contrastive) since they're
just callbacks that hook into different points.
- EMA-based methods (BYOL, data2vec) naturally fit via `on_train_batch_end`.
- Lightning's `manual_optimization` can be used when strategies need separate
optimizer steps.

### 5.6 How FoundryModule integrates strategies

```python
class FoundryModule(L.LightningModule):
    def _shared_step(self, stage, batch):
        # Let strategies modify inputs
        for cb in self._ssl_strategies:
            batch = cb.modify_forward_inputs(self.model, batch)

        # Standard forward
        model_inputs, targets, weights, task_index = self._unpack_batch(batch)
        outputs, intermediates = self.model(
            **model_inputs, return_intermediates=True
        )

        # Supervised losses
        total_loss = self._compute_task_losses(stage, outputs, targets, weights)

        # SSL losses (from strategies)
        for cb in self._ssl_strategies:
            ssl_loss = cb.compute_ssl_loss(
                self.model, batch, outputs, intermediates
            )
            if ssl_loss is not None:
                total_loss = total_loss + ssl_loss
                self.log(f"{stage}/{cb.__class__.__name__}_loss", ssl_loss)

        self.log(f"{stage}/loss", total_loss, prog_bar=True)
        return total_loss
```

---

## 6. Task catalog (Hydra YAML)

### 6.1 Task YAML configs

Each task is a composable Hydra config file:

```yaml
# configs/tasks/motor_imagery_5class.yaml
name: motor_imagery_5class

head:
  _target_: foundry.tasks.heads.ReadoutHead
  output_dim: 5

target_extractor:
  _target_: foundry.tasks.targets.TargetExtractor
  timestamp_key: motor_imagery_trials.timestamps
  value_key: motor_imagery_trials.movement_ids

loss:
  _target_: foundry.tasks.losses.CrossEntropyTaskLoss
  label_smoothing: 0.0

metrics:
  _target_: foundry.tasks.metrics.classification_metrics
  num_classes: 5

class_names: [Rest, Left hand, Right hand, Feet, Tongue]
metric_summary_modes:
  acc: max
  f1: max
  auroc: max
  loss: min
```

```yaml
# configs/tasks/motor_imagery_left_right.yaml
name: motor_imagery_left_right

head:
  _target_: foundry.tasks.heads.ReadoutHead
  output_dim: 2

target_extractor:
  _target_: foundry.tasks.targets.TargetExtractor
  timestamp_key: motor_imagery_trials.timestamps
  value_key: motor_imagery_trials.movement_ids
  label_map:
    1: 0
    2: 1

loss:
  _target_: foundry.tasks.losses.CrossEntropyTaskLoss

metrics:
  _target_: foundry.tasks.metrics.classification_metrics
  num_classes: 2

class_names: [Left hand, Right hand]
metric_summary_modes:
  acc: max
  auroc: max
  loss: min
```

```yaml
# configs/tasks/ajile_pose_estimation.yaml
name: ajile_pose_estimation

head:
  _target_: foundry.tasks.heads.ReadoutHead
  output_dim: 18

target_extractor:
  _target_: foundry.tasks.targets.TargetExtractor
  timestamp_key: pose_trajectories.timestamps
  value_key: pose_trajectories.values

loss:
  _target_: foundry.tasks.losses.MSETaskLoss

metrics:
  _target_: foundry.tasks.metrics.regression_metrics

metric_summary_modes:
  r2: max
  mse: min
  mae: min
  loss: min
```

### 6.2 Python catalog (for programmatic access)

For code that needs task configs without Hydra (tests, notebooks, tools),
a Python catalog provides the same information:

```python
# foundry/tasks/catalog.py
from foundry.tasks.config import TaskConfig
from foundry.tasks.heads import ReadoutHead
from foundry.tasks.targets import TargetExtractor
from foundry.tasks.losses import CrossEntropyTaskLoss, MSETaskLoss
from foundry.tasks.metrics import classification_metrics, regression_metrics


MOTOR_IMAGERY_5CLASS = TaskConfig(
    name="motor_imagery_5class",
    head={"_target_": "foundry.tasks.heads.ReadoutHead", "output_dim": 5},
    target_extractor={
        "_target_": "foundry.tasks.targets.TargetExtractor",
        "timestamp_key": "motor_imagery_trials.timestamps",
        "value_key": "motor_imagery_trials.movement_ids",
    },
    loss={"_target_": "foundry.tasks.losses.CrossEntropyTaskLoss"},
    metrics={"_target_": "foundry.tasks.metrics.classification_metrics", "num_classes": 5},
    class_names=["Rest", "Left hand", "Right hand", "Feet", "Tongue"],
    metric_summary_modes={"acc": "max", "f1": "max", "auroc": "max", "loss": "min"},
)

MOTOR_IMAGERY_LEFT_RIGHT = TaskConfig(
    name="motor_imagery_left_right",
    head={"_target_": "foundry.tasks.heads.ReadoutHead", "output_dim": 2},
    target_extractor={
        "_target_": "foundry.tasks.targets.TargetExtractor",
        "timestamp_key": "motor_imagery_trials.timestamps",
        "value_key": "motor_imagery_trials.movement_ids",
        "label_map": {1: 0, 2: 1},
    },
    loss={"_target_": "foundry.tasks.losses.CrossEntropyTaskLoss"},
    metrics={"_target_": "foundry.tasks.metrics.classification_metrics", "num_classes": 2},
    class_names=["Left hand", "Right hand"],
    metric_summary_modes={"acc": "max", "auroc": "max", "loss": "min"},
)

AJILE_POSE_ESTIMATION = TaskConfig(
    name="ajile_pose_estimation",
    head={"_target_": "foundry.tasks.heads.ReadoutHead", "output_dim": 18},
    target_extractor={
        "_target_": "foundry.tasks.targets.TargetExtractor",
        "timestamp_key": "pose_trajectories.timestamps",
        "value_key": "pose_trajectories.values",
    },
    loss={"_target_": "foundry.tasks.losses.MSETaskLoss"},
    metrics={"_target_": "foundry.tasks.metrics.regression_metrics"},
    metric_summary_modes={"r2": "max", "mse": "min", "mae": "min", "loss": "min"},
)
```

The Python catalog and YAML configs are equivalent representations. The YAML
is canonical for Hydra experiments; the Python catalog is for tests and scripts.

---

## 7. How models use task configs

### 7.1 Model constructor

```python
class POYOEEGModel(nn.Module):
    def __init__(
        self,
        tokenizer: EEGTokenizer,
        task_configs: dict[str, TaskConfig],
        embed_dim: int,
        # ... other args ...
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self._task_configs = task_configs

        # Build readout heads from task configs + embed_dim
        heads = {}
        for name, cfg in task_configs.items():
            head_kwargs = dict(cfg.head)
            head_kwargs.pop("_target_", None)
            heads[name] = ReadoutHead(embed_dim=embed_dim, **head_kwargs)

        self.router = ReadoutRouter(heads)

        # Task embedding
        self.task_emb = Embedding(
            self.router.num_tasks, embed_dim, init_scale=emb_init_scale
        )
        # ... rest of init ...

    @property
    def task_configs(self) -> dict[str, TaskConfig]:
        return self._task_configs

    def forward(
        self,
        *,
        input_values,
        input_timestamps,
        # ... other inputs ...
        task_index=None,
        return_intermediates: bool = False,
    ):
        inputs = self.tokenizer(...)
        latents = self.backbone.encoder(inputs, ...)
        processed = self.backbone.processor(latents, latent_timestamp_emb)
        output_latents = self.backbone.decoder(processed, ...)
        predictions = self.router(output_latents, task_index)

        if return_intermediates:
            return predictions, {
                "encoder_output": latents,
                "processor_output": processed,
                "decoder_output": output_latents,
            }
        return predictions
```

The model is a pure `nn.Module`. Its forward pass is always the same:
`encode → process → decode → route`. No conditional SSL branching, no
`_get_ssl_readout()` checks.

### 7.2 Tokenization with target extractors

Target extraction is a data concern handled by `TargetExtractor` callables.
The model's `tokenize()` method uses them:

```python
def tokenize(self, data: Data) -> dict:
    # ... input tokenization (signal, channels, etc.) ...

    # Target extraction — delegate to TargetExtractors from task configs
    all_timestamps = []
    task_indices = []
    target_values = {}
    target_weights = {}

    for name in self.router._task_names:
        cfg = self._task_configs[name]
        extractor_kwargs = dict(cfg.target_extractor)
        extractor_kwargs.pop("_target_", None)
        extractor = TargetExtractor(**extractor_kwargs)

        targets = extractor(data)
        if targets["timestamps"] is None:
            continue

        idx = self.router.get_task_index_by_name(name)
        all_timestamps.append(targets["timestamps"])
        task_indices.append(idx)
        target_values[name] = targets["values"]
        target_weights[name] = targets.get(
            "weights", np.ones_like(targets["timestamps"], dtype=np.float32)
        )

    timestamps, batch_idx = _chain_timestamps(all_timestamps)
    task_index = torch.tensor([task_indices[i] for i in batch_idx])

    return {
        # ... input fields ...
        "task_index": pad8(task_index),
        "target_values": chain(target_values, allow_missing_keys=True),
        "target_weights": chain(target_weights, allow_missing_keys=True),
    }
```

Note: `TargetExtractor` instances are lightweight frozen dataclasses, so
constructing them per-sample is negligible. Alternatively they can be cached
at model init — both approaches work.

### 7.3 Baselines

```python
class Linear(BaselineEEGModel):
    def __init__(self, task_configs, num_channels=64, num_samples=128):
        super().__init__(num_channels=num_channels, num_samples=num_samples)
        embed_dim = num_channels * num_samples

        heads = {}
        for name, cfg in task_configs.items():
            head_kwargs = dict(cfg.head)
            head_kwargs.pop("_target_", None)
            heads[name] = ReadoutHead(embed_dim=embed_dim, **head_kwargs)

        self.router = ReadoutRouter(heads)

    def forward(self, *, input_values, task_index=None, **kwargs):
        x = self._normalize_input_shape(input_values)
        x = x.reshape(x.size(0), -1)
        if not self.router._single_task:
            batch_size = x.shape[0]
            n_out = task_index.shape[1]
            x = x.unsqueeze(1).expand(batch_size, n_out, -1)
        return self.router(x, task_index)
```

---

## 8. Training module (Lightning-native)

```python
# foundry/training/module.py
import lightning as L
import torch
import torch.nn as nn
from hydra.utils import instantiate
from torchmetrics import MetricCollection


class FoundryModule(L.LightningModule):
    """Single training module for all task types.

    Delegates loss computation to per-task loss functions and metric
    management to per-task MetricCollections. No isinstance checks,
    no classification/regression split.
    """

    def __init__(
        self,
        model: nn.Module,
        learning_rate: float = 1e-4,
        weight_decay: float = 0.01,
        cwt_lr_multiplier: float = 1.0,
    ):
        super().__init__()
        self.model = model
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.cwt_lr_multiplier = cwt_lr_multiplier
        self.save_hyperparameters(ignore=["model"])

        # Build per-task loss functions and metrics from task configs
        self._task_losses = nn.ModuleDict()
        self.train_metrics = nn.ModuleDict()
        self.val_metrics = nn.ModuleDict()

        for name, cfg in model.task_configs.items():
            self._task_losses[name] = instantiate(cfg.loss)

            metrics_cfg = cfg.metrics
            if metrics_cfg:
                metrics_kwargs = dict(metrics_cfg)
                target = metrics_kwargs.pop("_target_")
                factory = _import(target)
                self.train_metrics[name] = factory(**metrics_kwargs).clone(
                    prefix=f"train/{name}_"
                )
                self.val_metrics[name] = factory(**metrics_kwargs).clone(
                    prefix=f"val/{name}_"
                )

        # Collect SSL strategies from trainer callbacks
        self._ssl_strategies: list = []

    def setup(self, stage=None):
        """Discover SSL strategies from trainer callbacks."""
        if self.trainer:
            from foundry.training.strategies import SSLStrategy
            self._ssl_strategies = [
                cb for cb in self.trainer.callbacks
                if isinstance(cb, SSLStrategy)
            ]

    def forward(self, **kwargs):
        return self.model(**kwargs)

    def training_step(self, batch, batch_idx):
        return self._shared_step("train", batch)

    def validation_step(self, batch, batch_idx):
        return self._shared_step("val", batch)

    def _shared_step(self, stage: str, batch: dict) -> torch.Tensor:
        # Let SSL strategies modify inputs
        for strategy in self._ssl_strategies:
            batch = strategy.modify_forward_inputs(self.model, batch)

        model_inputs, target_values, target_weights, task_index = (
            self._unpack_batch(batch)
        )

        # Forward pass (always request intermediates if strategies exist)
        if self._ssl_strategies:
            outputs, intermediates = self.model(
                **model_inputs, return_intermediates=True
            )
        else:
            outputs = self.model(**model_inputs)
            intermediates = {}

        # Supervised task losses
        total_loss = self._compute_task_losses(
            stage, outputs, target_values, target_weights, task_index
        )

        # SSL losses
        for strategy in self._ssl_strategies:
            ssl_loss = strategy.compute_ssl_loss(
                self.model, batch, outputs, intermediates
            )
            if ssl_loss is not None:
                total_loss = total_loss + ssl_loss
                self.log(f"{stage}/{strategy.__class__.__name__}_loss", ssl_loss)

        self.log(f"{stage}/loss", total_loss, prog_bar=True)
        return total_loss

    def _compute_task_losses(
        self,
        stage: str,
        outputs: dict[str, torch.Tensor],
        target_values: dict[str, torch.Tensor],
        target_weights: dict[str, torch.Tensor],
        task_index: torch.Tensor,
    ) -> torch.Tensor:
        total_loss = torch.tensor(0.0, device=self.device)
        total_sequences = 0

        for name, cfg in self.model.task_configs.items():
            preds = outputs.get(name)
            target = target_values.get(name)
            if preds is None or target is None or target.numel() == 0:
                continue

            weights = target_weights.get(name, 1.0)
            loss = self._task_losses[name](preds, target, weights)

            # Sequence-weighted aggregation (preserves current behavior)
            idx = self.model.router.get_task_index_by_name(name)
            num_sequences = torch.any(task_index == idx, dim=1).sum()
            total_loss = total_loss + loss * num_sequences
            total_sequences += num_sequences

            self.log(f"{stage}/{name}_loss", loss)

            # Metrics
            metrics = self.train_metrics if stage == "train" else self.val_metrics
            if name in metrics:
                metric_preds, metric_target = self._prepare_for_metrics(
                    cfg, preds, target
                )
                metrics[name].update(metric_preds, metric_target)
                self.log_dict(metrics[name], on_step=False, on_epoch=True)

        if total_sequences > 0:
            total_loss = total_loss / total_sequences

        return total_loss

    def _prepare_for_metrics(self, cfg, predictions, targets):
        """Transform predictions for metric consumption based on task kind."""
        if cfg.kind == "multiclass":
            return torch.softmax(predictions, dim=-1), targets
        elif cfg.kind == "binary":
            return torch.softmax(predictions, dim=-1)[:, 1], targets
        return predictions, targets

    def _unpack_batch(self, batch: dict):
        target_values = batch.pop("target_values")
        target_weights = batch.pop("target_weights")
        batch.pop("session_id", None)
        batch.pop("absolute_start", None)
        batch.pop("eval_mask", None)
        task_index = batch["task_index"]
        return batch, target_values, target_weights, task_index

    def _build_param_groups(self) -> list[dict]:
        if self.cwt_lr_multiplier == 1.0:
            return [{"params": list(self.parameters()),
                     "lr": self.learning_rate,
                     "weight_decay": self.weight_decay}]

        cwt_params = []
        other_params = []
        for name, param in self.named_parameters():
            if ".cwt." in name:
                cwt_params.append(param)
            else:
                other_params.append(param)

        groups = [{"params": other_params,
                   "lr": self.learning_rate,
                   "weight_decay": self.weight_decay}]
        if cwt_params:
            groups.append({"params": cwt_params,
                           "lr": self.learning_rate * self.cwt_lr_multiplier,
                           "weight_decay": self.weight_decay})
        return groups

    def configure_optimizers(self):
        param_groups = self._build_param_groups()
        optimizer = torch.optim.AdamW(param_groups)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=self.trainer.max_epochs if self.trainer else 100
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "interval": "epoch"},
        }

    def on_fit_start(self):
        self._configure_wandb_metric_summaries()

    def _configure_wandb_metric_summaries(self):
        from lightning.pytorch.loggers import WandbLogger
        if not isinstance(self.logger, WandbLogger):
            return
        experiment = self.logger.experiment
        for name, cfg in self.model.task_configs.items():
            for metric_name, mode in cfg.metric_summary_modes.items():
                for prefix in ("train", "val"):
                    experiment.define_metric(
                        f"{prefix}/{name}_{metric_name}", summary=mode
                    )
            for prefix in ("train", "val"):
                experiment.define_metric(f"{prefix}/{name}_loss", summary="min")
        experiment.define_metric("train/loss", summary="min")
        experiment.define_metric("val/loss", summary="min")

    def transfer_batch_to_device(self, batch, device, dataloader_idx):
        from lightning.fabric.utilities.apply_func import move_data_to_device
        from lightning_utilities.core.apply_func import apply_to_collection
        batch = apply_to_collection(
            batch, dtype=torch.Tensor,
            function=lambda t: t.float() if t.dtype == torch.float64 else t,
        )
        return move_data_to_device(batch, device)
```

### 8.1 What the training module handles

- Batch unpacking and device transfer
- Per-task loss computation (delegated to instantiated loss modules)
- Sequence-weighted multitask loss aggregation (preserved from current code)
- Metric updates (using task-config-driven MetricCollections)
- SSL strategy integration (via callback hooks)
- Optimizer/scheduler setup (CWT param groups preserved)
- WandB metric summary configuration

### 8.2 What it does NOT handle

- Loss function details (delegated to `TaskLoss` modules)
- Metric selection (delegated to config-driven factories)
- SSL execution logic (delegated to `SSLStrategy` callbacks)
- Label remapping (handled in `TargetExtractor`, data layer)
- Confusion matrix plotting (delegated to `ConfusionMatrixCallback`)
- Model architecture decisions (model owns its forward pass)

---

## 9. Callbacks

### 9.1 Confusion matrix callback

```python
# foundry/training/callbacks.py
import lightning as L


class ConfusionMatrixCallback(L.Callback):
    """Logs confusion matrices for classification tasks at epoch end.

    Discovers classification tasks from model.task_configs — no need to
    inspect readout.kind or maintain a separate class_names dict.
    """

    def __init__(self):
        super().__init__()
        self._confusion_matrices = {}

    def on_fit_start(self, trainer, pl_module):
        from torchmetrics.classification import ConfusionMatrix

        for name, cfg in pl_module.model.task_configs.items():
            if cfg.kind in ("binary", "multiclass"):
                task_type = cfg.kind
                num_classes = cfg.output_dim
                self._confusion_matrices[name] = ConfusionMatrix(
                    task=task_type, num_classes=num_classes
                ).to(pl_module.device)

    def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        # Update confusion matrices from validation predictions
        # (predictions are available via the stored batch outputs)
        pass  # Implementation follows current pattern

    def on_validation_epoch_end(self, trainer, pl_module):
        for name, cm in self._confusion_matrices.items():
            cfg = pl_module.model.task_configs[name]
            matrix = cm.compute()
            fig = _plot_confusion_matrix(
                matrix, name, cfg.class_names
            )
            _log_figure(pl_module.logger, f"val/{name}_confusion_matrix", fig)
            cm.reset()
```

### 9.2 Existing callbacks (unchanged)

`VocabInitializerCallback`, `EffectiveBatchSizeCallback`, and
`ParameterWatcherCallback` are unaffected by this refactor.

---

## 10. Dataset integration

### 10.1 Dataset mixin

```python
# foundry/data/datasets/mixins.py
from foundry.tasks.config import TaskConfig


class TaskMixin:
    """Mixin for datasets that declare which tasks they support."""

    AVAILABLE_TASKS: dict[str, TaskConfig] = {}

    @classmethod
    def get_task(cls, name: str) -> TaskConfig:
        if name not in cls.AVAILABLE_TASKS:
            raise ValueError(
                f"Unknown task '{name}'. Available: {list(cls.AVAILABLE_TASKS)}"
            )
        return cls.AVAILABLE_TASKS[name]

    @classmethod
    def get_tasks(cls, names: list[str] | None = None) -> dict[str, TaskConfig]:
        if names is None:
            return dict(cls.AVAILABLE_TASKS)
        return {n: cls.get_task(n) for n in names}
```

### 10.2 Dataset usage

```python
from foundry.tasks.catalog import (
    MOTOR_IMAGERY_5CLASS,
    MOTOR_IMAGERY_LEFT_RIGHT,
    MOTOR_IMAGERY_RIGHT_FEET,
)


class SchalkWolpawPhysionet2009(TaskMixin, EEGDatasetMixin, Dataset):
    AVAILABLE_TASKS = {
        "motor_imagery_5class": MOTOR_IMAGERY_5CLASS,
        "motor_imagery_left_right": MOTOR_IMAGERY_LEFT_RIGHT,
        "motor_imagery_right_feet": MOTOR_IMAGERY_RIGHT_FEET,
    }
```

### 10.3 Datamodule

```python
class PhysionetDataModule(NeuralDataModule):
    TASK_TO_READOUT = {
        "MotorImagery": ["motor_imagery_5class"],
        "LeftRightImagery": ["motor_imagery_left_right"],
        "RightHandFeetImagery": ["motor_imagery_right_feet"],
    }

    @classmethod
    def get_tasks_for_experiment(cls, task_type: str) -> dict[str, TaskConfig]:
        task_names = cls.TASK_TO_READOUT[task_type]
        return SchalkWolpawPhysionet2009.get_tasks(task_names)
```

`READOUT_CLASS_NAMES` is removed — class names live on `TaskConfig`.

### 10.4 Class weight computation

```python
def compute_class_weights(
    task_configs: dict[str, TaskConfig],
    dataset,
    split: str = "train",
) -> dict[str, list[float]]:
    """Compute inverse-frequency class weights from the dataset.

    Uses TargetExtractor from each classification task config to
    walk the dataset and count label frequencies.
    """
    weights = {}
    for name, cfg in task_configs.items():
        if cfg.kind not in ("binary", "multiclass"):
            continue

        ext_kwargs = dict(cfg.target_extractor)
        ext_kwargs.pop("_target_", None)
        extractor = TargetExtractor(**ext_kwargs)

        # Count labels across sampling intervals
        label_counts = _count_labels(dataset, split, extractor)
        weights[name] = _inverse_frequency_weights(label_counts)

    return weights
```

This replaces the current `compute_class_weights` that inspects
`MappedCrossEntropyLoss` internals. The label mapping is already baked into
`TargetExtractor`, so counts reflect the mapped labels directly.

---

## 11. Hydra experiment wiring

### 11.1 `main.py`

```python
def _build_model_and_data(cfg):
    _populate_data_driven_hyperparams(cfg)

    DataModuleClass = get_class(cfg.data._target_)
    task_configs = DataModuleClass.get_tasks_for_experiment(cfg.data.task_type)

    model = instantiate(cfg.model, task_configs=task_configs)
    tokenizer = model.tokenize if hasattr(model, "tokenize") else None
    datamodule = instantiate(cfg.data, tokenizer=tokenizer)

    return model, datamodule


def _build_lightning_module(cfg, model, datamodule):
    return instantiate(cfg.module, model=model)
```

Note: class weights are no longer passed to the Lightning module constructor.
Instead they're set as a loss parameter in the task config:

```yaml
tasks:
  motor_imagery_5class:
    loss:
      _target_: foundry.tasks.losses.CrossEntropyTaskLoss
      class_weights: [0.2, 0.3, 0.3, 0.1, 0.1]  # or computed at runtime
```

Or computed dynamically and injected before model construction:

```python
task_configs = DataModuleClass.get_tasks_for_experiment(cfg.data.task_type)
if cfg.get("auto_class_weights"):
    weights = compute_class_weights(task_configs, dataset, "train")
    for name, w in weights.items():
        task_configs[name].loss["class_weights"] = w
```

### 11.2 Module config (single)

```yaml
# configs/module/default.yaml
_target_: foundry.training.FoundryModule

learning_rate: ${hyperparameters.learning_rate}
weight_decay: ${hyperparameters.weight_decay}
cwt_lr_multiplier: ${hyperparameters.cwt_lr_multiplier}
```

The `classification.yaml` / `regression.yaml` split is eliminated. One module
handles both — the per-task loss and metrics come from the task configs, not
the module type.

### 11.3 SSL experiment config

```yaml
# configs/experiment/poyo_mae_physionet.yaml
# @package _global_

defaults:
  - override /model: poyo_eeg
  - override /data: physionet/default
  - override /module: default

data:
  task_type: MotorImagery

trainer:
  callbacks:
    intermediate_mae:
      _target_: foundry.training.strategies.IntermediateMAEStrategy
      mask_after_layer: 1
      mask_ratio: 0.5
      decoder_dim: 256

hyperparameters:
  batch_size: 64
  learning_rate: 3e-4
```

SSL is configured as a trainer callback — not a special readout, not a
separate model class, not a different Lightning module.

### 11.4 Experiment YAML (supervised, unchanged feel)

```yaml
# configs/experiment/poyo_ajile_behavior.yaml
# @package _global_

defaults:
  - override /model: poyo_eeg
  - override /data: ajile/singlesess
  - override /module: default

data:
  task_type: behavior

run:
  name: ajile_behavior

hyperparameters:
  batch_size: 100
  sequence_length: 1.0
```

---

## 12. End-to-end data flow

```
┌─────────────────────────────────────────────────────────────────┐
│  1. TASK CONFIG (YAML or Python catalog)                        │
│                                                                 │
│  motor_imagery_left_right:                                      │
│    head: ReadoutHead(output_dim=2)                              │
│    target_extractor: TargetExtractor(label_map={1:0, 2:1}, ...) │
│    loss: CrossEntropyTaskLoss()                                 │
│    metrics: classification_metrics(num_classes=2)               │
└──────────────────────────┬──────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────────┐
│  2. MODEL CONSTRUCTION (main.py)                                │
│                                                                 │
│  task_configs = DataModule.get_tasks_for_experiment("LeftRight") │
│  model = POYOEEGModel(task_configs=task_configs, embed_dim=256) │
│    └─ router.heads["motor_imagery_left_right"] =                │
│         ReadoutHead(embed_dim=256, output_dim=2)                │
└──────────────────────────┬──────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────────┐
│  3. TOKENIZATION (per-sample, CPU)                              │
│                                                                 │
│  model.tokenize(data):                                          │
│    extractor = TargetExtractor(                                 │
│        timestamp_key="motor_imagery_trials.timestamps",         │
│        value_key="motor_imagery_trials.movement_ids",           │
│        label_map={1: 0, 2: 1},                                 │
│    )                                                            │
│    targets = extractor(data)                                    │
│      → {"timestamps": [...], "values": [0, 1, 0, ...]}         │
└──────────────────────────┬──────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────────┐
│  4. FORWARD PASS (GPU, pure nn.Module)                          │
│                                                                 │
│  outputs = model(input_values=..., task_index=...)              │
│    backbone_embs = backbone.encoder → processor → decoder       │
│    router(backbone_embs, task_index):                           │
│      head.forward(embs) → (N, 2) logits   [single-task fast]   │
└──────────────────────────┬──────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────────┐
│  5. LOSS + METRICS (FoundryModule._shared_step)                 │
│                                                                 │
│  loss_fn = CrossEntropyTaskLoss()  # instantiated from config   │
│  loss = loss_fn(preds, targets, weights)                        │
│                                                                 │
│  metric_preds = softmax(preds)[:, 1]  # binary → prob class 1  │
│  metrics.update(metric_preds, targets)                          │
│    → acc, f1, auroc, ...                                        │
└─────────────────────────────────────────────────────────────────┘
```

---

## 13. What stays in torchbrain


| Keeps                                                               | Notes                     |
| ------------------------------------------------------------------- | ------------------------- |
| `torch_brain.dataset.Dataset`                                       | Base dataset class        |
| `torch_brain.data.collate`, `chain`, `pad8`, `pad2d`, `track_batch` | Batching utilities        |
| `torch_brain.data.sampler.RandomFixedWindowSampler`                 | Sampling                  |
| `torch_brain.transforms.Compose`                                    | Transform pipeline        |
| `torch_brain.nn.Embedding`, `InfiniteVocabEmbedding`                | Embedding layers          |
| `torch_brain.nn.RotaryTimeEmbedding`, `RotaryCrossAttention`, etc.  | Attention building blocks |
| `torch_brain.utils.create_linspace_latent_tokens`                   | Utility                   |



| Moves to Foundry                                         | Replacement                                                 |
| -------------------------------------------------------- | ----------------------------------------------------------- |
| `MultitaskReadout`                                       | `ReadoutRouter` + `ReadoutHead` per task                    |
| `prepare_for_multitask_readout`                          | `TargetExtractor` per task (data layer)                     |
| `Loss`, `CrossEntropyLoss`, `MSELoss`                    | `CrossEntropyTaskLoss`, `MSETaskLoss` etc. (training layer) |
| `register_modality`, `ModalitySpec`, `MODALITY_REGISTRY` | `TaskConfig` (YAML + Python catalog)                        |
| `DataType` enum                                          | `TaskConfig.kind` (inferred property)                       |


---

## 14. New package layout

```
foundry/
├── tasks/
│   ├── __init__.py          # re-exports
│   ├── config.py            # TaskConfig dataclass
│   ├── heads.py             # ReadoutHead, MLPReadoutHead
│   ├── targets.py           # TargetExtractor
│   ├── losses.py            # CrossEntropyTaskLoss, MSETaskLoss, FocalTaskLoss
│   ├── metrics.py           # classification_metrics(), regression_metrics()
│   └── catalog.py           # Python constants (MOTOR_IMAGERY_5CLASS, etc.)
├── models/
│   ├── readout.py           # ReadoutRouter
│   ├── poyo_eeg.py          # updated (pure nn.Module forward)
│   ├── baselines.py         # updated
│   └── ...
├── training/
│   ├── module.py            # FoundryModule (single, unified)
│   ├── strategies.py        # SSLStrategy, IntermediateMAEStrategy, etc.
│   ├── callbacks.py         # ConfusionMatrixCallback, existing callbacks
│   └── ...
├── data/
│   ├── datasets/
│   │   ├── mixins.py        # TaskMixin (replaces ModalityMixin)
│   │   └── ...
│   └── datamodules/
│       └── ...
└── core.py                  # updated protocols
configs/
├── tasks/                   # NEW: per-task YAML configs
│   ├── motor_imagery_5class.yaml
│   ├── motor_imagery_left_right.yaml
│   ├── ajile_pose_estimation.yaml
│   └── ...
├── module/
│   └── default.yaml         # replaces classification.yaml + regression.yaml
└── ...
```

`foundry/data/datasets/modalities.py` is deleted entirely.

---

## 15. Migration order


| Step | Scope                                                                                                                 | Breakage                          | Strategy       |
| ---- | --------------------------------------------------------------------------------------------------------------------- | --------------------------------- | -------------- |
| 1    | Add `foundry/tasks/` package: `TargetExtractor`, `ReadoutHead`, task losses, metrics factories, `TaskConfig`, catalog | None — new code only              | Additive       |
| 2    | Add `foundry/models/readout.py` (`ReadoutRouter`) and `configs/tasks/*.yaml`                                          | None — new code only              | Additive       |
| 3a   | Add `LegacyTaskAdapter` that wraps `ModalitySpec` in a `TaskConfig` interface                                         | None — adapter pattern            | Bridge         |
| 3b   | Port `FoundryModule` to use `TaskConfig` interface (with adapter for old specs)                                       | Internal — same external behavior | Refactor       |
| 4    | Port **Physionet** models + dataset + datamodule to new system                                                        | First external breaking change    | Vertical slice |
| 5    | Port **Ajile** (behavior + pose estimation)                                                                           | Incremental                       | Vertical slice |
| 6    | Port **Neurosoft**                                                                                                    | Incremental                       | Vertical slice |
| 7    | Add `foundry/training/strategies.py` + SSL strategy implementations                                                   | None — new code, opt-in           | Additive       |
| 8    | Delete `modalities.py`, old `ModalityMixin`, `resolve_readout_specs`, adapter                                         | Cleanup                           | Final          |
| 9    | Pin `torch-brain` to post-`2426bbb` main                                                                              | Final dependency change           | Finish         |


### 15.1 Adapter pattern for safe migration

The key insight for safe migration is a `LegacyTaskAdapter` that makes old
`ModalitySpec` objects look like new `TaskConfig` objects:

```python
# foundry/tasks/_compat.py (temporary, deleted in step 8)
from foundry.tasks.config import TaskConfig


def adapt_modality_spec(name: str, spec) -> TaskConfig:
    """Wrap a legacy ModalitySpec as a TaskConfig for incremental migration."""
    from torch_brain.registry import DataType

    if spec.type in (DataType.BINARY, DataType.MULTINOMIAL):
        loss_target = "foundry.tasks.losses.CrossEntropyTaskLoss"
        metrics_target = "foundry.tasks.metrics.classification_metrics"
        metrics_kwargs = {"num_classes": spec.dim}
    else:
        loss_target = "foundry.tasks.losses.MSETaskLoss"
        metrics_target = "foundry.tasks.metrics.regression_metrics"
        metrics_kwargs = {}

    return TaskConfig(
        name=name,
        head={"_target_": "foundry.tasks.heads.ReadoutHead", "output_dim": spec.dim},
        target_extractor={
            "_target_": "foundry.tasks.targets.TargetExtractor",
            "timestamp_key": spec.timestamp_key,
            "value_key": spec.value_key,
        },
        loss={"_target_": loss_target},
        metrics={"_target_": metrics_target, **metrics_kwargs},
    )
```

This allows migrating the training module first (step 3b) while datasets and
models still use the old registry. Each subsequent step removes one layer of
adapters until the old code is gone.

---

## 16. Comparison: v2 (`ReadoutModule`) vs v3 (decomposed)


| Dimension                      | v2                                                   | v3                                                             |
| ------------------------------ | ---------------------------------------------------- | -------------------------------------------------------------- |
| **Responsibilities per class** | 5 (forward, targets, loss, metrics, backbone wrap)   | 1 each                                                         |
| **Testability**                | Must instantiate nn.Module to test target extraction | TargetExtractor testable with no GPU, no embed_dim             |
| **YAML overridability**        | Can't swap loss without new Python subclass          | Loss, metrics, head all overridable from CLI                   |
| **SSL approach**               | `wrap_backbone()` inverts control hierarchy          | SSLStrategy callback; model stays pure                         |
| **Single-task performance**    | Always iterates router                               | Fast path: direct head call                                    |
| **Model exportability**        | Model forward depends on readout state               | Model forward is deterministic, compilable                     |
| **Lightning alignment**        | Loss/metrics on model subcomponents                  | Loss/metrics on LightningModule (where Lightning expects them) |
| **Composability**              | One monolithic class per task variant                | Mix-and-match head + loss + metrics + extractor                |
| **Migration safety**           | Big-bang step 3                                      | Adapter pattern enables piece-by-piece                         |


---

## 17. Decisions made (resolved from v2 open questions)

1. `**class_names` source**: Lives on `TaskConfig`. `DataModule.READOUT_CLASS_NAMES`
  is removed.
2. **Class weight computation**: Accepts `dict[str, TaskConfig]`. Uses
  `TargetExtractor` from each config to walk sampling intervals. Computed
   class weights are injected into the loss config before model construction.
3. **Batch key rename**: batch key is `task_index` (prep PR completed before the
  main refactor).
4. `**data.config` injection**: Removed. `TargetExtractor` handles target
  extraction directly — the hook that injected
   `data.config["multitask_readout"]` is unnecessary.
5. **WandB metric summaries**: Declared on `TaskConfig.metric_summary_modes`.
  The Lightning module reads them at `on_fit_start`. No method on the readout.
6. **Multiple SSL objectives**: Naturally supported — each is a separate
  `SSLStrategy` callback. Multiple strategies compose via sequential
   `modify_forward_inputs` + `compute_ssl_loss` calls.
7. **SSL-backbone coupling**: The model exposes intermediates via
  `return_intermediates=True`. SSL strategies consume whatever the model
   provides — coupling is at the protocol level ("model offers intermediates"),
   not the implementation level ("readout iterates processor layers").

---

## 18. Remaining open questions

1. `**return_intermediates` overhead**: Should models always compute/return
  intermediates (small dict of tensor references, negligible memory), or gate
   it behind a flag that's only set when SSL strategies are present? Current
   proposal: gate behind the flag, since supervised-only runs (the majority)
   get zero overhead.
2. **Contrastive dual-view execution**: The model must run the backbone twice
  for contrastive learning. Options: (a) the strategy calls `model.forward()`
   twice from within `compute_ssl_loss`; (b) the dataloader packs two views and
   the strategy modifies batch structure in `modify_forward_inputs`. Leaning
   toward (a) for clarity.
3. **Per-task loss weighting**: Current system uses sequence-weighted averaging.
  Should we also support configurable per-task loss coefficients (e.g.
   `task_configs[name].loss_weight = 0.5`) for multitask balancing?
4. **Inference pipeline**: `predict_step` should skip SSL strategies and loss
  computation. Should it also support per-task prediction routing, or just
   return the raw model output dict?
5. **Strategy parameter optimization**: SSL strategy parameters (decoder,
  projector) need to be included in the optimizer. Should strategies declare
   their parameters via a method, or should `FoundryModule` automatically
   collect parameters from all callbacks that are `nn.Module` subclasses?

