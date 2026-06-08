# Technical Issues Report

This is an excellent, well-thought-out design document. Moving from a monolithic god-object registry (v2) to a decomposed, single-responsibility architecture (v3) aligns perfectly with PyTorch Lightning and Hydra idioms. It significantly improves testability, model exportability, and configuration flexibility.

However, looking closely at the implementation details and your open questions, there are a few critical technical issues, runtime bugs, and performance traps that will cause issues during execution.

## 1. Critical Code-Level Issues & Bugs

🐛 **The `np.empty_like` Garbage Memory Bug**

In `TargetExtractor.__call__` (Section 4.1), you have the following logic for label mapping:

```python
if self.label_map is not None:
    mapped = np.empty_like(values)
    for src, dst in self.label_map.items():
        mapped[values == src] = dst
    values = mapped
```

**The Issue:**  
`np.empty_like` allocates uninitialized memory. If `values` contains any ID that is not explicitly defined in `self.label_map` (such as padding tokens, out-of-bounds tokens, or background classes), those slots will retain arbitrary garbage values from your RAM. This will cause silent, erratic training failures or label crashes downstream.

**The Fix:**  
Initialize your array safely, or copy the original values if unmapped elements should remain unchanged:

```python
# If unmapped values should be preserved:
mapped = values.copy()

# If unmapped values should be explicitly flagged (e.g., as an ignore index):
mapped = np.full_like(values, fill_value=-100)
```

---

📉 **Token-Level Shape Mismatches in `ReadoutRouter`**

In `ReadoutRouter.forward` (Section 4.4), you use boolean masking to select tokens/sequences for a specific task:

```python
mask = task_index == idx
outputs[name] = self.heads[name](output_embs[mask])
```

If `output_embs` has a shape of `(Batch, Tokens, EmbedDim)` and `task_index` has a shape of `(Batch, Tokens)`, applying `output_embs[mask]` collapses the tensor into a 2D shape of `(NumSelectedTokens, EmbedDim)`.

Consequently, `outputs[name]` becomes a 2D tensor of shape `(NumSelectedTokens, OutputDim)`.

**The Issue:**  
In `FoundryModule._compute_task_losses`, you fetch `target = target_values.get(name)`. If `target` is provided by the dataloader as a 2D padded array of shape `(Batch, Tokens)`, passing it directly into the loss function alongside your collapsed 2D preds tensor will trigger a shape mismatch exception. Your loss layer must know exactly how to mask or align the target tensor to match the router's collapsed token-level output.

## 2. Resolving Your Open Questions

### Q1: `return_intermediates` Overhead

Recommendation: Keep it gated behind a flag.

Using a conditional check like `return_intermediates=bool(self._ssl_strategies)` is a clean approach. It prevents your model from building and maintaining a dictionary of tensor references during pure supervised runs, completely eliminating overhead and potential memory graph retention issues where it isn't needed.

### Q2: Contrastive Dual-View Execution Performance Trap

Your Leaning: `(a)` The strategy calls `model.forward()` twice from within `compute_ssl_loss`.

**Recommendation:** Avoid Option `(a)`. If your strategy calls `model.forward()` twice inside the loss computation step, it means the backbone will execute three times total per batch (once during the main `_shared_step` pass and twice more inside the contrastive loss function). This introduces a massive, unnecessary computational bottleneck.

**The Solution:** Choose a modified version of Option `(b)`. Collate your dual views along the batch dimension within your data loader or dataset step. In `modify_forward_inputs`, stack them so they pass through the model backbone in a single, unified forward pass:

```python
# In your Contrastive Strategy / Data layer
# Combine view1 and view2 into a single batch dimension: (2 * Batch, Tokens, ...)
combined_inputs = torch.cat([view1, view2], dim=0)
```

This maximizes GPU parallelization, ensures your normalization layers (`BatchNorm`/`LayerNorm`) process both views consistently, and keeps the backbone execution count to exactly one pass.

### Q3: Per-Task Loss Weighting

Recommendation: Yes, absolutely implement this.

Multi-task learning architectures commonly suffer from gradient dominance issues, where a single regression task with large MSE values overpowers a classification task with smaller Cross-Entropy gradients. Adding a configurable loss_weight: float = 1.0 parameter to your TaskConfig provides an essential knob for balancing multi-task optimization scales.

### Q4 & Q5: SSL Strategy Parameter Optimization Lifecycle

**The Issue:**  
You initialize strategy parameters (like `_decoder` or `_projector`) during the callback's `on_fit_start` hook.

In PyTorch Lightning, `LightningModule.configure_optimizers()` is called **before** `on_fit_start`. Additionally, the trainer's optimizer builder only tracks parameters returned by `self.parameters()` on the core module. Because your strategy modules are nested inside a separate callback class, the optimizer will completely miss them, and their weights will never update.

**The Solution:**  

1. Strategies with trainable parameters must expose them via a dedicated method (e.g., `strategy.get_trainable_parameters()`).
2. Discover and register these parameters inside the `FoundryModule` initialization or `setup("fit")` step—not during `on_fit_start`.

```python
# Inside FoundryModule.setup()
def setup(self, stage=None):
    if self.trainer:
        self._ssl_strategies = [cb for cb in self.trainer.callbacks if isinstance(cb, SSLStrategy)]
        
        # Explicitly register strategy parameters as submodules so they migrate to the correct GPU device
        for i, strategy in enumerate(self._ssl_strategies):
            if hasattr(strategy, "modules_to_optimize"):
                for name, module in strategy.modules_to_optimize.items():
                    self.add_module(f"ssl_{i}_{name}", module)
```

Your `_build_param_groups()` method can then safely iterate through these registered submodules to append them to your optimizer configuration.

## 3. Architecture Checklist for Implementation

[ ] Fix TargetExtractor: Change np.empty_like to values.copy() or use a designated fill value to avoid allocating random garbage memory.

[ ] Align Target Masking: Ensure that FoundryModule._compute_task_losses applies the exact same token-level mask to target values that the ReadoutRouter applies to embeddings.

[ ] Collect Parameters Early: Ensure callback parameters are registered to the core LightningModule during setup so they can be parsed by configure_optimizers.

[ ] Batch Contrastive Views: Run contrastive views through the backbone simultaneously via batch concatenation instead of issuing multiple sequential forward calls.