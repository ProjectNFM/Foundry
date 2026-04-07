# Embeddings Package Reorganization - Implementation Summary

## Overview

The `foundry/models/embeddings` package has been reorganized to provide clearer naming and better structural organization that mirrors the actual data flow pipeline. All changes are fully backward-compatible.

## New Structure

### Subpackages Created

```
foundry/models/embeddings/
├── channel/                    # Channel processing stage
│   ├── __init__.py
│   ├── processors.py          # Channel strategies (Fixed, PerChannel, SpatialProjection)
│   └── spatial_projectors.py  # Spatial projector implementations (Linear, Session, Perceiver)
├── temporal/                   # Temporal embedding stage
│   ├── __init__.py
│   ├── patch_linear.py        # Linear patch-based temporal embedding
│   ├── patch_mlp.py           # MLP patch-based temporal embedding
│   ├── patch_cnn.py           # CNN patch-based temporal embedding
│   ├── per_timepoint.py       # Per-timepoint continuous temporal embedding
│   └── cwt.py                 # CWT continuous temporal embedding
├── patching.py                 # Patching operations (patch_signal, compute_patch_timestamps)
├── activations.py              # Activation function helper
└── legacy.py                   # Deprecated base classes
```

### New Module Files

All functionality has been moved to the new subpackages while maintaining the original implementation logic.

## Backward Compatibility

### 1. Shim Files

All original module files have been converted to shims that:
- Import from the new locations
- Emit `DeprecationWarning` when imported
- Re-export all original symbols

**Shim files:**
- `base.py` → imports from `legacy.py` and `activations.py`
- `channel_strategies.py` → imports from `channel/`
- `spatial.py` → imports from `channel/`
- `linear.py` → imports from `temporal/`
- `mlp.py` → imports from `temporal/`
- `cnn.py` → imports from `temporal/`
- `per_timepoint.py` → imports from `temporal/`
- `cwt.py` → imports from `temporal/`

Note: `patching.py` is the new canonical location (not a shim).

### 2. Re-exports

Core symbols are re-exported from `foundry/models/embeddings/__init__.py`:

```python
# These are available at package level:
from foundry.models.embeddings import PatchLinearEmbedding
from foundry.models.embeddings import ChannelStrategy
from foundry.models.embeddings import patch_signal
```

## New Recommended Imports

### Channel Processing

```python
# New way (recommended)
from foundry.models.embeddings.channel import (
    ChannelStrategy,
    FixedChannelStrategy,
    PerChannelStrategy,
    SpatialProjectionStrategy,
    LinearSpatialProjector,
    SessionSpatialProjector,
    PerceiverSpatialProjector,
)

# Legacy shim modules have been removed.
```

### Temporal Embeddings

```python
# New way (recommended)
from foundry.models.embeddings.temporal import (
    PatchLinearEmbedding,
    PatchMLPEmbedding,
    PatchCNNEmbedding,
    PerTimepointEmbedding,
    CWTEmbedding,
    ContinuousCWTLayer,
)

# Legacy shim modules have been removed.
```

### Patch Operations

```python
# New way (recommended)
from foundry.models.embeddings.patching import (
    patch_signal,
    compute_patch_timestamps,
)

# Also available from main package
from foundry.models.embeddings import patch_signal, compute_patch_timestamps
```

### Utilities

```python
# New way (recommended)
from foundry.models.embeddings.activations import get_activation

# Legacy shim modules have been removed.
```

## Updated Internal Code

The following files have been updated to use the new import paths:

### Core Library
- `foundry/models/tokenizer.py` - Updated to import from new locations

### Tests
- `tests/test_models/test_gpu_patching.py`
- `tests/test_models/test_channel_strategies.py`
- `tests/test_models/test_tokenizer.py`

## Class Name Changes

Some classes were renamed to be more descriptive:

| Old Name | New Name | Status |
|----------|----------|--------|
| `LinearEmbedding` | `PatchLinearEmbedding` | Use new name |
| `MLPEmbedding` | `PatchMLPEmbedding` | Use new name |
| `CNNEmbedding` | `PatchCNNEmbedding` | Use new name |
| `PerTimepointEmbedding` | (unchanged) | No change |
| `CWTEmbedding` | (unchanged) | No change |

## Migration Guide

### For Library Users

Migrate imports to the new package structure:

1. **Update imports to use new subpackages:**
   ```python
   # Before
   from foundry.models.embeddings.linear import LinearEmbedding

   # After
   from foundry.models.embeddings.temporal import PatchLinearEmbedding
   ```

2. **Use new class names**:
   ```python
   # Before
   LinearEmbedding(embed_dim=64, num_input_channels=8, patch_samples=25)
   
   # After
   PatchLinearEmbedding(embed_dim=64, num_input_channels=8, patch_samples=25)
   ```

### For Library Developers

When adding new temporal embeddings:
- Place patch-based embeddings in `foundry/models/embeddings/temporal/patch_*.py`
- Place continuous embeddings in `foundry/models/embeddings/temporal/*.py`
- Export from `foundry/models/embeddings/temporal/__init__.py`
- Re-export from `foundry/models/embeddings/__init__.py`

## Benefits of the New Organization

1. **Clearer separation of concerns:**
   - Channel processing (`channel/`)
   - Temporal embedding (`temporal/`)
   - Patch operations (`patching.py`)

2. **More intuitive module names:**
   - `patch_linear.py` clearly indicates patch-based linear embedding
   - `per_timepoint.py` clearly indicates continuous timepoint-by-timepoint processing
   - `patching.py` contains all patching utilities

3. **Better discoverability:**
   - Related functionality grouped in subpackages
   - Deprecated code isolated in `legacy.py`
   - Simple utilities as standalone modules

4. **Mirrors the actual data flow:**
   ```
   Raw signal → Channel processing → Optional patching → Temporal embedding
   ```

## Testing

All existing tests pass with the new organization:
- Backward compatibility verified
- Import paths updated in test files
- No functional changes to any classes

## Future Work

In a future major version, the deprecation shims can be removed:
1. Remove old module files (`base.py`, `channel_strategies.py`, etc.)
2. Optionally remove old class name aliases
3. Update documentation to use new import paths exclusively

## Documentation Location

This implementation follows the plan documented in:
`.cursor/plans/embeddings_naming_reorganization_0f7474b1.plan.md`
