"""
# Architecture Diagram: How core.py Protocols Tie Everything Together

## System Architecture (After Refactoring)

```
┌─────────────────────────────────────────────────────────────────────┐
│                           main.py                                   │
│  (Training Entry Point)                                             │
└──────────────────────┬──────────────────────────────────────────────┘
                       │
        ┌──────────────┼──────────────┐
        │              │              │
        ▼              ▼              ▼
    ┌────────┐   ┌──────────┐   ┌──────────────┐
    │ Model  │   │DataModule│   │    Trainer   │
    └────┬───┘   └────┬─────┘   └──────┬───────┘
         │            │                │
         │            │                │ (adds callback)
         │            │                │
         ▼            ▼                ▼
    ┌─────────────────────────────────────────────┐
    │         foundry/core.py PROTOCOLS           │
    ├─────────────────────────────────────────────┤
    │                                             │
    │  1. NeuralModel Protocol                    │
    │     ├─ forward(**kwargs) -> Dict            │
    │     └─ readout_specs -> Dict                │
    │                                             │
    │  2. Tokenizable Protocol                    │
    │     └─ tokenize(data: Data) -> Dict         │
    │                                             │
    │  3. VocabManager Protocol                   │
    │     ├─ initialize_session_vocab()           │
    │     ├─ initialize_channel_vocab()           │
    │     └─ has_lazy_vocabs()                    │
    │                                             │
    └─────────────────────────────────────────────┘
         ▲              ▲             ▲
         │              │             │
    (satisfies)    (satisfies)    (satisfies)
         │              │             │
         │              │             │
    ┌────┴──────┬───────┴────────┬────┴──────────┐
    │           │                │               │
    │           │                │               │
▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼
│                 MODELS (Can be ANY of these)            │
├─────────────────────────────────────────────────────────┤
│                                                         │
│  POYOEEGModel                                           │
│  ├─ forward() ✓                NeuralModel              │
│  ├─ readout_specs ✓                                     │
│  ├─ tokenize() ✓                Tokenizable             │
│  ├─ initialize_session_vocab() ✓  VocabManager          │
│  ├─ initialize_channel_vocab() ✓                        │
│  └─ has_lazy_vocabs() ✓                                 │
│                                                         │
│  EEGNetEncoder                                          │
│  ├─ forward() ✓                NeuralModel              │
│  ├─ readout_specs ✓                                     │
│  └─ (no tokenize, no vocab manager)                     │
│                                                         │
│  ShallowConvNet                                         │
│  ├─ forward() ✓                NeuralModel              │
│  ├─ readout_specs ✓                                     │
│  └─ (no tokenize, no vocab manager)                     │
│                                                         │
│  SimpleEEGClassifier                                    │
│  ├─ forward() ✓                NeuralModel              │
│  ├─ readout_specs ✓                                     │
│  └─ (no tokenize, no vocab manager)                     │
│                                                         │
│  Future: fMRIModel, PETModel, iEEGModel, etc.           │
│  ├─ forward() ✓                NeuralModel              │
│  ├─ readout_specs ✓                                     │
│  └─ (modality-specific tokenize/vocab if needed)        │
│                                                         │
▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲
    │            │                │
    └────┬───────┴─────────┬──────┘
         │                 │
    ┌────▼─────┐      ┌────▼────────┐
    │  Training│      │  Data Flow  │
    └──────────┘      └─────────────┘
         │                 │
         ├─ NeuralTask     ├─ NeuralDataModule
         │  (works with    │  (generic for any
         │   any model     │   dataset)
         │   satisfying    │
         │   NeuralModel)  ├─ PhysionetDataModule
         │                 │  (extends base,
         │                 │   specific to
         │                 │   PhysioNet)
         │                 │
         │                 ├─ Transforms:
         │                 │  ├─ RescaleSignal
         │                 │  │  (configurable
         │                 │  │   field: "eeg",
         │                 │  │   "fmri", etc.)
         │                 │  └─ Patching
         │                 │
         │                 └─ Tokenizer:
         │                    (optional)
         │                    └─ model.tokenize()
         │
         └─ Callbacks:
            └─ VocabInitializerCallback
               (initializes any model
                satisfying VocabManager
                protocol)
```

## Data Flow for POYOEEGModel (Example)

```
Raw EEG Data (128 channels, 1000 Hz)
         │
         ▼
   RescaleSignal (field="eeg")
   └─ Multiplies by 1e5
         │
         ▼
   Patching (patch_duration=0.1s)
   └─ (T, C) → (P, C, S)
         │
         ▼
   POYOEEGModel.tokenize() ← Satisfies Tokenizable protocol
   ├─ Flattens patches: (P×C, S)
   ├─ Creates input embeddings
   ├─ Generates latent tokens
   ├─ Prepares output queries
   └─ Returns: {
         "input_values": ...,
         "input_timestamps": ...,
         "input_channel_index": ...,
         "target_values": {...},
         ...
      }
         │
         ▼
   POYOEEGModel.forward() ← Satisfies NeuralModel protocol
   ├─ Applies input embedding
   ├─ Adds context (channel + session embeddings)
   ├─ Passes through PerceiverIOBackbone
   └─ Returns: {
         "motor_imagery_right_feet": logits[B, 2],
         ...
      }
         │
         ▼
   NeuralTask (training step) ← Works with ANY NeuralModel
   ├─ Computes loss
   ├─ Updates metrics
   └─ Backpropagates
```

## Alternative Flow for EEGNetEncoder (Simpler Model)

```
Raw EEG Data
         │
         ▼
   RescaleSignal (field="eeg")
         │
         ▼
   Patching
         │
         ▼
   DataModule SKIPS tokenization
   (no tokenize() method on this model)
   └─ Returns raw patches directly
         │
         ▼
   EEGNetEncoder.forward() ← Satisfies NeuralModel protocol
   ├─ Temporal + spatial convolution
   ├─ Pooling
   ├─ Classification layer
   └─ Returns: {
         "motor_imagery_right_feet": logits[B, 2],
      }
         │
         ▼
   NeuralTask (training step) ← Same training loop!
   ├─ Computes loss
   ├─ Updates metrics
   └─ Backpropagates
```

## How Protocols Enable This Flexibility

### Before (Tightly Coupled):

```
DataModule
  └─ Assumes model.tokenize() exists
  └─ Assumes model.session_emb.initialize_vocab()
  └─ Assumes model.channel_emb.initialize_vocab()
  └─ **FAILS if any assumption violated**

EEGTask
  └─ Assumes model is EEG-specific
  └─ **FAILS for fMRI or other modalities**
```

### After (Loosely Coupled via Protocols):

```
DataModule
  └─ Checks: hasattr(model, "tokenize")
  └─ If YES: uses model.tokenize()
  └─ If NO: returns raw data
  └─ **Works with both**

Trainer with VocabInitializerCallback
  └─ Checks: isinstance(model, VocabManager)  # protocol check
  └─ If YES: initializes vocabs
  └─ If NO: skips initialization
  └─ **Works with both**

NeuralTask
  └─ Works with ANY model satisfying NeuralModel protocol
  └─ **Works with EEG, fMRI, PET, iEEG, or anything else**
```

## Key Protocol Relationships

```
┌─────────────────────────────────────────────────┐
│ Every Model is Different                        │
├─────────────────────────────────────────────────┤
│ But they all satisfy NeuralModel:               │
│                                                 │
│   ✓ POYOEEGModel                                │
│   ✓ EEGNetEncoder                               │
│   ✓ ShallowConvNet                              │
│   ✓ SimpleEEGClassifier                         │
│   ✓ Any future model with forward() +           │
│     readout_specs property                      │
│                                                 │
└─────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────┐
│ Some Models are Tokenizable                     │
├─────────────────────────────────────────────────┤
│ (can convert raw data to tensors)               │
│                                                 │
│   ✓ POYOEEGModel (has tokenize())               │
│   ✓ Future custom tokenizers                    │
│   ✗ EEGNetEncoder (expects preprocessed data)   │
│                                                 │
└─────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────┐
│ Some Models need VocabManager                   │
├─────────────────────────────────────────────────┤
│ (have lazy vocabularies to initialize)          │
│                                                 │
│   ✓ POYOEEGModel (has session/channel embeds)   │
│   ✗ EEGNetEncoder (no vocab)                    │
│   ✗ ShallowConvNet (no vocab)                   │
│                                                 │
└─────────────────────────────────────────────────┘
```

## Summary

The protocols in `foundry/core.py` are like a **contract** that says:
- "If you have these methods, you work with our system"
- No inheritance needed
- No registration needed
- Just implement the methods!

This enables:
1. **Multi-modality support** (EEG, fMRI, PET, etc.)
2. **Flexible models** (simple CNNs to complex Perceiver)
3. **Reusable components** (same training loop for all)
4. **Type safety** (static checkers can verify compatibility)
5. **Easy extension** (add new models without modifying core)
"""
