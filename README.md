# Foundry

Foundry is a small, modular EEG experimentation library. It focuses on
composable building blocks (embeddings, backbones, and readouts) and keeps
the core surface area intentionally small so experiments stay flexible.

## Modules

- `foundry.data`: datasets and datamodules
- `foundry.models`: embeddings, backbones, and a reference `EEGModel`
- `foundry.transforms`: preprocessing and patching utilities

## Design goals

- **Modular by default**: components are plain `nn.Module` objects
- **Minimal glue code**: avoid deep inheritance or hidden protocols
- **Composable experiments**: swap pieces without rewriting pipelines