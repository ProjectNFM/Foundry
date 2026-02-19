from torch_brain.registry import ModalitySpec, MODALITY_REGISTRY

def resolve_readout_specs(
    readout_specs: list[ModalitySpec | str] | dict[str, ModalitySpec]
) -> dict[str, ModalitySpec]:
    """Resolve string modality names to ModalitySpec objects.

    Args:
        readout_specs: List or dict of ModalitySpec objects or string modality names

    Returns:
        Dictionary mapping modality names to ModalitySpec objects
    """
    if isinstance(readout_specs, dict):
        return readout_specs

    resolved = {}
    for spec in readout_specs:
        if isinstance(spec, str):
            if spec not in MODALITY_REGISTRY:
                raise ValueError(
                    f"Unknown modality '{spec}' in registry. "
                    f"Available: {list(MODALITY_REGISTRY.keys())}"
                )
            resolved[spec] = MODALITY_REGISTRY[spec]
        else:
            for name, registry_spec in MODALITY_REGISTRY.items():
                if registry_spec.id == spec.id:
                    resolved[name] = spec
                    break
            else:
                raise ValueError(
                    f"ModalitySpec with id {spec.id} not found in registry"
                )
    return resolved