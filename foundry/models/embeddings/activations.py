import torch.nn as nn


def get_activation(activation: str) -> nn.Module:
    activations = {
        "relu": nn.ReLU(),
        "gelu": nn.GELU(),
        "silu": nn.SiLU(),
        "tanh": nn.Tanh(),
        "sigmoid": nn.Sigmoid(),
        "leaky_relu": nn.LeakyReLU(),
        "elu": nn.ELU(),
    }
    if activation.lower() not in activations:
        raise ValueError(
            f"Unknown activation: {activation}. "
            f"Available: {list(activations.keys())}"
        )
    return activations[activation.lower()]


__all__ = ["get_activation"]
