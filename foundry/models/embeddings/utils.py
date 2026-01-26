import torch.nn as nn


def get_activation(activation: str) -> nn.Module:
    """
    Get activation function by name.

    Args:
        activation: Name of activation function (relu, gelu, silu, tanh, etc.)

    Returns:
        Activation module instance
    """
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
