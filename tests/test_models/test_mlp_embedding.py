import pytest
import torch

from foundry.models import MLPEmbedding


class TestMLPEmbedding:
    def test_is_nn_module(self):
        embedding = MLPEmbedding(
            embed_dim=64,
            num_input_channels=8,
            patch_samples=50,
            hidden_dims=[128],
            activation="gelu",
        )
        assert isinstance(embedding, torch.nn.Module)

    def test_initialization(self, embed_dim):
        hidden_dims = [128, 64]
        embedding = MLPEmbedding(
            embed_dim=embed_dim,
            num_input_channels=8,
            patch_samples=50,
            hidden_dims=hidden_dims,
            activation="gelu",
        )
        assert embedding.embed_dim == embed_dim
        assert embedding.num_input_channels == 8
        assert embedding.patch_samples == 50

    def test_forward_pass_basic(self, embed_dim, batch_size):
        num_channels = 8
        patch_samples = 50
        num_patches = 4

        embedding = MLPEmbedding(
            embed_dim=embed_dim,
            num_input_channels=num_channels,
            patch_samples=patch_samples,
            hidden_dims=[128, 64],
            activation="relu",
        )

        input_values = torch.randn(
            batch_size, num_patches, num_channels, patch_samples
        )
        output = embedding(input_values)

        assert output.shape == (batch_size, num_patches, embed_dim)

    def test_different_activations(self, embed_dim, batch_size):
        num_channels = 8
        patch_samples = 50

        for activation in ["relu", "gelu", "silu", "tanh"]:
            embedding = MLPEmbedding(
                embed_dim=embed_dim,
                num_input_channels=num_channels,
                patch_samples=patch_samples,
                hidden_dims=[64],
                activation=activation,
            )
            input_values = torch.randn(
                batch_size, 5, num_channels, patch_samples
            )
            output = embedding(input_values)
            assert output.shape == (batch_size, 5, embed_dim)

    def test_invalid_activation(self, embed_dim):
        with pytest.raises(ValueError, match="Unknown activation"):
            MLPEmbedding(
                embed_dim=embed_dim,
                num_input_channels=8,
                patch_samples=50,
                hidden_dims=[64],
                activation="invalid_activation",
            )

    def test_single_hidden_layer(self, embed_dim, batch_size):
        num_channels = 8
        patch_samples = 50

        embedding = MLPEmbedding(
            embed_dim=embed_dim,
            num_input_channels=num_channels,
            patch_samples=patch_samples,
            hidden_dims=[128],
            activation="relu",
        )
        input_values = torch.randn(batch_size, 10, num_channels, patch_samples)
        output = embedding(input_values)
        assert output.shape == (batch_size, 10, embed_dim)

    def test_multiple_hidden_layers(self, embed_dim, batch_size):
        num_channels = 8
        patch_samples = 50

        embedding = MLPEmbedding(
            embed_dim=embed_dim,
            num_input_channels=num_channels,
            patch_samples=patch_samples,
            hidden_dims=[256, 128, 64],
            activation="gelu",
        )
        input_values = torch.randn(batch_size, 10, num_channels, patch_samples)
        output = embedding(input_values)
        assert output.shape == (batch_size, 10, embed_dim)

    def test_mlp_structure(self, embed_dim):
        hidden_dims = [128, 64]
        embedding = MLPEmbedding(
            embed_dim=embed_dim,
            num_input_channels=8,
            patch_samples=50,
            hidden_dims=hidden_dims,
            activation="relu",
        )

        # 1 output layer + 2 * num_hidden (linear + activation per hidden)
        expected_layers = 1 + 2 * len(hidden_dims)
        assert len(embedding.mlp) == expected_layers

    def test_forward_pass_single_batch(self, embed_dim):
        num_channels = 4
        patch_samples = 25

        embedding = MLPEmbedding(
            embed_dim=embed_dim,
            num_input_channels=num_channels,
            patch_samples=patch_samples,
            hidden_dims=[128],
            activation="gelu",
        )
        input_values = torch.randn(1, 20, num_channels, patch_samples)
        output = embedding(input_values)
        assert output.shape == (1, 20, embed_dim)

    def test_forward_pass_large_batch(self, embed_dim):
        num_channels = 8
        patch_samples = 50
        batch_size = 16

        embedding = MLPEmbedding(
            embed_dim=embed_dim,
            num_input_channels=num_channels,
            patch_samples=patch_samples,
            hidden_dims=[128, 64],
            activation="silu",
        )
        input_values = torch.randn(batch_size, 5, num_channels, patch_samples)
        output = embedding(input_values)
        assert output.shape == (batch_size, 5, embed_dim)

    def test_device_placement_cpu(self, embed_dim, batch_size):
        num_channels = 8
        patch_samples = 50

        embedding = MLPEmbedding(
            embed_dim=embed_dim,
            num_input_channels=num_channels,
            patch_samples=patch_samples,
            hidden_dims=[128],
            activation="relu",
        )
        embedding = embedding.to("cpu")

        input_values = torch.randn(batch_size, 4, num_channels, patch_samples)
        output = embedding(input_values)

        assert output.device.type == "cpu"
        assert next(embedding.mlp.parameters()).device.type == "cpu"

    def test_device_placement_cuda(self, embed_dim, batch_size):
        if not torch.cuda.is_available():
            return

        num_channels = 8
        patch_samples = 50

        embedding = MLPEmbedding(
            embed_dim=embed_dim,
            num_input_channels=num_channels,
            patch_samples=patch_samples,
            hidden_dims=[128],
            activation="relu",
        )
        embedding = embedding.to("cuda")

        input_values = torch.randn(
            batch_size, 4, num_channels, patch_samples, device="cuda"
        )
        output = embedding(input_values)

        assert output.device.type == "cuda"
        assert next(embedding.mlp.parameters()).device.type == "cuda"

    def test_accepts_kwargs(self, embed_dim, batch_size):
        num_channels = 8
        patch_samples = 50

        embedding = MLPEmbedding(
            embed_dim=embed_dim,
            num_input_channels=num_channels,
            patch_samples=patch_samples,
            hidden_dims=[64],
            activation="gelu",
        )

        input_values = torch.randn(batch_size, 4, num_channels, patch_samples)
        output = embedding(
            input_values, input_channel_index=torch.zeros(batch_size, 8)
        )
        assert output.shape == (batch_size, 4, embed_dim)
