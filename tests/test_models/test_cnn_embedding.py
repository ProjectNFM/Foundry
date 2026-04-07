import pytest
import torch

from foundry.models.embeddings.temporal import PatchCNNEmbedding as CNNEmbedding


class TestCNNEmbedding:
    def test_is_nn_module(self):
        embedding = CNNEmbedding(
            embed_dim=64,
            num_input_channels=8,
            patch_samples=50,
            num_filters=32,
            kernel_size=3,
        )
        assert isinstance(embedding, torch.nn.Module)

    def test_initialization(self, embed_dim):
        embedding = CNNEmbedding(
            embed_dim=embed_dim,
            num_input_channels=8,
            patch_samples=50,
            num_filters=64,
            kernel_size=3,
            activation="gelu",
        )
        assert embedding.embed_dim == embed_dim
        assert embedding.num_input_channels == 8
        assert embedding.patch_samples == 50

    def test_forward_pass_basic(self, embed_dim, batch_size):
        num_channels = 8
        patch_samples = 50
        num_patches = 4

        embedding = CNNEmbedding(
            embed_dim=embed_dim,
            num_input_channels=num_channels,
            patch_samples=patch_samples,
            num_filters=32,
            kernel_size=3,
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
            embedding = CNNEmbedding(
                embed_dim=embed_dim,
                num_input_channels=num_channels,
                patch_samples=patch_samples,
                num_filters=32,
                kernel_size=3,
                activation=activation,
            )
            input_values = torch.randn(
                batch_size, 5, num_channels, patch_samples
            )
            output = embedding(input_values)
            assert output.shape == (batch_size, 5, embed_dim)

    def test_invalid_activation(self, embed_dim):
        with pytest.raises(ValueError, match="Unknown activation"):
            CNNEmbedding(
                embed_dim=embed_dim,
                num_input_channels=8,
                patch_samples=50,
                num_filters=32,
                kernel_size=3,
                activation="invalid_activation",
            )

    def test_different_kernel_sizes(self, embed_dim, batch_size):
        num_channels = 8
        patch_samples = 50

        for kernel_size in [3, 5, 7]:
            embedding = CNNEmbedding(
                embed_dim=embed_dim,
                num_input_channels=num_channels,
                patch_samples=patch_samples,
                num_filters=32,
                kernel_size=kernel_size,
                activation="relu",
            )
            input_values = torch.randn(
                batch_size, 5, num_channels, patch_samples
            )
            output = embedding(input_values)
            assert output.shape == (batch_size, 5, embed_dim)

    def test_different_num_filters(self, embed_dim, batch_size):
        num_channels = 8
        patch_samples = 50

        for num_filters in [16, 32, 64, 128]:
            embedding = CNNEmbedding(
                embed_dim=embed_dim,
                num_input_channels=num_channels,
                patch_samples=patch_samples,
                num_filters=num_filters,
                kernel_size=3,
                activation="gelu",
            )
            input_values = torch.randn(
                batch_size, 5, num_channels, patch_samples
            )
            output = embedding(input_values)
            assert output.shape == (batch_size, 5, embed_dim)

    def test_cnn_structure(self, embed_dim):
        embedding = CNNEmbedding(
            embed_dim=embed_dim,
            num_input_channels=8,
            patch_samples=50,
            num_filters=32,
            kernel_size=3,
            activation="relu",
        )

        assert len(embedding.cnn) == 4

    def test_conv_layer_dimensions(self, embed_dim):
        num_channels = 8
        embedding = CNNEmbedding(
            embed_dim=embed_dim,
            num_input_channels=num_channels,
            patch_samples=50,
            num_filters=32,
            kernel_size=3,
            activation="relu",
        )

        conv_layer = embedding.cnn[0]
        assert conv_layer.in_channels == num_channels
        assert conv_layer.out_channels == 32
        assert conv_layer.kernel_size == (3,)

    def test_forward_pass_single_batch(self, embed_dim):
        num_channels = 4
        patch_samples = 25

        embedding = CNNEmbedding(
            embed_dim=embed_dim,
            num_input_channels=num_channels,
            patch_samples=patch_samples,
            num_filters=32,
            kernel_size=3,
            activation="gelu",
        )
        input_values = torch.randn(1, 20, num_channels, patch_samples)
        output = embedding(input_values)
        assert output.shape == (1, 20, embed_dim)

    def test_forward_pass_large_batch(self, embed_dim):
        num_channels = 8
        patch_samples = 50
        batch_size = 16

        embedding = CNNEmbedding(
            embed_dim=embed_dim,
            num_input_channels=num_channels,
            patch_samples=patch_samples,
            num_filters=64,
            kernel_size=5,
            activation="silu",
        )
        input_values = torch.randn(batch_size, 5, num_channels, patch_samples)
        output = embedding(input_values)
        assert output.shape == (batch_size, 5, embed_dim)

    def test_device_placement_cpu(self, embed_dim, batch_size):
        num_channels = 8
        patch_samples = 50

        embedding = CNNEmbedding(
            embed_dim=embed_dim,
            num_input_channels=num_channels,
            patch_samples=patch_samples,
            num_filters=32,
            kernel_size=3,
            activation="relu",
        )
        embedding = embedding.to("cpu")

        input_values = torch.randn(batch_size, 4, num_channels, patch_samples)
        output = embedding(input_values)

        assert output.device.type == "cpu"
        assert next(embedding.cnn.parameters()).device.type == "cpu"

    def test_device_placement_cuda(self, embed_dim, batch_size):
        if not torch.cuda.is_available():
            return

        num_channels = 8
        patch_samples = 50

        embedding = CNNEmbedding(
            embed_dim=embed_dim,
            num_input_channels=num_channels,
            patch_samples=patch_samples,
            num_filters=32,
            kernel_size=3,
            activation="relu",
        )
        embedding = embedding.to("cuda")

        input_values = torch.randn(
            batch_size, 4, num_channels, patch_samples, device="cuda"
        )
        output = embedding(input_values)

        assert output.device.type == "cuda"
        assert next(embedding.cnn.parameters()).device.type == "cuda"

    def test_accepts_kwargs(self, embed_dim, batch_size):
        num_channels = 8
        patch_samples = 50

        embedding = CNNEmbedding(
            embed_dim=embed_dim,
            num_input_channels=num_channels,
            patch_samples=patch_samples,
            num_filters=32,
            kernel_size=3,
            activation="gelu",
        )

        input_values = torch.randn(batch_size, 4, num_channels, patch_samples)
        output = embedding(
            input_values, input_channel_index=torch.zeros(batch_size, 8)
        )
        assert output.shape == (batch_size, 4, embed_dim)
