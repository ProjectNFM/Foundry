import pytest
import torch

from foundry.models import CNNEmbedding


class TestCNNEmbedding:
    def test_initialization(self, embed_dim):
        embedding = CNNEmbedding(
            embed_dim=embed_dim,
            num_filters=64,
            kernel_size=3,
            activation="gelu",
        )
        assert embedding.embed_dim == embed_dim
        assert embedding.num_filters == 64
        assert embedding.kernel_size == 3
        assert embedding.activation == "gelu"
        assert len(embedding.projections) == 0

    def test_forward_pass_basic(self, embed_dim, batch_size):
        embedding = CNNEmbedding(
            embed_dim=embed_dim,
            num_filters=32,
            kernel_size=3,
            activation="relu",
        )

        num_tokens = 10
        patch_samples = 50

        input_values = torch.randn(batch_size, num_tokens, patch_samples)
        output = embedding(input_values)

        assert output.shape == (batch_size, num_tokens, embed_dim)
        assert len(embedding.projections) == 1

    def test_forward_pass_different_shapes(self, embed_dim, batch_size):
        embedding = CNNEmbedding(
            embed_dim=embed_dim,
            num_filters=64,
            kernel_size=5,
            activation="gelu",
        )

        shape1 = (batch_size, 10, 50)
        shape2 = (batch_size, 15, 100)

        input1 = torch.randn(*shape1)
        input2 = torch.randn(*shape2)

        output1 = embedding(input1)
        output2 = embedding(input2)

        assert output1.shape == (batch_size, 10, embed_dim)
        assert output2.shape == (batch_size, 15, embed_dim)
        assert len(embedding.projections) == 2

    def test_get_projection_caching(self, embed_dim):
        embedding = CNNEmbedding(
            embed_dim=embed_dim,
            num_filters=32,
            kernel_size=3,
            activation="relu",
        )

        patch_samples = 50

        proj1 = embedding.get_projection(patch_samples)
        proj2 = embedding.get_projection(patch_samples)

        assert proj1 is proj2
        assert len(embedding.projections) == 1

    def test_different_activations(self, embed_dim, batch_size):
        activations = ["relu", "gelu", "silu", "tanh"]

        for activation in activations:
            embedding = CNNEmbedding(
                embed_dim=embed_dim,
                num_filters=32,
                kernel_size=3,
                activation=activation,
            )
            input_values = torch.randn(batch_size, 5, 30)
            output = embedding(input_values)
            assert output.shape == (batch_size, 5, embed_dim)

    def test_invalid_activation(self, embed_dim):
        with pytest.raises(ValueError, match="Unknown activation"):
            embedding = CNNEmbedding(
                embed_dim=embed_dim,
                num_filters=32,
                kernel_size=3,
                activation="invalid_activation",
            )
            embedding.get_projection(50)

    def test_different_kernel_sizes(self, embed_dim, batch_size):
        kernel_sizes = [3, 5, 7]

        for kernel_size in kernel_sizes:
            embedding = CNNEmbedding(
                embed_dim=embed_dim,
                num_filters=32,
                kernel_size=kernel_size,
                activation="relu",
            )
            patch_samples = 50
            input_values = torch.randn(batch_size, 5, patch_samples)
            output = embedding(input_values)
            assert output.shape == (batch_size, 5, embed_dim)

    def test_different_num_filters(self, embed_dim, batch_size):
        num_filters_list = [16, 32, 64, 128]

        for num_filters in num_filters_list:
            embedding = CNNEmbedding(
                embed_dim=embed_dim,
                num_filters=num_filters,
                kernel_size=3,
                activation="gelu",
            )
            input_values = torch.randn(batch_size, 5, 30)
            output = embedding(input_values)
            assert output.shape == (batch_size, 5, embed_dim)

    def test_cnn_structure(self, embed_dim):
        embedding = CNNEmbedding(
            embed_dim=embed_dim,
            num_filters=32,
            kernel_size=3,
            activation="relu",
        )
        projection = embedding.get_projection(50)

        assert len(projection) == 4

    def test_forward_pass_single_batch(self, embed_dim):
        embedding = CNNEmbedding(
            embed_dim=embed_dim,
            num_filters=32,
            kernel_size=3,
            activation="gelu",
        )
        input_values = torch.randn(1, 20, 100)
        output = embedding(input_values)
        assert output.shape == (1, 20, embed_dim)

    def test_forward_pass_large_batch(self, embed_dim):
        embedding = CNNEmbedding(
            embed_dim=embed_dim,
            num_filters=64,
            kernel_size=5,
            activation="silu",
        )
        batch_size = 16
        input_values = torch.randn(batch_size, 5, 30)
        output = embedding(input_values)
        assert output.shape == (batch_size, 5, embed_dim)

    def test_conv_output_dimensions(self, embed_dim):
        embedding = CNNEmbedding(
            embed_dim=embed_dim,
            num_filters=32,
            kernel_size=3,
            activation="relu",
        )
        patch_samples = 50

        projection = embedding.get_projection(patch_samples)
        conv_layer = projection[0]

        assert conv_layer.in_channels == 1
        assert conv_layer.out_channels == 32
        assert conv_layer.kernel_size == (3,)

    def test_device_placement_cpu(self, embed_dim, batch_size):
        embedding = CNNEmbedding(
            embed_dim=embed_dim,
            num_filters=32,
            kernel_size=3,
            activation="relu",
        )
        embedding = embedding.to("cpu")

        input_values = torch.randn(batch_size, 10, 50)
        output = embedding(input_values)

        assert output.device.type == "cpu"
        projection = embedding.projections["50"]
        assert next(projection.parameters()).device.type == "cpu"

    def test_device_placement_cuda(self, embed_dim, batch_size):
        if not torch.cuda.is_available():
            return

        embedding = CNNEmbedding(
            embed_dim=embed_dim,
            num_filters=32,
            kernel_size=3,
            activation="relu",
        )
        embedding = embedding.to("cuda")

        input_values = torch.randn(batch_size, 10, 50, device="cuda")
        output = embedding(input_values)

        assert output.device.type == "cuda"
        projection = embedding.projections["50"]
        assert next(projection.parameters()).device.type == "cuda"

    def test_dynamic_projection_inherits_device(self, embed_dim, batch_size):
        if not torch.cuda.is_available():
            return

        embedding = CNNEmbedding(
            embed_dim=embed_dim,
            num_filters=32,
            kernel_size=3,
            activation="relu",
        )
        embedding = embedding.to("cuda")

        input_values_50 = torch.randn(batch_size, 10, 50, device="cuda")
        output_50 = embedding(input_values_50)

        input_values_100 = torch.randn(batch_size, 10, 100, device="cuda")
        output_100 = embedding(input_values_100)

        assert output_50.device.type == "cuda"
        assert output_100.device.type == "cuda"
        assert next(embedding.projections["50"].parameters()).device.type == "cuda"
        assert next(embedding.projections["100"].parameters()).device.type == "cuda"
