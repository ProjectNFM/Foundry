import torch

from foundry.models import LinearEmbedding


class TestLinearEmbedding:
    def test_initialization(self, embed_dim):
        embedding = LinearEmbedding(
            embed_dim=embed_dim, num_channels=8, patch_samples=50
        )
        assert embedding.embed_dim == embed_dim
        assert embedding.num_channels == 8
        assert embedding.patch_samples == 50

    def test_forward_pass_basic(self, embed_dim, batch_size):
        num_channels = 8
        patch_samples = 50
        num_patches = 4

        embedding = LinearEmbedding(
            embed_dim=embed_dim,
            num_channels=num_channels,
            patch_samples=patch_samples,
        )

        input_values = torch.randn(
            batch_size, num_patches, num_channels, patch_samples
        )
        output = embedding(input_values)

        assert output.shape == (batch_size, num_patches, embed_dim)

    def test_projection_dimensions(self, embed_dim):
        num_channels = 8
        patch_samples = 50

        embedding = LinearEmbedding(
            embed_dim=embed_dim,
            num_channels=num_channels,
            patch_samples=patch_samples,
        )

        assert embedding.projection.in_features == num_channels * patch_samples
        assert embedding.projection.out_features == embed_dim
        assert embedding.projection.weight.shape == (
            embed_dim,
            num_channels * patch_samples,
        )
        assert embedding.projection.bias.shape == (embed_dim,)

    def test_forward_pass_single_batch(self, embed_dim):
        num_channels = 4
        patch_samples = 25

        embedding = LinearEmbedding(
            embed_dim=embed_dim,
            num_channels=num_channels,
            patch_samples=patch_samples,
        )

        input_values = torch.randn(1, 10, num_channels, patch_samples)
        output = embedding(input_values)

        assert output.shape == (1, 10, embed_dim)

    def test_forward_pass_large_batch(self, embed_dim):
        num_channels = 16
        patch_samples = 100
        batch_size = 16

        embedding = LinearEmbedding(
            embed_dim=embed_dim,
            num_channels=num_channels,
            patch_samples=patch_samples,
        )

        input_values = torch.randn(batch_size, 5, num_channels, patch_samples)
        output = embedding(input_values)

        assert output.shape == (batch_size, 5, embed_dim)

    def test_device_placement_cpu(self, embed_dim, batch_size):
        num_channels = 8
        patch_samples = 50

        embedding = LinearEmbedding(
            embed_dim=embed_dim,
            num_channels=num_channels,
            patch_samples=patch_samples,
        )
        embedding = embedding.to("cpu")

        input_values = torch.randn(batch_size, 4, num_channels, patch_samples)
        output = embedding(input_values)

        assert output.device.type == "cpu"
        assert embedding.projection.weight.device.type == "cpu"

    def test_device_placement_cuda(self, embed_dim, batch_size):
        if not torch.cuda.is_available():
            return

        num_channels = 8
        patch_samples = 50

        embedding = LinearEmbedding(
            embed_dim=embed_dim,
            num_channels=num_channels,
            patch_samples=patch_samples,
        )
        embedding = embedding.to("cuda")

        input_values = torch.randn(
            batch_size, 4, num_channels, patch_samples, device="cuda"
        )
        output = embedding(input_values)

        assert output.device.type == "cuda"
        assert embedding.projection.weight.device.type == "cuda"

    def test_accepts_kwargs(self, embed_dim, batch_size):
        num_channels = 8
        patch_samples = 50

        embedding = LinearEmbedding(
            embed_dim=embed_dim,
            num_channels=num_channels,
            patch_samples=patch_samples,
        )

        input_values = torch.randn(batch_size, 4, num_channels, patch_samples)
        channel_index = torch.randint(0, 100, (batch_size, num_channels))

        output = embedding(input_values, input_channel_index=channel_index)
        assert output.shape == (batch_size, 4, embed_dim)
