import numpy as np
import torch

from foundry.models import (
    LinearEmbedding,
    FixedChannelWindowEmbedding,
    EmbeddingBase,
)


class TestLinearEmbedding:
    def test_class_hierarchy(self):
        embedding = LinearEmbedding(
            embed_dim=64, num_channels=8, patch_samples=50
        )
        assert isinstance(embedding, FixedChannelWindowEmbedding)
        assert isinstance(embedding, EmbeddingBase)

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


class TestFixedChannelWindowPretokenize:
    def test_pretokenize_pads_channels(self):
        embedding = LinearEmbedding(
            embed_dim=64, num_channels=8, patch_samples=50
        )
        patches = np.random.randn(4, 3, 50).astype(np.float32)
        tokens = np.array([10, 20, 30])

        result = embedding.pretokenize(patches, tokens)

        assert result["input_values"].shape == (4, 8, 50)
        assert result["input_channel_index"].shape == (8,)
        assert result["input_mask"].shape == (8,)
        assert result["input_mask"][:3].all()
        assert not result["input_mask"][3:].any()
        assert (result["input_values"][:, 3:, :] == 0).all()

    def test_pretokenize_truncates_channels(self):
        embedding = LinearEmbedding(
            embed_dim=64, num_channels=4, patch_samples=50
        )
        patches = np.random.randn(4, 8, 50).astype(np.float32)
        tokens = np.arange(8)

        result = embedding.pretokenize(patches, tokens)

        assert result["input_values"].shape == (4, 4, 50)
        assert result["input_channel_index"].shape == (4,)
        assert result["input_mask"].all()

    def test_pretokenize_exact_channels(self):
        embedding = LinearEmbedding(
            embed_dim=64, num_channels=8, patch_samples=50
        )
        patches = np.random.randn(4, 8, 50).astype(np.float32)
        tokens = np.arange(8)

        result = embedding.pretokenize(patches, tokens)

        assert result["input_values"].shape == (4, 8, 50)
        assert result["input_mask"].all()
        np.testing.assert_array_equal(result["input_values"].numpy(), patches)

    def test_pretokenize_preserves_channel_tokens(self):
        embedding = LinearEmbedding(
            embed_dim=64, num_channels=8, patch_samples=50
        )
        patches = np.random.randn(4, 5, 50).astype(np.float32)
        tokens = np.array([100, 200, 300, 400, 500])

        result = embedding.pretokenize(patches, tokens)

        np.testing.assert_array_equal(
            result["input_channel_index"][:5].numpy(), tokens
        )
        assert (result["input_channel_index"][5:] == 0).all()
