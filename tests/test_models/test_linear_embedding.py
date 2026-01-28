import torch

from foundry.models import LinearEmbedding


class TestLinearEmbedding:
    def test_initialization(self, embed_dim):
        embedding = LinearEmbedding(embed_dim=embed_dim)
        assert embedding.embed_dim == embed_dim
        assert len(embedding.projections) == 0

    def test_forward_pass_basic(self, embed_dim, batch_size):
        embedding = LinearEmbedding(embed_dim=embed_dim)

        num_tokens = 10
        patch_samples = 50

        input_values = torch.randn(batch_size, num_tokens, patch_samples)
        output = embedding(input_values)

        assert output.shape == (batch_size, num_tokens, embed_dim)
        assert len(embedding.projections) == 1

    def test_forward_pass_different_shapes(self, embed_dim, batch_size):
        embedding = LinearEmbedding(embed_dim=embed_dim)

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
        embedding = LinearEmbedding(embed_dim=embed_dim)

        patch_samples = 50

        proj1 = embedding.get_projection(patch_samples)
        proj2 = embedding.get_projection(patch_samples)

        assert proj1 is proj2
        assert len(embedding.projections) == 1

    def test_get_projection_different_dimensions(self, embed_dim):
        embedding = LinearEmbedding(embed_dim=embed_dim)

        proj1 = embedding.get_projection(50)
        proj2 = embedding.get_projection(100)
        proj3 = embedding.get_projection(75)

        assert proj1 is not proj2
        assert proj1 is not proj3
        assert proj2 is not proj3
        assert len(embedding.projections) == 3

    def test_projection_output_dimensions(self, embed_dim):
        embedding = LinearEmbedding(embed_dim=embed_dim)

        patch_samples = 50
        projection = embedding.get_projection(patch_samples)

        assert projection.in_features == patch_samples
        assert projection.out_features == embed_dim

    def test_forward_pass_single_batch(self, embed_dim):
        embedding = LinearEmbedding(embed_dim=embed_dim)

        input_values = torch.randn(1, 20, 100)
        output = embedding(input_values)

        assert output.shape == (1, 20, embed_dim)

    def test_forward_pass_large_batch(self, embed_dim):
        embedding = LinearEmbedding(embed_dim=embed_dim)

        batch_size = 16
        input_values = torch.randn(batch_size, 5, 30)
        output = embedding(input_values)

        assert output.shape == (batch_size, 5, embed_dim)

    def test_projection_initialization(self, embed_dim):
        embedding = LinearEmbedding(embed_dim=embed_dim)
        projection = embedding.get_projection(50)

        assert hasattr(projection, "weight")
        assert hasattr(projection, "bias")
        assert projection.weight.shape == (embed_dim, 50)
        assert projection.bias.shape == (embed_dim,)
