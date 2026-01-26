import pytest
import torch

from foundry.models import PatchEmbedding


class TestPatchEmbedding:
    def test_initialization(self, embed_dim):
        embedding = PatchEmbedding(embed_dim=embed_dim)
        assert embedding.embed_dim == embed_dim
        assert len(embedding.projections) == 0

    def test_forward_pass_basic(self, embed_dim, batch_size):
        embedding = PatchEmbedding(embed_dim=embed_dim)
        
        num_patches = 10
        time_steps = 50
        channels = 4
        
        input_values = torch.randn(batch_size, num_patches, time_steps, channels)
        output = embedding(input_values)
        
        assert output.shape == (batch_size, num_patches, embed_dim)
        assert len(embedding.projections) == 1

    def test_forward_pass_different_shapes(self, embed_dim, batch_size):
        embedding = PatchEmbedding(embed_dim=embed_dim)
        
        shape1 = (batch_size, 10, 50, 4)
        shape2 = (batch_size, 15, 100, 8)
        
        input1 = torch.randn(*shape1)
        input2 = torch.randn(*shape2)
        
        output1 = embedding(input1)
        output2 = embedding(input2)
        
        assert output1.shape == (batch_size, 10, embed_dim)
        assert output2.shape == (batch_size, 15, embed_dim)
        assert len(embedding.projections) == 2

    def test_get_projection_caching(self, embed_dim):
        embedding = PatchEmbedding(embed_dim=embed_dim)
        
        time_steps, channels = 50, 4
        
        proj1 = embedding.get_projection(time_steps, channels)
        proj2 = embedding.get_projection(time_steps, channels)
        
        assert proj1 is proj2
        assert len(embedding.projections) == 1

    def test_get_projection_different_dimensions(self, embed_dim):
        embedding = PatchEmbedding(embed_dim=embed_dim)
        
        proj1 = embedding.get_projection(50, 4)
        proj2 = embedding.get_projection(100, 8)
        proj3 = embedding.get_projection(50, 8)
        
        assert proj1 is not proj2
        assert proj1 is not proj3
        assert proj2 is not proj3
        assert len(embedding.projections) == 3

    def test_projection_output_dimensions(self, embed_dim):
        embedding = PatchEmbedding(embed_dim=embed_dim)
        
        time_steps, channels = 50, 4
        projection = embedding.get_projection(time_steps, channels)
        
        assert projection.in_features == time_steps * channels
        assert projection.out_features == embed_dim

    def test_forward_pass_single_batch(self, embed_dim):
        embedding = PatchEmbedding(embed_dim=embed_dim)
        
        input_values = torch.randn(1, 20, 100, 6)
        output = embedding(input_values)
        
        assert output.shape == (1, 20, embed_dim)

    def test_forward_pass_large_batch(self, embed_dim):
        embedding = PatchEmbedding(embed_dim=embed_dim)
        
        batch_size = 16
        input_values = torch.randn(batch_size, 5, 30, 2)
        output = embedding(input_values)
        
        assert output.shape == (batch_size, 5, embed_dim)

    def test_projection_initialization(self, embed_dim):
        embedding = PatchEmbedding(embed_dim=embed_dim)
        projection = embedding.get_projection(50, 4)
        
        assert hasattr(projection, "weight")
        assert hasattr(projection, "bias")
        assert projection.weight.shape == (embed_dim, 50 * 4)
        assert projection.bias.shape == (embed_dim,)
