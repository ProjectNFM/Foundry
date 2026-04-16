import torch

from foundry.models.reconstruction_head import ReconstructionHead


class TestReconstructionHead:
    def test_output_shape(self):
        head = ReconstructionHead(embed_dim=128, output_dim=470)
        x = torch.randn(10, 128)
        out = head(x)
        assert out.shape == (10, 470)

    def test_custom_hidden_dim(self):
        head = ReconstructionHead(embed_dim=128, output_dim=50, hidden_dim=256)
        x = torch.randn(5, 128)
        out = head(x)
        assert out.shape == (5, 50)

    def test_default_hidden_dim_equals_embed_dim(self):
        head = ReconstructionHead(embed_dim=64, output_dim=32)
        linear1 = head.net[1]
        assert linear1.in_features == 64
        assert linear1.out_features == 64

    def test_gradient_flow(self):
        head = ReconstructionHead(embed_dim=64, output_dim=100)
        x = torch.randn(3, 64, requires_grad=True)
        out = head(x)
        out.sum().backward()
        assert x.grad is not None

    def test_single_output_dim(self):
        head = ReconstructionHead(embed_dim=32, output_dim=1)
        x = torch.randn(8, 32)
        out = head(x)
        assert out.shape == (8, 1)

    def test_batch_independence(self):
        head = ReconstructionHead(embed_dim=64, output_dim=32)
        head.eval()
        x = torch.randn(4, 64)
        out_batch = head(x)
        out_single = torch.stack(
            [head(x[i : i + 1]).squeeze(0) for i in range(4)]
        )
        torch.testing.assert_close(out_batch, out_single)
