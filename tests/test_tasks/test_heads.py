import torch

from foundry.tasks.heads import MLPReadoutHead, ReadoutHead


class TestReadoutHead:
    def test_output_shape(self):
        head = ReadoutHead(embed_dim=256, output_dim=5)
        embeddings = torch.randn(4, 256)

        output = head(embeddings)

        assert output.shape == (4, 5)


class TestMLPReadoutHead:
    def test_output_shape_default_depth(self):
        head = MLPReadoutHead(embed_dim=128, output_dim=10)
        embeddings = torch.randn(3, 128)

        output = head(embeddings)

        assert output.shape == (3, 10)

    def test_configurable_depth_and_relu_activation(self):
        head = MLPReadoutHead(
            embed_dim=64,
            output_dim=7,
            hidden_dim=32,
            num_layers=3,
            activation="relu",
        )
        embeddings = torch.randn(2, 64)

        output = head(embeddings)

        assert output.shape == (2, 7)
        assert isinstance(head.net[1], torch.nn.ReLU)

    def test_gelu_activation(self):
        head = MLPReadoutHead(embed_dim=64, output_dim=3, activation="gelu")

        assert isinstance(head.net[1], torch.nn.GELU)
