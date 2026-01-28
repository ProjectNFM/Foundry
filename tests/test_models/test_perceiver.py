import torch

from foundry.models import (
    PerceiverDecoder,
    PerceiverEncoder,
    PerceiverIOBackbone,
    PerceiverProcessor,
)


class TestPerceiverEncoder:
    def test_initialization(self, embed_dim):
        encoder = PerceiverEncoder(embed_dim=embed_dim)
        assert encoder.embed_dim == embed_dim

    def test_forward_pass_basic(self, embed_dim, batch_size):
        encoder = PerceiverEncoder(embed_dim=embed_dim)

        num_latents = 20
        num_inputs = 100
        dim_head = 64

        latents = torch.randn(batch_size, num_latents, embed_dim)
        inputs = torch.randn(batch_size, num_inputs, embed_dim)
        latent_timestamp_emb = torch.randn(
            batch_size, num_latents, dim_head * 2
        )
        input_timestamp_emb = torch.randn(batch_size, num_inputs, dim_head * 2)

        output = encoder(
            latents=latents,
            inputs=inputs,
            latent_timestamp_emb=latent_timestamp_emb,
            input_timestamp_emb=input_timestamp_emb,
        )

        assert output.shape == (batch_size, num_latents, embed_dim)

    def test_forward_pass_with_mask(self, embed_dim, batch_size):
        encoder = PerceiverEncoder(embed_dim=embed_dim)

        num_latents = 20
        num_inputs = 100
        dim_head = 64

        latents = torch.randn(batch_size, num_latents, embed_dim)
        inputs = torch.randn(batch_size, num_inputs, embed_dim)
        latent_timestamp_emb = torch.randn(
            batch_size, num_latents, dim_head * 2
        )
        input_timestamp_emb = torch.randn(batch_size, num_inputs, dim_head * 2)
        input_mask = torch.ones(batch_size, num_inputs, dtype=torch.bool)

        output = encoder(
            latents=latents,
            inputs=inputs,
            latent_timestamp_emb=latent_timestamp_emb,
            input_timestamp_emb=input_timestamp_emb,
            input_mask=input_mask,
        )

        assert output.shape == (batch_size, num_latents, embed_dim)

    def test_custom_heads(self, embed_dim, batch_size):
        encoder = PerceiverEncoder(embed_dim=embed_dim, cross_heads=4)

        latents = torch.randn(batch_size, 20, embed_dim)
        inputs = torch.randn(batch_size, 100, embed_dim)
        latent_timestamp_emb = torch.randn(batch_size, 20, 64 * 2)
        input_timestamp_emb = torch.randn(batch_size, 100, 64 * 2)

        output = encoder(
            latents=latents,
            inputs=inputs,
            latent_timestamp_emb=latent_timestamp_emb,
            input_timestamp_emb=input_timestamp_emb,
        )

        assert output.shape == (batch_size, 20, embed_dim)


class TestPerceiverProcessor:
    def test_initialization(self, embed_dim):
        processor = PerceiverProcessor(embed_dim=embed_dim, depth=2)
        assert processor.embed_dim == embed_dim
        assert len(processor.layers) == 2

    def test_forward_pass_basic(self, embed_dim, batch_size):
        processor = PerceiverProcessor(embed_dim=embed_dim, depth=2)

        num_latents = 20
        dim_head = 64

        latents = torch.randn(batch_size, num_latents, embed_dim)
        latent_timestamp_emb = torch.randn(
            batch_size, num_latents, dim_head * 2
        )

        output = processor(
            latents=latents, latent_timestamp_emb=latent_timestamp_emb
        )

        assert output.shape == (batch_size, num_latents, embed_dim)

    def test_different_depths(self, embed_dim, batch_size):
        for depth in [1, 2, 4, 6]:
            processor = PerceiverProcessor(embed_dim=embed_dim, depth=depth)
            assert len(processor.layers) == depth

            latents = torch.randn(batch_size, 20, embed_dim)
            latent_timestamp_emb = torch.randn(batch_size, 20, 64 * 2)

            output = processor(
                latents=latents, latent_timestamp_emb=latent_timestamp_emb
            )
            assert output.shape == (batch_size, 20, embed_dim)

    def test_custom_heads(self, embed_dim, batch_size):
        processor = PerceiverProcessor(
            embed_dim=embed_dim, depth=2, self_heads=12
        )

        latents = torch.randn(batch_size, 20, embed_dim)
        latent_timestamp_emb = torch.randn(batch_size, 20, 64 * 2)

        output = processor(
            latents=latents, latent_timestamp_emb=latent_timestamp_emb
        )

        assert output.shape == (batch_size, 20, embed_dim)


class TestPerceiverDecoder:
    def test_initialization(self, embed_dim):
        decoder = PerceiverDecoder(embed_dim=embed_dim)
        assert decoder.embed_dim == embed_dim

    def test_forward_pass_basic(self, embed_dim, batch_size):
        decoder = PerceiverDecoder(embed_dim=embed_dim)

        num_queries = 30
        num_latents = 20
        dim_head = 64

        queries = torch.randn(batch_size, num_queries, embed_dim)
        latents = torch.randn(batch_size, num_latents, embed_dim)
        query_timestamp_emb = torch.randn(batch_size, num_queries, dim_head * 2)
        latent_timestamp_emb = torch.randn(
            batch_size, num_latents, dim_head * 2
        )

        output = decoder(
            queries=queries,
            latents=latents,
            query_timestamp_emb=query_timestamp_emb,
            latent_timestamp_emb=latent_timestamp_emb,
        )

        assert output.shape == (batch_size, num_queries, embed_dim)

    def test_custom_heads(self, embed_dim, batch_size):
        decoder = PerceiverDecoder(embed_dim=embed_dim, cross_heads=4)

        queries = torch.randn(batch_size, 30, embed_dim)
        latents = torch.randn(batch_size, 20, embed_dim)
        query_timestamp_emb = torch.randn(batch_size, 30, 64 * 2)
        latent_timestamp_emb = torch.randn(batch_size, 20, 64 * 2)

        output = decoder(
            queries=queries,
            latents=latents,
            query_timestamp_emb=query_timestamp_emb,
            latent_timestamp_emb=latent_timestamp_emb,
        )

        assert output.shape == (batch_size, 30, embed_dim)


class TestPerceiverIOBackbone:
    def test_initialization(self, embed_dim):
        backbone = PerceiverIOBackbone(embed_dim=embed_dim, depth=2)
        assert backbone.embed_dim == embed_dim
        assert hasattr(backbone, "encoder")
        assert hasattr(backbone, "processor")
        assert hasattr(backbone, "decoder")

    def test_forward_pass_end_to_end(self, embed_dim, batch_size):
        backbone = PerceiverIOBackbone(embed_dim=embed_dim, depth=2)

        num_inputs = 100
        num_latents = 20
        num_outputs = 30
        dim_head = 64

        inputs = torch.randn(batch_size, num_inputs, embed_dim)
        input_timestamp_emb = torch.randn(batch_size, num_inputs, dim_head * 2)
        latents = torch.randn(batch_size, num_latents, embed_dim)
        latent_timestamp_emb = torch.randn(
            batch_size, num_latents, dim_head * 2
        )
        output_queries = torch.randn(batch_size, num_outputs, embed_dim)
        output_timestamp_emb = torch.randn(
            batch_size, num_outputs, dim_head * 2
        )

        output = backbone(
            inputs=inputs,
            input_timestamp_emb=input_timestamp_emb,
            latents=latents,
            latent_timestamp_emb=latent_timestamp_emb,
            output_queries=output_queries,
            output_timestamp_emb=output_timestamp_emb,
        )

        assert output.shape == (batch_size, num_outputs, embed_dim)

    def test_forward_pass_with_mask(self, embed_dim, batch_size):
        backbone = PerceiverIOBackbone(embed_dim=embed_dim, depth=2)

        num_inputs = 100
        num_latents = 20
        num_outputs = 30
        dim_head = 64

        inputs = torch.randn(batch_size, num_inputs, embed_dim)
        input_timestamp_emb = torch.randn(batch_size, num_inputs, dim_head * 2)
        input_mask = torch.ones(batch_size, num_inputs, dtype=torch.bool)
        latents = torch.randn(batch_size, num_latents, embed_dim)
        latent_timestamp_emb = torch.randn(
            batch_size, num_latents, dim_head * 2
        )
        output_queries = torch.randn(batch_size, num_outputs, embed_dim)
        output_timestamp_emb = torch.randn(
            batch_size, num_outputs, dim_head * 2
        )

        output = backbone(
            inputs=inputs,
            input_timestamp_emb=input_timestamp_emb,
            latents=latents,
            latent_timestamp_emb=latent_timestamp_emb,
            output_queries=output_queries,
            output_timestamp_emb=output_timestamp_emb,
            input_mask=input_mask,
        )

        assert output.shape == (batch_size, num_outputs, embed_dim)

    def test_custom_configuration(self, embed_dim, batch_size):
        backbone = PerceiverIOBackbone(
            embed_dim=embed_dim,
            depth=4,
            cross_heads=2,
            self_heads=12,
            dim_head=32,
        )

        inputs = torch.randn(batch_size, 100, embed_dim)
        input_timestamp_emb = torch.randn(batch_size, 100, 32 * 2)
        latents = torch.randn(batch_size, 20, embed_dim)
        latent_timestamp_emb = torch.randn(batch_size, 20, 32 * 2)
        output_queries = torch.randn(batch_size, 30, embed_dim)
        output_timestamp_emb = torch.randn(batch_size, 30, 32 * 2)

        output = backbone(
            inputs=inputs,
            input_timestamp_emb=input_timestamp_emb,
            latents=latents,
            latent_timestamp_emb=latent_timestamp_emb,
            output_queries=output_queries,
            output_timestamp_emb=output_timestamp_emb,
        )

        assert output.shape == (batch_size, 30, embed_dim)

    def test_different_sequence_lengths(self, embed_dim, batch_size):
        backbone = PerceiverIOBackbone(embed_dim=embed_dim, depth=2)

        dim_head = 64

        test_cases = [
            (50, 10, 20),
            (200, 40, 60),
            (500, 100, 150),
        ]

        for num_inputs, num_latents, num_outputs in test_cases:
            inputs = torch.randn(batch_size, num_inputs, embed_dim)
            input_timestamp_emb = torch.randn(
                batch_size, num_inputs, dim_head * 2
            )
            latents = torch.randn(batch_size, num_latents, embed_dim)
            latent_timestamp_emb = torch.randn(
                batch_size, num_latents, dim_head * 2
            )
            output_queries = torch.randn(batch_size, num_outputs, embed_dim)
            output_timestamp_emb = torch.randn(
                batch_size, num_outputs, dim_head * 2
            )

            output = backbone(
                inputs=inputs,
                input_timestamp_emb=input_timestamp_emb,
                latents=latents,
                latent_timestamp_emb=latent_timestamp_emb,
                output_queries=output_queries,
                output_timestamp_emb=output_timestamp_emb,
            )

            assert output.shape == (batch_size, num_outputs, embed_dim)
