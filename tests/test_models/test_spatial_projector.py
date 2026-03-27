import torch

from foundry.models import SessionSpatialProjector


class TestSessionSpatialProjector:
    def _make_projector(self, session_configs=None, num_sources=4, **kwargs):
        if session_configs is None:
            session_configs = {"A": 8, "B": 16}
        return SessionSpatialProjector(
            session_configs=session_configs,
            num_sources=num_sources,
            **kwargs,
        )

    def test_initialization(self):
        proj = self._make_projector()
        assert proj.num_sources == 4
        assert "A" in proj.session_layers
        assert "B" in proj.session_layers
        assert proj.shared_mlp is None

    def test_initialization_with_shared_hidden(self):
        proj = self._make_projector(shared_hidden_dim=32)
        assert proj.shared_hidden_dim == 32
        assert proj.shared_mlp is not None

        layer_a = proj.session_layers["A"]
        assert layer_a.in_features == 8
        assert layer_a.out_features == 32

    def test_session_layer_dimensions(self):
        proj = self._make_projector()
        assert proj.session_layers["A"].in_features == 8
        assert proj.session_layers["A"].out_features == 4
        assert proj.session_layers["B"].in_features == 16
        assert proj.session_layers["B"].out_features == 4

    def test_forward_output_shape(self, batch_size):
        num_sources = 6
        proj = self._make_projector(
            session_configs={"S": 10}, num_sources=num_sources
        )

        Max_C, Max_T = 10, 200
        x = torch.randn(batch_size, Max_C, Max_T)
        out = proj(
            x,
            session_ids=["S"] * batch_size,
            channel_counts=[10] * batch_size,
            seq_lens=[200] * batch_size,
        )
        assert out.shape == (batch_size, num_sources, Max_T)

    def test_forward_heterogeneous_sessions(self):
        proj = self._make_projector(
            session_configs={"A": 8, "B": 16}, num_sources=4
        )
        Max_C, Max_T = 16, 100
        x = torch.randn(2, Max_C, Max_T)

        out = proj(
            x,
            session_ids=["A", "B"],
            channel_counts=[8, 16],
            seq_lens=[100, 100],
        )
        assert out.shape == (2, 4, Max_T)

    def test_time_padding_rezeroed(self):
        proj = self._make_projector(session_configs={"S": 4}, num_sources=2)
        Max_T = 100
        x = torch.randn(1, 4, Max_T)
        valid_len = 60

        out = proj(
            x,
            session_ids=["S"],
            channel_counts=[4],
            seq_lens=[valid_len],
        )
        assert (out[0, :, valid_len:] == 0.0).all()

    def test_forward_with_shared_mlp(self):
        proj = self._make_projector(
            session_configs={"S": 8}, num_sources=4, shared_hidden_dim=16
        )
        x = torch.randn(2, 8, 50)
        out = proj(
            x,
            session_ids=["S", "S"],
            channel_counts=[8, 8],
            seq_lens=[50, 50],
        )
        assert out.shape == (2, 4, 50)

    def test_forward_variable_channel_counts(self):
        proj = self._make_projector(
            session_configs={"A": 4, "B": 8}, num_sources=3
        )
        Max_C, Max_T = 8, 80
        x = torch.zeros(2, Max_C, Max_T)
        x[0, :4, :] = torch.randn(4, Max_T)
        x[1, :8, :] = torch.randn(8, Max_T)

        out = proj(
            x,
            session_ids=["A", "B"],
            channel_counts=[4, 8],
            seq_lens=[80, 80],
        )
        assert out.shape == (2, 3, Max_T)

    def test_gradients_flow(self):
        proj = self._make_projector(session_configs={"S": 4}, num_sources=2)
        x = torch.randn(1, 4, 50, requires_grad=True)
        out = proj(x, session_ids=["S"], channel_counts=[4], seq_lens=[50])
        out.sum().backward()
        assert x.grad is not None
        assert x.grad.shape == x.shape

    def test_device_placement_cpu(self):
        proj = self._make_projector(session_configs={"S": 4}, num_sources=2)
        proj = proj.to("cpu")
        x = torch.randn(1, 4, 50)
        out = proj(x, session_ids=["S"], channel_counts=[4], seq_lens=[50])
        assert out.device.type == "cpu"

    def test_device_placement_cuda(self):
        if not torch.cuda.is_available():
            return
        proj = self._make_projector(session_configs={"S": 4}, num_sources=2)
        proj = proj.to("cuda")
        x = torch.randn(1, 4, 50, device="cuda")
        out = proj(x, session_ids=["S"], channel_counts=[4], seq_lens=[50])
        assert out.device.type == "cuda"
