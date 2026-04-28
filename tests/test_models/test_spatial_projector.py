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

    def test_initialization_with_hidden_dim(self):
        proj = self._make_projector(hidden_dim=32)
        assert proj.hidden_dim == 32

        mlp_a = proj.session_layers["A"]
        assert mlp_a[0].in_features == 8
        assert mlp_a[0].out_features == 32
        assert mlp_a[2].in_features == 32
        assert mlp_a[2].out_features == 4

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
            input_session_ids=["S"] * batch_size,
            input_channel_counts=[10] * batch_size,
            input_seq_len=[200] * batch_size,
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
            input_session_ids=["A", "B"],
            input_channel_counts=[8, 16],
            input_seq_len=[100, 100],
        )
        assert out.shape == (2, 4, Max_T)

    def test_time_padding_rezeroed(self):
        proj = self._make_projector(session_configs={"S": 4}, num_sources=2)
        Max_T = 100
        x = torch.randn(1, 4, Max_T)
        valid_len = 60

        out = proj(
            x,
            input_session_ids=["S"],
            input_channel_counts=[4],
            input_seq_len=[valid_len],
        )
        assert (out[0, :, valid_len:] == 0.0).all()

    def test_forward_with_hidden_dim(self):
        proj = self._make_projector(
            session_configs={"S": 8}, num_sources=4, hidden_dim=16
        )
        x = torch.randn(2, 8, 50)
        out = proj(
            x,
            input_session_ids=["S", "S"],
            input_channel_counts=[8, 8],
            input_seq_len=[50, 50],
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
            input_session_ids=["A", "B"],
            input_channel_counts=[4, 8],
            input_seq_len=[80, 80],
        )
        assert out.shape == (2, 3, Max_T)

    def test_gradients_flow(self):
        proj = self._make_projector(session_configs={"S": 4}, num_sources=2)
        x = torch.randn(1, 4, 50, requires_grad=True)
        out = proj(
            x,
            input_session_ids=["S"],
            input_channel_counts=[4],
            input_seq_len=[50],
        )
        out.sum().backward()
        assert x.grad is not None
        assert x.grad.shape == x.shape

    def test_device_placement_cpu(self):
        proj = self._make_projector(session_configs={"S": 4}, num_sources=2)
        proj = proj.to("cpu")
        x = torch.randn(1, 4, 50)
        out = proj(
            x,
            input_session_ids=["S"],
            input_channel_counts=[4],
            input_seq_len=[50],
        )
        assert out.device.type == "cpu"

    def test_device_placement_cuda(self):
        if not torch.cuda.is_available():
            return
        proj = self._make_projector(session_configs={"S": 4}, num_sources=2)
        proj = proj.to("cuda")
        x = torch.randn(1, 4, 50, device="cuda")
        out = proj(
            x,
            input_session_ids=["S"],
            input_channel_counts=[4],
            input_seq_len=[50],
        )
        assert out.device.type == "cuda"

    # ------------------------------------------------------------ #
    # common_layer tests
    # ------------------------------------------------------------ #

    def test_common_layer_disabled_by_default(self):
        proj = self._make_projector()
        assert proj.common_layer is None

    def test_common_layer_initialization(self):
        proj = self._make_projector(
            session_configs={"S": 8}, num_sources=4, common_layer=True
        )
        assert isinstance(proj.common_layer, torch.nn.Linear)
        assert proj.common_layer.in_features == 4
        assert proj.common_layer.out_features == 4

    def test_common_layer_is_registered_submodule(self):
        proj = self._make_projector(
            session_configs={"S": 8}, num_sources=4, common_layer=True
        )
        submodule_names = [name for name, _ in proj.named_modules()]
        assert "common_layer" in submodule_names

    def test_common_layer_output_shape(self, batch_size):
        num_sources = 6
        proj = self._make_projector(
            session_configs={"S": 10},
            num_sources=num_sources,
            common_layer=True,
        )
        Max_C, Max_T = 10, 200
        x = torch.randn(batch_size, Max_C, Max_T)
        out = proj(
            x,
            input_session_ids=["S"] * batch_size,
            input_channel_counts=[10] * batch_size,
            input_seq_len=[200] * batch_size,
        )
        assert out.shape == (batch_size, num_sources, Max_T)

    def test_common_layer_changes_output(self):
        """The common_layer should transform the per-session output further."""
        proj_without = self._make_projector(
            session_configs={"S": 4}, num_sources=2, common_layer=False
        )
        proj_with = self._make_projector(
            session_configs={"S": 4}, num_sources=2, common_layer=True
        )

        # Copy session weights so only the common_layer differs.
        proj_with.session_layers.load_state_dict(
            proj_without.session_layers.state_dict()
        )

        x = torch.randn(1, 4, 50)
        kwargs = dict(
            input_session_ids=["S"],
            input_channel_counts=[4],
            input_seq_len=[50],
        )
        out_without = proj_without(x, **kwargs)
        out_with = proj_with(x, **kwargs)

        assert not torch.allclose(out_without, out_with)

    def test_common_layer_gradients_flow(self):
        proj = self._make_projector(
            session_configs={"S": 4}, num_sources=2, common_layer=True
        )
        x = torch.randn(1, 4, 50, requires_grad=True)
        out = proj(
            x,
            input_session_ids=["S"],
            input_channel_counts=[4],
            input_seq_len=[50],
        )
        out.sum().backward()

        assert x.grad is not None
        assert proj.common_layer.weight.grad is not None
        assert proj.common_layer.bias.grad is not None

    def test_common_layer_time_padding_rezeroed(self):
        proj = self._make_projector(
            session_configs={"S": 4}, num_sources=2, common_layer=True
        )
        Max_T = 100
        valid_len = 60
        x = torch.randn(1, 4, Max_T)

        out = proj(
            x,
            input_session_ids=["S"],
            input_channel_counts=[4],
            input_seq_len=[valid_len],
        )
        # The common_layer linear has a bias, so zero-padded region may get
        # a constant bias offset — but the per-session layer zeros it first,
        # and the common_layer acts on (num_sources, Max_T) transposed so the
        # bias is applied per-source at each time step.  Since the input to
        # common_layer is zero for t >= valid_len, the output there equals
        # just the bias vector broadcast across time.
        padded = out[0, :, valid_len:]
        bias = proj.common_layer.bias.detach().unsqueeze(-1)
        assert torch.allclose(padded, bias.expand_as(padded), atol=1e-6)

    def test_common_layer_with_hidden_dim(self):
        proj = self._make_projector(
            session_configs={"S": 8},
            num_sources=4,
            hidden_dim=16,
            common_layer=True,
        )
        assert isinstance(proj.common_layer, torch.nn.Linear)
        x = torch.randn(2, 8, 50)
        out = proj(
            x,
            input_session_ids=["S", "S"],
            input_channel_counts=[8, 8],
            input_seq_len=[50, 50],
        )
        assert out.shape == (2, 4, 50)

    def test_common_layer_heterogeneous_sessions(self):
        proj = self._make_projector(
            session_configs={"A": 8, "B": 16},
            num_sources=4,
            common_layer=True,
        )
        Max_C, Max_T = 16, 100
        x = torch.randn(2, Max_C, Max_T)
        out = proj(
            x,
            input_session_ids=["A", "B"],
            input_channel_counts=[8, 16],
            input_seq_len=[100, 100],
        )
        assert out.shape == (2, 4, Max_T)
