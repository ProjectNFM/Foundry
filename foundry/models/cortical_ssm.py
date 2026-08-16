import math
from typing import Dict

import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint

from foundry.models.readout import build_readout_router
from foundry.tasks.config import TaskConfig
from foundry.models.baselines import BaselineEEGModel

def hippo_n_eigenvalues(state_dim: int) -> torch.Tensor:

    n = torch.arange(state_dim, dtype=torch.float64)
    real = -0.5 * torch.ones(state_dim, dtype=torch.float64)
    imag = math.pi * n

    return torch.complex(real, imag).to(torch.complex64)


def parallel_associative_scan(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    def scan_fn(A, B):
        A = A.clone()
        B = B.clone()
        T = A.shape[1]
        offset = 1
        while offset < T:
            A_shifted = torch.cat(
                [torch.ones_like(A[:, :offset]), A[:, : T - offset]], dim=1
            )
            B_shifted = torch.cat(
                [torch.zeros_like(B[:, :offset]), B[:, : T - offset]], dim=1
            )
            B = A * B_shifted + B
            A = A * A_shifted
            offset *= 2
        return B

    if A.requires_grad or B.requires_grad:
        return checkpoint(scan_fn, A, B, use_reentrant=False)
    return scan_fn(A, B)

class S5SSM(nn.Module):
    def __init__(
      self,
      state_dim: int,
      input_dim: int,
      output_dim: int | None = None,
      dt_min: float = 1e-3,
      dt_max: float = 1e-1,
    ):
        super().__init__()
        output_dim = output_dim or input_dim
        self.state_dim = state_dim
        self.input_dim = input_dim
        self.output_dim = output_dim

        Lambda = hippo_n_eigenvalues(state_dim)

        self.log_neg_Lambda_re = nn.Parameter(torch.log(-Lambda.real.clone()))
        self.Lambda_im = nn.Parameter(Lambda.imag.clone())

        self.B_re = nn.Parameter(torch.randn(state_dim, input_dim) / math.sqrt(input_dim))
        self.B_im = nn.Parameter(torch.randn(state_dim, input_dim) / math.sqrt(input_dim))
        self.C_re = nn.Parameter(torch.randn(output_dim, state_dim) / math.sqrt(state_dim))
        self.C_im = nn.Parameter(torch.randn(output_dim, state_dim) / math.sqrt(state_dim))

        if output_dim == input_dim:
            self.D = nn.Parameter(torch.ones(output_dim))
        else:
            self.register_parameter("D", None)

        log_dt = torch.rand(state_dim) * (math.log(dt_max) - math.log(dt_min)) + math.log(dt_min)
        self.log_step = nn.Parameter(log_dt)

    def _discretize(self):
        Lambda = torch.complex(-torch.exp(self.log_neg_Lambda_re), self.Lambda_im)
        step = torch.exp(self.log_step).to(Lambda.dtype)

        # e ^ (lambda * delta)
        Lambda_bar = torch.exp(Lambda * step)

        # ~B
        B = torch.complex(self.B_re, self.B_im)

        # B_bar = lambda ^ -1 (lambda_bar - I) 
        B_Bar = ((Lambda_bar - 1.0) / Lambda).unsqueeze(-1) * B
        return Lambda_bar, B_Bar

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        Lambda_bar, B_bar = self._discretize()

        x_c = x.to(B_bar.dtype)
        Bx = torch.einsum("qp, btp->btq", B_bar, x_c)

        batch_size, seq_len, _ = Bx.shape
        A_bcast = Lambda_bar.view(1, 1, -1).expand(batch_size, seq_len, -1)

        # h = lambda_bar * h_t-1 + B_bar * x_t
        h = parallel_associative_scan(A_bcast, Bx)

        C = torch.complex(self.C_re, self.C_im)

        # C_bar * h_t
        y= torch.einsum("oq, btq->bto", C, h).real

        if self.D is not None:
            y = y + x * self.D
        return y

class TemporalLayerNorm(nn.Module):

    def __init__(self, num_channels: int, eps: float = 1e-5):

        super().__init__()
        self.weight = nn.Parameter(torch.ones(num_channels))
        self.bias = nn.Parameter(torch.zeros(num_channels))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mean = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, unbiased=False, keepdim=True)
        x_norm = (x- mean) / torch.sqrt(var + self.eps)
        affine_shape = [1] * (x.dim() - 2) + [-1, 1]
        return x_norm * self.weight.view(*affine_shape) + self.bias.view(*affine_shape)

def build_morelet_filterbank(
        num_freqs: int,
        freq_min: float,
        freq_max: float,
        seq_len: int,
        sampling_rate: float,
        omega0: float = 5.0,
        device=None,
        dtype: torch.dtype = torch.float32,
):
    freqs = torch.linspace(freq_min, freq_max, num_freqs, device=device, dtype=dtype)
    t = torch.arange(seq_len, device=device, dtype=dtype) - (seq_len // 2)
    s = omega0 * sampling_rate / (2 * math.pi * freqs)
    t_over_s = t.view(1, -1) / s.view(-1, 1)
    norm = (1.0 / torch.sqrt(s)).view(-1, 1) * (math.pi ** -0.25)
    envelope = torch.exp(-0.5 * t_over_s)
    real = norm * torch.cos(omega0 * t_over_s) * envelope
    imag = norm * torch.sin(omega0 * t_over_s) * envelope

    return real, imag

class WaveletConv(nn.Module):

    def __init__(
            self,
            num_freqs: int,
            sampling_rate: float,
            freq_min: float = 4.0,
            freq_max: float = 40.0,
            omega0: float = 5.0,
            conv_kernel_size: int | None = None,
    ):

        super().__init__()
        self.num_freqs = num_freqs
        self.sampling_rate = sampling_rate
        self.freq_min = freq_min
        self.freq_max = freq_max
        self.omega0 = omega0

        kernel_size = conv_kernel_size or max(3, int(round(sampling_rate / 2)))

        if kernel_size % 2 == 0:
            kernel_size += 1
        self.conv_branch = nn.Conv1d(
            1, num_freqs, kernel_size=kernel_size, padding="same", bias=False
        )

        self.norm_cwt = TemporalLayerNorm(num_freqs)
        self.norm_conv = TemporalLayerNorm(num_freqs)

    def _cwt_branch(self, x: torch.Tensor) -> torch.Tensor:

        batch_size, num_channels, seq_len = x.shape
        real_k, imag_k = build_morelet_filterbank(
            num_freqs=self.num_freqs,
            freq_min=self.freq_min,
            freq_max=self.freq_max,
            seq_len=seq_len,
            sampling_rate=self.sampling_rate,
            omega0=self.omega0,
            device=x.device,
            dtype=x.dtype,
        )

        weight_real = real_k.unsqueeze(1)
        weight_imag = imag_k.unsqueeze(1)

        flat = x.reshape(batch_size * num_channels, 1, seq_len)
        pad = seq_len // 2
        real_out = nn.functional.conv1d(flat, weight_real, padding=pad)[..., :seq_len]
        imag_out = nn.functional.conv1d(flat, weight_imag, padding=pad)[..., :seq_len]
        magnitude = torch.sqrt(real_out**2 + imag_out**2 + 1e-8)

        return magnitude.reshape(batch_size, num_channels, self.num_freqs, seq_len)

    def _conv_branch(self, x: torch.Tensor) -> torch.Tensor:

        batch_size, num_channels, seq_len = x.shape
        flat = x.reshape(batch_size * num_channels, 1, seq_len)
        out = self.conv_branch(flat)

        return out.reshape(batch_size, num_channels, self.num_freqs, seq_len)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:

        batch_size, num_channels, seq_len = x.shape

        cwt_feat = self._cwt_branch(x)
        conv_feat = self._conv_branch(x)

        cwt_flat = cwt_feat.reshape(batch_size * num_channels, self.num_freqs, seq_len)
        conv_flat = conv_feat.reshape(batch_size * num_channels, self.num_freqs, seq_len)
        cwt_norm = self.norm_cwt(cwt_flat).reshape(batch_size, num_channels, self.num_freqs, seq_len)
        conv_norm = self.norm_conv(conv_flat).reshape(batch_size, num_channels, self.num_freqs, seq_len)

        # 1/2 LayerNorm(CTW(x_m)) + 1/2 LayerNorm(Conv1D(x_m))
        return 0.5 * cwt_norm + 0.5 * conv_norm

class SSMFFNBlock(nn.Module):

    def __init__(
            self,
            num_features: int,
            state_dim: int,
            ffn_dim: int | None = None,
            bidirectional: bool = False,
            dropout_rate: float = 0.1,
    ):

        super().__init__()
        ffn_dim = ffn_dim or num_features * 2
        self.bidirectional = bidirectional

        self.norm = TemporalLayerNorm(num_features)
        self.ssm_fwd = S5SSM(state_dim=state_dim, input_dim=num_features, output_dim=num_features)
        if bidirectional:
            self.ssm_bwd = S5SSM(state_dim=state_dim, input_dim=num_features, output_dim=num_features)
            self.mix = nn.Linear(2 * num_features, num_features)

        self.ffn = nn.Sequential(
            nn.Linear(num_features, ffn_dim),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(ffn_dim, num_features),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        x_norm = self.norm(x)
        ssm_in = x_norm.transpose(1, 2)

        y_fwd = self.ssm_fwd(ssm_in)
        if self.bidirectional:
            y_bwd = self.ssm_bwd(ssm_in.flip(dims=[1])).flip(dims=[1])
            ssm_out = self.mix(torch.cat([y_fwd, y_bwd], dim=-1))
        else:
            ssm_out = y_fwd

        ffn_out = self.ffn(ssm_out)
        out = ffn_out + x_norm.transpose(1, 2)

        return out.transpose(1, 2)

class FrequencySSM(nn.Module):

    def __init__(
            self,
            num_channels: int,
            num_freqs: int,
            state_dim: int,
            num_layers: int,
            ffn_dim: int | None = None,
            bidirectional: bool = False,
            dropout_rate: float = 0.1,
    ):
        super().__init__()
        self.num_channels = num_channels
        self.num_freqs = num_freqs
        self.blocks = nn.ModuleList(
            [
                SSMFFNBlock(
                    num_features=num_channels,
                    state_dim=state_dim,
                    ffn_dim=ffn_dim,
                    bidirectional=bidirectional,
                    dropout_rate=dropout_rate,
                )
                for _ in range(num_layers)
            ]
        )

    def forward(self, x_tilde: torch.Tensor) -> torch.Tensor:

        batch_size, num_channels, num_freqs, seq_len = x_tilde.shape

        u = x_tilde.permute(0, 2, 1, 3).reshape(batch_size * num_freqs, num_channels, seq_len)
        for block in self.blocks:
            u = block(u)
        u = u.reshape(batch_size, num_freqs, num_channels, seq_len).permute(0, 2, 1, 3)

        return u

class ChannelSSM(nn.Module):

    def __init__(
            self,
            num_channels: int,
            num_freqs: int,
            state_dim: int,
            num_layers: int,
            ffn_dim: int | None = None,
            bidirectional: bool = False,
            dropout_rate: float = 0.1,
    ):
        super().__init__()
        self.num_channels = num_channels
        self.num_freqs = num_freqs
        self.blocks = nn.ModuleList(
            [
                SSMFFNBlock(
                    num_features=num_freqs,
                    state_dim=state_dim,
                    ffn_dim=ffn_dim,
                    bidirectional=bidirectional,
                    dropout_rate=dropout_rate,
                )
                for _ in range(num_layers)
            ]
        )

    def forward(self, x_tilde: torch.Tensor) -> torch.Tensor:

        batch_size, num_channels, num_freqs, seq_len = x_tilde.shape

        v = x_tilde.reshape(batch_size * num_channels, num_freqs, seq_len)
        for block in self.blocks:
            v = block(v)
        v = v.reshape(batch_size, num_channels, num_freqs, seq_len)

        return v

class CorticalSSM(BaselineEEGModel):

    def __init__(
            self,
            task_configs: dict[str, TaskConfig],
            num_channels: int = 32,
            sampling_rate: float = 2000.0,
            num_freqs: int = 20,
            freq_min: float = 4.0,
            freq_max: float = 150.0,
            morlet_omega0: float = 5.0,
            conv_kernel_size: int | None = None,
            state_dim: int = 32,
            num_layers: int = 2,
            ffn_dim: int | None = None,
            bidirectional: bool = False,
            fusion_dim: int = 128,
            dropout_rate: float = 0.1,
    ):

        super().__init__(
            num_channels=num_channels,
            num_samples=None,
            task_configs=task_configs,
        )
        self.num_freqs = num_freqs

        self.wavelet_conv = WaveletConv(
            num_freqs=num_freqs,
            sampling_rate=sampling_rate,
            freq_min=freq_min,
            freq_max=freq_max,
            omega0=morlet_omega0,
            conv_kernel_size=conv_kernel_size,
        )

        self.frequency_ssm = FrequencySSM(
            num_channels=num_channels,
            num_freqs=num_freqs,
            state_dim=state_dim,
            num_layers=num_layers,
            ffn_dim=ffn_dim,
            bidirectional=bidirectional,
            dropout_rate=dropout_rate,
        )

        self.channel_ssm = ChannelSSM(
            num_channels=num_channels,
            num_freqs=num_freqs,
            state_dim=state_dim,
            num_layers=num_layers,
            ffn_dim=ffn_dim,
            bidirectional=bidirectional,
            dropout_rate=dropout_rate,
        )


        fusion_in_dim = 2 * num_channels * num_freqs
        self.fusion_ffn = nn.Sequential(
            nn.Linear(fusion_in_dim, fusion_dim),
            nn.GELU(),
            nn.Dropout(dropout_rate),
        )

        self.router = self._build_router(fusion_dim)

    def extract_features(self, input_values: torch.Tensor) -> torch.Tensor:

        x = self._normalize_input_shape(input_values)

        x_tilde = self.wavelet_conv(x)

        u = self.frequency_ssm(x_tilde)
        v = self.channel_ssm(x_tilde)

        u_pooled = u.mean(dim=-1).reshape(u.size(0), -1)
        v_pooled = v.mean(dim=-1).reshape(v.size(0), -1)

        fused = torch.cat([u_pooled, v_pooled], dim=-1)
        return self.fusion_ffn(fused)

    def forward(
            self,
            *,
            input_values: torch.Tensor,
            task_index: torch.Tensor,
            **kwargs,
    ) -> Dict[str, torch.Tensor]:

        embed = self.extract_features(input_values)

        batch_size = embed.shape[0]
        n_out = task_index.shape[1]
        x = embed.unsqueeze(1).expand(batch_size, n_out, -1)

        return self._route_readout(x, task_index)