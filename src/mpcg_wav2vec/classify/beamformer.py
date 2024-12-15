"""Time-varying sinc delay-and-sum beamformer for collapsing multichannel PCG to one channel.

A tiny transformer predicts a per-sample fractional delay for each microphone; each channel is
then fractionally delayed by dynamic sinc interpolation, squared, and summed across channels
(sum-of-squares delay-and-sum). This is the only channel mixer the paper's vest runs use.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class _DelayPredictor(nn.Module):
    def __init__(self, num_mics: int, d_model: int = 32, nhead: int = 4, num_layers: int = 2):
        super().__init__()
        self.input_proj = nn.Conv1d(num_mics, d_model, kernel_size=1)
        layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=64,
                                           batch_first=True)
        self.encoder = nn.TransformerEncoder(layer, num_layers=num_layers)
        self.output_proj = nn.Linear(d_model, num_mics)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # (B, M, T) -> (B, M, T)
        h = self.input_proj(x).transpose(1, 2)
        h = self.encoder(h)
        return self.output_proj(h).transpose(1, 2)


class TimeVaryingSincBeamformer(nn.Module):
    def __init__(self, num_mics: int, fs: float, max_delay_s: float = 0.01, kernel_size: int = 41):
        super().__init__()
        self.num_mics = num_mics
        self.max_delay_samples = max_delay_s * fs
        self.kernel_size = kernel_size
        self.half_k = kernel_size // 2
        self.delay_predictor = _DelayPredictor(num_mics)
        self.register_buffer("t_idx", torch.arange(-self.half_k, self.half_k + 1).float())
        self.register_buffer("window", torch.hamming_window(kernel_size, periodic=False))

    def _delay_channel(self, x: torch.Tensor, delays: torch.Tensor) -> torch.Tensor:
        """Fractionally delay ``x`` (B, T) by per-sample ``delays`` (B, T) via sinc interpolation."""
        b, t = x.shape
        k = self.kernel_size
        kernel = torch.sinc(self.t_idx.view(1, 1, k) - delays.unsqueeze(-1)) * self.window.view(1, 1, k)
        kernel = kernel / kernel.sum(dim=-1, keepdim=True)

        padded = F.pad(x.unsqueeze(1), (self.half_k, self.half_k), mode="reflect").unsqueeze(2)
        unfolded = F.unfold(padded, kernel_size=(1, k)).transpose(1, 2)  # (B, T, K)
        return torch.einsum("btk,btk->bt", unfolded, kernel)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # (B, M, T) -> (B, T)
        delays = torch.clamp(self.delay_predictor(x), 0.0, self.max_delay_samples)
        aligned = [self._delay_channel(x[:, m, :], delays[:, m, :]) ** 2 for m in range(x.shape[1])]
        return torch.stack(aligned, dim=1).sum(dim=1)
