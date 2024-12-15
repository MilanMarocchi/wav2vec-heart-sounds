"""DiffWave: a class- and mel-conditioned discrete-step DDPM vocoder for heart sounds.

Architecture (unchanged in spirit from Kong et al. 2020, re-expressed here):

    waveform -> 1x1 in-projection -> stack of dilated gated residual blocks -> skip sum
             -> 1x1 -> 1x1 out-projection -> predicted noise

Each residual block is conditioned on the diffusion-step embedding, an upsampled mel
conditioner, and a class-label embedding. Config lives in :class:`DiffWaveConfig`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from math import sqrt

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .schedules import DiffusionStepEmbedding, NoiseSchedule


@dataclass
class DiffWaveConfig:
    sample_rate: int = 4000
    n_mels: int = 80
    n_fft: int = 1024
    hop_length: int = 256
    residual_layers: int = 30
    residual_channels: int = 64
    dilation_cycle: int = 10
    step_hidden: int = 512
    num_classes: int = 2
    label_dim: int = 32
    train_beta: tuple[float, float, int] = (1e-4, 0.05, 50)
    inference_betas: tuple = (0.0001, 0.001, 0.01, 0.05, 0.2, 0.5)

    def training_schedule(self) -> NoiseSchedule:
        return NoiseSchedule.linear(*self.train_beta)

    def upsample_factors(self) -> tuple[int, int]:
        """Two ConvTranspose strides whose product upsamples mel frames to samples (= hop)."""
        hop = self.hop_length
        for a in range(int(sqrt(hop)), 0, -1):
            if hop % a == 0:
                return a, hop // a
        return 1, hop


def _conv1d(*args, **kwargs) -> nn.Conv1d:
    layer = nn.Conv1d(*args, **kwargs)
    nn.init.kaiming_normal_(layer.weight)
    return layer


def _match_time(x: torch.Tensor, length: int) -> torch.Tensor:
    """Crop or edge-pad a ``[B, C, T]`` tensor to exactly ``length`` samples along time."""
    if x.shape[-1] > length:
        return x[..., :length]
    if x.shape[-1] < length:
        return F.pad(x, (0, length - x.shape[-1]))
    return x


class MelUpsampler(nn.Module):
    """Upsample a mel-spectrogram along time to waveform resolution via two transposed convs."""

    def __init__(self, factors: tuple[int, int]):
        super().__init__()
        self.blocks = nn.ModuleList()
        for f in factors:
            pad = f // 2
            self.blocks.append(nn.ConvTranspose2d(1, 1, (3, 2 * f), stride=(1, f), padding=(1, pad)))

    def forward(self, mel: torch.Tensor) -> torch.Tensor:
        x = mel.unsqueeze(1)
        for block in self.blocks:
            x = F.leaky_relu(block(x), 0.4)
        return x.squeeze(1)


class ResidualBlock(nn.Module):
    def __init__(self, n_mels: int, channels: int, dilation: int, step_hidden: int, label_dim: int):
        super().__init__()
        self.dilated = _conv1d(channels, 2 * channels, 3, padding=dilation, dilation=dilation)
        self.step_proj = nn.Linear(step_hidden, channels)
        self.cond_proj = _conv1d(n_mels, 2 * channels, 1)
        self.label_proj = _conv1d(label_dim, 2 * channels, 1)
        self.out_proj = _conv1d(channels, 2 * channels, 1)

    def forward(self, x, step_embed, conditioner, label_embed):
        y = x + self.step_proj(step_embed).unsqueeze(-1)
        y = self.dilated(y) + self.cond_proj(conditioner) + self.label_proj(label_embed)
        gate, filt = y.chunk(2, dim=1)
        y = torch.sigmoid(gate) * torch.tanh(filt)
        residual, skip = self.out_proj(y).chunk(2, dim=1)
        return (x + residual) / sqrt(2.0), skip


class DiffWave(nn.Module):
    def __init__(self, config: DiffWaveConfig):
        super().__init__()
        self.config = config
        c = config.residual_channels

        self.input_projection = _conv1d(1, c, 1)
        self.step_embedding = DiffusionStepEmbedding(len(config.training_schedule()),
                                                     hidden=config.step_hidden)
        self.mel_upsampler = MelUpsampler(config.upsample_factors())
        self.label_embedding = nn.Embedding(config.num_classes, config.label_dim)

        self.residual_blocks = nn.ModuleList(
            ResidualBlock(config.n_mels, c, 2 ** (i % config.dilation_cycle),
                          config.step_hidden, config.label_dim)
            for i in range(config.residual_layers)
        )
        self.skip_projection = _conv1d(c, c, 1)
        self.output_projection = _conv1d(c, 1, 1)
        nn.init.zeros_(self.output_projection.weight)

    def forward(self, audio, step, conditioner, label) -> torch.Tensor:
        x = F.relu(self.input_projection(audio.unsqueeze(1)))
        step_embed = self.step_embedding(step)
        conditioner = self.mel_upsampler(conditioner)
        conditioner = _match_time(conditioner, x.shape[-1])
        label_embed = self.label_embedding(label).squeeze(1).unsqueeze(-1)

        skip = 0.0
        for block in self.residual_blocks:
            x, s = block(x, step_embed, conditioner, label_embed)
            skip = skip + s
        x = skip / sqrt(len(self.residual_blocks))
        x = F.relu(self.skip_projection(x))
        return self.output_projection(x)
