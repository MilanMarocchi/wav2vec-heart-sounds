"""WaveGrad: a class- and mel-conditioned continuous-noise-level diffusion vocoder.

A U-net over the waveform: down-sampling ``DBlock``s extract features that produce FiLM
(shift, scale) modulations, and up-sampling ``UBlock``s decode from the mel-spectrogram while
being modulated by those FiLM outputs (Chen et al. 2020). Config in :class:`WaveGradConfig`.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from .schedules import NoiseLevelEncoding, NoiseSchedule


@dataclass
class WaveGradConfig:
    sample_rate: int = 4000
    n_mels: int = 128
    hop_length: int = 300
    num_classes: int = 2
    label_dim: int = 32
    train_beta: tuple[float, float, int] = (1e-6, 0.01, 1000)

    def training_schedule(self) -> NoiseSchedule:
        return NoiseSchedule.linear(*self.train_beta)


class _Conv1d(nn.Conv1d):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        nn.init.orthogonal_(self.weight)
        if self.bias is not None:
            nn.init.zeros_(self.bias)


class FiLM(nn.Module):
    """Produce (shift, scale) modulations from features, a noise level and a class label."""

    def __init__(self, in_ch: int, out_ch: int, num_classes: int, label_dim: int):
        super().__init__()
        self.label_embedding = nn.Embedding(num_classes, label_dim)
        self.label_proj = nn.Conv1d(label_dim, in_ch, 3, padding=1)
        self.encoding = NoiseLevelEncoding(in_ch)
        self.input_conv = nn.Conv1d(in_ch, in_ch, 3, padding=1)
        self.output_conv = nn.Conv1d(in_ch, out_ch * 2, 3, padding=1)
        for conv in (self.input_conv, self.output_conv):
            nn.init.xavier_uniform_(conv.weight)
            nn.init.zeros_(conv.bias)

    def forward(self, x, noise_level, label):
        label_embed = self.label_proj(self.label_embedding(label).squeeze(1).unsqueeze(-1))
        x = self.input_conv(x + label_embed)
        x = self.encoding(F.leaky_relu(x, 0.2), noise_level)
        return self.output_conv(x).chunk(2, dim=1)


class DBlock(nn.Module):
    """Down-sampling residual block."""

    def __init__(self, in_ch: int, out_ch: int, factor: int):
        super().__init__()
        self.factor = factor
        self.residual = _Conv1d(in_ch, out_ch, 1)
        self.convs = nn.ModuleList([
            _Conv1d(in_ch, out_ch, 3, dilation=1, padding=1),
            _Conv1d(out_ch, out_ch, 3, dilation=2, padding=2),
            _Conv1d(out_ch, out_ch, 3, dilation=4, padding=4),
        ])

    def forward(self, x):
        size = x.shape[-1] // self.factor
        residual = F.interpolate(self.residual(x), size=size)
        x = F.interpolate(x, size=size)
        for conv in self.convs:
            x = conv(F.leaky_relu(x, 0.2))
        return x + residual


class UBlock(nn.Module):
    """Up-sampling residual block modulated by a FiLM (shift, scale)."""

    def __init__(self, in_ch: int, out_ch: int, factor: int, dilations):
        super().__init__()
        self.factor = factor
        self.skip = _Conv1d(in_ch, out_ch, 1)
        self.conv_a = nn.ModuleList([
            _Conv1d(in_ch, out_ch, 3, dilation=dilations[0], padding=dilations[0]),
            _Conv1d(out_ch, out_ch, 3, dilation=dilations[1], padding=dilations[1]),
        ])
        self.conv_b = nn.ModuleList([
            _Conv1d(out_ch, out_ch, 3, dilation=dilations[2], padding=dilations[2]),
            _Conv1d(out_ch, out_ch, 3, dilation=dilations[3], padding=dilations[3]),
        ])

    def forward(self, x, shift, scale):
        size = x.shape[-1] * self.factor
        skip = self.skip(F.interpolate(x, size=size))

        h = self.conv_a[0](F.interpolate(F.leaky_relu(x, 0.2), size=size))
        h = self.conv_a[1](F.leaky_relu(shift + scale * h, 0.2))
        x = skip + h

        h = self.conv_b[0](F.leaky_relu(shift + scale * x, 0.2))
        h = self.conv_b[1](F.leaky_relu(shift + scale * h, 0.2))
        return x + h


class WaveGrad(nn.Module):
    def __init__(self, config: WaveGradConfig):
        super().__init__()
        self.config = config
        nc, ld = config.num_classes, config.label_dim

        self.first_conv = _Conv1d(config.n_mels, 768, 3, padding=1)
        self.downsample = nn.ModuleList([
            _Conv1d(1, 32, 5, padding=2),
            DBlock(32, 128, 2),
            DBlock(128, 128, 2),
            DBlock(128, 256, 3),
            DBlock(256, 512, 5),
        ])
        self.film = nn.ModuleList([
            FiLM(32, 128, nc, ld),
            FiLM(128, 128, nc, ld),
            FiLM(128, 256, nc, ld),
            FiLM(256, 512, nc, ld),
            FiLM(512, 512, nc, ld),
        ])
        self.upsample = nn.ModuleList([
            UBlock(768, 512, 5, [1, 2, 1, 2]),
            UBlock(512, 512, 5, [1, 2, 1, 2]),
            UBlock(512, 256, 3, [1, 2, 4, 8]),
            UBlock(256, 128, 2, [1, 2, 4, 8]),
            UBlock(128, 128, 2, [1, 2, 4, 8]),
        ])
        self.last_conv = _Conv1d(128, 1, 3, padding=1)

    def forward(self, audio, conditioner, noise_level, label) -> torch.Tensor:
        x = audio.unsqueeze(1)
        modulations = []
        for film, block in zip(self.film, self.downsample):
            x = block(x)
            modulations.append(film(x, noise_level, label))

        # Keep exactly audio_len / hop mel frames so the upsample path matches the audio length.
        frames = audio.shape[-1] // self.config.hop_length
        conditioner = conditioner[..., :frames]
        x = self.first_conv(conditioner)
        for block, (shift, scale) in zip(self.upsample, reversed(modulations)):
            x = block(x, shift, scale)
        return self.last_conv(x)
