"""Shared diffusion noise schedules and conditioning embeddings.

Both generators are epsilon-prediction denoisers. The maths common to them lives here so the
model files hold only architecture:

* :class:`NoiseSchedule` — a discrete beta schedule with cached ``alpha``/``alpha_cum`` terms
  and the continuous ``sqrt(alpha_cum)`` noise-levels used for WaveGrad-style training.
* :func:`step_embedding` — sinusoidal embedding of an integer diffusion step (DiffWave).
* :class:`NoiseLevelEncoding` — continuous-noise-level positional encoding (WaveGrad).
"""

from __future__ import annotations

from dataclasses import dataclass
from math import log

import numpy as np
import torch
import torch.nn as nn


@dataclass(frozen=True)
class NoiseSchedule:
    betas: np.ndarray

    @classmethod
    def linear(cls, start: float, end: float, steps: int) -> "NoiseSchedule":
        return cls(np.linspace(start, end, steps, dtype=np.float64))

    def __len__(self) -> int:
        return len(self.betas)

    @property
    def alphas(self) -> np.ndarray:
        return 1.0 - self.betas

    @property
    def alpha_cumprod(self) -> np.ndarray:
        return np.cumprod(self.alphas)

    def training_noise_levels(self) -> np.ndarray:
        """cumprod(1-beta) — variance retained by the signal at each discrete step (DiffWave)."""
        return self.alpha_cumprod

    def continuous_noise_levels(self) -> np.ndarray:
        """sqrt(cumprod(1-beta)) prefixed with 1.0 (WaveGrad continuous level lookup)."""
        return np.concatenate([[1.0], np.sqrt(self.alpha_cumprod)])


def step_embedding(steps: torch.Tensor, dim: int = 128, max_freq_exp: float = 4.0) -> torch.Tensor:
    """Sinusoidal embedding of (possibly fractional) diffusion steps -> ``[N, dim]``."""
    half = dim // 2
    freqs = 10.0 ** (torch.arange(half, device=steps.device, dtype=torch.float32) * max_freq_exp / (half - 1))
    args = steps.float().unsqueeze(-1) * freqs.unsqueeze(0)
    return torch.cat([torch.sin(args), torch.cos(args)], dim=-1)


class DiffusionStepEmbedding(nn.Module):
    """Cached step table + MLP, supporting fractional-step interpolation for fast sampling."""

    def __init__(self, num_steps: int, dim: int = 128, hidden: int = 512):
        super().__init__()
        table = step_embedding(torch.arange(num_steps), dim=dim)
        self.register_buffer("table", table, persistent=False)
        self.proj1 = nn.Linear(dim, hidden)
        self.proj2 = nn.Linear(hidden, hidden)

    def forward(self, step: torch.Tensor) -> torch.Tensor:
        if step.dtype in (torch.int32, torch.int64):
            x = self.table[step]
        else:
            lo = step.floor().long()
            hi = step.ceil().long()
            frac = (step - lo).unsqueeze(-1)
            x = self.table[lo] + (self.table[hi] - self.table[lo]) * frac
        x = torch.nn.functional.silu(self.proj1(x))
        return torch.nn.functional.silu(self.proj2(x))


class NoiseLevelEncoding(nn.Module):
    """Add a Gaussian-Fourier encoding of a continuous noise level to a feature map ``[B, C, T]``."""

    def __init__(self, channels: int):
        super().__init__()
        self.channels = channels

    def forward(self, x: torch.Tensor, noise_level: torch.Tensor) -> torch.Tensor:
        half = self.channels // 2
        steps = torch.arange(half, device=x.device, dtype=x.dtype) / half
        enc = noise_level.unsqueeze(1) * torch.exp(-log(1e4) * steps.unsqueeze(0))
        enc = torch.cat([torch.sin(enc), torch.cos(enc)], dim=-1)
        return x + enc.unsqueeze(-1)
