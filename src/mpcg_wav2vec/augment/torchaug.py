"""Batched, device-aware waveform augmentation.

Applies the tensor-friendly augmentations from the paper to a whole batch ``[B, T]`` at once on
the GPU: additive white noise, sinusoidal volume modulation, baseline wander, cubic-spline
amplitude warp (as a grouped 1-D convolution) and random parametric EQ (causal Butterworth
sections). Each augmentation is applied to the batch with an independent per-sample Bernoulli
mask, so unaugmented samples pass through untouched.

Time-stretch (rubberband) and HPSS have no exact tensor form and stay on the NumPy path
(:mod:`mpcg_wav2vec.augment.pipelines`); use this module for the fast on-device subset.
"""

from __future__ import annotations

import numpy as np
import torch
from scipy import signal as sp

from .pipelines import AugmentConfig

_NOISE_STDS = (0.0001, 0.001, 0.01)


def _normalise(x: torch.Tensor) -> torch.Tensor:
    x = x - x.mean(dim=-1, keepdim=True)
    peak = x.abs().amax(dim=-1, keepdim=True).clamp_min(1e-12)
    return (x / peak).clamp(-1.0, 1.0)


def _mask(batch: int, prob: float, device) -> torch.Tensor:
    return (torch.rand(batch, 1, device=device) < prob).float()


def _apply(x: torch.Tensor, transformed: torch.Tensor, prob: float) -> torch.Tensor:
    m = _mask(x.shape[0], prob, x.device)
    return _normalise(m * transformed + (1.0 - m) * x)


def add_white_noise(x: torch.Tensor) -> torch.Tensor:
    std = float(np.random.choice(_NOISE_STDS))
    scale = torch.rand(x.shape[0], 1, device=x.device) * 0.1
    return x + scale * std * torch.randn_like(x)


def sinusoidal_envelope(x: torch.Tensor, fs: int) -> torch.Tensor:
    b = x.shape[0]
    t = torch.arange(x.shape[-1], device=x.device) / fs
    mod = torch.zeros(b, x.shape[-1], device=x.device)
    for lo, hi in ((0.05, 0.5), (0.001, 0.05)):
        amp = (0.01 + torch.rand(b, 1, device=x.device) * 0.24)
        freq = (lo + torch.rand(b, 1, device=x.device) * (hi - lo))
        phase = torch.rand(b, 1, device=x.device)
        mod = mod + amp * torch.sin(2 * np.pi * (freq * t + phase))
    return x * (1.0 + mod)


def baseline_wander(x: torch.Tensor, fs: int) -> torch.Tensor:
    b = x.shape[0]
    t = torch.arange(x.shape[-1], device=x.device) / fs
    drift = torch.zeros(b, x.shape[-1], device=x.device)
    for lo, hi in ((0.05, 0.5), (0.001, 0.05)):
        amp = (0.01 + torch.rand(b, 1, device=x.device) * 0.19)
        freq = (lo + torch.rand(b, 1, device=x.device) * (hi - lo))
        phase = torch.rand(b, 1, device=x.device)
        drift = drift + amp * torch.sin(2 * np.pi * (freq * t + phase))
    return x + drift


def amplitude_warp(x: torch.Tensor, num_points: int = 12, kernel: int = 65) -> torch.Tensor:
    """Per-sample smooth gain curve applied as a depthwise 1-D convolution."""
    b, t = x.shape
    control = torch.linspace(0, kernel - 1, num_points)
    amps = 0.7 + torch.rand(b, num_points) * 0.6
    grid = torch.arange(kernel).float()
    # linear interpolation of the control points into a length-`kernel` curve
    idx = torch.clamp((grid / (kernel - 1) * (num_points - 1)), max=num_points - 1)
    lo = idx.floor().long()
    hi = idx.ceil().long()
    frac = (idx - lo).unsqueeze(0)
    curve = amps[:, lo] + (amps[:, hi] - amps[:, lo]) * frac      # [B, kernel]
    curve = curve / curve.sum(dim=-1, keepdim=True)
    curve = curve.to(x.device).unsqueeze(1)                        # [B, 1, K]
    padded = torch.nn.functional.pad(x.unsqueeze(1), (kernel // 2, kernel // 2), mode="reflect")
    out = torch.nn.functional.conv1d(padded.reshape(1, b, -1), curve, groups=b)
    return out.reshape(b, -1)[:, :t]


def parametric_eq(x: torch.Tensor, fs: float, low: float, high: float, num_bands: int = 5) -> torch.Tensor:
    """Blend with a stack of random narrow band-pass sections (applied per batch, shared bands)."""
    import torchaudio.functional as AF
    nyq = fs / 2.0
    coloured = x
    for _ in range(num_bands):
        b_low = float(np.random.uniform(low, 0.95 * high))
        b_high = float(np.random.uniform(b_low + 0.05 * (high - low), high))
        b, a = sp.butter(1, [b_low / nyq, b_high / nyq], btype="band")
        coloured = AF.lfilter(coloured, torch.tensor(a, device=x.device, dtype=x.dtype),
                              torch.tensor(b, device=x.device, dtype=x.dtype),
                              clamp=False, batching=True)
    return _normalise(_normalise(coloured) / 50.0 + _normalise(x))


def augment_pcg_batch(x: torch.Tensor, fs: int, cfg: AugmentConfig | None = None) -> torch.Tensor:
    """Apply the on-device PCG augmentation subset to a batch ``[B, T]``."""
    cfg = cfg or AugmentConfig()
    x = _normalise(x)
    x = _apply(x, add_white_noise(x), cfg.prob_noise / 4)
    x = _apply(x, sinusoidal_envelope(x, fs), cfg.prob_wandering_volume)
    x = _apply(x, parametric_eq(x, fs, 2, 500), cfg.prob_banding)
    x = _apply(x, add_white_noise(x), cfg.prob_noise / 4)
    return x
