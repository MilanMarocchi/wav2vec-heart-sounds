"""Batched, device-aware tensor implementations of the preprocessing chain.

These mirror the NumPy functions in this package but operate on ``[B, T]`` (or ``[B, C, T]``)
tensors so a whole batch is filtered, normalised, despiked and segmented in one shot on the GPU.
Filter coefficients are designed once with SciPy and applied causally with ``torchaudio``'s
``lfilter``, matching the paper's preprocessing (Abbott et al. 2025) and the NumPy path within
floating-point tolerance while running far faster on batches.

Resampling and segmentation are also provided. Time-stretch and HPSS have no exact tensor
equivalent and stay on the NumPy augmentation path.
"""

from __future__ import annotations

import torch
import torchaudio.functional as AF
from scipy import signal as sp

from .segment import WindowSpec

# Band edges shared with the NumPy chain (cutoffs normalised by the sampling rate).
PCG_BAND = (25.0, 450.0)
ECG_BAND = (2.0, 40.0)


def _to_batched(x: torch.Tensor) -> tuple[torch.Tensor, bool]:
    if x.dim() == 1:
        return x.unsqueeze(0), True
    return x, False


def _butter_ba(cutoff: float, fs: float, btype: str, order, device, dtype):
    b, a = sp.butter(order, cutoff / fs, btype=btype)  # cutoff normalised by fs (paper convention)
    return (torch.tensor(b, device=device, dtype=dtype),
            torch.tensor(a, device=device, dtype=dtype))


def _causal(x: torch.Tensor, b: torch.Tensor, a: torch.Tensor) -> torch.Tensor:
    return AF.lfilter(x, a, b, clamp=False, batching=True)


def lowpass(x: torch.Tensor, fs: float, cutoff: float, order: int = 2) -> torch.Tensor:
    b, a = _butter_ba(cutoff, fs, "lowpass", order, x.device, x.dtype)
    return _causal(x, b, a)


def highpass(x: torch.Tensor, fs: float, cutoff: float, order: int = 2) -> torch.Tensor:
    b, a = _butter_ba(cutoff, fs, "highpass", order, x.device, x.dtype)
    return _causal(x, b, a)


def bandpass_cascade(x: torch.Tensor, fs: float, low: float, high: float, order: int = 2) -> torch.Tensor:
    return highpass(lowpass(x, fs, high, order=order), fs, low, order=order)


def resample(x: torch.Tensor, fs_in: float, fs_out: float) -> torch.Tensor:
    if fs_in == fs_out:
        return x
    return AF.resample(x, int(round(fs_in)), int(round(fs_out)))


def abs_max_normalise(x: torch.Tensor) -> torch.Tensor:
    x = torch.nan_to_num(x)
    x = x - x.mean(dim=-1, keepdim=True)
    peak = x.abs().amax(dim=-1, keepdim=True).clamp_min(1e-12)
    return (x / peak).clamp(-1.0, 1.0)


def remove_spikes(x: torch.Tensor, fs: float, threshold: float = 3.0, max_iterations: int = 1000) -> torch.Tensor:
    """Batched Schmidt spike removal over ``[B, T]`` (500 ms windows)."""
    x, squeezed = _to_batched(x)
    x = x.clone()
    B, T = x.shape
    win = round(float(fs) / 2.0)
    if win < 1 or T < win:
        return x.squeeze(0) if squeezed else x

    nfull = T - (T % win)
    frames = x[:, :nfull].reshape(B, -1, win).clone()  # [B, W, win]
    for _ in range(max_iterations):
        maa = frames.abs().amax(dim=2)                 # [B, W]
        median = maa.median(dim=1, keepdim=True).values
        active = (maa > threshold * median).any(dim=1)
        if not bool(active.any()):
            break
        worst = maa.argmax(dim=1)
        for bi in torch.nonzero(active, as_tuple=False).flatten().tolist():
            window = frames[bi, worst[bi]]
            peak = int(window.abs().argmax())
            signs = torch.sign(window)
            crossings = torch.nonzero((signs[1:] - signs[:-1]).abs() > 1, as_tuple=False).flatten()
            before = crossings[crossings < peak]
            after = crossings[crossings >= peak]
            start = int(before[-1] + 1) if before.numel() else 0
            end = int(after[0]) if after.numel() else win - 1
            frames[bi, worst[bi], start:end] = 1e-4
    x[:, :nfull] = frames.reshape(B, nfull)
    return x.squeeze(0) if squeezed else x


def preprocess_pcg(x: torch.Tensor, fs_in: float, fs_out: float, *, despike: bool = True) -> torch.Tensor:
    x, squeezed = _to_batched(x)
    x = resample(x, fs_in, fs_out)
    if despike:
        x = remove_spikes(x, fs_out)
    x = bandpass_cascade(x, fs_out, *PCG_BAND, order=2)
    x = abs_max_normalise(x)
    return x.squeeze(0) if squeezed else x


def preprocess_ecg(x: torch.Tensor, fs_in: float, fs_out: float) -> torch.Tensor:
    x, squeezed = _to_batched(x)
    x = resample(x, fs_in, fs_out)
    x = bandpass_cascade(x, fs_out, *ECG_BAND, order=2)
    x = abs_max_normalise(x)
    return x.squeeze(0) if squeezed else x


def segment(x: torch.Tensor, fs: float, spec: WindowSpec) -> torch.Tensor:
    """Split ``[B, T]`` into overlapping windows ``[B, N, win]`` via ``unfold``."""
    x, squeezed = _to_batched(x)
    win = spec.window_len(fs)
    hop = spec.hop_len(fs)
    start = int(round(spec.start_pad_s * fs))
    x = x[:, start:]
    if x.shape[-1] < win:
        x = torch.nn.functional.pad(x, (0, win - x.shape[-1]))
    windows = x.unfold(dimension=-1, size=win, step=hop)
    return windows.squeeze(0) if squeezed else windows
