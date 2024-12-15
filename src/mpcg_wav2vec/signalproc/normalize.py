"""Amplitude normalisers: abs-max, min-max, z-score and k-peak."""

from __future__ import annotations

import numpy as np
import torch

_EPS = 1e-8


def interpolate_nans(x: np.ndarray) -> np.ndarray:
    """Linearly interpolate over any NaN samples in place-safe fashion."""
    x = np.asarray(x, dtype=np.float64).copy()
    mask = np.isnan(x)
    if mask.any() and (~mask).any():
        x[mask] = np.interp(np.flatnonzero(mask), np.flatnonzero(~mask), x[~mask])
    return x


def abs_max_normalise(x: np.ndarray) -> np.ndarray:
    """Zero-mean, divide by peak absolute value, clip to [-1, 1].

    This is the project's canonical ``normalise_signal`` / ``standardise_signal``.
    """
    x = interpolate_nans(x)
    x = x - np.mean(x)
    peak = np.max(np.abs(x))
    if peak > 0:
        x = x / peak
    return np.clip(x, -1.0, 1.0)


def minmax_normalise(x: np.ndarray, lo: float = -1.0, hi: float = 1.0) -> np.ndarray:
    """True min-max rescale into [lo, hi]."""
    x = np.asarray(x, dtype=np.float64)
    span = x.max() - x.min()
    if span <= 0:
        return np.full_like(x, (lo + hi) / 2.0)
    return (x - x.min()) / span * (hi - lo) + lo


def minmax_normalise_torch(x: torch.Tensor, lo: float = -1.0, hi: float = 1.0) -> torch.Tensor:
    span = x.max() - x.min()
    return (x - x.min()) / (span + _EPS) * (hi - lo) + lo


def z_normalise(x: np.ndarray, axis: int = 0) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    return (x - x.mean(axis=axis)) / (x.std(axis=axis) + _EPS)


def z_normalise_torch(x: torch.Tensor) -> torch.Tensor:
    """Per-channel z-score over the time dimension of a ``[B, C, T]`` tensor."""
    mean = x.mean(dim=-1, keepdim=True)
    std = x.std(dim=-1, unbiased=False, keepdim=True)
    return (x - mean) / (std + _EPS)


def kpeak_normalise(x: np.ndarray, k: int = 3, lo: float = -1.0, hi: float = 1.0) -> np.ndarray:
    """Rescale using the mean of the ``k`` largest / smallest samples as the range.

    More robust to isolated spikes than plain min-max.
    """
    x = np.asarray(x, dtype=np.float64)
    ordered = np.sort(x)
    lo_ref = ordered[:k].mean()
    hi_ref = ordered[-k:].mean()
    span = hi_ref - lo_ref
    if span <= 0:
        return np.full_like(x, (lo + hi) / 2.0)
    return lo + (x - lo_ref) / span * (hi - lo)


def kpeak_normalise_torch(x: torch.Tensor, k: int = 26, lo: float = -1.0, hi: float = 1.0,
                          dim: int = -1) -> torch.Tensor:
    hi_ref = torch.topk(x, k=k, dim=dim, largest=True).values.mean()
    lo_ref = -torch.topk(-x, k=k, dim=dim, largest=True).values.mean()
    return lo + (x - lo_ref) / (hi_ref - lo_ref + _EPS) * (hi - lo)


def to_torch_normalised(signal: torch.Tensor) -> torch.Tensor:
    """abs-max normalise a torch signal, returning a 1-D float tensor."""
    arr = abs_max_normalise(signal.detach().cpu().numpy())
    return torch.from_numpy(arr).reshape(-1).float()
