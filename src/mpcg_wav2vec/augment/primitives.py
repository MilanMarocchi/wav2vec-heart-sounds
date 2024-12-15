"""Individual waveform augmentation operations.

These reproduce the paper's conventional augmentations (HPSS recombination, additive white
noise, cubic-spline amplitude warp, time-stretch, random parametric EQ, baseline wander,
sinusoidal volume modulation), factored into small pure functions. All operations return an
abs-max-normalised signal so they compose cleanly.
"""

from __future__ import annotations

import random

import librosa
import numpy as np
import pyrubberband as pyrb
from scipy import signal as sp
from scipy.interpolate import CubicSpline

from ..signalproc.normalize import abs_max_normalise, minmax_normalise

_NOISE_STDS = (0.0001, 0.001, 0.01)


def randfloat(lo: float, hi: float) -> float:
    return lo + random.random() * (hi - lo)


# --- time / amplitude ------------------------------------------------------

def time_stretch(x: np.ndarray, fs: int, rate: float, keep_length: bool = False) -> np.ndarray:
    y = pyrb.time_stretch(x, fs, rate=rate)
    if keep_length:
        y = y[: len(x)]
    return y


def random_crop(x: np.ndarray, length: int) -> np.ndarray:
    if len(x) <= length:
        return x
    start = random.randint(0, len(x) - length)
    return x[start:start + length]


def add_white_noise(x: np.ndarray) -> np.ndarray:
    std = random.choice(_NOISE_STDS)
    return abs_max_normalise(x + randfloat(0.0, 0.1) * np.random.normal(0.0, std, x.shape))


def amplitude_warp(x: np.ndarray, num_points: int = 12, amp_range=(0.7, 1.3)) -> np.ndarray:
    """Convolve with a smooth, unit-sum cubic-spline gain curve."""
    n = len(x)
    control = np.linspace(0, n - 1, num_points)
    amps = np.random.uniform(amp_range[0], amp_range[1], size=num_points)
    curve = CubicSpline(control, amps, bc_type="natural")(np.arange(n))
    curve = curve / np.sum(curve)
    return np.convolve(x, curve, mode="same")


def sinusoidal_envelope(x: np.ndarray, fs: int, a_lo: float = 0.01, a_hi: float = 0.25) -> np.ndarray:
    t = np.arange(x.size) / fs
    fast = randfloat(a_lo, a_hi) * np.sin(2 * np.pi * (randfloat(0.05, 0.5) * t + randfloat(0, 1)))
    slow = randfloat(a_lo, a_hi) * np.sin(2 * np.pi * (randfloat(0.001, 0.05) * t + randfloat(0, 1)))
    return abs_max_normalise(x * (1.0 + fast + slow))


def baseline_wander(x: np.ndarray, fs: int) -> np.ndarray:
    t = np.arange(x.size) / fs
    drift = randfloat(0.01, 0.2) * np.sin(2 * np.pi * (randfloat(0.05, 0.5) * t + randfloat(0, 1)))
    drift += randfloat(0.01, 0.2) * np.sin(2 * np.pi * (randfloat(0.001, 0.05) * t + randfloat(0, 1)))
    return abs_max_normalise(x + drift)


def parametric_eq(x: np.ndarray, fs: float, low: float, high: float, num_bands: int = 5) -> np.ndarray:
    """Blend the signal with a stack of random narrow band-pass sections (subtle colouring)."""
    nyq = fs / 2.0
    coloured = np.asarray(x, dtype=np.float64)
    for _ in range(num_bands):
        b_low = np.random.uniform(low, 0.95 * high)
        b_high = random.choice([np.random.uniform(b_low + 0.05 * (high - low), high),
                                b_low + (high - low) / num_bands])
        sos = sp.iirfilter(1, [b_low / nyq, b_high / nyq], btype="band", ftype="butter", output="sos")
        coloured = sp.sosfilt(sos, coloured)
    return abs_max_normalise(abs_max_normalise(coloured) / 50.0 + abs_max_normalise(x))


# --- harmonic/percussive source separation ---------------------------------

def _hpss_split(y: np.ndarray, n_fft: int, hop: int, margin, kernel):
    spec = librosa.stft(y, n_fft=n_fft, hop_length=hop, win_length=n_fft)
    harm, perc = librosa.decompose.hpss(spec, margin=margin, kernel_size=kernel)
    resid = spec - (harm + perc)
    inv = lambda s: librosa.istft(s, n_fft=n_fft, hop_length=hop, win_length=n_fft)
    return inv(harm), inv(perc), inv(resid)


def hpss_recombine(x: np.ndarray, include_residual: bool = True) -> tuple[np.ndarray, int]:
    """Two-stage HPSS decomposition, randomly re-weighting the components back together.

    ``include_residual=True`` keeps the spectral residual of each stage (7 components), matching
    the synchronised PCG+ECG augmentation. ``include_residual=False`` keeps only the harmonic and
    percussive components of the second stage (4 components), matching the single-channel PCG
    augmentation.
    """
    n_fft1 = random.choice([512, 1024, 2048])
    hop1 = random.choice([16, 32, 64, 128])
    n_fft2 = random.choice([512, 1024, 2048])
    hop2 = random.choice([16, 32, 64, 128])
    margin1 = (randfloat(1.0, 2.0), randfloat(1.0, 2.0))
    margin2 = (randfloat(1.0, 4.0), randfloat(1.0, 4.0))
    kernel1 = (random.randint(5, 30), random.randint(5, 30))
    kernel2 = (random.randint(5, 30), random.randint(5, 30))

    harm, perc, resid = _hpss_split(x, n_fft1, hop1, margin1, kernel1)
    h1, p1, r1 = _hpss_split(harm, n_fft2, hop2, margin2, kernel2)
    h2, p2, r2 = _hpss_split(perc, n_fft2, hop2, margin2, kernel2)

    parts = [h1, p1, r1, h2, p2, r2, resid] if include_residual else [h1, p1, h2, p2]
    n = min(len(p) for p in parts)
    parts = [p[:n] for p in parts]

    mix1 = abs_max_normalise(sum(randfloat(0.01, 10) * p for p in parts))
    mix2 = abs_max_normalise(sum(randfloat(0.01, 10) * abs_max_normalise(p) for p in parts))
    return abs_max_normalise(mix1 + randfloat(0.01, 0.05) * mix2), n
