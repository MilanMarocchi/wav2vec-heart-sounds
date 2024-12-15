"""Envelope extraction helpers."""

from __future__ import annotations

import numpy as np
from scipy import signal as sp

from .filters import butter_lowpass


def hilbert_envelope(x: np.ndarray) -> np.ndarray:
    """Analytic-signal amplitude envelope."""
    return np.abs(sp.hilbert(np.asarray(x, dtype=np.float64)))


def homomorphic_envelope(x: np.ndarray, fs: float, cutoff: float = 8.0, order: int = 6) -> np.ndarray:
    """Low-pass the log-envelope then exponentiate (classic homomorphic envelogram)."""
    if cutoff >= 0.5 * fs:
        raise ValueError(f"cutoff {cutoff} Hz is above Nyquist for fs={fs}")
    env = hilbert_envelope(x)
    env = np.maximum(env, np.finfo(float).eps)
    smoothed = butter_lowpass(np.log(env), fs, cutoff, order=order)
    return np.exp(smoothed)
