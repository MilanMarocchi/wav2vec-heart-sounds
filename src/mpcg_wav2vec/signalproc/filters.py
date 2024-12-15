"""Butterworth and FIR filters used throughout the pipeline.

The PCG/ECG preprocessing band filters follow the convention used in the paper
(Abbott et al. 2025, arXiv:2410.10125): causal second-order Butterworth low- and high-pass
stages whose cutoff is normalised by the sampling rate. Generic zero-phase helpers are also
provided for envelope extraction and band decomposition.
"""

from __future__ import annotations

import numpy as np
from scipy import signal as sp


def _nyquist(fs: float) -> float:
    return 0.5 * fs


def _as_float(x: np.ndarray) -> np.ndarray:
    return np.asarray(x, dtype=np.float64)


# --- preprocessing band filters (causal, sampling-rate normalised) ---------

def lowpass(x: np.ndarray, fs: float, cutoff: float, order: int = 2) -> np.ndarray:
    """Causal Butterworth low-pass; cutoff normalised by the sampling rate."""
    sos = sp.butter(order, cutoff / fs, btype="lowpass", output="sos")
    return sp.sosfilt(sos, _as_float(x))


def highpass(x: np.ndarray, fs: float, cutoff: float, order: int = 2) -> np.ndarray:
    """Causal Butterworth high-pass; cutoff normalised by the sampling rate."""
    sos = sp.butter(order, cutoff / fs, btype="highpass", output="sos")
    return sp.sosfilt(sos, _as_float(x))


def bandpass_cascade(x: np.ndarray, fs: float, low: float, high: float, order: int = 2) -> np.ndarray:
    """Low-pass at ``high`` then high-pass at ``low`` — the PCG/ECG preprocessing band."""
    return highpass(lowpass(x, fs, high, order=order), fs, low, order=order)


# --- generic zero-phase filters (Nyquist normalised) -----------------------

def butter_bandpass(x: np.ndarray, fs: float, low: float, high: float, order: int = 4) -> np.ndarray:
    """Zero-phase Butterworth band-pass between ``low`` and ``high`` Hz."""
    nyq = _nyquist(fs)
    sos = sp.butter(order, [low / nyq, high / nyq], btype="bandpass", output="sos")
    return sp.sosfiltfilt(sos, _as_float(x))


def butter_lowpass(x: np.ndarray, fs: float, cutoff: float, order: int = 4) -> np.ndarray:
    sos = sp.butter(order, cutoff / _nyquist(fs), btype="lowpass", output="sos")
    return sp.sosfiltfilt(sos, _as_float(x))


def butter_highpass(x: np.ndarray, fs: float, cutoff: float, order: int = 4) -> np.ndarray:
    sos = sp.butter(order, cutoff / _nyquist(fs), btype="highpass", output="sos")
    return sp.sosfiltfilt(sos, _as_float(x))


def notch(x: np.ndarray, fs: float, freq: float, q: float = 30.0) -> np.ndarray:
    """Zero-phase IIR notch at ``freq`` Hz with quality factor ``q``."""
    b, a = sp.iirnotch(freq / _nyquist(fs), q)
    return sp.filtfilt(b, a, _as_float(x))


def notch_chain(x: np.ndarray, fs: float, freqs, q: float = 55.0) -> np.ndarray:
    """Apply several notch filters in sequence (e.g. mains hum + harmonics)."""
    y = _as_float(x)
    for f in freqs:
        if f < _nyquist(fs):
            y = notch(y, fs, f, q)
    return y


def band_stop(x: np.ndarray, fs: float, low: float, high: float, order: int = 4) -> np.ndarray:
    """Zero-phase Butterworth band-stop between ``low`` and ``high`` Hz."""
    nyq = _nyquist(fs)
    sos = sp.butter(order, [low / nyq, high / nyq], btype="bandstop", output="sos")
    return sp.sosfiltfilt(sos, _as_float(x))


def fir_subbands(fs: float, taps: int = 61, edges=(45.0, 80.0, 200.0)) -> list[np.ndarray]:
    """Four Hamming-window FIR band filters (LP / BP / BP / HP) for band decomposition."""
    nyq = _nyquist(fs)
    e0, e1, e2 = edges
    return [
        sp.firwin(taps, e0 / nyq, window="hamming", pass_zero="lowpass"),
        sp.firwin(taps, [e0 / nyq, e1 / nyq], window="hamming", pass_zero="bandpass"),
        sp.firwin(taps, [e1 / nyq, e2 / nyq], window="hamming", pass_zero="bandpass"),
        sp.firwin(taps, e2 / nyq, window="hamming", pass_zero="highpass"),
    ]


def decompose_bands(x: np.ndarray, fs: float, **kwargs) -> np.ndarray:
    """Return a ``[num_bands, T]`` array of FIR-filtered sub-bands."""
    filters = fir_subbands(fs, **kwargs)
    return np.stack([sp.filtfilt(b, [1.0], _as_float(x)) for b in filters], axis=0)
