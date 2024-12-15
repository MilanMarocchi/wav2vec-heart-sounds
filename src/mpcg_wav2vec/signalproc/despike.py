"""Schmidt spike removal for PCG signals.

Reference: S. E. Schmidt et al., "Segmentation of heart sound recordings by a duration-
dependent hidden Markov model" (2010). The algorithm repeatedly finds the window whose maximum
absolute amplitude (MAA) most exceeds the median MAA, locates the offending spike between its
surrounding zero crossings, and flattens it, until no window exceeds ``threshold`` * median.
"""

from __future__ import annotations

import numpy as np

_FLOOR = 1e-4


def _zero_crossings(window: np.ndarray) -> np.ndarray:
    """Indices where the sign of ``window`` flips."""
    return np.where(np.abs(np.diff(np.sign(window))) > 1)[0]


def _spike_bounds(window: np.ndarray, peak: int) -> tuple[int, int]:
    """Return the [start, end) sample range of the spike around ``peak``."""
    crossings = _zero_crossings(window)
    before = crossings[crossings < peak]
    after = crossings[crossings >= peak]
    start = int(before[-1] + 1) if before.size else 0
    end = int(after[0]) if after.size else window.size - 1
    return start, end


def remove_spikes(signal: np.ndarray, fs: float, threshold: float = 3.0,
                  max_iterations: int = 1000) -> np.ndarray:
    """Return ``signal`` with high-amplitude spikes flattened (500 ms analysis windows)."""
    signal = np.asarray(signal, dtype=np.float64).copy()
    win = round(float(fs) / 2.0)
    if win < 1 or signal.size < win:
        return signal

    n_full = signal.size - (signal.size % win)
    # Column j holds samples [j*win : (j+1)*win); mutating the view mutates ``frames`` in place.
    frames = signal[:n_full].reshape(-1, win).T  # shape [win, num_windows]

    for _ in range(max_iterations):
        maas = np.max(np.abs(frames), axis=0)
        median = np.median(maas)
        if median == 0 or not np.any(maas > threshold * median):
            break
        w = int(np.argmax(maas))
        peak = int(np.argmax(np.abs(frames[:, w])))
        start, end = _spike_bounds(frames[:, w], peak)
        frames[start:end, w] = _FLOOR

    signal[:n_full] = frames.T.reshape(-1)
    return signal
