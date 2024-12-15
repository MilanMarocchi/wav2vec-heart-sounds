"""Fixed-length overlapping time-window segmentation.

The paper segments recordings into overlapping windows: 4 s for CinC/Training-A, 2 s for the
vest data, with 0.25 s overlap and the first 0.3 s of every recording discarded. This is applied
directly on the waveform.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .preprocess import fit_length


@dataclass(frozen=True)
class WindowSpec:
    window_s: float
    overlap_s: float = 0.25
    start_pad_s: float = 0.3

    def window_len(self, fs: float) -> int:
        return int(round(self.window_s * fs))

    def hop_len(self, fs: float) -> int:
        return max(1, int(round((self.window_s - self.overlap_s) * fs)))


def window_starts(n_samples: int, fs: float, spec: WindowSpec) -> list[int]:
    start = int(round(spec.start_pad_s * fs))
    hop = spec.hop_len(fs)
    win = spec.window_len(fs)
    if n_samples <= start:
        return []
    last = max(start, n_samples - win)
    return list(range(start, last + 1, hop)) or [start]


def segment(signal: np.ndarray, fs: float, spec: WindowSpec) -> np.ndarray:
    """Split ``signal`` (``[T]`` or ``[T, C]``) into ``[N, win]`` or ``[N, win, C]`` windows."""
    signal = np.asarray(signal)
    win = spec.window_len(fs)
    windows = []
    for s in window_starts(signal.shape[0], fs, spec):
        chunk = signal[s:s + win]
        chunk, _ = fit_length(chunk, win)
        windows.append(chunk)
    if not windows:
        empty_shape = (0, win) if signal.ndim == 1 else (0, win, signal.shape[1])
        return np.zeros(empty_shape, dtype=signal.dtype)
    return np.stack(windows, axis=0)
