"""Mel-spectrogram builders and log-scaled normalisation used as diffusion conditioning."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
import torchaudio
from scipy.signal import chirp as _chirp


@dataclass(frozen=True)
class MelConfig:
    """Parameters for a conditioning mel-spectrogram.

    ``f_max`` is the only field that differs between PCG (500 Hz) and ECG (200 Hz).
    """
    sample_rate: int
    n_fft: int
    hop_length: int
    win_length: int | None = None
    n_mels: int = 80
    f_min: float = 0.125
    f_max: float = 500.0

    def build(self) -> torchaudio.transforms.MelSpectrogram:
        return torchaudio.transforms.MelSpectrogram(
            sample_rate=self.sample_rate,
            n_fft=self.n_fft,
            win_length=self.win_length or self.n_fft,
            hop_length=self.hop_length,
            f_min=self.f_min,
            f_max=self.f_max,
            n_mels=self.n_mels,
            power=1.0,
            normalized=True,
        )


def log_mel(signal: torch.Tensor, transform: torchaudio.transforms.MelSpectrogram) -> torch.Tensor:
    """Mel-spectrogram in dB, shifted/scaled into [0, 1] (matches the diffusion conditioner)."""
    mel = transform(signal)
    mel = 20.0 * torch.log10(torch.clamp(mel, min=1e-5)) - 20.0
    return torch.clamp((mel + 100.0) / 100.0, 0.0, 1.0)


def add_chirp(x: np.ndarray, fs: float) -> np.ndarray:
    """Add a full-band linear chirp for spectral-reference plots."""
    t = np.arange(len(x)) / fs
    wave = np.asarray(_chirp(t, f0=0, f1=fs / 2, t1=t[-1] if len(t) else 1.0, method="linear"))
    peak = np.max(np.abs(wave)) or 1.0
    wave = wave / peak * max(0.5, float(np.max(np.abs(x))) if len(x) else 0.5)
    return x + wave
