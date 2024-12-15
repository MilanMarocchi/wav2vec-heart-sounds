"""Registry mapping a generator name to its model, loss, sampler and conditioning transform.

Each lookup builds a fresh config so nothing leaks between calls.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import torch.nn as nn

from ..signalproc import MelConfig
from .diffwave import DiffWave, DiffWaveConfig
from .samplers import diffwave_sample, wavegrad_sample
from .trainer import LossStrategy, diffwave_loss, wavegrad_loss
from .wavegrad import WaveGrad, WaveGradConfig

# f_max differs by signal type; everything else about the conditioning mel is model-defined.
_F_MAX = {"ecg": 200.0, "pcg": 500.0, "pcg_ref": 500.0}


def _next_pow2(n: int) -> int:
    return 1 << (n - 1).bit_length()


@dataclass
class GeneratorSpec:
    build_model: Callable[[int], nn.Module]
    loss: LossStrategy
    sample: Callable
    mel: Callable[[str], MelConfig]
    sample_rate: int
    hop_length: int
    crop_frames: int


def _diffwave_mel(signal: str) -> MelConfig:
    return MelConfig(sample_rate=4000, n_fft=1024, hop_length=256, n_mels=80,
                     f_max=_F_MAX.get(signal, 500.0))


def _wavegrad_mel(signal: str) -> MelConfig:
    win = 300 * 4
    return MelConfig(sample_rate=4000, n_fft=_next_pow2(win), win_length=win,
                     hop_length=300, n_mels=128, f_max=_F_MAX.get(signal, 500.0))


REGISTRY: dict[str, GeneratorSpec] = {
    "diffwave": GeneratorSpec(
        build_model=lambda num_classes: DiffWave(DiffWaveConfig(num_classes=num_classes)),
        loss=diffwave_loss,
        sample=lambda model, cond, label, **kw: diffwave_sample(model, cond, label, **kw),
        mel=_diffwave_mel,
        sample_rate=4000, hop_length=256, crop_frames=96,
    ),
    "wavegrad": GeneratorSpec(
        build_model=lambda num_classes: WaveGrad(WaveGradConfig(num_classes=num_classes)),
        loss=wavegrad_loss,
        sample=lambda model, cond, label, **kw: wavegrad_sample(model, cond, label, **kw),
        mel=_wavegrad_mel,
        sample_rate=4000, hop_length=300, crop_frames=96,
    ),
}


def get_spec(name: str) -> GeneratorSpec:
    key = name.lower()
    if key not in REGISTRY:
        raise ValueError(f"Unknown generator '{name}'. Options: {sorted(REGISTRY)}")
    return REGISTRY[key]
