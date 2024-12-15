"""Denoising-diffusion generators (DiffWave, WaveGrad) for synthetic heart-sound augmentation."""

from .diffwave import DiffWave, DiffWaveConfig
from .generate import generate_dataset
from .registry import REGISTRY, GeneratorSpec, get_spec
from .samplers import diffwave_sample, wavegrad_sample
from .schedules import NoiseSchedule
from .trainer import GenerativeTrainer, diffwave_loss, wavegrad_loss
from .wavegrad import WaveGrad, WaveGradConfig

__all__ = [
    "DiffWave",
    "DiffWaveConfig",
    "WaveGrad",
    "WaveGradConfig",
    "NoiseSchedule",
    "GenerativeTrainer",
    "diffwave_loss",
    "wavegrad_loss",
    "diffwave_sample",
    "wavegrad_sample",
    "generate_dataset",
    "get_spec",
    "GeneratorSpec",
    "REGISTRY",
]
