"""Traditional (non-generative) waveform augmentation."""

from .pipelines import (
    AugmentConfig,
    augment_ecg,
    augment_multi_pcg,
    augment_pcg,
    augment_pcg_ecg,
)
from .torchaug import augment_pcg_batch

__all__ = [
    "AugmentConfig",
    "augment_pcg",
    "augment_ecg",
    "augment_pcg_ecg",
    "augment_multi_pcg",
    "augment_pcg_batch",
]
