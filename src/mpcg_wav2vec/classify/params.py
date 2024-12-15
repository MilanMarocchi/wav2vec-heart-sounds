"""Per-setting classifier hyperparameters.

``head_hidden`` values follow the paper's per-dataset choices (CinC uses a wider three-layer
head; Training-A/vest use a single hidden layer).
"""

from __future__ import annotations

from dataclasses import dataclass, replace

from .wav2vec import Wav2VecConfig


@dataclass(frozen=True)
class TrainingArgs:
    epochs: int = 20
    optimizer: str = "sgd"
    lr: float = 1e-3
    weight_decay: float = 1e-5
    batch_size: int = 64


_MODEL_PRESETS = {
    "cinc": Wav2VecConfig(num_classes=2, num_channels=1, head_hidden=(512, 512, 512), fs=16000),
    "training-a": Wav2VecConfig(num_classes=2, num_channels=1, head_hidden=(512,), fs=4125),
    "training-a-ecg": Wav2VecConfig(num_classes=2, num_channels=1, head_hidden=(128,), fs=4125),
    "vest": Wav2VecConfig(num_classes=2, num_channels=6, head_hidden=(256,), fs=4125),
}


def model_config(setting: str, **overrides) -> Wav2VecConfig:
    base = _MODEL_PRESETS.get(setting, Wav2VecConfig())
    return replace(base, **overrides) if overrides else base


def training_args(setting: str, **overrides) -> TrainingArgs:
    base = TrainingArgs()
    return replace(base, **overrides) if overrides else base
