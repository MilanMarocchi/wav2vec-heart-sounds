"""Builders for the classifier models used by the paper."""

from __future__ import annotations

from .fusion import EncoderFusion, two_branch_pcg_ecg
from .wav2vec import Wav2VecClassifier, Wav2VecConfig


def build_wav2vec(config: Wav2VecConfig) -> Wav2VecClassifier:
    return Wav2VecClassifier(config)


def build_two_branch(pcg_config: Wav2VecConfig, ecg_config: Wav2VecConfig,
                     num_classes: int = 2) -> EncoderFusion:
    """Fresh (untrained) two-branch model; branches are trained separately upstream."""
    return two_branch_pcg_ecg(Wav2VecClassifier(pcg_config), Wav2VecClassifier(ecg_config), num_classes)
