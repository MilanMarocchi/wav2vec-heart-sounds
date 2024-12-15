"""Per-channel encoder fusion, including the two-branch PCG+ECG ``big_rnn:2:wav2vec`` model.

Each input channel is passed through its own Wav2Vec encoder; the mean-pooled 768-d features are
concatenated and classified by a shared MLP. For PCG+ECG this is the ``big_rnn:2:wav2vec``
topology from the paper: train a PCG encoder on channel 0 and an ECG encoder on channel 1, drop
their heads, then fit the fusion classifier on the concatenated features.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from .wav2vec import _HIDDEN, Wav2VecClassifier


class EncoderFusion(nn.Module):
    def __init__(self, branches: list[Wav2VecClassifier], num_classes: int = 2, hidden: int = 128):
        super().__init__()
        self.branches = nn.ModuleList(branches)
        feat = _HIDDEN * len(branches)
        self.classifier = nn.Sequential(
            nn.Linear(feat, 2 * hidden), nn.ReLU(),
            nn.Linear(2 * hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """``x``: [B, T, C] with one channel per branch."""
        if x.dim() != 3 or x.shape[2] != len(self.branches):
            raise ValueError(f"Expected [B, T, {len(self.branches)}] input, got {tuple(x.shape)}")
        features = [branch.encode(x[:, :, i]) for i, branch in enumerate(self.branches)]
        return self.classifier(torch.cat(features, dim=1))


def two_branch_pcg_ecg(pcg_branch: Wav2VecClassifier, ecg_branch: Wav2VecClassifier,
                       num_classes: int = 2) -> EncoderFusion:
    return EncoderFusion([pcg_branch, ecg_branch], num_classes=num_classes)
