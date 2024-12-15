"""Cross-cutting constants and small helpers shared across the pipeline."""

from __future__ import annotations

import torch

from .signalproc.segment import WindowSpec

# Classification sample rates found optimal by the paper's grid search.
CLASSIFY_FS_CINC = 16000
CLASSIFY_FS_DEFAULT = 4125

# Sample rate the diffusion generators operate at.
GENERATIVE_FS = 4000

# Per-dataset segmentation windows (0.25 s overlap, 0.3 s start pad throughout).
WINDOWS = {
    "cinc": WindowSpec(window_s=4.0),
    "training-a": WindowSpec(window_s=4.0),
    "vest": WindowSpec(window_s=2.0),
}


def default_window(dataset: str) -> WindowSpec:
    return WINDOWS.get(dataset, WindowSpec(window_s=4.0))


def get_device(prefer_cuda: bool = True) -> torch.device:
    if prefer_cuda and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")
