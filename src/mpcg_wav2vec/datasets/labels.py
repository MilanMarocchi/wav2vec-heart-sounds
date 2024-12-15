"""Label vocabularies and a class-balanced sampler.

Classification is binary (0 = normal, 1 = abnormal). The generative models condition on a
dataset-specific label vocabulary; those vocabularies are kept here so the diffusion embedding
size stays consistent between training and generation.
"""

from __future__ import annotations

import itertools

import torch
from torch.utils.data import WeightedRandomSampler

# Binary classification classes.
BINARY_LABELS = (0, 1)

# Generative conditioning vocabularies (index order defines the embedding rows).
LABEL_SETS: dict[str, tuple] = {
    "training-a": (-1, 1),
    "training-a-extended": ("Normal", "Benign", "MVP", "MPC", "AD"),
    "ticking-heart-multi": (-1, 1),
    "ticking-heart-extended": tuple(
        f"C{a}X{b}{s}" for a, b in itertools.permutations(range(1, 7), 2) for s in "NA"
    ),
    "cinc-channels": tuple(f"{c}{s}" for c in (2, 3, 4, 5, 6) for s in "NA"),
    "multichannel-mixed": (0, 1, 2),
}


def label_set(dataset: str) -> tuple:
    try:
        return LABEL_SETS[dataset]
    except KeyError as exc:
        raise NotImplementedError(f"No label vocabulary for dataset '{dataset}'") from exc


def num_classes(dataset: str) -> int:
    return len(label_set(dataset))


def label_to_index(dataset: str, label) -> int:
    return label_set(dataset).index(label)


def index_to_label(dataset: str, index: int):
    return label_set(dataset)[index]


def balanced_sampler(labels) -> WeightedRandomSampler:
    """Weighted sampler that draws each class with equal probability."""
    labels = torch.as_tensor(list(labels), dtype=torch.long)
    counts = torch.bincount(labels)
    counts = torch.clamp(counts.float(), min=1.0)
    weights = (1.0 / counts)[labels]
    return WeightedRandomSampler(weights=weights, num_samples=len(weights), replacement=True)
