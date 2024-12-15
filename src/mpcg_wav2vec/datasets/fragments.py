"""In-memory fragment dataset shared by the CinC and vest classifiers.

A *fragment* is one fixed-length window of a recording, tagged with its binary label and source
patient. A :class:`FragmentDataset` optionally materialises augmented copies of each fragment
(balanced so the minority class receives more copies) and applies per-sample channel selection.

Fragments live in memory and augmentation is applied lazily in ``__getitem__`` so each epoch
sees fresh augmentations.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Sequence

import numpy as np
import torch
from torch.utils.data import Dataset

AugmentFn = Callable[[np.ndarray, int], np.ndarray]


@dataclass
class Fragment:
    waveform: np.ndarray   # [T] (mono) or [T, C] (multichannel)
    label: int
    patient: str


class FragmentDataset(Dataset):
    def __init__(
        self,
        fragments: Sequence[Fragment],
        fs: int,
        augment_num: int = 0,
        augment_fn: AugmentFn | None = None,
        balance: bool = True,
        channel: int = -1,
        cache_augmented: bool = False,
    ):
        self.fs = fs
        self.augment_fn = augment_fn
        self.channel = channel
        self.cache_augmented = cache_augmented
        self._augment_cache: dict[int, np.ndarray] = {}
        self._items: list[tuple[Fragment, bool]] = []

        counts = _class_counts([f.label for f in fragments])
        max_count = max(counts.values()) if counts else 1
        for frag in fragments:
            self._items.append((frag, False))  # original
            if augment_num > 0 and augment_fn is not None:
                copies = augment_num
                if balance and counts.get(frag.label, 0) > 0:
                    copies = int(round(augment_num * max_count / counts[frag.label]))
                self._items.extend((frag, True) for _ in range(copies))

    @property
    def labels(self) -> list[int]:
        return [frag.label for frag, _ in self._items]

    def __len__(self) -> int:
        return len(self._items)

    def __getitem__(self, idx: int) -> dict:
        frag, augmented = self._items[idx]
        wave = frag.waveform
        if augmented and self.augment_fn is not None:
            if self.cache_augmented and idx in self._augment_cache:
                wave = self._augment_cache[idx]
            else:
                wave = self.augment_fn(wave, self.fs)
                if self.cache_augmented:
                    wave = np.asarray(wave, dtype=np.float32)
                    self._augment_cache[idx] = np.ascontiguousarray(wave)
        wave = np.asarray(wave, dtype=np.float32)
        if wave.ndim == 2 and self.channel != -1:
            wave = wave[:, self.channel]
        return {
            "waveform": torch.from_numpy(np.ascontiguousarray(wave)),
            "label": int(frag.label),
            "patient": frag.patient,
        }


def _class_counts(labels) -> dict[int, int]:
    counts: dict[int, int] = {}
    for label in labels:
        counts[label] = counts.get(label, 0) + 1
    return counts
