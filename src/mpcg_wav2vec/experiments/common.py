"""Shared helpers for the ablation runners."""

from __future__ import annotations

import json
from pathlib import Path

from torch.utils.data import DataLoader

from ..datasets.cinc import pad_collate
from ..datasets.labels import balanced_sampler


def make_loader(dataset, batch_size: int, train: bool, num_workers: int = 0) -> DataLoader:
    sampler = balanced_sampler(dataset.labels) if train else None
    return DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=sampler,
        shuffle=(train and sampler is None),
        num_workers=num_workers,
        collate_fn=pad_collate,
    )


def append_result(results_json: str | None, record: dict) -> None:
    if not results_json:
        return
    path = Path(results_json)
    path.parent.mkdir(parents=True, exist_ok=True)
    existing = json.loads(path.read_text()) if path.exists() else []
    existing.append(record)
    path.write_text(json.dumps(existing, indent=2, default=str))
