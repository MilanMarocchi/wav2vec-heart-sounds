"""Generate patient-level, label-stratified train/valid/test split CSVs.

The classifier loaders read a reference CSV with a ``patient`` column, a ``label`` column, and one
``split`` column per fold (``split``, ``split2``, …) valued ``train``/``valid``/``test``. This
module builds that CSV from CinC-style ``REFERENCE.csv`` label files (rows ``record,label`` with
label in {-1, 1}) or from an explicit label mapping, producing independent patient-level random
splits per fold — matching the patient-level random-split protocol used in the paper.
"""

from __future__ import annotations

import csv
import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class SplitRatios:
    train: float = 0.6
    valid: float = 0.2
    test: float = 0.2

    def __post_init__(self):
        total = self.train + self.valid + self.test
        if abs(total - 1.0) > 1e-6:
            raise ValueError(f"split ratios must sum to 1.0, got {total}")


def read_cinc_labels(data_dir: str) -> dict[str, int]:
    """Read ``<data_dir>/REFERENCE.csv`` (rows ``record,label``) into {record: label}."""
    path = os.path.join(data_dir, "REFERENCE.csv")
    labels: dict[str, int] = {}
    with open(path, newline="") as fh:
        for row in csv.reader(fh):
            if len(row) >= 2 and row[0]:
                labels[row[0].strip()] = int(row[1])
    if not labels:
        raise ValueError(f"no labels read from {path}")
    return labels


def _patient_of(record: str, patient_fn=None) -> str:
    return patient_fn(record) if patient_fn else record


def make_splits(
    labels: dict[str, int],
    *,
    folds: int = 5,
    ratios: SplitRatios | None = None,
    seed: int = 42,
    patient_fn=None,
) -> pd.DataFrame:
    """Return a DataFrame with columns ``patient, label, split[, split2, …]``.

    Splitting is done at the patient level (all records of a patient land in the same subset) and
    stratified by the patient's label, independently for each fold.
    """
    ratios = ratios or SplitRatios()
    records = sorted(labels)

    # Patient -> label (a patient's records share a label in these datasets; take the first).
    patient_label: dict[str, int] = {}
    for rec in records:
        patient_label.setdefault(_patient_of(rec, patient_fn), labels[rec])
    patients = sorted(patient_label)

    columns: dict[str, dict[str, str]] = {}
    for fold in range(1, folds + 1):
        rng = np.random.default_rng(seed + fold)
        assignment: dict[str, str] = {}
        # Stratify: assign within each label class so class balance is preserved per subset.
        for label in sorted(set(patient_label.values())):
            members = [p for p in patients if patient_label[p] == label]
            rng.shuffle(members)
            n = len(members)
            n_test = int(round(n * ratios.test))
            n_valid = int(round(n * ratios.valid))
            for i, p in enumerate(members):
                if i < n_test:
                    assignment[p] = "test"
                elif i < n_test + n_valid:
                    assignment[p] = "valid"
                else:
                    assignment[p] = "train"
        columns["split" if fold == 1 else f"split{fold}"] = assignment

    rows = []
    for rec in records:
        patient = _patient_of(rec, patient_fn)
        row = {"patient": rec, "label": labels[rec]}
        for col, assignment in columns.items():
            row[col] = assignment[patient]
        rows.append(row)
    return pd.DataFrame(rows)


def write_splits(df: pd.DataFrame, out_path: str | Path) -> str:
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    return str(out)


def make_splits_from_dirs(data_dirs: list[str], **kwargs) -> pd.DataFrame:
    """Combine CinC ``REFERENCE.csv`` labels from several directories, then split."""
    labels: dict[str, int] = {}
    for d in data_dirs:
        labels.update(read_cinc_labels(d))
    return make_splits(labels, **kwargs)
