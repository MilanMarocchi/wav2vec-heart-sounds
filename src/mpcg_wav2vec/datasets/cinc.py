"""CinC 2016 loaders: single-channel PCG and synchronised Training-A PCG+ECG.

Expected on-disk layout (the standard PhysioNet CinC 2016 format):

* ``data_dir/<patient>.hea`` + signal file readable by ``wfdb`` (channel 0 = PCG; for the
  PCG+ECG Training-A records, channel 1 = ECG).
* a reference CSV with a ``patient`` column, a binary label column (``abnormality``/``label``),
  and per-fold ``split`` columns (``split`` for fold 1, ``split<fold>`` otherwise) whose values
  are ``train`` / ``valid`` / ``test``. Pass ``subset='all'`` to ignore the split column.
"""

from __future__ import annotations

import os

import numpy as np
import pandas as pd
import torch
import wfdb
from tqdm import tqdm

from ..augment import AugmentConfig, augment_pcg, augment_pcg_ecg
from ..signalproc import WindowSpec, preprocess_ecg, preprocess_pcg, segment
from .fragments import Fragment, FragmentDataset, _class_counts

_LABEL_COLUMNS = ("abnormality", "label", "diagnosis")


def read_split(csv_path: str, subset: str, fold: int = 1) -> pd.DataFrame:
    df = pd.read_csv(csv_path, comment="#")
    if subset != "all":
        col = "split" if fold == 1 else f"split{fold}"
        df = df[df[col] == subset]
    return df


def _label_column(df: pd.DataFrame) -> str:
    for col in _LABEL_COLUMNS:
        if col in df.columns:
            return col
    raise KeyError(f"No label column ({_LABEL_COLUMNS}) in split CSV columns {list(df.columns)}")


def _binary_label(raw) -> int:
    """Map CinC labels to {0: normal, 1: abnormal}. Accepts -1/1 or 0/1 encodings."""
    return 1 if int(raw) == 1 else 0


def _read_record(data_dir: str, patient: str):
    rec = wfdb.rdrecord(os.path.join(data_dir, str(patient)))
    return rec.p_signal, rec.fs


def build_fragments(
    data_dir: str,
    csv_path: str,
    subset: str,
    *,
    fs_out: int,
    window: WindowSpec,
    ecg: bool = False,
    fold: int = 1,
    augment_num: int = 0,
    augment_config: AugmentConfig | None = None,
    balance_augment: bool = True,
) -> list[Fragment]:
    """Load, optionally augment full patient records, then window into fragments."""
    df = read_split(csv_path, subset, fold)
    label_col = _label_column(df)
    fragments: list[Fragment] = []
    cfg = augment_config or AugmentConfig()
    labels = [_binary_label(row[label_col]) for _, row in df.iterrows()]
    counts = _class_counts(labels)
    max_count = max(counts.values()) if counts else 1

    kind = "PCG+ECG" if ecg else "PCG"
    for _, row in tqdm(df.iterrows(), total=len(df),
                       desc=f"Loading CinC {kind} [{subset}]", unit="rec"):
        patient = str(row["patient"])
        label = _binary_label(row[label_col])
        try:
            signal, fs = _read_record(data_dir, patient)
        except (FileNotFoundError, ValueError):
            continue

        pcg = preprocess_pcg(signal[:, 0], fs, fs_out)
        if ecg and signal.shape[1] > 1:
            ecg_sig = preprocess_ecg(signal[:, 1], fs, fs_out)
            n = min(len(pcg), len(ecg_sig))
            base_signal = np.stack([pcg[:n], ecg_sig[:n]], axis=1)      # [T, 2]
        else:
            base_signal = pcg                                           # [T]

        _append_segmented(fragments, base_signal, fs_out, window, label, patient)

        copies = augment_num
        if balance_augment and counts.get(label, 0) > 0:
            copies = int(round(augment_num * max_count / counts[label]))
        for copy_idx in range(copies):
            aug_signal = _pcg_augment(base_signal, fs_out, cfg)
            _append_segmented(
                fragments, aug_signal, fs_out, window, label, f"{patient}#aug{copy_idx + 1}"
            )
    return fragments


def _append_segmented(
    fragments: list[Fragment],
    signal: np.ndarray,
    fs: int,
    window: WindowSpec,
    label: int,
    patient: str,
) -> None:
    windows = segment(signal, fs, window)
    for w in windows:
        fragments.append(Fragment(waveform=w, label=label, patient=patient))


def _pcg_augment(wave: np.ndarray, fs: int, cfg: AugmentConfig) -> np.ndarray:
    if wave.ndim == 2:  # PCG+ECG pair
        ecg_aug, pcg_aug = augment_pcg_ecg(wave[:, 1], wave[:, 0], fs, cfg)
        n = min(len(pcg_aug), len(ecg_aug))
        return np.stack([pcg_aug[:n], ecg_aug[:n]], axis=1)
    return augment_pcg(wave, fs, cfg)


def cinc_dataset(
    data_dir: str,
    csv_path: str,
    subset: str,
    *,
    fs_out: int,
    window: WindowSpec,
    ecg: bool = False,
    fold: int = 1,
    augment_num: int = 0,
    augment_config: AugmentConfig | None = None,
    channel: int = -1,
) -> FragmentDataset:
    fragments = build_fragments(
        data_dir, csv_path, subset, fs_out=fs_out, window=window, ecg=ecg, fold=fold,
        augment_num=augment_num, augment_config=augment_config,
    )
    return FragmentDataset(fragments, fs=fs_out, channel=channel)


def pad_collate(batch: list[dict]) -> dict:
    """Zero-pad variable-length waveforms to the longest in the batch.

    Handles both mono ``[T]`` and multichannel ``[T, C]`` fragments.
    """
    waves = [b["waveform"] for b in batch]
    max_len = max(w.shape[0] for w in waves)
    multichannel = waves[0].dim() == 2
    if multichannel:
        c = waves[0].shape[1]
        out = torch.zeros(len(waves), max_len, c)
    else:
        out = torch.zeros(len(waves), max_len)
    for i, w in enumerate(waves):
        out[i, : w.shape[0]] = w
    return {
        "waveform": out,
        "label": torch.tensor([b["label"] for b in batch], dtype=torch.long),
        "patient": [b["patient"] for b in batch],
    }
