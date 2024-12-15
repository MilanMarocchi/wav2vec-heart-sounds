"""Multichannel wearable-vest PCG loader.

The processed vest recordings are multichannel WAV files (one recording per patient, read with
``scipy.io.wavfile``). Channels follow a fixed layout: PCG microphones 1-7 occupy WAV columns
0-6, the ECG lead ``E`` is column 7 and a second ECG ``E2`` is column 8. Recordings are located
by matching the patient id (from the reference CSV ``patient`` column) against WAV filenames.

The reference CSV shares the CinC format (``patient`` column, binary label column, per-fold
``split`` columns). Channels are given as 1-indexed microphone numbers (matching ``-H 1,2,3,4,5,6``).
"""

from __future__ import annotations

import os
from functools import partial

import numpy as np
from scipy.io import wavfile
from tqdm import tqdm

from ..augment import AugmentConfig, augment_multi_pcg
from ..signalproc import WindowSpec, preprocess_ecg, preprocess_pcg, segment
from .cinc import _binary_label, _label_column, read_split
from .fragments import Fragment, FragmentDataset

# Processed-vest channel layout: microphone / lead -> WAV column index.
VEST_CHANNEL_MAP: dict[object, int] = {1: 0, 2: 1, 3: 2, 4: 3, 5: 4, 6: 5, 7: 6, "E": 7, "E2": 8}


def read_vest_wav(path: str) -> tuple[np.ndarray, int]:
    """Read a multichannel WAV as float32 ``[T, C]`` (integer PCM scaled to [-1, 1])."""
    fs, signal = wavfile.read(path)
    if np.issubdtype(signal.dtype, np.integer):
        signal = signal.astype(np.float32) / np.iinfo(signal.dtype).max
    else:
        signal = signal.astype(np.float32)
    if signal.ndim == 1:
        signal = signal[:, None]
    return signal, fs


def _patient_files(data_dir: str, patient: str) -> list[str]:
    return sorted(
        os.path.join(data_dir, f)
        for f in os.listdir(data_dir)
        if patient in f and f.lower().endswith(".wav")
    )


def _preprocess_channel(column_signal: np.ndarray, fs: int, fs_out: int, is_ecg: bool) -> np.ndarray:
    return preprocess_ecg(column_signal, fs, fs_out) if is_ecg else preprocess_pcg(column_signal, fs, fs_out)


def build_fragments(
    data_dir: str,
    csv_path: str,
    subset: str,
    *,
    fs_out: int,
    window: WindowSpec,
    channels: list,
    fold: int = 1,
) -> list[Fragment]:
    df = read_split(csv_path, subset, fold)
    label_col = _label_column(df)
    columns = [(c, VEST_CHANNEL_MAP[c]) for c in channels if c in VEST_CHANNEL_MAP]
    fragments: list[Fragment] = []

    for _, row in tqdm(df.iterrows(), total=len(df),
                       desc=f"Loading vest ({len(columns)}ch) [{subset}]", unit="rec"):
        patient = str(row["patient"])
        label = _binary_label(row[label_col])
        for wav_path in _patient_files(data_dir, patient):
            signal, fs = read_vest_wav(wav_path)
            processed = [
                _preprocess_channel(signal[:, col], fs, fs_out, is_ecg=(name in ("E", "E2")))
                for name, col in columns
                if col < signal.shape[1]
            ]
            if not processed:
                continue
            n = min(len(ch) for ch in processed)
            stacked = np.stack([ch[:n] for ch in processed], axis=1)   # [T, C]
            for w in segment(stacked, fs_out, window):                  # [N, win, C]
                fragments.append(Fragment(waveform=w, label=label, patient=patient))
    return fragments


def _multi_augment(wave: np.ndarray, fs: int, cfg: AugmentConfig) -> np.ndarray:
    channels = [wave[:, i] for i in range(wave.shape[1])]
    augmented = augment_multi_pcg(channels, fs, cfg)
    n = min(len(c) for c in augmented)
    return np.stack([c[:n] for c in augmented], axis=1)


def vest_dataset(
    data_dir: str,
    csv_path: str,
    subset: str,
    *,
    fs_out: int,
    window: WindowSpec,
    channels: list,
    fold: int = 1,
    augment_num: int = 0,
    augment_config: AugmentConfig | None = None,
    channel: int = -1,
) -> FragmentDataset:
    fragments = build_fragments(data_dir, csv_path, subset, fs_out=fs_out, window=window,
                                channels=channels, fold=fold)
    augment_fn = partial(_multi_augment, cfg=augment_config or AugmentConfig())
    return FragmentDataset(fragments, fs=fs_out, augment_num=augment_num,
                           augment_fn=augment_fn, channel=channel)
