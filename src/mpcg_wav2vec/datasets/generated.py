"""Loader for synthetic waveform datasets produced by ``generative.generate_dataset``.

A generated dataset is a directory of WAV files plus a ``REFERENCE.csv`` manifest with columns
``patient,label,file``. Labels are already binary (0 = normal, 1 = abnormal). This turns such a
directory into the same :class:`Fragment` list the real loaders produce, so real and synthetic
data mix transparently in the training schedule.
"""

from __future__ import annotations

import os

import numpy as np
import pandas as pd
from scipy.io import wavfile
from tqdm import tqdm

from ..signalproc import WindowSpec, abs_max_normalise, resample, segment
from .fragments import Fragment


def _load_wav(path: str, fs_out: int) -> np.ndarray:
    sr, wave = wavfile.read(path)
    x = np.asarray(wave, dtype=np.float64)
    if x.ndim == 2:  # collapse to mono if needed
        x = x.mean(axis=1)
    return abs_max_normalise(resample(x, sr, fs_out))


def generated_fragments(manifest_dir: str, *, fs_out: int, window: WindowSpec,
                        proportion: float = 1.0, seed: int = 0) -> list[Fragment]:
    """Read a generated dataset directory into windowed fragments."""
    manifest = os.path.join(manifest_dir, "REFERENCE.csv")
    df = pd.read_csv(manifest)
    if proportion < 1.0:
        df = df.sample(frac=proportion, random_state=seed)

    fragments: list[Fragment] = []
    for _, row in tqdm(df.iterrows(), total=len(df), desc="Loading synthetic", unit="wav"):
        path = os.path.join(manifest_dir, str(row["file"]))
        if not os.path.exists(path):
            continue
        label = 1 if int(row["label"]) == 1 else 0
        wave = _load_wav(path, fs_out)
        for w in segment(wave, fs_out, window):
            fragments.append(Fragment(waveform=w, label=label, patient=str(row["patient"])))
    return fragments
