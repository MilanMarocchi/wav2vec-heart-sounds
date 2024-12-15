"""Real recorded-noise sources used for clinical-noise augmentation.

PCG noise comes from the EPHNOGRAM auxiliary channels; ECG noise from the MIT-BIH Noise Stress
Test records (electrode motion ``em``, baseline wander ``bw``, muscle artefact ``ma``). Both
paths are supplied by the caller; if a record cannot be read we fall back to silence/Gaussian
noise rather than crashing a training run.
"""

from __future__ import annotations

import glob
import os
import random

import numpy as np
import wfdb
from scipy import signal as sp

from ..signalproc.normalize import abs_max_normalise
from .primitives import randfloat, random_crop


def _load_record(path: str, max_seconds: float = -1.0) -> wfdb.Record:
    header = wfdb.rdheader(path)
    total = header.sig_len
    want = total if max_seconds <= -1.0 else round(max_seconds * header.fs)
    if total > want:
        start = random.randint(0, total - want)
        return wfdb.rdrecord(path, sampfrom=start, sampto=start + want)
    return wfdb.rdrecord(path)


def pcg_noise(fs: float, length: int, ephnogram_dir: str) -> np.ndarray:
    """Random EPHNOGRAM AUX-channel noise, scaled down and cropped to ``length`` samples."""
    files = glob.glob(os.path.join(ephnogram_dir, "*.hea"))
    for _ in range(50):
        try:
            rec = _load_record(random.choice(files).removesuffix(".hea"))
            names = rec.sig_name
            aux1 = sp.resample_poly(rec.p_signal[:, names.index("AUX1")], int(fs), rec.fs)
            aux2 = sp.resample_poly(rec.p_signal[:, names.index("AUX2")], int(fs), rec.fs)
            aux1 = random.choice([0.0, randfloat(0.0, 0.05)]) * abs_max_normalise(random_crop(aux1, length))
            aux2 = random.choice([0.0, randfloat(0.0, 0.05)]) * abs_max_normalise(random_crop(aux2, length))
            combined = aux1 + aux2
            if np.max(np.abs(combined)) > 0:
                combined = abs_max_normalise(combined)
            return combined
        except (ValueError, IndexError):
            continue
    return np.zeros(length)


def ecg_noise(fs: float, length: int, mit_dir: str) -> np.ndarray:
    """Sum of scaled MIT-BIH em/bw/ma noise, cropped to ``length`` samples."""
    try:
        parts = []
        scales = {"em": (0.0, 0.25), "bw": (0.0, 0.5), "ma": (0.0, 0.25)}
        for name, (lo, hi) in scales.items():
            rec = _load_record(os.path.join(mit_dir, name))
            sig = sp.resample_poly(rec.p_signal[:, 0], int(fs), rec.fs)
            parts.append(random.choice([0.0, randfloat(lo, hi)]) * abs_max_normalise(random_crop(sig, length)))
        return sum(parts)
    except (FileNotFoundError, ValueError, IndexError):
        return np.zeros(length)
