"""High-level preprocessing chains for PCG and ECG (Abbott et al. 2025).

* PCG: NaN-interpolate -> resample -> Schmidt despike -> 25-450 Hz band -> abs-max normalise
* ECG: NaN-interpolate -> resample -> 2-40 Hz band -> abs-max normalise

The band is a causal low-pass at the high edge followed by a high-pass at the low edge
(see ``filters.bandpass_cascade``).
"""

from __future__ import annotations

import numpy as np

from . import filters
from .despike import remove_spikes
from .normalize import abs_max_normalise, interpolate_nans
from .resample import resample

# Band edges (Hz) used by the paper's preprocessing.
PCG_BAND = (25.0, 450.0)
ECG_BAND = (2.0, 40.0)


def preprocess_pcg(pcg: np.ndarray, fs_in: float, fs_out: float, *, despike: bool = True) -> np.ndarray:
    x = interpolate_nans(pcg)
    x = resample(x, fs_in, fs_out)
    if despike:
        x = remove_spikes(x, fs_out)
    x = filters.bandpass_cascade(x, fs_out, *PCG_BAND, order=2)
    return abs_max_normalise(x)


def preprocess_ecg(ecg: np.ndarray, fs_in: float, fs_out: float) -> np.ndarray:
    x = interpolate_nans(ecg)
    x = resample(x, fs_in, fs_out)
    x = filters.bandpass_cascade(x, fs_out, *ECG_BAND, order=2)
    return abs_max_normalise(x)


def preprocess_four_bands(pcg: np.ndarray, fs: float) -> np.ndarray:
    """Return a ``[T, 4]`` FIR band decomposition (four-band PCG split)."""
    return filters.decompose_bands(np.asarray(pcg).squeeze(), fs).T


def fit_length(array, length: int):
    """Pad with zeros or crop ``array`` along axis 0 to exactly ``length`` samples.

    Works for numpy arrays and torch tensors. Returns ``(array, valid_length)`` where
    ``valid_length`` is the number of real (non-padded) samples.
    """
    import torch

    orig = array.shape[0]
    if orig < length:
        pad = length - orig
        if torch.is_tensor(array):
            tail = torch.zeros((pad, *array.shape[1:]), dtype=array.dtype, device=array.device)
            array = torch.cat([array, tail], dim=0)
        else:
            widths = ((0, pad),) + tuple((0, 0) for _ in range(array.ndim - 1))
            array = np.pad(array, widths, mode="constant")
    elif orig > length:
        array = array[:length]
    return array, min(orig, length)
