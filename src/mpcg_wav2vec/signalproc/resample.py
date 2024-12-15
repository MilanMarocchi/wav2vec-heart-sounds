"""Rational-factor polyphase resampling."""

from __future__ import annotations

from math import gcd

import numpy as np
from scipy import signal as sp


def resample(x: np.ndarray, fs_in: float, fs_out: float) -> np.ndarray:
    """Resample ``x`` from ``fs_in`` to ``fs_out`` via polyphase filtering.

    The up/down factors are reduced by their GCD so ``resample_poly`` runs with the smallest
    integer ratio (e.g. 2000 -> 4125 becomes up=33, down=16).
    """
    if fs_in == fs_out:
        return np.asarray(x)
    up = int(round(fs_out))
    down = int(round(fs_in))
    g = gcd(up, down)
    return sp.resample_poly(x, up // g, down // g)
