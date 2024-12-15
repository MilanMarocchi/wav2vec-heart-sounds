"""Composed, probabilistic augmentation pipelines for PCG / ECG / multichannel PCG.

The single-channel PCG and synchronised PCG+ECG pipelines reproduce the augmentation used for the
Training-A / CinC runs: HPSS recombination, additive white noise, a small time-stretch, wandering
volume, parametric-EQ banding, baseline wander (ECG) and recorded clinical noise, applied with the
probabilities below. The multichannel (vest) pipeline keeps its own probabilities since it augments
all channels jointly.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..signalproc.normalize import abs_max_normalise, minmax_normalise
from . import primitives as P
from .noise_sources import ecg_noise, pcg_noise

# Time-stretch ranges (rate multipliers): a near-identity micro-stretch for single-channel PCG,
# a wider stretch for the synchronised pair.
PCG_STRETCH = (1.004, 1.006)
PAIR_STRETCH = (0.8, 1.2)


@dataclass
class AugmentConfig:
    ephnogram_dir: str = ""
    mit_dir: str = ""
    prob_hpss: float = 0.75
    prob_noise: float = 0.30          # split across applications (prob_noise / 4 each)
    prob_time_warp: float = 0.25
    prob_wandering_volume: float = 0.75
    prob_banding: float = 0.25
    prob_baseline_wander: float = 0.30
    prob_real_noise: float = 0.5


def _chance(p: float) -> bool:
    return np.random.rand() < p


def augment_pcg(pcg: np.ndarray, fs: int, cfg: AugmentConfig | None = None) -> np.ndarray:
    """Single-channel PCG augmentation (HPSS uses 4 components; small time-stretch, no mag warp)."""
    cfg = cfg or AugmentConfig()
    x = minmax_normalise(pcg.copy())
    if _chance(cfg.prob_hpss):
        x, _ = P.hpss_recombine(x, include_residual=False)
    if _chance(cfg.prob_noise / 4):
        x = P.add_white_noise(x)
    if _chance(cfg.prob_time_warp):
        x = abs_max_normalise(P.time_stretch(x, fs, P.randfloat(*PCG_STRETCH)))
    if _chance(cfg.prob_wandering_volume):
        x = P.sinusoidal_envelope(x, fs)
    if _chance(cfg.prob_noise / 4):
        x = P.add_white_noise(x)
    if _chance(cfg.prob_banding):
        x = P.parametric_eq(x, fs, 2, 500)
    if _chance(cfg.prob_real_noise) and cfg.ephnogram_dir:
        x = x + pcg_noise(fs, len(x), cfg.ephnogram_dir)
    return abs_max_normalise(x)


def augment_ecg(ecg: np.ndarray, fs: int, cfg: AugmentConfig | None = None) -> np.ndarray:
    cfg = cfg or AugmentConfig()
    x = minmax_normalise(ecg.copy())
    if _chance(cfg.prob_noise / 4):
        x = P.add_white_noise(x)
    if _chance(cfg.prob_baseline_wander):
        x = P.baseline_wander(x, fs)
    if _chance(cfg.prob_time_warp):
        x = abs_max_normalise(P.time_stretch(x, fs, P.randfloat(*PAIR_STRETCH)))
    if _chance(cfg.prob_noise / 4):
        x = P.add_white_noise(x)
    if _chance(cfg.prob_banding):
        x = P.parametric_eq(x, fs, 0.25, 100)
    if _chance(cfg.prob_real_noise) and cfg.mit_dir:
        x = x + ecg_noise(fs, len(x), cfg.mit_dir)
    return abs_max_normalise(x)


def augment_pcg_ecg(ecg: np.ndarray, pcg: np.ndarray, fs: int,
                    cfg: AugmentConfig | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Jointly augment a synchronised ECG/PCG pair (HPSS 7 components; shared time-stretch)."""
    cfg = cfg or AugmentConfig()
    e = minmax_normalise(ecg.copy())
    p = minmax_normalise(pcg.copy())

    if _chance(cfg.prob_hpss):
        p, n = P.hpss_recombine(p, include_residual=True)
        e = e[:n]
    if _chance(cfg.prob_noise / 4):
        p = P.add_white_noise(p)
    if _chance(cfg.prob_noise / 4):
        e = P.add_white_noise(e)
    if _chance(cfg.prob_baseline_wander):
        e = P.baseline_wander(e, fs)
    if _chance(cfg.prob_time_warp):
        rate = P.randfloat(*PAIR_STRETCH)
        e = abs_max_normalise(P.time_stretch(e, fs, rate))
        p = abs_max_normalise(P.time_stretch(p, fs, rate))
    if _chance(cfg.prob_wandering_volume):
        p = P.sinusoidal_envelope(p, fs)
    if _chance(cfg.prob_noise / 4):
        p = P.add_white_noise(p)
    if _chance(cfg.prob_noise / 4):
        e = P.add_white_noise(e)
    if _chance(cfg.prob_banding):
        p = P.parametric_eq(p, fs, 2, 500)
    if _chance(cfg.prob_banding):
        e = P.parametric_eq(e, fs, 0.25, 100)
    if _chance(cfg.prob_real_noise) and cfg.mit_dir:
        e = e + ecg_noise(fs, len(e), cfg.mit_dir)
    if _chance(cfg.prob_real_noise) and cfg.ephnogram_dir:
        p = p + pcg_noise(fs, len(p), cfg.ephnogram_dir)
    return abs_max_normalise(e), abs_max_normalise(p)


# Vest / multichannel probabilities (all channels augmented identically to preserve timing).
_MULTI_PROB_NOISE = 0.30
_MULTI_PROB_TIME_WARP = 0.35
_MULTI_PROB_WANDER = 0.75
_MULTI_PROB_REAL_NOISE = 0.25
_MULTI_STRETCH = (0.7, 1.3)


def augment_multi_pcg(channels: list[np.ndarray], fs: int,
                      cfg: AugmentConfig | None = None) -> list[np.ndarray]:
    """Augment every PCG channel identically so cross-channel timing/phase is preserved."""
    cfg = cfg or AugmentConfig()
    chans = [abs_max_normalise(c.copy()) for c in channels]

    if _chance(_MULTI_PROB_NOISE / 4):
        chans = [P.add_white_noise(c) for c in chans]
    if _chance(_MULTI_PROB_TIME_WARP):
        rate = P.randfloat(*_MULTI_STRETCH)
        chans = [abs_max_normalise(P.time_stretch(c, fs, rate, keep_length=True)) for c in chans]
    if _chance(_MULTI_PROB_WANDER):
        t = np.arange(chans[0].size) / fs
        mod = (P.randfloat(0.01, 0.25) * np.sin(2 * np.pi * (P.randfloat(0.05, 0.5) * t + P.randfloat(0, 1)))
               + P.randfloat(0.01, 0.25) * np.sin(2 * np.pi * (P.randfloat(0.001, 0.05) * t + P.randfloat(0, 1))))
        chans = [abs_max_normalise(c * (1.0 + mod)) for c in chans]
    if _chance(_MULTI_PROB_NOISE / 4):
        chans = [P.add_white_noise(c) for c in chans]
    if _chance(_MULTI_PROB_REAL_NOISE) and cfg.ephnogram_dir:
        shared = pcg_noise(fs, len(chans[0]), cfg.ephnogram_dir)
        chans = [abs_max_normalise(c + shared) for c in chans]
    return chans
