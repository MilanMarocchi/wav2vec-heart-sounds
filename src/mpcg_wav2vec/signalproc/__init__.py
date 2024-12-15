"""Signal-processing primitives: filtering, normalisation, despiking, resampling, segmentation."""

from .despike import remove_spikes
from .envelopes import hilbert_envelope, homomorphic_envelope
from .filters import (
    band_stop,
    bandpass_cascade,
    butter_bandpass,
    butter_highpass,
    butter_lowpass,
    decompose_bands,
    notch,
    notch_chain,
)
from .normalize import (
    abs_max_normalise,
    kpeak_normalise,
    kpeak_normalise_torch,
    minmax_normalise,
    minmax_normalise_torch,
    to_torch_normalised,
    z_normalise,
    z_normalise_torch,
)
from .preprocess import (
    ECG_BAND,
    PCG_BAND,
    fit_length,
    preprocess_ecg,
    preprocess_four_bands,
    preprocess_pcg,
)
from .resample import resample
from .segment import WindowSpec, segment
from .spectrogram import MelConfig, add_chirp, log_mel
from . import torchproc

__all__ = [
    "remove_spikes",
    "hilbert_envelope",
    "homomorphic_envelope",
    "butter_bandpass",
    "butter_lowpass",
    "butter_highpass",
    "bandpass_cascade",
    "band_stop",
    "notch",
    "notch_chain",
    "decompose_bands",
    "abs_max_normalise",
    "minmax_normalise",
    "minmax_normalise_torch",
    "z_normalise",
    "z_normalise_torch",
    "kpeak_normalise",
    "kpeak_normalise_torch",
    "to_torch_normalised",
    "preprocess_pcg",
    "preprocess_ecg",
    "preprocess_four_bands",
    "fit_length",
    "PCG_BAND",
    "ECG_BAND",
    "resample",
    "WindowSpec",
    "segment",
    "MelConfig",
    "log_mel",
    "add_chirp",
    "torchproc",
]
