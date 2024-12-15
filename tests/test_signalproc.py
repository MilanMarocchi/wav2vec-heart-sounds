import numpy as np
import pytest

from mpcg_wav2vec import signalproc as sp
from mpcg_wav2vec.config import default_window


def _tone(freq, fs, seconds=3.0):
    t = np.arange(int(fs * seconds)) / fs
    return np.sin(2 * np.pi * freq * t)


def test_bandpass_attenuates_out_of_band():
    fs = 1000
    inband = _tone(100, fs)
    below = _tone(5, fs)
    above = _tone(480, fs)
    assert np.mean(sp.butter_bandpass(inband, fs, 25, 450) ** 2) > 0.3   # passes
    assert np.mean(sp.butter_bandpass(below, fs, 25, 450) ** 2) < 0.05    # blocked
    assert np.mean(sp.butter_bandpass(above, fs, 25, 450) ** 2) < 0.05    # blocked


def test_despike_removes_spike():
    fs = 1000
    x = _tone(40, fs).copy()
    x[1500] = 50.0
    cleaned = sp.remove_spikes(x, fs)
    assert np.max(np.abs(cleaned)) < 5.0


def test_resample_length():
    x = _tone(50, 2000, seconds=3.0)
    y = sp.resample(x, 2000, 4125)
    assert abs(len(y) - round(len(x) * 4125 / 2000)) <= 1


def test_normalisers_ranges():
    x = np.random.randn(5000) * 3 + 2
    assert np.isclose(sp.minmax_normalise(x).min(), -1.0)
    assert np.isclose(sp.minmax_normalise(x).max(), 1.0)
    assert np.max(np.abs(sp.abs_max_normalise(x))) <= 1.0 + 1e-6


def test_preprocess_pcg_finite_and_bounded():
    fs = 2000
    x = _tone(90, fs) + 0.5 * _tone(600, fs)
    out = sp.preprocess_pcg(x, fs, 4125)
    assert np.all(np.isfinite(out))
    assert np.max(np.abs(out)) <= 1.0 + 1e-6


def test_segmentation_window_count_and_length():
    fs = 1000
    spec = default_window("vest")  # 2 s windows, 0.25 overlap
    x = _tone(40, fs, seconds=10.0)
    windows = sp.segment(x, fs, spec)
    assert windows.shape[1] == spec.window_len(fs) == 2000
    assert windows.shape[0] >= 4


def test_log_mel_range():
    import torch
    fs = 4000
    mc = sp.MelConfig(sample_rate=fs, n_fft=1024, hop_length=256, n_mels=80, f_max=500)
    mel = sp.log_mel(torch.from_numpy(_tone(100, fs)).float(), mc.build())
    assert mel.min() >= 0.0 and mel.max() <= 1.0
