import numpy as np
import torch

from mpcg_wav2vec.augment import AugmentConfig, augment_pcg_batch
from mpcg_wav2vec.signalproc import preprocess_pcg, torchproc
from mpcg_wav2vec.signalproc.segment import WindowSpec


def test_batch_augment_shape_and_bounds():
    x = torch.randn(8, 4125)
    cfg = AugmentConfig(prob_hpss=0.0, prob_real_noise=0.0)
    out = augment_pcg_batch(x, fs=4125, cfg=cfg)
    assert out.shape == x.shape
    assert torch.isfinite(out).all()
    assert out.abs().max() <= 1.0 + 1e-5


def test_batch_preprocess_shapes():
    x = torch.randn(4, 2000 * 3)
    out = torchproc.preprocess_pcg(x, 2000, 4125)
    assert out.shape[0] == 4 and torch.isfinite(out).all()
    windows = torchproc.segment(out, 4125, WindowSpec(window_s=2.0))
    assert windows.dim() == 3 and windows.shape[0] == 4


def test_tensor_and_numpy_preprocess_agree():
    """The batched tensor preprocessing tracks the NumPy preprocessing closely."""
    fs = 2000
    t = np.arange(fs * 4) / fs
    signal = np.sin(2 * np.pi * 90 * t) + 0.5 * np.sin(2 * np.pi * 300 * t)
    numpy_out = preprocess_pcg(signal, fs, 4125)
    tensor_out = torchproc.preprocess_pcg(torch.tensor(signal, dtype=torch.float64), fs, 4125).numpy()
    n = min(len(numpy_out), len(tensor_out))
    corr = np.corrcoef(numpy_out[:n], tensor_out[:n])[0, 1]
    assert corr > 0.999
    assert np.max(np.abs(numpy_out[:n] - tensor_out[:n])) < 5e-3
