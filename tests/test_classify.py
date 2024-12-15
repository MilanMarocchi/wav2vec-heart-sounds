import os

import torch

from mpcg_wav2vec.classify import ConfusionMatrix, TimeVaryingSincBeamformer

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")


def test_confusion_matrix_metrics():
    cm = ConfusionMatrix()
    cm.update([0, 1, 1, 0, 1], [0, 1, 0, 0, 1])
    stats = cm.stats()
    assert abs(stats["accuracy"] - 0.8) < 1e-9
    assert 0.0 <= stats["uar"] <= 1.0
    assert -1.0 <= stats["mcc"] <= 1.0


def test_beamformer_collapses_channels():
    bf = TimeVaryingSincBeamformer(6, fs=4125)
    out = bf(torch.randn(2, 6, 1500))
    assert out.shape == (2, 1500)


def test_wav2vec_forward_shapes():
    pytest = __import__("pytest")
    try:
        from mpcg_wav2vec.classify import Wav2VecClassifier, Wav2VecConfig
        model = Wav2VecClassifier(Wav2VecConfig(num_channels=1, fs=4125))
    except Exception as exc:  # pragma: no cover - only when weights unavailable offline
        pytest.skip(f"wav2vec2 weights unavailable: {exc}")
    logits = model(torch.randn(2, 4125))
    assert logits.shape == (2, 2)
