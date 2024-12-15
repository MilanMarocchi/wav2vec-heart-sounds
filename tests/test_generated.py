import numpy as np
from scipy.io import wavfile

from mpcg_wav2vec.datasets.generated import generated_fragments
from mpcg_wav2vec.signalproc import WindowSpec


def _write_generated(tmp_path, n=4, fs=4125, seconds=4.0):
    manifest = tmp_path / "REFERENCE.csv"
    lines = ["patient,label,file"]
    for i in range(n):
        wave = (np.random.randn(int(fs * seconds)) * 0.1).astype("float32")
        name = f"p{i % 2}_{i}.wav"
        wavfile.write(str(tmp_path / name), fs, wave)
        lines.append(f"p{i % 2},{i % 2},{name}")
    manifest.write_text("\n".join(lines))
    return tmp_path


def test_generated_fragments_roundtrip(tmp_path):
    _write_generated(tmp_path)
    frags = generated_fragments(str(tmp_path), fs_out=4125, window=WindowSpec(window_s=2.0))
    assert len(frags) > 0
    assert {f.label for f in frags} <= {0, 1}
    assert frags[0].waveform.shape[0] == int(2.0 * 4125)


def test_generated_fragments_proportion(tmp_path):
    _write_generated(tmp_path, n=10)
    full = generated_fragments(str(tmp_path), fs_out=4125, window=WindowSpec(window_s=2.0))
    half = generated_fragments(str(tmp_path), fs_out=4125, window=WindowSpec(window_s=2.0), proportion=0.5)
    assert len(half) <= len(full)
