import numpy as np
from scipy.io import wavfile

from mpcg_wav2vec.datasets.vest import VEST_CHANNEL_MAP, read_vest_wav, vest_dataset
from mpcg_wav2vec.signalproc import WindowSpec


def _make_vest_dataset(tmp_path, patients=("v0001", "v0002"), fs=4000, seconds=8, n_channels=9):
    data_dir = tmp_path / "vest"
    data_dir.mkdir()
    for i, p in enumerate(patients):
        # int16 multichannel WAV in the processed layout [T, C]
        sig = (np.random.randn(fs * seconds, n_channels) * 3000).astype(np.int16)
        wavfile.write(str(data_dir / f"{p}_rec01.wav"), fs, sig)
    csv = tmp_path / "vest.csv"
    lines = ["patient,label,split"]
    for i, p in enumerate(patients):
        lines.append(f"{p},{i % 2},train")
    csv.write_text("\n".join(lines))
    return str(data_dir), str(csv)


def test_read_vest_wav_normalises_int(tmp_path):
    sig = (np.random.randn(1000, 9) * 3000).astype(np.int16)
    path = tmp_path / "x.wav"
    wavfile.write(str(path), 4000, sig)
    out, fs = read_vest_wav(str(path))
    assert fs == 4000 and out.shape == (1000, 9)
    assert out.dtype == np.float32 and np.abs(out).max() <= 1.0


def test_vest_dataset_builds_multichannel_fragments(tmp_path):
    data_dir, csv = _make_vest_dataset(tmp_path)
    ds = vest_dataset(data_dir, csv, "train", fs_out=4125, window=WindowSpec(window_s=2.0),
                      channels=[1, 2, 3, 4, 5, 6])
    assert len(ds) > 0
    item = ds[0]
    wave = item["waveform"]
    assert wave.dim() == 2 and wave.shape[1] == 6      # [T, 6 channels]
    assert item["label"] in (0, 1)


def test_channel_map_matches_processed_layout():
    assert [VEST_CHANNEL_MAP[c] for c in (1, 2, 3, 4, 5, 6, 7)] == [0, 1, 2, 3, 4, 5, 6]
    assert VEST_CHANNEL_MAP["E"] == 7
