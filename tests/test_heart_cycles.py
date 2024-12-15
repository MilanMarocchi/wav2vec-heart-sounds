import json
import random

import numpy as np

from mpcg_wav2vec.datasets import heart_cycles as hc
from mpcg_wav2vec.datasets.generative import GenerativeDataset, GenRecord
from mpcg_wav2vec.signalproc import MelConfig


def _seg_file(tmp_path, joins, fs):
    path = tmp_path / "p0.json"
    path.write_text(json.dumps({"segments": [[j] for j in joins], "last_index": joins[-1], "fs": fs}))
    return str(path)


def test_load_join_indices_resamples(tmp_path):
    path = _seg_file(tmp_path, [0, 500, 1000, 1500], fs=1000)
    joins = hc.load_join_indices(path, fs_out=2000)  # 2x -> doubled indices, drop the 0
    assert joins == [1000, 2000, 3000]


def test_split_and_rebuild_length():
    signal = np.sin(2 * np.pi * 3 * np.arange(4000) / 100.0)
    joins = list(range(0, 4000, 400))
    cycles = hc.split_cycles(signal, joins)
    assert len(cycles) >= 2
    out = hc.rebuild(cycles, target_len=4000, fade_samples=40)
    assert len(out) >= 4000 and np.all(np.isfinite(out))


def test_rearrange_consistent_across_signals():
    cycles = {"ref": [np.full(10, i) for i in range(6)],
              "con": [np.full(10, -i) for i in range(6)]}
    out = hc.rearrange(cycles, prob_contiguous=0.0, random_start=True, rng=random.Random(0))
    # ref[k] value i implies con[k] value -i -> same ordering applied to both
    ref_order = [int(seg[0]) for seg in out["ref"]]
    con_order = [int(seg[0]) for seg in out["con"]]
    assert ref_order == [-v for v in con_order]
    assert sorted(ref_order) == list(range(6))


def test_generative_dataset_uses_segments(tmp_path):
    fs, hop, frames = 4000, 256, 96
    seg = _seg_file(tmp_path, list(range(0, fs * 8, fs // 2)), fs)  # ~0.5s cycles over 8s
    rec = GenRecord(reference=np.random.randn(fs * 8), conditioning=np.random.randn(fs * 8),
                    label=1, patient="p0", segment_path=seg)
    ds = GenerativeDataset([rec], fs=fs, mel=MelConfig(sample_rate=fs, n_fft=1024, hop_length=hop, n_mels=80),
                           crop_frames=frames, hop_length=hop, rearrange_cycles=True)
    item = ds[0]
    assert item["ref_audio"].shape[0] == hop * frames
    assert item["con_spec"].shape[-1] == frames
