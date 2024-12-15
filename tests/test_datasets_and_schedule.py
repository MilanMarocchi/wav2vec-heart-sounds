from pathlib import Path

import numpy as np

from mpcg_wav2vec import augment as A
from mpcg_wav2vec.datasets import labels, load_schedule
from mpcg_wav2vec.datasets.cinc import pad_collate
from mpcg_wav2vec.datasets.fragments import Fragment, FragmentDataset

DATA = Path(__file__).resolve().parents[1] / "data"


def test_schedule_parses_paper_config():
    sched = load_schedule(DATA / "gen_config_rnn_paper_training_a.json")
    keys = [s.key for s in sched.stages]
    assert keys == ["data1", "data2", "data1", "data3", "data1"]
    assert sched.datasets["data2"].proportion == 0.3
    assert sched.datasets["data2"].gen_data is True
    assert sched.stages[0].epochs == 10


def test_fragment_dataset_balances_and_collates():
    fs = 4125
    frags = [Fragment(np.random.randn(fs).astype(np.float32), lbl, f"p{lbl}_{i}")
             for i in range(3) for lbl in (0, 1)]
    ds = FragmentDataset(frags, fs=fs, augment_num=2,
                         augment_fn=lambda w, f: A.augment_pcg(w, f, A.AugmentConfig(prob_hpss=0, prob_real_noise=0)))
    assert len(ds) > len(frags)
    batch = pad_collate([ds[0], ds[1], ds[2]])
    assert batch["waveform"].shape[0] == 3
    assert batch["label"].shape[0] == 3


def test_fragment_dataset_can_cache_augmented_items():
    fs = 8
    calls = {"count": 0}
    frag = Fragment(np.arange(fs, dtype=np.float32), 0, "p0")

    def augment(wave, _fs):
        calls["count"] += 1
        return wave + calls["count"]

    ds = FragmentDataset([frag], fs=fs, augment_num=1, augment_fn=augment, cache_augmented=True)
    first = ds[1]["waveform"].clone()
    second = ds[1]["waveform"].clone()

    assert calls["count"] == 1
    assert np.allclose(first.numpy(), second.numpy())


def test_label_vocabularies():
    assert labels.num_classes("training-a-extended") == 5
    assert labels.label_to_index("training-a", 1) == 1
    assert labels.balanced_sampler([0, 0, 1]) is not None
