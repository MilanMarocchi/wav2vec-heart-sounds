"""Synthetic-augmentation schedule runner (single-channel PCG).

Trains a Wav2Vec classifier through a staged schedule that interleaves real CinC data with
synthetic WaveGrad / DiffWave data, as described in the paper. The schedule JSON (see
``data/gen_config_*.json``) lists real and generated datasets with per-stage epoch counts,
``augment_num`` augmentation multipliers and ``proportion`` subsampling; stages are trained in
order and the best validation-MCC checkpoint is retained.
"""

from __future__ import annotations

import torch

from ..augment import AugmentConfig
from ..classify import SupervisedTrainer, Wav2VecClassifier, Wav2VecConfig, evaluate
from ..config import get_device
from ..datasets.cinc import _pcg_augment, build_fragments
from ..datasets.fragments import FragmentDataset
from ..datasets.generated import generated_fragments
from ..datasets.schedule import Schedule, load_schedule
from ..signalproc import WindowSpec
from functools import partial
from .common import append_result, make_loader


def _stage_fragments(spec, fs, window, proportion):
    if isinstance(spec.gen_data, bool) and spec.gen_data:
        return generated_fragments(spec.path, fs_out=fs, window=window, proportion=proportion)
    return build_fragments(spec.path, spec.split, "train", fs_out=fs, window=window, ecg=False)


def run(
    schedule_path: str,
    *,
    fs: int = 4125,
    window_s: float = 4.0,
    random_init: bool = False,
    augment_config: AugmentConfig | None = None,
    batch_size: int = 64,
    optimizer: str = "sgd",
    lr: float = 1e-3,
    max_batches: int | None = None,
    results_json: str | None = None,
    log_dir: str | None = None,
    run_label: str = "",
) -> dict:
    device = get_device()
    schedule: Schedule = load_schedule(schedule_path)
    cfg = augment_config or AugmentConfig()
    window = WindowSpec(window_s=window_s)
    augment_fn = partial(_pcg_augment, cfg=cfg)

    valid_frags = build_fragments(schedule.valid_set.data, schedule.valid_set.split, "valid",
                                  fs_out=fs, window=window, ecg=False)
    test_frags = build_fragments(schedule.test_set.data, schedule.test_set.split, "test",
                                 fs_out=fs, window=window, ecg=False)
    valid_ds = FragmentDataset(valid_frags, fs=fs, augment_num=0)
    test_ds = FragmentDataset(test_frags, fs=fs, augment_num=0)

    model = Wav2VecClassifier(Wav2VecConfig(num_classes=2, num_channels=1,
                                            random_init=random_init, fs=fs)).to(device)
    trainer = SupervisedTrainer(model, device, optimizer_name=optimizer, lr=lr, log_dir=log_dir)

    for spec, epochs, _letskip in schedule.resolved_stages():
        frags = _stage_fragments(spec, fs, window, float(spec.proportion))
        stage_ds = FragmentDataset(frags, fs=fs, augment_num=spec.augment_num, augment_fn=augment_fn)
        trainer.fit(make_loader(stage_ds, batch_size, True),
                    make_loader(valid_ds, batch_size, False), epochs, max_batches)

    metrics = evaluate(model, make_loader(test_ds, batch_size, False), device, max_batches)
    record = {"schedule": schedule_path, "fs": fs, "random_init": random_init,
              "run_label": run_label, **metrics}
    append_result(results_json, record)
    return record
