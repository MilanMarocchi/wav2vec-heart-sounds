"""CinC single-channel PCG and Training-A PCG+ECG classifier runner.

Supports three settings via ``mode``:

* ``pcg``      — single-channel PCG classifier (CinC / Training-A channel 0).
* ``ecg``      — single-channel ECG classifier (Training-A channel 1).
* ``pcg_ecg``  — two-branch fusion: train a PCG encoder and an ECG encoder, then fit the fusion
                 classifier on their concatenated features (``big_rnn:2:wav2vec`` topology).

A leave-source-database-out variant (:func:`run_leave_out_db`) trains on several CinC databases
and evaluates on a held-out one.
"""

from __future__ import annotations

import torch

from ..augment import AugmentConfig
from ..classify import (
    SupervisedTrainer,
    Wav2VecClassifier,
    Wav2VecConfig,
    evaluate,
    two_branch_pcg_ecg,
)
from ..config import default_window, get_device
from ..datasets.cinc import build_fragments
from ..datasets.fragments import FragmentDataset
from .common import append_result, make_loader


def _fragment_dataset(fragments, fs, channel):
    return FragmentDataset(fragments, fs=fs, channel=channel)


def run(
    data_dir: str,
    csv_path: str,
    *,
    mode: str = "pcg",
    dataset: str = "training-a",
    fs: int = 4125,
    window_s: float = 4.0,
    epochs: int = 20,
    augment: bool = True,
    augment_num: int = 15,
    random_init: bool = False,
    reference_train_rnn: bool = False,
    augment_config: AugmentConfig | None = None,
    batch_size: int = 64,
    fold: int = 1,
    optimizer: str = "sgd",
    lr: float = 1e-3,
    max_batches: int | None = None,
    results_json: str | None = None,
    log_dir: str | None = None,
    run_label: str = "",
) -> dict:
    from ..signalproc import WindowSpec

    device = get_device()
    cfg = augment_config or AugmentConfig()
    aug_num = augment_num if augment else 0
    # Legacy "reference" RNN regime: train for half the epochs and augment the validation set
    # (with half as many copies) instead of keeping it clean.
    train_epochs = max(1, epochs // 2) if reference_train_rnn else epochs
    valid_aug = (aug_num // 2) if (reference_train_rnn and augment) else 0
    window = WindowSpec(window_s=window_s)
    two_branch = mode == "pcg_ecg"
    load_ecg = mode in ("ecg", "pcg_ecg")

    frags = {
        "train": build_fragments(
            data_dir, csv_path, "train", fs_out=fs, window=window, ecg=load_ecg, fold=fold,
            augment_num=aug_num, augment_config=cfg,
        ),
        "valid": build_fragments(
            data_dir, csv_path, "valid", fs_out=fs, window=window, ecg=load_ecg, fold=fold,
            augment_num=valid_aug, augment_config=cfg,
        ),
        "test": build_fragments(data_dir, csv_path, "test", fs_out=fs, window=window,
                                ecg=load_ecg, fold=fold),
    }

    def branch(channel: int, label: str) -> Wav2VecClassifier:
        model = Wav2VecClassifier(Wav2VecConfig(num_classes=2, num_channels=1,
                                                random_init=random_init, fs=fs))
        valid_channel = 0 if not load_ecg else channel
        train_ds = _fragment_dataset(frags["train"], fs, channel)
        valid_ds = _fragment_dataset(frags["valid"], fs, valid_channel)
        trainer = SupervisedTrainer(model, device, optimizer_name=optimizer, lr=lr, log_dir=log_dir)
        trainer.fit(make_loader(train_ds, batch_size, True),
                    make_loader(valid_ds, batch_size, False), train_epochs, max_batches, label=label)
        return model

    if two_branch:
        pcg_branch = branch(0, label="[1/3 PCG branch]")
        ecg_branch = branch(1, label="[2/3 ECG branch]")
        model = two_branch_pcg_ecg(pcg_branch, ecg_branch).to(device)
        train_ds = _fragment_dataset(frags["train"], fs, -1)
        valid_ds = _fragment_dataset(frags["valid"], fs, -1)
        test_ds = _fragment_dataset(frags["test"], fs, -1)
        SupervisedTrainer(model, device, optimizer_name=optimizer, lr=lr, log_dir=log_dir).fit(
            make_loader(train_ds, batch_size, True),
            make_loader(valid_ds, batch_size, False), train_epochs, max_batches, label="[3/3 fusion]")
        topology = "big_rnn:2:wav2vec"
    else:
        channel = 1 if mode == "ecg" else 0
        model = branch(channel, label=f"[{mode}]").to(device)
        test_ds = _fragment_dataset(frags["test"], fs, channel if load_ecg else 0)
        topology = "wav2vec"

    metrics = evaluate(model, make_loader(test_ds, batch_size, False), device, max_batches)
    record = {
        "mode": mode, "dataset": dataset, "fs": fs, "epochs": epochs, "train_epochs": train_epochs,
        "augment": augment, "augment_num": aug_num, "random_init": random_init,
        "reference_train_rnn": reference_train_rnn, "topology": topology,
        "fold": fold, "run_label": run_label,
        **metrics,
    }
    append_result(results_json, record)
    return record


def run_leave_out_db(
    databases: dict[str, tuple[str, str]],
    holdout: str,
    *,
    fs: int = 4125,
    window_s: float = 4.0,
    epochs: int = 20,
    augment: bool = True,
    random_init: bool = False,
    reference_train_rnn: bool = False,
    augment_config: AugmentConfig | None = None,
    batch_size: int = 64,
    optimizer: str = "sgd",
    lr: float = 1e-3,
    max_batches: int | None = None,
    results_json: str | None = None,
) -> dict:
    """Train single-channel PCG on every database except ``holdout``; evaluate on ``holdout``.

    ``databases`` maps a database name to a ``(data_dir, csv_path)`` pair.
    """
    from ..signalproc import WindowSpec

    device = get_device()
    cfg = augment_config or AugmentConfig()
    window = WindowSpec(window_s=window_s)
    aug_num = 15 if augment else 0
    train_epochs = max(1, epochs // 2) if reference_train_rnn else epochs
    valid_aug = (aug_num // 2) if (reference_train_rnn and augment) else 0

    train_frags, valid_frags = [], []
    for name, (data_dir, csv_path) in databases.items():
        if name == holdout:
            continue
        train_frags += build_fragments(
            data_dir, csv_path, "train", fs_out=fs, window=window,
            augment_num=aug_num, augment_config=cfg,
        )
        valid_frags += build_fragments(
            data_dir, csv_path, "valid", fs_out=fs, window=window,
            augment_num=valid_aug, augment_config=cfg,
        )

    holdout_dir, holdout_csv = databases[holdout]
    test_frags = build_fragments(holdout_dir, holdout_csv, "all", fs_out=fs, window=window)

    model = Wav2VecClassifier(Wav2VecConfig(num_classes=2, num_channels=1,
                                            random_init=random_init, fs=fs)).to(device)
    trainer = SupervisedTrainer(model, device, optimizer_name=optimizer, lr=lr)
    trainer.fit(make_loader(_fragment_dataset(train_frags, fs, 0), batch_size, True),
                make_loader(_fragment_dataset(valid_frags, fs, 0), batch_size, False),
                train_epochs, max_batches)

    metrics = evaluate(model, make_loader(_fragment_dataset(test_frags, fs, 0),
                                          batch_size, False), device, max_batches)
    record = {"mode": "pcg", "leave_out_db": holdout, "fs": fs, "epochs": epochs,
              "train_epochs": train_epochs, "augment": augment, "random_init": random_init,
              "reference_train_rnn": reference_train_rnn, **metrics}
    append_result(results_json, record)
    return record
