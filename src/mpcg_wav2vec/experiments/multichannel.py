"""Multichannel vest PCG ablation runner (single Wav2Vec + sinc beamformer)."""

from __future__ import annotations

from ..augment import AugmentConfig
from ..classify import (
    ContrastiveFocalLoss,
    NeuralSVM,
    SupervisedTrainer,
    Wav2VecClassifier,
    Wav2VecConfig,
    evaluate,
)
from ..config import get_device
from ..datasets.vest import vest_dataset
from ..signalproc import WindowSpec
from .common import append_result, make_loader


def run(
    data_dir: str,
    csv_path: str,
    *,
    channels: list[int] | None = None,
    fs: int = 4125,
    window_s: float = 2.0,
    epochs: int = 20,
    augment: bool = True,
    random_init: bool = False,
    lora: bool = True,
    freeze_encoder: bool = False,
    fit_svm: bool = True,
    loss: str = "ce",
    augment_config: AugmentConfig | None = None,
    batch_size: int = 16,
    fold: int = 1,
    optimizer: str = "adamw",
    lr: float = 1e-4,
    max_batches: int | None = None,
    results_json: str | None = None,
    log_dir: str | None = None,
    run_label: str = "",
) -> dict:
    device = get_device()
    channels = channels or [1, 2, 3, 4, 5, 6]
    cfg = augment_config or AugmentConfig()
    window = WindowSpec(window_s=window_s)
    aug_num = 15 if augment else 0

    model = Wav2VecClassifier(Wav2VecConfig(
        num_classes=2, num_channels=len(channels), random_init=random_init,
        lora=lora and not random_init, freeze_encoder=freeze_encoder, fs=fs,
    )).to(device)

    def dataset(subset, augment_num):
        return vest_dataset(data_dir, csv_path, subset, fs_out=fs, window=window,
                            channels=channels, fold=fold, augment_num=augment_num,
                            augment_config=cfg)

    train_ds = dataset("train", aug_num)
    valid_ds = dataset("valid", 0)
    test_ds = dataset("test", 0)

    criterion = ContrastiveFocalLoss(num_classes=2) if loss == "contrastive-focal" else None
    trainer = SupervisedTrainer(model, device, optimizer_name=optimizer, lr=lr,
                                criterion=criterion, log_dir=log_dir)
    trainer.fit(make_loader(train_ds, batch_size, True),
                make_loader(valid_ds, batch_size, False), epochs, max_batches)

    metrics = {"mlp": evaluate(model, make_loader(test_ds, batch_size, False), device, max_batches)}
    if fit_svm:
        svm = NeuralSVM(model, device).fit(make_loader(train_ds, batch_size, False))
        metrics["svm"] = svm.evaluate(make_loader(test_ds, batch_size, False))

    record = {
        "channels": channels, "fs": fs, "epochs": epochs, "augment": augment,
        "random_init": random_init, "lora": lora, "freeze_encoder": freeze_encoder,
        "loss": loss, "fold": fold, "run_label": run_label, **metrics,
    }
    append_result(results_json, record)
    return record
