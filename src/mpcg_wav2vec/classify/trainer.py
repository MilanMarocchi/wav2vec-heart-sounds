"""Supervised training loop with best-epoch (by validation MCC) checkpointing."""

from __future__ import annotations

import copy

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from .losses import ContrastiveFocalLoss
from .metrics import ConfusionMatrix


def build_optimizer(model: nn.Module, name: str = "sgd", lr: float = 1e-3,
                    weight_decay: float = 1e-5, momentum: float = 0.9, extra_params=None):
    params = [p for p in model.parameters() if p.requires_grad]
    params += [p for p in (extra_params or []) if p.requires_grad]
    if name == "sgd":
        opt = torch.optim.SGD(params, lr=lr, momentum=momentum, weight_decay=weight_decay)
        sched = torch.optim.lr_scheduler.StepLR(opt, step_size=7, gamma=0.1)
        return opt, sched
    if name == "adam":
        return torch.optim.Adam(params, lr=lr, weight_decay=weight_decay), None
    if name == "adamw":
        return torch.optim.AdamW(params, lr=lr, weight_decay=weight_decay), None
    raise ValueError(f"Unknown optimizer '{name}'")


class SupervisedTrainer:
    def __init__(self, model: nn.Module, device: torch.device, optimizer_name: str = "sgd",
                 lr: float = 1e-3, weight_decay: float = 1e-5, batch_transform=None,
                 criterion: nn.Module | None = None, log_dir: str | None = None):
        self.model = model.to(device)
        self.device = device
        # A feature-aware loss (ContrastiveFocalLoss) also has trainable parameters (centres),
        # so it is moved to the device and its parameters join the optimizer.
        self.criterion = (criterion or nn.CrossEntropyLoss()).to(device)
        self.feature_loss = isinstance(self.criterion, ContrastiveFocalLoss)
        extra_params = list(self.criterion.parameters()) if self.feature_loss else []
        self.optimizer, self.scheduler = build_optimizer(model, optimizer_name, lr, weight_decay,
                                                         extra_params=extra_params)
        # Optional on-device augmentation applied to each training batch (e.g. augment.torchaug).
        self.batch_transform = batch_transform
        self.max_grad_norm = 5.0
        self._clip_params = list(model.parameters()) + extra_params
        self.writer = None
        if log_dir:
            from torch.utils.tensorboard.writer import SummaryWriter
            self.writer = SummaryWriter(log_dir)
        self.epoch = 0

    def _run_epoch(self, loader: DataLoader, train: bool, max_batches: int | None,
                   desc: str = "") -> ConfusionMatrix:
        self.model.train(train)
        cm = ConfusionMatrix()
        total = max_batches if max_batches is not None else len(loader)
        phase = "train" if train else "valid"
        bar = tqdm(loader, total=total, leave=False, desc=f"{desc}{phase}", unit="batch")
        running = 0.0
        for i, batch in enumerate(bar):
            if max_batches is not None and i >= max_batches:
                break
            x = batch["waveform"].to(self.device)
            y = batch["label"].to(self.device)
            if train and self.batch_transform is not None:
                x = self.batch_transform(x)
            with torch.set_grad_enabled(train):
                if self.feature_loss:
                    features = self.model.encode(x)
                    logits = self.model.head(features)
                    loss = self.criterion(features, logits, y)
                else:
                    logits = self.model(x)
                    loss = self.criterion(logits, y)
                if train:
                    self.optimizer.zero_grad(set_to_none=True)
                    loss.backward()
                    nn.utils.clip_grad_norm_(self._clip_params, self.max_grad_norm)
                    self.optimizer.step()
            cm.update(y.tolist(), logits.argmax(dim=1).tolist())
            running += float(loss.detach())
            bar.set_postfix(loss=f"{running / (i + 1):.3f}", mcc=f"{cm.stats()['mcc']:.3f}")
        if train and self.scheduler is not None:
            self.scheduler.step()
        return cm

    def fit(self, train_loader: DataLoader, valid_loader: DataLoader | None, epochs: int,
            max_batches: int | None = None, label: str = "") -> float:
        best_mcc = -1.0
        best_state = copy.deepcopy(self.model.state_dict())
        prefix = f"{label} " if label else ""
        epoch_bar = tqdm(range(1, epochs + 1), desc=f"{prefix}epochs", unit="epoch")
        tag = label.strip("[] ").replace(" ", "_") or "run"
        for epoch in epoch_bar:
            self.epoch += 1
            train_cm = self._run_epoch(train_loader, True, max_batches, desc=f"{prefix}e{epoch} ")
            line = f"{prefix}epoch {epoch}: train {train_cm}"
            self._log(f"{tag}/train", train_cm)
            if valid_loader is not None:
                valid_cm = self._run_epoch(valid_loader, False, max_batches, desc=f"{prefix}e{epoch} ")
                mcc = valid_cm.stats()["mcc"]
                line += f" | valid {valid_cm}"
                self._log(f"{tag}/valid", valid_cm)
                if mcc > best_mcc:
                    best_mcc = mcc
                    best_state = copy.deepcopy(self.model.state_dict())
                epoch_bar.set_postfix(valid_mcc=f"{mcc:.3f}", best=f"{best_mcc:.3f}")
            tqdm.write(line)
        if valid_loader is not None:
            self.model.load_state_dict(best_state)
        return best_mcc

    def _log(self, prefix: str, cm: ConfusionMatrix) -> None:
        if self.writer is None:
            return
        for name, value in cm.stats().items():
            self.writer.add_scalar(f"{prefix}/{name}", value, self.epoch)
        self.writer.flush()
