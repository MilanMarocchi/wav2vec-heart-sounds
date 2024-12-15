"""Training loop for the diffusion generators.

Both models are trained by epsilon-prediction with an L1 loss; only how the noisy input and the
model call are formed differs, so that part is a small per-model *loss strategy* callable and
everything else (AMP, grad clipping, checkpointing, validation) is shared.
"""

from __future__ import annotations

import os
from typing import Callable

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

LossStrategy = Callable[[nn.Module, dict, torch.device], torch.Tensor]


def diffwave_loss(model: nn.Module, batch: dict, device: torch.device) -> torch.Tensor:
    ref = batch["ref_audio"].to(device)
    con = batch["con_spec"].to(device)
    label = batch["label"].to(device)
    levels = torch.tensor(model.config.training_schedule().training_noise_levels(),
                          dtype=torch.float32, device=device)
    t = torch.randint(0, len(levels), (ref.shape[0],), device=device)
    noise_scale = levels[t].unsqueeze(1)
    noise = torch.randn_like(ref)
    noisy = noise_scale.sqrt() * ref + (1.0 - noise_scale).sqrt() * noise
    predicted = model(noisy, t, con, label).squeeze(1)
    return nn.functional.l1_loss(predicted, noise)


def wavegrad_loss(model: nn.Module, batch: dict, device: torch.device) -> torch.Tensor:
    ref = batch["ref_audio"].to(device)
    con = batch["con_spec"].to(device)
    label = batch["label"].to(device)
    levels = torch.tensor(model.config.training_schedule().continuous_noise_levels(),
                          dtype=torch.float32, device=device)
    steps = len(levels) - 1
    s = torch.randint(1, steps + 1, (ref.shape[0],), device=device)
    lo, hi = levels[s - 1], levels[s]
    noise_scale = (lo + torch.rand(ref.shape[0], device=device) * (hi - lo)).unsqueeze(1)
    noise = torch.randn_like(ref)
    noisy = noise_scale * ref + (1.0 - noise_scale ** 2).sqrt() * noise
    predicted = model(noisy, con, noise_scale.squeeze(1), label).squeeze(1)
    return nn.functional.l1_loss(predicted, noise)


class GenerativeTrainer:
    def __init__(self, model: nn.Module, optimizer, loss_strategy: LossStrategy,
                 device: torch.device, model_dir: str, *, fp16: bool = False,
                 max_grad_norm: float | None = 1.0, log_dir: str | None = None,
                 sampler=None, sample_every: int = 10):
        self.model = model
        self.optimizer = optimizer
        self.loss_strategy = loss_strategy
        self.device = device
        self.model_dir = model_dir
        self.max_grad_norm = max_grad_norm
        self.scaler = torch.amp.GradScaler("cuda", enabled=fp16)
        self.fp16 = fp16
        self.step = 0
        self.best_valid = float("inf")
        # Optional TensorBoard logging + periodic generated-sample audio/mel.
        self.sampler = sampler
        self.sample_every = sample_every
        self.writer = None
        if log_dir:
            from torch.utils.tensorboard.writer import SummaryWriter
            self.writer = SummaryWriter(log_dir)
        os.makedirs(model_dir, exist_ok=True)

    def _autocast(self):
        return torch.amp.autocast("cuda", enabled=self.fp16)

    def train_step(self, batch: dict) -> float:
        self.optimizer.zero_grad(set_to_none=True)
        with self._autocast():
            loss = self.loss_strategy(self.model, batch, self.device)
        self.scaler.scale(loss).backward()
        if self.max_grad_norm is not None:
            self.scaler.unscale_(self.optimizer)
            nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
        self.scaler.step(self.optimizer)
        self.scaler.update()
        self.step += 1
        return float(loss.detach())

    @torch.no_grad()
    def validate(self, loader: DataLoader, max_batches: int | None = None) -> float:
        self.model.eval()
        total, count = 0.0, 0
        for i, batch in enumerate(loader):
            total += float(self.loss_strategy(self.model, batch, self.device))
            count += 1
            if max_batches is not None and i + 1 >= max_batches:
                break
        self.model.train()
        return total / max(1, count)

    def train(self, train_loader: DataLoader, epochs: int, valid_loader: DataLoader | None = None,
              max_train_batches: int | None = None):
        self.model.train()
        model_name = type(self.model).__name__
        total = max_train_batches if max_train_batches is not None else len(train_loader)
        # Fix one batch as the conditioner for periodic sample logging.
        self._sample_batch = next(iter(train_loader)) if (self.writer and self.sampler) else None
        for epoch in range(1, epochs + 1):
            running = 0.0
            n = 0
            bar = tqdm(train_loader, total=total, desc=f"{model_name} e{epoch}/{epochs}",
                       unit="batch", leave=False)
            for i, batch in enumerate(bar):
                loss = self.train_step(batch)
                if not np.isfinite(loss):
                    raise RuntimeError(f"non-finite loss at step {self.step}")
                running += loss
                n += 1
                bar.set_postfix(L1=f"{running / n:.4f}")
                if max_train_batches is not None and i + 1 >= max_train_batches:
                    break
            train_loss = running / max(1, n)
            msg = f"{model_name} epoch {epoch}: train L1={train_loss:.4f}"
            if self.writer is not None:
                self.writer.add_scalar("gen/train_L1", train_loss, epoch)
            if valid_loader is not None:
                valid_loss = self.validate(valid_loader, max_train_batches)
                msg += f" valid L1={valid_loss:.4f}"
                if self.writer is not None:
                    self.writer.add_scalar("gen/valid_L1", valid_loss, epoch)
                if valid_loss < self.best_valid:
                    self.best_valid = valid_loss
                    self.save("weights-best")
            self._log_sample(epoch)
            tqdm.write(msg)
            self.save("weights")

    @torch.no_grad()
    def _log_sample(self, epoch: int) -> None:
        """Generate one clip from a fixed conditioner and log its audio + mel to TensorBoard."""
        if self.writer is None or self.sampler is None or self._sample_batch is None:
            return
        if epoch % self.sample_every != 0:
            return
        cond = self._sample_batch["con_spec"][:1].to(self.device)
        label = self._sample_batch["label"][:1].to(self.device)
        audio, sr = self.sampler(self.model, cond[0], label)
        wave = audio.squeeze(0).float().cpu()
        peak = wave.abs().max().clamp_min(1e-6)
        self.writer.add_audio("gen/sample", (wave / peak).unsqueeze(0), epoch, sample_rate=sr)
        self.writer.add_image("gen/con_spec", cond[0].cpu().unsqueeze(0), epoch)
        self.writer.flush()

    def save(self, name: str) -> str:
        path = os.path.join(self.model_dir, f"{name}.pt")
        torch.save({"step": self.step, "model": self.model.state_dict(),
                    "optimizer": self.optimizer.state_dict(),
                    "config": self.model.config}, path)
        return path

    def restore(self, path: str) -> bool:
        if not path or not os.path.exists(path):
            return False
        ckpt = torch.load(path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(ckpt["model"])
        self.optimizer.load_state_dict(ckpt["optimizer"])
        self.step = ckpt.get("step", 0)
        return True
