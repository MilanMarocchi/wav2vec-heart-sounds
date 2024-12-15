"""Reverse-diffusion samplers for DiffWave and WaveGrad.

Both return ``(audio[B, T], sample_rate)`` where ``T = hop_length * mel_frames``. DiffWave uses
the standard fast-sampling schedule alignment; WaveGrad supports optional step sub-sampling for
a matching fast path.
"""

from __future__ import annotations

import numpy as np
import torch

from .schedules import NoiseSchedule


def _prepare(conditioner: torch.Tensor, label, device):
    if conditioner.dim() == 2:
        conditioner = conditioner.unsqueeze(0)
    conditioner = conditioner.to(device)
    label = torch.as_tensor(label, device=device).reshape(-1)
    if label.numel() == 1 and conditioner.shape[0] > 1:
        label = label.expand(conditioner.shape[0])
    return conditioner, label


def _align_fast_steps(train_sched: NoiseSchedule, infer_sched: NoiseSchedule) -> np.ndarray:
    """Map each inference step onto a (fractional) training step by matching alpha_cumprod."""
    train_cum = train_sched.alpha_cumprod
    infer_cum = infer_sched.alpha_cumprod
    steps = []
    for s in range(len(infer_cum)):
        for t in range(len(train_cum) - 1):
            if train_cum[t + 1] <= infer_cum[s] <= train_cum[t]:
                frac = ((train_cum[t] ** 0.5 - infer_cum[s] ** 0.5)
                        / (train_cum[t] ** 0.5 - train_cum[t + 1] ** 0.5))
                steps.append(t + frac)
                break
    return np.asarray(steps, dtype=np.float32)


@torch.no_grad()
def diffwave_sample(model, conditioner: torch.Tensor, label, *, fast: bool = True):
    cfg = model.config
    device = next(model.parameters()).device
    conditioner, label = _prepare(conditioner, label, device)

    train_sched = cfg.training_schedule()
    infer_sched = NoiseSchedule(np.asarray(cfg.inference_betas, dtype=np.float64)) if fast else train_sched
    mapped_steps = _align_fast_steps(train_sched, infer_sched) if fast else np.arange(len(train_sched), dtype=np.float32)

    beta = infer_sched.betas
    alpha = infer_sched.alphas
    alpha_cum = infer_sched.alpha_cumprod

    n_samples = cfg.hop_length * conditioner.shape[-1]
    audio = torch.randn(conditioner.shape[0], n_samples, device=device)

    for n in range(len(alpha) - 1, -1, -1):
        step = torch.full((conditioner.shape[0],), float(mapped_steps[n]), device=device)
        predicted = model(audio, step, conditioner, label).squeeze(1)
        audio = (audio - beta[n] / (1 - alpha_cum[n]) ** 0.5 * predicted) / alpha[n] ** 0.5
        if n > 0:
            sigma = ((1.0 - alpha_cum[n - 1]) / (1.0 - alpha_cum[n]) * beta[n]) ** 0.5
            audio = audio + sigma * torch.randn_like(audio)
        audio = audio.clamp(-1.0, 1.0)
    return audio, cfg.sample_rate


@torch.no_grad()
def wavegrad_sample(model, conditioner: torch.Tensor, label, *, num_steps: int | None = None):
    cfg = model.config
    device = next(model.parameters()).device
    conditioner, label = _prepare(conditioner, label, device)

    sched = cfg.training_schedule()
    beta = sched.betas
    alpha = sched.alphas
    alpha_cum = sched.alpha_cumprod
    noise_scale = np.sqrt(alpha_cum)

    order = range(len(alpha) - 1, -1, -1)
    if num_steps is not None and num_steps < len(alpha):
        order = np.unique(np.linspace(0, len(alpha) - 1, num_steps).round().astype(int))[::-1]

    n_samples = cfg.hop_length * conditioner.shape[-1]
    audio = torch.randn(conditioner.shape[0], n_samples, device=device)

    for n in order:
        level = torch.full((conditioner.shape[0],), float(noise_scale[n]), device=device)
        predicted = model(audio, conditioner, level, label).squeeze(1)
        c2 = (1 - alpha[n]) / (1 - alpha_cum[n]) ** 0.5
        audio = (audio - c2 * predicted) / alpha[n] ** 0.5
        if n > 0:
            sigma = ((1.0 - alpha_cum[n - 1]) / (1.0 - alpha_cum[n]) * beta[n]) ** 0.5
            audio = audio + sigma * torch.randn_like(audio)
        audio = audio.clamp(-1.0, 1.0)
    return audio, cfg.sample_rate
