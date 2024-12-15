"""Wav2Vec 2.0 classifier for heart sounds.

One model covers every configuration the paper exercises:

* pretrained encoder (``facebook/wav2vec2-base-960h``) or random init (isolates pre-training),
* full fine-tune, frozen encoder, or LoRA adapters (``q_proj``/``v_proj``),
* single-channel input, or multichannel input collapsed to one channel by the time-varying sinc
  beamformer before the encoder.

The mean-pooled encoder output feeds a small MLP head. Forward returns raw class logits.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch
import torch.nn as nn
from transformers import Wav2Vec2Config, Wav2Vec2Model

from .beamformer import TimeVaryingSincBeamformer

_HIDDEN = 768  # wav2vec2-base hidden size


@dataclass
class Wav2VecConfig:
    num_classes: int = 2
    num_channels: int = 1
    head_hidden: tuple[int, ...] = (256,)
    pretrained_name: str = "facebook/wav2vec2-base-960h"
    random_init: bool = False
    lora: bool = False
    freeze_encoder: bool = False
    fs: int = 4125


def _build_head(hidden: tuple[int, ...], num_classes: int) -> nn.Sequential:
    layers: list[nn.Module] = []
    prev = _HIDDEN
    for h in hidden:
        layers += [nn.Linear(prev, h), nn.ReLU()]
        prev = h
    layers.append(nn.Linear(prev, num_classes))
    return nn.Sequential(*layers)


def _build_encoder(cfg: Wav2VecConfig) -> nn.Module:
    if cfg.random_init:
        return Wav2Vec2Model(Wav2Vec2Config.from_pretrained(cfg.pretrained_name))
    return Wav2Vec2Model.from_pretrained(cfg.pretrained_name)


def _apply_lora(encoder: nn.Module) -> nn.Module:
    from peft import LoraConfig, get_peft_model
    lora_cfg = LoraConfig(r=8, lora_alpha=16, target_modules=["q_proj", "v_proj"], lora_dropout=0.05)
    return get_peft_model(encoder, lora_cfg)


class Wav2VecClassifier(nn.Module):
    def __init__(self, config: Wav2VecConfig):
        super().__init__()
        self.config = config
        self.encoder = _build_encoder(config)

        if config.freeze_encoder:
            for p in self.encoder.parameters():
                p.requires_grad = False
        elif config.lora:
            self.encoder = _apply_lora(self.encoder)

        self.channel_mixer = (
            TimeVaryingSincBeamformer(config.num_channels, config.fs)
            if config.num_channels > 1 else None
        )
        self.head = _build_head(config.head_hidden, config.num_classes)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Return mean-pooled encoder features for a waveform batch."""
        if x.dim() == 3:                       # [B, T, C] -> [B, C, T]
            x = x.transpose(1, 2)
        if self.channel_mixer is not None:
            x = self.channel_mixer(x)          # [B, C, T] -> [B, T]
        elif x.dim() == 3:
            x = x.squeeze(1) if x.shape[1] == 1 else x.mean(dim=1)
        features = self.encoder(x).last_hidden_state   # [B, T', 768]
        return features.mean(dim=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.encode(x))
