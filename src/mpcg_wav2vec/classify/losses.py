"""Feature-aware classification loss for the multichannel (vest) runs.

Combines a supervised contrastive term over the pooled encoder features, a cross-entropy term over
the logits, and an optional center-loss term that pulls features toward per-class centres. This
matches the objective the vest classifier was trained with; the single-channel / Training-A runs
use plain cross-entropy.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class CenterLoss(nn.Module):
    """Penalise the squared distance of each feature to its (learnable) class centre."""

    def __init__(self, num_classes: int, feature_dim: int):
        super().__init__()
        self.centers = nn.Parameter(torch.randn(num_classes, feature_dim))

    def forward(self, features: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        return ((features - self.centers[labels]) ** 2).sum(dim=1).mean()


def supervised_contrastive(features: torch.Tensor, labels: torch.Tensor,
                           temperature: float = 0.7) -> torch.Tensor:
    """Pull same-class features together / push different-class apart (cosine similarity)."""
    feats = F.normalize(features, dim=1)
    sim = feats @ feats.t() / temperature
    sim = sim - sim.max(dim=1, keepdim=True).values.detach()

    same = labels.unsqueeze(0) == labels.unsqueeze(1)
    self_mask = torch.eye(len(labels), dtype=torch.bool, device=features.device)
    positives = same & ~self_mask

    exp_sim = torch.exp(sim) * (~self_mask)
    log_prob = sim - torch.log(exp_sim.sum(dim=1, keepdim=True) + 1e-8)
    pos_counts = positives.sum(dim=1)
    valid = pos_counts > 0
    if not valid.any():
        return features.new_zeros(())
    mean_log_prob = (log_prob * positives).sum(dim=1)[valid] / pos_counts[valid]
    return -mean_log_prob.mean()


class ContrastiveFocalLoss(nn.Module):
    def __init__(self, num_classes: int, feature_dim: int = 768, *, alpha: float = 0.5,
                 beta: float = 0.2, center_weight: float = 0.01, temperature: float = 0.7,
                 use_center: bool = True):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.center_weight = center_weight
        self.temperature = temperature
        self.center_loss = CenterLoss(num_classes, feature_dim) if use_center else None

    def forward(self, features: torch.Tensor, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        contrastive = supervised_contrastive(features, labels, self.temperature)
        classification = F.cross_entropy(logits, labels)
        total = self.beta * contrastive + self.alpha * classification
        if self.center_loss is not None:
            total = total + self.center_weight * self.center_loss(features, labels)
        return total
