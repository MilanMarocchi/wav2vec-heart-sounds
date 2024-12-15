import torch

from mpcg_wav2vec.classify import CenterLoss, ContrastiveFocalLoss
from mpcg_wav2vec.classify.losses import supervised_contrastive


def test_center_loss_positive_and_differentiable():
    feats = torch.randn(8, 16, requires_grad=True)
    labels = torch.randint(0, 2, (8,))
    loss = CenterLoss(2, 16)(feats, labels)
    assert loss.item() >= 0
    loss.backward()
    assert feats.grad is not None


def test_supervised_contrastive_lower_when_separated():
    labels = torch.tensor([0, 0, 1, 1])
    separated = torch.tensor([[5.0, 0.0], [5.0, 0.1], [-5.0, 0.0], [-5.0, 0.1]])
    mixed = torch.tensor([[1.0, 0.0], [-1.0, 0.0], [1.0, 0.1], [-1.0, 0.1]])
    assert supervised_contrastive(separated, labels) < supervised_contrastive(mixed, labels)


def test_contrastive_focal_forward():
    loss_fn = ContrastiveFocalLoss(num_classes=2, feature_dim=16)
    feats = torch.randn(6, 16)
    logits = torch.randn(6, 2)
    labels = torch.randint(0, 2, (6,))
    loss = loss_fn(feats, logits, labels)
    assert torch.isfinite(loss) and loss.item() >= 0
    # the loss carries trainable centre parameters
    assert any(p.requires_grad for p in loss_fn.parameters())
