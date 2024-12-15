"""SVM side-classifier over frozen encoder features (vest ablations).

Fits an SVM on the mean-pooled Wav2Vec features (via ``model.encode``) after univariate feature
selection. This probes representation quality independently of the MLP head.
"""

from __future__ import annotations

import numpy as np
import torch
from sklearn.feature_selection import SelectKBest
from sklearn.svm import SVC
from torch.utils.data import DataLoader

from .metrics import ConfusionMatrix


class NeuralSVM:
    def __init__(self, model, device: torch.device, k_best: int = 80):
        self.model = model
        self.device = device
        self.k_best = k_best
        self.selector: SelectKBest | None = None
        self.svm: SVC | None = None

    @torch.no_grad()
    def _features(self, loader: DataLoader):
        self.model.eval()
        feats, labels = [], []
        for batch in loader:
            x = batch["waveform"].to(self.device)
            feats.append(self.model.encode(x).cpu().numpy())
            labels.extend(int(v) for v in batch["label"].tolist())
        return np.concatenate(feats, axis=0), np.asarray(labels)

    def fit(self, loader: DataLoader) -> "NeuralSVM":
        features, labels = self._features(loader)
        k = min(self.k_best, features.shape[1])
        self.selector = SelectKBest(k=k)
        selected = self.selector.fit_transform(features, labels)
        self.svm = SVC()
        self.svm.fit(selected, labels)
        return self

    def evaluate(self, loader: DataLoader) -> dict:
        assert self.svm is not None and self.selector is not None, "call fit() first"
        features, labels = self._features(loader)
        preds = self.svm.predict(self.selector.transform(features))
        cm = ConfusionMatrix()
        cm.update(labels.tolist(), preds.tolist())
        return cm.stats()
