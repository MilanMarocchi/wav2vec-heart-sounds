"""Fragment-level and patient-level evaluation.

Fragment-level scores every window independently. Patient-level aggregates each patient's
fragment logits (mean softmax) into a single prediction, matching the paper's patient-level
reporting.
"""

from __future__ import annotations

from collections import defaultdict

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from .metrics import ConfusionMatrix


@torch.no_grad()
def evaluate(model, loader: DataLoader, device: torch.device, max_batches: int | None = None,
             desc: str = "Evaluating") -> dict:
    model.eval()
    fragment_cm = ConfusionMatrix()
    patient_logits: dict[str, list[torch.Tensor]] = defaultdict(list)
    patient_true: dict[str, int] = {}

    total = max_batches if max_batches is not None else len(loader)
    for i, batch in enumerate(tqdm(loader, total=total, desc=desc, unit="batch", leave=False)):
        if max_batches is not None and i >= max_batches:
            break
        x = batch["waveform"].to(device)
        y = batch["label"]
        logits = model(x).cpu()
        fragment_cm.update(y.tolist(), logits.argmax(dim=1).tolist())
        for j, patient in enumerate(batch["patient"]):
            patient_logits[patient].append(logits[j])
            patient_true[patient] = int(y[j])

    patient_cm = ConfusionMatrix()
    for patient, logit_list in patient_logits.items():
        mean_prob = F.softmax(torch.stack(logit_list).mean(dim=0), dim=0)
        patient_cm.update([patient_true[patient]], [int(mean_prob.argmax())])

    return {"fragment": fragment_cm.stats(), "patient": patient_cm.stats()}
