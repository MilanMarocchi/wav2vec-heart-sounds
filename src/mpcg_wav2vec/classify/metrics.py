"""Binary classification metrics.

One confusion-matrix accumulator and the standard MCC. Reported metrics match the paper:
accuracy, UAR (mean of sensitivity/specificity), sensitivity, specificity, NPV, precision, F1,
MCC.
"""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass
class ConfusionMatrix:
    tp: int = 0
    tn: int = 0
    fp: int = 0
    fn: int = 0

    def update(self, y_true, y_pred) -> None:
        for t, p in zip(y_true, y_pred):
            t, p = int(t), int(p)
            if t == 1 and p == 1:
                self.tp += 1
            elif t == 0 and p == 0:
                self.tn += 1
            elif t == 0 and p == 1:
                self.fp += 1
            else:
                self.fn += 1

    @property
    def total(self) -> int:
        return self.tp + self.tn + self.fp + self.fn

    def _safe(self, num, den) -> float:
        return num / den if den else 0.0

    def stats(self) -> dict[str, float]:
        sens = self._safe(self.tp, self.tp + self.fn)
        spec = self._safe(self.tn, self.tn + self.fp)
        ppv = self._safe(self.tp, self.tp + self.fp)
        npv = self._safe(self.tn, self.tn + self.fn)
        f1 = self._safe(2 * ppv * sens, ppv + sens)
        denom = math.sqrt((self.tp + self.fp) * (self.tp + self.fn)
                          * (self.tn + self.fp) * (self.tn + self.fn))
        mcc = (self.tp * self.tn - self.fp * self.fn) / denom if denom else 0.0
        return {
            "accuracy": self._safe(self.tp + self.tn, self.total),
            "uar": 0.5 * (sens + spec),
            "sensitivity": sens,
            "specificity": spec,
            "npv": npv,
            "precision": ppv,
            "f1": f1,
            "mcc": mcc,
        }

    def __str__(self) -> str:
        s = self.stats()
        return (f"acc={s['accuracy']:.4f} uar={s['uar']:.4f} sens={s['sensitivity']:.4f} "
                f"spec={s['specificity']:.4f} mcc={s['mcc']:.4f}")
