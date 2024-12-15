"""Aggregate ablation result JSON files into mean/std tables.

The ``classify-*`` commands append one record per run to a results JSON (metrics nested under
``fragment``/``patient``, or ``mlp``/``svm`` for the vest runs, alongside config fields). This
turns those per-fold records into per-condition mean/std summaries for reporting.
"""

from __future__ import annotations

import json
from pathlib import Path

METRIC_KEYS = ("accuracy", "uar", "sensitivity", "specificity", "npv", "precision", "f1", "mcc")


def load_results(path: str | Path) -> list[dict]:
    data = json.loads(Path(path).read_text())
    return data if isinstance(data, list) else [data]


def flatten_metrics(record: dict, prefix: str = "") -> dict[str, float]:
    """Collect numeric metric leaves as dotted paths, e.g. ``patient.mcc`` or ``mlp.patient.uar``."""
    out: dict[str, float] = {}
    for key, value in record.items():
        path = f"{prefix}{key}"
        if isinstance(value, dict):
            out.update(flatten_metrics(value, prefix=f"{path}."))
        elif key in METRIC_KEYS and isinstance(value, (int, float)):
            out[path] = float(value)
    return out


def group_key(record: dict, group_by: list[str]) -> str:
    parts = []
    for field in group_by:
        if field in record and not isinstance(record[field], dict):
            parts.append(f"{field}={record[field]}")
    return ", ".join(parts) if parts else "all"


def summarize(records: list[dict], group_by: list[str] | None = None) -> dict[str, dict[str, tuple[float, float, int]]]:
    """Return ``{group: {metric_path: (mean, std, n)}}`` aggregated across records."""
    group_by = group_by or ["run_label"]
    groups: dict[str, dict[str, list[float]]] = {}
    for record in records:
        key = group_key(record, group_by)
        metrics = flatten_metrics(record)
        bucket = groups.setdefault(key, {})
        for name, value in metrics.items():
            bucket.setdefault(name, []).append(value)

    summary: dict[str, dict[str, tuple[float, float, int]]] = {}
    for key, metrics in groups.items():
        summary[key] = {}
        for name, values in metrics.items():
            arr = _mean_std(values)
            summary[key][name] = (arr[0], arr[1], len(values))
    return summary


def _mean_std(values: list[float]) -> tuple[float, float]:
    n = len(values)
    mean = sum(values) / n
    var = sum((v - mean) ** 2 for v in values) / n if n else 0.0
    return mean, var ** 0.5


def to_markdown(summary: dict, metrics: list[str] | None = None) -> str:
    """Render a summary as a Markdown table (mean±std). ``metrics`` selects/orders columns."""
    all_metrics = sorted({m for group in summary.values() for m in group})
    if metrics:
        all_metrics = [m for m in all_metrics if any(m == sel or m.endswith("." + sel) for sel in metrics)]
    header = "| condition | n | " + " | ".join(all_metrics) + " |"
    sep = "|" + "---|" * (len(all_metrics) + 2)
    lines = [header, sep]
    for key in sorted(summary):
        n = max((v[2] for v in summary[key].values()), default=0)
        cells = []
        for m in all_metrics:
            if m in summary[key]:
                mean, std, _ = summary[key][m]
                cells.append(f"{mean:.4f}±{std:.4f}")
            else:
                cells.append("-")
        lines.append(f"| {key} | {n} | " + " | ".join(cells) + " |")
    return "\n".join(lines)
