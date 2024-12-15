"""Heart-cycle rearrangement for generator training.

Reads a per-recording segmentation index file, cuts each signal at the cardiac-cycle joins, and
rebuilds a training signal by either rotating the cycles (contiguous) or shuffling groups of
cycles and crossfading them back together. This diversifies the generator's training targets
while preserving realistic cycle morphology, and keeps every signal (reference, conditioning)
aligned by cutting them all at the same joins.

Segmentation file format (JSON): ``{"segments": [[i0, i1, ...], ...], "last_index": int,
"fs": int}`` where each group's first index marks a cycle boundary at the segmentation rate.
"""

from __future__ import annotations

import json
import random
from pathlib import Path

import numpy as np


def load_join_indices(seg_path: str | Path, fs_out: float) -> list[int]:
    """Return sorted cardiac-cycle cut points (in ``fs_out`` samples) from a segmentation file."""
    data = json.loads(Path(seg_path).read_text())
    groups, fs_seg = data["segments"], data["fs"]
    joins = sorted({int(g[0]) for g in groups if len(g) and g[0] > 0})
    if fs_out != fs_seg:
        joins = [round(j * fs_out / fs_seg) for j in joins]
    return joins


def split_cycles(signal: np.ndarray, joins: list[int]) -> list[np.ndarray]:
    """Split ``signal`` into segments between consecutive join points."""
    joins = [j for j in joins if 0 < j < len(signal)]
    return [signal[a:b] for a, b in zip(joins[:-1], joins[1:]) if b > a]


def _crossfade(a: np.ndarray, b: np.ndarray, n: int) -> np.ndarray:
    """Correlation-aware crossfade join of two segments over ``n`` samples."""
    if n <= 1 or len(a) < n or len(b) < n:
        return np.concatenate([a, b])
    tail, head = a[-n:], b[:n]
    if np.var(tail) < 1e-5 or np.var(head) < 1e-5:
        fade_in = np.linspace(0.0, 1.0, n)
    else:
        r = np.corrcoef(tail, head)[0, 1]
        r = 0.0 if np.isnan(r) else abs(r)
        t = np.linspace(-1.0, 1.0, n)
        skew = (9 / 16) * np.sin(np.pi / 2 * t) + (1 / 16) * np.sin(3 * np.pi / 2 * t)
        even = np.sqrt(np.clip(0.5 / (1 + r) - ((1 - r) / (1 + r)) * skew ** 2, 0.0, None))
        fade_in = np.clip(even + skew, 0.0, 1.0)
    blended = tail * (1.0 - fade_in) + head * fade_in
    return np.concatenate([a[:-n], blended, b[n:]])


def rebuild(cycles: list[np.ndarray], target_len: int, fade_samples: int) -> np.ndarray:
    """Crossfade-concatenate cycles (looping if needed) until ``target_len`` samples are reached."""
    if not cycles:
        return np.zeros(target_len)
    out = cycles[0]
    i = 1
    guard = 0
    while len(out) < target_len:
        out = _crossfade(out, cycles[i % len(cycles)], fade_samples)
        i += 1
        guard += 1
        if guard > 10 * len(cycles) + 4:
            break
    return out


def rearrange(cycles_by_signal: dict[str, list[np.ndarray]], *, prob_contiguous: float = 0.0,
              random_start: bool = True, rng: random.Random | None = None) -> dict[str, list[np.ndarray]]:
    """Reorder cycles identically across all signals: rotate (contiguous) or shuffle cycle groups."""
    rng = rng or random.Random()
    num = min((len(v) for v in cycles_by_signal.values()), default=0)
    if num < 2:
        return cycles_by_signal
    indices = list(range(num))

    if rng.random() <= prob_contiguous:
        start = rng.randint(0, num - 1) if random_start else 0
        order = indices[start:] + indices[:start]
    else:
        group_sizes = rng.choice([[1], [rng.randint(1, 4) for _ in range(5)]])
        groups, i, s = [], 0, 0
        while i < num:
            g = group_sizes[s % len(group_sizes)]
            groups.append(indices[i:i + g])
            i += g
            s += 1
        rng.shuffle(groups)
        order = [i for group in groups for i in group]

    return {name: [cycles[i] for i in order] for name, cycles in cycles_by_signal.items()}
