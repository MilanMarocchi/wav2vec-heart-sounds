"""Generate synthetic waveform datasets from a trained generator.

Iterates a :class:`GenerativeDataset`, samples a waveform per item conditioned on that item's
mel-spectrogram and label, and writes each as a WAV plus a ``REFERENCE.csv`` manifest that the
classifier's schedule loader can consume as a generated dataset.
"""

from __future__ import annotations

import csv
import os

import numpy as np
import torch
from scipy.io import wavfile
from tqdm import tqdm

from ..signalproc.normalize import abs_max_normalise


@torch.no_grad()
def generate_dataset(model, spec, dataset, output_dir: str, *, device, per_item: int = 1,
                     sampler_kwargs: dict | None = None) -> str:
    """Sample ``per_item`` waveforms for each dataset item and write them under ``output_dir``.

    Returns the path to the written ``REFERENCE.csv``.
    """
    os.makedirs(output_dir, exist_ok=True)
    sampler_kwargs = sampler_kwargs or {}
    model.eval()
    manifest_path = os.path.join(output_dir, "REFERENCE.csv")

    with open(manifest_path, "w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["patient", "label", "file"])
        for idx in tqdm(range(len(dataset)), desc="generating"):
            item = dataset[idx]
            con_spec = item["con_spec"].to(device)
            label = item["label"]
            for copy in range(per_item):
                audio, sr = spec.sample(model, con_spec, label, **sampler_kwargs)
                wave = abs_max_normalise(audio.squeeze(0).cpu().numpy()).astype(np.float32)
                name = f"{item['patient']}_{idx}_{copy}"
                path = os.path.join(output_dir, f"{name}.wav")
                wavfile.write(path, sr, wave)
                writer.writerow([item["patient"], label, os.path.basename(path)])
    return manifest_path
