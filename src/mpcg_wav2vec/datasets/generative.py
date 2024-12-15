"""Datasets that feed the DiffWave / WaveGrad generators.

Each item provides a *reference* waveform (the target the model learns to generate) and a
*conditioning* waveform (encoded to a mel-spectrogram, ``con_spec``) plus an integer class
label. Everything is at the generator sample rate (4 kHz) and cropped to a fixed number of
samples derived from ``crop_frames * hop_length`` so batches stack cleanly.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

import numpy as np
import torch
import torchaudio
import wfdb
from torch.utils.data import Dataset

from ..signalproc import MelConfig, abs_max_normalise, add_chirp, log_mel, resample
from ..signalproc.preprocess import fit_length
from . import heart_cycles
from .cinc import _binary_label, _label_column, read_split
from .labels import label_to_index


@dataclass
class GenRecord:
    reference: np.ndarray            # target waveform at fs
    conditioning: np.ndarray         # conditioning waveform at fs
    label: int
    patient: str
    segment_path: str | None = None  # per-recording cardiac-cycle segmentation file


def _fade(x: np.ndarray, n: int = 128) -> np.ndarray:
    if len(x) < 2 * n:
        return x
    x = x.copy()
    x[:n] *= np.linspace(0.0, 1.0, n)
    x[-n:] *= np.linspace(1.0, 0.0, n)
    return x


class GenerativeDataset(Dataset):
    def __init__(self, records: list[GenRecord], fs: int, mel: MelConfig,
                 crop_frames: int, hop_length: int, *, rearrange_cycles: bool = True,
                 prob_contiguous: float = 0.0, random_start: bool = True, fade_ms: float = 10.0):
        self.records = records
        self.fs = fs
        self.mel_transform = mel.build()
        self.crop_frames = crop_frames
        self.crop = crop_frames * hop_length
        self.rearrange_cycles = rearrange_cycles
        self.prob_contiguous = prob_contiguous
        self.random_start = random_start
        self.fade_samples = int(round(fade_ms / 1000.0 * fs))

    def __len__(self) -> int:
        return len(self.records)

    def _rebuild_from_cycles(self, rec: GenRecord):
        """Return (reference, conditioning) rebuilt from shuffled cardiac cycles, or None."""
        joins = heart_cycles.load_join_indices(rec.segment_path, self.fs)
        ref_cycles = heart_cycles.split_cycles(abs_max_normalise(rec.reference), joins)
        con_cycles = heart_cycles.split_cycles(abs_max_normalise(rec.conditioning), joins)
        if len(ref_cycles) < 2 or len(con_cycles) < 2:
            return None
        arranged = heart_cycles.rearrange(
            {"ref": ref_cycles, "con": con_cycles},
            prob_contiguous=self.prob_contiguous, random_start=self.random_start,
        )
        ref = heart_cycles.rebuild(arranged["ref"], self.crop, self.fade_samples)
        con = heart_cycles.rebuild(arranged["con"], self.crop, self.fade_samples)
        return ref, con

    def __getitem__(self, idx: int) -> dict:
        rec = self.records[idx]
        rebuilt = None
        if self.rearrange_cycles and rec.segment_path:
            try:
                rebuilt = self._rebuild_from_cycles(rec)
            except (OSError, KeyError, ValueError):
                rebuilt = None
        if rebuilt is not None:
            ref, con = rebuilt
        else:
            ref, con = abs_max_normalise(rec.reference), abs_max_normalise(rec.conditioning)

        ref = _fade(ref)
        con = _fade(con)
        ref, _ = fit_length(ref, self.crop)
        con, _ = fit_length(con, self.crop)

        ref_t = torch.from_numpy(np.ascontiguousarray(ref)).float()
        con_t = torch.from_numpy(np.ascontiguousarray(con)).float()
        con_spec = log_mel(con_t, self.mel_transform)
        # A centred STFT yields one extra frame; keep exactly crop_frames so the upsampled
        # conditioner matches the crop_frames * hop_length waveform length.
        if con_spec.shape[-1] >= self.crop_frames:
            con_spec = con_spec[..., : self.crop_frames]
        else:
            pad = self.crop_frames - con_spec.shape[-1]
            con_spec = torch.nn.functional.pad(con_spec, (0, pad))
        chirp = torch.from_numpy(add_chirp(ref, self.fs)).float()

        return {
            "ref_audio": ref_t,
            "con_audio": con_t,
            "con_spec": con_spec,
            "label": int(rec.label),
            "seg_wave": ref_t.clone(),
            "chirp_wave": chirp,
            "patient": rec.patient,
        }


def cinc_generative_dataset(
    data_dir: str,
    csv_path: str,
    subset: str,
    *,
    fs: int,
    mel: MelConfig,
    crop_frames: int,
    hop_length: int,
    label_vocab: str = "training-a",
    condition_on_ecg: bool = False,
    fold: int = 1,
    segment_dir: str | None = None,
    rearrange_cycles: bool = True,
    prob_contiguous: float = 0.0,
) -> GenerativeDataset:
    """Build a generator dataset from CinC records (PCG reference, PCG-or-ECG conditioning).

    When ``segment_dir`` is given, each record is paired with its ``<segment_dir>/<patient>.json``
    cardiac-cycle segmentation so training can rearrange heart cycles.
    """
    df = read_split(csv_path, subset, fold)
    label_col = _label_column(df)
    records: list[GenRecord] = []

    for _, row in df.iterrows():
        patient = str(row["patient"])
        raw_label = -1 if _binary_label(row[label_col]) == 0 else 1
        label = label_to_index(label_vocab, raw_label)
        try:
            rec = wfdb.rdrecord(os.path.join(data_dir, patient))
        except (FileNotFoundError, ValueError):
            continue
        sig, sr = rec.p_signal, rec.fs
        pcg = resample(sig[:, 0], sr, fs)
        con_channel = 1 if (condition_on_ecg and sig.shape[1] > 1) else 0
        con = resample(sig[:, con_channel], sr, fs)
        seg_path = os.path.join(segment_dir, f"{patient}.json") if segment_dir else None
        if seg_path and not os.path.exists(seg_path):
            seg_path = None
        records.append(GenRecord(reference=pcg, conditioning=con, label=label, patient=patient,
                                 segment_path=seg_path))
    return GenerativeDataset(records, fs=fs, mel=mel, crop_frames=crop_frames, hop_length=hop_length,
                             rearrange_cycles=rearrange_cycles, prob_contiguous=prob_contiguous)
