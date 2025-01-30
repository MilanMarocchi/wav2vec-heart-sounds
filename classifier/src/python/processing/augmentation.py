"""
    augmentation.py
    Author: Leigh Abbott

    Purpose: Data augmentation on pcg and ecg signals
"""
from processing.filtering import (
    standardise_signal,
    stretch_resample,
    random_parametric_eq,
    random_crop,
)
from util.paths import (
    EPHNOGRAM,
    MIT,
)

import librosa
import random
import numpy as np
import scipy.signal as ssg
import wfdb
import glob
import os


def randfloat(low: float, high: float) -> float:
    return low + random.random() * (high - low)


def get_record(path: str, max_sig_len_s: float = -1.0) -> wfdb.Record:

    header = wfdb.rdheader(path)
    sig_len = header.sig_len
    fs = header.fs

    if max_sig_len_s <= -1.0:
        target_sig_len = sig_len
    else:
        target_sig_len = round(max_sig_len_s * fs) # type: ignore

    if sig_len > target_sig_len:
        sampfrom = random.randint(0, sig_len - target_sig_len)
        sampto = sampfrom + target_sig_len
    else:
        sampfrom = 0
        sampto = sig_len

    rec = wfdb.rdrecord(path, sampfrom=sampfrom, sampto=sampto)
    return rec


def get_pcg_noise(target_sr: float, len_record: int, path: str = "") -> np.ndarray:

    if path == "":
        path = EPHNOGRAM
    valid_files = glob.glob(f"{path}/*.hea")

    num_tries = 0

    while num_tries < 50:

        try:

            num_tries += 1

            valid_file = random.choice(valid_files)

            record = get_record(valid_file.removesuffix('.hea'))
            pcg_noise_1 = record.p_signal[:, record.sig_name.index('AUX1')]
            pcg_noise_2 = record.p_signal[:, record.sig_name.index('AUX2')]
            pcg_noise_1 = ssg.resample_poly(pcg_noise_1, target_sr, record.fs)
            pcg_noise_2 = ssg.resample_poly(pcg_noise_2, target_sr, record.fs)
            pcg_noise_1 = standardise_signal(random_crop(pcg_noise_1, len_record))
            pcg_noise_2 = standardise_signal(random_crop(pcg_noise_2, len_record))
            pcg_noise_1 = random.choice([0, randfloat(0.0, 0.05)]) * pcg_noise_1
            pcg_noise_2 = random.choice([0, randfloat(0.0, 0.05)]) * pcg_noise_2

            return pcg_noise_1 + pcg_noise_2

        except ValueError:
            pass

    return np.zeros(len_record)


def get_ecg_noise(target_sr: float, len_record: int, path: str = "") -> np.ndarray:

    if path == "":
        path = MIT

    em_noise = get_record(os.path.join(path,'em'))
    bw_noise = get_record(os.path.join(path,'bw'))
    ma_noise = get_record(os.path.join(path,'ma'))

    em_noise = ssg.resample_poly(em_noise.p_signal[:, 0], target_sr, em_noise.fs)
    bw_noise = ssg.resample_poly(bw_noise.p_signal[:, 0], target_sr, bw_noise.fs)
    ma_noise = ssg.resample_poly(ma_noise.p_signal[:, 0], target_sr, ma_noise.fs)

    em_noise = random.choice([0, randfloat(0.0, 0.25)]) * standardise_signal(random_crop(em_noise, len_record))
    bw_noise = random.choice([0, randfloat(0.0, 0.5)]) * standardise_signal(random_crop(bw_noise, len_record))
    ma_noise = random.choice([0, randfloat(0.0, 0.25)]) * standardise_signal(random_crop(ma_noise, len_record))

    return em_noise + bw_noise + ma_noise


def augment_multi_pcg(orig_multi_pcg_wav: list, sr,
                    prob_noise=0.30, prob_baseline_wander=0.30,
                    prob_wandering_volume=0.75, prob_time_warp=0.25,
                    prob_hpss=0.75, prob_banding=0.25,
                    prob_real_noise=0.5,
                    EPHNOGRAM="") -> list[np.ndarray]:
    """
    For multichannel pcg recordings, to ensure the same augmentation occurs on all channels
    """
    aug_pcg_wav = list()
    pcg_multi_wav = list()

    for orig_pcg_wav in orig_multi_pcg_wav:
        pcg_wav = orig_pcg_wav.copy()
        pcg_wav = standardise_signal(pcg_wav)
        pcg_multi_wav.append(pcg_wav)

    if np.random.rand() < prob_hpss:

        n_fft_1 = random.choice([512, 1024, 2048])
        win_len_1 = n_fft_1
        hop_len_1 = random.choice([16, 32, 64, 128])
        margin_1 = (randfloat(1.0, 2.0), randfloat(1.0, 2.0))
        kernel_1 = (random.randint(5, 30), random.randint(5, 30))

        n_fft_2 = random.choice([512, 1024, 2048])
        win_len_2 = n_fft_2
        hop_len_2 = random.choice([16, 32, 64, 128])
        margin_2 = (randfloat(1.0, 4.0), randfloat(1.0, 4.0))
        kernel_2 = (random.randint(5, 30), random.randint(5, 30))

        for idx, pcg_wav in enumerate(pcg_multi_wav):

            decomp = librosa.stft(
                pcg_wav,
                n_fft=n_fft_1,
                hop_length=hop_len_1,
                win_length=win_len_1,
            )

            harmon, percus = librosa.decompose.hpss(
                decomp,
                margin=margin_1,
                kernel_size=kernel_1,
            )

            y_1 = librosa.istft(
                harmon,
                n_fft=n_fft_1,
                hop_length=hop_len_1,
                win_length=win_len_1,
            )

            y_2 = librosa.istft(
                percus,
                n_fft=n_fft_1,
                hop_length=hop_len_1,
                win_length=win_len_1,
            )

            decomp = librosa.stft(
                y_1,
                n_fft=n_fft_2,
                hop_length=hop_len_2,
                win_length=win_len_2,
            )

            harmon, percus = librosa.decompose.hpss(
                decomp,
                margin=margin_2,
                kernel_size=kernel_2,
            )

            y_11 = librosa.istft(
                harmon,
                n_fft=n_fft_2,
                hop_length=hop_len_2,
                win_length=win_len_2,
            )

            y_12 = librosa.istft(
                percus,
                n_fft=n_fft_2,
                hop_length=hop_len_2,
                win_length=win_len_2,
            )

            decomp = librosa.stft(
                y_2,
                n_fft=n_fft_2,
                hop_length=hop_len_2,
                win_length=win_len_2,
            )

            harmon, percus = librosa.decompose.hpss(
                decomp,
                margin=margin_2,
                kernel_size=kernel_2,
            )

            y_21 = librosa.istft(
                harmon,
                n_fft=n_fft_2,
                hop_length=hop_len_2,
                win_length=win_len_2,
            )

            y_22 = librosa.istft(
                percus,
                n_fft=n_fft_2,
                hop_length=hop_len_2,
                win_length=win_len_2,
            )

            min_len = min(len(y_i) for y_i in (y_11, y_12, y_21, y_22))

            pcg_wav_1 = standardise_signal(
                1 * randfloat(0.01, 10)*y_11[:min_len]
                + 1 * randfloat(0.01, 10)*y_12[:min_len]
                + 1 * randfloat(0.01, 10)*y_21[:min_len]
                + 1 * randfloat(0.01, 10)*y_22[:min_len]
            )

            pcg_wav_2 = standardise_signal(
                1 * randfloat(0.01, 10)*standardise_signal(y_11[:min_len])
                + 1 * randfloat(0.01, 10)*standardise_signal(y_12[:min_len])
                + 1 * randfloat(0.01, 10)*standardise_signal(y_21[:min_len])
                + 1 * randfloat(0.01, 10)*standardise_signal(y_22[:min_len])
            )

            pcg_multi_wav[idx] = standardise_signal(pcg_wav_1 + randfloat(0.01, 0.05)*pcg_wav_2)

    if np.random.rand() < prob_noise / 4:
        noise_std = random.choice([0.0001, 0.001, 0.01])

        for idx, pcg_wav in enumerate(pcg_multi_wav):
            pcg_wav += randfloat(0, 0.1) * np.random.normal(0, noise_std, pcg_wav.shape)
            pcg_multi_wav[idx] = standardise_signal(pcg_wav)

    if np.random.rand() < prob_time_warp:
        time_stretch_factor = randfloat(1.004, 1.006)

        for idx, pcg_wav in enumerate(pcg_multi_wav):
            pcg_wav = stretch_resample(pcg_wav, sr, time_stretch_factor)
            pcg_multi_wav[idx] = standardise_signal(pcg_wav)

    if np.random.rand() < prob_wandering_volume:
        t = np.arange(pcg_multi_wav[0].size) / sr
        vol_mod_1 = randfloat(0.01, 0.25) * np.sin(2 * np.pi * (randfloat(0.05, 0.5) * t + randfloat(0, 1)))
        vol_mod_2 = randfloat(0.01, 0.25) * np.sin(2 * np.pi * (randfloat(0.001, 0.05) * t + randfloat(0, 1)))
        for idx, pcg_wav in enumerate(pcg_multi_wav):
            pcg_wav *= (1 + vol_mod_1 + vol_mod_2)
            pcg_multi_wav[idx] = standardise_signal(pcg_wav)

    if np.random.rand() < prob_noise / 4:
        noise_std = random.choice([0.0001, 0.001, 0.01])
        for idx, pcg_wav in enumerate(pcg_multi_wav):
            pcg_wav += randfloat(0, 0.1) * np.random.normal(0, noise_std, pcg_wav.shape)
            pcg_multi_wav[idx] = standardise_signal(pcg_wav)

    if np.random.rand() < prob_banding:
        for idx, pcg_wav in enumerate(pcg_multi_wav):
            pcg_wav = random_parametric_eq(pcg_wav, sr, low=2, high=500)
            pcg_multi_wav[idx] = standardise_signal(pcg_wav)

    if np.random.rand() < prob_real_noise:
        pcg_noise = get_pcg_noise(sr, len(pcg_multi_wav[0]), EPHNOGRAM)
        for idx in range(len(pcg_multi_wav)):
            pcg_multi_wav[idx] += pcg_noise

    return pcg_multi_wav 


def augment_pcg(orig_pcg_wav: np.ndarray, sr,
                    prob_noise=0.30, prob_baseline_wander=0.30,
                    prob_wandering_volume=0.75, prob_time_warp=0.25,
                    prob_hpss=0.75, prob_banding=0.25,
                    prob_real_noise=0.5,
                    EPHNOGRAM="") -> np.ndarray:

    pcg_wav = orig_pcg_wav.copy()

    pcg_wav = standardise_signal(pcg_wav)

    if np.random.rand() < prob_hpss:

        n_fft_1 = random.choice([512, 1024, 2048])
        win_len_1 = n_fft_1
        hop_len_1 = random.choice([16, 32, 64, 128])
        margin_1 = (randfloat(1.0, 2.0), randfloat(1.0, 2.0))
        kernel_1 = (random.randint(5, 30), random.randint(5, 30))

        n_fft_2 = random.choice([512, 1024, 2048])
        win_len_2 = n_fft_2
        hop_len_2 = random.choice([16, 32, 64, 128])
        margin_2 = (randfloat(1.0, 4.0), randfloat(1.0, 4.0))
        kernel_2 = (random.randint(5, 30), random.randint(5, 30))

        decomp = librosa.stft(
            pcg_wav,
            n_fft=n_fft_1,
            hop_length=hop_len_1,
            win_length=win_len_1,
        )

        harmon, percus = librosa.decompose.hpss(
            decomp,
            margin=margin_1,
            kernel_size=kernel_1,
        )

        y_1 = librosa.istft(
            harmon,
            n_fft=n_fft_1,
            hop_length=hop_len_1,
            win_length=win_len_1,
        )

        y_2 = librosa.istft(
            percus,
            n_fft=n_fft_1,
            hop_length=hop_len_1,
            win_length=win_len_1,
        )

        decomp = librosa.stft(
            y_1,
            n_fft=n_fft_2,
            hop_length=hop_len_2,
            win_length=win_len_2,
        )

        harmon, percus = librosa.decompose.hpss(
            decomp,
            margin=margin_2,
            kernel_size=kernel_2,
        )

        y_11 = librosa.istft(
            harmon,
            n_fft=n_fft_2,
            hop_length=hop_len_2,
            win_length=win_len_2,
        )

        y_12 = librosa.istft(
            percus,
            n_fft=n_fft_2,
            hop_length=hop_len_2,
            win_length=win_len_2,
        )

        decomp = librosa.stft(
            y_2,
            n_fft=n_fft_2,
            hop_length=hop_len_2,
            win_length=win_len_2,
        )

        harmon, percus = librosa.decompose.hpss(
            decomp,
            margin=margin_2,
            kernel_size=kernel_2,
        )

        y_21 = librosa.istft(
            harmon,
            n_fft=n_fft_2,
            hop_length=hop_len_2,
            win_length=win_len_2,
        )

        y_22 = librosa.istft(
            percus,
            n_fft=n_fft_2,
            hop_length=hop_len_2,
            win_length=win_len_2,
        )

        min_len = min(len(y_i) for y_i in (y_11, y_12, y_21, y_22))

        pcg_wav_1 = standardise_signal(
            1 * randfloat(0.01, 10)*y_11[:min_len]
            + 1 * randfloat(0.01, 10)*y_12[:min_len]
            + 1 * randfloat(0.01, 10)*y_21[:min_len]
            + 1 * randfloat(0.01, 10)*y_22[:min_len]
        )

        pcg_wav_2 = standardise_signal(
            1 * randfloat(0.01, 10)*standardise_signal(y_11[:min_len])
            + 1 * randfloat(0.01, 10)*standardise_signal(y_12[:min_len])
            + 1 * randfloat(0.01, 10)*standardise_signal(y_21[:min_len])
            + 1 * randfloat(0.01, 10)*standardise_signal(y_22[:min_len])
        )

        pcg_wav = standardise_signal(pcg_wav_1 + randfloat(0.01, 0.05)*pcg_wav_2)

    if np.random.rand() < prob_noise / 4:
        noise_std = random.choice([0.0001, 0.001, 0.01])
        pcg_wav += randfloat(0, 0.1) * np.random.normal(0, noise_std, pcg_wav.shape)
        pcg_wav = standardise_signal(pcg_wav)


    if np.random.rand() < prob_time_warp:
        time_stretch_factor = randfloat(1.004, 1.006)
        pcg_wav = stretch_resample(pcg_wav, sr, time_stretch_factor)
        pcg_wav = standardise_signal(pcg_wav)

    if np.random.rand() < prob_wandering_volume:
        t = np.arange(pcg_wav.size) / sr
        vol_mod_1 = randfloat(0.01, 0.25) * np.sin(2 * np.pi * (randfloat(0.05, 0.5) * t + randfloat(0, 1)))
        vol_mod_2 = randfloat(0.01, 0.25) * np.sin(2 * np.pi * (randfloat(0.001, 0.05) * t + randfloat(0, 1)))
        pcg_wav *= (1 + vol_mod_1 + vol_mod_2)
        pcg_wav = standardise_signal(pcg_wav)

    if np.random.rand() < prob_noise / 4:
        noise_std = random.choice([0.0001, 0.001, 0.01])
        pcg_wav += randfloat(0, 0.1) * np.random.normal(0, noise_std, pcg_wav.shape)
        pcg_wav = standardise_signal(pcg_wav)

    if np.random.rand() < prob_banding:
        pcg_wav = random_parametric_eq(pcg_wav, sr, low=2, high=500)
        pcg_wav = standardise_signal(pcg_wav)

    if np.random.rand() < prob_real_noise:
        pcg_wav += get_pcg_noise(sr, len(pcg_wav), EPHNOGRAM)

    return pcg_wav


def augment_signals(orig_ecg_wav: np.ndarray, orig_pcg_wav: np.ndarray, sr: float,
                    prob_noise=0.30, prob_baseline_wander=0.30,
                    prob_wandering_volume=0.75, prob_time_warp=0.25,
                    prob_hpss=0.75, prob_banding=0.25,
                    prob_real_noise=0.5,
                    MIT="", EPHNOGRAM="") -> tuple[np.ndarray, np.ndarray]:

    ecg_wav = orig_ecg_wav.copy()
    pcg_wav = orig_pcg_wav.copy()

    ecg_wav = standardise_signal(ecg_wav)
    pcg_wav = standardise_signal(pcg_wav)

    if np.random.rand() < prob_hpss:

        n_fft_1 = random.choice([512, 1024, 2048])
        win_len_1 = n_fft_1
        hop_len_1 = random.choice([16, 32, 64, 128])
        margin_1 = (randfloat(1.0, 2.0), randfloat(1.0, 2.0))
        kernel_1 = (random.randint(5, 30), random.randint(5, 30))

        n_fft_2 = random.choice([512, 1024, 2048])
        win_len_2 = n_fft_2
        hop_len_2 = random.choice([16, 32, 64, 128])
        margin_2 = (randfloat(1.0, 4.0), randfloat(1.0, 4.0))
        kernel_2 = (random.randint(5, 30), random.randint(5, 30))

        decomp = librosa.stft(
            pcg_wav,
            n_fft=n_fft_1,
            hop_length=hop_len_1,
            win_length=win_len_1,
        )

        harmon, percus = librosa.decompose.hpss(
            decomp,
            margin=margin_1,
            kernel_size=kernel_1,
        )

        y_1 = librosa.istft(
            harmon,
            n_fft=n_fft_1,
            hop_length=hop_len_1,
            win_length=win_len_1,
        )

        y_2 = librosa.istft(
            percus,
            n_fft=n_fft_1,
            hop_length=hop_len_1,
            win_length=win_len_1,
        )

        decomp = librosa.stft(
            y_1,
            n_fft=n_fft_2,
            hop_length=hop_len_2,
            win_length=win_len_2,
        )

        harmon, percus = librosa.decompose.hpss(
            decomp,
            margin=margin_2,
            kernel_size=kernel_2,
        )

        y_11 = librosa.istft(
            harmon,
            n_fft=n_fft_2,
            hop_length=hop_len_2,
            win_length=win_len_2,
        )

        y_12 = librosa.istft(
            percus,
            n_fft=n_fft_2,
            hop_length=hop_len_2,
            win_length=win_len_2,
        )

        decomp = librosa.stft(
            y_2,
            n_fft=n_fft_2,
            hop_length=hop_len_2,
            win_length=win_len_2,
        )

        harmon, percus = librosa.decompose.hpss(
            decomp,
            margin=margin_2,
            kernel_size=kernel_2,
        )

        y_21 = librosa.istft(
            harmon,
            n_fft=n_fft_2,
            hop_length=hop_len_2,
            win_length=win_len_2,
        )

        y_22 = librosa.istft(
            percus,
            n_fft=n_fft_2,
            hop_length=hop_len_2,
            win_length=win_len_2,
        )

        min_len = min(len(y_i) for y_i in (y_11, y_12, y_21, y_22))

        pcg_wav_1 = standardise_signal(
            1 * randfloat(0.01, 10)*y_11[:min_len]
            + 1 * randfloat(0.01, 10)*y_12[:min_len]
            + 1 * randfloat(0.01, 10)*y_21[:min_len]
            + 1 * randfloat(0.01, 10)*y_22[:min_len]
        )

        pcg_wav_2 = standardise_signal(
            1 * randfloat(0.01, 10)*standardise_signal(y_11[:min_len])
            + 1 * randfloat(0.01, 10)*standardise_signal(y_12[:min_len])
            + 1 * randfloat(0.01, 10)*standardise_signal(y_21[:min_len])
            + 1 * randfloat(0.01, 10)*standardise_signal(y_22[:min_len])
        )

        pcg_wav = standardise_signal(pcg_wav_1 + randfloat(0.01, 0.05)*pcg_wav_2)
        ecg_wav = ecg_wav[:min_len]

    if np.random.rand() < prob_noise / 4:
        noise_std = random.choice([0.0001, 0.001, 0.01])
        pcg_wav += randfloat(0, 0.1) * np.random.normal(0, noise_std, pcg_wav.shape)
        pcg_wav = standardise_signal(pcg_wav)

    if np.random.rand() < prob_noise / 4:
        noise_std = random.choice([0.0001, 0.001, 0.01])
        ecg_wav += randfloat(0, 0.1) * np.random.normal(0, noise_std, ecg_wav.shape)
        ecg_wav = standardise_signal(ecg_wav)

    if np.random.rand() < prob_baseline_wander:
        t = np.arange(ecg_wav.size) / sr
        baseline_wander = randfloat(0.01, 0.2) * np.sin(2 * np.pi * (randfloat(0.05, 0.5) * t + randfloat(0, 1)))
        baseline_wander += randfloat(0.01, 0.2) * np.sin(2 * np.pi * (randfloat(0.001, 0.05) * t + randfloat(0, 1)))
        ecg_wav += baseline_wander
        ecg_wav = standardise_signal(ecg_wav)

    if np.random.rand() < prob_time_warp:
        time_stretch_factor = randfloat(1.004, 1.006)
        ecg_wav = stretch_resample(ecg_wav, sr, time_stretch_factor)
        pcg_wav = stretch_resample(pcg_wav, sr, time_stretch_factor)
        pcg_wav = standardise_signal(pcg_wav)
        ecg_wav = standardise_signal(ecg_wav)

    if np.random.rand() < prob_wandering_volume:
        t = np.arange(pcg_wav.size) / sr
        vol_mod_1 = randfloat(0.01, 0.25) * np.sin(2 * np.pi * (randfloat(0.05, 0.5) * t + randfloat(0, 1)))
        vol_mod_2 = randfloat(0.01, 0.25) * np.sin(2 * np.pi * (randfloat(0.001, 0.05) * t + randfloat(0, 1)))
        pcg_wav *= (1 + vol_mod_1 + vol_mod_2)
        pcg_wav = standardise_signal(pcg_wav)

    if np.random.rand() < prob_noise / 4:
        noise_std = random.choice([0.0001, 0.001, 0.01])
        pcg_wav += randfloat(0, 0.1) * np.random.normal(0, noise_std, pcg_wav.shape)
        pcg_wav = standardise_signal(pcg_wav)

    if np.random.rand() < prob_noise / 4:
        noise_std = random.choice([0.0001, 0.001, 0.01])
        ecg_wav += randfloat(0, 0.1) * np.random.normal(0, noise_std, ecg_wav.shape)
        ecg_wav = standardise_signal(ecg_wav)

    if np.random.rand() < prob_banding:
        pcg_wav = random_parametric_eq(pcg_wav, sr, low=2, high=500)
        pcg_wav = standardise_signal(pcg_wav)

    if np.random.rand() < prob_banding:
        ecg_wav = random_parametric_eq(ecg_wav, sr, low=0.25, high=100)
        ecg_wav = standardise_signal(ecg_wav)

    if np.random.rand() < prob_real_noise:
        ecg_wav += get_ecg_noise(sr, len(ecg_wav), MIT)

    if np.random.rand() < prob_real_noise:
        pcg_wav += get_pcg_noise(sr, len(pcg_wav), EPHNOGRAM)

    return ecg_wav, pcg_wav
