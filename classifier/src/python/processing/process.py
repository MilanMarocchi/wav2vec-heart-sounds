"""
    preprocessing.py
    Author: Milan Marocchi

    Purpose: Run any preprocessing that is required for signals before training
"""

from processing.filtering import (
    resample,
    low_pass_butter,
    high_pass_butter,
    normalise_signal,
    pre_filter_ecg,
    spike_removal_python,
    create_band_filters
)

import numpy as np
import scipy.signal as ssg

def pre_process_pcg_orig(pcg: np.ndarray, fs: int, fs_new: int) -> np.ndarray:
    pcg = resample(pcg, fs, fs_new)

    pcg = low_pass_butter(pcg, 2, 400, fs_new)
    pcg = high_pass_butter(pcg, 2, 25, fs_new)
    pcg = normalise_signal(pcg)
    # FIXME: Get this working
    # pcg = spike_removal_python(pcg, fs_new)

    return pcg


def pre_process_ecg_orig(ecg: np.ndarray, fs: int, fs_new: int) -> np.ndarray:
    ecg = resample(ecg, fs, fs_new)
    ecg = low_pass_butter(ecg, 2, 60, fs_new)
    ecg = high_pass_butter(ecg, 2, 2, fs_new)
    ecg = normalise_signal(ecg)

    ecg = pre_filter_ecg(ecg, fs_new)

    return ecg


def pre_process_orig_four_bands(pcg: np.ndarray, fs: int) -> np.ndarray:
    data = np.zeros((len(pcg), 4))

    pcg = pcg.squeeze()

    b = create_band_filters(fs)
    for i in range(4):
        data[:, i] = ssg.filtfilt(b[i], 1, pcg)

    return data


def normalise_array_length(array, normalised_length):
    """
    Pad or crop the array to have a shape of (2500, second_dim_size).

    :param array: The input array.
    :param normalised_length: Length to normalise array to.
    :return: Array with shape (2500, second_dim_size).
    """
    pad_amount = 0
    # Pad or crop the first dimension to 2500
    if len(array) < normalised_length:
        # Pad
        pad_amount = normalised_length - len(array)
        array = np.pad(array, (0, pad_amount), mode='constant')
    elif len(array) > normalised_length:
        # Crop
        array = array[:normalised_length]

    pad_idx = len(array) - pad_amount

    return array, pad_idx


def normalise_2d_array_length(array, normalised_length):
    """
    Pad or crop the array to have a shape of (2500, second_dim_size).

    :param array: The input array.
    :param normalised_length: Length to normalise array to.
    :return: Array with shape (2500, second_dim_size).
    """
    pad_amount = 0
    # Pad or crop the first dimension to 2500
    if array.shape[0] < normalised_length:
        # Pad
        pad_amount = normalised_length - array.shape[0]
        array = np.pad(array, ((0, pad_amount), (0, 0)), mode='constant')
    elif array.shape[0] > normalised_length:
        # Crop
        array = array[:normalised_length, :]

    pad_idx = len(array) - pad_amount

    return array, pad_idx