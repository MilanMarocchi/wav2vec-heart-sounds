#!/usr/bin/env pipenv-shebang
"""
    explore.py
    Author : Milan Marocchi

    Front end for exploration of various signals
"""
from typing import Callable
import matplotlib
import matplotlib.pyplot as plt
import numpy as np

from util.fileio import (
    get_cinc_sig,
    read_ticking_PCG,
    read_ticking_ECG,
    get_patients,
)

from processing.filtering import (
    resample,
    low_pass_butter,
    high_pass_butter,
    normalise_signal,
    pre_filter_ecg,
    spike_removal_python,
    create_band_filters,
    noise_canc,
)

import click
import os

def plot_time_nm_signal(signal: list[np.ndarray], title: str):
    t = range(len(signal[0]))
    plt.figure()
    plt.subplot(3, 1, 1)
    plt.plot(range(len(signal[0])), signal[0])
    plt.title("PCG Channel")
    plt.subplot(3, 1, 2)
    plt.plot(range(len(signal[1])), signal[1])
    plt.title("PCG NM Channel")
    plt.subplot(3, 1, 3)
    plt.plot(range(len(signal[2])), signal[2])
    plt.title("PCG with noise canc Channel")

    plt.show()

def plot_time_signal(signal: np.ndarray, title: str):
    t = range(len(signal))
    plt.plot(t, signal)
    plt.title(title)

    plt.show()

def plot_time_signals(signals: list[np.ndarray], title: str):
    plt.figure()
    plt.subplot(4, 2, 1)
    plt.plot(range(len(signals[0])), signals[0])
    plt.title("Channel 1")
    plt.subplot(4, 2, 2)
    plt.plot(range(len(signals[1])), signals[1])
    plt.title("Channel 2")
    plt.subplot(4, 2, 3)
    plt.plot(range(len(signals[2])), signals[2])
    plt.title("Channel 3")
    plt.subplot(4, 2, 4)
    plt.plot(range(len(signals[3])), signals[3])
    plt.title("Channel 4")
    plt.subplot(4, 2, 5)
    plt.plot(range(len(signals[4])), signals[4])
    plt.title("Channel 5")
    plt.subplot(4, 2, 6)
    plt.plot(range(len(signals[5])), signals[5])
    plt.title("Channel 6")
    plt.subplot(4, 2, 7)
    plt.plot(range(len(signals[6])), signals[6])
    plt.title("Channel 7")
    plt.subplot(4, 2, 8)
    plt.plot(range(len(signals[7])), signals[7])
    plt.title("ECG Channel")

    plt.show()

# Option functions
def read_and_display_PCG_ECG(file_name: str):
    file_name = file_name.replace(".wav", "")
    signal1, fs = read_ticking_PCG(file_name, 1)
    signal2, fs = read_ticking_PCG(file_name, 2)
    signal3, fs = read_ticking_PCG(file_name, 3)
    signal4, fs = read_ticking_PCG(file_name, 4)
    signal5, fs = read_ticking_PCG(file_name, 5)
    signal6, fs = read_ticking_PCG(file_name, 6)
    signal7, fs = read_ticking_PCG(file_name, 7)
    ecg, fs = read_ticking_ECG(file_name)

    signal1 = preprocess_pcg(signal1, fs)
    signal2 = preprocess_pcg(signal1, fs)
    signal3 = preprocess_pcg(signal1, fs)
    signal4 = preprocess_pcg(signal1, fs)
    signal5 = preprocess_pcg(signal1, fs)
    signal6 = preprocess_pcg(signal1, fs)
    signal7 = preprocess_pcg(signal1, fs)
    ecg = preprocess_pcg(ecg, fs)

    plot_time_signals([
        signal1,
        signal2,
        signal3,
        signal4,
        signal5,
        signal6,
        signal7,
        ecg],
    "PCG ECG signal"
    )

def read_and_display_PCG(file_name: str):
    file_name = file_name.replace(".wav", "")
    signal, fs = read_ticking_PCG(file_name, 2)

    pcg = preprocess_pcg(signal, fs)

    plot_time_signal(pcg, "PCG signal")

def read_and_display_ECG(file_name: str):
    file_name = file_name.replace(".wav", "")
    signal, fs = read_ticking_ECG(file_name)

    ecg = preprocess_ecg(signal, fs)

    plot_time_signal(ecg, "ECG signal")

def read_and_display_PCG_NM(file_name: str):
    file_name = file_name.replace(".wav", "")
    singal, fs = read_ticking_PCG(file_name, 2)
    signal_nm, _ = read_ticking_PCG(file_name, 2, noise_mic=True)

    signal_canc = noise_canc(signal_nm, singal, fs=fs)
    signal_canc = preprocess_pcg(signal_canc, fs)

    singal = preprocess_pcg(singal, fs)

    plot_time_nm_signal([singal, signal_nm, signal_canc], "PCG signal")

def read_and_display_cinc_PCG(file_name: str):
    file_name = file_name.replace(".wav", "")
    signal, fs = get_cinc_sig(file_name, "PCG")

    pcg = preprocess_pcg(signal, fs)

    plot_time_signal(pcg, "PCG signal")

def read_and_display_cinc_ECG(file_name: str):
    file_name = file_name.replace(".wav", "")
    signal, fs = get_cinc_sig(file_name, "ECG")

    ecg = preprocess_ecg(signal, fs)

    plot_time_signal(ecg, "ECG signal")

def read_and_display_cinc_PCG_ECG(file_name: str):
    file_name = file_name.replace(".wav", "")
    pcg, fs = get_cinc_sig(file_name, "PCG")
    ecg, fs = get_cinc_sig(file_name, "ECG")

    plot_time_signal(pcg, "PCG signal")
    plot_time_signal(ecg, "ECG signal")

# Preprocessing
def preprocess_pcg(signal: np.ndarray, fs: int) -> np.ndarray:
    fs_new = 1000
    pcg = resample(signal, fs, fs_new)

    pcg = low_pass_butter(pcg, 2, 400, fs_new)
    pcg = high_pass_butter(pcg, 2, 25, fs_new)
    pcg = normalise_signal(pcg)
    # FIXME: Get this working
    # pcg = spike_removal_python(pcg, fs_new)

    return pcg


def preprocess_ecg(signal: np.ndarray, fs: int) -> np.ndarray:
    fs_new = 1000
    ecg = resample(signal, fs, fs_new)
    ecg = low_pass_butter(ecg, 2, 60, fs_new)
    ecg = high_pass_butter(ecg, 2, 2, fs_new)
    ecg = normalise_signal(ecg)

    ecg = pre_filter_ecg(ecg, fs_new)

    return ecg


def run_operation_on_all(signals: list[np.ndarray], fs: int, operation: Callable[[np.ndarray, int], np.ndarray]):
    """Run processing/plotting operations on all signals"""
    for signal in signals:
        operation(signal, fs)


# Option map containing function for each option.
ticking_option_map = {
    "PCG": read_and_display_PCG,
    "ECG": read_and_display_ECG,
    "BOTH": read_and_display_PCG_ECG,
    "NM": read_and_display_PCG_NM,
}

cinc_option_map = {
    "PCG": read_and_display_cinc_PCG,
    "ECG": read_and_display_cinc_ECG,
    "BOTH": read_and_display_cinc_PCG_ECG,
}

@click.command()
@click.option(
    "-D",
    "--data_dir",
    required=True,
    help="Path to data to explore."
)
@click.option(
    "-B",
    "--database",
    required=True,
    help="What database to explore."
)
@click.option(
    "-A",
    "--all_data",
    is_flag=True,
    help="Whether to go through all data or not."
)
def explore(
    data_dir,
    database,
    all_data,
):
    if database == "ticking-heart":
        option_map = ticking_option_map
    elif database == "cinc":
        option_map = cinc_option_map
    else:
        raise ValueError(f'Database is not supported: {database=}')

    loop = True
    while loop:
        signals, _ = get_patients(os.path.join(data_dir, "REFERENCE.csv"))
        for signal in signals:
            print(signal)

        if all_data:
            for signal in signals:
                option_map["BOTH"](os.path.join(os.path.abspath(data_dir), signal))
            break

        file = "None"
        while file not in signals:
            file = input("What signal? ")
            print()

            if file not in signals:
                print("Invalid signal\n")
            else:
                option = input("PCG or ECG or BOTH or NM? ")
                print()

        option_map[option](os.path.join(os.path.abspath(data_dir), file)) # type: ignore

        loop = input("Exit[y/n]? ") != "y"

if __name__ == "__main__":
    explore()
