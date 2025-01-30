"""
    filtering.py
    Author : Milan Marocchi

    Various filters
"""

from typing import Any
import numpy as np
import os
import math
import scipy.signal as ssg
from scipy.io import loadmat 
import random
import librosa
import pywt

import logging
logging.basicConfig(level=logging.INFO)

# Matlab engine
ENG = None


def start_matlab(matlab_location: str):
    print(matlab_location)
    if matlab_location != '':
        try:
            import matlab.engine
            global ENG
            ENG = matlab.engine.start_matlab()
            ENG.addpath(ENG.genpath(str(matlab_location)), nargout=0)  # type: ignore
            logging.info('STARTED MATLAB')
        except ImportError as e:
            logging.error('Matlab engine not installed --- trying anyway')
            logging.error(e)


def stop_matlab():
    if ENG is not None:
        ENG.exit()  # type: ignore
        logging.info('STOPPED MATLAB')


def stretch_resample(signal: np.ndarray, sample_rate: int, time_stretch_factor: float) -> np.ndarray:
    signal = librosa.effects.time_stretch(signal, rate=time_stretch_factor)
    signal = librosa.resample(signal, orig_sr=round(sample_rate * time_stretch_factor), target_sr=sample_rate)
    return signal


def random_crop(signal: np.ndarray, len_crop: int) -> np.ndarray:
    start = random.randint(0, len(signal) - len_crop)
    end = start + len_crop
    return signal[start:end]


def random_parametric_eq(signal: np.ndarray, sr: float, low: float, high: float, num_bands: int = 5) -> np.ndarray:
    equalised_signal = np.copy(signal)

    for _ in range(num_bands):

        b_low = np.random.uniform(low=low, high=0.95*high)
        b_high = random.choice([np.random.uniform(low=b_low+0.05*(high-low), high=high), b_low+(high-low)/num_bands])

        sos = ssg.iirfilter(N=1, Wn=[b_low / (sr / 2), b_high / (sr / 2)], btype='band',
                            analog=False, ftype='butter', output='sos')

        equalised_signal = np.asarray(ssg.sosfilt(sos, equalised_signal))

    return standardise_signal(standardise_signal(equalised_signal)/50 + standardise_signal(signal))


def interpolate_nans(a: np.ndarray) -> np.ndarray:
    mask = np.isnan(a)
    a[mask] = np.interp(np.flatnonzero(mask),
                        np.flatnonzero(~mask),
                        a[~mask])
    return a


def normalise_signal(signal: np.ndarray) -> np.ndarray:
    signal = interpolate_nans(signal)

    signal -= np.mean(signal)
    signal /= np.max(np.abs(signal))
    signal = np.clip(signal, -1, 1)

    return signal


def standardise_signal(signal: np.ndarray) -> np.ndarray:
    return normalise_signal(signal)


def bandpass(signal: np.ndarray, fs: float, low: float, high: float) -> np.ndarray:
    nyquist_freq = 0.5 * fs
    low /= nyquist_freq
    high /= nyquist_freq

    sos = ssg.butter(1, [low, high], 'bandpass', analog=False, output='sos',)
    signal = ssg.sosfiltfilt(sos, signal)

    return signal


def notchfilter(signal: np.ndarray, fs: float, notch: float, Q: float) -> np.ndarray:
    nyquist_freq = 0.5 * fs
    notch /= nyquist_freq

    b, a = ssg.iirnotch(notch, Q)
    signal = ssg.filtfilt(b, a, signal)

    return signal


def pre_filter_ecg(signal: np.ndarray, fs: float) -> np.ndarray:
    signal = notchfilter(signal, fs, 50, 55)
    signal = notchfilter(signal, fs, 60, 55)
    signal = notchfilter(signal, fs, 100, 55)
    signal = notchfilter(signal, fs, 120, 55)
    signal = bandpass(signal, fs, 0.25, 150)
    # signal = wavefilt(signal, 'sym4', 4)
    # signal = bandpass(signal, fs, 0.5, 70)
    return signal


def create_band_filters(fs: int) -> list[np.ndarray]:
    N = 61
    sr = fs
    wn = 45 * 2 / sr
    b1 = ssg.firwin(N, wn, window='hamming', pass_zero='lowpass') # type: ignore
    wn = [45 * 2 / sr, 80 * 2 / sr]
    b2 = ssg.firwin(N, wn, window='hamming', pass_zero='bandpass') # type: ignore
    wn = [80 * 2 / sr, 200 * 2 / sr]
    b3 = ssg.firwin(N, wn, window='hamming', pass_zero='bandpass') # type: ignore
    wn = 200 * 2 / sr
    b4 = ssg.firwin(N, wn, window='hamming', pass_zero='highpass') # type: ignore

    return [b1, b2, b3, b4]


def spike_removal_python(original_signal: np.ndarray, fs: float) -> np.ndarray:
    """Python implementation of schmidt spike removal"""
    # Find the window size (500 ms)
    windowsize = int(np.round(fs/2))

    # Find any samples outside of a integer number of windows:
    trailingsamples = len(original_signal) % windowsize

    # Reshape the signal into a number of windows:
    sampleframes = original_signal[:len(original_signal)-trailingsamples].reshape(-1, windowsize).T

    # Find the MAAs:
    MAAs = np.max(np.abs(sampleframes), axis=0)

    # While there are still samples greater than 3* the median value of the MAAs, then remove those spikes:
    while np.any(MAAs > np.median(MAAs)*3):
        # Find the window with the max MAA:
        window_num = np.argmax(MAAs)

        # Find the postion of the spike within that window:
        spike_position = np.argmax(np.abs(sampleframes[:, window_num]))

        # Finding zero crossings (where there may not be actual 0 values, just a change from positive to negative):
        zero_crossings = np.concatenate([(np.abs(np.diff(np.sign(sampleframes[:, window_num]))) > 1) , np.zeros(1)])

        # Find the start of the spike, finding the last zero crossing before spike position. If that is empty, take the start of the window:
        if len(np.where(zero_crossings[:spike_position] == True)[0]) == 0:
            spike_start = 1
        else:
            spike_start = np.where(zero_crossings[:spike_position] == True)[0][-1] + 1

        # Find the end of the spike, finding the first zero crossing after spike position. If that is empty, take the end of the window:
        zero_crossings[:spike_position] = 0
        if len(np.where(zero_crossings == True)[0]) == 0:
            spike_end = windowsize
        else:
            spike_end = np.where(zero_crossings == True)[0][0]

        # Set to Zero
        sampleframes[spike_start:spike_end, window_num] = 0.0001

        # Recalculate MAAs
        MAAs = np.max(np.abs(sampleframes), axis=0)

        if np.all(np.isnan(MAAs)) or np.max(MAAs) == np.max(np.abs(sampleframes), axis=0).max():
            break

    despiked_signal = sampleframes.T.flatten()

    # Add the trailing samples back to the signal:
    despiked_signal = np.append(despiked_signal, original_signal[len(despiked_signal):])

    return despiked_signal


def spike_removal(signal: np.ndarray, fs: float, matlab_location: str = "") -> np.ndarray:
    signal = np.array(signal).reshape(-1, 1)
    signal = ENG.schmidt_spike_removal(signal, float(fs))  # type: ignore
    signal = np.asarray(signal).flatten()
    return signal


def get_segment_time(pcg: np.ndarray, fs_old: float, fs_new: float, time: float = 1.25) -> list[list[int]]:
    """Gets the PCG segments based on time."""
    pcg_resampled = ssg.resample_poly(pcg, fs_new, fs_old)

    sample_increment = round(fs_new * time)
    seg_idxs = [[i, i, i, i] for i in range(sample_increment, len(pcg_resampled), sample_increment)]

    return seg_idxs


def get_hand_label_seg_pcg(path: str, filename: str) -> list[list[Any]]:

    segment_info = loadmat(os.path.join(path, f"{filename}"))
    segment_info = segment_info['state_ans']

    # Remember to adjust index for python instead of matlab
    breakpoint()

    return segment_info


def get_segment_pcg(pcg: np.ndarray, fs_old: float, fs_new: float) -> np.ndarray:
    """Gets the PCG segments using mixture of MATLAB and python."""
    pcg_resampled = ssg.resample_poly(pcg, fs_new, fs_old) # type: ignore

    pcg_resampled = ENG.butterworth_low_pass_filter(pcg_resampled, 2, 400, fs_new) # type: ignore
    pcg_resampled = ENG.butterworth_high_pass_filter(pcg_resampled, 2, 25, fs_new) # type: ignore
    pcg_resampled = np.array(pcg_resampled).reshape(-1, 1)
    pcg_resampled = ENG.schmidt_spike_removal(pcg_resampled, float(fs_new))  # type: ignore

    assigned_states = ENG.segmentation(pcg_resampled, fs_new) # type: ignore
    seg_idxs = np.asarray(ENG.get_states(assigned_states), dtype=int) - 1 # type: ignore

    return seg_idxs


def resample(signal: np.ndarray, fs_old: float, fs_new: float) -> np.ndarray:
    return ssg.resample_poly(signal, fs_new, fs_old)


def low_pass_butter(signal: np.ndarray, order: int, fc: float, fs: float) -> np.ndarray:
    wn = fc / fs
    b, a = ssg.butter(order, wn, btype="lowpass")

    return np.asarray(ssg.lfilter(b, a, signal))


def high_pass_butter(signal: np.ndarray, order: int, fc: float, fs: float) -> np.ndarray:
    wn = fc / fs
    b, a = ssg.butter(order, wn, btype="highpass")

    return np.asarray(ssg.lfilter(b, a, signal))


def delay_signal(signal: np.ndarray, delay: int) -> np.ndarray:
    """
       Delays a signal by the specified delay
    """
    hh = np.concatenate((
        np.zeros(delay),
        np.ones(1),
        np.zeros(delay)),
        dtype="float32"
    )

    delayed_signal = np.asarray(ssg.lfilter(hh.flatten(), 1, signal))

    return delayed_signal


def correlations(xdn: np.ndarray, ydn: np.ndarray, FL: int) -> tuple[np.ndarray, np.ndarray, float, np.ndarray]:
    """
        Calculates the correlation matrix and crosscorrelation vector
    """
    DL = max(np.shape(xdn))
    RXX: np.ndarray = np.zeros((FL, FL), dtype="float32")
    rxy: np.ndarray = np.zeros((FL, 1), dtype="float32")
    ryy: float = 0

    yp: np.ndarray = np.zeros(DL, dtype="float32")
    for ii in range(FL, DL, 1):
        xv = xdn[ii:ii-FL:-1].reshape(-1, 1)
        RXX = RXX + xv @ np.transpose(xv)
        rxy = rxy + xv * ydn[ii]
        ryy = ryy + ydn[ii] ** 2
        yp[ii] = ydn[ii]

    return RXX, rxy, ryy, yp


def optimal_weights(RXX: np.ndarray, rxy: np.ndarray, ryy: float, FL: int, DL: float) -> np.ndarray:
    """
        Finds the optimal weights
    """
    # TODO : Look into if float32 removes the hacks.
    err0 = 0.005

    egv = np.linalg.eigvals(RXX)
    w = np.linalg.lstsq(RXX + err0 * (egv[np.argmax(np.abs(egv))]) *
                        np.eye(RXX.shape[0]), rxy, rcond=-1)[0]  # To extract the correct soln
    err = float(((ryy - 2 * np.transpose(w) @ rxy + np.transpose(w) @ RXX @ w) / (DL - FL))[0])  # extract as a float

    passes = 0
    err_prev = 0
    while abs(err0 - err) > 0.0001 or passes <= 2:
        if err == err_prev:  # To get the same result as matlab.
            passes += 1
        w = np.linalg.lstsq(RXX + err0 * (egv[np.argmax(np.abs(egv))]) *
                            np.eye(RXX.shape[0]), rxy, rcond=-1)[0]  # To extract the correct soln
        err_prev = err0
        err0 = err
        err = float(((ryy - 2 * np.transpose(w) @ rxy + np.transpose(w)
                    @ RXX @ w) / (DL - FL))[0])  # To extract as a float

    return w


def weiner_filter(xdn: np.ndarray, ydn: np.ndarray, FL: int, DL: float) -> np.ndarray:
    """
        Runs the weiner filter algorithm
    """
    RXX, rxy, ryy, yp = correlations(xdn, ydn, FL)
    w = optimal_weights(RXX, rxy, ryy, FL, DL)

    # apply weiner filter
    yhat = ssg.lfilter(w.flatten(), 1, xdn)
    en1 = np.transpose(yp) - yhat

    return en1


def milan_hp(signal: np.ndarray, fc: float, fs: float) -> np.ndarray:
    """
        IIR High pass filter to help the noise cancellation weiner filter
    """
    wn = fc / fs
    b, a = ssg.butter(2, wn, btype="highpass")

    signal = np.asarray(ssg.lfilter(b, a, signal))

    return signal


def noise_canc(xdn: np.ndarray, ydn: np.ndarray, fc: float = 150, fs: float = 1000) -> np.ndarray:
    """
    Noise cancellation using weiner filter and hp

    xdn is background noise,
    ydn is the signal with background noise
    """
    FL = 64
    DL = max(np.shape(xdn))

    ydn = delay_signal(ydn, math.floor(FL/2)) 
    xdn_hp = milan_hp(xdn, fc, fs)

    en = weiner_filter(xdn_hp, ydn, FL, DL)

    return en


def wavelet_denoise(signal: np.ndarray, wavelet: str) -> np.ndarray:
    coeffs = pywt.wavedec(signal, wavelet)

    threshold = np.sqrt(2*np.log(len(signal)))
    coeffs[1:] = (pywt.threshold(i, value=threshold, mode='soft') for i in coeffs[1:])

    denoised_signal = pywt.waverec(coeffs, wavelet)

    return denoised_signal