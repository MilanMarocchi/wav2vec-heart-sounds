import numpy as np
import torch
import scipy

ENG = None


def start_matlab(matlab_location):
    if matlab_location != '':
        try:
            import matlab.engine
            global ENG
            ENG = matlab.engine.start_matlab()
            ENG.addpath(ENG.genpath(matlab_location), nargout=0)  # type: ignore
            print('STARTED MATLAB')
        except ImportError as e:
            print('Matlab engine not installed --- trying anyway')
            print(e)


def stop_matlab():
    if ENG is not None:
        ENG.exit()  # type: ignore
        print('STOPPED MATLAB')


def interpolate_nans(a):
    mask = np.isnan(a)
    a[mask] = np.interp(np.flatnonzero(mask),
                        np.flatnonzero(~mask),
                        a[~mask])
    return a


def create_spectrogram(signal, transform):
    spectrogram = transform(signal)
    spectrogram = 20 * torch.log10(torch.clamp(spectrogram, min=1e-5)) - 20
    spectrogram = torch.clamp((spectrogram + 100) / 100, 0.0, 1.0)
    return spectrogram


def resample(signal, old_fs, new_fs):
    return scipy.signal.resample_poly(signal, new_fs, old_fs)


def standardise_signal(signal):
    signal = signal.astype(np.float64)
    signal = interpolate_nans(signal)

    signal -= np.mean(signal)
    signal /= np.max(np.abs(signal))
    signal = np.clip(signal, -1, 1)

    return signal


def standardise_torch_signal(signal):
    return torch.from_numpy(standardise_signal(signal.cpu().numpy())).squeeze(0).float()


def fade_signal(signal, num_fade_samples):
    fade_in = np.linspace(0, 1, num_fade_samples)
    fade_out = np.linspace(1, 0, num_fade_samples)
    signal[:num_fade_samples] *= fade_in
    signal[-num_fade_samples:] *= fade_out
    return signal


def spike_removal(signal, fs):
    signal = np.array(signal).reshape(-1, 1)
    signal = ENG.schmidt_spike_removal(signal, float(fs))  # type: ignore
    signal = np.asarray(signal).flatten()
    return signal


def bandpass(signal, fs, low, high):
    nyquist_freq = 0.5 * fs
    low /= nyquist_freq
    high /= nyquist_freq

    sos = scipy.signal.butter(1, [low, high], 'bandpass', analog=False, output='sos',)
    signal = scipy.signal.sosfiltfilt(sos, signal)

    return signal


# def wavefilt(signal, wavelet, level):
#     signal = standardise_signal(signal)
#     coeffs = pywt.wavedec(signal, wavelet, level)
#
#     sigma = np.median(np.abs(coeffs[-1] - np.median(coeffs[-1]))) / 0.6745
#     var = np.var(coeffs[-1])
#     threshold = sigma**2 / np.sqrt(max(var - sigma**2, 0))
#
#     coeffs = [pywt.threshold(coeff, threshold, mode='soft')
#               if i > 0
#               else coeff
#               for i, coeff in enumerate(coeffs)]
#     return pywt.waverec(coeffs, wavelet)


def notchfilter(signal, fs, notch, Q):
    nyquist_freq = 0.5 * fs
    notch /= nyquist_freq

    b, a = scipy.signal.iirnotch(notch, Q)
    signal = scipy.signal.filtfilt(b, a, signal)

    return signal


def pre_filter_ecg(signal, fs):
    signal = notchfilter(signal, fs, 50, 55)
    signal = notchfilter(signal, fs, 60, 55)
    signal = notchfilter(signal, fs, 100, 55)
    signal = notchfilter(signal, fs, 120, 55)
    signal = bandpass(signal, fs, 0.25, 150)
    # signal = wavefilt(signal, 'sym4', 4)
    # signal = bandpass(signal, fs, 0.5, 70)
    return signal


def mid_filter_ecg(signal, fs):
    signal = bandpass(signal, fs, 0.25, 150)
    return signal


def post_filter_ecg(signal, fs):
    signal = notchfilter(signal, fs, 50, 55)
    signal = notchfilter(signal, fs, 60, 55)
    signal = notchfilter(signal, fs, 100, 55)
    signal = notchfilter(signal, fs, 120, 55)
    return bandpass(signal, fs, 0.25, 70)


def pre_filter_pcg(signal, fs):
    signal = bandpass(signal, fs, 2, 500)
    # signal = spike_removal(signal, fs)
    # signal = wavefilt(signal, 'db10', 4)
    # signal = bandpass(signal, fs, 5, 400)
    return signal


def mid_filter_pcg(signal, fs):
    signal = bandpass(signal, fs, 2, 500)
    return signal


def post_filter_pcg(signal, fs):
    return bandpass(signal, fs, 5, 450)


def get_pcg_segs_idx(pcg, old_fs, new_fs):
    pcg = resample(pcg, old_fs, new_fs)

    pcg = ENG.butterworth_low_pass_filter(pcg, 2, 400, new_fs)  # type: ignore
    pcg = ENG.butterworth_high_pass_filter(pcg, 2, 25, new_fs)  # type: ignore
    pcg = np.array(pcg).reshape(-1, 1)
    pcg = ENG.schmidt_spike_removal(pcg, float(new_fs))  # type: ignore

    assigned_states = ENG.segmentation(pcg, new_fs)  # type: ignore
    seg_idxs = np.asarray(ENG.get_states(assigned_states), dtype=int) - 1  # type: ignore

    return seg_idxs, new_fs


def add_chirp(audio_signal, fs):
    t = np.arange(len(audio_signal)) / fs

    chirp_signal = scipy.signal.chirp(t, f0=0, f1=fs/2, t1=t[-1], method='linear')
    chirp_signal = (chirp_signal / np.max(np.abs(chirp_signal))) * max(0.5, np.max(np.abs(audio_signal)))

    return audio_signal + chirp_signal
