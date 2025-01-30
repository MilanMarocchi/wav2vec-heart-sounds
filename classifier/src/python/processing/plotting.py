"""
  author: Milan Marocchi
  purpose : Contains usefull plotting abstractions
"""

from ssqueezepy import cwt, imshow

import matplotlib as mpl
mpl.use('Agg')  # type: ignore
import matplotlib.pyplot as plt
from matplotlib.colors import Colormap
import librosa.display
import numpy as np


def plot_dwt(details: np.ndarray, approx: np.ndarray, xlim=(-300,300), **line_kwargs):
    """For plotting the coefficients from the dwt"""
    xvals = np.zeros(0)
    for i in range(len(details)):
        plt.subplot(len(details)+1,1,i+1)
        d = details[len(details)-1-i]
        half = len(d)//2
        xvals = np.arange(-half,-half+len(d))* 2**i
        plt.plot(xvals, d, **line_kwargs)
        plt.xlim(xlim)
        plt.title("detail[{}]".format(i))
    plt.subplot(len(details)+1,1,len(details)+1)
    plt.title("approx")
    plt.plot(xvals, approx, **line_kwargs)
    plt.xlim(xlim)

#####
# These functions are modified version from the librosa source
#####
def power_to_db(S: np.ndarray, ref_value: float = 1.0, amin:float = 1e-10, top_db: float = 80.0) -> np.ndarray:
    S = np.asarray(S)

    magnitude = S

    log_spec = 10.0 * np.log10(np.maximum(amin, magnitude))
    log_spec -= 10 * np.log10(np.maximum(amin, ref_value))

    log_spec[np.isnan(log_spec)] = np.inf

    if top_db is not None:
        log_spec[log_spec != np.inf] = np.maximum(
            log_spec[log_spec != np.inf], log_spec[log_spec != np.inf].max() - top_db)

    return log_spec


def amplitude_to_db(S: np.ndarray, ref_value: float = 1.0, amin: float = 1e-5, top_db: float = 80.0) -> np.ndarray:
    S = np.asarray(S)

    magnitude = np.abs(S)

    ref_value = np.abs(1.0)
    amin = 1e-5

    power = np.square(magnitude, out=magnitude)

    return power_to_db(power, ref_value=ref_value**2, amin=amin**2, top_db=top_db)
#####


class SignalPlotGenerator():
    """
    Interface for a plot generator to create plots
    """

    def __init__(self):
        raise NotImplementedError()

    def plot(self, signal: np.ndarray, Fs: int):
        raise NotImplementedError()

    def set_ecg(self, ecg: bool):
        self.ecg = ecg

    def get_cmap(self) -> Colormap:
        cmap_jet = plt.cm.get_cmap('jet')
        return cmap_jet


# NOTE: Now assuming that multiple arrays passed in so need to iterate through all of them.
def plot_signal(signal: np.ndarray, plotter: SignalPlotGenerator, Fs: int = 1000, title: str = "", labels: bool = True, colorbar: bool = True, hide_axis: bool = False, path: str = ""):
    """
    For plotting spectrograms/scalograms nicely.
    """
    fig = plt.figure()

    plotter.plot(signal, Fs)

    if hide_axis:
        plt.axis('off')
    if colorbar:
        plt.colorbar()
    if title != "":
        plt.title(title)
    if labels:
        plt.ylabel("Frequency")
        plt.xlabel("Time")

    if path == "":
        plt.show()
        plt.close()
    else:
        plt.savefig(path, bbox_inches='tight', pad_inches=0)
        plt.close(fig)


class STFTPlotGenerator(SignalPlotGenerator):
    """
    Generates STFTs
    """

    def __init__(self):
        self.ecg: bool = False
        self.clim: bool = False

    def spec_segment(self, y: np.ndarray) -> np.ndarray:
        return librosa.stft(y=y, center=False, pad_mode="symmetric", n_fft=100, hop_length=50)

    def plot(self, signal: np.ndarray, Fs: int = 1000):
        if self.ecg:
            # Tweak so it displays better
            signal[:, -1] = 0.015 * signal[:, -1]

        y = signal[:, 0]
        S = self.spec_segment(y)

        for i in range(1, len(signal[0])):
            y = signal[:, i]
            S_1 = self.spec_segment(y)

            # Collect the rest of the spectrogram
            S = np.concatenate((S, S_1), axis=1)


        # Convert to db and display
        S_dB = amplitude_to_db(S)
        librosa.display.specshow(S_dB, sr=Fs, cmap=self.get_cmap(), shading='gouraud')

        plt.clim([-50, 50]) # type: ignore


class MelSTFTPlotGenerator(SignalPlotGenerator):
    """
    Generates Mel-STFTs
    """

    def __init__(self):
        self.ecg: bool = False

    def spec_segment(self, y: np.ndarray, Fs: int) -> np.ndarray:
        return librosa.feature.melspectrogram(y=y, sr=Fs, center=False, pad_mode="symmetric", n_mels=32, n_fft=100, hop_length=50)

    def plot(self, signal: np.ndarray, Fs=1000):
        if self.ecg:
            # Tweak so it displays better
            signal[:, -1] = 0.01 * signal[:, -1]

        y = signal[:, 0]
        S = self.spec_segment(y, Fs)

        for i in range(1, len(signal[0])):
            y = signal[:, i]
            S_1 = self.spec_segment(y, Fs)

            # Collect the rest of the spectrogram
            S = np.concatenate((S, S_1), axis=1)

        # Convert to db and display
        S_dB = amplitude_to_db(S)
        img = librosa.display.specshow(S_dB, sr=Fs, cmap=self.get_cmap(), shading='gouraud')


class ScalogramPlotGenerator(SignalPlotGenerator):
    """
    Generates Scalograms
    """

    def __init__(self):
        self.ecg: bool = False

    def scal_segment(self, y: np.ndarray) -> np.ndarray:
        Wx_sig, _ = cwt(y, 'morlet') # type: ignore

        return np.asarray(Wx_sig)

    def plot(self, signal: np.ndarray, Fs: int = 1000):
        if self.ecg:
            signal[:, :-1] = 11 * signal[:, :-1]

        y = signal[:, 0]
        Wx = self.scal_segment(y)

        for i in range(1, len(signal[0])):
            y = signal[:, i]
            Wx_sig = self.scal_segment(y)
            Wx = np.concatenate((Wx, Wx_sig), axis=1) # type: ignore

        # Display
        imshow(Wx, ticks=False, borders=False, show=False, abs=1, cmap=self.get_cmap())


class SignalPlotterFactory():
    """
    Creates the appropriate signal plotter
    """

    def create(self, transform: str) -> SignalPlotGenerator:
        if transform == "stft":
            return STFTPlotGenerator()
        elif transform == "mel-stft":
            return MelSTFTPlotGenerator()
        elif transform == "wave":
            return ScalogramPlotGenerator()
        else:
            raise Exception(f"Invalid transform passed to plotter factory: {transform}")
