import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import UnivariateSpline


def stretch_signal_using_spline(signal):
    """
    Stretch a signal to be twice as long using spline interpolation.

    Parameters:
    - signal: array-like
        Original signal values.

    Returns:
    - stretched_signal: array-like
        Stretched signal values.
    """

    # Create original x-values based on the length of the signal
    x = np.arange(len(signal))

    # Create new x-values that are twice as dense
    x_new = np.linspace(x[0], x[-1], 2*len(x))

    # Perform spline interpolation
    stretched_signal = spline_interpolation(x, signal, x_new)

    return stretched_signal


def spline_interpolation(x, y, x_new):
    """Helper function to perform spline interpolation."""
    spline = UnivariateSpline(x, y, s=0)
    return spline(x_new)


def stretch_signal(signal):

    len_sig = len(signal)

    original_indices = np.linspace(0, len_sig - 1, len_sig)

    stretched_indices = np.linspace(0, len_sig - 1, 2 * len_sig)

    stretched_signal = np.interp(stretched_indices, original_indices, signal)

    return stretched_signal


# Example usage:
t = np.linspace(0, 2*np.pi, 15)
signal = np.sin(t) + 2 * np.cos(t)
stretched_signal = stretch_signal_using_spline(signal)
another_stretched_signal = stretch_signal(signal)
t = np.linspace(0, 2*np.pi, 30)
signal = np.sin(t) + 2 * np.cos(t)

# If you want to visualize the result:
plt.figure(figsize=(10, 6))
plt.plot(signal, label='Original Signal')
plt.plot(stretched_signal, label='Stretched Signal', linestyle='--')
plt.plot(another_stretched_signal, label='Another Stretched Signal', linestyle='--')
plt.legend()
plt.show()
