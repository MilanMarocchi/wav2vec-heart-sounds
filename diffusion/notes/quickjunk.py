import matplotlib.pyplot as plt
import numpy as np


def equal_voltage_crossfade(t):
    # Define a_e(t)
    a_e = 0.5 * np.ones_like(t)

    # Define a_o(t)
    a_o = np.where(np.abs(t) <= 1,
                   (9/16) * np.sin(np.pi/2 * t) + (1/16) * np.sin(3*np.pi/2 * t),
                   0.5 * np.sign(t))

    # Calculate a(t)
    a = a_e + a_o

    return a


def equal_power_crossfade(t):
    # Define a_e(t)
    a_e = np.where(np.abs(t) <= 1,
                   0.5 * (np.sqrt(0.5 * (1 + t)) + np.sqrt(0.5 * (1 - t))),
                   0.5 * np.ones_like(t))

    # Define a_o(t)
    a_o = np.where(np.abs(t) <= 1,
                   0.5 * (np.sqrt(0.5 * (1 + t)) - np.sqrt(0.5 * (1 - t))),
                   0.5 * np.sign(t))

    # Calculate a(t)
    a = a_e + a_o

    return a


def crossfade(x, y, num_samples):
    # Ensure the signals are numpy arrays
    x = np.array(x)
    y = np.array(y)

    # Calculate the correlation coefficient r
    r = np.corrcoef(x[:num_samples], y[:num_samples])[0, 1]

    # Define the time variable t
    t = np.linspace(-1, 1, num_samples)

    # Define the odd part o(t)
    o = np.piecewise(t, [t < -1, (t >= -1) & (t < 1), t >= 1],
                     [lambda t: 0.5 * np.sign(t),
                      lambda t: (9/16)*np.sin(np.pi/2 * t) + (1/16)*np.sin(3*np.pi/2 * t),
                      lambda t: 0.5 * np.sign(t)])

    # Calculate the even part e(t)
    e = np.sqrt((0.5 / (1 + r)) - ((1 - r) / (1 + r)) * o**2)

    # Calculate the crossfade function a(t)
    a = e + o

    # Perform the crossfade
    crossfaded_signal = a * y[:num_samples] + (1 - a) * x[:num_samples]

    return crossfaded_signal


def crossfade_and_concatenate(curr_signal, next_signal, num_fade_samples, equal_length):
    # Ensure the signals are numpy arrays
    curr_signal = np.array(curr_signal)
    next_signal = np.array(next_signal)

    # Calculate the correlation coefficient r
    r = abs(np.corrcoef(curr_signal[-num_fade_samples:], next_signal[:num_fade_samples])[0, 1])

    print(f'{r=}')

    # Define the time variable t
    t = np.linspace(-1, 1, num_fade_samples)

    # Define the odd part o(t)
    o = np.piecewise(t, [t < -1, (t >= -1) & (t < 1), t >= 1],
                     [lambda t: 0.5 * np.sign(t),
                      lambda t: (9/16)*np.sin(np.pi/2 * t) + (1/16)*np.sin(3*np.pi/2 * t),
                      lambda t: 0.5 * np.sign(t)])

    # Calculate the even part e(t)
    e = np.sqrt((0.5 / (1 + r)) - ((1 - r) / (1 + r)) * o**2)

    # Calculate the crossfade function a(t)
    a = e + o

    # Perform the crossfade
    crossfaded_segment = curr_signal[-num_fade_samples:] * (1 - a) + next_signal[:num_fade_samples] * a

    if equal_length:
        crossfaded_segment = stretch_signal(crossfaded_segment)

    # Concatenate the signals
    concatenated_signal = np.hstack((curr_signal[:-num_fade_samples],
                                     crossfaded_segment,
                                     next_signal[num_fade_samples:]))

    return concatenated_signal


def crossfade_and_concatenate_linear(curr_signal, next_signal, num_fade_samples, equal_length):

    fade_out = np.linspace(1, 0, num_fade_samples)
    fade_in = np.linspace(0, 1, num_fade_samples)

    crossfaded_segment = (curr_signal[-num_fade_samples:] * fade_out +
                          next_signal[:num_fade_samples] * fade_in)

    if equal_length:
        crossfaded_segment = stretch_signal(crossfaded_segment)

    concatenated_signal = np.hstack((curr_signal[:-num_fade_samples],
                                     crossfaded_segment,
                                     next_signal[num_fade_samples:]))

    return concatenated_signal


def crossfade_and_concatenate_power(curr_signal, next_signal, num_fade_samples, equal_length):

    t = np.linspace(0, np.pi / 2, num_fade_samples)

    fade_out_curve = np.cos(t)
    fade_in_curve = np.sin(t)

    crossfaded_segment = (curr_signal[-num_fade_samples:] * fade_out_curve +
                          next_signal[:num_fade_samples] * fade_in_curve)

    if equal_length:
        crossfaded_segment = stretch_signal(crossfaded_segment)

    concatenated_signal = np.hstack((curr_signal[:-num_fade_samples],
                                     crossfaded_segment,
                                     next_signal[num_fade_samples:]))

    return concatenated_signal


def crossfade_and_concatenate_mix(curr_signal, next_signal, num_fade_samples, equal_length):

    # Calculate the correlation coefficient r
    r = abs(np.corrcoef(curr_signal[-num_fade_samples:], next_signal[:num_fade_samples])[0, 1])

    print(f'{r=}')

    # Define the time variable t
    t = np.linspace(-1, 1, num_fade_samples)

    # Define the odd part o(t)
    o = np.piecewise(t, [t < -1, (t >= -1) & (t < 1), t >= 1],
                     [lambda t: 0.5 * np.sign(t),
                      lambda t: (9/16)*np.sin(np.pi/2 * t) + (1/16)*np.sin(3*np.pi/2 * t),
                      lambda t: 0.5 * np.sign(t)])

    # Calculate the even part e(t)
    e = np.sqrt((0.5 / (1 + r)) - ((1 - r) / (1 + r)) * o**2)

    # Calculate the crossfade function a(t)
    a = e + o

    t_equal_power = np.linspace(0, np.pi / 2, num_fade_samples)

    fade_out_equal_power = np.cos(t_equal_power)
    fade_out_linear = np.linspace(1, 0, num_fade_samples)
    fade_out_corr = 1 - a

    fade_in_equal_power = np.sin(t_equal_power)
    fade_in_linear = np.linspace(0, 1, num_fade_samples)
    fade_in_corr = a

    fade_in = (fade_in_linear + fade_in_equal_power + fade_in_corr) / 3
    fade_out = (fade_out_linear + fade_out_equal_power + fade_out_corr) / 3

    # fade_in = fade_in_linear
    # fade_out = fade_out_linear

    # Perform the crossfade
    crossfaded_segment = curr_signal[-num_fade_samples:] * fade_out + next_signal[:num_fade_samples] * fade_in

    if equal_length:
        crossfaded_segment = stretch_signal(crossfaded_segment)

    # Concatenate the signals
    concatenated_signal = np.hstack((curr_signal[:-num_fade_samples],
                                     crossfaded_segment,
                                     next_signal[num_fade_samples:]))

    return concatenated_signal


def stretch_signal(signal):
    # Create an array of indices for the original signal
    original_indices = np.linspace(0, len(signal) - 1, len(signal))

    # Create an array of indices for the stretched signal
    stretched_indices = np.linspace(0, len(signal) - 1, 2 * len(signal) - 1)

    # Use numpy's interp function to perform linear interpolation
    stretched_signal = np.interp(stretched_indices, original_indices, signal)

    return stretched_signal


# Example usage:
t = np.linspace(-1.5, 1.5, 1000)

a_sig1 = 5 * np.cos(t)
a_sig2 = 8 * np.cos(3*t) + 4

# a_sig1 = 1 * np.cos(t)
# a_sig2 = 1.1 * -np.sin(t)

fig, (ax1, ax2) = plt.subplots(2, 1)

for i, (func, desc) in enumerate([(crossfade_and_concatenate, 'experiment'),
                                  (crossfade_and_concatenate_mix, 'mix'),
                                  (crossfade_and_concatenate_power, 'power'),
                                  (crossfade_and_concatenate_linear, 'linear')]):
    a_res = func(a_sig1[:len(t)//2], a_sig2[len(t)//2:], 100, equal_length=False)
    ax1.plot(t[:len(a_res)], a_res, label=f'{desc}', linestyle='--')
    a_res = func(a_sig1[:len(t)//2], a_sig2[len(t)//2:], 100, equal_length=True)
    ax2.plot(t[:len(a_res)], a_res, label=f'{desc} equal', linestyle='--')

# You can plot these using matplotlib to visualize the functions
# plt.plot(t, a_voltage, label="Equal Voltage")
# plt.plot(t, a_power, label="Equal Power")
ax1.plot(t, a_sig1, label="Cos1")
ax1.plot(t, a_sig2, label="Cos2")
ax2.plot(t, a_sig1, label="Cos1")
ax2.plot(t, a_sig2, label="Cos2")
fig.legend()
plt.show()
