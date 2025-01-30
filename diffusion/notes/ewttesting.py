import numpy as np
import matplotlib.pyplot as plt
from PyEMD import EMD
from scipy.signal import find_peaks

signal = "notes/sig3.csv"  # sig1,sig2,sig3,eeg or any other csv file with a signal

data = np.loadtxt(signal, delimiter=",")

if len(data.shape) > 1:

    f = data[:, 0]

else:

    f = data

# f = f - np.mean(f)


# Load the PCG signal (replace with your data)
pcg_signal = f

print(pcg_signal.shape)


# Perform Empirical Mode Decomposition (EMD)
emd = EMD()
imfs = emd(pcg_signal)

segment_length = len(pcg_signal) // 50  # Adjust the length as needed

print(f'{segment_length=}')

# Randomly select a segment from the signal (you can use other selection methods)
start_index = np.random.randint(0, len(pcg_signal) - segment_length)
end_index = start_index + segment_length

# Estimate the noise standard deviation (sigma)
# Using a representative portion of an IMF (you can adjust the selection)
representative_portion = imfs[0][start_index:end_index]  # Adjust indices
sigma = np.std(representative_portion)

# Define a threshold for denoising (you can adjust this)
threshold = 3 * sigma  # Adjust as needed

# Apply soft thresholding to each IMF
denoised_imfs = [np.where(np.abs(imf) < threshold, 0, imf) for imf in imfs]

# Reconstruct the denoised PCG signal
denoised_pcg_signal = np.sum(denoised_imfs, axis=0)

# Optional: Further post-processing, such as peak detection
peaks, _ = find_peaks(denoised_pcg_signal, height=0.2, distance=200)

# Plot the original and denoised PCG signals
plt.figure(figsize=(12, 6))
plt.subplot(211)
plt.plot(pcg_signal, label='Original PCG Signal')
plt.title('Original PCG Signal')
plt.subplot(212)
plt.plot(denoised_pcg_signal, label='Denoised PCG Signal')
plt.plot(peaks, denoised_pcg_signal[peaks], 'ro', label='Detected Peaks')
plt.title('Denoised PCG Signal with Peak Detection')
plt.tight_layout()
plt.legend()
plt.show()
