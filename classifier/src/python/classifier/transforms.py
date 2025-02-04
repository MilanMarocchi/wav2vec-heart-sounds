"""
    transforms.py
    Author: Milan Marocchi
    
    Purpose: Create transforms for ml.
"""

import random
import torch
from torchvision import transforms
import numpy as np
from processing.filtering import (
    time_stretch_crop,
    band_stop,
)

class RandomStretch:
    """Applies a random stretch to a signal"""

    def __init__(self, fs: int):
        self.fs = fs

    def __call__(self, audio: np.ndarray):
        if random.random() > 0.8:

            stretch_factor = 0.96 + random.random() * (1.04 - 0.96)
            audio = time_stretch_crop(audio, self.fs, stretch_factor)

        return audio

class RandomTimeFreqMask:
    """Applies a random line mask in the spectrogram of the audio"""
    def __init__(self, thickness: float, fs: int):
        self.thickness = thickness
        self.fs = fs

    def __call__(self, audio: np.ndarray):
        sig_len = len(audio)
        assert sig_len > 1000, "Correct way to get sig len for multi-channel"

        time_thickness = int(self.thickness * sig_len)
        freq_thickness = int(self.thickness * self.fs)

        if random.random() > 0.8:
            if random.random() > 0.5:
                # Time masking
                time = random.randint(0, len(audio) - time_thickness - 1)

                # check multi channel
                if audio.ndim == 1:
                    audio[time:time + time_thickness] = 0
                else:
                    num_channels = audio.shape[1]
                    channel = random.randint(0, num_channels - 1)
                    audio[time:time + time_thickness, channel]
            else:
                # Frequency masking
                frequency = random.randint(1, self.fs - freq_thickness - 1)

                # check multi channel
                if audio.ndim == 1:
                    audio = band_stop(audio, self.fs, frequency, frequency + freq_thickness)
                else:
                    num_channels = audio.shape[1]
                    channel = random.randint(0, num_channels - 1) 
                    audio[:, channel] = band_stop(audio[:, channel], self.fs, frequency, frequency + freq_thickness)

        return audio

def get_pil_transform_numpy(size: int = 224) -> transforms.Compose:
    """
    Transform to get an image to show for xai but outputting a numpy image
    """

    transf = transforms.Compose([
        transforms.Resize((size, size)),
        transforms.ToTensor(),
        transforms.Lambda(lambda x: (x.numpy()).astype(np.float64))  # Convert tensor to np array
    ])

    return transf

def numpy_to_tensor(y):
    return torch.from_numpy(y.copy())

def get_pil_transform(size: int = 224) -> transforms.Compose:
    """
    Transform to get an image to show for xai
    """
    transf = transforms.Compose([
        transforms.Resize((size, size)),
        transforms.ToTensor(),
    ])

    return transf


def get_normalise_transform(size: int = 224) -> transforms.Compose:
    """
    Applies the pre-processing transforms to the image as done for classification
    """
    transf = transforms.Compose([
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    return transf


def get_preprocess_transform(size: int = 224) -> transforms.Compose:
    """
    Applies the pre-processing transforms to the image as done for classification
    """
    transf = transforms.Compose([
        transforms.Resize((size, size)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    return transf


def create_data_transforms(is_inception: bool = False) -> dict[str, transforms.Compose]:
    """
    Creates data transforms for training and classifying
    """
    size = 299 if is_inception else 224

    data_transforms = {
        'train': transforms.Compose([
            transforms.Resize((size, size)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'valid': transforms.Compose([
            transforms.Resize((size, size)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'test': transforms.Compose([
            transforms.Resize((size, size)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
    }

    return data_transforms

def create_audio_data_transforms(fs: int) -> dict[str, transforms.Compose]:

    thickness = 0.09

    data_transforms = {
        'train': transforms.Compose([
            numpy_to_tensor,
            RandomStretch(fs),
            RandomTimeFreqMask(thickness, fs), # type: ignore,
            RandomTimeFreqMask(thickness / 2, fs), # type: ignore
        ]),
        'valid': transforms.Compose([
            numpy_to_tensor,
        ]),
        'test': transforms.Compose([
            numpy_to_tensor,
        ])
    }

    return data_transforms
