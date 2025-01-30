"""
    fileio.py
    Author : Milan Marocchi

    Reading and writing records
"""

import wfdb
import json
from json import JSONEncoder
import numpy as np
import pandas as pd
import scipy.io as sio

from processing.filtering import normalise_signal


def read_ticking_PCG(filename, channel, noise_mic=False, max_len=None):
    """Read in the PCG from the ticking heart data."""
    channels = {
        '1': 1,
        '2': 2,
        '3': 4,
        '4': 5,
        '5': 6,
        '6': 7,
        '7': 8,
    }

    filename += ".wav"
    signal, fs = read_signal_wav(filename)
    wav_channel = channels[str(channel)]
    wav_channel = wav_channel + 7 if noise_mic else wav_channel - 1

    return signal[:, wav_channel], fs


def read_ticking_ECG(filename, max_len=None):
    """Read in the ECG from the ticking heart data."""
    ecg_channel = 10
    filename += ".wav"
    signal, fs = read_signal_wav(filename)

    return signal[:, ecg_channel], fs


def read_signal_wav(filename):
    """
    Reads in a signal from a wav file then converts it into the same format that matlab would output.
    Outputs the sampling freq as well as the signal
    """
    if ".wav" not in filename:
        filename += ".wav"

    Fs, signal = sio.wavfile.read(filename)

    if signal.dtype == np.int16:
        max_val = np.iinfo(np.int16).max
    elif signal.dtype == np.int32:
        max_val = np.iinfo(np.int32).max
    elif signal.dtype == np.int64:
        max_val = np.iinfo(np.int64).max
    elif signal.dtype == np.float32 or signal.dtype == np.float64:
        return signal.astype(np.float32), Fs
    else:
        raise ValueError("Unsupported data type")

    # Convert to float 32
    signal = (signal / max_val).astype(np.float32)

    return signal, Fs


def save_signal_wav(signal, fs, path):
    """
    Saves a signal as a wav file to the specified path.
    """
    if ".wav" not in path:
        path += ".wav"

    sio.wavfile.write(path, fs, signal)


def get_cinc_record(path, max_len=None):

    header = wfdb.rdheader(path)
    sig_len = header.sig_len
    fs = header.fs

    if max_len is None:
        target_sig_len = sig_len
    else:
        target_sig_len = min(round(max_len * fs), sig_len)

    rec = wfdb.rdrecord(path, sampfrom=0, sampto=target_sig_len)
    return rec


def get_cinc_sig(path, name, max_len=None):

    record = get_cinc_record(path, max_len)
    fs = record.fs
    signal = record.p_signal[:, record.sig_name.index(name)]
    signal = normalise_signal(signal)

    return signal , fs


def get_patients(path, training_a=False):
    """
    Gets the labels from the cinc data
    """
    training_a_exclude = ['a0041', 'a0117', 'a0220', 'a0233']

    # Check to see if there are headers 
    with open(path, 'r') as file_in:

        line = file_in.readline()
        if line[0] == "#":
            line = file_in.readline()

        if 'patient' in line and 'abnormality' in line:
            patients_header = 'patient'
            label_header = 'abnormality'
            header = 0
        else:
            patients_header = 0
            label_header = 1
            header = None

    patient_data = pd.read_csv(path, comment='#', header=header)
    patients = list(patient_data[patients_header])
    labels = list(patient_data[label_header])

    patients = [patient for patient in patients if patient not in training_a_exclude]

    return patients, labels


def get_patients_split(path, subset):
    """
    Gets the label data from the splits file
    """
    patients = pd.read_csv(path, comment='#')
    patients = patients[patients['split'] == subset]

    return patients


def get_patients_segments(path):
    """
    Gets the segment information from the patient file
    """
    data = read_json_numpy(path)

    return data['segments'], data['last_index'], data['fs']



class NumpyEncoder(JSONEncoder):
    """
    Class to encode numpy data to a list to be stored in a json file.
    """

    def default(self, o):
        if isinstance(o, np.ndarray):
            return o.tolist()
        return JSONEncoder.default(self, o)


def write_json_numpy(data, filepath):
    """
    Writes to a json using the numpy encoder
    """
    # NOTE: Add error checking so format is enforced.

    with open(filepath, "w") as out_file:
        json.dump(data, out_file, cls=NumpyEncoder)


def read_json_numpy(filepath):
    """
    Reads from a json file and decodes the array to a numpy array.
    Excepts the format that is used when written
    """
    with open(filepath, "r") as in_file:
        json_data = json.load(in_file)

    return json_data
