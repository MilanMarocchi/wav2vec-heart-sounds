#!/usr/bin/env pipenv-shebang
"""
    generate_aug_data.py
    Author: Milan Marocchi

    Purpose: Generate augmented data
"""

from concurrent.futures import ProcessPoolExecutor, as_completed
import click
import os
import numpy as np
import wfdb
import logging
logging.basicConfig(level=logging.ERROR)

from tqdm.auto import tqdm

from util.fileio import (
    get_patients,
    save_signal_wav,
    get_cinc_sig,
    read_ticking_PCG,
)

from processing.augmentation import (
    augment_pcg,
    augment_multi_pcg,
    augment_signals
)

def create_multi_wav(pcg_multi):
    # First 7 channels are heart facing
    # Last 7 channels are background mic facing
    ticking_wav = np.zeros((len(pcg_multi[0]), 16))

    # Create massive wav file so it is in the same format.
    ticking_wav[:, 0] = pcg_multi[0]
    ticking_wav[:, 1] = pcg_multi[1]
    ticking_wav[:, 2] = pcg_multi[2]
    ticking_wav[:, 4] = pcg_multi[3]
    ticking_wav[:, 5] = pcg_multi[4]
    ticking_wav[:, 6] = pcg_multi[5]
    ticking_wav[:, 7] = pcg_multi[6]
    ticking_wav[:, 8] = pcg_multi[7]
    ticking_wav[:, 9] = pcg_multi[8]
    ticking_wav[:, 11] = pcg_multi[9]
    ticking_wav[:, 12] = pcg_multi[10]
    ticking_wav[:, 13] = pcg_multi[11]
    ticking_wav[:, 14] = pcg_multi[12]
    ticking_wav[:, 15] = pcg_multi[13]

    return ticking_wav


def save_cinc_signals(sig_dict, fs, REFERENCE, path, label): 
    save_dir = os.path.dirname(os.path.abspath(path))

    # Save using wfdb
    patient = path.split("/")[-1]
    sigs_dict = {sig_name: sig for sig_name, sig in sig_dict.items()}

    wfdb.wrsamp(
        f'{patient}',
        fs=fs,
        units=['mV' for _ in sigs_dict],
        sig_name=list(sigs_dict.keys()),
        p_signal=np.stack([(sig) for sig in sigs_dict.values()], axis=1),
        write_dir=save_dir,
    )

    for sig_name, signal in sigs_dict.items():
        filename = f'{patient}_{sig_name}.wav'
        save_signal_wav(signal.astype(np.float32), fs, os.path.join(save_dir, filename))

    with open(REFERENCE, "a") as fp:
        fp.write(f"{patient},{label}\n")

def save_ticking_signals(ticking_wav, fs, REFERENCE, path, label):
    # Save the signals to the destination
    save_signal_wav(ticking_wav.astype(np.float32), fs, path)
    # Add signals to reference csv
    patient = path.split("/")[-1]
    with open(REFERENCE, "a") as fp:
        fp.write(f"{patient},{label}\n")


def augment_patient(patient, labels, idx, input_dir, output_dir, dataset, num_augments,
                     REFERENCE, MAX_LEN_S, MIT, EPHNOGRAM):

    pcg_multi = list()

    pcg = ecg = np.zeros((1))
    fs = 2000

    path = os.path.join(input_dir, patient)
    out_path = os.path.join(output_dir, patient)

    if dataset in ["training-b", "training-c", "training-d", "training-e", "training-f"]:
        pcg, fs = get_cinc_sig(path, "PCG", max_len=MAX_LEN_S)
    elif dataset == "training-a":
        pcg, fs = get_cinc_sig(path, "PCG", max_len=MAX_LEN_S)
        ecg, fs = get_cinc_sig(path, "ECG", max_len=MAX_LEN_S)
    elif dataset == "ticking-heart":
        for pcg_channel in range(1, 8):
            pcg, fs = read_ticking_PCG(path, pcg_channel, max_len=MAX_LEN_S)
            pcg_multi.append(pcg)
        for pcg_channel in range(1, 8):
            pcg, fs = read_ticking_PCG(path, pcg_channel, True, max_len=MAX_LEN_S)
            pcg_multi.append(pcg)
    else:
        raise Exception(f"{dataset=} is not supported.")

    for i in range(num_augments-1):
        if dataset == "training-a":
            ecg_aug, pcg_aug = augment_signals(ecg, pcg, fs, MIT=MIT, EPHNOGRAM=EPHNOGRAM)
            signals_dict = {"PCG": pcg_aug[:60*fs], "ECG": ecg_aug[:60*fs]}
            save_cinc_signals(signals_dict, fs, REFERENCE, f"{out_path}_aug{i}", labels[idx])
        elif dataset == "ticking-heart":
            pcg_aug_multi = augment_multi_pcg(pcg_multi, fs, EPHNOGRAM=EPHNOGRAM)
            ticking_wav = create_multi_wav(pcg_aug_multi)
            save_ticking_signals(ticking_wav, fs, REFERENCE, f"{out_path}_aug{i}", labels[idx])
        else:
            pcg_aug = augment_pcg(pcg, fs)
            signals_dict = {"PCG": pcg_aug[:60*fs]}
            save_cinc_signals(signals_dict, fs, REFERENCE, f"{out_path}_aug{i}", labels[idx])

    # Save the original signal to this location 
    if dataset == "training-a":
        signals_dict = {"PCG": pcg, "ECG": ecg}
        save_cinc_signals(signals_dict, fs, REFERENCE, out_path, labels[idx])
    elif dataset == "ticking-heart":
        ticking_wav_orig = create_multi_wav(pcg_multi)
        save_ticking_signals(ticking_wav_orig, fs, REFERENCE, out_path, labels[idx])
    else:
        signals_dict = {"PCG": pcg}
        save_cinc_signals(signals_dict, fs, REFERENCE, out_path, labels[idx])

@click.command()
@click.option(
    '--input_dir', 
    '-I', 
    required=True, 
    help="The dataset to use for augmentation."
)
@click.option(
    '--output_dir', 
    '-O', 
    required=True, 
    help="The location to store the signals."
)
@click.option(
    '--reference', 
    '-R', 
    default="REFERENCE.csv", 
    help="The name of the reference file that stores the labels."
)
@click.option(
    '--mit_database_path', 
    '-M', 
    required=True,
    help="The path to the mit database to be used for augmentation."
)
@click.option(
    '--mit_database_path', 
    '-M', 
    required=True,
    help="The path to the mit database to be used for augmentation."
)
@click.option(
    '--ephnogram_database_path', 
    '-E', 
    required=True,
    help="The path to the ephongram database to be used for augmentation."
)
@click.option(
    '--reference', 
    '-R', 
    default="REFERENCE.csv", 
    help="The name of the reference file that stores the labels."
)
@click.option(
    "-D",
    "--dataset",
    required=True,
    help="Name of the dataset."
)
@click.option(
    "-N",
    "--num_augments",
    required=True,
    type=int,
    help="Number of extra times augment patients (i.e 10 for 10 times as many)"
)
def cli(
    input_dir,
    output_dir,
    mit_database_path,
    ephnogram_database_path,
    reference,
    dataset,
    num_augments,
    **kwargs
):
    del kwargs

    MAX_LEN_S = 60 # Longest signal len
    fs = 2000      # Sampling frequency

    MIT = os.path.abspath(mit_database_path)
    EPHNOGRAM = os.path.abspath(ephnogram_database_path)
    REFERENCE = os.path.join(output_dir, "REFERENCE.csv")

    input_dir = os.path.abspath(input_dir)
    output_dir = os.path.abspath(output_dir)
    os.makedirs(output_dir, exist_ok=True)

    patients, labels = get_patients(os.path.join(input_dir, reference), dataset=="training-a")

    # FIXME: Make this multiprocess
    num_workers = os.cpu_count() - 2 # type: ignore
    futures = []

    # Clear the file
    with open(REFERENCE, "w") as fp:
        fp.write("")

    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        for idx, patient in enumerate(patients):
            futures.append(executor.submit(
                augment_patient, 
                patient, 
                labels, 
                idx, 
                input_dir, 
                output_dir, 
                dataset, 
                num_augments,
                REFERENCE,
                MAX_LEN_S, 
                MIT, 
                EPHNOGRAM
            ))

        for future in tqdm(as_completed(futures), total=len(futures)):
            try:
                future.result()
            except Exception as e:
                print(f"Task generated an exception: {e}")
                executor.shutdown()
                raise e

if __name__ == "__main__":
    cli()