#!/usr/bin/env pipenv-shebang
"""
    segmentation.py
    Author : Milan Marocchi

    Front end for segmentation algorithm
"""
from processing.filtering import (
    start_matlab,
    stop_matlab,
    get_segment_pcg,
    get_segment_time,
    get_hand_label_seg_pcg,
    resample,
)
from util.fileio import (
    get_patients,
    get_cinc_sig,
    write_json_numpy,
    read_ticking_PCG,
)
from util.paths import (
    MATLAB_PATH
)
from tqdm.auto import tqdm

import os
import click
import logging
logging.basicConfig(level=logging.INFO)


@click.command()
@click.option(
    "-I",
    "--in_path",
    required=True,
    help="Path where the data is stored"
)
@click.option(
    "-O",
    "--out_path",
    required=True,
    help="Path where to store the output file, including the name of the file."

)
@click.option(
    "-R",
    "--reference",
    default="REFERENCE.csv",
    help="Name of the reference file for the dataset."
)
@click.option(
    "-D",
    "--dataset",
    required=True,
    help="Name of the dataset."
)
@click.option(
    "-T",
    "--time_segment",
    help="Segment based on time instead of springer segmentation."
)
@click.option(
    "-H",
    "--hand_labelled",
    help="The path to the hand labelled segmentation to be parsed."
)
def segment(
    out_path, 
    in_path, 
    reference, 
    dataset, 
    time_segment, 
    hand_labelled, 
    **kwargs
):
    del kwargs

    MAX_LEN_S = 60 # Longest signal len
    FS = 1000      # Sampling frequency

    os.makedirs(os.path.join(out_path), exist_ok=True)

    # read the csv and populate files
    patients, _ = get_patients(os.path.join(in_path, reference), dataset=="training-a")

    start_matlab(MATLAB_PATH)
    try:
        for patient in tqdm(patients):
            path = os.path.join(in_path, patient)

            if dataset in ["training-a", "training-b", "training-c", "training-d", "training-e", "training-f"]:
                pcg, fs_old = get_cinc_sig(path, "PCG", MAX_LEN_S)
            elif dataset == "ticking-heart":
                # FIXME: Figure out the best channel for this or change it so they all get their own segmentation
                pcg_channel = 2
                pcg, fs_old = read_ticking_PCG(path, pcg_channel, max_len=MAX_LEN_S)
            else:
                raise Exception(f"{dataset=} is not supported.")

            if hand_labelled is None:
                segmentation_info = {"fs": FS}

                # Extract segment information
                if time_segment is not None:
                    segments = get_segment_time(pcg, fs_old, FS, float(time_segment))
                else:
                    segments = get_segment_pcg(pcg, fs_old, FS)

                # Resample the pcg
                pcg = resample(pcg, fs_old, FS)

                segmentation_info['segments'] = segments # type: ignore
                segmentation_info['last_index'] = len(pcg) - 1
            else:
                filename = f"{patient}_StateAns.mat"
                segments = get_hand_label_seg_pcg(os.path.abspath(hand_labelled), filename)
                # Need to account for 1kHz instead of 2kHz
                FS = 2000

                segmentation_info = {
                    'fs': FS,
                    'segments': segments,
                    'last_index': len(pcg) - 1
                }

            write_json_numpy(segmentation_info, os.path.join(out_path, f"{patient}.json"))
    except Exception as e:
        logging.error("Error occured:", str(e))
        raise e
    finally:
        stop_matlab()

if __name__ == "__main__":
    segment()
