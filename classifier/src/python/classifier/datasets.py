"""
    datasets.py
    Author: Milan Marocchi

    Purpose: For all custom datasets 
"""

from util.fileio import (
    get_cinc_sig,
    get_patients_segments,
    get_patients_split,
    read_ticking_PCG,
)
from processing.plotting import (
    SignalPlotterFactory,
    plot_signal,
)
from processing.filtering import (
    noise_canc,
    normalise_signal,
)
from processing.process import (
    pre_process_ecg_orig,
    pre_process_pcg_orig,
    pre_process_orig_four_bands,
    normalise_array_length,
    normalise_2d_array_length,
)
from processing.segments import (
    get_seg_join_idx,
    get_seg_time_join_idx,
    resample_segments,
)
from processing.augmentation import (
    augment_signals,
    augment_pcg,
)

from torch.utils.data import Dataset
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm
from scipy.io import wavfile

import os
import PIL.Image
import torch
import numpy as np
import scipy.signal as ssg
import random

class Fragment():
    """
    To hold fragment information
    """

    def __init__(self, patient_name: str, fragment: int, segment_start: int, segment_end: int):
        self.patient_name = patient_name
        self.fragment = fragment
        self.segment_start = segment_start
        self.segment_end = segment_end

    def __hash__(self):
        return hash((self.patient_name, self.fragment, self.segment_start, self.segment_end))

    def __eq__(self, other):
        if not isinstance(other, Fragment):
            return False
        return (
            self.patient_name == other.patient_name and
            self.fragment == other.fragment and
            self.segment_start == other.segment_start and
            self.segment_end == other.segment_end
        )

class Patient():
    """
    TO hold information about patients
    """

    FS = 1000

    def __init__(self, name: str, segments_path: str, label, segmentation):
        self.name = name
        self.segments_path = segments_path
        self.label = label
        self.segmentation = segmentation
        self.fragments = list()


    def read_segments(self):
        segments, last_idx, fs = get_patients_segments(self.segments_path)

        if self.segmentation == "heart":
            segments = get_seg_join_idx(segments, last_idx)
        elif self.segmentation == "time":
            segments = get_seg_time_join_idx(segments, last_idx)
        else:
            raise Exception("{segmentation=} is not supported.")

        segments, _ = resample_segments(segments, fs, self.FS) 

        return segments, last_idx


class PatientCollection():
    """To hold all the patients."""
    NUM_FRAGMENTS = 100000
    OVERLAP = 250 # samples

    def __init__(self, segments_path, splits_path, subset, segmentation="heart"):
        self.segments_path = segments_path
        self.splits_path = splits_path
        self.subset = subset
        self.segmentation = segmentation

        self.patients: dict = dict()
        self.fragments: list = list()
        self.idx_fragments: dict = dict()

        self.read_patients()

    def read_patients(self):
        patients_data = get_patients_split(self.splits_path, self.subset)

        for _ , patient in patients_data.iterrows():
            patient_name = patient['patient']
            patient = Patient(
                patient_name,
                os.path.join(self.segments_path, f"{patient_name}.json"),
                patient['abnormality'],
                self.segmentation,
            )
            self.patients[patient_name] = patient

            segments, last_idx = patient.read_segments()

            self.read_segments(patient_name, segments, last_idx)

        # create reverse map
        self.idx_fragments = {self.fragments[x] : x for x in range(len(self.fragments))}

    def read_segments(self, patient_name, segments, last_idx):
        for idx, segment in enumerate(segments):
            if idx == self.NUM_FRAGMENTS and self.subset not in ['train', 'valid']:
                break

            if self.segmentation == 'time':
                start_index = 300 if idx == 0 else (segments[idx-1] - idx * self.OVERLAP)
                end_index = segment if idx == 0 else segment - idx * self.OVERLAP
            elif self.segmentation == 'heart':
                start_index = segment
                end_index = segments[idx+1] if idx < (len(segments) - 1) and idx < self.NUM_FRAGMENTS else last_idx
            else:
                raise ValueError(f"{self.segmentation=}: is not supported.")

            fragment = Fragment(patient_name, idx, start_index, end_index)
            self.fragments.append(fragment)
            self.patients[patient_name].fragments.append(fragment)

    def merge_collection(self, patient_collection):
        """Merges another PatientCollection to this one."""
        self.fragments.extend(patient_collection.fragments)
        self.patients.update(patient_collection.patients)
        # create reverse map
        self.idx_fragments = {x: self.fragments[x] for x in range(len(self.fragments))}

    def get_patient(self, name):
        return self.patients[name]

    def get_fragment(self, idx):
        return self.fragments[idx]


class SyntheticPatientCollection(PatientCollection):

    def __init__(self, segments_path, splits_path, subset, segmentation="heart"):
        super().__init__(
            segments_path,
            splits_path,
            subset,
            segmentation
        )

    def read_segments(self, patient_name, segments, last_idx):
        idx = random.randint(1, len(segments) - 1)
        segment = segments[idx]

        if self.segmentation == 'time':
            start_index = 0 if idx == 0 else (segments[idx-1] - idx * self.OVERLAP)
            end_index = segment if idx == 0 else segment - idx * self.OVERLAP
        elif self.segmentation == 'heart':
            start_index = segment
            end_index = segments[idx+1] if idx < (len(segments) - 1) and idx < self.NUM_FRAGMENTS else last_idx
        else:
            raise ValueError(f"{self.segmentation=}: is not supported.")

        fragment = Fragment(patient_name, idx, start_index, end_index)
        self.fragments.append(fragment)
        self.patients[patient_name].fragments.append(fragment)


class HeartDataset(Dataset):
    """Base dataset for heart signals."""

    FS = 1000
    FILE_EXT = "png"
    AUGMENT_NUM = 5

    AUGMENT_BALANCE = True

    def __init__(
            self,
            data_dir,
            splits_path,
            segments_path,
            subset,
            output_dir,
            ecg=False,
            segmentation='heart',
            four_band=True,
            augmentation=False,
            skip_data_valid=False,
            sig_len=1.5,
            **kwargs
    ):
        """
        Dataset for preprocessed heart signals (Creates the images)

        :param str data_dir: The directory where the data is stored
        :param str splits_path: The path to the splits file
        :param str segments_path: The path to the segment files
        :param str subset: The subset this dataset is (train/valid/test)
        :param str plotter: The plotter to use (stft/mel/wave)
        """
        # Is the signal length in seconds
        self.SIG_LEN = sig_len

        self.data_dir = data_dir
        self.output_dir = output_dir

        self.segments_path = segments_path
        self.splits_path = splits_path

        self.four_band = four_band
        self.subset = subset

        self.classes = ["0", "1"]

        self.ecg = ecg

        self.augmentation = augmentation
        self.skip_data_valid = skip_data_valid

        self.patients: PatientCollection = PatientCollection(
            self.segments_path,
            self.splits_path,
            self.subset,
            segmentation=segmentation,
        )

        self.setup_all_data()

    def __len__(self):
        return len(self.patients.fragments)

    def __getitem__(self, idx):
        raise NotImplementedError("This class must be inherited not used directly.")

    def get_label(self, idx):
        """Get label for an index."""
        fragment = self.patients.get_fragment(idx)
        patient_name = fragment.patient_name
        label = f"{patient_name}.{self.patients.get_patient(patient_name).label}.{fragment.fragment}"

        return label

    def check_create_data(self, out_path):
        """Check if data must be created."""
        return not os.path.exists(out_path)

    def read_data(self, patient_name):
        """Reads in the data"""
        raise NotImplementedError("This class must be inherited not used directly.")

    def process_data(self, patient_name, pcg, fs_p, ecg, fs_e):
        """processes the data"""
        raise NotImplementedError("This class must be inherited not used directly.")
        
    def save_data(self, data, out_path, fs=None):
        """Save the data for lookup later."""
        raise NotImplementedError("This class must be inherited not used directly.")

    def setup_data(self, fragment):
        """Read in and then preprocess the data"""
        patient_name = fragment.patient_name
        pcg, fs_p, ecg, fs_e = self.read_data(patient_name)
        data = self.process_data(patient_name, pcg, fs_p, ecg, fs_e)

        return data

    def create_augment_patients(self):
        augmented_patients = {}
        augmented_fragments = []

        # Figure out how many extra augmented patients should be generated to ensure a balanced dataset
        balance = 0
        total = 0
        patient = None
        for idx in range(len(self.patients.fragments)):
            fragment = self.patients.get_fragment(idx)
            curr_patient = self.patients.get_patient(fragment.patient_name)
            if curr_patient != patient:
                patient = curr_patient
                label = 1 if patient.label == 1 else -1
                balance += label * self.AUGMENT_NUM
                total += 1 * self.AUGMENT_NUM

        pos_extra_aug = (self.AUGMENT_NUM // 2) * (total // abs(balance)) if balance < 0 else 0
        neg_extra_aug = (self.AUGMENT_NUM // 2) * (total // abs(balance)) if balance > 0 else 0

        # Add in the augmentation patients and fragments
        for idx in range(len(self.patients.fragments)):

            fragment = self.patients.get_fragment(idx)
            patient = self.patients.get_patient(fragment.patient_name)
            label = 1 if patient.label == 1 else -1
            augment_num = self.AUGMENT_NUM if self.subset == "train" else int(self.AUGMENT_NUM / 2)

            if self.AUGMENT_BALANCE:
                extra_augment_num = (pos_extra_aug if label == 1 else neg_extra_aug)
                augment_num += (extra_augment_num if self.subset == "train" else int(extra_augment_num / 2))

            for i in range(augment_num):
                patient_name_aug = f"{fragment.patient_name}_aug{i}"
                if not patient_name_aug in augmented_patients:
                    patient_aug = Patient(
                        patient_name_aug,
                        patient.segments_path,
                        patient.label,
                        patient.segmentation,
                    )
                    augmented_patients[f"{patient_name_aug}"] = patient_aug

                else:
                    patient_aug = augmented_patients[f"{patient_name_aug}"]

                fragment_aug = Fragment(
                    patient_name_aug,
                    fragment.fragment,
                    fragment.segment_start,
                    fragment.segment_end
                )
                augmented_fragments.append(fragment_aug)
                patient_aug.fragments.append(fragment_aug)


        self.patients.fragments.extend(augmented_fragments)
        self.patients.patients.update(augmented_patients)
        # create reverse map with updated augments
        self.patients.idx_fragments = {self.patients.fragments[x] : x for x in range(len(self.patients.fragments))}

    def setup_all_data(self):
        """Preproccess and create all images."""
        num_workers = os.cpu_count() - 2 # type: ignore
        futures = []

        if self.subset in ["train", "valid", "test"] and self.augmentation:
            self.create_augment_patients()

        if self.skip_data_valid:
            return

        with ProcessPoolExecutor(max_workers=num_workers) as executor:
            for patient in self.patients.patients.values():
                #self._setup_data(patient)
                futures.append(executor.submit(self._setup_data, patient))

            for future in tqdm(as_completed(futures), total=len(futures)):
                try:
                    future.result()
                except Exception as e:
                    print(f"Task generated an exception: {e}")
                    executor.shutdown()
                    raise e

    def _setup_data(self, patient):
        data = None
        out_dir = os.path.abspath(self.output_dir)

        for fragment in patient.fragments:

            idx = self.patients.idx_fragments[fragment]
            out_path = os.path.join(out_dir, f"{self.get_label(idx)}.{self.FILE_EXT}")

            if self.check_create_data(out_path):
                os.makedirs(out_dir, exist_ok=True)

                if data is None:
                    data = self.setup_data(fragment)

                if data.ndim == 2:
                    data_crop = data[fragment.segment_start:fragment.segment_end]
                    data_crop, _ = normalise_2d_array_length(data_crop, int(self.SIG_LEN * self.FS))
                elif data.ndim == 3:
                    data_crop = data[:, fragment.segment_start:fragment.segment_end]
                    data_reshape = np.zeros((data.shape[0], int(self.SIG_LEN * self.FS), data.shape[2]))

                    for i in range(len(data_crop)):
                        data_reshape[i], _ = normalise_2d_array_length(data_crop[i], int(self.SIG_LEN * self.FS))
                    data_crop = data_reshape
                else:
                    raise ValueError(f"Unsupported array of dimension: {data.ndim=}")

                self.save_data(data_crop, out_path, fs=self.FS)


class HeartImageDataset(HeartDataset):
    """Dataset for preprocessed heart signals (Creates the images)."""

    def __init__(
            self,
            data_dir,
            splits_path,
            segments_path,
            subset,
            output_dir,
            plotter="stft",
            ecg=False,
            segmentation='heart',
            transform=None,
            four_band=True,
            augmentation=False,
            skip_data_valid=False,
            sig_len=1.5,
            **kwargs
    ):
        """
        Dataset for preprocessed heart signals (Creates the images)

        :param str data_dir: The directory where the data is stored
        :param str splits_path: The path to the splits file
        :param str segments_path: The path to the segment files
        :param str subset: The subset this dataset is (train/valid/test)
        :param str plotter: The plotter to use (stft/mel/wave)
        """
        self.transform = transform
        self.plotter = SignalPlotterFactory().create(plotter)

        super().__init__(
            data_dir,
            splits_path,
            segments_path,
            subset,
            output_dir,
            ecg=ecg,
            segmentation=segmentation,
            four_band=four_band,
            augmentation=augmentation,
            skip_data_valid=skip_data_valid,
            sig_len=sig_len,
        )

    def __len__(self):
        return len(self.patients.fragments)

    def __getitem__(self, idx):

        out_dir = os.path.abspath(self.output_dir)
        label = self.get_label(idx)
        out_path = os.path.join(out_dir, f"{label}.png")

        image_pil = PIL.Image.open(out_path).convert('RGB')

        if self.transform:
            image_pil = self.transform(image_pil)

        return image_pil, label

    def get_label(self, idx):
        """Get label for an index."""
        fragment = self.patients.get_fragment(idx)
        patient_name = fragment.patient_name
        label = f"{patient_name}.{self.patients.get_patient(patient_name).label}.{fragment.fragment}"

        return label

    def check_create_data(self, out_path):
        """Check if data must be created."""
        return not os.path.exists(out_path)

    def read_data(self, patient_name):
        """Reads in the data"""
        pcg, fs_p = get_cinc_sig(os.path.join(self.data_dir, patient_name.split("_aug")[0]), "PCG", 60)

        if self.ecg:
            ecg, fs_e = get_cinc_sig(os.path.join(self.data_dir, patient_name.split("_aug")[0]), "ECG", 60)
        else:
            ecg = fs_e = None

        return pcg, fs_p, ecg, fs_e

    def process_data(self, patient_name, pcg, fs_p, ecg, fs_e):
        """processes the data"""
        if self.augmentation and "_aug" in patient_name:
            orig_pcg_len = len(pcg)

            if ecg is not None:
                orig_ecg_len = len(ecg)
                ecg, pcg = augment_signals(ecg, pcg, fs_p)
                ecg, _ = normalise_array_length(ecg, orig_ecg_len)
            else:
                pcg = augment_pcg(pcg, fs_p)

            pcg, _ = normalise_array_length(pcg, orig_pcg_len)

        pcg = pre_process_pcg_orig(pcg, fs_p, self.FS)

        if self.four_band:
            pcg = pre_process_orig_four_bands(pcg, self.FS)
        else:
            pcg = pcg.reshape(-1, 1)

        if ecg is not None:
            ecg = pre_process_ecg_orig(ecg, fs_e, self.FS)
            ecg = ecg.reshape(-1, 1)
            data = np.hstack((pcg, ecg))
        else:
            data = pcg

        return data

    def save_data(self, data, out_path, fs=None):
        """Save the data for lookup later."""
        plot_signal(
            data,
            self.plotter,
            labels=False,
            colorbar=False,
            hide_axis=True,
            path=out_path
        )


class SyntheticHeartImageDatabase(HeartImageDataset):

    def __init__(
            self,
            data_dir,
            splits_path,
            segments_path,
            subset,
            output_dir,
            plotter="stft",
            ecg=False,
            segmentation='heart',
            transform=None,
            four_band=True,
            augmentation=False,
            skip_data_valid=False,
            sig_len=1.5,
            **kwargs
    ):
        self.SIG_LEN = sig_len
        self.transform = transform
        self.plotter = SignalPlotterFactory().create(plotter)

        self.data_dir = data_dir
        self.output_dir = output_dir

        self.segments_path = segments_path
        self.splits_path = splits_path

        self.four_band = four_band
        self.subset = subset

        self.classes = ["0", "1"]

        self.ecg = ecg

        self.augmentation = augmentation
        self.skip_data_valid = skip_data_valid

        self.patients: SyntheticPatientCollection = SyntheticPatientCollection(
            self.segments_path,
            self.splits_path,
            self.subset,
            segmentation=segmentation,
        )

        self.setup_all_data()



# NOTE: This one is for all of cinc
class HeartImageDatabase(HeartImageDataset):
    """Dataset for preprocessed heart signals (Creates the images for all of cinc)."""

    def __init__(
            self,
            data_dir,
            splits_path,
            segments_path,
            subset,
            output_dir,
            plotter="stft",
            databases='training-a:training-b:training-c:training-d:training-e:training-f',
            segmentation='heart',
            transform=None,
            augmentation=False,
            four_band=True,
            sig_len=1.5,
            **kwargs
    ):
        """
        Dataset for preprocessed heart signals (Creates the images)

        :param str data_dir: The directory where the data is stored
        :param str splits_path: The path to the splits file
        :param str segments_path: The path to the segment files
        :param str subset: The subset this dataset is (train/valid/test)
        :param str plotter: The plotter to use (stft/mel/wave)
        """
        # Signal length is in seconds.
        self.SIG_LEN = sig_len

        self.data_dir = data_dir
        self.output_dir = output_dir

        self.segments_path = segments_path
        self.splits_path = splits_path

        self.four_band = four_band
        self.subset = subset

        self.transform = transform
        self.plotter = SignalPlotterFactory().create(plotter)

        self.classes = ["0", "1"]

        self.databases = self.get_databases(databases)

        self.augmentation = augmentation

        for idx, database in enumerate(self.databases):
            if idx == 0:
                self.patients: PatientCollection = PatientCollection(
                    os.path.join(self.segments_path, database),
                    os.path.join(self.splits_path, f"{database}.csv"),
                    self.subset,
                    segmentation=segmentation
                )
            else:
                patient_col = PatientCollection(
                    os.path.join(self.segments_path, database),
                    os.path.join(self.splits_path, f"{database}.csv"),
                    self.subset,
                    segmentation=segmentation
                )
                self.patients.merge_collection(patient_col)

        self.setup_all_data()

    def __len__(self):
        return len(self.patients.fragments)

    def get_databases(self, databases):
        """Get the datasets inside the database."""
        return databases.split(":")

    def read_data(self, patient_name):
        """Reads in the data"""
        database = patient_name[0]
        data_dir = os.path.join(self.data_dir, f"training-{database}")
        pcg, fs_p = get_cinc_sig(os.path.join(data_dir, patient_name.split("_aug")[0]), "PCG", 60)

        return pcg, fs_p, None, None

class HeartAudioDatabase(HeartImageDatabase):

    FILE_EXT = "wav"
    AUGMENT_NUM = 30

    def __init__(
            self,
            data_dir,
            splits_path,
            segments_path,
            subset,
            output_dir,
            plotter="stft",
            databases='training-a:training-b:training-c:training-d:training-e:training-f',
            segmentation='heart',
            transform=None,
            augmentation=False,
            four_band=True,
            sig_len=1.5,
            fs=16000,
            skip_data_valid=False,
            **kwargs
    ):
        self.CLASSIFY_FS = fs
        self.skip_data_valid = skip_data_valid

        super().__init__(
            data_dir,
            splits_path,
            segments_path,
            subset,
            output_dir,
            plotter=plotter,
            databases=databases,
            segmentation=segmentation,
            transform=transform,
            augmentation=augmentation,
            four_band=four_band,
            sig_len=sig_len,
        )

    def save_data(self, data, out_path, fs=None):
        # create wav
        wavfile.write(out_path, self.FS, data.astype(np.float32))

    def __getitem__(self, idx):

        out_dir = os.path.abspath(self.output_dir)
        label = self.get_label(idx)
        out_path = os.path.join(out_dir, f"{label}.{self.FILE_EXT}")

        fs, data = wavfile.read(out_path)
        data = ssg.resample_poly(data, self.CLASSIFY_FS, fs)

        # Normalise the data
        data = normalise_signal(data)

        return data, label


class MultiImageHeartDataset(HeartImageDataset):
    """Dataset for preprocessed heart signals for multi-input models (Creates the multiple images)."""

    def __init__(
            self,
            data_dir,
            splits_path,
            segments_path,
            subset,
            output_dir,
            plotter="stft",
            ecg=False,
            segmentation='heart',
            transform=None,
            four_band=True,
            augmentation=False,
            num_inputs=2,
            sig_len=1.5,
            **kwargs
    ):
        """
        Dataset for preprocessed heart signals (Creates the images)

        :param str data_dir: The directory where the data is stored
        :param str splits_path: The path to the splits file
        :param str segments_path: The path to the segment files
        :param str subset: The subset this dataset is (train/valid/test)
        :param str plotter: The plotter to use (stft/mel/wave)
        """
        self.num_inputs = int(num_inputs)

        super().__init__(
            data_dir,
            splits_path,
            segments_path,
            subset,
            output_dir,
            plotter=plotter,
            ecg=ecg,
            segmentation=segmentation,
            transform=transform,
            augmentation=augmentation,
            four_band=four_band,
            sig_len=sig_len
        )

    def __getitem__(self, idx):

        out_dir = os.path.abspath(self.output_dir)
        label = self.get_label(idx)


        images = []
        for i in range(int(self.num_inputs)):
            image_path = os.path.join(out_dir, str(i), f"{label}.png")
            image_pil = PIL.Image.open(image_path).convert('RGB')

            if self.transform:
                image_pil = self.transform(image_pil)

            images.append(image_pil)

        # Conver to stacked tensors
        image_pil = torch.stack(tuple(images))

        return image_pil, label

    def check_create_data(self, out_path):
        """Check if data must be created."""
        for i in range(self.num_inputs):
            out_dir = "/".join(out_path.split("/")[:-1])
            file_name = out_path.split("/")[-1]
            file_path = os.path.join(out_dir, str(i), file_name)

            if os.path.exists(file_path):
                return False

        return True

    def save_data(self, data, out_path, fs=None):
        out_dir = "/".join(out_path.split("/")[:-1])
        file_name = out_path.split("/")[-1]

        for i in range(self.num_inputs):

            file_path = os.path.join(out_dir, str(i), file_name)
            os.makedirs(os.path.join(out_dir, str(i)), exist_ok=True)

            # FIXME: Make this not hardcoded. As currently just splits for each band
            subset_data = data[:, :-1] if i == 0 else data[:, -1].reshape(-1, 1)

            # create image
            plot_signal(
                subset_data,
                self.plotter,
                labels=False,
                colorbar=False,
                hide_axis=True,
                path=file_path
            )

class SyntheticMultiHeartImageDatabase(MultiImageHeartDataset):

    def __init__(
            self,
            data_dir,
            splits_path,
            segments_path,
            subset,
            output_dir,
            plotter="stft",
            ecg=False,
            segmentation='heart',
            transform=None,
            four_band=True,
            augmentation=False,
            skip_data_valid=False,
            sig_len=1.5,
            num_inputs=2,
            **kwargs
    ):
        self.SIG_LEN = sig_len
        self.transform = transform
        self.plotter = SignalPlotterFactory().create(plotter)

        self.data_dir = data_dir
        self.output_dir = output_dir

        self.segments_path = segments_path
        self.splits_path = splits_path

        self.four_band = four_band
        self.subset = subset

        self.classes = ["0", "1"]

        self.ecg = ecg
        self.num_inputs = num_inputs

        self.augmentation = augmentation
        self.skip_data_valid = skip_data_valid

        self.patients: SyntheticPatientCollection = SyntheticPatientCollection(
            self.segments_path,
            self.splits_path,
            self.subset,
            segmentation=segmentation,
        )

        self.setup_all_data()


class TickingHeartImageDataset(HeartImageDataset):
    """Dataset for ticking heart signals."""

    def __init__(
            self,
            data_dir,
            splits_path,
            segments_path,
            subset,
            output_dir,
            plotter="stft",
            ecg=False,
            segmentation='heart',
            transform=None,
            four_band=True,
            augmentation=False,
            sig_len=1.5,
            **kwargs
    ):
        super().__init__(
            data_dir,
            splits_path,
            segments_path,
            subset,
            output_dir,
            plotter=plotter,
            ecg=ecg,
            segmentation=segmentation,
            transform=transform,
            four_band=four_band,
            augmentation=augmentation,
            sig_len=sig_len
        )

    def read_data(self, patient_name):
        """Reads in the data"""
        pcg, fs_p = read_ticking_PCG(os.path.join(self.data_dir, patient_name), 2, max_len=60)
        pcg_nm, _ = read_ticking_PCG(os.path.join(self.data_dir, patient_name), 2, True, max_len=60)

        # FIXME: verify the noise cancellation stuff
        pcg = noise_canc(pcg_nm, pcg, fs=fs_p)

        return pcg, fs_p, None, None


class MultiTickingHeartImageDataset(MultiImageHeartDataset):
    """Dataset for ticking heart signals for models that take multiple inputs (channels of PCG)."""

    num_inputs = 6

    def __init__(
            self,
            data_dir,
            splits_path,
            segments_path,
            subset,
            output_dir,
            plotter="stft",
            ecg=False,
            segmentation='heart',
            transform=None,
            four_band=True,
            augmentation=False,
            sig_len=1.5,
            **kwargs
    ):
        super().__init__(
            data_dir,
            splits_path,
            segments_path,
            subset,
            output_dir,
            plotter=plotter,
            ecg=ecg,
            segmentation=segmentation,
            transform=transform,
            four_band=four_band,
            augmentation=augmentation,
            num_inputs=self.num_inputs,
            sig_len=sig_len
        )

    def read_data(self, patient_name):
        """Read in the data."""
        pcg_channels = []
        fs = 0

        for channel in range(1, int(self.num_inputs) + 1):
            pcg, fs_p = read_ticking_PCG(os.path.join(self.data_dir, patient_name.split("_aug")[0]), channel, max_len=60)
            pcg_nm, _ = read_ticking_PCG(os.path.join(self.data_dir, patient_name.split("_aug")[0]), channel, noise_mic=True, max_len=60)

            # FIXME: verify the noise cancellation stuff
            pcg = noise_canc(pcg_nm, pcg, fs=fs_p)

            pcg_channels.append(pcg)
            fs = fs_p

        return pcg_channels, fs, None, None


    def process_data(self, patient_name, pcg_channels, fs_p, ecg, fs_e):
        """to read in the data and process it to be plotted."""
        data = []

        for idx, pcg in enumerate(pcg_channels):
            if self.augmentation and "aug" in patient_name:
                orig_pcg_len = len(pcg)
                pcg = augment_pcg(pcg, fs_p)
                pcg, _ = normalise_array_length(pcg, orig_pcg_len)

            pcg = pre_process_pcg_orig(pcg, fs_p, self.FS)

            if self.four_band:
                pcg = pre_process_orig_four_bands(pcg, self.FS)
            else:
                pcg = pcg.reshape(-1, 1)

            data.append(pcg)

        if ecg is not None:
            ecg = pre_process_ecg_orig(ecg, fs_e, self.FS)
            ecg = ecg.reshape(-1, 1)
            data.append(ecg)

        data = np.asarray(data)

        return data

    def save_data(self, data, out_path, fs=None):
        # FIXME: Do this using os path stuff
        out_dir = "/".join(out_path.split("/")[:-1])
        file_name = out_path.split("/")[-1]

        for i in range(int(self.num_inputs)):

            file_path = os.path.join(out_dir, str(i), file_name)
            os.makedirs(os.path.join(out_dir, str(i)), exist_ok=True)
            subset_data = data[i]

            # create image
            plot_signal(
                subset_data,
                self.plotter,
                labels=False,
                colorbar=False,
                hide_axis=True,
                path=file_path
            )


class HeartAudioDataset(HeartDataset):

    # SIG_LEN is the number of seconds
    FS = 1000
    FILE_EXT = "wav"
    AUGMENT_NUM = 30

    def __init__(
            self,
            data_dir,
            splits_path,
            segments_path,
            subset,
            output_dir,
            ecg=False,
            segmentation='heart',
            four_band=True,
            augmentation=False,
            fs=16000,
            skip_data_valid=False,
            sig_len=1.5,
            channel=-1,
            **kwargs
    ):
        """
        Dataset for preprocessed heart signals (Creates the images)

        :param str data_dir: The directory where the data is stored
        :param str splits_path: The path to the splits file
        :param str segments_path: The path to the segment files
        :param str audio_dir: The path to the processed audio files
        :param str subset: The subset this dataset is (train/valid/test)
        :param str plotter: The plotter to use (stft/mel/wave)
        """
        self.CLASSIFY_FS = fs
        self.channel = -1

        super().__init__(
            data_dir,
            splits_path,
            segments_path,
            subset,
            output_dir,
            ecg=ecg,
            segmentation=segmentation,
            four_band=four_band,
            augmentation=augmentation,
            skip_data_valid=skip_data_valid,
            sig_len=sig_len
        )

    def __getitem__(self, idx):

        out_dir = os.path.abspath(self.output_dir)
        label = self.get_label(idx)
        out_path = os.path.join(out_dir, f"{label}.{self.FILE_EXT}")

        fs, data = wavfile.read(out_path)
        data = ssg.resample_poly(data, self.CLASSIFY_FS, fs)

        # Normalise the data
        data = normalise_signal(data)
        data = data[:, self.channel] if self.channel != -1 else data

        return data, label

    def read_data(self, patient_name):
        """Reads in the data"""
        pcg, fs_p = get_cinc_sig(os.path.join(self.data_dir, patient_name.split("_aug")[0]), "PCG", 60)

        if self.ecg:
            ecg, fs_e = get_cinc_sig(os.path.join(self.data_dir, patient_name.split("_aug")[0]), "ECG", 60)
        else:
            ecg = fs_e = None

        return pcg, fs_p, ecg, fs_e

    def process_data(self, patient_name, pcg, fs_p, ecg, fs_e):
        """processes the data"""
        if self.augmentation and "_aug" in patient_name:
            orig_pcg_len = len(pcg)

            if ecg is not None:
                orig_ecg_len = len(ecg)
                ecg, pcg = augment_signals(ecg, pcg, fs_p)
                ecg, _ = normalise_array_length(ecg, orig_ecg_len)
            else:
                pcg = augment_pcg(pcg, fs_p)

            pcg, _ = normalise_array_length(pcg, orig_pcg_len)

        pcg = pre_process_pcg_orig(pcg, fs_p, self.FS)

        if self.four_band:
            pcg = pre_process_orig_four_bands(pcg, self.FS)
        else:
            pcg = pcg.reshape(-1, 1)

        if ecg is not None:
            ecg = pre_process_ecg_orig(ecg, fs_e, self.FS)
            ecg = ecg.reshape(-1, 1)
            data = np.hstack((pcg, ecg))
        else:
            data = pcg

        return data

    def save_data(self, data, out_path, fs=None):
        # create wav
        wavfile.write(out_path, self.FS, data.astype(np.float32))


class SyntheticHeartAudioDataset(HeartAudioDataset):

    def __init__(
            self,
            data_dir,
            splits_path,
            segments_path,
            subset,
            output_dir,
            ecg=False,
            segmentation='heart',
            four_band=True,
            augmentation=False,
            fs=16000,
            skip_data_valid=False,
            sig_len=1.5,
            channel=-1,
            **kwargs
    ):

        self.CLASSIFY_FS = fs
        self.channel = -1
        self.SIG_LEN = sig_len

        self.data_dir = data_dir
        self.output_dir = output_dir

        self.segments_path = segments_path
        self.splits_path = splits_path

        self.four_band = four_band
        self.subset = subset

        self.classes = ["0", "1"]

        self.ecg = ecg

        self.augmentation = augmentation
        self.skip_data_valid = skip_data_valid

        self.patients: SyntheticPatientCollection = SyntheticPatientCollection(
            self.segments_path,
            self.splits_path,
            self.subset,
            segmentation=segmentation,
        )

        self.setup_all_data()

class TickingHeartAudioDataset(HeartAudioDataset):

    AUGMENT_NUM = 30
    FS = 1000
    num_channels = 6

    def __init__(
            self,
            data_dir,
            splits_path,
            segments_path,
            subset,
            audio_dir,
            ecg=False,
            segmentation='heart',
            transform=None,
            four_band=True,
            augmentation=False,
            fs=16000,
            skip_data_valid=False,
            sig_len=1.5,
            channel=-1,
            **kwargs
    ):
        self.channel = channel

        super().__init__(
            data_dir,
            splits_path,
            segments_path,
            subset,
            audio_dir,
            ecg=ecg,
            segmentation=segmentation,
            four_band=four_band,
            augmentation=augmentation,
            fs=fs,
            skip_data_valid=skip_data_valid,
            sig_len=sig_len,
        )

    def read_data(self, patient_name):
        """Reads in the data."""

        fs = 0
        pcg_channels = np.zeros(1)

        for channel in range(1, int(self.num_channels) + 1):
            pcg, fs_p = read_ticking_PCG(os.path.join(self.data_dir, patient_name.split("_aug")[0]), channel, max_len=60)
            pcg_nm, _ = read_ticking_PCG(os.path.join(self.data_dir, patient_name.split("_aug")[0]), channel, noise_mic=True, max_len=60)

            if (pcg_channels) == 1:
                pcg_channels = np.zeros((len(pcg), self.num_channels))

            # FIXME: verify the noise cancellation stuff
            pcg = noise_canc(pcg_nm, pcg, fs=fs_p)

            pcg_channels[:, channel] = pcg
            assert fs == fs_p
            fs = fs_p

        return pcg_channels, fs, None, None
