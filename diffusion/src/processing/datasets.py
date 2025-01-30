import os
import random
import datetime

import torch
import pandas as pd
import numpy as np
import scipy.io as sio

from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from xkcdpass import xkcd_password
from sklearn.model_selection import train_test_split
from tqdm.auto import tqdm

import generative.models

from processing.filtering import (
    standardise_signal,
    create_spectrogram,
    add_chirp,
    pre_filter_ecg,
    pre_filter_pcg,
    mid_filter_ecg,
    mid_filter_pcg,
    resample,
    fade_signal,
)

from processing.records import readrecord

from processing.segments import (
    load_segment_info,
    seg_idxs_from_mat,
    segment_signal_tuple,
    build_signal,
    shuffle_segments_multi,
    get_weighted_random_choice,
    create_segment_waveform,
    join_adjacent_segments,
    from_matform,
    resample_seg_joins,
)

from utils.utils import first_dict_value
from utils.reproducible import torch_gen, seed_worker


CINC_DATASET_POSSIBLE_LABELS = (
    -1,
    1
)


CINC_EXTENDED_DATASET_POSSIBLE_LABELS = (
    'Normal',
    'Benign',
    'MVP',
    'MPC',
    'AD',
)


def assign_split_old(annotations, ratios={'train': 0.6, 'valid': 0.2, 'test': 0.2}, random_state=None):

    assert (sum(ratios[r] for r in ratios) - 1) < 1E-3, f'{ratios=}'

    valid_ratio_adjusted = ratios['valid'] / (ratios['train'] + ratios['valid'])

    patients_remaining, patients_test = train_test_split(annotations,
                                                         test_size=ratios['test'],
                                                         random_state=random_state,
                                                         stratify=annotations['diagnosis'])

    patients_train, patients_valid = train_test_split(patients_remaining,
                                                      test_size=valid_ratio_adjusted,
                                                      random_state=random_state,
                                                      stratify=patients_remaining['diagnosis'])  # type: ignore

    splits = {'train': patients_train, 'valid': patients_valid, 'test': patients_test}

    splits = {split: splits[split].assign(split=split) for split in splits}  # type: ignore

    new_annotations = pd.concat([splits[split] for split in splits]).sort_values(by='patient')  # type: ignore

    return new_annotations, splits


def assign_split(annotations, stratify_cols, ratios={'train': 0.6, 'valid': 0.2, 'test': 0.2}, random_state=None):
    assert abs(sum(ratios[r] for r in ratios) - 1) < 1E-3, f'{ratios=}'

    for col in stratify_cols:
        assert col in annotations.columns, f'{col=} not found in annotations'

    sqi_zero_data = annotations[annotations['SQI'] == 0].copy()
    sqi_nonzero_data = annotations[annotations['SQI'] != 0].copy()

    valid_count_needed = int(len(annotations) * ratios['valid'])
    valid_nonzero_count = valid_count_needed - len(sqi_zero_data)

    non_valid_data, valid_nonzero_data = train_test_split(
        sqi_nonzero_data,
        test_size=valid_nonzero_count,
        random_state=random_state,
        stratify=sqi_nonzero_data[stratify_cols]
    )

    patients_train, patients_test = train_test_split(
        non_valid_data,
        test_size=ratios['test']/(ratios['train'] + ratios['test']),
        random_state=random_state,
        stratify=non_valid_data[stratify_cols]
    )

    patients_valid = pd.concat([sqi_zero_data, valid_nonzero_data])  # type: ignore

    splits = {'train': patients_train, 'valid': patients_valid, 'test': patients_test}

    splits = {split: splits[split].assign(split=split) for split in splits}  # type: ignore

    new_annotations = pd.concat([splits[split] for split in splits]).sort_values(by='patient')

    return new_annotations, splits


def display_split(annotations, splits):

    for split in splits:
        print(annotations[annotations['split'] == split])

    for split in splits:
        print(split, len(splits[split]))
        print(splits[split]['diagnosis'].value_counts())
        print(splits[split]['diagnosis'].value_counts(normalize=True))
        print(splits[split])

    annotations.info()


def create_split_name():

    wordlist = xkcd_password.generate_wordlist(xkcd_password.locate_wordfile())
    xkcd_name = xkcd_password.generate_xkcdpassword(wordlist, numwords=3, delimiter='', case='capitalize')

    assert xkcd_name is not None, f'{xkcd_name=}'

    now = datetime.datetime.now()
    mins_name = str(round((now - now.replace(hour=0, minute=0, second=0, microsecond=0)).total_seconds() // 60))

    date_name = now.strftime('%Y-%m%b-%d')

    save_name = '_'.join([
        xkcd_name,
        date_name,
        mins_name,
    ])

    return save_name


def get_possible_labels(dataset):
    if dataset == 'training-a':
        return CINC_DATASET_POSSIBLE_LABELS
    elif dataset == 'training-a-extended':
        return CINC_EXTENDED_DATASET_POSSIBLE_LABELS
    raise NotImplementedError(f'{dataset=}')


def get_index_label(dataset, index):
    possible_labels = get_possible_labels(dataset)
    return possible_labels[index]


def get_label_index(dataset, label):
    possible_labels = get_possible_labels(dataset)
    assert label in possible_labels, f'{dataset=}, {possible_labels=}, {label=}'
    return torch.tensor([possible_labels.index(label)])


def merge_and_validate_training_a(online_appendix_path, reference_path, reference_sqi_path):
    online_appendix = pd.read_csv(online_appendix_path)

    reference = pd.read_csv(reference_path, header=None, names=["patient", "abnormality"])

    reference_sqi = pd.read_csv(reference_sqi_path, header=None, names=["patient", "abnormality", "SQI"])

    filtered_online_appendix = online_appendix[online_appendix['Database'] == 'training-a']

    merged_data = pd.merge(filtered_online_appendix, reference,
                           left_on='Challenge record name', right_on='patient', how='inner')

    merged_data = pd.merge(merged_data, reference_sqi, on='patient', how='inner')

    assert all(merged_data['abnormality_x'] == merged_data['abnormality_y']
               ), "Discrepancy found in 'abnormality' values!"

    final_data = merged_data[['patient', 'Diagnosis', 'abnormality_x', 'SQI']]
    final_data.columns = ['patient', 'diagnosis', 'abnormality', 'SQI']

    return final_data


class CINCDataset(Dataset):

    def __init__(self, base_dir, dataset, annotations_file, fs, split='train', min_num_samples=None):

        self.dataset = dataset
        self.data_dir = os.path.join(base_dir, 'training-a')
        self.states_dir = os.path.join(base_dir, 'annotations', 'hand_corrected', 'training-a_StateAns')
        self.split = split
        self.fs = fs
        self._min_num_samples = min_num_samples
        self.annotations = self._get_annotations(annotations_file)
        self.patient_data = self._get_patient_data()
        self._count = len(self.patient_data)

    def _get_patient_data(self):

        patient_data = []

        for _, annotation in tqdm(self.annotations.iterrows(),
                                  total=len(self.annotations),
                                  desc=f'Creating {self.split} dataset'):

            patient = annotation.patient
            diagnosis = annotation.diagnosis.strip()
            split = annotation['split']

            if split != self.split:
                continue

            assert diagnosis in get_possible_labels(self.dataset), f'{diagnosis=}'
            info = self._get_patient_info(patient, diagnosis)

            if info is not None:
                patient_data.append(info)

        return patient_data

    def _get_annotations(self, annotations_file):
        return pd.read_csv(annotations_file, skiprows=1)

    def __len__(self):
        return self._count

    def _get_patient_seg_info(self, patient):
        return load_segment_info(os.path.join(self.data_dir, f'{patient}_seg_info.json'))

    def _get_patient_segments(self, patient):
        seg_info = sio.matlab.loadmat(os.path.join(self.states_dir, f'{patient}_StateAns.mat'))['state_ans']
        return seg_info

    def _get_patient_signals(self, patient):
        return readrecord(os.path.join(self.data_dir, patient))

    def _get_patient_info(self, patient, diagnosis):

        label = get_label_index('training-a-extended', diagnosis)

        sigs, old_fs = self._get_patient_signals(patient)

        new_fs = self.fs

        old_sig_len = len(first_dict_value(sigs))

        chirp_wave = add_chirp(np.zeros_like(sigs['ECG']), old_fs)

        sigs = {name: resample(sig, old_fs, new_fs) for name, sig in sigs.items()}

        sig_len = len(first_dict_value(sigs))

        assert old_sig_len * (new_fs / old_fs) == sig_len, f'{old_sig_len=}, {sig_len=}, {old_fs=}, {self.fs=}'
        assert len(sigs['PCG']) == len(sigs['ECG']), f'{len(sigs["PCG"])=}, {len(sigs["ECG"])=}'

        segs = self._get_patient_segments(patient)
        seg_info = from_matform(segs)

        sigs = {name: standardise_signal(sig) for name, sig in sigs.items()}

        sigs['ECG'] = pre_filter_ecg(sigs['ECG'], new_fs)
        sigs['PCG'] = pre_filter_pcg(sigs['PCG'], new_fs)

        sigs = {name: standardise_signal(sig) for name, sig in sigs.items()}

        assert not (self.split != 'valid' and len(seg_info) == 0), f'{seg_info}=, {self.split=}'
        if len(seg_info) == 0:
            return dict(
                label=label,
                patient=patient,
                diagnosis=diagnosis,
                ecg=sigs['ECG'],
                pcg=sigs['PCG'],
            )

        seg_wave = create_segment_waveform(seg_info, old_sig_len)

        seg_wave = resample(seg_wave, old_fs, new_fs)
        chirp_wave = resample(chirp_wave, old_fs, new_fs)

        seg_join_idxs = seg_idxs_from_mat(segs)[1:-1]
        seg_join_idxs, _ = resample_seg_joins(seg_join_idxs, old_fs, new_fs)

        del old_fs

        ecg_segs = segment_signal_tuple(sigs['ECG'], seg_join_idxs)
        pcg_segs = segment_signal_tuple(sigs['PCG'], seg_join_idxs)
        seg_segs = segment_signal_tuple(seg_wave, seg_join_idxs)
        chirp_segs = segment_signal_tuple(chirp_wave, seg_join_idxs)

        ecg_segs = tuple(seg for seg in ecg_segs)
        pcg_segs = tuple(seg for seg in pcg_segs)
        seg_segs = tuple(seg for seg in seg_segs)
        chirp_segs = tuple(seg for seg in chirp_segs)

        assert len(ecg_segs) == len(pcg_segs) == len(seg_segs) == len(chirp_segs), (
            f'{len(ecg_segs)=},{len(pcg_segs)=},{len(seg_segs)=},{len(chirp_segs)=}')

        for i, (e, p, s, c) in enumerate(zip(ecg_segs, pcg_segs, seg_segs, chirp_segs)):
            assert len(e) == len(p) == len(s) == len(c) >= 100, (
                f'{len(e)=},{len(p)=},{len(s)=},{len(c)=},{i=}')

        return dict(
            label=label,
            patient=patient,
            diagnosis=diagnosis,
            ecg_segs=ecg_segs,
            pcg_segs=pcg_segs,
            ecg=sigs['ECG'],
            pcg=sigs['PCG'],
            seg_segs=seg_segs,
            chirp_segs=chirp_segs,
        )

    def __getitem__(self, index):

        return self.patient_data[index].copy()


def get_weighted_sampler(dataset):

    labels = torch.tensor([patient_info['label']
                           for patient_info in dataset])

    class_counts = torch.tensor([sum(labels == t)
                                for t in torch.unique(labels, sorted=True)])

    weights = 1.0 / class_counts

    sample_weights = [float(weights[label])
                      for label in labels]

    assert (len(labels) == len(sample_weights)
            and len(weights) == len(class_counts)
            ), f'{len(labels)=}, {len(class_counts)=}, {len(weights)=}, {len(sample_weights)=}'

    return WeightedRandomSampler(weights=sample_weights, num_samples=len(sample_weights), replacement=True)


class Collator:
    def __init__(self, sample_rate, samples_per_frame, crop_mel_frames, prob_orig_wave, random_start,
                 transform_con, transform_ref, reference_sig_name, condition_sig_name):

        self.sr = sample_rate
        self.samples_per_frame = samples_per_frame
        self.crop_mel_frames = crop_mel_frames
        self.crop_samples = crop_mel_frames * samples_per_frame

        self.prob_full_wav = prob_orig_wave
        self.random_start = random_start

        self.transform_con = transform_con
        self.transform_ref = transform_ref

        self.ref_sig_name = reference_sig_name
        self.con_sig_name = condition_sig_name

        self.sig_names = (reference_sig_name, condition_sig_name)

    def build_signals(self, orig_segs):

        segs = orig_segs
        num_zeros = 0
        # segs, num_zeros = filter_segments(orig_segs,
        #                                   target_signal='pcg',
        #                                   sample_rate=self.sr,
        #                                   threshold_factor=10,
        #                                   )

        num_segs = len(first_dict_value(segs))

        if random.random() <= self.prob_full_wav:

            segs = join_adjacent_segments(segs, num_segments=num_segs, random_start=self.random_start)

        else:

            random_chance = random.random()
            if random_chance < (1 / 3):
                segs = shuffle_segments_multi(
                    segs,
                    group_sizes=[get_weighted_random_choice(num_segs//2) for _ in range(5)],
                    num_segments=num_segs)
            elif random_chance < (2 / 3):
                segs = shuffle_segments_multi(
                    segs,
                    group_sizes=[random.randint(1, 4) for _ in range(5)],
                    num_segments=num_segs)
            else:
                segs = shuffle_segments_multi(
                    segs,
                    group_sizes=[1],
                    num_segments=num_segs)

        num_fade_samples = round(0.01*self.sr)

        built_signals = {name: build_signal(
            segs,
            num_fade_samples=num_fade_samples,
            target_num_samples=self.crop_samples
        ) for name, segs in segs.items()}

        return built_signals, num_zeros

    def collate(self, minibatch):

        for record in minibatch:

            built_signals, zero_segs = self.build_signals({
                sig_name: record[f'{sig_name}_segs']
                for sig_name in [self.ref_sig_name, self.con_sig_name, 'seg', 'chirp']
            })

            record['flag'] = zero_segs

            built_signals['ecg'] = mid_filter_ecg(built_signals['ecg'], self.sr)
            built_signals['pcg'] = mid_filter_pcg(built_signals['pcg'], self.sr)

            built_signals = {sig_name: standardise_signal(signal)
                             for sig_name, signal in built_signals.items()}

            sig_shape = first_dict_value(built_signals).shape
            for name, sig in built_signals.items():
                assert sig.shape == sig_shape, f'{name=}, {sig.shape=}, {sig_shape=}'

            record['ref_audio'] = built_signals[self.ref_sig_name]
            record['con_audio'] = built_signals[self.con_sig_name]
            record['seg_wave'] = built_signals['seg']
            record['chirp_wave'] = built_signals['chirp']

            start = 0
            end = start + self.crop_mel_frames
            record['con_spec'] = create_spectrogram(torch.from_numpy(
                built_signals[self.con_sig_name]).float(), self.transform_con)
            record['con_spec'] = record['con_spec'].T[start:end].T

            assert record['con_spec'].shape[1] >= self.crop_mel_frames, (
                f'{record["con_spec"].shape[1]=},{self.crop_mel_frames=}')

            start *= self.samples_per_frame
            end *= self.samples_per_frame

            for wave_name in ['ref_audio', 'con_audio', 'seg_wave', 'chirp_wave']:

                record[wave_name] = record[wave_name][start:end]
                record[wave_name] = standardise_signal(record[wave_name])
                record[wave_name] = fade_signal(record[wave_name], round(0.01*self.sr))
                record[wave_name] = np.pad(record[wave_name],
                                           (0, (end-start) - len(record[wave_name])),
                                           mode='constant')

        return_dict = {
            stack_name: torch.from_numpy(
                np.stack([record[stack_name] for record in minibatch if stack_name in record]))
            for stack_name in ['ref_audio', 'con_spec', 'label', 'con_audio', 'seg_wave', 'chirp_wave']
        }

        return_dict = {
            stack_name: stack_val.float() if stack_name not in ['label'] else stack_val
            for stack_name, stack_val in return_dict.items()
        }

        return_dict.update({'patient': [record['patient'] for record in minibatch],
                            'flag': [record['flag'] for record in minibatch]})  # type: ignore

        return return_dict


def get_dataloaders(annotations_file, input_dir,
                    reference_sig_name, condition_sig_name,
                    dataset, generative_model):

    params = generative.models.get_params(generative_model)

    transform_ref = generative.models.get_transform(generative_model=generative_model, signal_name=reference_sig_name)
    transform_con = generative.models.get_transform(generative_model=generative_model, signal_name=condition_sig_name)
    post_transform_ref = generative.models.get_post_transform(signal_name=reference_sig_name)
    post_transform_con = generative.models.get_post_transform(signal_name=condition_sig_name)

    min_num_samples = params['crop_mel_frames'] * params['sample_rate']

    shared_dataset_kwargs = dict(
        base_dir=input_dir,
        dataset=dataset,
        annotations_file=annotations_file,
        min_num_samples=min_num_samples,
        fs=params['sample_rate'],
    )

    dataset_train = CINCDataset(
        split='train',
        **shared_dataset_kwargs,
    )

    dataset_valid = CINCDataset(
        split='valid',
        **shared_dataset_kwargs,
    )

    dataset_test = CINCDataset(
        split='test',
        **shared_dataset_kwargs,
    )

    shared_collator_kwargs = dict(
        sample_rate=params['sample_rate'],
        samples_per_frame=params['hop_samples'],
        crop_mel_frames=params['crop_mel_frames'],
        transform_ref=transform_ref,
        transform_con=transform_con,
        reference_sig_name=reference_sig_name,
        condition_sig_name=condition_sig_name,
        prob_orig_wave=1,  # TODO: delete this
        random_start=False,
    )

    train_collator = Collator(
        # prob_orig_wave=0.25,
        # random_start=True,
        **shared_collator_kwargs,
    )

    valid_collator = Collator(
        # prob_orig_wave=1,
        # random_start=True,
        **shared_collator_kwargs,
    )

    test_collator = Collator(
        # prob_orig_wave=1,
        # random_start=True,
        **shared_collator_kwargs,
    )

    shared_dataloader_kwargs = dict(
        batch_size=params['batch_size'],
        pin_memory=True,
        num_workers=0,
        worker_init_fn=seed_worker,
        generator=torch_gen,
    )

    dataloader_train = DataLoader(
        dataset_train,
        # shuffle=True,
        drop_last=True,
        sampler=get_weighted_sampler(dataset_train),
        collate_fn=train_collator.collate,
        **shared_dataloader_kwargs,  # type: ignore
    )

    dataloader_valid = DataLoader(
        dataset_valid,
        shuffle=False,
        drop_last=False,
        collate_fn=valid_collator.collate,
        **shared_dataloader_kwargs,  # type: ignore
    )

    dataloader_test = DataLoader(
        dataset_test,
        shuffle=False,
        drop_last=False,
        collate_fn=test_collator.collate,
        **shared_dataloader_kwargs,  # type: ignore
    )

    datasets = {'train': dataset_train, 'valid': dataset_valid, 'test': dataset_test}
    dataloaders = {'train': dataloader_train, 'valid': dataloader_valid, 'test': dataloader_test}

    data = {
        'datasets': datasets,
        'dataloaders': dataloaders,
        'transform_ref': transform_ref,
        'transform_con': transform_con,
        'post_transform_ref': post_transform_ref,
        'post_transform_con': post_transform_con,
        'params': params,
    }

    return data
