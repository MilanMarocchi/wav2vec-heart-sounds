import os
import time
import random

import matplotlib.pyplot as plt
import sounddevice as sd
import numpy as np
import wfdb
import librosa

# from sklearn.decomposition import PCA
from tqdm.auto import tqdm

import utils.plotting

from utils.utils import pretty_print_shapes, add_beep
from processing.datasets import get_possible_labels
from processing.filtering import standardise_signal


def get_patients_from_datasets(datasets, dataset):

    patients = {dset: {diagnosis: {} for diagnosis in get_possible_labels(dataset)} for dset in datasets}
    for dset in datasets:
        patients[dset].update({'all': {}})

    for dset_name, dset in datasets.items():
        for item in dset:
            patient_dict = {item['patient']: []}
            patient = item['patient']
            patients[dset_name]['all'][patient] = patient_dict
            patients[dset_name][item['diagnosis']][patient] = patient_dict

    return patients


# def unbatch_patient_data(patient_information):
#
#     patient_dict = {}
#
#     for patient in patient_information:
#         reformatted_data = {k: [] for k in patient_information[patient][0]}
#
#         for i in range(len(patient_information[patient])):
#             for k in reformatted_data:
#                 reformatted_data[k].append(patient_information[patient][i][k])
#
#         patient_dict[patient] = reformatted_data
#
#     return patient_dict


def get_patient_information_from_dataloaders(dataloaders, datasets, num_runs, dataset):

    patient_information = get_patients_from_datasets(datasets, dataset)
    patient_information = {dset_k: {cat_k: {pat: []
                                            for pat in cat_v}
                                    for cat_k, cat_v in dset_v.items()}
                           for dset_k, dset_v in patient_information.items()}

    for _ in tqdm(range(num_runs), desc=f'Sampling {num_runs} times'):

        for split, dataloader in dataloaders.items():

            for _, batch in enumerate(dataloader):

                for unbatch_idx, patient in enumerate(batch['patient']):
                    patient_info = {k: batch[k][unbatch_idx] for k in batch}
                    patient_information[split]['all'][patient].append(patient_info)
                    diagnosis = get_possible_labels(dataset)[batch['label'][unbatch_idx].item()]
                    patient_information[split][diagnosis][patient].append(patient_info)

    return patient_information


def demonstrate_sampling(data, dataset):

    patient_information = get_patient_information_from_dataloaders(
        data['dataloaders'], data['datasets'], num_runs=5, dataset=dataset)

    print(pretty_print_shapes(patient_information, depth=4))

    diagnosis_counts = {diagnosis: 0 for diagnosis in get_possible_labels(dataset)}
    patient_counts = {diagnosis: [] for diagnosis in get_possible_labels(dataset)}

    for diag, patients in patient_information['train'].items():
        if diag == 'all':
            continue
        for patient, patient_data in patients.items():
            diagnosis_counts[diag] += len(patient_data)
            patient_counts[diag].append(patient)

    _, axes = plt.subplots(1, 2)

    axes[0].bar(list(diagnosis_counts.keys()), list(diagnosis_counts.values()))
    axes[0].set_title('Total Sampling Rate for Each Class')
    axes[0].set_ylabel('Count')

    # total_count = sum(abnormal_patients.values()) + sum(normal_patients.values())
    # total_patient_num = len(abnormal_patients) + len(normal_patients)

    # weights_normal = [1 / total_patient_num * 100 for _ in normal_patients.values()]
    # weights_abnormal = [1 / total_patient_num * 100 for _ in abnormal_patients.values()]

    # axes[1].hist([list(normal_patients.values()), list(abnormal_patients.values())],
    #              label=['normal', 'abnormal'],
    #              bins=20, align='left')
    # axes[1].set_title('Frequency Distribution for Patient Sampling')
    # axes[1].set_xlabel('Number of times sampled')
    # axes[1].set_ylabel('Number of patients')
    # axes[1].grid(axis='both')

    # fig.tight_layout()
    # fig.show()

    # fig, ax = plt.subplots(1, 1)
    # ax.bar(normal_patients.keys(), normal_patients.values(), color='blue', label='Normal')
    # ax.bar(abnormal_patients.keys(), abnormal_patients.values(), color='red', label='Abnormal')
    # ax.set_title('Sampling Frequency for Each Patient')
    # ax.set_xlabel('Patient ID')
    # ax.set_ylabel('Number of times sampled')
    # ax.legend()

    # ax.set_xticklabels(list(normal_patients.keys()) + list(abnormal_patients.keys()),
    #                    rotation=45, ha='right', fontsize=8)

    # fig.tight_layout()
    # fig.show()

    plt.show()


def plot_and_listen(patient_info, patient, dataset, sr, ref_transform, con_transform, listen=False, max_sigs=None):
    if max_sigs is None:
        num_sigs = len(patient_info)
    else:
        num_sigs = min(max_sigs, len(patient_info))

    assert num_sigs > 0, f'{num_sigs=}'

    print(patient, patient_info[0]['label'])
    num_plot_types = 8
    fig, axes = plt.subplots(num_plot_types, num_sigs, squeeze=False)

    for i in range(num_sigs):

        utils.plotting.plot_wav(axes[0][i], patient_info[i]['ref_audio'], sr)
        utils.plotting.plot_wav(axes[1][i], patient_info[i]['con_audio'], sr)
        utils.plotting.plot_wav(axes[2][i], patient_info[i]['seg_wave'], sr)
        utils.plotting.plot_stft(axes[3][i], patient_info[i]['ref_audio'], sr)
        utils.plotting.plot_stft(axes[4][i], patient_info[i]['con_audio'], sr)
        utils.plotting.plot_stft(axes[5][i], patient_info[i]['chirp_wave'], sr)
        utils.plotting.plot_spec(axes[6][i], patient_info[i]['ref_audio'], transform=ref_transform)
        utils.plotting.plot_spec(axes[7][i], patient_info[i]['con_audio'], transform=con_transform)

        for j in range(num_plot_types):
            if i not in [0, 6, 7]:
                axes[j][i].sharex(axes[j][0])

            axes[j][i].set_xlabel('')
            axes[j][i].set_xticks([])
            axes[j][i].set_xticklabels([])
            axes[j][i].set_ylabel('')
            axes[j][i].set_yticks([])
            axes[j][i].set_yticklabels([])
            axes[j][i].set_title('')
            axes[j][i].patch.set_visible(False)
            axes[j][i].set_frame_on(False)

        axes[0][0].set_ylabel('PCG WAV', fontsize='large')
        axes[1][0].set_ylabel('ECG WAV', fontsize='large')
        axes[2][0].set_ylabel('SEG WAV', fontsize='large')
        axes[3][0].set_ylabel('PCG STFT', fontsize='large')
        axes[4][0].set_ylabel('ECG STFT', fontsize='large')
        axes[5][0].set_ylabel('CHIRP STFT', fontsize='large')
        axes[6][0].set_ylabel('PCG MEL', fontsize='large')
        axes[7][0].set_ylabel('ECG MEL', fontsize='large')

    diagnosis = get_possible_labels(dataset)[patient_info[0]['label'].item()]
    fig.suptitle(f'{patient=}, {diagnosis=}')

    fig.tight_layout()
    fig.subplots_adjust(wspace=0.1, hspace=0.1)
    fig.show()

    if listen:
        audio_data = np.concatenate([
            np.concatenate([
                add_beep(round((1 + i/num_sigs)*sr/4), num_beeps=i+1, fs=sr, amplitude=0.05),
                patient_info[i]['ref_audio']
            ])
            for i in range(num_sigs)])
        sd.play(audio_data, sr)


def demonstrate_shuffling(data, dataset):

    # for gui in mpl.rcsetup.interactive_bk:  # type: ignore
    #     print(gui)

    patient_information = get_patient_information_from_dataloaders(
        data['dataloaders'], data['datasets'], num_runs=5, dataset=dataset)

    stop = False
    try:
        for dset_name, dset in patient_information.items():
            if stop:
                break
            assert dset_name in ['train', 'valid', 'test'], f'{dset_name=}'
            for diagnosis, split in dset.items():
                if stop:
                    break
                if diagnosis in ['all']:
                    continue

                if diagnosis not in ['MVP', 'MPC', 'AD']:
                    continue

                for patient_name, patient_info in sorted(split.items(),
                                                         key=lambda kv: len(kv[1]),
                                                         reverse=True):
                    if stop:
                        break

                    # skip = True
                    # for i in range(len(patient_info)):
                    #     info = patient_info[i]['flag']
                    #     if info != 0:
                    #         skip = False

                    # if skip:
                    #     continue

                    # if patient_name not in ['a0001', 'a0028', 'a0100', 'a0296', 'a0039', 'a0093', 'a0105']:
                    #     continue

                    print(patient_name, dset_name)
                    print(len(patient_info))

                    plot_and_listen(patient_info, patient_name, dataset,
                                    sr=data['params']['sample_rate'], max_sigs=8, listen=True,
                                    ref_transform=data['transform_ref'],
                                    con_transform=data['transform_con'])
                    stop = input('Continue? ').upper() != 'Y'
                    sd.stop()
                    plt.close('all')
    finally:
        sd.stop()
        plt.close('all')


def compare_hpss_params(y, sr, param_list, patient_name):

    # Prepare the figure
    num_params = len(param_list)
    fig, axes = plt.subplots(num_params, 8, figsize=(20, 4 * num_params), constrained_layout=True)

    combined_waveforms = []

    for i, (stft_params, hpss_params) in enumerate(param_list):
        # Compute the STFT
        D = librosa.stft(y, **stft_params)
        D_mag_dB = librosa.amplitude_to_db(np.abs(D), ref=np.max)
        vmin = D_mag_dB.min()
        vmax = D_mag_dB.max()

        # Perform HPSS
        harmonic, percussive = librosa.decompose.hpss(D, **hpss_params)

        # Convert the harmonic and percussive components back to time domain
        y_harmonic = librosa.istft(harmonic)
        y_percussive = librosa.istft(percussive)
        y_combined = y_harmonic + y_percussive

        combined_waveforms.extend([y, y_percussive, y_harmonic, y_combined])

        # Plot waveform and spectrogram for original, harmonic, percussive, and combined
        for j, component in enumerate([y, y_percussive, y_harmonic, y_combined]):
            # Select the axis for the waveform and spectrogram
            ax_waveform = axes[i, j*2]
            ax_spectrogram = axes[i, j*2+1]

            # Waveform plot
            librosa.display.waveshow(component, sr=sr, alpha=0.5, ax=ax_waveform,
                                     color=['red', 'green', 'blue', 'black'][j],
                                     linewidth=0.5, linestyle='--')
            ax_waveform.set_title(
                f'{"Original" if j == 0 else "Percussive" if j == 1 else "Harmonic" if j == 2 else "Combined"}'
                f'Waveform - Params {i+1}')
            ax_waveform.set_xlabel('Time (s)')
            ax_waveform.set_ylabel('Amplitude')
            ax_waveform.label_outer()  # Hide labels for shared axes

            # Spectrogram plot
            component_stft = librosa.stft(component, **stft_params)
            D_component_db = librosa.amplitude_to_db(np.abs(component_stft), ref=np.max)
            img = librosa.display.specshow(D_component_db, sr=sr, x_axis='time', y_axis='log',
                                           ax=ax_spectrogram, cmap='jet',
                                           vmin=vmin, vmax=vmax)
            ax_spectrogram.set_title(
                f'{"Original" if j == 0 else "Percussive" if j == 1 else "Harmonic" if j == 2 else "Combined"}'
                f'STFT - Params {i+1}')
            ax_spectrogram.set_xlabel('Time (s)')
            ax_spectrogram.set_ylabel('Frequency (Hz)')
            ax_spectrogram.label_outer()  # Hide labels for shared axes
            fig.colorbar(img, ax=ax_spectrogram, format='%+2.0f dB')

            if j not in [0]:
                axes[i][j*2].sharey(axes[i][0])
                axes[i][j*2].sharex(axes[i][0])
                axes[i][j*2+1].sharex(axes[i][1])

    fig.suptitle(patient_name)

    num_sigs = len(combined_waveforms)

    audio_data = np.concatenate([
        np.concatenate([
            add_beep(round(sr/8), beep_duration=0.25, num_beeps=((i // 4) + 1), fs=sr, amplitude=0.05),
            add_beep(round((1 + i/num_sigs)*sr/4), num_beeps=((i % 4) + 1), fs=sr, amplitude=0.05),
            combined_waveforms[i]
        ])
        for i in range(num_sigs)
        if i == 0 or i % 4 not in [0, 3]])

    fig.show()

    audio_data = np.concatenate([np.zeros(shape=[sr*20], dtype=audio_data.dtype), audio_data])

    sd.play(audio_data, sr)


def compare_hpss_param_pairs(y, sr, parameter_combination_pairs, plot_title):
    # Prepare the figure
    num_params = len(parameter_combination_pairs)
    num_groups = 6

    fig, axes = plt.subplots(num_params, num_groups*2, figsize=(25, 30), constrained_layout=True)

    y = standardise_signal(y)

    y_max = np.max(np.abs(y))
    combined_waveforms = []

    for i, (first_params, second_params) in enumerate(parameter_combination_pairs):

        # Compute the STFT
        D = librosa.stft(y, **first_params[0])
        D_mag_dB = librosa.amplitude_to_db(np.abs(D), ref=y_max)
        vmin = D_mag_dB.min()
        vmax = D_mag_dB.max()

        # Perform the first HPSS
        background, residual = librosa.decompose.hpss(D, **first_params[1])

        y_background = librosa.istft(background, **first_params[0])
        y_residual = librosa.istft(residual, **first_params[0])

        D_residual = librosa.stft(y_residual, **second_params[0])

        # Perform the second HPSS on the residual
        murmurs, heart_sounds = librosa.decompose.hpss(D_residual, **second_params[1])

        y_murmurs = librosa.istft(murmurs, **second_params[0])
        y_heart_sounds = librosa.istft(heart_sounds, **second_params[0])

        min_len = min(len(y_i) for y_i in (y_background, y_murmurs, y_heart_sounds))

        y_enhanced = standardise_signal(
            0.1 * y_background[:min_len]
            + 2 * y_murmurs[:min_len]
            + y_heart_sounds[:min_len]
        )

        waveforms = [y, y_background, y_residual, y_murmurs, y_heart_sounds, y_enhanced]

        combined_waveforms.extend(waveforms)

        plot_stft_params = {'n_fft': 2048,
                            'hop_length': 512,
                            'win_length': 2048,
                            'window': 'hann',
                            }

        # Plot waveform and spectrogram for original, harmonic, first percussive, new percussive, and final residual
        for j, component in enumerate(waveforms):
            # Select the axis for the waveform and spectrogram
            ax_waveform = axes[i, j*2]
            ax_spectrogram = axes[i, j*2+1]

            # Waveform plot
            librosa.display.waveshow(standardise_signal(component), sr=sr, alpha=0.5, ax=ax_waveform,
                                     color=['red', 'green', 'blue', 'orange', 'purple', 'black'][j],
                                     linewidth=0.5, linestyle='--')
            ax_waveform.set_title(
                ('Original' if j == 0
                 else 'Background' if j == 1
                 else 'Residual' if j == 2
                 else 'Murmurs' if j == 3
                 else 'Transient' if j == 4
                 else 'ENHANCE')
                + f' Wave {i+1}')
            ax_waveform.set_xlabel('Time (s)')
            ax_waveform.set_ylabel('Amplitude')
            ax_waveform.label_outer()  # Hide labels for shared axes

            # Spectrogram plot
            # if j < 5:  # Avoid recomputing STFT for the combined signal
            component_stft = librosa.stft(component, **plot_stft_params)
            D_component_db = librosa.amplitude_to_db(np.abs(component_stft), ref=y_max)
            img = librosa.display.specshow(D_component_db, sr=sr, x_axis='time', y_axis='log',
                                           ax=ax_spectrogram, cmap='jet',
                                           vmin=vmin, vmax=vmax)
            ax_spectrogram.set_title(
                ('Original' if j == 0
                 else 'Background' if j == 1
                 else 'Residual' if j == 2
                 else 'Murmurs' if j == 3
                 else 'Heart Sounds' if j == 4
                 else 'ENHANCE')
                + f' STFT {i+1}')
            ax_spectrogram.set_xlabel('Time (s)')
            ax_spectrogram.set_ylabel('Frequency (Hz)')
            ax_spectrogram.label_outer()  # Hide labels for shared axes
            fig.colorbar(img, ax=ax_spectrogram, format='%+2.0f dB')

            # Ensure the plots share axes where appropriate
            if j not in [0]:
                axes[i][j*2].sharey(axes[i][0])
                axes[i][j*2+1].sharex(axes[i][1])

    fig.suptitle(plot_title)

    # Combine waveforms, skipping the combined signal to avoid redundancy
    num_sigs = len(combined_waveforms)
    audio_data = np.concatenate([
        np.concatenate([
            add_beep(round(sr/8), beep_duration=0.25, num_beeps=((i // num_groups) + 1), fs=sr, amplitude=0.025),
            add_beep(round((1 + i/num_sigs)*sr/num_groups), num_beeps=((i % num_groups) + 1), fs=sr, amplitude=0.025),
            standardise_signal(combined_waveforms[i])
        ])
        for i in range(num_sigs)
        # if i == 0 or i % num_groups not in [0, num_groups-1]])
    ])

    # Display the figure
    fig.show()

    # Prepare the audio with a silence at the beginning
    audio_data = np.concatenate([np.zeros(shape=[sr*20], dtype=audio_data.dtype), audio_data])

    # Play the audio
    sd.play(audio_data, sr)


def demonstrate_augmentation(data, dataset):
    del dataset

    random.seed(time.time_ns())
    random.shuffle(data['datasets']['train'].patient_data)

    stop = False

    try:
        for dataset in data['datasets']:
            if stop:
                break
            for patient in data['datasets'][dataset]:
                if stop:
                    break

                patient_name = patient['patient']
                diagnosis = patient['diagnosis']

                if patient_name not in ['a0071', 'a0313', 'a0024', 'a0309']:
                    continue

                # if diagnosis not in ['Benign']:
                #     continue

                ref_audio = np.concatenate(patient['pcg_segs'][:8], axis=0, dtype=float)

                parameter_combination_pairs = [
                    (
                        # First HPSS for constant/background sounds
                        ({
                            'n_fft': 512,
                            'hop_length': 128,
                            'win_length': 512,
                            'window': 'hann'
                        }, {
                            'kernel_size': (17, 17),
                            'margin': (1.6, 1.6),
                        }),
                        # Second HPSS for transient sounds
                        ({
                            'n_fft': 448,
                            'hop_length': 112,
                            'win_length': 448,
                            'window': 'hann'
                        }, {
                            'kernel_size': (29, 29),
                            'margin': (1.3, 2.2),
                        }),
                    ),
                    (
                        # First HPSS for constant/background sounds
                        ({
                            'n_fft': 512,
                            'hop_length': 128,
                            'win_length': 512,
                            'window': 'hann'
                        }, {
                            'kernel_size': (24, 24),
                            'margin': (1.6, 1.6),
                        }),
                        # Second HPSS for transient sounds
                        ({
                            'n_fft': 1024,
                            'hop_length': 256,
                            'win_length': 1024,
                            'window': 'hann'
                        }, {
                            'kernel_size': (21, 21),
                            'margin': (1.1, 2.5),
                        }),
                    ),
                    (
                        # First HPSS for constant/background sounds
                        ({
                            'n_fft': 2048,
                            'hop_length': 512,
                            'win_length': 2048,
                            'window': 'hann'
                        }, {
                            'kernel_size': (31, 31),
                            'margin': (1.8, 1.8),
                        }),
                        # Second HPSS for transient sounds
                        ({
                            'n_fft': 448,
                            'hop_length': 112,
                            'win_length': 448,
                            'window': 'hann'
                        }, {
                            'kernel_size': (29, 29),
                            'margin': (1.3, 2.2),
                        }),
                    ),
                    (
                        # First HPSS for constant/background sounds
                        ({
                            'n_fft': 1792,         # Detailed frequency resolution
                            'hop_length': 448,     # Consistent 1/4 ratio
                            'win_length': 1792,
                            'window': 'hann'
                        }, {
                            'kernel_size': (37, 37),  # Broader strokes
                            'margin': (1.75, 1.75),   # Balanced HPSS
                        }),
                        # Second HPSS for transient sounds
                        ({
                            'n_fft': 512,
                            'hop_length': 128,
                            'win_length': 512,
                            'window': 'hann'
                        }, {
                            'kernel_size': (25, 25),
                            'margin': (1.2, 2.0),
                        }),
                    ),
                    (
                        # First HPSS for constant/background sounds
                        ({
                            'n_fft': 640,          # Trade-off between time and frequency resolution
                            'hop_length': 160,     # 1/4 of n_fft
                            'win_length': 640,
                            'window': 'hann'
                        }, {
                            'kernel_size': (22, 22),  # Moderately small kernel
                            'margin': (1.5, 2.0),    # Balancing percussive and harmonic elements
                        }),
                        # Second HPSS for transient sounds
                        ({
                            'n_fft': 512,
                            'hop_length': 128,
                            'win_length': 512,
                            'window': 'hann'
                        }, {
                            'kernel_size': (27, 27),
                            'margin': (1.1, 2.3),
                        }),
                    ),
                ]

                compare_hpss_param_pairs(ref_audio, 4000, parameter_combination_pairs,
                                         '+'.join([patient_name, diagnosis]))
                stop = input('Continue? ').upper() != 'Y'
                sd.stop()
                plt.close('all')

    finally:
        sd.stop()
        plt.close('all')


class RandomDataPointSampler:
    def __init__(self, df):
        self.df = df.copy()
        self.original_indexes = set(self.df.index)
        self.reset_iterator()

        # Calculate weights based on the inverse frequency of diagnosis and recording location combinations
        combination_counts = df.groupby(['Diagnosis', 'Signal record site']).size()
        self.df['weight'] = df.apply(
            lambda row: 1 / combination_counts[(row['Diagnosis'], row['Signal record site'])], axis=1)

    def reset_iterator(self):
        self.indexes = self.original_indexes.copy()

    def __iter__(self):
        self.reset_iterator()  # Reset the iterator when a new iteration starts
        return self

    def __next__(self):
        if not self.indexes:
            raise StopIteration

        available_weights = self.df.loc[list(self.indexes), 'weight']
        # Normalize weights to sum to 1 for the available indexes
        available_weights /= available_weights.sum()

        chosen_index = np.random.choice(available_weights.index, p=available_weights)
        self.indexes.remove(chosen_index)

        return self.df.loc[chosen_index].drop('weight')

    def __len__(self):
        return len(self.df)


def sample_record(filename, target_sig_len_s, target_symbol, target_proportion,
                  min_anns, max_tries=50, max_Q_props=0.1):
    header = wfdb.rdheader(filename)
    sig_len = header.sig_len
    old_fs = header.fs
    assert old_fs is not None

    target_sig_len = target_sig_len_s * old_fs

    best_rec = None
    best_ann = None
    best_symbol_count = 0

    for _ in range(max_tries):
        start = random.randint(0, sig_len - target_sig_len)
        end = start + target_sig_len

        rec = wfdb.rdrecord(filename, sampfrom=start, sampto=end)
        ann = wfdb.rdann(filename, 'atr', sampfrom=start, sampto=end, shift_samps=True)

        assert ann.symbol is not None

        symbol_count = ann.symbol.count(target_symbol)
        total_symbols = len(ann.symbol)

        if symbol_count > best_symbol_count:
            best_rec, best_ann, best_symbol_count = rec, ann, symbol_count

        if total_symbols < min_anns:
            continue

        if ann.symbol.count('Q') > max_Q_props * total_symbols:
            continue

        if symbol_count / total_symbols >= target_proportion:
            return rec, ann

    return best_rec, best_ann


def plot_selected_records(round_num, selected_records_with_symbol, df, folder, target_sig_len_s=20):

    for _, (pcg_class, records) in enumerate(selected_records_with_symbol.items()):
        if len(records) > round_num:

            record_name, target_symbol = records[round_num]
            target_proportion = df[df['record name'] == record_name][target_symbol].values[0]
            filename = os.path.join(folder, record_name)

            print(df[df['record name'] == record_name])
            rec, ann = sample_record(filename, target_sig_len_s, target_symbol,
                                     target_proportion, min_anns=round(target_sig_len_s*0.4))

            wfdb.plot_wfdb(rec, ann, plot_sym=True, title=f'{pcg_class=}, {target_symbol=}')


def normalize_prioritization(prioritization_dict):
    normalized_dict = {}
    for pcg, priorities in prioritization_dict.items():
        total = sum([value for _, value in priorities])
        normalized_priorities = [(annotation, value/total) for annotation, value in priorities]
        normalized_dict[pcg] = normalized_priorities
    return normalized_dict


def plot_record_distributions(df, selected_records, columns_to_normalize):
    selected_df = df[df['record name'].isin([item for sublist in selected_records.values() for item, _ in sublist])]
    avg_proportions = selected_df.groupby(selected_df['record name'].map(lambda x: next(
        (k for k, v in selected_records.items() if x in v), None)))[columns_to_normalize].mean()

    bar_width = 0.15
    index = np.arange(len(avg_proportions.index))

    _, ax = plt.subplots(figsize=(12, 7))
    for i, column in enumerate(columns_to_normalize):
        ax.bar(index + i*bar_width, avg_proportions[column], bar_width, label=column)

    ax.set_xlabel('PCG Classification')
    ax.set_ylabel('Average Proportion')
    ax.set_title('Average Proportions of ECG Annotations for Each PCG Classification')
    ax.set_xticks(index + bar_width * (len(columns_to_normalize) - 1) / 2)
    ax.set_xticklabels(avg_proportions.index)
    ax.legend()

    plt.tight_layout()
    plt.show()


def demonstrate_syms_prioritisation(df, input_dir):

    df = df.rename({'record_name': 'record name'}, axis='columns')

    columns_to_normalize = ["N", "+", "Q", "S", "V"]
    df[columns_to_normalize] = df[columns_to_normalize].div(df[columns_to_normalize].sum(axis=1), axis=0)
    df = df[df['Q'] <= 0.1]

    prioritization = {
        'Aortic disease': [('+', 40), ('V', 30), ('S', 20), ('N', 1), ('Q', 0)],
        'Miscellaneous': [('+', 30), ('V', 25), ('S', 25), ('N', 1), ('Q', 0)],
        'Mitral valve prolapse': [('V', 40), ('S', 30), ('+', 25), ('N', 1), ('Q', 0)],
        'Benign': [('N', 55), ('+', 25), ('S', 10), ('V', 10), ('Q', 0)],
        'Normal': [('N', 80), ('+', 10), ('V', 5), ('S', 5), ('Q', 0)],
    }

    prioritization = normalize_prioritization(prioritization)

    for pcg, prios in prioritization.items():
        weights = [p[1] for p in prios]
        labels = [p[0] for p in prios]
        assert len(labels) == len(set(labels)), f'{labels=}'

        assert 0.99 < sum(weights) < 1.01, f'{prios=}, {sum(weights)=}'

    k = 200
    selected_records_with_symbol = {pcg: [] for pcg in prioritization}

    for _ in tqdm(range(k)):
        for pcg, priorities in prioritization.items():

            annotations, weights = zip(*priorities)
            chosen_annotation = np.random.choice(annotations, p=weights)

            df_sorted = df.sort_values(by=chosen_annotation, ascending=False)
            if chosen_annotation != 'N':
                df_sorted = df_sorted[df_sorted['N'] < df_sorted['N'].mean()]

            for _, row in df_sorted.iterrows():
                record_name = row['record name']
                if record_name not in [item[0]
                                       for sublist in selected_records_with_symbol.values()
                                       for item in sublist]:
                    selected_records_with_symbol[pcg].append((record_name, chosen_annotation))
                    break

    # plot_record_distributions(df, selected_records_with_symbol, columns_to_normalize)

    while round_num := int(input(f'Round num? (0 to stop, {k} is highest) ')):
        plot_selected_records(round_num-1, selected_records_with_symbol, df, input_dir)


def plot_and_listen_generative(patient_info, patient, diagnosis, sr, ref_transform, con_transform, listen=False):
    num_plot_types = 6
    fig, axes = plt.subplots(num_plot_types, 1, squeeze=False)

    utils.plotting.plot_wav(axes[0][0], patient_info['gen_sig'], sr)
    utils.plotting.plot_wav(axes[1][0], patient_info['con_sig'], sr)
    utils.plotting.plot_stft(axes[2][0], patient_info['gen_sig'], sr)
    utils.plotting.plot_stft(axes[3][0], patient_info['con_sig'], sr)
    utils.plotting.plot_spec(axes[4][0], patient_info['gen_sig'], transform=ref_transform)
    utils.plotting.plot_spec(axes[5][0], patient_info['con_sig'], transform=con_transform)

    for j in range(num_plot_types):

        axes[j][0].set_xlabel('')
        axes[j][0].set_xticks([])
        axes[j][0].set_xticklabels([])
        axes[j][0].set_ylabel('')
        axes[j][0].set_yticks([])
        axes[j][0].set_yticklabels([])
        axes[j][0].set_title('')
        axes[j][0].patch.set_visible(False)
        axes[j][0].set_frame_on(False)

    axes[0][0].set_ylabel('PCG WAV', fontsize='large')
    axes[1][0].set_ylabel('ECG WAV', fontsize='large')
    axes[2][0].set_ylabel('PCG STFT', fontsize='large')
    axes[3][0].set_ylabel('ECG STFT', fontsize='large')
    axes[4][0].set_ylabel('PCG MEL', fontsize='large')
    axes[5][0].set_ylabel('ECG MEL', fontsize='large')

    fig.suptitle(f'{patient=}, {diagnosis=}')

    fig.tight_layout()
    fig.subplots_adjust(wspace=0.1, hspace=0.1)
    fig.show()

    if listen:
        audio_data = np.concatenate([
            np.concatenate([
                add_beep(round((1 + i/5)*sr/4), num_beeps=i+1, fs=sr, amplitude=0.05),
                patient_info['gen_sig']
            ])
            for i in range(5)])
        sd.play(audio_data, sr)


def demonstrate_generated_icentia(dataset):

    random.seed(time.time_ns())
    random.shuffle(dataset.patient_data)

    stop = False

    try:
        for patient in dataset:
            if stop:
                break

            patient_name = patient['patient']
            diagnosis = patient['diagnosis']

            plot_and_listen_generative(
                patient_info=patient,
                patient=patient_name,
                diagnosis=diagnosis,
                sr=4000,
                ref_transform=dataset._transform_gen,
                con_transform=dataset._transform_con,
                listen=True
            )

            stop = input('Continue? ').upper() != 'Y'
            sd.stop()
            plt.close('all')

    finally:
        sd.stop()
        plt.close('all')
