import os
import random

from copy import deepcopy
from collections import Counter

import wfdb
import wfdb.processing as wproc
import torch
import torchaudio
import numpy as np
import pandas as pd

from tqdm.auto import tqdm

from processing.filtering import (
    start_matlab,
    stop_matlab,
    standardise_signal,
    pre_filter_ecg,
    mid_filter_ecg,
    pre_filter_pcg,
    get_pcg_segs_idx,
    resample,
    add_chirp,
)

from processing.segments import (
    get_seg_info,
    save_segment_info,
    resample_segments,
    get_shortest_signal_len,
    create_segment_waveform,
    segment_signal,
    get_seg_join_idx,
    shift_seg_idxs,
    some_shuffle_segments_fn,
    build_signal,
)

from utils.utils import first_dict_value

from utils.plotting import plot_multi_signals_HACKY


def get_record(path, max_sig_len_s=None):

    header = wfdb.rdheader(path)
    sig_len = header.sig_len
    fs = header.fs

    if max_sig_len_s is None:
        target_sig_len = sig_len
    else:
        target_sig_len = round(max_sig_len_s * fs)

    if sig_len > target_sig_len:
        sampfrom = random.randint(0, sig_len - target_sig_len)
        sampto = sampfrom + target_sig_len
    else:
        sampfrom = 0
        sampto = sig_len

    rec = wfdb.rdrecord(path, sampfrom=sampfrom, sampto=sampto)
    return rec


def save_signals(sigs_dict, group_name, output_dir, fs):
    os.makedirs(output_dir, exist_ok=True)

    sigs_dict = {sig_name: standardise_signal(sig) for sig_name, sig in sigs_dict.items()}
    wfdb.wrsamp(f'{group_name}',
                fs=fs,
                units=['mV' for _ in sigs_dict],
                sig_name=list(sigs_dict.keys()),
                p_signal=np.stack([standardise_signal(sig) for sig in sigs_dict.values()], axis=1),
                write_dir=output_dir)

    for sig_name, signal in sigs_dict.items():
        filename = f'{group_name}_{sig_name}.wav'
        signal = torch.from_numpy(signal).unsqueeze(0)
        torchaudio.save(os.path.join(output_dir, filename), signal, fs)  # type: ignore


def process_psigs(multi_sigs_dict, old_fs, new_fs, output_dir, patient_name):

    seg_idxs = seg_fs = None

    min_len = get_shortest_signal_len(multi_sigs_dict)
    crop_start = round(0.1*new_fs)
    crop_end = min_len - round(0.1*new_fs)

    multi_sigs_dict = {name: sig[crop_start:crop_end] for name, sig in multi_sigs_dict.items()}
    multi_sigs_dict = {name: standardise_signal(sig) for name, sig in multi_sigs_dict.items()}

    if 'PCG' in multi_sigs_dict:
        seg_idxs, seg_fs = get_pcg_segs_idx(multi_sigs_dict['PCG'].copy(), old_fs, new_fs)
        multi_sigs_dict['PCG_FILT'] = pre_filter_pcg(multi_sigs_dict['PCG'], old_fs)

    if 'ECG' in multi_sigs_dict:
        multi_sigs_dict['ECG_FILT'] = pre_filter_ecg(multi_sigs_dict['ECG'], old_fs)

    multi_sigs_dict = {name: resample(sig, old_fs, new_fs) for name, sig in multi_sigs_dict.items()}

    min_len = get_shortest_signal_len(multi_sigs_dict)
    crop_start = round(0.1*new_fs)
    crop_end = min_len - round(0.1*new_fs)
    assert crop_end > crop_start, f'{crop_start=}, {crop_end=}'
    multi_sigs_dict = {name: sig[crop_start:crop_end] for name, sig in multi_sigs_dict.items()}

    save_signals(multi_sigs_dict, patient_name, output_dir, new_fs)

    if seg_idxs is not None:
        seg_idxs, seg_fs = resample_segments(seg_idxs, seg_fs, new_fs)
        assert seg_fs == new_fs, f'{seg_fs=} != {new_fs=}'
        seg_idxs = shift_seg_idxs(seg_idxs, crop_start)
        save_segment_info(os.path.join(output_dir, f'{patient_name}_seg_info.json'),
                          get_seg_info(seg_idxs))


def examine_psigs(multi_sigs_dict, old_fs, new_fs, plot_name='anonymous'):

    min_len = get_shortest_signal_len(multi_sigs_dict)
    crop_start = round(0.1*new_fs)
    crop_end = min_len - round(0.1*new_fs)
    assert crop_end > crop_start, f'{crop_start=}, {crop_end=}'
    multi_sigs_dict = {name: sig[crop_start:crop_end] for name, sig in multi_sigs_dict.items()}

    multi_sigs_dict = {name: standardise_signal(sig) for name, sig in multi_sigs_dict.items()}

    multi_sigs_dict['ECG_FILT'] = pre_filter_ecg(multi_sigs_dict['ECG'], old_fs)

    seg_idxs, seg_fs = get_pcg_segs_idx(multi_sigs_dict['PCG'].copy(), old_fs, new_fs)

    multi_sigs_dict['PCG_FILT'] = pre_filter_pcg(multi_sigs_dict['PCG'], old_fs)

    multi_sigs_dict = {name: resample(sig, old_fs, new_fs) for name, sig in multi_sigs_dict.items()}
    min_len = get_shortest_signal_len(multi_sigs_dict)

    # M is in the middle
    multi_sigs_dict['MSEGS'] = create_segment_waveform(seg_idxs, min_len)

    crop_start = round(0.1*new_fs)
    crop_end = min_len - round(0.1*new_fs)
    assert crop_end > crop_start, f'{crop_start=}, {crop_end=}'
    multi_sigs_dict = {name: sig[crop_start:crop_end] for name, sig in multi_sigs_dict.items()}

    seg_idxs, seg_fs = resample_segments(seg_idxs, seg_fs, new_fs)
    assert seg_fs == new_fs, f'{seg_fs=} != {new_fs=}'
    seg_idxs = shift_seg_idxs(seg_idxs, crop_start)

    multi_sigs_dict = {name: standardise_signal(sig) for name, sig in sorted(multi_sigs_dict.items())}
    multi_sigs_dict.update({f'{name}+CHIRP': standardise_signal(add_chirp(np.zeros_like(sig), new_fs))
                            for name, sig in sorted(multi_sigs_dict.items())})

    orig_multi_sigs_dict = deepcopy(multi_sigs_dict)

    seg_join_idxs = get_seg_join_idx(seg_idxs, len(first_dict_value(multi_sigs_dict)))

    multi_segs_dict = {name: segment_signal(sig, seg_join_idxs) for name, sig in multi_sigs_dict.items()}

    target_num_samples = len(first_dict_value(multi_sigs_dict))
    num_fade_samples = round(0.01*new_fs)

    no_fade_sigs_dict = {
        name: build_signal(segs,
                           num_fade_samples=0,
                           target_num_samples=target_num_samples)
        for name, segs in multi_segs_dict.items()
    }

    unshuffled_sigs_dict = {
        name: build_signal(segs,
                           num_fade_samples=num_fade_samples,
                           target_num_samples=target_num_samples)
        for name, segs in multi_segs_dict.items()
    }

    shuffled_segs_dict = some_shuffle_segments_fn(multi_segs_dict)

    shuffled_sigs_dict = {
        name: build_signal(segments=segs,
                           num_fade_samples=num_fade_samples,
                           target_num_samples=target_num_samples)
        for name, segs in shuffled_segs_dict.items()
    }

    plot_multi_signals_HACKY({
        'origorig': orig_multi_sigs_dict,
        'nofade': no_fade_sigs_dict,
        'unshuffled': unshuffled_sigs_dict,
        'shuffled': shuffled_sigs_dict,
    }, title=plot_name, sr=new_fs)


def get_psig(rec, name):
    return rec.p_signal[:, rec.sig_name.index(name)]


def get_psigs_from_record(path, max_sig_len_s):
    rec = get_record(path, max_sig_len_s=max_sig_len_s)
    return {sig_name: get_psig(rec, sig_name) for sig_name in rec.sig_name}, rec.fs


def readrecord(path):
    rec = get_record(path)
    return {sig_name: get_psig(rec, sig_name) for sig_name in rec.sig_name}, rec.fs


def create_sym_counts(patient_paths):
    data = []
    all_symbols = set()

    for record_name, filepath in tqdm(patient_paths.items(), desc='Counting symbols'):

        ann = wfdb.rdann(filepath, 'atr')
        # header = wfdb.rdheader(filepath)

        symbols = ann.symbol if ann.symbol is not None else []
        counts = Counter(symbols)
        all_symbols.update(symbols)
        row = {
            'record_name': record_name,
            # 'sig_len': header.sig_len,
            # 'sig_name': ','.join(header.sig_name),  # type: ignore
            **counts,
        }

        data.append(row)

    for row in data:
        for sym in all_symbols:
            row.setdefault(sym, 0)

    sym_counts = pd.DataFrame(data)
    return sym_counts


def compute_heart_rate(filename, start, end):
    header = wfdb.rdheader(filename)
    xqrs = wproc.xqrs_detect(wfdb.rdrecord(filename).p_signal[:, 0], fs=header.fs,
                             sampfrom=start, sampto=end, verbose=False)
    mean_hr = wproc.calc_mean_hr(wproc.calc_rr(xqrs, fs=header.fs), fs=header.fs)
    return mean_hr


def sample_record(filename, target_sig_len_s, target_symbol, target_proportion,
                  min_anns, max_tries=50, max_Q_props=0.1):
    header = wfdb.rdheader(filename)
    sig_len = header.sig_len
    old_fs = header.fs
    assert old_fs is not None

    target_sig_len = target_sig_len_s * old_fs

    best_rec = best_ann = best_start = best_end = None
    best_symbol_count = 0

    for _ in range(max_tries):
        start = random.randint(0, sig_len - target_sig_len)
        end = start + target_sig_len

        rec = wfdb.rdrecord(filename, sampfrom=start, sampto=end)
        ann = wfdb.rdann(filename, 'atr', sampfrom=start, sampto=end, shift_samps=True)

        assert ann.symbol is not None

        symbol_count = ann.symbol.count(target_symbol)
        total_symbols = len(ann.symbol)

        if symbol_count > best_symbol_count or best_rec is None:
            best_rec, best_ann, best_symbol_count, best_start, best_end = rec, ann, symbol_count, start, end

        if total_symbols < min_anns:
            continue

        if ann.symbol.count('Q') > max_Q_props * total_symbols:
            continue

        if symbol_count / total_symbols >= target_proportion:
            best_rec, best_ann, best_start, best_end = rec, ann, start, end
            break

    best_hr = compute_heart_rate(filename, best_start, best_end)

    return best_rec, best_ann, best_hr


def normalize_prioritization(prioritization_dict):
    normalized_dict = {}
    for pcg, priorities in prioritization_dict.items():
        total = sum([value for _, value in priorities])
        normalized_priorities = [(annotation, value/total) for annotation, value in priorities]
        normalized_dict[pcg] = normalized_priorities
    return normalized_dict


def compute_annotation_proportions(ann, total_symbols):
    symbols = ["N", "+", "Q", "S", "V"]
    proportions = {}
    for symbol in symbols:
        proportions[symbol] = ann.symbol.count(symbol) / total_symbols
    return proportions


def process_ecg_priorities(df_orig, input_dir, output_dir,
                           target_sig_len_s, num_abnormal_each,
                           num_normal, new_fs=1000):

    columns_to_normalize = ["N", "+", "Q", "S", "V"]
    df_norm = df_orig.copy()
    df_norm[columns_to_normalize] = df_norm[columns_to_normalize].div(df_norm[columns_to_normalize].sum(axis=1), axis=0)
    df_norm = df_norm[df_norm['Q'] <= 0.1]

    prioritization = {
        'AD': [('+', 40), ('V', 30), ('S', 20), ('N', 1), ('Q', 0)],
        'MPC': [('+', 30), ('V', 25), ('S', 25), ('N', 1), ('Q', 0)],
        'MVP': [('V', 40), ('S', 30), ('+', 25), ('N', 1), ('Q', 0)],
        'Benign': [('N', 55), ('+', 25), ('S', 10), ('V', 10), ('Q', 0)],
        'Normal': [('N', 999), ('+', 10), ('V', 5), ('S', 5), ('Q', 0)],
    }

    prioritization = normalize_prioritization(prioritization)

    for pcg, prios in prioritization.items():
        weights = [p[1] for p in prios]
        labels = [p[0] for p in prios]
        assert len(labels) == len(set(labels)), f'{labels=}'

        assert 0.99 < sum(weights) < 1.01, f'{prios=}, {sum(weights)=}'

    selected_records_with_symbol = {pcg: [] for pcg in prioritization}

    num_runs = max(num_normal, num_abnormal_each)

    for i in tqdm(range(num_runs)):  # type: ignore
        for pcg, priorities in prioritization.items():

            if i >= num_abnormal_each and pcg != 'Normal':
                continue

            if i >= num_normal and pcg == 'Normal':
                continue

            annotations, weights = zip(*priorities)
            chosen_annotation = np.random.choice(annotations, p=weights)

            df_sorted = df_norm.sort_values(by=chosen_annotation, ascending=False)
            if chosen_annotation != 'N':
                df_sorted = df_sorted[df_sorted['N'] < df_sorted['N'].mean()]

            for _, row in df_sorted.iterrows():
                record_name = row['record_name']
                if record_name not in [item[0]
                                       for sublist in selected_records_with_symbol.values()
                                       for item in sublist]:
                    selected_records_with_symbol[pcg].append((record_name, chosen_annotation))
                    break

    records_data = []

    for pcg_class, records in tqdm(selected_records_with_symbol.items(), position=0):
        for round_num in tqdm(range(len(records)), position=1, leave=False):

            record_name, target_symbol = records[round_num]
            target_proportion = df_norm[df_norm['record_name'] == record_name][target_symbol].values[0]
            filename = os.path.join(input_dir, record_name)

            rec, ann, hr = sample_record(filename, target_sig_len_s, target_symbol,
                                         target_proportion, min_anns=round(target_sig_len_s*0.4))

            if rec is None or ann is None:
                continue

            assert ann.symbol is not None

            total_symbols = len(ann.symbol)
            proportions = compute_annotation_proportions(ann, total_symbols)

            records_data.append({
                'record_name': record_name,
                'pcg_class': pcg_class,
                'heart_rate': hr,
                'target_symbol': target_symbol,
                **proportions,
            })

            multi_sigs_dict = {'ECG': rec.p_signal.flatten()}
            multi_sigs_dict = {name: standardise_signal(sig) for name, sig in multi_sigs_dict.items()}
            multi_sigs_dict = {name: resample(sig, rec.fs, new_fs) for name, sig in multi_sigs_dict.items()}
            multi_sigs_dict['ECG_FILT'] = pre_filter_ecg(multi_sigs_dict['ECG'], new_fs)
            multi_sigs_dict['ECG_FILT'] = mid_filter_ecg(multi_sigs_dict['ECG_FILT'], new_fs)
            save_signals(multi_sigs_dict, record_name, os.path.join(output_dir, pcg_class), new_fs)

    records_df = pd.DataFrame(records_data)
    records_df.to_csv(os.path.join(output_dir, 'records_info.csv'), index=False, sep=',')
    df_norm.to_csv(os.path.join(output_dir, 'patient_props_info.csv'), index=False, sep=',')
    df_orig.to_csv(os.path.join(output_dir, 'patient_counts_info.csv'), index=False, sep=',')


def process_icentia_dataset(syms_records_df, input_dir, output_dir, matlab_path,
                            num_normal, num_abnormal_each,
                            target_sig_len_s=180, new_fs=1000):

    path, patient = None, None
    os.makedirs(output_dir, exist_ok=True)

    start_matlab(matlab_path)
    try:
        process_ecg_priorities(syms_records_df, input_dir=input_dir, output_dir=output_dir,
                               num_abnormal_each=num_abnormal_each, num_normal=num_normal,
                               target_sig_len_s=target_sig_len_s, new_fs=new_fs)
    except Exception as e:
        print(f'{path=}, {patient=}')
        raise e
    finally:
        stop_matlab()


def process_ecg_pcg_dataset(paths, output_dir, matlab_path, new_fs=1000, max_sig_len_s=180):

    path, patient = None, None
    os.makedirs(output_dir, exist_ok=True)

    start_matlab(matlab_path)
    try:
        for _, path in enumerate(tqdm(paths)):
            patient = os.path.basename(path)
            psigs, old_fs = get_psigs_from_record(path, max_sig_len_s=max_sig_len_s)
            process_psigs(psigs, old_fs, new_fs, output_dir, patient)
    except Exception as e:
        print(f'{path=}, {patient=}')
        raise e
    finally:
        stop_matlab()


def examine_signals(paths, matlab_path, new_fs=1000, max_sig_len_s=30):

    random.shuffle(paths)

    start_matlab(matlab_path)
    try:
        for _, path in enumerate(tqdm(paths)):

            user_wants_another_patient = True
            user_wants_to_continue = True

            while user_wants_another_patient:
                user_wants_another_patient = False
                user_input = input('Continue? ')
                if user_input[0] == 'a':
                    user_wants_another_patient = True
                    patient = user_input
                    psigs, old_fs = get_psigs_from_record(path, max_sig_len_s=max_sig_len_s)
                    if 'PCG' not in psigs:
                        continue
                    examine_psigs(psigs, old_fs, new_fs, patient)
                    continue
                elif user_input[0].upper() != 'Y':
                    user_wants_to_continue = False

            if not user_wants_to_continue:
                print('User chose exit')
                break

            patient = os.path.basename(path)
            psigs, old_fs = get_psigs_from_record(path, max_sig_len_s=max_sig_len_s)
            if 'PCG' not in psigs:
                continue
            examine_psigs(psigs, old_fs, new_fs, patient)

    finally:
        stop_matlab()


def examine_patients(patient_paths, sampler, matlab_path, new_fs=1000, max_sig_len_s=30):

    start_matlab(matlab_path)
    try:
        for _, row in enumerate(tqdm(sampler, total=len(sampler))):
            user_wants_another_patient = True
            user_wants_to_continue = True

            while user_wants_another_patient:
                user_wants_another_patient = False
                user_input = input('Continue? ')
                if user_input[0] == 'a':
                    user_wants_another_patient = True
                    patient = user_input
                    psigs, old_fs = get_psigs_from_record(patient_paths[patient], max_sig_len_s=max_sig_len_s)
                    if 'PCG' not in psigs:
                        continue
                    examine_psigs(psigs, old_fs, new_fs, patient)
                    continue
                elif user_input[0].upper() != 'Y':
                    user_wants_to_continue = False

            if not user_wants_to_continue:
                print('User chose exit')
                break

            print(row)
            patient = row['Record name']
            path = patient_paths[patient]
            psigs, old_fs = get_psigs_from_record(path, max_sig_len_s=max_sig_len_s)
            if 'PCG' not in psigs:
                continue

            chart_name = '_'.join([row[col_name] for col_name in
                                   ['Record name', 'Diagnosis', 'Signal record site']])

            examine_psigs(psigs, old_fs, new_fs, chart_name)

    finally:
        stop_matlab()
