"""
    segments.py
    Author: Leigh Abbott

    Purpose: Segment the signals into their heart segments
"""

import numpy as np


def resample_segments(seg_idxs: list, old_fs: float, new_fs: float) -> tuple[list, float]:

    adj_seg_idxs = seg_idxs.copy()

    if old_fs != new_fs:

        for i in range(len(seg_idxs)):
            for j in range(len(seg_idxs[i])):
                adj_seg_idxs[i][j] = round(adj_seg_idxs[i][j] * (new_fs / old_fs))

    return adj_seg_idxs, new_fs


def get_shortest_signal_len(signals_dict: dict[str, np.ndarray]) -> int:
    return min(len(sig) for sig in signals_dict.values())


def crop_signal_ends(signal: np.ndarray, crop_start: int, crop_end: int) -> np.ndarray:
    return signal[crop_start:crop_end]


def shift_seg_idxs(seg_idxs: np.ndarray, shift: int) -> list[list]:
    return [[idx - shift for idx in idx_group]
            for idx_group in seg_idxs]


def crop_seg_idxs(seg_idxs: np.ndarray, crop_start: int, crop_end: int) -> list:

    adj_seg_idxs = []

    shift_idxs = shift_seg_idxs(seg_idxs, crop_start)

    for seg_idx_group in shift_idxs:
        if all(0 <= idx < crop_end for idx in seg_idx_group):
            adj_seg_idxs.append(seg_idx_group)

    return adj_seg_idxs


def crop_sigs_segs(signals_dict: dict[str, np.ndarray], crop_start: int, crop_end: int, seg_idxs: list = list()) -> tuple[dict[str, np.ndarray], list]:

    start = crop_start
    shortest_len = min([len(sig) for _, sig in signals_dict.items()])
    end = shortest_len - crop_end

    for sig_name, sig in signals_dict.items():
        old_len = len(signals_dict[sig_name])
        signals_dict[sig_name] = sig[start:end]
        new_len = len(signals_dict[sig_name])
        assert new_len == shortest_len - (crop_start + crop_end), f'{old_len=}, {new_len=}, {crop_end=}'

    adj_seg_idxs = list()
    if len(seg_idxs) > 0:
        adj_seg_idxs = list()

        for seg_idx in seg_idxs:
            adj_seg_idx = [idx - start for idx in seg_idx]
            if all(0 <= idx < end for idx in adj_seg_idx):
                adj_seg_idxs.append(adj_seg_idx)

    return signals_dict, adj_seg_idxs


def get_seg_time_join_idx(seg_idxs: list, last_len: int) -> list:

    num_groups = len(seg_idxs)
    seg_joins = list()

    for i in range(num_groups-1):
        join = seg_idxs[i][0]
        seg_joins.append(join)

    return seg_joins


def get_seg_join_idx(seg_idxs: list, last_len: int) -> list:

    seg_ind = 0
    num_ind = len(seg_idxs[0])

    seg_joins = []
    num_groups = len(seg_idxs)

    for i in range(num_groups-1):
        curr_group_end = seg_idxs[i][seg_ind]
        next_group_start = seg_idxs[(i + 1)][seg_ind]

        if i == num_groups - 1:
            assert next_group_start == 0, f'{next_group_start=}'
            next_group_start += last_len

        if curr_group_end < 0:
            continue

        assert next_group_start > curr_group_end > 0, f'{curr_group_end=}, {next_group_start=}'

        join = curr_group_end
        seg_joins.append(join)

    return seg_joins


# THE LEIGH SPECIAL
def leigh_get_seg_join_idx(seg_idxs: list, last_len: int) -> list:

    seg_ind = 3
    num_ind = len(seg_idxs[0])

    seg_joins = []
    num_groups = len(seg_idxs)

    for i in range(num_groups-1):
        curr_group_end = seg_idxs[i][seg_ind]
        next_group_start = seg_idxs[(i + 1)][(seg_ind + 1) % num_ind]

        if i == num_groups - 1:
            # assert next_group_start == 0, f'{next_group_start=}'
            next_group_start += last_len

        if curr_group_end < 0:
            continue

        assert next_group_start > curr_group_end > 0, f'{curr_group_end=}, {next_group_start=}'

        join = round((curr_group_end + next_group_start) / 2)
        seg_joins.append(join)

    return seg_joins

def from_matform(matform: list) -> list:
    sample_points = []

    latest_state = None

    for s in matform[0:-1]:
        sample_num = s[0][0][0] - 1
        state = s[1][0]

        assert state in ['S1', 'systole', 'S2', 'diastole', '(N', 'N)'], f'{state=}'
        if state == 'S1':
            if latest_state is not None:
                sample_points.append(latest_state)
            latest_state = [sample_num]
        elif state == '(N':
            latest_state = None
        elif latest_state is not None:
            latest_state.append(sample_num)
        else:
            continue

    return sample_points


def get_seg_offset(seg_idxs: list, seg_ind: int) -> int:

    num_cycles = len(seg_idxs) - 1
    num_seg_idx = seg_idxs[num_cycles][seg_ind] - seg_idxs[0][seg_ind]
    seg_offset = round(0.25 * (num_seg_idx / num_cycles))

    return seg_offset


def get_heart_sound_durations(seg_idxs: list) -> tuple[list, list]:

    s1_i, sys_i, s2_i, dia_i = 0, 1, 2, 3

    s1_durs = []
    s2_durs = []

    for idx_group in seg_idxs:

        s1_s, sys_s, s2_s, dia_s = idx_group[s1_i], idx_group[sys_i], idx_group[s2_i], idx_group[dia_i]

        s1_durs.append(sys_s - s1_s)
        s2_durs.append(dia_s - s2_s)

    return s1_durs, s2_durs


def create_segment_waveform(seg_idxs: list, signal_len: int) -> np.ndarray:

    s1_i, sys_i, s2_i, dia_i = 0, 1, 2, 3

    seg_signal = np.zeros(signal_len)

    for idx_group in seg_idxs:

        s1_s, sys_s, s2_s, dia_s = idx_group[s1_i], idx_group[sys_i], idx_group[s2_i], idx_group[dia_i]

        seg_signal[s1_s:sys_s] = s1_i
        seg_signal[sys_s:s2_s] = sys_i
        seg_signal[s2_s:dia_s] = s2_i
        seg_signal[dia_s:] = dia_i

    return seg_signal

