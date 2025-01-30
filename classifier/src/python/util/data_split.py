"""
Put an thing here 
"""

from xkcdpass import xkcd_password 
import datetime
import pandas as pd
from sklearn.model_selection import train_test_split

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

def merge_and_validate_cinc_dataset(online_appendix_path, reference_path, reference_sqi_path, database):
    online_appendix = pd.read_csv(online_appendix_path)

    reference = pd.read_csv(reference_path, header=None, names=["patient", "abnormality"])

    reference_sqi = pd.read_csv(reference_sqi_path, header=None, names=["patient", "abnormality", "SQI"])

    filtered_online_appendix = online_appendix[online_appendix['Database'] == database]

    merged_data = pd.merge(filtered_online_appendix, reference,
                           left_on='Challenge record name', right_on='patient', how='inner')

    merged_data = pd.merge(merged_data, reference_sqi, on='patient', how='inner')

    assert all(merged_data['abnormality_x'] == merged_data['abnormality_y']
               ), "Discrepancy found in 'abnormality' values!"

    final_data = merged_data[['patient', 'Diagnosis', 'abnormality_x', 'SQI']]
    final_data.columns = ['patient', 'diagnosis', 'abnormality', 'SQI']

    final_data['diagnosis'] = [s.strip() for s in final_data['diagnosis']]

    return final_data


def merge_and_validate_ticking_dataset(reference_path):

    reference = pd.read_csv(reference_path, header=None, names=["patient", "abnormality"])

    return reference

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


def display_split(annotations, splits):

    for split in splits:
        print(annotations[annotations['split'] == split])

    for split in splits:
        print(split, len(splits[split]))
        if 'diagnosis' in splits[split]:
            print(splits[split]['diagnosis'].value_counts())
            print(splits[split]['diagnosis'].value_counts(normalize=True))
        print(splits[split])

    annotations.info()


def get_possible_labels(dataset):
    if dataset == 'training-a':
        return CINC_DATASET_POSSIBLE_LABELS
    elif dataset in ['training-a-extended', 'training-a', 'training-b', 'training-c', 'training-d', 'training-e', 'training-f']:
        return CINC_EXTENDED_DATASET_POSSIBLE_LABELS
    raise NotImplementedError(f'{dataset=}')


def assign_split(annotations, ratios={'train': 0.6, 'valid': 0.2, 'test': 0.2}, random_state=None):

    assert (sum(ratios[r] for r in ratios) - 1) < 1E-3, f'{ratios=}'

    valid_ratio_adjusted = ratios['valid'] / (ratios['train'] + ratios['valid'])

    patients_remaining, patients_test = train_test_split(annotations,
                                                         test_size=ratios['test'],
                                                         random_state=random_state)

    patients_train, patients_valid = train_test_split(patients_remaining,
                                                      test_size=valid_ratio_adjusted,
                                                      random_state=random_state)

    splits = {'train': patients_train, 'valid': patients_valid, 'test': patients_test}

    splits = {split: splits[split].assign(split=split) for split in splits}  # type: ignore

    new_annotations = pd.concat([splits[split] for split in splits]).sort_values(by='patient')  # type: ignore

    return new_annotations, splits


def assign_split_extended(annotations, ratios={'train': 0.6, 'valid': 0.2, 'test': 0.2}, random_state=None):

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


def assign_split_noisy_val(annotations, stratify_cols, ratios={'train': 0.6, 'valid': 0.2, 'test': 0.2}, random_state=None):
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
