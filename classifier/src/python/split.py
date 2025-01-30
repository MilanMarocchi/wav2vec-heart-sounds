#!/usr/bin/env pipenv-shebang
from util.data_split import (
    create_split_name,
    merge_and_validate_cinc_dataset,
    merge_and_validate_ticking_dataset,
    display_split,
    get_possible_labels,
    assign_split,
    assign_split_extended,
)
from collections import defaultdict

import os
import logging
import pandas as pd
import click

@click.command()
@click.option(
    "-I",
    "--input_dir",
    required=True,
    help="Path where the data is stored"
)
@click.option(
    "-O",
    "--output_path",
    default='',
    help="Path where to store the output file, including the name of the file."
)
@click.option(
    "-D",
    "--dataset",
    required=True,
    help="Name of the dataset, can be extended",
)
@click.option(
    "-S",
    "--data_split",
    default="0.6:0.2:0.2",
    help="The dataset split to use train:valid:test e.g. (0.6:0.2:0.2)"
)
def split(input_dir, output_path, dataset, data_split, **kwargs):
    logging.info(f'{kwargs=}')

    print(f'Creating split for the {dataset} dataset')

    patients_excluded = []

    patient_missing_files = defaultdict(list)

    num_removed = 0
    old_len = -1

    cinc_datasets = ['training-a', 'training-b', 'training-c', 'training-d', 'training-e', 'training-f']

    if dataset in cinc_datasets:

        online_appendix_path = os.path.join(input_dir, 'annotations', 'Online_Appendix_training_set.csv')
        reference_sqi_path = os.path.join(input_dir, 'annotations', 'updated', dataset, 'REFERENCE_withSQI.csv')
        reference_path = os.path.join(input_dir, dataset, 'REFERENCE.csv')

        annotations = merge_and_validate_cinc_dataset(
            online_appendix_path,
            reference_path,
            reference_sqi_path,
            dataset
        )

        old_len = len(annotations)

        for patient in annotations['patient']:
            if dataset == 'training-a':
                required_files = [os.path.join(input_dir, 'training-a', f'{patient}.{extension}')
                                for extension in ['hea', 'dat', 'wav']]
            else:
                required_files = [os.path.join(input_dir, dataset, f'{patient}.{extension}')
                                for extension in ['hea', 'wav']]
            if not all(os.path.exists(file) for file in required_files):
                for file in required_files:
                    if not os.path.exists(file):
                        patient_missing_files[patient].append(file)
                patients_excluded.append(patient)

        logging.info(f'{patient_missing_files=}')

        num_removed = len(patient_missing_files)

    elif dataset == 'ticking-heart':

        reference_path = os.path.join(input_dir, 'REFERENCE.csv')
        annotations = merge_and_validate_ticking_dataset(reference_path)

        old_len = len(annotations)

    else:
        raise Exception("Dataset is not supported")


    print('Annotations before split...')
    print(annotations)

    excluded_patients_df = annotations[annotations['patient'].isin(  # type: ignore
        patient_missing_files.keys())].assign(split='ignore')  # type: ignore

    annotations = annotations[~annotations['patient'].isin(  # type: ignore
        patient_missing_files.keys())]  # type: ignore

    new_len = len(annotations)

    assert old_len - new_len == num_removed, f'{(old_len, new_len, num_removed)=}'

    print('Excluding the following from the split...')
    print(excluded_patients_df)

    train = float(data_split.split(":")[0])
    valid = float(data_split.split(":")[1])
    test = float(data_split.split(":")[2])

    if dataset in cinc_datasets:
        annotations, splits = assign_split_extended(annotations=annotations,  # type: ignore
                                                    ratios={'train': train, 'valid': valid, 'test': test},
                                                    random_state=None)
    else:
        annotations, splits = assign_split(annotations=annotations,  # type: ignore
                                                    ratios={'train': train, 'valid': valid, 'test': test},
                                                    random_state=None)

    annotations = pd.concat([annotations, excluded_patients_df], axis=0).sort_values(by='patient')  # type: ignore

    if output_path == '':
        split_name = create_split_name()
        output_path = os.path.join('splits', f'{split_name}.csv')

    else:
        split_name = os.path.basename(output_path).removesuffix('.csv')

    print('Annotations after split...')
    display_split(annotations, splits)

    print(f'Saving annotations file to {output_path}')

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w') as file:
        file.write(f'# Saved as {split_name}\n')
        annotations.to_csv(file, sep=',', index=False)

if __name__ == "__main__":
    split()
