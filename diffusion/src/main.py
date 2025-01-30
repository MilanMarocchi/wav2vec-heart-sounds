import os
import logging
import shutil

from collections import defaultdict

import torch
import click
import pandas as pd
import matplotlib as mpl

from torch.optim import Adam
from tqdm.auto import tqdm

import utils.reproducible  # type: ignore
import utils.sampling_demonstration
import utils.interactive
import utils.augmentation_demo

import processing.datasets as datasets

import generative.models
import generative.learner

from processing.records import (
    process_ecg_pcg_dataset,
    examine_patients,
    examine_signals,
    create_sym_counts,
    process_icentia_dataset,
    save_signals,
)

from processing.filtering import (
    mid_filter_ecg,
    mid_filter_pcg,
)

from processing.icentia import (
    get_dataloader,
    generate_icentia_data,
    GenerativeIcentiaDataset,
)

try:
    import colored_traceback  # type: ignore
    colored_traceback.add_hook()
except ImportError:
    pass

SUPPORTED_2SIGNAL_DATASETS = [
    # 'training-a',
    'training-a-extended',
    # 'ephnogram',
]


SUPPORTED_1SIGNAL_DATASETS = [
    'icentia11k',
]

SUPPORTED_MODELS = [
    'DiffWave',
    'WaveGrad',
]


@click.group(context_settings={'show_default': True})
@click.option('--LOG_LEVEL', type=click.Choice(['INFO', 'DEBUG']), default='INFO', help='Debug flag level')
@click.pass_context
def cli(ctx, **kwargs):

    for kwarg in kwargs:
        print(kwarg, kwargs[kwarg])
        ctx.obj[kwarg] = kwargs[kwarg]

    logging.basicConfig(level=getattr(logging, kwargs['log_level'], None))


@cli.command()
@click.option('-i', '--input-dir',
              type=str, required=True, help='Location of directory to do preprocessing on')
@click.option('-o', '--output-dir',
              type=str, default='', help='Location to store the preprocessed files')
@click.option('-d', '--dataset',
              type=click.Choice([*SUPPORTED_2SIGNAL_DATASETS, *SUPPORTED_2SIGNAL_DATASETS]),
              required=True, help='Name of the dataset to preprocess')
@click.option('-m', '--matlab-path',
              type=str, default='', help='MATLAB code edited by Milan')
@click.pass_context
def preprocess(ctx, input_dir, output_dir, dataset, matlab_path, **kwargs):

    logging.info(f'{ctx.obj=}')
    logging.info(f'{kwargs=}')

    print(f'{input_dir=}, {output_dir=}, {dataset=}')

    paths = None

    if dataset == 'training-a':

        annotations = pd.read_csv(os.path.join(input_dir, 'REFERENCE.csv'), names=['Record Name', 'Diagnosis'])
        paths = [os.path.join(input_dir, patient)  # type: ignore
                 for patient in annotations['Record Name']]

    elif dataset == 'ephnogram':

        annotations = pd.read_csv(os.path.join(input_dir, 'ECGPCGSpreadsheet.csv')
                                  ).dropna(axis=0, how='all').dropna(axis=1, how='all')
        paths = [os.path.join(input_dir, 'WFDB', patient)  # type: ignore
                 for patient in annotations['Record Name']]

    else:
        raise Exception(f'{dataset=} is not supported')

    print(annotations)

    if output_dir == '':
        output_dir = os.path.join('data', 'processed', dataset)

    process_ecg_pcg_dataset(paths, output_dir, matlab_path)


@cli.command()
@click.option('-i', '--input-dir',
              type=str, required=True, help='Location of data directory')
@click.option('-o', '--output-path',
              type=str, default='', help='If not provided, a randomly generated name is used')
@click.option('-d', '--dataset',
              type=click.Choice(SUPPORTED_2SIGNAL_DATASETS),
              required=True, help='Name of the dataset to preprocess')
@click.pass_context
def make_split(ctx, input_dir, output_path, dataset, **kwargs):

    logging.info(f'{ctx.obj=}')
    logging.info(f'{kwargs=}')

    print(f'Creating split for the {dataset} dataset')

    patients_excluded = []

    patient_missing_files = defaultdict(list)

    num_removed = 0
    old_len = -1

    if dataset == 'training-a-extended':

        annotations = pd.read_csv(
            os.path.join(input_dir, 'annotations', 'Online_Appendix_training_set.csv')
        ).rename({'Challenge record name': 'patient',
                  'Diagnosis': 'diagnosis'}, axis='columns')

        online_appendix_path = os.path.join(input_dir, 'annotations', 'Online_Appendix_training_set.csv')
        reference_sqi_path = os.path.join(input_dir, 'annotations', 'updated', 'training-a', 'REFERENCE_withSQI.csv')
        reference_path = os.path.join(input_dir, 'training-a', 'REFERENCE.csv')

        annotations = datasets.merge_and_validate_training_a(online_appendix_path=online_appendix_path,
                                                             reference_path=reference_path,
                                                             reference_sqi_path=reference_sqi_path)

        old_len = len(annotations)

        for patient in annotations['patient']:
            assert patient.startswith('a'), f'{patient=}'
            diagnosis = annotations[annotations['patient'] == patient]['diagnosis'].item().strip()  # type: ignore
            assert diagnosis in datasets.get_possible_labels(dataset), f'{diagnosis=}'
            required_files = [os.path.join(input_dir, 'training-a', f'{patient}.{extension}')
                              for extension in ['hea', 'dat', 'wav']]
            if not all(os.path.exists(file) for file in required_files):
                for file in required_files:
                    if not os.path.exists(file):
                        patient_missing_files[patient].append(file)
                patients_excluded.append(patient)

        logging.info(f'{patient_missing_files=}')

        num_removed = len(patient_missing_files)

    else:

        raise Exception(f'Dataset {dataset} is not supported')

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

    annotations, splits = datasets.assign_split(annotations=annotations,  # type: ignore
                                                stratify_cols=['diagnosis', 'SQI'],
                                                ratios={'train': 0.6, 'valid': 0.2, 'test': 0.2},
                                                random_state=None)

    annotations = pd.concat([annotations, excluded_patients_df], axis=0).sort_values(by='patient')  # type: ignore

    if output_path == '':
        split_name = datasets.create_split_name()
        output_path = os.path.join('splits', f'{split_name}.csv')

    else:
        split_name = os.path.basename(output_path).removesuffix('.csv')

    print('Annotations after split...')
    datasets.display_split(annotations, splits)

    print(f'Saving annotations file to {output_path}')

    with open(output_path, 'w') as file:
        file.write(f'# Saved as {split_name}\n')
        annotations.to_csv(file, sep=',', index=False)


@cli.command()
@click.option('-t', '--time-interval-m',
              type=int, required=True, help='Save interval in minutes')
@click.option('-w', '--weights',
              type=str, default='', help='Path to weights checkpoint')
@click.option('-m', '--model-dir',
              type=str, default='', help='Where to save weights and summaries')
@click.option('-i', '--input-dir',
              type=str, default='data/processed/training-a', help='CinC dataset location')
@click.option('-s', '--split-annotations',
              type=str, required=True, help='Defines the train/valid/test split')
@click.option('-d', '--dataset',
              type=click.Choice(SUPPORTED_2SIGNAL_DATASETS),
              required=True, help='Name of the dataset to preprocess')
@click.option('-r', '--reference-signal-name',
              type=str, required=True, help='Which signal to use as the reference')
@click.option('-c', '--condition-signal-name',
              type=str, required=True, help='Which signal to use as the condition')
@click.option('-g', '--generative-model',
              type=click.Choice(SUPPORTED_MODELS),
              required=True, help='Which generative model to use')
@click.pass_context
def train_model(ctx, model_dir, input_dir, weights, time_interval_m, split_annotations, dataset,
                reference_signal_name, condition_signal_name, generative_model, **kwargs):

    logging.info(f'{ctx.obj=}')
    logging.info(f'{kwargs=}')

    mpl.use('Agg')

    if model_dir == '':
        split_name = os.path.basename(split_annotations).removesuffix('.csv')
        model_dir = os.path.join('modelout',
                                 f'{condition_signal_name}2{reference_signal_name}',
                                 f'{generative_model}',
                                 f'{split_name}')
        os.makedirs(model_dir, exist_ok=True)

    data = datasets.get_dataloaders(
        generative_model=generative_model,
        annotations_file=split_annotations,
        input_dir=input_dir,
        reference_sig_name=reference_signal_name,
        condition_sig_name=condition_signal_name,
        dataset=dataset,
    )

    USE_CUDA = torch.cuda.is_available()
    DEVICE = torch.device('cuda' if USE_CUDA else 'cpu')
    assert USE_CUDA, f'{USE_CUDA=}, {DEVICE=}'

    params = generative.models.get_params(generative_model)
    model = generative.models.get_generative_model(generative_model)(params).to(DEVICE)

    optimizer = Adam(model.parameters(), lr=data['params'].learning_rate)
    learner = generative.models.get_learner(generative_model)(
        model_dir=model_dir,
        model=model,
        dataloader_train=data['dataloaders']['train'],
        dataloader_valid=data['dataloaders']['valid'],
        optimizer=optimizer,
        params=params,
        dataset=dataset,
        ref_sig_name=reference_signal_name,
        con_sig_name=condition_signal_name,
        transform_ref=data['transform_ref'],
        transform_con=data['transform_con'],
        post_transform_ref=data['post_transform_ref'],
        post_transform_con=data['post_transform_con'],
        final_sr=2000,
    )

    restored_checkpoint = learner.restore_from_checkpoint(weights)

    print(f'Restored from checkpoint at {restored_checkpoint}'
          if restored_checkpoint != ''
          else 'Did not restore from a checkpoint')

    assert restored_checkpoint or weights == '', f'{restored_checkpoint=}, {weights=}'

    learner.train(save_interval_m=time_interval_m)


@cli.command()
@click.option('-i', '--input-dir',
              type=str, default='data/processed/training-a', help='CinC dataset location')
@click.option('-s', '--split-annotations',
              type=str, required=True, help='Defines the train/valid/test split')
@click.option('-r', '--reference-signal',
              type=str, default='pcg', help='Which signal to generate')
@click.option('-c', '--condition-signal',
              type=str, default='ecg', help='Which signal to use as the condition')
@click.option('-d', '--dataset',
              type=click.Choice(SUPPORTED_2SIGNAL_DATASETS),
              required=True, help='Name of the dataset to preprocess')
@click.option('-g', '--generative-model',
              type=click.Choice(SUPPORTED_MODELS),
              required=True, help='Which generative model to use')
@click.pass_context
def demonstrate_dataloaders(ctx, input_dir, split_annotations,
                            reference_signal, condition_signal,
                            dataset, generative_model, **kwargs):

    logging.info(f'{ctx.obj=}')
    logging.info(f'{kwargs=}')

    data = datasets.get_dataloaders(
        generative_model=generative_model,
        annotations_file=split_annotations,
        input_dir=input_dir,
        reference_sig_name=reference_signal,
        condition_sig_name=condition_signal,
        dataset=dataset,
    )

    # utils.sampling_demonstration.demonstrate_sampling(data, dataset)

    utils.sampling_demonstration.demonstrate_shuffling(data, dataset)


@cli.command()
@click.option('-i', '--input-dir',
              type=str, default='data/processed/training-a', help='CinC dataset location')
@click.option('-s', '--split-annotations',
              type=str, required=True, help='Defines the train/valid/test split')
@click.option('-r', '--reference-signal',
              type=str, default='pcg', help='Which signal to generate')
@click.option('-c', '--condition-signal',
              type=str, default='ecg', help='Which signal to use as the condition')
@click.option('-d', '--dataset',
              type=click.Choice(SUPPORTED_2SIGNAL_DATASETS),
              required=True, help='Name of the dataset to preprocess')
@click.option('-g', '--generative-model',
              type=click.Choice(SUPPORTED_MODELS),
              required=True, help='Which generative model to use')
@click.pass_context
def demonstrate_augmentation(ctx, input_dir, split_annotations,
                             reference_signal, condition_signal,
                             dataset, generative_model, **kwargs):

    logging.info(f'{ctx.obj=}')
    logging.info(f'{kwargs=}')

    data = datasets.get_dataloaders(
        generative_model=generative_model,
        annotations_file=split_annotations,
        input_dir=input_dir,
        reference_sig_name=reference_signal,
        condition_sig_name=condition_signal,
        dataset=dataset,
    )

    # utils.sampling_demonstration.demonstrate_sampling(data, dataset)

    utils.sampling_demonstration.demonstrate_augmentation(data, dataset)


@cli.command()
@click.option('-i', '--input-dir',
              type=str, default='data/processed/training-a', help='CinC dataset location')
@click.option('-s', '--split-annotations',
              type=str, required=True, help='Defines the train/valid/test split')
@click.option('-r', '--reference-signal',
              type=str, default='pcg', help='Which signal to generate')
@click.option('-c', '--condition-signal',
              type=str, default='ecg', help='Which signal to use as the condition')
@click.option('-d', '--dataset',
              type=click.Choice(SUPPORTED_2SIGNAL_DATASETS),
              required=True, help='Name of the dataset to preprocess')
@click.option('-g', '--generative-model',
              type=click.Choice(SUPPORTED_MODELS),
              required=True, help='Which generative model to use')
@click.pass_context
def interactive_augmentation(ctx, input_dir, split_annotations,
                             reference_signal, condition_signal,
                             dataset, generative_model, **kwargs):

    logging.info(f'{ctx.obj=}')
    logging.info(f'{kwargs=}')

    data = datasets.get_dataloaders(
        generative_model=generative_model,
        annotations_file=split_annotations,
        input_dir=input_dir,
        reference_sig_name=reference_signal,
        condition_sig_name=condition_signal,
        dataset=dataset,
    )

    # utils.sampling_demonstration.demonstrate_sampling(data, dataset)

    hpss_app = utils.interactive.HPSSApp(patient_data=utils.interactive.get_patient_data(data))
    hpss_app.mainloop()


@cli.command()
@click.option('-i', '--input-dir',
              type=str, required=True, help='Location of directory containing data')
@click.option('-d', '--dataset',
              type=click.Choice(SUPPORTED_2SIGNAL_DATASETS),
              required=True, help='Name of the dataset to preprocess')
@click.option('-l', '--labels-file',
              type=str, required=True, help='Path to the labels csv')
@click.option('-m', '--matlab-path',
              type=str, default='', help="MATLAB code edited by Milan")
@click.pass_context
def examine_labels(ctx, input_dir, dataset, labels_file, matlab_path, **kwargs):

    logging.info(f'{ctx.obj=}')
    logging.info(f'{kwargs=}')

    print(f'{input_dir=}, {dataset=}, {labels_file=}')

    paths = None

    if dataset == 'training-a':

        annotations = pd.read_csv(os.path.join(input_dir, 'REFERENCE.csv'), names=['Record Name', 'Diagnosis'])
        paths = {patient: os.path.join(input_dir, patient)  # type: ignore
                 for patient in annotations['Record Name']}

        labels_file = pd.read_csv(labels_file)

    else:
        raise Exception(f'{dataset=} is not supported')

    examine_patients(
        patient_paths=paths,
        sampler=utils.sampling_demonstration.RandomDataPointSampler(labels_file),
        matlab_path=matlab_path,
    )


@cli.command()
@click.option('-i', '--input-dir',
              type=str, required=True, help='Location of directory to do preprocessing on')
@click.option('-d', '--dataset',
              type=click.Choice(SUPPORTED_2SIGNAL_DATASETS),
              required=True, help='Name of the dataset to preprocess')
@click.option('-m', '--matlab-path',
              type=str, default='', help="MATLAB code edited by Milan")
@click.pass_context
def examine_dataset(ctx, input_dir, dataset, matlab_path, **kwargs):

    logging.info(f'{ctx.obj=}')
    logging.info(f'{kwargs=}')

    print(f'{input_dir=}, {dataset=}')

    paths = None

    if dataset == 'training-a':

        annotations = pd.read_csv(os.path.join(input_dir, 'REFERENCE.csv'), names=['Record Name', 'Diagnosis'])
        paths = [os.path.join(input_dir, patient)  # type: ignore
                 for patient in annotations['Record Name']]

    elif dataset == 'ephnogram':

        annotations = pd.read_csv(os.path.join(input_dir, 'ECGPCGSpreadsheet.csv')
                                  ).dropna(axis=0, how='all').dropna(axis=1, how='all')
        paths = [os.path.join(input_dir, 'WFDB', patient)  # type: ignore
                 for patient in annotations['Record Name']]

    else:
        raise Exception(f'{dataset=} is not supported')

    print(annotations)

    examine_signals(paths, matlab_path)


@cli.command()
@click.option('-i', '--input-dir',
              type=str, required=True, help='Location of directory with sym attributes')
@click.option('-o', '--output-path',
              type=str, default='', help='Location to store the count annotations file')
@click.option('-d', '--dataset',
              type=click.Choice(SUPPORTED_1SIGNAL_DATASETS),
              required=True, help='Name of the dataset to process')
@click.pass_context
def count_patient_syms(ctx, input_dir, output_path, dataset, **kwargs):

    logging.info(f'{ctx.obj=}')
    logging.info(f'{kwargs=}')

    if dataset == 'icentia11k':

        required_extensions = ['hea', 'dat', 'atr']
        base_extension = 'hea'

        valid_paths = {}

        if output_path == '':
            output_path = os.path.join(input_dir, 'patient_sym_counts.csv')

        for record_name in os.listdir(input_dir):

            if not record_name.endswith(base_extension):
                continue

            record_name = record_name.removesuffix(base_extension).removesuffix('.')

            if record_name == 'p10323_s00':
                continue

            record_path = os.path.join(input_dir, record_name)

            assert all([os.path.exists('.'.join([record_path, extension]))
                       for extension in required_extensions]), f'{record_path=}'

            valid_paths[record_name] = record_path

        valid_paths = {k: valid_paths[k] for k in sorted(valid_paths)}

        dataframe = create_sym_counts(valid_paths)
        dataframe.to_csv(output_path, sep=',', index=False)

        # print({k: v for i, (k, v) in enumerate(valid_paths.items()) if i < 5 or i > len(valid_paths) - 5})


@cli.command()
@click.option('-i', '--input-dir',
              type=str, required=True, help='Location of directory to do preprocessing on')
@click.option('-o', '--output-dir',
              type=str, default='', help='Location to store the preprocessed files')
@click.option('-s', '--syms-annotations',
              type=str, required=True, help='Counts for each record')
@click.option('-d', '--dataset',
              type=click.Choice(SUPPORTED_1SIGNAL_DATASETS),
              required=True, help='Name of the dataset to process')
@click.option('-n', '--num-normal',
              type=int, required=True, help='Number of normal samples')
@click.option('-a', '--num-abnormal-each',
              type=int, required=True, help='Number of each abnormal category')
@click.option('-m', '--matlab-path',
              type=str, default='', help='MATLAB code edited by Milan')
@click.pass_context
def preprocess_patient_syms(ctx, input_dir, output_dir, num_normal, num_abnormal_each,
                            dataset,  syms_annotations, matlab_path, **kwargs):

    logging.info(f'{ctx.obj=}')
    logging.info(f'{kwargs=}')

    if dataset == 'icentia11k':

        syms_records_df = pd.read_csv(syms_annotations)

        process_icentia_dataset(syms_records_df=syms_records_df,  matlab_path=matlab_path,
                                input_dir=input_dir, output_dir=output_dir,
                                num_normal=num_normal, num_abnormal_each=num_abnormal_each,
                                target_sig_len_s=15, new_fs=4000)
    else:

        raise NotImplementedError(f'Dataset {dataset} is not supported yet')


@cli.command()
@click.option('-i', '--input-dir',
              type=str, required=True, help='Path to preprocessed icentia files')
@click.option('-o', '--output-dir',
              type=str, default='', help='Location to store the generated files')
@click.option('-w', '--weights',
              type=str, required=True, help='Full path to weights file (normal and abnormal)')
@click.option('-s', '--split-annotations',
              type=str, required=True, help='Name of split annotations')
@click.option('-d', '--device',
              type=str,  default='cpu', help='Which device to use')
@click.option('-g', '--generative-model',
              type=click.Choice(SUPPORTED_MODELS),
              required=True, help='Which generative model to use')
@click.pass_context
def generate_from_icentia(ctx, input_dir, output_dir, weights, device, split_annotations, generative_model, **kwargs):

    logging.info(f'{ctx.obj=}')
    logging.info(f'{kwargs=}')

    if output_dir == '':

        output_dir = os.path.join(
            'data', 'generated', 'icentia',
            f'{generative_model}',
            os.path.basename(split_annotations),
            os.path.basename(weights),
        )

    os.makedirs(output_dir, exist_ok=True)

    params = generative.models.get_params(generative_model)
    model = generative.models.get_generative_model(generative_model)(params)
    transform = generative.models.get_transform(generative_model, 'ecg')

    checkpoint = torch.load(weights)
    model.load_state_dict(checkpoint['model'], strict=True)

    if 'cuda' in device:
        assert torch.cuda.is_available(), f'{device=}, {torch.cuda.is_available()=}'

    model = model.to(device)

    icentia_dataloader = get_dataloader(
        base_dir=input_dir,
        dataset='training-a-extended',
        fs=params['sample_rate'],
        batch_size=16,
        transform=transform,
    )

    generate_icentia_data(
        model=model,
        dataset='training-a-extended',
        icentia_dataloader=icentia_dataloader,
        output_dir=output_dir
    )


@cli.command()
@click.option('-i', '--input-dir',
              type=str, required=True, help='Path to generative icentia files')
@click.option('-o', '--output-dir',
              type=str, default='', help='Location to store the generated files')
@click.option('-s', '--split-annotations',
              type=str, default='', help='Defines the train/valid/test split')
@click.option('-d', '--dataset',
              type=click.Choice(SUPPORTED_2SIGNAL_DATASETS),
              required=True, help='Name of the dataset to preprocess')
@click.pass_context
def flatten_generative_dataset(ctx, input_dir, output_dir, split_annotations, dataset, **kwargs):

    logging.info(f'{ctx.obj=}')
    logging.info(f'{kwargs=}')

    possible_labels = datasets.get_possible_labels(dataset)
    header_suffix = '.hea'
    all_suffix = ['.hea', '_PCG.wav', '.dat', '_ECG.wav']

    normal_category = 'Normal'

    assert normal_category in possible_labels, f'{normal_category=}, {possible_labels=}'

    data = []

    os.makedirs(output_dir, exist_ok=True)

    all_filepaths = []

    for diagnosis in os.listdir(input_dir):
        assert diagnosis in possible_labels, f'{diagnosis=}, {possible_labels=}'
        folder = os.path.join(input_dir, diagnosis)
        for filename in os.listdir(folder):
            assert any(filename.endswith(suffix) for suffix in all_suffix), f'{filename=}'
            if not filename.endswith(header_suffix):
                continue
            else:
                patient = filename.removesuffix(header_suffix)
                filepaths = [os.path.join(folder, f'{patient}{suffix}') for suffix in all_suffix]
                assert all(os.path.exists(filepath) for filepath in filepaths), f'{filepaths=}'
                data.append({
                    'patient': patient,
                    'diagnosis': diagnosis,
                    'abnormality': -1 if diagnosis == normal_category else 1,
                    'SQI': 1,
                    'split': 'train',
                })
                all_filepaths.extend(filepaths)

    dataframe = pd.DataFrame(data).sort_values(by='patient')

    for src in all_filepaths:
        dest = os.path.join(output_dir, os.path.basename(src))
        shutil.copy2(src, dest)

    output_path = os.path.join(output_dir, 'REFERENCE.csv')

    with open(output_path, 'w') as file:
        if split_annotations != '':
            split_name = os.path.basename(split_annotations).removesuffix('.csv')
            file.write(f'# Generated from {split_name}\n')
        dataframe.to_csv(file, sep=',', index=False)

    print(f'Saved to {output_dir}')


@cli.command()
@click.option('-i', '--input-dir',
              type=str, required=True, help='Path to reference dataset')
@click.option('-o', '--output-dir',
              type=str, default='', help='Location to store the generated files')
@click.option('-s', '--split-annotations',
              type=str, default='', help='Defines the train/valid/test split')
@click.option('-d', '--dataset',
              type=click.Choice(SUPPORTED_2SIGNAL_DATASETS),
              required=True, help='Name of the dataset to preprocess')
@click.pass_context
def standardise_reference_dataset(ctx, input_dir, output_dir, split_annotations, dataset, **kwargs):

    logging.info(f'{ctx.obj=}')
    logging.info(f'{kwargs=}')

    possible_labels = datasets.get_possible_labels(dataset)

    normal_category = 'Normal'

    assert normal_category in possible_labels, f'{normal_category=}, {possible_labels=}'

    data = datasets.get_dataloaders(
        generative_model='DiffWave',  # actually doesn't matter here
        annotations_file=split_annotations,
        input_dir=input_dir,
        reference_sig_name='pcg',
        condition_sig_name='ecg',
        dataset=dataset,
    )

    os.makedirs(output_dir, exist_ok=True)

    for dataset in data['datasets'].values():
        for row in tqdm(dataset):
            diagnosis = row['diagnosis']
            assert diagnosis in possible_labels, f'{diagnosis=}, {possible_labels=}'
            patient = row['patient']
            ecg = mid_filter_ecg(row['ecg'], dataset.fs)
            pcg = mid_filter_pcg(row['pcg'], dataset.fs)
            save_signals({'ECG': ecg, 'PCG': pcg},
                         patient,
                         output_dir,
                         dataset.fs)

    shutil.copy2(split_annotations,
                 os.path.join(output_dir, 'REFERENCE.csv'))

    print(f'Saved to {output_dir}')


@cli.command()
@click.option('-i', '--input-dir',
              type=str, required=True, help='Location of directory to do preprocessing on')
@click.pass_context
def examine_augmentation(ctx, input_dir, **kwargs):

    logging.info(f'{ctx.obj=}')
    logging.info(f'{kwargs=}')

    dataframe = pd.read_csv(os.path.join(input_dir, 'REFERENCE.csv'), comment='#')
    print(dataframe.head())

    rows = []

    for _, row in dataframe.iterrows():
        patient = row['patient']
        rows.append({
            'ecg': os.path.join(input_dir, f'{patient}_ECG.wav'),
            'pcg': os.path.join(input_dir, f'{patient}_PCG.wav'),
            'diagnosis': row['diagnosis'],
            'patient': row['patient'],
        })

    utils.augmentation_demo.sig_augment(rows)


@cli.command()
@click.option('-i', '--input-dir',
              type=str, required=True, help='Location of directory to do preprocessing on')
@click.option('-d', '--dataset',
              type=click.Choice(SUPPORTED_2SIGNAL_DATASETS),
              required=True, help='Name of the dataset to preprocess')
@click.option('-g', '--generative-model',
              type=click.Choice(SUPPORTED_MODELS),
              required=True, help='Which generative model to use')
@click.pass_context
def examine_generative_dataset(ctx, input_dir, dataset, generative_model, **kwargs):

    logging.info(f'{ctx.obj=}')
    logging.info(f'{kwargs=}')

    if dataset == 'training-a-extended':
        params = generative.models.get_params(generative_model)
        transform_con = generative.models.get_transform(generative_model, 'ecg')
        transform_gen = generative.models.get_transform(generative_model, 'pcg')
        gen_dataset = GenerativeIcentiaDataset(base_dir=input_dir, dataset=dataset, fs=params['sample_rate'],
                                               transform_con=transform_con, transform_gen=transform_gen)

        utils.sampling_demonstration.demonstrate_generated_icentia(gen_dataset)

    else:
        raise NotImplementedError(f'Dataset {dataset} is not supported')


if __name__ == '__main__':

    print(f'{utils.reproducible.SEED=}')
    cli(obj={})  # type: ignore
