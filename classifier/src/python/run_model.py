#!/usr/bin/env pipenv-shebang
"""
    run_model.py
    Author: Milan Marocchi

    Purpose: To train and test models
"""
from util.schedule import (
    get_schedule
)
from classifier.transforms import (
    create_audio_data_transforms,
    create_data_transforms
)
from classifier.dataloaders import (
    create_dataloaders,
    create_multi_model_dataloader,
)
from classifier.training import (
    FineTunerTrainer
)
from classifier.testing import (
    FineTunerFragmentTester,
    FineTunerPatientTester,
)
from classifier.model_factory import (
    ModelFactory
)
from classifier.datasets import (
    HeartAudioDatabase,
    HeartDataset,
    HeartImageDatabase,
    HeartImageDataset,
    MultiImageHeartDataset,
    SyntheticHeartAudioDataset,
    SyntheticHeartImageDatabase,
    SyntheticMultiHeartImageDatabase,
    TickingHeartImageDataset,
    MultiTickingHeartImageDataset,
    HeartAudioDataset,
    TickingHeartAudioDataset,
)


import click
import torch
import os
import json
import logging
logging.basicConfig(level=logging.ERROR)


def get_dataset(database, is_gen=False, is_rnn=False):

    if not is_rnn:
        if database == 'ticking-heart':
            Dataset = TickingHeartImageDataset
            augmentation = True
        elif is_gen:
            Dataset = SyntheticHeartImageDatabase
            augmentation = False
        else:
            Dataset = HeartImageDataset
            augmentation = True
    else:
        if database == 'ticking-heart':
            Dataset = TickingHeartAudioDataset
            augmentation = True
        elif is_gen:
            # FIXME: Make a synthetic audio database
            Dataset = SyntheticHeartAudioDataset
            augmentation = False
        else:
            Dataset = HeartAudioDataset
            augmentation = True

    return Dataset, augmentation


def parse_schedule(schedule_str):
    """Parse the schedule string."""
    schedule = []
    for item in schedule_str.split(','):
        dataset, epochs, letskip = item.strip().split(':')
        schedule.append((dataset, int(epochs), letskip == '1'))

    return schedule


def parse_model(model_str):
    """
    Parse the models string.
    @returns: composite type, aux type, number of models, if it is an rnn based model
    """
    models = model_str.split(":")
    rnn_models = (
        "bilstm",
        "wav2vec",
        "wav2vec-cnn",
        "cnn-bilstm",
    )

    single_models = (
        "vgg",
        "inception",
        "resnet",
        "wav2vec",
        "wav2vec-cnn",
        "bilstm",
        "cnn-bilstm",
    )

    multi_models = (
        "ensemble",
        "big",
        "big_rnn",
    )

    if len(models) > 1:
        large_type = models[0]
        aux_type = models[2]
        num_models = models[1]

        if large_type not in multi_models:
            raise Exception(f"Invalid Model string: {model_str}")

    elif len(models) == 1 and models[0] in single_models:
        large_type = None
        aux_type = models[0]
        num_models = 1

    else:
        raise Exception(f"Invalid Model string: {model_str}")

    return large_type, aux_type, num_models, aux_type in rnn_models


def parse_dataset(dataset, is_rnn):
    Dataset = MultiDataset = HeartDataset

    if is_rnn:
        if "ticking-heart" in dataset:
            Dataset = MultiDataset = TickingHeartAudioDataset
        else:
            Dataset = MultiDataset = HeartAudioDataset
    else:
        if "ticking-heart" in dataset:
            Dataset = TickingHeartImageDataset
            MultiDataset = MultiTickingHeartImageDataset
        else:
            Dataset = HeartImageDataset
            MultiDataset = MultiImageHeartDataset

    return Dataset, MultiDataset

@click.group()
def cli():
    pass


@click.command()
@click.option('--data_dir', '-D', required=True, help="The dataset to use if training.")
@click.option('--split_path', '-P', required=True, help="The path of the file with the train/test/valid split.")
@click.option('--segment_dir', '-Z', required=True, help="The directory where the segment info is stored.")
@click.option('--model_str', '-M', required=True, help="The type of model to use [resnet/vgg/inception/big:<resnet/vgg/inception>:2/ensemble:<model>:2].")
@click.option('--twod_transform', '-R', required=True, help="The type of spectrogram [stft/mel-stft/wave].")
@click.option('--image_dir', '-I', required=True, help="Path to the image dir to save generated images.")
@click.option('--trained_model_path', '-T', default=None, help="The path to a pre-trained model.")
@click.option('--database', '-B', default='training_a', help='The database being used.')
@click.option('--segmentation', '-A', default='heart', help='The type of segmentation to be used [heart/time].')
@click.option('--four_bands', '-F', is_flag=True, help='To use four bands of pcg.')
@click.option('--fs', '-G', default=16000, help='Frequency to resample to for classification')
@click.option('--sig_len', '-L', required=True, type=float, help='The length of the signal in samples')
def test_model(
        data_dir,
        split_path,
        segment_dir,
        model_str,
        twod_transform,
        image_dir,
        trained_model_path,
        database,
        segmentation,
        four_bands,
        fs,
        sig_len,
        **kwargs
):
    """Tests an already trained model"""
    del kwargs

    large_type, aux_type, num_models, is_rnn = parse_model(model_str)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    data_transforms = create_data_transforms(aux_type == "inception")

    Dataset, MultiDataset = parse_dataset(database, is_rnn)
    data_transforms = create_data_transforms(aux_type == "inception")


    if large_type is None:
        datasets = {'test': Dataset(
            data_dir,
            split_path,
            segment_dir,
            'test',
            image_dir,
            plotter=twod_transform,
            ecg=(database == "training-a"),
            segmentation=segmentation,
            transform=data_transforms['test'],
            four_band=four_bands,
            sig_len=sig_len,
            fs=fs,
        )}
    else:
        datasets = {'test': MultiDataset(
            data_dir,
            split_path,
            segment_dir,
            'test',
            image_dir,
            plotter=twod_transform,
            ecg=(database == "training-a"),
            segmentation=segmentation,
            transform=data_transforms['test'],
            four_band=four_bands,
            num_inputs=num_models,
            fs=fs,
            sig_len=sig_len
        )}

    class_names = next(iter(datasets.values())).classes
    dataloader = create_dataloaders(datasets, aux_type)

    models_factory = ModelFactory(device, class_names, freeze=True)
    model = models_factory.load_model(trained_model_path, aux_type == "inception", aux_type == "wav2vec" and large_type is None, aux_type == "wav2vec" and large_type is not None)

    fragment_tester = FineTunerFragmentTester(model, dataloader)
    patient_tester = FineTunerPatientTester(model, dataloader)

    print()
    fragment_tester.test()
    print()
    patient_tester.test()
    print()


@click.command()
@click.option('--twod_transform', '-R', required=True, help="The type of spectrogram [stft/mel-stft/wave].")
@click.option('--model_str', '-M', required=True, help="The type of model to use [resnet/vgg/inception/big:<resnet/vgg/inception>:2/ensemble:<model>:2].")
@click.option('--image_dir', '-I', required=True, help="Path to the image dir to save generated images.")
@click.option('--output_path', '-O', default='', help="The path to save the model.")
@click.option('--trained_model_path', '-T', default=None, help="The path to a pre-trained model.")
@click.option('--schedule', '-S', default=None, help='Specify the dataloader schedule in the format [dataset1:epochs1,dataset2:epochs2,..]')
@click.option('--optimizer', '-Q', default='sgd', help='The optimizer to use [sgd/adam]')
@click.option('--database', '-D', required=True , help='The database being used.')
@click.option('--segmentation', '-A', default='heart', help='The type of segmentation to be used [heart/time].')
@click.option('--four_bands', '-F', is_flag=True, help='To use four bands of pcg.')
@click.option('--sig_len', '-L', required=True, type=float, help='The length of the signal in samples')
@click.option('--skip_data_valid', '-C', is_flag=True, help='To skip checking if all data is generated for a speedup.')
def train_gen_model(
        twod_transform,
        model_str,
        image_dir,
        output_path,
        trained_model_path,
        schedule,
        optimizer,
        database,
        segmentation,
        four_bands,
        sig_len,
        skip_data_valid,
        **kwargs
):
    """Trains a model using the data dir, split file and segments dir. Also allows for generative data."""
    del kwargs

    model = models_factory = dataloader =  None
    datasets = dict()
    trainer = FineTunerTrainer()
    large_type, aux_type, _ , is_rnn = parse_model(model_str)

    schedule_dict = get_schedule(schedule)
    data_transforms = create_data_transforms(aux_type == "inception")
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    Dataset, augmentation = get_dataset(database, False, is_rnn=is_rnn)

    valid_set = schedule_dict["valid_set"]
    datasets["valid"] = Dataset(
        valid_set["data"],
        valid_set["split"],
        valid_set["segment"],
        "valid",
        image_dir,
        plotter=twod_transform,
        ecg=(database == "training-a"),
        segmentation=segmentation,
        transform=data_transforms["valid"],
        four_band=four_bands,
        sig_len=sig_len,
        augmentation=True,
        skip_data_valid=skip_data_valid
    )

    test_set = schedule_dict["test_set"]
    datasets["test"] = Dataset(
        test_set["data"],
        test_set["split"],
        test_set["segment"],
        "test",
        image_dir,
        plotter=twod_transform,
        ecg=(database == "training-a"),
        segmentation=segmentation,
        transform=data_transforms["valid"],
        four_band=four_bands,
        sig_len=sig_len,
        augmentation=True,
        skip_data_valid=skip_data_valid
    )

    train_datasets = schedule_dict["datasets"]
    schedule = [(train_datasets[x["key"]], x["epochs"], x["letskip"]) for x in schedule_dict["schedule"]]

    for dataset, num_epochs, letskip in schedule:

        Dataset, augmentation = get_dataset(database, dataset["gen_data"], is_rnn=is_rnn)

        # Create the datasets
        datasets["train"] = Dataset(
            dataset["path"],
            dataset["split"],
            dataset["segment"],
            "train",
            image_dir,
            plotter=twod_transform,
            ecg=(database == "training-a"),
            segmentation=segmentation,
            transform=data_transforms["train"],
            four_band=four_bands,
            sig_len=sig_len,
            augmentation=augmentation,
            skip_data_valid=skip_data_valid #type: ignore
        )

        class_names = next(iter(datasets.values())).classes

        if model is None:
            models_factory = ModelFactory(device, class_names, freeze=False, optimizer_type=optimizer)
            if large_type is not None:
                raise Exception("Should be using train_multi_model command.")
            else:
                model = models_factory.create_cnn_model(aux_type)
                dataloader = create_dataloaders(datasets, aux_type)

            if trained_model_path is not None:
                model = models_factory.load_model(trained_model_path, aux_type == "inception", aux_type == "wav2vec")

        trained_model, _ = trainer.train(model, dataloader, num_epochs, letskip=letskip)

        if output_path != "":
            trainer.save_model(trained_model, output_path)

    # Test the final model
    fragment_tester = FineTunerFragmentTester(model, dataloader)
    patient_tester = FineTunerPatientTester(model, dataloader)

    print()
    fragment_tester.test()
    print()
    patient_tester.test()
    print()



@click.command()
@click.option('--twod_transform', '-R', required=True, help="The type of spectrogram [stft/mel-stft/wave].")
@click.option('--model_str', '-M', required=True, help="The type of model to use [resnet/vgg/inception/big:<resnet/vgg/inception>:2/ensemble:<model>:2].")
@click.option('--image_dir', '-I', required=True, help="Path to the image dir to save generated images.")
@click.option('--output_path', '-O', default='', help="The path to save the model.")
@click.option('--trained_model_path', '-T', default=None, help="The path to a pre-trained model.")
@click.option('--schedule', '-S', default=None, help='Specify the dataloader schedule in the format [dataset1:epochs1,dataset2:epochs2,..]')
@click.option('--optimizer', '-Q', default='sgd', help='The optimizer to use [sgd/adam]')
@click.option('--database', '-B', default='training_a', help='The database being used.')
@click.option('--segmentation', '-A', default='heart', help='The type of segmentation to be used [heart/time].')
@click.option('--four_bands', '-F', is_flag=True, help='To use four bands of pcg.')
@click.option('--aux_trained_model_path', '-X', help='The path to a aux pre-trained model.')
@click.option('--sig_len', '-L', required=True, type=float, help='The length of the signal in samples')
@click.option('--skip_data_valid', '-C', is_flag=True, help='To skip checking if all data is generated for a speedup.')
def train_multi_gen_model(
        twod_transform,
        model_str,
        image_dir,
        output_path,
        trained_model_path,
        schedule,
        optimizer,
        database,
        segmentation,
        four_bands,
        aux_trained_model_path,
        sig_len,
        **kwargs
):
    """Trains a model using the data dir, split file and segments dir. Also allows for generative data."""
    del kwargs

    model = individual_dataloader = models_factory = dataloader =  None
    multi_datasets = dict()
    individual_datasets = list()
    trainer = FineTunerTrainer()
    large_type, aux_type, num_models, _ = parse_model(model_str)

    schedule_dict = get_schedule(schedule)
    data_transforms = create_data_transforms(aux_type == "inception")
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    valid_set = schedule_dict["valid_set"]
    multi_datasets["valid"] = MultiImageHeartDataset(
        valid_set["data"],
        valid_set["split"],
        valid_set["segment"],
        "valid",
        image_dir,
        twod_transform,
        ecg=(database == "training-a"),
        segmentation=segmentation,
        transform=data_transforms["valid"],
        four_band=four_bands,
        sig_len=sig_len,
        augmentation=False
    )

    test_set = schedule_dict["test_set"]
    multi_datasets["test"] = MultiImageHeartDataset(
        test_set["data"],
        test_set["split"],
        test_set["segment"],
        "test",
        image_dir,
        twod_transform,
        ecg=(database == "training-a"),
        segmentation=segmentation,
        transform=data_transforms["valid"],
        four_band=four_bands,
        sig_len=sig_len,
        augmentation=False
    )

    individual_datasets = [{ 
        "valid": HeartImageDataset(
            valid_set["data"],
            valid_set["split"],
            valid_set["segment"],
            "valid",
            os.path.join(image_dir, str(i)),
            twod_transform,
            ecg=False,
            segmentation=segmentation,
            transform=data_transforms["valid"],
            four_band=four_bands,
            sig_len=sig_len,
            augmentation=False
        ),
        "test": HeartImageDataset(
            test_set["data"],
            test_set["split"],
            test_set["segment"],
            "test",
            os.path.join(image_dir, str(i)),
            twod_transform,
            ecg=False,
            segmentation=segmentation,
            transform=data_transforms["valid"],
            four_band=four_bands,
            sig_len=sig_len,
            augmentation=False
        )}
    for i in range(int(num_models))]


    train_datasets = schedule_dict["datasets"]
    schedule = [(train_datasets[x["key"]], x["epochs"], x["letskip"]) for x in schedule_dict["schedule"]]

    for dataset, num_epochs, letskip in schedule:

        # FIXME: Replace with new get dataset function
        if database == 'ticking-heart':
            Dataset = TickingHeartImageDataset
            MultiDataset = MultiTickingHeartImageDataset
            augmentation = True
        elif dataset["gen_data"]:
            Dataset = SyntheticHeartImageDatabase
            MultiDataset = SyntheticMultiHeartImageDatabase
            augmentation = False
        else:
            Dataset = HeartImageDataset
            MultiDataset = MultiImageHeartDataset
            augmentation = True

        # Create the datasets
        multi_datasets["train"] = MultiDataset(
            dataset["path"],
            dataset["split"],
            dataset["segment"],
            "train",
            image_dir,
            twod_transform,
            ecg=(database == "training-a"),
            segmentation=segmentation,
            transform=data_transforms["train"],
            four_band=four_bands,
            sig_len=sig_len,
            augmentation=augmentation,
        )

        for i in range(int(num_models)):
            individual_datasets[i]["train"] = Dataset(
                dataset["path"],
                dataset["split"],
                dataset["segment"],
                "train",
                os.path.join(image_dir, str(i)),
                plotter=twod_transform,
                ecg=False,
                segmentation=segmentation,
                transform=data_transforms["train"],
                four_band=four_bands,
                sig_len=sig_len,
                augmentation=False
            )

        class_names = next(iter(multi_datasets.values())).classes

        models_factory = ModelFactory(device, class_names, freeze=True, optimizer_type=optimizer)
        dataloader, individual_dataloader = create_multi_model_dataloader(multi_datasets, individual_datasets, num_models, aux_type)

        if model is None:
            if large_type is not None:
                aux_models = []
                for i in range(int(num_models)):
                    if aux_trained_model_path is not None:
                        model = models_factory.load_model(aux_trained_model_path, aux_type=="inception", aux_type == "wav2vec")
                    else:
                        model = models_factory.create_cnn_model(aux_type)
                    model, _ = trainer.train(model, individual_dataloader[i], num_epochs)
                    aux_models.append(model.model_ft)

                if large_type == "big":
                    model = models_factory.create_cnn_model("big", aux_models, aux_type)
                elif large_type == "ensemble":
                    model = models_factory.create_cnn_model("ensemble", aux_models, aux_type)
            else:
                raise Exception("Should be using train_gen_model command.")

        letskip = True
        trained_model, _ = trainer.train(model, dataloader, num_epochs//2, letskip=letskip)

        if output_path is not None:
            trainer.save_model(trained_model, output_path)

@click.command()
@click.option('--model_str', '-M', required=True, help="The type of model to use [resnet/vgg/inception/big:<resnet/vgg/inception>:2/ensemble:<model>:2].")
@click.option('--audio_dir', '-I', required=True, help="Path to the audio dir to save processed audio.")
@click.option('--output_path', '-O', default='', help="The path to save the model.")
@click.option('--trained_model_path', '-T', default=None, help="The path to a pre-trained model.")
@click.option('--schedule', '-S', default=None, help='The path to the schedule json file')
@click.option('--optimizer', '-Q', default='sgd', help='The optimizer to use [sgd/adam]')
@click.option('--database', '-B', default='training_a', help='The database being used.')
@click.option('--segmentation', '-A', default='heart', help='The type of segmentation to be used [heart/time].')
@click.option('--four_bands', '-F', is_flag=True, help='To use four bands of pcg.')
@click.option('--fs', '-G', default=16000, help='Frequency to resample to for classification')
@click.option('--aux_trained_model_path', '-X', help='The path to a aux pre-trained model.')
@click.option('--sig_len', '-L', required=True, type=float, help='The length of the signal in seconds')
@click.option('--skip_data_valid', '-C', is_flag=True, help='To skip checking if all data is generated for a speedup.')
def train_rnn_gen_model(
        model_str,
        audio_dir,
        output_path,
        trained_model_path,
        schedule,
        optimizer,
        database,
        segmentation,
        four_bands,
        fs,
        sig_len,
        skip_data_valid,
        **kwargs
):
    """Trains a model using the data dir, split file and segments dir."""
    del kwargs

    model = models_factory = dataloader =  None
    datasets = dict()
    trainer = FineTunerTrainer()
    large_type, aux_type, num_models, is_rnn = parse_model(model_str)

    schedule_dict = get_schedule(schedule)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    Dataset, augmentation = get_dataset(database, False, is_rnn=is_rnn)
    transforms = create_audio_data_transforms(fs)

    valid_set = schedule_dict["valid_set"]
    datasets["valid"] = Dataset(
        valid_set["data"],
        valid_set["split"],
        valid_set["segment"],
        "valid",
        audio_dir,
        ecg=(database == "training-a"),
        segmentation=segmentation,
        transform=transforms["valid"],
        four_band=four_bands,
        sig_len=sig_len,
        augmentation=augmentation,
        fs=fs,
        skip_data_valid=skip_data_valid
    )

    test_set = schedule_dict["test_set"]
    datasets["test"] = Dataset(
        test_set["data"],
        test_set["split"],
        test_set["segment"],
        "test",
        audio_dir,
        ecg=(database == "training-a"),
        segmentation=segmentation,
        transform=transforms["test"],
        four_band=four_bands,
        sig_len=sig_len,
        augmentation=augmentation,
        fs=fs,
        skip_data_valid=skip_data_valid
    )

    train_datasets = schedule_dict["datasets"]
    schedule = [(train_datasets[x["key"]], x["epochs"], x["letskip"]) for x in schedule_dict["schedule"]]

    for dataset, num_epochs, letskip in schedule:

        Dataset, augmentation = get_dataset(database, dataset["gen_data"], is_rnn=is_rnn)

        # Create the datasets
        datasets["train"] = Dataset(
            dataset["path"],
            dataset["split"],
            dataset["segment"],
            "train",
            audio_dir,
            ecg=(database == "training-a"),
            segmentation=segmentation,
            transform=transforms["train"],
            four_band=four_bands,
            sig_len=sig_len,
            fs=fs,
            augmentation=augmentation,
            skip_data_valid=skip_data_valid #type: ignore
        )

        class_names = next(iter(datasets.values())).classes
        num_features = 4 if four_bands else 1
        num_features += 1 if database=="training-a" else 0

        if model is None:
            models_factory = ModelFactory(device, class_names, freeze=False, optimizer_type=optimizer)
            if large_type is not None:
                aux_models = []
                for i in range(int(num_models)):
                    # Change the datasets channel to an specific one
                    for key in datasets.keys():
                        datasets[key].channel = i

                    dataloader = create_dataloaders(datasets, aux_type)
                    aux_model = models_factory.create_rnn_model(aux_type, num_features)
                    aux_model, _ = trainer.train(aux_model, dataloader, num_epochs)
                    aux_models.append(aux_model.model_ft)

                if large_type in ("big_rnn", "ensemble"):
                    model = models_factory.create_rnn_model(large_type, int(sig_len*fs), models=aux_models, aux_model_code=aux_type)
            else:
                model = models_factory.create_rnn_model(aux_type, int(sig_len*fs))

            # Change the datasets channel to all again
            for key in datasets.keys():
                datasets[key].channel = -1
            if database == "training-a-pcg":
                for key in datasets.keys():
                    datasets[key].channel = 0
            elif database == "training-a-ecg":
                for key in datasets.keys():
                    datasets[key].channel = 1
            dataloader = create_dataloaders(datasets, aux_type)

            if trained_model_path is not None:
                model = models_factory.load_model(trained_model_path, aux_type == "inception", aux_type == "wav2vec" or aux_type == "wav2vec-cnn")

        trained_model, _ = trainer.train(model, dataloader, num_epochs, letskip=letskip)

        if output_path != "":
            trainer.save_model(trained_model, output_path)

        # Test the final model
        fragment_tester = FineTunerFragmentTester(model, dataloader)
        patient_tester = FineTunerPatientTester(model, dataloader)

        print()
        fragment_tester.test()
        print()
        patient_tester.test()
        print()


@click.command()
@click.option('--data_dir', '-D', required=True, help="The dataset to use if training.")
@click.option('--split_path', '-P', required=True, help="The path of the file with the train/test/valid split.")
@click.option('--segment_dir', '-Z', required=True, help="The directory where the segment info is stored.")
@click.option('--twod_transform', '-R', required=True, help="The type of spectrogram [stft/mel-stft/wave].")
@click.option('--model_str', '-M', required=True, help="The type of model to use [resnet/vgg/inception/big:<resnet/vgg/inception>:2/ensemble:<model>:2].")
@click.option('--image_dir', '-I', required=True, help="Path to the image dir to save generated images.")
@click.option('--num_epochs', '-N', default=20, help="The number of epochs to run for training.")
@click.option('--output_path', '-O', default='', help="The path to save the model.")
@click.option('--trained_model_path', '-T', default=None, help="The path to a pre-trained model.")
@click.option('--schedule', '-S', default=None, help='Specify the dataloader schedule in the format [dataset1:epochs1,dataset2:epochs2,..]')
@click.option('--optimizer', '-Q', default='sgd', help='The optimizer to use [sgd/adam]')
@click.option('--database', '-B', default='training_a', help='The database being used.')
@click.option('--segmentation', '-A', default='heart', help='The type of segmentation to be used [heart/time].')
@click.option('--four_bands', '-F', is_flag=True, help='To use four bands of pcg.')
@click.option('--aux_trained_model_path', '-X', help='The path to a aux pre-trained model.')
@click.option('--sig_len', '-L', required=True, type=float, help='The length of the signal in samples')
def train_multi_model(
        data_dir,
        split_path,
        segment_dir,
        twod_transform,
        model_str,
        image_dir,
        num_epochs,
        output_path,
        trained_model_path,
        schedule,
        optimizer,
        database,
        segmentation,
        four_bands,
        aux_trained_model_path,
        sig_len,
        **kwargs
):
    del kwargs

    model = individual_dataloader = None
    trainer = FineTunerTrainer()
    large_type, aux_type, num_models, _ = parse_model(model_str)
    data_transforms = create_data_transforms(aux_type == "inception")
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    if schedule:
        schedule = parse_schedule(schedule)
        print(f'{schedule=}')
    else:
        schedule = [(data_dir, num_epochs, False)]

    if database == 'ticking-heart':
        Dataset = TickingHeartImageDataset
        MultiDataset = MultiTickingHeartImageDataset
    else:
        Dataset = HeartImageDataset
        MultiDataset = MultiImageHeartDataset

    for _, num_epochs, _ in schedule:

        # Create the datasets
        phases = ('train', 'valid')

        multi_datasets = {p: MultiDataset(
            data_dir,
            split_path,
            segment_dir,
            p,
            twod_transform,
            image_dir,
            ecg=(database == "training-a"),
            segmentation=segmentation,
            transform=data_transforms[p],
            four_band=four_bands,
            augmentation=False,
            num_inputs=num_models,
            sig_len=sig_len
        ) for p in phases}

        individual_datasets = [
                { p : Dataset(
                    data_dir,
                    split_path,
                    segment_dir,
                    p,
                    twod_transform,
                    os.path.join(image_dir, str(i)),
                    ecg=(database == "training-a"),
                    segmentation=segmentation,
                    augmentation=False,
                    transform=data_transforms[p],
                    four_band=four_bands,
                    sig_len=sig_len
                ) for p in phases}
            for i in range(int(num_models))]

        class_names = next(iter(multi_datasets.values())).classes

        models_factory = ModelFactory(device, class_names, freeze=True, optimizer_type=optimizer)
        dataloader, individual_dataloader = create_multi_model_dataloader(multi_datasets, individual_datasets, num_models, aux_type)

        if large_type is not None:
            aux_models = []
            for i in range(int(num_models)):
                if aux_trained_model_path is not None:
                    model = models_factory.load_model(aux_trained_model_path, aux_type=="inception", aux_type == "wav2vec")
                else:
                    model = models_factory.create_cnn_model(aux_type)
                model, _ = trainer.train(model, individual_dataloader[i], num_epochs//2)
                aux_models.append(model.model_ft)

            if large_type == "big":
                model = models_factory.create_cnn_model("big", aux_models, aux_type)
            elif large_type == "ensemble":
                model = models_factory.create_cnn_model("ensemble", aux_models, aux_type)
        else:
            raise Exception("Should be using train_model command.")

        if trained_model_path is not None:
            model = models_factory.load_model(trained_model_path, aux_type == "inception", aux_type == "wav2vec")

        trained_model, _ = trainer.train(model, dataloader, num_epochs)

        if output_path is not None:
            trainer.save_model(trained_model, output_path)

@click.command()
@click.option('--data_dir', '-D', required=True, help="The dataset to use if training.")
@click.option('--split_path', '-P', required=True, help="The path of the file with the train/test/valid split.")
@click.option('--segment_dir', '-Z', required=True, help="The directory where the segment info is stored.")
@click.option('--twod_transform', '-R', required=True, help="The type of spectrogram [stft/mel-stft/wave].")
@click.option('--model_str', '-M', required=True, help="The type of model to use [resnet/vgg/inception/big:<resnet/vgg/inception>:2/ensemble:<model>:2].")
@click.option('--image_dir', '-I', required=True, help="Path to the image dir to save generated images.")
@click.option('--num_epochs', '-N', default=20, help="The number of epochs to run for training.")
@click.option('--output_path', '-O', default='', help="The path to save the model.")
@click.option('--trained_model_path', '-T', default=None, help="The path to a pre-trained model.")
@click.option('--schedule', '-S', default=None, help='Specify the dataloader schedule in the format [dataset1:epochs1,dataset2:epochs2,..]')
@click.option('--optimizer', '-Q', default='sgd', help='The optimizer to use [sgd/adam]')
@click.option('--database', '-B', default='training_a', help='The database being used.')
@click.option('--segmentation', '-A', default='heart', help='The type of segmentation to be used [heart/time].')
@click.option('--four_bands', '-F', is_flag=True, help='To use four bands of pcg.')
@click.option('--sig_len', '-L', required=True, type=float, help='The length of the signal in samples')
def train_model(
        data_dir,
        split_path,
        segment_dir,
        twod_transform,
        model_str,
        image_dir,
        num_epochs,
        output_path,
        trained_model_path,
        schedule,
        optimizer,
        database,
        segmentation,
        four_bands,
        sig_len,
        **kwargs
):
    """Trains a model using the data dir, split file and segments dir."""
    model = None
    trainer = FineTunerTrainer()
    large_type, aux_type, _ , _ = parse_model(model_str)

    if schedule:
        schedule = parse_schedule(schedule)
        print(f'{schedule=}')
    else:
        schedule = [(data_dir, num_epochs, False)]

    for dataset, num_epochs, letskip in schedule:

        data_transforms = create_data_transforms(aux_type == "inception")
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        if database == 'ticking-heart':
            Dataset = TickingHeartImageDataset
        else:
            Dataset = HeartImageDataset

        # Create the datasets
        phases = ('train', 'valid')
        datasets = {p: Dataset(
            data_dir,
            split_path,
            segment_dir,
            p,
            twod_transform,
            image_dir,
            ecg=(database == "training-a"),
            segmentation=segmentation,
            transform=data_transforms[p],
            four_band=four_bands,
            sig_len=sig_len
        ) for p in phases}

        class_names = next(iter(datasets.values())).classes

        models_factory = ModelFactory(device, class_names, freeze=False, optimizer_type=optimizer)
        if large_type is not None:
            raise Exception("Should be using train_multi_model command.")
        else:
            model = models_factory.create_cnn_model(aux_type)
            dataloader = create_dataloaders(datasets, aux_type)

        trainer = FineTunerTrainer()

        if trained_model_path is not None:
            model = models_factory.load_model(trained_model_path, aux_type == "inception", aux_type == "wav2vec")

        trained_model, _ = trainer.train(model, dataloader, num_epochs)

        if output_path is not None:
            trainer.save_model(trained_model, output_path)


@click.command()
@click.option('--data_dir', '-D', required=True, help="The dataset to use if training.")
@click.option('--split_path', '-P', required=True, help="The path of the file with the train/test/valid split.")
@click.option('--segment_dir', '-Z', required=True, help="The directory where the segment info is stored.")
@click.option('--model_str', '-M', required=True, help="The type of model to use [wav2vec/bilstm//big:<model>:2/ensemble:<model>:2].")
@click.option('--audio_dir', '-I', required=True, help="Path to the dir to save processed audio.")
@click.option('--num_epochs', '-N', default=20, help="The number of epochs to run for training.")
@click.option('--output_path', '-O', default='', help="The path to save the model.")
@click.option('--trained_model_path', '-T', default=None, help="The path to a pre-trained model.")
@click.option('--schedule', '-S', default=None, help='Specify the dataloader schedule in the format [dataset1:epochs1,dataset2:epochs2,..]')
@click.option('--optimizer', '-Q', default='sgd', help='The optimizer to use [sgd/adam]')
@click.option('--database', '-B', default='training_a', help='The database being used.')
@click.option('--segmentation', '-A', default='heart', help='The type of segmentation to be used [heart/time].')
@click.option('--four_bands', '-F', is_flag=True, help='To use four bands of pcg.')
@click.option('--fs', '-G', default=16000, help='Frequency to resample to for classification')
@click.option('--sig_len', '-L', required=True, type=float, help='The length of the signal in samples')
@click.option('--skip_data_valid', '-C', is_flag=True, help='To skip checking if all data is generated for a speedup.')
def train_rnn_model(
        data_dir,
        split_path,
        segment_dir,
        model_str,
        num_epochs,
        audio_dir,
        output_path,
        trained_model_path,
        schedule,
        optimizer,
        database,
        segmentation,
        four_bands,
        fs,
        sig_len,
        skip_data_valid,
        **kwargs
):
    """Trains a model using the data dir, split file and segments dir."""
    del kwargs

    model = None
    trainer = FineTunerTrainer()
    large_type, aux_type, num_models, _ = parse_model(model_str)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    if database == "ticking-heart":
        Dataset = TickingHeartAudioDataset 
    else: 
        Dataset = HeartAudioDataset

    # Create the datasets
    phases = ('train', 'valid', 'test')
    datasets = {p: Dataset(
        data_dir,
        split_path,
        segment_dir,
        p,
        audio_dir,
        ecg=(database == "training-a"),
        segmentation=segmentation,
        augmentation=True,
        four_band=four_bands,
        fs=fs,
        skip_data_valid=skip_data_valid,
        sig_len=sig_len,
    ) for p in phases}

    class_names = next(iter(datasets.values())).classes
    num_features = 4 if four_bands else 1
    num_features += 1 if database=="training-a" else 0

    models_factory = ModelFactory(device, class_names, freeze=False, optimizer_type=optimizer)
    if large_type is not None:
        aux_models = []
        for i in range(int(num_models)):
            # Change the datasets channel to an specific one
            for key in datasets.keys():
                datasets[key].channel = i

            dataloader = create_dataloaders(datasets, aux_type)
            aux_model = models_factory.create_rnn_model(aux_type, num_features)
            aux_model, _ = trainer.train(aux_model, dataloader, num_epochs // 2)
            aux_models.append(aux_model.model_ft)

        if large_type in ("big_rnn", "ensemble"):
            model = models_factory.create_rnn_model(large_type, sig_len, models=aux_models, aux_model_code=aux_type)
    else:
        model = models_factory.create_rnn_model(aux_type, int(sig_len*fs))

    # Change the datasets channel to all again
    for key in datasets.keys():
        datasets[key].channel = -1
    dataloader = create_dataloaders(datasets, aux_type)

    if trained_model_path is not None:
        model = models_factory.load_model(trained_model_path, aux_type == "inception", aux_type == "wav2vec")

    trained_model, _ = trainer.train(model, dataloader, num_epochs)

    if output_path is not None:
        trainer.save_model(trained_model, output_path)

    fragment_tester = FineTunerFragmentTester(model, dataloader)
    patient_tester = FineTunerPatientTester(model, dataloader)

    print()
    fragment_tester.test()
    print()
    patient_tester.test()
    print()


@click.command()
@click.option('--data_dir', '-D', required=True, help="The dataset to use if training.")
@click.option('--split_path', '-P', required=True, help="The path of the file with the train/test/valid split.")
@click.option('--segment_dir', '-Z', required=True, help="The directory where the segment info is stored.")
@click.option('--twod_transform', '-R', required=True, help="The type of spectrogram [stft/mel-stft/wave].")
@click.option('--model_str', '-M', required=True, help="The type of model to use [wav2vec].")
@click.option('--image_dir', '-I', required=True, help="Path to the image dir to save generated images.")
@click.option('--num_epochs', '-N', default=20, help="The number of epochs to run for training.")
@click.option('--output_path', '-O', default='', help="The path to save the model.")
@click.option('--trained_model_path', '-T', default=None, help="The path to a pre-trained model.")
@click.option('--schedule', '-S', default=None, help='Specify the dataloader schedule in the format [dataset1:epochs1,dataset2:epochs2,..]')
@click.option('--optimizer', '-Q', default='sgd', help='The optimizer to use [sgd/adam]')
@click.option('--database', '-B', default='training-a:training-b:training-c:training-d:training-e:training-f', help='The database being used.')
@click.option('--segmentation', '-A', default='heart', help='The type of segmentation to be used [heart/time].')
@click.option('--four_bands', '-F', is_flag=True, help='To use four bands of pcg.')
@click.option('--fs', '-G', default=16000, help='Frequency to resample to for classification')
@click.option('--skip_data_valid', '-C', is_flag=True, help='To skip checking if all data is generated for a speedup.')
@click.option('--sig_len', '-L', required=True, type=float, help='The length of the signal in samples')
def train_cinc_model(
        data_dir,
        split_path,
        segment_dir,
        twod_transform,
        model_str,
        image_dir,
        num_epochs,
        output_path,
        trained_model_path,
        schedule,
        optimizer,
        database,
        segmentation,
        four_bands,
        fs,
        skip_data_valid,
        sig_len,
        **kwargs
):
    """Trains a model using the data dir, split file and segments dir."""
    del kwargs

    print(output_path)

    model = individual_dataloader = None
    trainer = FineTunerTrainer()
    large_type, aux_type, num_models, is_rnn = parse_model(model_str)

    if schedule:
        schedule = parse_schedule(schedule)
        print(f'{schedule=}')
    else:
        schedule = [(data_dir, num_epochs, False)]

    sched = ''
    for dataset, num_epochs, letskip in schedule:
        sched += f'{dataset[0]}{num_epochs}{letskip}'

    for dataset, num_epochs, letskip in schedule:

        data_transforms = create_data_transforms(aux_type == "inception")
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        Dataset = HeartImageDatabase if not is_rnn else HeartAudioDatabase

        # Create the datasets
        phases = ('train', 'valid', 'test')
        datasets = {p: Dataset(
            data_dir,
            split_path,
            segment_dir,
            p,
            image_dir,
            plotter=twod_transform,
            databases=database,
            segmentation=segmentation,
            transform=data_transforms[p],
            four_band=four_bands,
            augmentation=True,
            sig_len=sig_len,
            fs=fs,
            skip_data_valid=skip_data_valid
        ) for p in phases}

        class_names = next(iter(datasets.values())).classes
        models_factory = ModelFactory(device, class_names, freeze=False, optimizer_type=optimizer)

        if large_type is not None:
            raise ValueError("Should be using train_multi_model command.")

        if is_rnn:
            model = models_factory.create_rnn_model(aux_type, int(sig_len*fs))
        else:
            model = models_factory.create_cnn_model(aux_type)
        dataloader = create_dataloaders(datasets, aux_type)

        trainer = FineTunerTrainer()

        if trained_model_path is not None:
            model = models_factory.load_model(trained_model_path, aux_type == "inception", aux_type == "wav2vec")

        trained_model, _ = trainer.train(model, dataloader, num_epochs)

        if output_path is not None:
            trainer.save_model(trained_model, output_path)

        fragment_tester = FineTunerFragmentTester(model, dataloader)
        patient_tester = FineTunerPatientTester(model, dataloader)

        print()
        fragment_tester.test()
        print()
        patient_tester.test()
        print()


cli.add_command(train_multi_model, 'train_multi_model')
cli.add_command(train_model, 'train_model')
cli.add_command(train_gen_model, 'train_gen_model')
cli.add_command(train_multi_gen_model, 'train_multi_gen_model')
cli.add_command(train_rnn_gen_model, 'train_rnn_gen_model')
cli.add_command(train_rnn_model, 'train_rnn_model')
cli.add_command(train_cinc_model, 'train_cinc_model')
cli.add_command(test_model, 'test_model')


if __name__ == "__main__":
    cli()
