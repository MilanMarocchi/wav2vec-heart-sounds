"""
    dataloaders.py
    Author: Milan Marocchi

    Purpose: Contains construction of the dataloaders used by various models
"""

from typing import Sequence
import torch

from torch.utils.data import WeightedRandomSampler

def get_sampler(dataset):
    labels = torch.zeros(len(dataset['train']), dtype=torch.long)
    for idx, (_, label) in enumerate(dataset['train']):
        label_stripped = int(label.split('.')[1])
        labels[idx] = label_stripped

    class_counts = torch.tensor([(labels == i).sum() for i in torch.unique(torch.tensor(labels))])
    class_weights = 1. / class_counts.float()
    print(class_weights)
    sample_weights: torch.Tensor = class_weights[labels]

    test_sampler = WeightedRandomSampler(
        weights=sample_weights, # type: ignore
        num_samples=len(sample_weights), 
        replacement=True
    )

    return test_sampler

def create_multi_model_dataloader(dataset, individual_datasets, num_models, model_code):
    """Create the dataloaders for the ensemble type of model."""
    if model_code == "resnet":
        batch_sizes = 32
        workers = 8
    elif model_code == "vgg":
        batch_sizes = 32
        workers = 4
    elif model_code in ["inception", "yamnet", "bilstm", "cnn-bilstm"]:
        batch_sizes = 32
        workers = 6
    elif model_code == "wav2vec":
        batch_sizes = 32
        workers = 1
    elif model_code == "wav2vec-cnn":
        batch_sizes = 64
        workers = 4
    else:
        raise Exception("Invalid model code")

    multi_dataloader = create_dataloaders(dataset, model_code, batch_sizes, workers)

    individual_dataloaders = []
    for i in range(int(num_models)):
        dataloader = create_dataloaders(individual_datasets[i], model_code)
        individual_dataloaders.append(dataloader)

    return multi_dataloader, individual_dataloaders


def create_dataloaders(dataset, model_code, batch_sizes=None, workers=None):
    """Create the dataloaders for a model."""
    if batch_sizes is None and workers is None:
        if model_code == "resnet":
            batch_sizes = 32
            workers = 8
        elif model_code == "vgg":
            batch_sizes = 32
            workers = 8
        elif model_code in ["inception", "yamnet"]:
            batch_sizes = 32
            workers = 8
        elif model_code == "wav2vec":
            batch_sizes = 64
            workers = 1
        elif model_code == "wav2vec-cnn":
            batch_sizes = 64
            workers = 8
        elif model_code in ["bilstm", "cnn-bilstm"]:
            batch_sizes = 64
            workers = 16
        else:
            raise Exception("Invalid model code")

    dataloader = {}

    # Setup weighted sampler for training
    test_sampler = get_sampler(dataset) if 'train' in dataset.keys() else None

    # Setup dataloaders, if a previous dataloader is supplied only update train
    phases = dataset.keys()
    for phase in phases:
        if phase == 'train':
            dloader = torch.utils.data.DataLoader(
                dataset[phase],
                batch_size=batch_sizes,
                num_workers=workers,
                sampler=test_sampler
            )
        else:
            dloader = torch.utils.data.DataLoader(
                dataset[phase],
                batch_size=batch_sizes,
                shuffle=True,
                num_workers=workers
            )

        dataloader[phase] = dloader

    return dataloader
