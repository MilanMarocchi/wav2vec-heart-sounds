"""
    model_factory.py
    Author: Milan Marocchi

    Purpose: To create models
"""

from classifier.models import (
    CNN2DBiLSTM,
    TorchModel,
    EnsembleModel,
    BigModel,
    BigRNNModel,
    Wav2Vec,
    BiLSTM,
    CNNBiLSTM,
    Wav2VecCNN,
)
from torchvision import models
from torch.optim import lr_scheduler
import os
import torch
import torch.nn as nn
import torch.optim as optim


HERE = os.path.abspath(os.getcwd())


def get_optimizer(params, optimizer_type):
    """
    Returns the required optimiser.
    """
    if optimizer_type == 'sgd':
        return optim.SGD(params, lr=0.001, momentum=0.9, weight_decay=1e-5)
    elif optimizer_type == 'adam':
        return optim.Adam(params, lr=0.001, betas=(0.9, 0.999), eps=1e-8, weight_decay=1e-5)
    elif optimizer_type == 'rmsprop':
        return optim.RMSprop(params, lr=1e-4, weight_decay=6.1148e-5, momentum=0.17562)
    else:
        raise NotImplementedError("Invalid Optimiser")


def get_CNN2DBiLSTM(device, class_names, signal_len, optimizer_type='sgd'):
    """Create a 2d Image CNN BiLSTM model for traininig"""
    num_classes = len(class_names)

    cnn_ft = models.resnet50(weights="IMAGENET1K_V1")
    fts = list(cnn_ft.children())[:-1]
    cnn_ft = nn.Sequential(*fts)

    model_ft = CNN2DBiLSTM(num_classes, device, cnn_ft, signal_len)
    model_ft = model_ft.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer_ft = get_optimizer(model_ft.parameters(), optimizer_type)
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=7, gamma=0.1)

    return TorchModel(model_ft, criterion, optimizer_ft, exp_lr_scheduler, False)


def get_CNNBILSTM(device, class_names, signal_len, optimizer_type='adam'):
    """Create a CNN BiLSTM model for training"""
    num_classes = len(class_names)

    model_ft = CNNBiLSTM(num_classes, device, signal_len=signal_len)
    model_ft = model_ft.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer_ft = get_optimizer(model_ft.parameters(), optimizer_type)
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=7, gamma=0.1) if optimizer_type == 'adam' else None

    return TorchModel(model_ft, criterion, optimizer_ft, exp_lr_scheduler, False)

def get_BILSTM(device, class_names, signal_len, optimizer_type='adam'):
    """Create a BiLSTM model for training."""
    num_classes = len(class_names)

    model_ft = BiLSTM(len(class_names), device, signal_len)
    model_ft = model_ft.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer_ft = get_optimizer(model_ft.parameters(), optimizer_type)
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=7, gamma=0.1) if optimizer_type == 'adam' else None

    return TorchModel(model_ft, criterion, optimizer_ft, exp_lr_scheduler, False)


def get_WAV2VEC(device, class_names, signal_len, optimizer_type='adam'):
    """Create a WAV2VEC Pretrained model."""
    num_classes = len(class_names)

    model_ft = Wav2Vec(len(class_names))
    model_ft = model_ft.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer_ft = get_optimizer(model_ft.parameters(), optimizer_type)
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=7, gamma=0.1) if optimizer_type == 'sgd' else None

    return TorchModel(model_ft, criterion, optimizer_ft, exp_lr_scheduler, False, is_wav2vec=True)


def get_WAV2VECCNN(device, class_names, signal_len, optimizer_type='rmsprop'):
    """Create a WAV2VEC Pretrained model."""
    num_classes = len(class_names)

    model_ft = Wav2VecCNN(len(class_names), signal_len)
    model_ft = model_ft.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer_ft = get_optimizer(model_ft.parameters(), optimizer_type)
    if optimizer_type == 'sgd':
        exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=7, gamma=0.1) 
    else:
        exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=4, gamma=0.02444)

    return TorchModel(model_ft, criterion, optimizer_ft, exp_lr_scheduler, False, is_wav2vec=True)


def get_YAMNET(optimizer_type='sgd'):
    """
    Create a YAMNET model
    """
    raise NotImplementedError("Has not yet been implemented.")


def get_Inception(device, class_names, optimizer_type='sgd'):
    """
    Returns the inceptionv3 model
    """
    num_classes = len(class_names)

    model_ft = models.inception_v3(weights="Inception_V3_Weights.IMAGENET1K_V1")
    # Handle the auxilary net
    num_ftrs = model_ft.AuxLogits.fc.in_features # type: ignore
    model_ft.AuxLogits.fc = nn.Linear(num_ftrs, num_classes) # type: ignore
    # Handle the primary net
    num_ftrs = model_ft.fc.in_features
    model_ft.fc = nn.Linear(num_ftrs, num_classes)

    model_ft = model_ft.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer_ft = get_optimizer(model_ft.parameters(), optimizer_type)

    exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=7, gamma=0.1) if optimizer_type == 'sgd' else None

    return TorchModel(model_ft, criterion, optimizer_ft, exp_lr_scheduler, True)


def get_VGG(device, class_names, optimizer_type='sgd'):
    """
    Returns the VGG model.
    """
    model_ft = models.vgg19(weights="VGG19_Weights.IMAGENET1K_V1")

    number_features = model_ft.classifier[6].in_features
    features = list(model_ft.classifier.children())[:-1]  # Removes the last layer
    features.extend([torch.nn.Linear(number_features, len(class_names))])
    model_ft.classifier = nn.Sequential(*features)

    model_ft = model_ft.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer_ft = get_optimizer(model_ft.parameters(), optimizer_type)

    exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=7, gamma=0.1) if optimizer_type == 'sgd' else None

    return TorchModel(model_ft, criterion, optimizer_ft, exp_lr_scheduler, False)


def get_RESNET(device, optimizer_type='sgd'):
    """
    Returns the RESNET model for classifying heart sounds.
    """
    model_ft = models.resnet50(weights="IMAGENET1K_V1")
    num_ftrs = model_ft.fc.in_features

    model_ft.fc = nn.Linear(num_ftrs, 2)

    model_ft = model_ft.to(device)

    criterion = nn.CrossEntropyLoss()

    optimizer_ft = get_optimizer(model_ft.parameters(), optimizer_type)

    exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=7, gamma=0.1) if optimizer_type == 'sgd' else None

    return TorchModel(model_ft, criterion, optimizer_ft, exp_lr_scheduler, False)


def get_ensemble(device, ensemble_models, num_classes, optimizer_type='sgd'):
    """
    Returns the RESNET ensemble model
    """
    ensemble_model = EnsembleModel(ensemble_models, num_classes)

    for param in ensemble_model.parameters():
        param.requires_grad = False

    for param in ensemble_model.classifier.parameters():
        param.requires_grad = True

    ensemble_ft = ensemble_model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer_ft = get_optimizer(ensemble_ft.parameters(), optimizer_type)
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=7, gamma=0.1) if optimizer_type == 'sgd' else None

    return TorchModel(ensemble_ft, criterion, optimizer_ft, exp_lr_scheduler, False)


def get_big(device, ensemble_models,  num_classes, model_code, optimizer_type='sgd'):
    """
    Returns the RESNET ensemble model
    """
    big_model = BigModel(ensemble_models, num_classes, model_code)

    for param in big_model.parameters():
        param.requires_grad = False 

    for param in big_model.classifier.parameters():
        param.requires_grad = True

    big_ft = big_model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer_ft = get_optimizer(big_ft.parameters(), optimizer_type)
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=7, gamma=0.1) if optimizer_type == 'sgd' else None

    return TorchModel(big_ft, criterion, optimizer_ft, exp_lr_scheduler, False)


def get_big_rnn(device, ensemble_models,  num_classes, model_code, optimizer_type='sgd'):
    """
    Returns the RESNET ensemble model
    """
    big_model = BigRNNModel(ensemble_models, num_classes, model_code)

    for param in big_model.parameters():
        param.requires_grad = True

    for param in big_model.classifier.parameters():
        param.requires_grad = True

    big_ft = big_model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer_ft = get_optimizer(big_ft.parameters(), optimizer_type)
    if optimizer_type == 'sgd':
        exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=7, gamma=0.1) 
    else:
        exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=4, gamma=0.02444)

    return TorchModel(big_ft, criterion, optimizer_ft, exp_lr_scheduler, False, 'wav2vec' in model_code)


class ModelFactory():
    """
    Model factory to make it easier to change between models and stuff
    """

    def __init__(self, device, class_names, freeze=False, optimizer_type='sgd'):
        self.device = device
        self.class_names = class_names
        self.freeze = freeze
        self.optimizer_type = optimizer_type

    def create_cnn_model(self, model_code, models=None, aux_model_code=None):
        """
        Creates the model specified by the model_code
        """
        if model_code == "resnet":
            return get_RESNET(self.device, optimizer_type=self.optimizer_type)
        elif model_code == "vgg":
            return get_VGG(self.device, self.class_names, optimizer_type=self.optimizer_type)
        elif model_code == "inception":
            return get_Inception(self.device, self.class_names, optimizer_type=self.optimizer_type)
        elif model_code == "yamnet":
            return get_YAMNET(optimizer_type=self.optimizer_type)
        elif model_code == "ensemble":
            if models is None:
                raise Exception("Must provide models to ensemble")

            return get_ensemble(self.device, models, len(self.class_names), optimizer_type=self.optimizer_type)
        elif model_code == "big":
            if models is None:
                raise Exception("Must provide models to ensemble")

            return get_big(self.device, models, len(self.class_names), aux_model_code, optimizer_type=self.optimizer_type)
        else:
            raise Exception(f"Invalid CNN model: {model_code=}")

    def create_rnn_model(self, model_code, signal_len, models=None, aux_model_code=None):
        """Creates the transformer model"""
        if model_code == "wav2vec":
            return get_WAV2VEC(self.device, self.class_names, signal_len, optimizer_type=self.optimizer_type)
        if model_code == "wav2vec-cnn":
            return get_WAV2VECCNN(self.device, self.class_names, signal_len, optimizer_type=self.optimizer_type)
        elif model_code == "bilstm":
            return get_BILSTM(self.device, self.class_names, signal_len, optimizer_type=self.optimizer_type)
        elif model_code == "cnn-bilstm":
            return get_CNNBILSTM(self.device, self.class_names, signal_len, optimizer_type=self.optimizer_type)
        elif model_code == "cnn-2d-bilstm":
            return get_CNN2DBiLSTM(self.device, self.class_names, signal_len, optimizer_type=self.optimizer_type)
        elif model_code == "ensemble":
            if models is None:
                raise Exception("Must provide models to ensemble")

            return get_ensemble(self.device, models, len(self.class_names), optimizer_type=self.optimizer_type)
        elif model_code == "big_rnn":
            if models is None:
                raise Exception("Must provide models to ensemble")

            return get_big_rnn(self.device, models, len(self.class_names), aux_model_code, optimizer_type=self.optimizer_type)
        else:
            raise Exception(f"Invalid RNN model: {model_code=}")


    def load_model(self, path, fs=4125, sig_len=4, is_inception=False, is_wav2vec=False, is_large_wav2vec=False, is_wav2veccnn=False, is_large_wav2veccnn=False):
        """
        Loads the model from a file to be used.
        """
        os.makedirs(os.path.dirname(path), exist_ok=True)

        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        if is_wav2vec:
            model_ft = Wav2Vec(2)
            model_ft.load_state_dict(torch.load(path, map_location=torch.device(device)))
            model_ft.to(device)
        elif is_large_wav2vec:
            model_ft = BigRNNModel([Wav2Vec(2), Wav2Vec(2)], 2, 'wav2vec')
            model_ft.load_state_dict(torch.load(path, map_location=torch.device(device)), strict=False)
            model_ft.to(device)
            [model.to(device) for model in model_ft.models]
        elif is_wav2veccnn:
            model_ft = Wav2VecCNN(2, fs*sig_len)
            model_ft.load_state_dict(torch.load(path, map_location=torch.device(device)))
            model_ft.to(device)
        elif is_large_wav2veccnn:
            model_ft = BigRNNModel([Wav2Vec(2), Wav2Vec(2)], 2, 'wav2vec')
            model_ft.load_state_dict(torch.load(path, map_location=torch.device(device)), strict=False)
            model_ft.to(device)
            [model.to(device) for model in model_ft.models]
        else:
            model_ft = torch.load(path)
            model_ft.to(device)


        optimizer_ft = get_optimizer(model_ft.parameters(), self.optimizer_type)
        exp_lr_scheduler = lr_scheduler.StepLR(
            optimizer_ft, step_size=7, gamma=0.1) if self.optimizer_type == 'sgd' else None

        model = TorchModel(model_ft, nn.CrossEntropyLoss(), optimizer_ft, exp_lr_scheduler, is_inception)


        return model
