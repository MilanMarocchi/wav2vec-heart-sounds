"""
    models.py
    Author: Milan Marocchi

    Purpose: Contains all the different models, with the idea of ease of swapping models
"""
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import Wav2Vec2Model, Wav2Vec2Processor
import copy

class CNN2DBiLSTM(nn.Module):
    def __init__(self, n_classes, device, cnn, signal_len):
        super(CNN2DBiLSTM, self).__init__()
        self.device = device
        self.hidden_size = 50
        self.num_layers = 2
        self.signal_len = signal_len

        self.cnn = cnn

        input_size = self.get_lstm_size()
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=self.hidden_size, num_layers=self.num_layers,
                            batch_first=True, bidirectional=True)

    def get_lstm_size(self):
        x = torch.zeros((1, 1, self.signal_len))
        out = self.cnn_layers(x)

        return out.shape[2]

    def forward(self, x):

        return x

class CNNBiLSTM(nn.Module):
    def __init__(self, n_classes, device, signal_len):
        super(CNNBiLSTM, self).__init__()
        self.device = device
        self.signal_len = signal_len
        self.hidden_size = 50
        self.num_layers = 2

        # CNN layers
        self.cnn_layers = nn.Sequential(
            nn.Conv1d(in_channels=1, out_channels=12, kernel_size=32, stride=42),
            nn.BatchNorm1d(12),
            nn.ReLU(),
            nn.Conv1d(in_channels=12, out_channels=12, kernel_size=32, stride=2),
            nn.BatchNorm1d(12),
            nn.ReLU(),
            nn.Conv1d(in_channels=12, out_channels=12, kernel_size=32, stride=2),
            nn.BatchNorm1d(12),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2)
        )

        input_size = self.get_lstm_size()
        # BiLSTM layers
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=self.hidden_size, num_layers=self.num_layers, 
                            batch_first=True, bidirectional=True)

        # Fully connected layer
        self.fc = nn.Linear(self.hidden_size*2, n_classes)

    def get_lstm_size(self):
        x = torch.zeros((1, 1, self.signal_len))
        out = self.cnn_layers(x)

        return out.shape[2]

    def forward(self, x):
        # Pass the input through CNN layers
        x = x.unsqueeze(1)
        x = self.cnn_layers(x)

        # Flatten the output for the LSTM layers
        x = x.permute(0, 2, 1)

        # Pass the CNN output to LSTM layers
        x, _ = self.lstm(x)

        # Take the output of the last time step
        x = x[:, -1, :]

        # Pass the LSTM output to the fully connected layer
        x = self.fc(x)

        x = F.log_softmax(x, dim=1)

        return x



class BiLSTM(nn.Module):

    def __init__(self, num_classes, device, num_features):
        super(BiLSTM, self).__init__()
        self.hidden_size = 200
        self.num_layers = 3
        self.lstm = nn.LSTM(int(num_features), self.hidden_size, self.num_layers, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(self.hidden_size * 2, num_classes)
        self.num_classes = num_classes
        self.device = device

    def forward(self, x):
        h0 = torch.zeros(self.num_layers * 2, x.size(0), self.hidden_size).to(self.device)
        c0 = torch.zeros(self.num_layers * 2, x.size(0), self.hidden_size).to(self.device)

        print(x.shape)
        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])  # Taking the output of the last time step

        out = F.log_softmax(out, dim=1)

        return out


class Wav2Vec(nn.Module):
    def __init__(self, num_classes):
        super(Wav2Vec, self).__init__()
        self.wav2vec = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base-960h")
        self.classifier = nn.Sequential(
            nn.Linear(768, 256),
            nn.ReLU(),
            nn.Linear(256, num_classes)
        )
        self.num_classes = num_classes

    def forward(self, x):
        # Without altering the feature extraction.
        # with torch.no_grad():
        out = self.wav2vec(x).last_hidden_state # type: ignore
        out = out.mean(dim=1)  # Global average pooling
        out = self.classifier(out)
        out = F.log_softmax(out, dim=1)
        return out 

    def extract_features(self, x):
        out = self.wav2vec(x).last_hidden_state # type: ignore
        out = out.mean(dim=1)  # Global average pooling
        return out

class Wav2VecCNN(nn.Module):
    # NOTE: signal_length is in number of samples
    def __init__(self, num_classes, signal_length):
        super(Wav2VecCNN, self).__init__()
        self.signal_length = 4125 * 4 # Hardcoded for now to get results
        self.ft_extractor = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base-960h").feature_extractor # type: ignore
        self.cnn_out_size = self._get_cnn_size()
        self.classifier = nn.Sequential(
            nn.Linear(self.cnn_out_size, 256),
            nn.ReLU(),
            nn.Linear(256, num_classes)
        )
        self.num_classes = num_classes

    def _get_cnn_size(self):
        input = torch.zeros(64, self.signal_length)
        print(self.signal_length)
        out = self.ft_extractor(input).mean(dim=2)

        return out.shape[-1]

    def forward(self, x):
        # Without altering the feature extraction.
        # with torch.no_grad():
        out = self.ft_extractor(x)
        out = out.mean(dim=2)  # Global average pooling
        out = self.classifier(out)
        out = F.log_softmax(out, dim=1)
        return out 

    def extract_features(self, x):
        out = self.ft_extractor(x)
        out = out.mean(dim=2)  # Global average pooling
        return out


class EnsembleModel(nn.Module):
    """
    Ensemble model to provide an ensemble of multiple models trained on single bands
    """

    def __init__(self, models, num_classes):
        super(EnsembleModel, self).__init__()
        self.models = models
        self.num_models = len(models)
        self.num_classes = num_classes
        self.classifier = nn.Linear(self.num_classes * self.num_models, self.num_classes)

    def forward(self, inputs):
        for i in range(len(inputs[0])):
            input = inputs[:, i]
            model = self.models[i]

            if i == 0:
                x_agg = model(input)
            else:
                x = model(input)
                x_agg = torch.cat((x_agg, x), dim=1) # type: ignore

        out = self.classifier(x_agg) # type: ignore

        return out


class BigModel(nn.Module):
    """
    Big model to provide the same as the ensemble without loss of information (maybe?)
    """

    def __init__(self, models, num_classes, model_code):
        super(BigModel, self).__init__()
        self.models = nn.ModuleList(models)
        self.hidden_layer = 128
        self.num_models = len(models)
        self.num_classes = num_classes
        self.model_code = model_code
        # Sets up models and gets num_features
        self.setup_models()
        self.classifier = nn.Sequential(
            nn.Linear(self.num_features * self.num_models, self.hidden_layer * 2),
            nn.ReLU(),
            nn.Linear(self.hidden_layer * 2, self.hidden_layer),
            nn.ReLU(),
            nn.Linear(self.hidden_layer, self.num_classes)
        )

    def setup_models(self):
        new_models = []
        if self.model_code == "resnet":
            self.num_features = self.models[0].fc.in_features
        elif self.model_code == "vgg":
            self.num_features = self.models[0].classifier[6].in_features
        elif self.model_code == "inception":
            self.num_features = self.models[0].fc.in_features
        else:
            raise ValueError(f"Unsupported model code: {self.model_code=}")

        for model in self.models:
            # Remove the last classification layer
            features = list(model.children())[:-1]
            new_model = nn.Sequential(*features)

            new_models.append(new_model)

        self.models = nn.ModuleList(new_models)

    def forward(self, inputs):
        # Get batch size and ensure the input tensor is of the expected shape
        batch_size = inputs.size(0)
        if inputs.dim() != 5 or inputs.size(2) != 3:
            raise ValueError("Expected 'inputs' tensor to have shape [batch_size, num_models, 3, 224, 224].")

        # Process each set of images (indexed by i) with its corresponding model
        x_aggregated_list = []
        for i in range(self.num_models):
            input_tensor = inputs[:, i, :, :, :]
            model = self.models[i]
            print(input_tensor.shape)
            x = model(input_tensor)  # Shape: [batch_size, 2048, 1, 1]
            if self.model_code == "inception":
                x_flat = x[0] # As we don't want aux logits/features
            x_flat = x.view(batch_size, -1)  # Shape: [batch_size, 2048]
            x_aggregated_list.append(x_flat)

        # Concatenate along the feature dimension
        x_agg = torch.cat(x_aggregated_list, dim=1)  # Shape: [batch_size, 2048 * num_models]

        out = self.classifier(x_agg)
        return out

    def extract_features(self, inputs):
        # Get batch size and ensure the input tensor is of the expected shape
        batch_size = inputs.size(0)
        if inputs.dim() != 5 or inputs.size(2) != 3:
            raise ValueError("Expected 'inputs' tensor to have shape [batch_size, num_models, 3, 224, 224].")

        # Process each set of images (indexed by i) with its corresponding model
        x_aggregated_list = []
        for i in range(self.num_models):
            input_tensor = inputs[:, i, :, :, :]
            model = self.models[i]
            print(input_tensor.shape)
            x = model(input_tensor)  # Shape: [batch_size, 2048, 1, 1]
            if self.model_code == "inception":
                x_flat = x[0] # As we don't want aux logits/features
            x_flat = x.view(batch_size, -1)  # Shape: [batch_size, 2048]
            x_aggregated_list.append(x_flat)

        # Concatenate along the feature dimension
        x_agg = torch.cat(x_aggregated_list, dim=1)  # Shape: [batch_size, 2048 * num_models]

        return x_agg


class BigRNNModel(BigModel):
    """
    Big model to provide the same as the ensemble without loss of information (maybe?)
    """
    def __init__(self, models, num_classes, model_code):
        super(BigRNNModel, self).__init__(models, num_classes, model_code)

    def setup_models(self):
        new_models = []
        if self.model_code == "wav2vec":
            self.num_features = self.models[0].wav2vec.config.hidden_size
        elif self.model_code == "wav2vec-cnn":
            self.num_features = self.models[0].cnn_out_size
        else:
            raise ValueError(f"Unsupported model code: {self.model_code=}")

        for model in self.models:
            # Remove the last classification layer
            if self.model_code == "wav2vec":
                new_model = model.wav2vec
            elif self.model_code == "wav2vec-cnn":
                new_model = model.ft_extractor
            else:
                features = list(model.children())[:-1]
                new_model = nn.Sequential(*features)

            new_models.append(new_model)

        self.models = nn.ModuleList(new_models)

    def forward(self, inputs):
        # Get batch size and ensure the input tensor is of the expected shape
        batch_size = inputs.size(0)
        if inputs.dim() != 3:
            raise ValueError("Expected 'inputs' tensor to have shape [batch_size, sequence_len, num_models/channels].")

        inputs = inputs.float()
        # Process each set of images (indexed by i) with its corresponding model
        x_aggregated_list = []
        for i in range(self.num_models):
            if self.model_code == "wav2vec":
                model = self.models[i]
                x = model(inputs[:, :, i]).last_hidden_state
                x = x.mean(dim=1)  # Global average pooling
                x_flat = x.view(batch_size, -1)  # Shape: [batch_size, 2048]
                x_aggregated_list.append(x)
            elif self.model_code == "wav2vec-cnn":
                model = self.models[i]
                x = model(inputs[:, :, i])
                x = x.mean(dim=2)  # Global average pooling
                x_flat = x.view(batch_size, -1)  # Shape: [batch_size, 2048]
                x_aggregated_list.append(x)
            else:
                model = self.models[i]
                x = model(inputs[:, :, i])  # Shape: [batch_size, 2048, 1, 1]
                x_flat = x.view(batch_size, -1)  # Shape: [batch_size, 2048]
                x_aggregated_list.append(x)


        # Concatenate along the feature dimension
        x_agg = torch.cat(x_aggregated_list, dim=1)  # Shape: [batch_size, 2048 * num_models]

        out = self.classifier(x_agg)
        return out


class MLModel():
    """
    MLModel to encapsulate functions increasing code reuse
    and decreasing complexity
    """

    def __init__(self):
        self.train = None
        self.set_mode = None
        self.clean_up = None
        self.get_weights = None
        self.set_weights = None
        self.classify_fragment = None
        self.classify_patient = None


class TorchModel(MLModel):
    """
    Encapsulates all data and functions to train and classify a Torch Model
    """

    def __init__(self, model_ft, criterion, optimizer_ft, exp_lr_scheduler, is_inception, is_wav2vec=False):
        self.model_ft = model_ft
        self.criterion = criterion
        self.optimizer_ft = optimizer_ft
        self.exp_lr_scheduler = exp_lr_scheduler
        self.is_inception = is_inception
        self.is_wav2vec = is_wav2vec

    def get_weights(self):
        return copy.deepcopy(self.model_ft.state_dict())

    def set_weights(self, weights):
        self.model_ft.load_state_dict(weights)

    def set_mode(self, phase):
        if phase == 'train':
            self.model_ft.train()  # Set model to training mode
        else:
            self.model_ft.eval()   # Set model to evaluate mode

    def _setup_labels(self, labels):
        labels = [1 if int(x.split('.')[1]) == 1 else 0 for x in labels]
        labels = torch.tensor(labels)

        return labels

    def train(self, inputs, labels, phase, runningCM):
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        inputs = inputs.to(device)
        labels = self._setup_labels(labels).to(device)

        # zero the parameter gradients
        self.optimizer_ft.zero_grad()

        # forward
        # track history if only in train
        with torch.set_grad_enabled(phase == 'train'):
            # Get model outputs and calculate loss
            # Special case for inception because in training it has an auxiliary output. In train
            #   mode we calculate the loss by summing the final output and the auxiliary output
            #   but in testing we only consider the final output.
            if self.is_inception and phase == 'train':
                # From https://discuss.pytorch.org/t/how-to-optimize-inception-model-with-auxiliary-classifiers/7958
                outputs, aux_outputs = self.model_ft(inputs)
                loss1 = self.criterion(outputs, labels)
                loss2 = self.criterion(aux_outputs, labels)
                loss = loss1 + 0.4*loss2
            else:
                outputs = self.model_ft(inputs)
                loss = self.criterion(outputs, labels)

            _, preds = torch.max(outputs, 1)

            # backward + optimize only if in training phase
            if phase == 'train':
                loss.backward()
                self.optimizer_ft.step()

            # statistics
            runningCM.update(y_true=labels.data.to("cpu"), y_pred=preds.to("cpu"), loss=float(loss.item()*inputs.size(0)))

    def clean_up(self, phase):
        if phase == 'train':
            if self.exp_lr_scheduler:
                self.exp_lr_scheduler.step()

    def classify_patient(self, inputs, labels, fragment_logits):
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        cpu_device = torch.device("cpu")

        inputs = inputs.to(device)
        full_labels = [".".join(x.split('/')[-1].split('.')[0:2]) for x in labels]
        labels = self._setup_labels(labels).to(device)

        outputs = self.model_ft(inputs)

        # Update fragment_logits
        for idx, label in enumerate(full_labels):
            fragment_logits[label].append(outputs[idx].data.to(cpu_device))

    def classify_fragment(self, inputs, labels, runningCM):
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        inputs = inputs.to(device)
        labels = self._setup_labels(labels).to(device)

        outputs = self.model_ft(inputs)
        loss = self.criterion(outputs, labels)

        _, preds = torch.max(outputs, 1)

        # statistics
        runningCM.update(y_true=labels.data.to("cpu"), y_pred=preds.to("cpu"), loss=float(loss.item()*inputs.size(0)))

    def save_model(self, path):
        """
        Stores the model into a file.
        """
        os.makedirs(os.path.dirname(path), exist_ok=True)
        if not self.is_wav2vec:
            torch.save(self.model_ft, path)
        else:
            torch.save(self.model_ft.state_dict(), path)

# FIXME: Add an implementation for a TF model when required.
