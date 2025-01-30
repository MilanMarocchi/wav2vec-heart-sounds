"""
    testing.py
    Author : Milan Marocchi

    Purpose : To run the classifier to classify heart signals.
"""

from sympy import O
from tqdm.auto import tqdm
from classifier.classify_stats import RunningBinaryConfusionMatrix
from collections import defaultdict
import torch.nn.functional as F
import torch
from typing import Tuple
import numpy as np


class Tester():

    def __init__(self, model, dataloaders):
        self.model = model
        self.dataloaders = dataloaders

    def test(self):
        raise NotImplementedError("Not implemented.")

    def _setup_labels(self, labels):
        # If not already ints split it.
        try:
            labels = [int(x) for x in labels]
        except ValueError:
            labels = [1 if int(x.split('.')[1]) == 1 else 0 for x in labels]
        labels = torch.tensor(labels)

        return labels

class FineTunerPatientTester(Tester):

    def __init__(self, model, dataloaders):
        super().__init__(model, dataloaders)

    def test(self):
        """
        classifies a dataset using a model and criterion.
          model: The model to use (pytorch model)
          dataloaders: The dataloaders to use (expects a dict of dataloaders)
          criterion: Criterion for loss
          device: device to load onto
        """
        # Set to evaluation mode.
        phase = "test"
        self.model.set_mode(phase)

        runningCM = RunningBinaryConfusionMatrix()

        # Create a default dict which will store all the predictions
        fragment_logits = defaultdict(list)

        for inputs, labels in self.dataloaders[phase]:
            self.model.classify_patient(inputs, labels, fragment_logits)

        # Deal with additions for runningCM
        for key in fragment_logits:
            agg_output = sum(fragment_logits[key]) / len(fragment_logits[key])
            agg_output = F.softmax(agg_output, dim=0) # type: ignore
            threshold = 0
            pred = 0 if agg_output[0] + threshold > agg_output[1] else 1

            label = 1 if int(key.split(".")[1]) == 1 else 0

            runningCM.update(y_true=[label], y_pred=[pred], loss=0)
            # logging.debug(f"labels: {label}, predictions: {pred}")

        print('Patient Stats:')
        print(f'{phase}')
        print(runningCM.display_stats(aliases=False))
        print(runningCM.base_stats)

        return runningCM.get_stats()['acc'], runningCM.display_stats()

    def roc_curve(self) -> Tuple[list[float], list[float], list[float]]:
        """
        Creates the roc curve.
        """
        phase = "test"
        self.model.set_mode(phase)
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        tprs = []
        fprs = []
        thresholds = np.linspace(-1,1,60).tolist()
        thresholds.append(0.0)
        thresholds.sort()
        for threshold in tqdm(thresholds, ncols=120):
            runningCM = RunningBinaryConfusionMatrix()

            model = self.model
            model.model_ft.to(device)
            fragment_logits = defaultdict(list)
            with torch.no_grad():
                for inputs, labels in self.dataloaders[phase]:
                    model.classify_patient(inputs, labels, fragment_logits)

                    for key in fragment_logits:
                        agg_output = sum(fragment_logits[key]) / len(fragment_logits[key])
                        agg_output = F.softmax(agg_output, dim=0) # type: ignore
                        preds = 0 if agg_output[0] + threshold > agg_output[1] else 1

                        label = 1 if int(key.split(".")[1]) == 1 else 0

                        runningCM.update(y_true=[label], y_pred=[preds], loss=0)

            stats = runningCM.get_stats()
            tprs.append(stats['tpr'])
            fprs.append(stats['fpr'])

        return tprs, fprs, thresholds



class FineTunerFragmentTester(Tester):

    def __init__(self, model, dataloaders):
        super().__init__(model, dataloaders)

    def test(self):
        """
        Runs the classifying algorithm on the model with the holdout test data.
        """
        # Set to evaluation mode.
        phase = "test"
        self.model.set_mode(phase)

        runningCM = RunningBinaryConfusionMatrix()

        for inputs, labels in self.dataloaders[phase]:
            self.model.classify_fragment(inputs, labels, runningCM)

        print('Fragment Stats:')
        print(f'{phase}')
        print(runningCM.display_stats(aliases=False))
        print(runningCM.base_stats)

    def embeddings(self, extract_features: bool = True) -> Tuple[np.ndarray, np.ndarray]: 
        phase = "test"
        self.model.set_mode(phase)
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        all_features = [] 
        all_labels = []
        with torch.no_grad():
            for inputs, labels in tqdm(self.dataloaders[phase]):
                input_vals = inputs.to(device)
                labels = self._setup_labels(labels).to(device)
                self.model = self.model.to(device)
                if extract_features:
                    fts = self.model.extract_features(input_vals)
                else:
                    fts = input_vals.flatten(start_dim=1)

                all_features.extend(fts.to("cpu"))
                all_labels.extend(labels.to("cpu")) 

        return np.asarray(all_features), np.asarray(all_labels)

    def roc_curve(self) -> Tuple[list[float], list[float], list[float]]:
        """
        Creates the roc curve.
        """
        phase = "test"
        model = self.model.model_ft
        model.eval()
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        tprs = []
        fprs = []
        thresholds = np.linspace(-1,1,60).tolist()
        thresholds.append(0.0)
        thresholds.sort()
        for threshold in tqdm(thresholds, ncols=120):
            runningCM = RunningBinaryConfusionMatrix()

            model = model.to(device)
            with torch.no_grad():
                for inputs, labels in self.dataloaders[phase]:

                    input_vals = inputs.to(device)
                    labels = self._setup_labels(labels).to(device)

                    logits = model(input_vals)
                    loss = F.cross_entropy(input_vals, labels) 
                    logits = F.softmax(logits, dim=1) # type: ignore
                    logits[:, 0] += threshold
                    _, preds = torch.max(logits, 1)

                    runningCM.update(y_true=labels.data.to("cpu"), y_pred=preds.to("cpu"), loss=float(loss.item()*input_vals.size(0)))

            stats = runningCM.get_stats()
            tprs.append(stats['tpr'])
            fprs.append(stats['fpr'])

        return tprs, fprs, thresholds
