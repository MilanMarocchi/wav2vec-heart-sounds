"""
    training.py
    Author : Milan Marocchi

    Contains the training code to train the ml algorithm.
    Allows for testing of different pre-processing and ml algorithms.
"""
import time
import math

from tqdm.auto import tqdm
from classifier.classify_stats import RunningBinaryConfusionMatrix

import torch
from torch.utils.tensorboard.writer import SummaryWriter

class Trainer():

    def __init__(self):
        pass

    def train_model(self, model, dataloaders, num_epochs=25):
        raise NotImplementedError("Needs to be implemented.")

    def save_model(self, model, output_path):
        raise NotImplementedError("Needs to be implemented.")

class FineTunerTrainer(Trainer):

    def __init__(self):
        self.writer = SummaryWriter()

    def update_summary_writer(self, stats: dict, aliases: dict, phase: str, epoch: int):
        for key in stats.keys():
            self.writer.add_scalar(f"{phase} : {aliases[key]}", stats[key], epoch)

    def train(self, model, dataloaders, num_epochs=25, letskip=False):
        """
        Allows for training of either a pytoch or tf model.
        Expects the model input to be an MLModel
        """
        since = time.time()
        val_qi_history = []

        best_model_wts = model.get_weights()
        best_epoch_measure = -1
        best_epoch = 0
        best_stats = None

        for epoch in tqdm(range(0 if letskip else 1, num_epochs+1), desc='Epoch'):
            # Each epoch has a training and validation phase
            for phase in ['train', 'valid']:

                model.set_mode(phase)

                total_inputs = 0
                runningCM = RunningBinaryConfusionMatrix()

                for inputs, labels in tqdm(dataloaders[phase]):
                    total_inputs += inputs.size(0)
                    model.train(inputs, labels, phase, runningCM)

                model.clean_up(phase)

                assert total_inputs == runningCM.total(), f'{total_inputs=}, {runningCM.total()=}'
                stats = runningCM.get_stats()
                epoch_measure = stats['mcc']
                self.update_summary_writer(stats, runningCM.aliases, phase, epoch)
                print(f'{phase}')
                print(runningCM.display_stats(aliases=False))
                print(runningCM.base_stats)
                print(f'{epoch_measure=:.4f}')

                # copy model
                if phase == 'valid' and not math.isnan(epoch_measure) and epoch_measure > best_epoch_measure:
                    best_epoch_measure = epoch_measure * 0.95 if epoch == 0 else epoch_measure
                    best_model_wts = model.get_weights()
                    best_epoch = epoch
                    best_stats = stats
                    print('Found new best')
                if phase == 'val':
                    val_qi_history.append(epoch_measure)

            print()

        assert best_stats is not None, f'{best_stats=}'
        time_elapsed = time.time() - since
        print('\t Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
        print('\t Best val measure: {:.4f}'.format(best_epoch_measure))
        print('\t Best val epoch: {}'.format(best_epoch))
        print('\t Best val stats:\n\t', ', '.join([f'{s}={best_stats[s]:.3f}' for s in best_stats]))

        # load best model weights
        model.set_weights(best_model_wts)

        return model, best_epoch

    def save_model(self, model, output_path):
        model.save_model(output_path)
