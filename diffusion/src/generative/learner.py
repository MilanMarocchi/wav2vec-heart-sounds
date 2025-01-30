import os
import datetime
import warnings

import torch
import torchaudio
import torch.nn as nn
import matplotlib.pyplot as plt

from torchaudio.functional import resample as tresample
from torch.utils.tensorboard.writer import SummaryWriter
from tqdm import tqdm, TqdmWarning

import utils.plotting

import processing.datasets

from processing.filtering import (
    standardise_signal,
    standardise_torch_signal,
    create_spectrogram,
)

from utils.utils import ensure_numpy

TQDM_COLS = 120
warnings.filterwarnings('ignore', category=TqdmWarning)


def _nested_map(struct, map_fn):
    if isinstance(struct, tuple):
        return tuple(_nested_map(x, map_fn) for x in struct)
    if isinstance(struct, list):
        return [_nested_map(x, map_fn) for x in struct]
    if isinstance(struct, dict):
        return {k: _nested_map(v, map_fn) for k, v in struct.items()}
    return map_fn(struct)


def state_dict_match(old_state_dict, new_state_dict):

    new_state_dict = {k: v for k, v in new_state_dict.items()
                      if k in old_state_dict and
                      ((not hasattr(v, 'shape') and not hasattr(v, 'shape'))
                       or (hasattr(v, 'shape') and hasattr(old_state_dict[k], 'shape')
                           and v.shape == old_state_dict[k].shape))
                      }

    old_state_dict.update(new_state_dict)
    return old_state_dict


def predict_data(model, output_wav_path, label, dataset, transform,
                 condition_sig_name, reference_sig_name,
                 condition_wav_path, reference_wav_path='',
                 multichannel='',  verbose=True):

    del condition_sig_name
    del reference_sig_name

    device = next(model.parameters()).device

    sr = model.params.sample_rate

    label = processing.datasets.get_label_index(dataset, label).to(device)

    multichannel_save_audio = []

    condition_signal, conditional_sr = torchaudio.load(condition_wav_path)  # type: ignore
    assert conditional_sr == sr, f'{conditional_sr=}, {sr=}'
    condition_signal = torch.from_numpy(standardise_signal(
        # post_filter_ecg(ensure_numpy(condition_signal), sr)
        ensure_numpy(condition_signal)
    )).squeeze(0).float()

    if verbose:
        print('Generating data...')

    spectrogram = create_spectrogram(condition_signal, transform).to(device)

    generated_signal, generated_sr = model.predict(
        spectrogram=spectrogram,
        label=label,
    )
    assert generated_sr == sr, f'{generated_sr=}, {sr=}'
    generated_signal = standardise_torch_signal(generated_signal)

    multichannel_save_audio.append(generated_signal)

    if multichannel != '':

        multichannel_save_audio.append(condition_signal)

        if dataset == 'training-a':
            alternate_signal, alternate_sr = model.predict(
                spectrogram=spectrogram,
                label=1-label,
            )
            assert alternate_sr == sr, f'{alternate_sr=}, {sr=}'
            alternate_signal = standardise_torch_signal(alternate_signal)

            multichannel_save_audio.append(alternate_signal)

        if reference_wav_path != '':

            reference_signal, reference_sr = torchaudio.load(reference_wav_path)  # type: ignore
            assert reference_sr == sr, f'{reference_sr=}, {sr=}'
            reference_signal = standardise_torch_signal(reference_signal)

            multichannel_save_audio.append(reference_signal)

        multichannel_save_audio = [a[:condition_signal.shape[0]] for a in multichannel_save_audio]
        multichannel_save_audio = torch.stack(multichannel_save_audio, dim=0)
        torchaudio.save(multichannel, multichannel_save_audio, sample_rate=generated_sr)  # type: ignore

    if verbose:
        print(f'Saving data to {output_wav_path}...')

    torchaudio.save(output_wav_path,   # type: ignore
                    torch.stack([generated_signal[:condition_signal.shape[0]]], dim=0),
                    sample_rate=generated_sr)

    if verbose:
        print('Data saved successfully')


class Learner():

    def __init__(self, model_dir, model, dataloader_train, dataloader_valid,
                 dataset, optimizer, params, final_sr,
                 ref_sig_name, con_sig_name,
                 transform_ref, transform_con,
                 post_transform_ref, post_transform_con,
                 *args, **kwargs):
        del args

        os.makedirs(model_dir, exist_ok=True)
        self.model_dir = model_dir
        self.model = model

        self.dataloader_train = dataloader_train
        self.dataloader_valid = dataloader_valid
        self.dataset = dataset

        self.optimizer = optimizer
        self.params = params
        self.autocast = torch.cuda.amp.autocast(enabled=kwargs.get('fp16', False))  # type: ignore
        self.scaler = torch.cuda.amp.GradScaler(enabled=kwargs.get('fp16', False))  # type: ignore

        self.step = 0
        self.grad_norm = None

        self._mel_transform_ref = transform_ref
        self._mel_transform_con = transform_con

        self._post_transform_ref = post_transform_ref
        self._post_transform_con = post_transform_con

        self.loss_fn = nn.L1Loss()

        self.summary_writer = None
        self.lowest_valid_loss = float('inf')

        self.ref_sig_name = ref_sig_name
        self.con_sig_name = con_sig_name

        self.model_name = 'Generative'
        self.final_sr = final_sr

    def train_step(self, *args, **kwargs):
        del args
        del kwargs
        raise NotImplementedError()

    def predict(self, *args, **kwargs):
        del args
        del kwargs
        raise NotImplementedError()

    def state_dict(self):
        if hasattr(self.model, 'module') and isinstance(self.model.module, nn.Module):
            model_state = self.model.module.state_dict()
        else:
            model_state = self.model.state_dict()
        return {
            'step': self.step,
            'model': {k: v.cpu() if isinstance(v, torch.Tensor) else v
                      for k, v in model_state.items()},
            'optimizer': {k: v.cpu() if isinstance(v, torch.Tensor) else v
                          for k, v in self.optimizer.state_dict().items()},
            'params': dict(self.params),
            'scaler': self.scaler.state_dict(),
        }

    def load_state_dict(self, state_dict):

        if hasattr(self.model, 'module') and isinstance(self.model.module, nn.Module):
            self.model.module.load_state_dict(state_dict_match(  # type: ignore
                self.model.module.state_dict(), state_dict['model']))
        else:
            self.model.load_state_dict(state_dict_match(
                self.model.state_dict(), state_dict['model']))

        optim_state_dict = self.optimizer.state_dict()
        if (len(optim_state_dict['param_groups'][0]['params'])
                == len(state_dict['optimizer']['param_groups'][0]['params'])):
            self.optimizer.load_state_dict(state_dict_match(
                self.optimizer.state_dict(), state_dict['optimizer']))

        self.scaler.load_state_dict(state_dict['scaler'])
        self.step = state_dict['step']

    def save_to_checkpoint(self, filename='weights', link_filename='weights'):
        save_basename = f'{filename}-{self.step}.pt'
        save_name = f'{self.model_dir}/{save_basename}'
        link_name = f'{self.model_dir}/{link_filename}.pt'
        torch.save(self.state_dict(), save_name)
        if os.name == 'nt':
            torch.save(self.state_dict(), link_name)
        else:
            if os.path.islink(link_name):
                os.unlink(link_name)
            os.symlink(save_basename, link_name)

    def restore_from_checkpoint(self, checkpoint_path=''):

        if checkpoint_path == '':
            checkpoint_path = f'{self.model_dir}/weights.pt'
        try:
            checkpoint = torch.load(checkpoint_path)
            self.load_state_dict(checkpoint)
            return checkpoint_path
        except FileNotFoundError:
            return ''

    def _valid(self, num, tqdm_step):
        device = next(self.model.parameters()).device

        valid_loss = None
        processed = False

        valid_features = None
        for features in tqdm(self.dataloader_valid, ncols=TQDM_COLS,
                             desc=f'({tqdm_step}/3) Validation set', position=1, leave=False):
            features = _nested_map(features, lambda x: x.to(device) if isinstance(x, torch.Tensor) else x)
            v_l = self.train_step(features, train=False)
            if valid_loss is None:
                valid_loss = v_l
            else:
                valid_loss += v_l

            if not processed:
                valid_features = self._get_some_features(features, num)
                processed = True

        assert valid_loss is not None, f'{valid_loss=}'

        return valid_features, valid_loss / len(self.dataloader_valid)

    def train(self, save_interval_m=30):
        device = next(self.model.parameters()).device
        loss, features = None, None

        step_info = {}

        start_time = datetime.datetime.now()
        last_save_time = start_time
        epoch_count = 0
        num_saves = 0

        bar_format = '{desc} {percentage:.1f}%|{bar}| {n:.1f}/{total_fmt} [{elapsed}<{remaining}'

        with tqdm(total=save_interval_m, position=0, ncols=120, leave=True, bar_format=bar_format) as save_pbar:
            while True:

                current_time = datetime.datetime.now()
                elapsed_time_h = (current_time - start_time).total_seconds() / (60**2)
                time_since_last_save_s = (current_time - last_save_time).total_seconds()

                desc = f'Epochs: {epoch_count} Saves: {num_saves} Elapsed: {elapsed_time_h:.1f}h'
                save_pbar.set_description(desc)

                for features in tqdm(self.dataloader_train, position=1, ncols=120, leave=False):

                    self.step += 1
                    features = _nested_map(features, lambda x: x.to(device) if isinstance(x, torch.Tensor) else x)
                    loss = self.train_step(features)

                    if torch.isnan(loss).any():
                        raise RuntimeError(f'Detected NaN loss at step {self.step}.')

                    step_info[self.step] = {'loss': loss, 'grad_norm': self.grad_norm}

                epoch_count += 1

                if time_since_last_save_s >= save_interval_m * 60 or num_saves == 0:
                    self._write_summary(features, step_info)
                    num_saves += 1
                    step_info = {}
                    last_save_time = datetime.datetime.now()
                    save_pbar.reset()
                else:
                    save_pbar.update(time_since_last_save_s / 60 - save_pbar.n)

    def _get_spectrogram(self, signal, sig_type):
        if sig_type == 'ref' or sig_type == 'gen':
            mel_transform = self._mel_transform_ref
        elif sig_type == 'con':
            mel_transform = self._mel_transform_con
        else:
            raise ValueError(f'sig_type must be ref or con, not {sig_type}')
        return torch.flip(create_spectrogram(signal, mel_transform), [1])

    def _get_some_features(self, features, num):
        batch_size = len(features['label'])
        example_features = {str(num): [] for num in range(5)}

        for index in range(batch_size):
            label = str(features['label'][index].item())
            if len(example_features[label]) < num or num < 0:
                feature_dict = {
                    feature_name: features[feature_name][index]
                    for feature_name in features.keys()
                }
                example_features[label].append(feature_dict)

        return example_features

    def _write_summary(self, features, step_info):
        writer = self.summary_writer or SummaryWriter(self.model_dir, purge_step=None)

        with torch.no_grad():

            valid_feature_dicts, valid_loss = self._valid(-1, 1)

            writer.add_scalar('valid/loss', valid_loss, self.step)

            for step in step_info:
                writer.add_scalar('train/loss', step_info[step]['loss'], step)
                writer.add_scalar('train/grad_norm', step_info[step]['grad_norm'], step)

            train_feature_dicts = self._get_some_features(features, -1)

            fig, (axes1, axes2) = plt.subplots(2, 2, figsize=(12.0, 12.0))

            self._write_features_to_summary(writer, train_feature_dicts, 'train', axes1, 2)
            self._write_features_to_summary(writer, valid_feature_dicts, 'valid', axes2, 3)

            for ax in [*axes1, *axes2]:
                ax.set(aspect='equal')

            # handles1, plot_labels1 = ax1.get_legend_handles_labels()
            # handles2, plot_labels2 = ax2.get_legend_handles_labels()

            # handles = [*handles1, *handles2]
            # plot_labels = [*plot_labels1, *plot_labels2]

            # unique = [(h, l) for i, (h, l) in enumerate(zip(handles, plot_labels)) if l not in plot_labels[:i]]
            # fig.legend(*zip(*unique),
            #            loc='upper center',
            #            bbox_to_anchor=(0.5, -0.05),
            #            fancybox=True,
            #            shadow=True,
            #            ncol=5)

            writer.add_figure('DimensionReduction', fig, self.step)

            plt.close('all')

            writer.flush()
            self.summary_writer = writer

            if valid_loss < self.lowest_valid_loss:
                self.lowest_valid_loss = valid_loss
                filename = f'weights-{self.step}-valid-loss{valid_loss:.4f}'
                self.save_to_checkpoint(filename=filename, link_filename='weights-valid')

            self.save_to_checkpoint()

    def _write_features_to_summary(self, writer, feature_dicts, split, axes, tqdm_step):
        dim_red_features = []

        for label, feature_list in tqdm(feature_dicts.items(), desc=f'({tqdm_step}/3)Writing {split}-split signals',
                                        ncols=TQDM_COLS, position=1, leave=False):

            string_label = processing.datasets.get_index_label(self.dataset, int(label))
            for feature_dict in feature_list:
                dim_red_features.append(self._write_signal(feature_dict, writer, string_label, split))

        ref_features = []
        gen_features = []
        ref_labels = []
        gen_labels = []

        for d in dim_red_features:
            ref_features.append(d['ref_features'])
            gen_features.append(d['gen_features'])
            ref_labels.append('ref-' + d['labels'])
            gen_labels.append('gen-' + d['labels'])

        utils.plotting.plot_pacmap(axes, split, ref_features, gen_features, ref_labels, gen_labels)

    def _write_signal(self, feature_dict, writer, string_label, split):
        ref_audio = feature_dict['ref_audio'] / torch.max(torch.abs(feature_dict['ref_audio']))
        con_audio = feature_dict['con_audio'] / torch.max(torch.abs(feature_dict['con_audio']))
        con_spec = feature_dict['con_spec']
        label = feature_dict['label']

        gen_audio, new_sr = self.predict(con_spec, label)
        gen_audio /= torch.max(torch.abs(gen_audio))

        seg_wave = feature_dict['seg_wave']
        chirp_wave = feature_dict['chirp_wave']

        orig_ref_audio = ref_audio.cpu().reshape(-1)
        orig_con_audio = con_audio.cpu().reshape(-1)
        orig_gen_audio = gen_audio.cpu().reshape(-1)

        assert new_sr == self.params.sample_rate, f'{new_sr=}, {self.params.sample_rate=}'

        gen_audio = tresample(torch.from_numpy(self._post_transform_ref(gen_audio[0].cpu().numpy(),
                                                                        self.params.sample_rate).copy()),
                              self.params.sample_rate, self.final_sr).float()
        ref_audio = tresample(torch.from_numpy(self._post_transform_ref(ref_audio.cpu().numpy(),
                                                                        self.params.sample_rate).copy()),
                              self.params.sample_rate, self.final_sr).float()

        con_audio = tresample(torch.from_numpy(self._post_transform_con(con_audio.cpu().numpy(),
                                                                        self.params.sample_rate).copy()),
                              self.params.sample_rate, self.final_sr).float()

        seg_wave = tresample(seg_wave, self.params.sample_rate, self.final_sr).float()
        chirp_wave = tresample(chirp_wave, self.params.sample_rate, self.final_sr).float()

        folder_prefix = f'feature-{string_label}-{split}'
        file_prefix = feature_dict['patient']

        def write_audio_and_spec(audio, sig_name, sig_type=None, old_sr=self.final_sr, new_sr=44100):
            if sig_type is None:
                sig_type = sig_name
            new_audio = tresample(audio, old_sr, new_sr)
            new_audio = new_audio / torch.max(torch.abs(new_audio))
            new_audio = new_audio.reshape(1, -1)
            writer.add_audio(f'{folder_prefix}-audio/{file_prefix}_{sig_name}',
                             new_audio,
                             self.step,
                             sample_rate=new_sr)
            writer.add_image(f'{folder_prefix}-spec/{file_prefix}_{sig_name}',
                             self._get_spectrogram(audio, sig_type),
                             self.step)

        ref_audio = ref_audio.cpu().reshape(1, -1)
        gen_audio = gen_audio.cpu().reshape(1, -1)
        con_audio = con_audio.cpu().reshape(1, -1)
        seg_wave = seg_wave.cpu().reshape(1, -1)
        chirp_wave = chirp_wave.cpu().reshape(1, -1)

        dim_red_features = {
            'ref_features': self._get_spectrogram(ref_audio, sig_type='ref')[0],
            'gen_features': self._get_spectrogram(gen_audio, sig_type='ref')[0],
            'labels': string_label,
        }

        seg_wave = 2 * (seg_wave - torch.min(seg_wave))/(torch.max(seg_wave) - torch.min(seg_wave)) - 1

        write_audio_and_spec(ref_audio, 'ref')
        write_audio_and_spec(gen_audio, 'gen')
        write_audio_and_spec(con_audio, 'con')

        fig, ax = plt.subplots(4, 3, figsize=(12.0, 12.0))

        ref_audio = ref_audio.reshape(-1)
        gen_audio = gen_audio.reshape(-1)
        con_audio = con_audio.reshape(-1)
        seg_wave = seg_wave.reshape(-1)
        chirp_wave = chirp_wave.reshape(-1)

        utils.plotting.plot_wav(ax[0][0], ref_audio, self.final_sr)
        utils.plotting.plot_wav(ax[1][0], gen_audio, self.final_sr)
        utils.plotting.plot_wav(ax[2][0], con_audio, self.final_sr)
        utils.plotting.plot_wav(ax[3][0], seg_wave, self.final_sr)

        utils.plotting.plot_stft(ax[0][1], ref_audio, sr=self.final_sr)
        utils.plotting.plot_stft(ax[1][1], gen_audio, sr=self.final_sr)
        utils.plotting.plot_stft(ax[2][1], con_audio, sr=self.final_sr)
        utils.plotting.plot_stft(ax[3][1], chirp_wave, sr=self.final_sr)

        utils.plotting.plot_psd(ax[0][2], ref_audio, self.final_sr)
        utils.plotting.plot_psd(ax[1][2], gen_audio, self.final_sr)
        utils.plotting.plot_psd(ax[2][2], con_audio, self.final_sr)

        ax[0][0].set_title('WAV')
        ax[0][1].set_title('STFT')
        ax[0][2].set_title('PSD')

        labels = ['REF', 'GEN', 'CON', 'SEG']
        for i, label in enumerate(labels):
            ax[i][0].text(-0.25, 0.5, label,
                          size='large', ha='center', rotation=90, va='center', transform=ax[i][0].transAxes)

        utils.plotting.plot_description(
            ax[3][2],
            ref_sig_name=self.ref_sig_name,
            gen_sig_name=self.ref_sig_name,
            con_sig_name=self.con_sig_name,
            patient=feature_dict['patient'],
            diagnosis=string_label,
            model_name=self.model_name,
        )

        fig.tight_layout()
        writer.add_figure(f'{folder_prefix}_comparisons/{file_prefix}_comparisons', fig, self.step)

        fig, ax = plt.subplots(4, 1, figsize=(6.0, 12.0))

        utils.plotting.plot_spec(ax[0], orig_ref_audio, self._mel_transform_ref)
        utils.plotting.plot_spec(ax[1], orig_gen_audio, self._mel_transform_ref)
        utils.plotting.plot_spec(ax[2], orig_con_audio, self._mel_transform_con)

        for i, label in enumerate(labels[:-1]):
            ax[i].text(-0.25, 0.5, label,
                       size='large', ha='center', rotation=90, va='center', transform=ax[i].transAxes)

        utils.plotting.plot_description(
            ax[3],
            ref_sig_name=self.ref_sig_name,
            gen_sig_name=self.ref_sig_name,
            con_sig_name=self.con_sig_name,
            patient=feature_dict['patient'],
            diagnosis=string_label,
            model_name=self.model_name,
        )

        fig.tight_layout()
        writer.add_figure(f'{folder_prefix}-specs-all/{file_prefix}_all', fig, self.step)

        return dim_red_features
