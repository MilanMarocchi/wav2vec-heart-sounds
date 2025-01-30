import numpy as np
import torch
import torchaudio
import torch.nn as nn
import torch.nn.functional as F

from math import sqrt

from generative.learner import Learner
from utils.utils import AttrDict


diffwave_params = AttrDict(
    # Training params
    batch_size=8,
    learning_rate=2e-4,
    max_grad_norm=None,

    # Data params
    sample_rate=4000,
    crop_mel_frames=96,
    n_mels=80,
    n_fft=1024,
    hop_samples=256,

    # Model params
    residual_layers=30,
    residual_channels=64,
    dilation_cycle_length=10,
    unconditional=False,
    noise_schedule=np.linspace(1e-4, 0.05, 50).tolist(),
    inference_noise_schedule=[0.0001, 0.001, 0.01, 0.05, 0.2, 0.5],
    num_classes=5,
    embedding_dim=32,
)


def diffwave_mel_spec_init_ecg(params):
    mel_args = {
        'sample_rate': params['sample_rate'],
        'win_length': params['n_fft'],
        'hop_length': params['hop_samples'],
        'n_fft': params['n_fft'],
        'f_min': 0.125,
        'f_max': 200,
        'n_mels': params['n_mels'],
        'power': 1.0,
        'normalized': True,
    }
    mel_spec_transform = torchaudio.transforms.MelSpectrogram(**mel_args)
    return mel_spec_transform


def diffwave_mel_spec_init_pcg(params):
    mel_args = {
        'sample_rate': params['sample_rate'],
        'win_length': params['n_fft'],
        'hop_length': params['hop_samples'],
        'n_fft': params['n_fft'],
        'f_min': 0.125,
        'f_max': 500,
        'n_mels': params['n_mels'],
        'power': 1.0,
        'normalized': True,
    }
    mel_spec_transform = torchaudio.transforms.MelSpectrogram(**mel_args)
    return mel_spec_transform


def Conv1d(*args, **kwargs):
    layer = nn.Conv1d(*args, **kwargs)
    nn.init.kaiming_normal_(layer.weight)
    return layer


@torch.jit.script  # type: ignore
def silu(x):
    return x * torch.sigmoid(x)


class DiffusionEmbedding(nn.Module):
    def __init__(self, max_steps):
        super().__init__()
        self.register_buffer('embedding', self._build_embedding(max_steps), persistent=False)
        self.projection1 = nn.Linear(128, 512)
        self.projection2 = nn.Linear(512, 512)

    def forward(self, diffusion_step):
        if diffusion_step.dtype in [torch.int32, torch.int64]:
            x = self.embedding[diffusion_step]  # type: ignore
        else:
            x = self._lerp_embedding(diffusion_step)
        x = self.projection1(x)
        x = silu(x)
        x = self.projection2(x)
        x = silu(x)
        return x

    def _lerp_embedding(self, t):
        low_idx = torch.floor(t).long()
        high_idx = torch.ceil(t).long()
        low = self.embedding[low_idx]  # type: ignore
        high = self.embedding[high_idx]  # type: ignore
        return low + (high - low) * (t - low_idx)

    def _build_embedding(self, max_steps):
        steps = torch.arange(max_steps).unsqueeze(1)  # [T,1]
        dims = torch.arange(64).unsqueeze(0)          # [1,64]
        table = steps * 10.0**(dims * 4.0 / 63.0)     # [T,64]
        table = torch.cat([torch.sin(table), torch.cos(table)], dim=1)
        return table


class SpectrogramUpsampler(nn.Module):
    def __init__(self, n_mels):
        del n_mels
        super().__init__()
        self.conv1 = nn.ConvTranspose2d(1, 1, (3, 32), stride=(1, 16), padding=(1, 8))
        self.conv2 = nn.ConvTranspose2d(1, 1,  (3, 32), stride=(1, 16), padding=(1, 8))

    def forward(self, x):
        x = torch.unsqueeze(x, 1)
        x = self.conv1(x)
        x = F.leaky_relu(x, 0.4)
        x = self.conv2(x)
        x = F.leaky_relu(x, 0.4)
        x = torch.squeeze(x, 1)
        return x


class ResidualBlock(nn.Module):
    def __init__(self, n_mels, residual_channels, dilation, label_embed_dim):
        super().__init__()

        self.dilated_conv = Conv1d(residual_channels, 2 * residual_channels, 3, padding=dilation, dilation=dilation)

        self.diffusion_projection = nn.Linear(512, residual_channels)
        self.conditioner_projection = Conv1d(n_mels, 2 * residual_channels, 1)
        self.label_projection = Conv1d(label_embed_dim, 2 * residual_channels, 3, padding=dilation, dilation=dilation)

        self.output_projection = Conv1d(residual_channels, 2 * residual_channels, 1)

    def forward(self, x, diffusion_step, conditioner, label_embed):

        diffusion_step = self.diffusion_projection(diffusion_step).unsqueeze(-1)
        y = x + diffusion_step

        conditioner = self.conditioner_projection(conditioner)
        label_embed = self.label_projection(label_embed.squeeze(1).unsqueeze(-1))

        y = self.dilated_conv(y) + conditioner + label_embed

        gate, filter = torch.chunk(y, 2, dim=1)
        y = torch.sigmoid(gate) * torch.tanh(filter)

        y = self.output_projection(y)

        residual, skip = torch.chunk(y, 2, dim=1)
        return (x + residual) / sqrt(2.0), skip


class DiffWave(nn.Module):
    def __init__(self, params):
        super().__init__()
        self.params = params

        self.input_projection = Conv1d(1, params.residual_channels, 1)
        self.diffusion_embedding = DiffusionEmbedding(len(params.noise_schedule))
        self.spectrogram_upsampler = SpectrogramUpsampler(params.n_mels)

        self.residual_layers = nn.ModuleList([
            ResidualBlock(params.n_mels, params.residual_channels, 2**(i %
                          params.dilation_cycle_length),
                          label_embed_dim=params.embedding_dim)
            for i in range(params.residual_layers)
        ])

        self.skip_projection = Conv1d(params.residual_channels, params.residual_channels, 1)
        self.output_projection = Conv1d(params.residual_channels, 1, 1)
        nn.init.zeros_(self.output_projection.weight)

        self.class_embedding = nn.Embedding(params.num_classes, params.embedding_dim)
        self.class_embedding.weight.data.fill_(0.001)

    def forward(self, ref_audio, diffusion_step, con_spec, label):

        x = ref_audio.unsqueeze(1)
        x = self.input_projection(x)
        x = F.relu(x)

        label_embed = self.class_embedding(label) if label is not None else None

        diffusion_step = self.diffusion_embedding(diffusion_step)
        con_spec = self.spectrogram_upsampler(con_spec)

        skip = None
        for layer in self.residual_layers:
            x, skip_connection = layer(x, diffusion_step, con_spec, label_embed)
            skip = skip_connection if skip is None else skip_connection + skip

        assert skip is not None, f'{skip=}'
        x = skip / sqrt(len(self.residual_layers))
        x = self.skip_projection(x)
        x = F.relu(x)
        x = self.output_projection(x)
        return x

    def predict(self, spectrogram, label, fast_sampling=False):
        device = next(self.parameters()).device
        with torch.no_grad():
            # Change in notation from the DiffWave paper for fast sampling.
            # DiffWave paper -> Implementation below
            # --------------------------------------
            # alpha -> talpha
            # beta -> training_noise_schedule
            # gamma -> alpha
            # eta -> beta
            training_noise_schedule = np.array(self.params.noise_schedule)
            inference_noise_schedule = (np.array(self.params.inference_noise_schedule)
                                        if fast_sampling else
                                        training_noise_schedule)

            talpha = 1 - training_noise_schedule
            talpha_cum = np.cumprod(talpha)

            beta = inference_noise_schedule
            alpha = 1 - beta
            alpha_cum = np.cumprod(alpha)

            T = []
            for s in range(len(inference_noise_schedule)):
                for t in range(len(training_noise_schedule) - 1):
                    if talpha_cum[t+1] <= alpha_cum[s] <= talpha_cum[t]:
                        twiddle = (talpha_cum[t]**0.5 - alpha_cum[s]**0.5) / (talpha_cum[t]**0.5 - talpha_cum[t+1]**0.5)
                        T.append(t + twiddle)
                        break
            T = np.array(T, dtype=np.float32)

            if len(spectrogram.shape) == 2:  # Expand rank 2 tensors by adding a batch dimension.
                spectrogram = spectrogram.unsqueeze(0)
                label = label.unsqueeze(0)

            spectrogram = spectrogram.to(device)
            label = label.to(device)
            audio = torch.randn(spectrogram.shape[0], self.params.hop_samples *
                                spectrogram.shape[-1], device=device)

            # noise_scale = torch.from_numpy(alpha_cum**0.5).float().unsqueeze(1).to(device)

            for n in range(len(alpha) - 1, -1, -1):
                c1 = 1 / alpha[n]**0.5
                c2 = beta[n] / (1 - alpha_cum[n])**0.5
                audio = c1 * (audio - c2 * self(
                    audio, torch.tensor([T[n]], device=audio.device), spectrogram, label
                ).squeeze(1))
                if n > 0:
                    noise = torch.randn_like(audio)
                    sigma = ((1.0 - alpha_cum[n-1]) / (1.0 - alpha_cum[n]) * beta[n])**0.5
                    audio += sigma * noise
                audio = torch.clamp(audio, -1.0, 1.0)

        return audio, self.params.sample_rate


class DiffWaveLearner(Learner):
    def __init__(self, *args, **kwargs):

        super().__init__(*args, **kwargs)

        beta = np.array(self.params.noise_schedule)
        noise_level = np.cumprod(1 - beta)
        self.noise_level = torch.tensor(noise_level.astype(np.float32))

        self.model_name = 'DiffWave'

    def predict(self, spectrogram, label, fast_sampling=False):
        return self.model.predict(spectrogram=spectrogram, label=label, fast_sampling=fast_sampling)

    def train_step(self, features, train=True):
        for param in self.model.parameters():
            param.grad = None

        ref_audio = features['ref_audio']
        con_spec = features['con_spec']
        label = features['label']

        N, _ = ref_audio.shape
        device = ref_audio.device
        self.noise_level = self.noise_level.to(device)

        with self.autocast:
            t = torch.randint(0, len(self.params.noise_schedule), [N], device=device)
            noise_scale = self.noise_level[t].unsqueeze(1)
            noise_scale_sqrt = noise_scale**0.5
            noise = torch.randn_like(ref_audio)
            noisy_audio = noise_scale_sqrt * ref_audio + (1.0 - noise_scale)**0.5 * noise

            predicted = self.model(noisy_audio, t, con_spec, label)
            loss = self.loss_fn(noise, predicted.squeeze(1))

        if train:
            self.scaler.scale(loss).backward()  # type: ignore
            self.scaler.unscale_(self.optimizer)
            self.grad_norm = nn.utils.clip_grad_norm_(  # type: ignore
                self.model.parameters(), self.params.max_grad_norm or 1e9)
            self.scaler.step(self.optimizer)
            self.scaler.update()

        return loss
