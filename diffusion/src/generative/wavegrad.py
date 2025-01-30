import numpy as np
import torch
import torchaudio
import torch.nn as nn
import torch.nn.functional as F

from math import log as ln

from generative.learner import Learner
from utils.utils import AttrDict

wavegrad_params = AttrDict(
    # Training params
    batch_size=8,
    learning_rate=2e-4,
    max_grad_norm=1.0,

    # Data params
    sample_rate=4000,
    crop_mel_frames=96,
    hop_samples=300,

    # Model params
    noise_schedule=np.linspace(1e-6, 0.01, 1000).tolist(),
    num_classes=5,
    embedding_dim=32,
)


def wavegrad_mel_spec_init_ecg(params):
    win = params['hop_samples'] * 4
    n_fft = 2**((win-1).bit_length())
    mel_args = {
        'sample_rate': params['sample_rate'],
        'hop_length': params['hop_samples'],
        'win_length': win,
        'n_fft': n_fft,
        'f_min': 0.125,
        'f_max': 200,
        'n_mels': 128,
        'power': 1.0,
        'normalized': True,
    }
    mel_spec_transform = torchaudio.transforms.MelSpectrogram(**mel_args)
    return mel_spec_transform


def wavegrad_mel_spec_init_pcg(params):
    win = params['hop_samples'] * 4
    n_fft = 2**((win-1).bit_length())
    mel_args = {
        'sample_rate': params['sample_rate'],
        'hop_length': params['hop_samples'],
        'win_length': win,
        'n_fft': n_fft,
        'f_min': 0.125,
        'f_max': 500,
        'n_mels': 128,
        'power': 1.0,
        'normalized': True,
    }
    mel_spec_transform = torchaudio.transforms.MelSpectrogram(**mel_args)
    return mel_spec_transform


class Conv1d(nn.Conv1d):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.orthogonal_(self.weight)
        nn.init.zeros_(self.bias)  # type: ignore


class PositionalEncoding(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x, noise_level):
        return (x + self._build_encoding(noise_level)[:, :, None])

    def _build_encoding(self, noise_level):
        count = self.dim // 2
        step = torch.arange(count, dtype=noise_level.dtype, device=noise_level.device) / count
        encoding = noise_level.unsqueeze(1) * torch.exp(-ln(1e4) * step.unsqueeze(0))
        encoding = torch.cat([torch.sin(encoding), torch.cos(encoding)], dim=-1)
        return encoding


class FiLM(nn.Module):
    def __init__(self, input_size, output_size, num_classes, embedding_dim):
        super().__init__()
        self.label_embedding = nn.Embedding(num_classes, embedding_dim)
        self.label_projection = nn.Conv1d(embedding_dim, input_size, 3, padding=1)
        self.encoding = PositionalEncoding(input_size)
        self.input_conv = nn.Conv1d(input_size, input_size, 3, padding=1)
        self.output_conv = nn.Conv1d(input_size, output_size * 2, 3, padding=1)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.input_conv.weight)
        nn.init.xavier_uniform_(self.output_conv.weight)
        nn.init.zeros_(self.input_conv.bias)  # type: ignore
        nn.init.zeros_(self.output_conv.bias)  # type: ignore
        self.label_embedding.weight.data.fill_(0.001)

    def forward(self, x, noise_scale, labels):
        label_embed = self.label_embedding(labels).squeeze(1).unsqueeze(2)
        label_embed = self.label_projection(label_embed)
        x += label_embed
        x = self.input_conv(x)
        x = F.leaky_relu(x, 0.2)
        x = self.encoding(x, noise_scale)
        shift, scale = torch.chunk(self.output_conv(x), 2, dim=1)
        return shift, scale


class UBlock(nn.Module):
    def __init__(self, input_size, hidden_size, factor, dilation):
        super().__init__()
        assert isinstance(dilation, (list, tuple))
        assert len(dilation) == 4

        self.factor = factor
        self.block1 = Conv1d(input_size, hidden_size, 1)
        self.block2 = nn.ModuleList([
            Conv1d(input_size, hidden_size, 3, dilation=dilation[0], padding=dilation[0]),
            Conv1d(hidden_size, hidden_size, 3, dilation=dilation[1], padding=dilation[1])
        ])
        self.block3 = nn.ModuleList([
            Conv1d(hidden_size, hidden_size, 3, dilation=dilation[2], padding=dilation[2]),
            Conv1d(hidden_size, hidden_size, 3, dilation=dilation[3], padding=dilation[3])
        ])

    def forward(self, x, film_shift, film_scale):
        block1 = F.interpolate(x, size=x.shape[-1] * self.factor)
        block1 = self.block1(block1)

        block2 = F.leaky_relu(x, 0.2)
        block2 = F.interpolate(block2, size=x.shape[-1] * self.factor)
        block2 = self.block2[0](block2)
        block2 = film_shift + film_scale * block2
        block2 = F.leaky_relu(block2, 0.2)
        block2 = self.block2[1](block2)

        x = block1 + block2

        block3 = film_shift + film_scale * x
        block3 = F.leaky_relu(block3, 0.2)
        block3 = self.block3[0](block3)
        block3 = film_shift + film_scale * block3
        block3 = F.leaky_relu(block3, 0.2)
        block3 = self.block3[1](block3)

        x = x + block3
        return x


class DBlock(nn.Module):
    def __init__(self, input_size, hidden_size, factor):
        super().__init__()
        self.factor = factor
        self.residual_dense = Conv1d(input_size, hidden_size, 1)
        self.conv = nn.ModuleList([
            Conv1d(input_size, hidden_size, 3, dilation=1, padding=1),
            Conv1d(hidden_size, hidden_size, 3, dilation=2, padding=2),
            Conv1d(hidden_size, hidden_size, 3, dilation=4, padding=4),
        ])

    def forward(self, x):
        size = x.shape[-1] // self.factor

        residual = self.residual_dense(x)
        residual = F.interpolate(residual, size=size)

        x = F.interpolate(x, size=size)
        for layer in self.conv:
            x = F.leaky_relu(x, 0.2)
            x = layer(x)

        return x + residual


class WaveGrad(nn.Module):
    def __init__(self, params):
        super().__init__()
        self.params = params
        self.downsample = nn.ModuleList([
            Conv1d(1, 32, 5, padding=2),
            DBlock(32, 128, 2),
            DBlock(128, 128, 2),
            DBlock(128, 256, 3),
            DBlock(256, 512, 5),
        ])
        self.film = nn.ModuleList([
            FiLM(32, 128, params.num_classes, params.embedding_dim),
            FiLM(128, 128, params.num_classes, params.embedding_dim),
            FiLM(128, 256, params.num_classes, params.embedding_dim),
            FiLM(256, 512, params.num_classes, params.embedding_dim),
            FiLM(512, 512, params.num_classes, params.embedding_dim),
        ])
        self.upsample = nn.ModuleList([
            UBlock(768, 512, 5, [1, 2, 1, 2]),
            UBlock(512, 512, 5, [1, 2, 1, 2]),
            UBlock(512, 256, 3, [1, 2, 4, 8]),
            UBlock(256, 128, 2, [1, 2, 4, 8]),
            UBlock(128, 128, 2, [1, 2, 4, 8]),
        ])
        self.first_conv = Conv1d(128, 768, 3, padding=1)
        self.last_conv = Conv1d(128, 1, 3, padding=1)

    def forward(self, audio, spectrogram, noise_scale, labels):
        x = audio.unsqueeze(1)
        downsampled = []
        for film, layer in zip(self.film, self.downsample):
            x = layer(x)
            downsampled.append(film(x, noise_scale, labels))

        x = self.first_conv(spectrogram)
        for layer, (film_shift, film_scale) in zip(self.upsample, reversed(downsampled)):
            x = layer(x, film_shift, film_scale)
        x = self.last_conv(x)
        return x

    def predict(self, spectrogram, label):
        device = next(self.parameters()).device
        with torch.no_grad():

            beta = np.array(self.params.noise_schedule)
            alpha = 1 - beta
            alpha_cum = np.cumprod(alpha)

            if len(spectrogram.shape) == 2:  # Expand rank 2 tensors by adding a batch dimension.
                spectrogram = spectrogram.unsqueeze(0)
            spectrogram = spectrogram.to(device)

            audio = torch.randn(spectrogram.shape[0], self.params.hop_samples * spectrogram.shape[-1], device=device)
            noise_scale = torch.from_numpy(alpha_cum**0.5).float().unsqueeze(1).to(device)

            for n in range(len(alpha) - 1, -1, -1):
                c1 = 1 / alpha[n]**0.5
                c2 = (1 - alpha[n]) / (1 - alpha_cum[n])**0.5
                audio = c1 * (audio - c2 * self(audio, spectrogram, noise_scale[n], label).squeeze(1))
                if n > 0:
                    noise = torch.randn_like(audio)
                    sigma = ((1.0 - alpha_cum[n-1]) / (1.0 - alpha_cum[n]) * beta[n])**0.5
                    audio += sigma * noise
                audio = torch.clamp(audio, -1.0, 1.0)

        return audio, self.params.sample_rate


class WaveGradLearner(Learner):

    def __init__(self, *args, **kwargs):

        super().__init__(*args, **kwargs)

        beta = np.array(self.params.noise_schedule)
        noise_level = np.cumprod(1 - beta)**0.5
        noise_level = np.concatenate([[1.0], noise_level], axis=0)
        self.noise_level = torch.tensor(noise_level.astype(np.float32))

        self.model_name = 'WaveGrad'

    def predict(self, spectrogram, label):
        return self.model.predict(spectrogram=spectrogram, label=label)

    def train_step(self, features, train=True):
        for param in self.model.parameters():
            param.grad = None

        ref_audio = features['ref_audio']
        con_spec = features['con_spec']
        label = features['label']

        N, _ = ref_audio.shape
        S = 1000
        device = ref_audio.device
        self.noise_level = self.noise_level.to(device)

        with self.autocast:
            s = torch.randint(1, S + 1, [N], device=ref_audio.device)
            l_a, l_b = self.noise_level[s-1], self.noise_level[s]
            noise_scale = l_a + torch.rand(N, device=ref_audio.device) * (l_b - l_a)
            noise_scale = noise_scale.unsqueeze(1)
            noise = torch.randn_like(ref_audio)
            noisy_audio = noise_scale * ref_audio + (1.0 - noise_scale**2)**0.5 * noise

            predicted = self.model(noisy_audio, con_spec, noise_scale.squeeze(1), label)
            loss = self.loss_fn(noise, predicted.squeeze(1))

        if train:
            self.scaler.scale(loss).backward()  # type: ignore
            self.scaler.unscale_(self.optimizer)
            self.grad_norm = nn.utils.clip_grad_norm_(  # type: ignore
                self.model.parameters(), self.params.max_grad_norm)
            self.scaler.step(self.optimizer)
            self.scaler.update()

        return loss
