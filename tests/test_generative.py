import numpy as np
import torch

from mpcg_wav2vec.datasets.generative import GenerativeDataset, GenRecord
from mpcg_wav2vec.generative import GenerativeTrainer, get_spec


def _batch(hop, n_mels, frames=8, batch=2):
    return {
        "ref_audio": torch.randn(batch, hop * frames),
        "con_spec": torch.rand(batch, n_mels, frames),
        "label": torch.randint(0, 2, (batch,)),
    }


def test_diffwave_train_and_sample():
    spec = get_spec("diffwave")
    model = spec.build_model(2)
    device = torch.device("cpu")
    loss = spec.loss(model, _batch(256, 80), device)
    assert torch.isfinite(loss)

    opt = torch.optim.Adam(model.parameters(), lr=1e-4)
    trainer = GenerativeTrainer(model, opt, spec.loss, device, "/tmp/mpcg_diffwave")
    assert np.isfinite(trainer.train_step(_batch(256, 80)))

    audio, sr = spec.sample(model, torch.rand(1, 80, 4), torch.tensor([1]), fast=True)
    assert sr == 4000 and audio.shape == (1, 256 * 4)


def test_wavegrad_train_and_sample():
    spec = get_spec("wavegrad")
    model = spec.build_model(2)
    device = torch.device("cpu")
    assert torch.isfinite(spec.loss(model, _batch(300, 128), device))

    audio, sr = spec.sample(model, torch.rand(1, 128, 4), torch.tensor([1]), num_steps=5)
    assert sr == 4000 and audio.shape == (1, 300 * 4)


def test_loss_from_generative_dataset(tmp_path):
    """Drive the real path: a raw waveform yields a centred-STFT mel (crop_frames+1 frames),
    which must still train without a length mismatch."""
    device = torch.device("cpu")
    for name in ("diffwave", "wavegrad"):
        spec = get_spec(name)
        recs = [GenRecord(reference=np.random.randn(spec.hop_length * spec.crop_frames),
                          conditioning=np.random.randn(spec.hop_length * spec.crop_frames),
                          label=i % 2, patient=f"p{i}") for i in range(2)]
        ds = GenerativeDataset(recs, fs=spec.sample_rate, mel=spec.mel("pcg"),
                               crop_frames=spec.crop_frames, hop_length=spec.hop_length)
        assert ds[0]["con_spec"].shape[-1] == spec.crop_frames
        model = spec.build_model(2)
        batch = {k: torch.stack([ds[0][k], ds[1][k]]) if torch.is_tensor(ds[0][k])
                 else torch.tensor([ds[0][k], ds[1][k]]) for k in ("ref_audio", "con_spec", "label")}
        assert torch.isfinite(spec.loss(model, batch, device))


def test_checkpoint_roundtrip(tmp_path):
    spec = get_spec("diffwave")
    model = spec.build_model(2)
    opt = torch.optim.Adam(model.parameters(), lr=1e-4)
    trainer = GenerativeTrainer(model, opt, spec.loss, torch.device("cpu"), str(tmp_path))
    trainer.step = 7
    path = trainer.save("weights")
    trainer.step = 0
    assert trainer.restore(path) and trainer.step == 7
