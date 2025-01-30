import os

import torchaudio
import pandas as pd

from tqdm.auto import tqdm
from torch.utils.data import Dataset, DataLoader

from processing.datasets import (
    get_possible_labels,
    get_label_index,
    get_index_label,
)

from utils.reproducible import (
    torch_gen,
    seed_worker
)

from processing.filtering import create_spectrogram, standardise_torch_signal

from processing.records import save_signals

from utils.utils import ensure_numpy


class IcentiaDataset(Dataset):

    def __init__(self, base_dir, dataset, fs, transform):

        self.base_dir = base_dir
        self.fs = fs
        self.dataset = dataset
        self._transform = transform
        self.annotations = self._get_annotations()
        self.patient_data = self._get_patient_data()
        self._count = len(self.patient_data)

    def _get_annotations(self):
        return pd.read_csv(os.path.join(self.base_dir, 'records_info.csv'))

    def _get_patient_data(self):

        patient_data = []

        for _, annotation in tqdm(self.annotations.iterrows(),
                                  total=len(self.annotations),
                                  desc='Reading processed icentia files'):

            patient = annotation.record_name
            diagnosis = annotation.pcg_class

            assert diagnosis in get_possible_labels(self.dataset), f'{diagnosis=}'
            info = self._get_patient_info(patient, diagnosis)

            assert info is not None
            patient_data.append(info)

        return patient_data

    def _get_patient_info(self, patient, diagnosis):

        label = get_label_index(self.dataset, diagnosis)
        input_path = os.path.join(self.base_dir, diagnosis, f'{patient}_ECG_FILT.wav')

        con_sig, con_sr = torchaudio.load(input_path)  # type: ignore
        con_sig = con_sig[0]

        con_spec = create_spectrogram(con_sig, self._transform)

        assert con_sr == self.fs, f'{con_sr=}, {self.fs=}'

        return dict(
            label=label,
            diagnosis=diagnosis,
            patient=patient,
            con_sig=con_sig,
            con_spec=con_spec,
        )

    def __len__(self):
        return self._count

    def __getitem__(self, index):

        return self.patient_data[index].copy()


class GenerativeIcentiaDataset(Dataset):

    def __init__(self, base_dir, dataset, fs, transform_con, transform_gen):

        self.base_dir = base_dir
        self.fs = fs
        self.dataset = dataset
        self._transform_con = transform_con
        self._transform_gen = transform_gen
        self.patient_data = self._get_patient_data()
        self._count = len(self.patient_data)

    def _get_annotations(self):
        return pd.read_csv(os.path.join(self.base_dir, 'records_info.csv'))

    def _get_patient_data(self):

        patient_data = []

        categories = os.listdir(self.base_dir)
        paths = [os.path.join(self.base_dir, cat) for cat in categories]

        for diagnosis, diag_fold in tqdm(zip(categories, paths), desc='Going through folders', position=1):
            assert diagnosis in get_possible_labels(self.dataset), f'{diagnosis=}'
            for file in tqdm(os.listdir(diag_fold), desc='Reading through patients', position=2, leave=False):
                if file.endswith('_ECG_CON.wav'):
                    patient = file.removesuffix('_ECG_CON.wav')

                else:
                    continue
                info = self._get_patient_info(patient, diagnosis)

                assert info is not None
                patient_data.append(info)

        return patient_data

    def _get_patient_info(self, patient, diagnosis):

        label = get_label_index(self.dataset, diagnosis)
        ecg_path = os.path.join(self.base_dir, diagnosis, f'{patient}_ECG_CON.wav')
        pcg_path = os.path.join(self.base_dir, diagnosis, f'{patient}_PCG_GEN.wav')

        gen_sig, gen_sr = torchaudio.load(pcg_path)  # type: ignore
        gen_sig = standardise_torch_signal(gen_sig[0])

        con_sig, con_sr = torchaudio.load(ecg_path)  # type: ignore
        con_sig = standardise_torch_signal(con_sig[0])

        con_spec = create_spectrogram(con_sig, self._transform_con)
        gen_spec = create_spectrogram(gen_sig, self._transform_gen)

        assert gen_sr == con_sr == self.fs, f'{gen_sr=}, {con_sr=}, {self.fs=}'

        return dict(
            label=label,
            diagnosis=diagnosis,
            patient=patient,
            con_sig=con_sig,
            con_spec=con_spec,
            ecg_sig=con_sig,
            ecg_spec=con_spec,
            gen_sig=gen_sig,
            gen_spec=gen_spec,
            pcg_sig=gen_sig,
            pcg_spec=gen_spec,
        )

    def __len__(self):
        return self._count

    def __getitem__(self, index):

        return self.patient_data[index].copy()


def get_dataloader(base_dir, dataset, fs, transform, batch_size=8):

    dataset = IcentiaDataset(base_dir=base_dir, dataset=dataset, fs=fs, transform=transform)
    dataloader = DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        drop_last=False,
        shuffle=False,
        num_workers=8,
        worker_init_fn=seed_worker,
        generator=torch_gen,
    )

    return dataloader


def generate_icentia_data(model, dataset, icentia_dataloader, output_dir):

    device = next(model.parameters()).device

    for data in tqdm(icentia_dataloader):
        con_sigs, labels, con_specs, patients = data['con_sig'], data['label'], data['con_spec'], data['patient']

        gen_sigs, gen_sr = model.predict(
            spectrogram=con_specs.to(device),
            label=labels.to(device),
        )

        collection = zip(con_sigs, gen_sigs, patients, labels)

        assert gen_sr == 4000, f'{gen_sr=}'

        for con_sig, gen_sig, patient, label in collection:

            diagnosis = get_index_label(dataset, label)

            save_generated(con_sig, gen_sig, patient, output_dir, diagnosis, gen_sr)


def save_generated(con_sig, gen_sig, record_name, output_dir, diagnosis, sr):

    output_subdir = os.path.join(output_dir, diagnosis)
    os.makedirs(output_subdir, exist_ok=True)
    # gen_output_path = os.path.join(output_subdir, f'{record_name}_PCG_GEN.wav')
    # con_output_path = os.path.join(output_subdir, f'{record_name}_ECG_CON.wav')

    gen_sig = gen_sig[:con_sig.shape[0]]

    gen_sig = ensure_numpy(gen_sig.cpu())
    con_sig = ensure_numpy(con_sig.cpu())

    # gen_sig = torch.stack([gen_sig.cpu()], dim=0)
    # con_sig = torch.stack([con_sig.cpu()], dim=0)

    # torchaudio.save(gen_output_path, gen_sig, sr)  # type: ignore
    # torchaudio.save(con_output_path, con_sig, sr)  # type: ignore

    save_signals({'ECG': con_sig, 'PCG': gen_sig},
                 record_name,
                 output_subdir,
                 sr)
