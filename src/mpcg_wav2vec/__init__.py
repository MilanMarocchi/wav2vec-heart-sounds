"""mPCG Wav2Vec: synthetic augmentation and heart-sound classification.

Subpackages:
    signalproc  filtering, normalisation, despiking, resampling, segmentation
    augment     traditional waveform augmentation
    datasets    CinC / vest / generative dataset loaders, labels, training schedules
    generative  DiffWave and WaveGrad diffusion generators
    classify    Wav2Vec 2.0 classifiers, training, evaluation, metrics
    experiments ablation runners for the paper's three settings
"""

__version__ = "0.1.0"

__all__ = ["signalproc", "augment", "datasets", "generative", "classify", "experiments", "config"]
