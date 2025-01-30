import processing.filtering as filtering
import generative.diffwave as diffwave
import generative.wavegrad as wavegrad


def get_transform(generative_model, signal_name):
    params = get_params(generative_model)
    if generative_model == 'DiffWave':
        if signal_name == 'ecg':
            return diffwave.diffwave_mel_spec_init_ecg(params)
        if signal_name == 'pcg':
            return diffwave.diffwave_mel_spec_init_pcg(params)
    if generative_model == 'WaveGrad':
        if signal_name == 'ecg':
            return wavegrad.wavegrad_mel_spec_init_ecg(params)
        if signal_name == 'pcg':
            return wavegrad.wavegrad_mel_spec_init_pcg(params)
    raise NotImplementedError(f'No transform found for {generative_model} + {signal_name}')


def get_params(generative_model):
    if generative_model == 'DiffWave':
        return diffwave.diffwave_params
    if generative_model == 'WaveGrad':
        return wavegrad.wavegrad_params
    raise NotImplementedError(f'No params found for {generative_model}')


def get_generative_model(generative_model):
    if generative_model == 'DiffWave':
        return diffwave.DiffWave
    if generative_model == 'WaveGrad':
        return wavegrad.WaveGrad
    raise NotImplementedError(f'Generative model {generative_model} not supported')


def get_learner(generative_model):
    if generative_model == 'DiffWave':
        return diffwave.DiffWaveLearner
    if generative_model == 'WaveGrad':
        return wavegrad.WaveGradLearner
    raise NotImplementedError(f'Generative model {generative_model} not supported')


def get_post_transform(signal_name):
    if signal_name == 'ecg':
        return filtering.post_filter_ecg
    if signal_name == 'pcg':
        return filtering.post_filter_pcg
    raise NotImplementedError(f'No post-transform filter found for {signal_name}')
