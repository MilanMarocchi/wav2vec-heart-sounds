import torch
import numpy as np


class AttrDict(dict):
    def __init__(self, *args, **kwargs):
        super(AttrDict, self).__init__(*args, **kwargs)
        self.__dict__ = self

    def override(self, attrs):
        if isinstance(attrs, dict):
            self.__dict__.update(**attrs)
        elif isinstance(attrs, (list, tuple, set)):
            for attr in attrs:
                self.override(attr)
        elif attrs is not None:
            raise NotImplementedError
        return self


def squeeze(data):
    if isinstance(data, np.ndarray):
        return np.squeeze(data)
    elif isinstance(data, torch.Tensor):
        return data.squeeze()
    else:
        raise ValueError(f'Data must be either a numpy array or a torch tensor, not {type(data)=}')


def ensure_numpy(data):
    if isinstance(data, np.ndarray):
        return data
    elif isinstance(data, torch.Tensor):
        return data.numpy()
    else:
        raise ValueError(f'Data must be either a numpy array or a torch tensor, not {type(data)=}')


def pretty_print_shapes(data, depth=-1):
    if depth == 0:
        return type(data)
    if isinstance(data, (np.ndarray, torch.Tensor)):
        return type(data), data.shape
    elif isinstance(data, dict):
        return {k: pretty_print_shapes(v, depth-1) for k, v in data.items()}
    elif isinstance(data, (list, tuple)):
        return [pretty_print_shapes(item, depth-1) for item in data]
    else:
        return type(data)


def make_ax_invisible(ax):
    ax.set_xlabel('')
    ax.set_xticks([])
    ax.set_xticklabels([])
    ax.set_ylabel('')
    ax.set_yticks([])
    ax.set_yticklabels([])
    ax.set_title('')
    ax.patch.set_visible(False)
    ax.set_frame_on(False)


def first_dict_value(d):
    return next(iter(d.values()))


def add_beep(f=500, beep_duration=0.5, fs=1000, pause_duration=0.1, amplitude=0.25, num_beeps=1):
    beep = amplitude * np.sin(2*np.pi*f*np.linspace(0, beep_duration, round(fs*beep_duration), endpoint=False))
    pause = np.zeros(round(fs * pause_duration))
    beep_with_pause = np.concatenate([beep, pause])
    multi_beep = np.tile(beep_with_pause, num_beeps)
    return multi_beep
