import librosa
import numpy as np
import matplotlib.pyplot as plt
import pacmap

from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler


from utils.utils import (
    squeeze,
    ensure_numpy,
    first_dict_value,
    make_ax_invisible,
)

from processing.filtering import create_spectrogram


label_colour = {
    'Normal': 'black',
    'Benign': 'blue',
    'MVP': 'red',
    'Aortic disease': 'pink',
    'Miscell': 'green',
}

label_marker = {
    'Normal': '$N$',
    'Benign': '$B$',
    'MVP': '$V$',
    'AD': '$A$',
    'MPC': '$C$',
}


def get_label_colour(label):

    if label.startswith('ref'):
        return 'red'
    elif label.startswith('gen'):
        return 'blue'
    elif label.startswith('con'):
        return 'green'
    else:
        return 'black'


def get_label_marker(label):
    for class_label, marker in label_marker.items():
        if class_label in label:
            return marker
    return '$Z$'


def plot_pacmap(axes, split, ref_features, gen_features, ref_labels, gen_labels):

    ref_features = [ref.flatten() for ref in ref_features]
    gen_features = [gen.flatten() for gen in gen_features]

    all_features = np.concatenate([ref_features, gen_features], axis=0)
    all_labels = np.concatenate([ref_labels, gen_labels], axis=0)

    scaler = MinMaxScaler((-1, 1))

    reduced_features = pacmap.PaCMAP(n_components=2, n_neighbors=2).fit_transform(all_features)
    reduced_features = scaler.fit_transform(reduced_features)
    assert reduced_features is not None, f'{reduced_features=}'

    for idx, (x, y) in enumerate(reduced_features):
        axes[0].scatter(
            x, y,
            c=get_label_colour(all_labels[idx]),
            label=all_labels[idx],
            marker=get_label_marker(all_labels[idx]),
        )

    axes[0].set_title(f'PaCMAP Visualisation for {split.capitalize()}')
    axes[0].set_xlabel('Component 1')
    axes[0].set_ylabel('Component 2')

    reduced_features = PCA(2).fit_transform(all_features)
    reduced_features = scaler.fit_transform(reduced_features)
    assert reduced_features is not None, f'{reduced_features=}'

    for idx, (x, y) in enumerate(reduced_features):
        axes[1].scatter(
            x, y,
            c=get_label_colour(all_labels[idx]),
            label=all_labels[idx],
            marker=get_label_marker(all_labels[idx]),
        )

    axes[1].set_title(f'PCA Visualisation for {split.capitalize()}')
    axes[1].set_xlabel('Component 1')
    axes[1].set_ylabel('Component 2')


def plot_spec(ax, audio, transform, cmap='jet'):
    ax.imshow(
        create_spectrogram(audio, transform),
        cmap=cmap,
        interpolation='none',
        aspect='auto',
        origin='lower',
    )
    ax.set_xlabel('Mel Frame')
    ax.set_ylabel('Mel Bin')


def plot_description(ax, ref_sig_name, gen_sig_name, con_sig_name, patient, diagnosis, model_name):
    descriptions = [
        f'Reference={ref_sig_name}',
        f'Generated={gen_sig_name}',
        f'Condition={con_sig_name}',
        f'Diagnosis={diagnosis}',
        f'Patient={patient}',
        f'Model={model_name}'
    ]

    max_left_length = max([desc.index('=') for desc in descriptions])
    max_right_length = max([len(desc) - desc.index('=') for desc in descriptions])

    aligned_descriptions = [''
                            + ' '*(max_left_length - len(desc.split('=')[0]))
                            + desc.split('=')[0]
                            + ' = '
                            + desc.split('=')[1]
                            + ' '*(max_right_length - len(desc.split('=')[1]))
                            for desc in descriptions]

    final_description = "\n".join(aligned_descriptions)

    ax.clear()

    ax.axis('off')

    ax.text(0.5, 0.5, final_description,
            transform=ax.transAxes,
            fontsize=14,
            verticalalignment='center',
            horizontalalignment='center',
            family='monospace')


def plot_psd(ax, signal, sr=1000):
    signal = squeeze(signal)
    ax.psd(signal, NFFT=1024, Fs=sr, linewidth=0.5, color='red')
    ax.set_yticks(np.arange(-150, 25, 10))
    ax.set_ylim(-155, 5)
    ax.set_xlim(0, sr//2)
    ax.set_xticks(np.arange(0, sr//2 + 1, sr // 10))
    ax.set_ylabel('Frequency (Hz)')
    ax.grid(True, which='both', linestyle='--', linewidth=0.5)


def plot_wav(ax, signal, sr=1000):

    t = np.linspace(0, len(signal) / sr, len(signal))

    ax.plot(t, signal, linewidth=0.5, color='blue')

    ax.set_xticks(np.arange(0, t[-1]+1, 1))
    ax.set_xlim(t[0], t[-1])

    ax.set_yticks(np.arange(-1, 1.1, 0.5))
    ax.set_ylim(-1, 1)

    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Amplitude (Normalised)')

    ax.grid(True, which='both', linestyle='--', linewidth=0.5)


def plot_stft(ax, signal, sr=1000, hop_length=50, n_fft=100):
    signal = ensure_numpy(signal)

    y = signal[:]
    S = librosa.stft(y=y[0:], center=False, n_fft=n_fft, hop_length=hop_length)

    S_dB = librosa.amplitude_to_db(np.abs(S))

    librosa.display.specshow(S_dB, sr=sr, ax=ax, n_fft=n_fft, hop_length=hop_length,
                             cmap=plt.cm.get_cmap('jet'), shading='gouraud',
                             x_axis='s', y_axis='linear')

    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Frequency (Hz)')
    ax.set_title('STFT Spectrogram')

    current_xlim = ax.get_xlim()
    xticks = [n//sr for n in range(0, len(signal)+1, sr)]
    xlabels = [str(n) for n in xticks]

    ax.set_xticks(xticks)
    ax.set_xticklabels(xlabels)
    ax.set_xlim(current_xlim)

    freq_ticks = np.linspace(0, sr/2, 6)
    ax.set_yticks(freq_ticks)
    ax.set_yticklabels([f'{int(freq)}' for freq in freq_ticks])


def plot_multi_signals_HACKY(signal_groups, title=None, sr=1000, plot_funcs=None):
    if plot_funcs is None:
        plot_funcs = [plot_wav, plot_stft]

    num_signals = len(first_dict_value(signal_groups)) // 2
    num_groups = len(signal_groups)
    num_funcs = len(plot_funcs)

    fig = plt.figure(constrained_layout=True)
    subfigs = fig.subfigures(1, num_groups, squeeze=False)

    for group_idx, (group_name, signals) in enumerate(signal_groups.items()):
        axes = subfigs[0][group_idx].subplots(num_signals, num_funcs, squeeze=False, sharex=True)
        for signal_idx, (signal_name, signal) in enumerate(signals.items()):
            if signal_idx == num_signals:
                break
            for func_idx, plot_func in enumerate(plot_funcs):
                if func_idx == 1 and signal_name == 'MSEGS':
                    signal = signals[f'{signal_name}+CHIRP']

                ax = axes[signal_idx][func_idx]
                plot_func(ax=ax, signal=signal, sr=sr)
                make_ax_invisible(ax)

                if func_idx == group_idx == 0:
                    ax.text(-0.25, 0.5, signal_name, size='large', ha='center',
                            rotation=90, va='center', transform=ax.transAxes)

        subfigs[0][group_idx].suptitle(group_name)

    if title:
        fig.suptitle(title, fontsize='x-large')

    fig.show()


def plot_multi_signals(signal_groups, title=None, sr=1000, plot_funcs=None):
    if plot_funcs is None:
        plot_funcs = [plot_wav, plot_stft]

    num_signals = len(first_dict_value(signal_groups))
    num_groups = len(signal_groups)
    num_funcs = len(plot_funcs)

    fig = plt.figure(constrained_layout=True)
    subfigs = fig.subfigures(1, num_groups, squeeze=False)

    for group_idx, (group_name, signals) in enumerate(signal_groups.items()):
        axes = subfigs[0][group_idx].subplots(num_signals, num_funcs, squeeze=False, sharex=True)
        for signal_idx, (signal_name, signal) in enumerate(signals.items()):
            for func_idx, plot_func in enumerate(plot_funcs):
                ax = axes[signal_idx][func_idx]
                plot_func(ax=ax, signal=signal, sr=sr)
                make_ax_invisible(ax)

                if func_idx == group_idx == 0:
                    ax.text(-0.25, 0.5, signal_name, size='large', ha='center',
                            rotation=90, va='center', transform=ax.transAxes)

        subfigs[0][group_idx].suptitle(group_name)

    if title:
        fig.suptitle(title, fontsize='x-large')

    fig.show()
