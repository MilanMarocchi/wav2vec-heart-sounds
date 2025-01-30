import tkinter as tk
from tkinter import ttk
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import librosa
import librosa.display
import numpy as np
from processing.filtering import standardise_signal
from sounddevice import play, stop
from utils.tickscale import TickScale


def get_patient_data(data):

    patient_data = []

    for dataset in data['datasets']:
        for patient in data['datasets'][dataset]:

            patient_data.append(dict(
                patient_name=patient['patient'],
                diagnosis=patient['diagnosis'],
                audio=np.concatenate(patient['pcg_segs'][:8], axis=0, dtype=float),
            ))

    return patient_data


class HPSSApp():
    def __init__(self, patient_data, master=None):
        if master is None:
            master = tk.Tk()
        self.master = master
        master.title('HPSS Parameter Adjustment')
        master.protocol("WM_DELETE_WINDOW", lambda: stop())
        master.bind('<Escape>', lambda _: self.destroy())
        master.bind('s', lambda _: stop())
        self.patient_data = patient_data

        self.num_groups = 6
        self._some_attrs = [
            'n_fft_1',
            'hop_length_1',
            'win_length_1',
            'margin_harmonic_1',
            'margin_percussive_1',
            'percus_kernel_1',
            'harmon_kernel_1',

            'n_fft_2',
            'hop_length_2',
            'win_length_2',
            'margin_harmonic_2',
            'margin_percussive_2',
            'percus_kernel_2',
            'harmon_kernel_2',

            'n_fft_display',
            'hop_length_display',
            'win_length_display',

            'mix_back',
            'mix_murm',
            'mix_tran',
        ]

        # A

        self.n_fft_1_a = tk.IntVar(value=2048)
        self.hop_length_1_a = tk.IntVar(value=64)
        self.win_length_1_a = tk.IntVar(value=256)
        self.margin_harmonic_1_a = tk.DoubleVar(value=1.0)
        self.margin_percussive_1_a = tk.DoubleVar(value=1.0)
        self.percus_kernel_1_a = tk.IntVar(value=37)
        self.harmon_kernel_1_a = tk.IntVar(value=37)

        self.n_fft_2_a = tk.IntVar(value=512)
        self.hop_length_2_a = tk.IntVar(value=128)
        self.win_length_2_a = tk.IntVar(value=512)
        self.margin_harmonic_2_a = tk.DoubleVar(value=1.0)
        self.margin_percussive_2_a = tk.DoubleVar(value=1.0)
        self.percus_kernel_2_a = tk.IntVar(value=25)
        self.harmon_kernel_2_a = tk.IntVar(value=25)

        self.n_fft_display_a = tk.IntVar(value=2048)
        self.hop_length_display_a = tk.IntVar(value=64)
        self.win_length_display_a = tk.IntVar(value=512)

        self.mix_back_a = tk.DoubleVar(value=0.2)
        self.mix_murm_a = tk.DoubleVar(value=2)
        self.mix_tran_a = tk.DoubleVar(value=1)

        # B

        self.n_fft_1_b = tk.IntVar(value=2048)
        self.hop_length_1_b = tk.IntVar(value=64)
        self.win_length_1_b = tk.IntVar(value=256)
        self.margin_harmonic_1_b = tk.DoubleVar(value=1.0)
        self.margin_percussive_1_b = tk.DoubleVar(value=1.0)
        self.percus_kernel_1_b = tk.IntVar(value=24)
        self.harmon_kernel_1_b = tk.IntVar(value=24)

        self.n_fft_2_b = tk.IntVar(value=512)
        self.hop_length_2_b = tk.IntVar(value=128)
        self.win_length_2_b = tk.IntVar(value=512)
        self.margin_harmonic_2_b = tk.DoubleVar(value=1.0)
        self.margin_percussive_2_b = tk.DoubleVar(value=1.0)
        self.percus_kernel_2_b = tk.IntVar(value=21)
        self.harmon_kernel_2_b = tk.IntVar(value=21)

        self.n_fft_display_b = tk.IntVar(value=2048)
        self.hop_length_display_b = tk.IntVar(value=64)
        self.win_length_display_b = tk.IntVar(value=512)

        self.mix_back_b = tk.DoubleVar(value=0.2)
        self.mix_murm_b = tk.DoubleVar(value=2)
        self.mix_tran_b = tk.DoubleVar(value=1)

        # FRAMES

        self.left_frame = ttk.Frame(master)
        self.left_frame.pack(side=tk.LEFT, fill=tk.Y)
        self.plot_frame = ttk.Frame(master)
        self.plot_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        self.right_frame = ttk.Frame(master)
        self.right_frame.pack(side=tk.RIGHT, fill=tk.Y)

        self.figure, self.axes = plt.subplots(self.num_groups, 4, figsize=(6, 6), constrained_layout=True)
        self.canvas = FigureCanvasTkAgg(self.figure, self.plot_frame)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        # PATIENT A

        self.stage_1_a_frame = tk.LabelFrame(self.left_frame, text='A stage 1')
        self.stage_1_a_frame.pack(fill='x', expand=True)

        TickScale(self.stage_1_a_frame, from_=128, to=4096, resolution=128, label='n_fft',
                  variable=self.n_fft_1_a, orient=tk.HORIZONTAL).pack(fill='x')
        TickScale(self.stage_1_a_frame, from_=16, to=1024, resolution=16, label='hop_length',
                  variable=self.hop_length_1_a, orient=tk.HORIZONTAL).pack(fill='x')
        TickScale(self.stage_1_a_frame, from_=128, to=4096, resolution=128, label='win_length',
                  variable=self.win_length_1_a, orient=tk.HORIZONTAL).pack(fill='x')
        TickScale(self.stage_1_a_frame, from_=1.0, to=5.0, resolution=0.1, label='harm_marg',
                  variable=self.margin_harmonic_1_a, orient=tk.HORIZONTAL).pack(fill='x')
        TickScale(self.stage_1_a_frame, from_=1.0, to=3.0, resolution=0.1, label='marg_perc',
                  variable=self.margin_percussive_1_a, orient=tk.HORIZONTAL).pack(fill='x')
        TickScale(self.stage_1_a_frame, from_=1, to=50, resolution=1, label='harmon_kernel',
                  variable=self.harmon_kernel_1_a, orient=tk.HORIZONTAL).pack(fill='x')
        TickScale(self.stage_1_a_frame, from_=1, to=50, resolution=1, label='percus_kernel',
                  variable=self.percus_kernel_1_a, orient=tk.HORIZONTAL).pack(fill='x')

        self.stage_2_a_frame = tk.LabelFrame(self.left_frame, text='A stage 2')
        self.stage_2_a_frame.pack(fill='x', expand=True)

        TickScale(self.stage_2_a_frame, from_=128, to=4096, resolution=128, label='n_fft',
                  variable=self.n_fft_2_a, orient=tk.HORIZONTAL).pack(fill='x')
        TickScale(self.stage_2_a_frame, from_=16, to=1024, resolution=16, label='hop_length',
                  variable=self.hop_length_2_a, orient=tk.HORIZONTAL).pack(fill='x')
        TickScale(self.stage_2_a_frame, from_=128, to=4096, resolution=128, label='win_length',
                  variable=self.win_length_2_a, orient=tk.HORIZONTAL).pack(fill='x')
        TickScale(self.stage_2_a_frame, from_=1.0, to=5.0, resolution=0.1, label='harm_marg',
                  variable=self.margin_harmonic_2_a, orient=tk.HORIZONTAL).pack(fill='x')
        TickScale(self.stage_2_a_frame, from_=1.0, to=3.0, resolution=0.1, label='marg_perc',
                  variable=self.margin_percussive_2_a, orient=tk.HORIZONTAL).pack(fill='x')
        TickScale(self.stage_2_a_frame, from_=1, to=50, resolution=1, label='harmon_kernel',
                  variable=self.harmon_kernel_2_a, orient=tk.HORIZONTAL).pack(fill='x')
        TickScale(self.stage_2_a_frame, from_=1, to=50, resolution=1, label='percus_kernel',
                  variable=self.percus_kernel_2_a, orient=tk.HORIZONTAL).pack(fill='x')

        self.stage_mix_a_frame = tk.LabelFrame(self.left_frame, text='A mix')
        self.stage_mix_a_frame.pack(fill='x', expand=True)

        TickScale(self.stage_mix_a_frame, from_=0.0, to=2, resolution=0.05, label='Background',
                  variable=self.mix_back_a, orient=tk.HORIZONTAL).pack(fill='x')
        TickScale(self.stage_mix_a_frame, from_=0.0, to=2, resolution=0.05, label='Murmur',
                  variable=self.mix_murm_a, orient=tk.HORIZONTAL).pack(fill='x')
        TickScale(self.stage_mix_a_frame, from_=0.0, to=2, resolution=0.05, label='Transient',
                  variable=self.mix_tran_a, orient=tk.HORIZONTAL).pack(fill='x')

        self.stage_display_a_frame = tk.LabelFrame(self.left_frame, text='A Display')
        self.stage_display_a_frame.pack(fill='x', expand=True)

        TickScale(self.stage_display_a_frame, from_=128, to=4096, resolution=128, label='n_fft',
                  variable=self.n_fft_display_a, orient=tk.HORIZONTAL).pack(fill='x')
        TickScale(self.stage_display_a_frame, from_=16, to=1024, resolution=16, label='hop_length',
                  variable=self.hop_length_display_a, orient=tk.HORIZONTAL).pack(fill='x')
        TickScale(self.stage_display_a_frame, from_=128, to=4096, resolution=128, label='win_length',
                  variable=self.win_length_display_a, orient=tk.HORIZONTAL).pack(fill='x')

        self.stage_controls_a_frame = tk.LabelFrame(self.left_frame, text='A Controls')
        self.stage_controls_a_frame.pack(fill='x', expand=True)
        self.selected_patient_a = tk.StringVar(master)

        self.patient_options_a = list(sorted([patient['patient_name'] for patient in self.patient_data]))
        self.selected_patient_a.set(self.patient_options_a[0])  # type: ignore

        self.patient_menu_a = ttk.Spinbox(
            self.stage_controls_a_frame,
            textvariable=self.selected_patient_a,
            values=self.patient_options_a  # type: ignore
        )
        self.patient_menu_a.pack()

        self.a_ind = tk.IntVar(value=0)
        TickScale(self.stage_controls_a_frame, label='Wav A',
                  variable=self.a_ind, from_=0, to=5, resolution=1).pack(fill='x')

        self.update_button_a = ttk.Button(self.stage_controls_a_frame, text="Update", command=self.update_plot)
        self.update_button_a.pack()

        self.a_normalise = tk.BooleanVar(value=True)
        ttk.Checkbutton(self.stage_controls_a_frame, text='Normalise A', variable=self.a_normalise).pack()
        self.play_button_A = ttk.Button(self.stage_controls_a_frame, text="Play A",
                                        command=lambda: play(standardise_signal(self.waveforms_a[self.a_ind.get()])
                                                             if self.a_normalise.get()
                                                             else self.waveforms_a[self.a_ind.get()], self.sr_a))
        self.play_button_A.pack()

        self.copy_button_a_to_b = ttk.Button(self.stage_controls_a_frame, text="Copy to B",
                                             command=lambda: self.copy_a_to_b())
        self.copy_button_a_to_b.pack()

        # PATIENT B

        self.stage_1_b_frame = tk.LabelFrame(self.right_frame, text='B stage 1')
        self.stage_1_b_frame.pack(fill='x', expand=True)

        TickScale(self.stage_1_b_frame, from_=128, to=4096, resolution=128, label='n_fft',
                  variable=self.n_fft_1_b, orient=tk.HORIZONTAL).pack(fill='x')
        TickScale(self.stage_1_b_frame, from_=16, to=1024, resolution=16, label='hop_length',
                  variable=self.hop_length_1_b, orient=tk.HORIZONTAL).pack(fill='x')
        TickScale(self.stage_1_b_frame, from_=128, to=4096, resolution=128, label='win_length',
                  variable=self.win_length_1_b, orient=tk.HORIZONTAL).pack(fill='x')
        TickScale(self.stage_1_b_frame, from_=1.0, to=3.0, resolution=0.1, label='harm_marg',
                  variable=self.margin_harmonic_1_b, orient=tk.HORIZONTAL).pack(fill='x')
        TickScale(self.stage_1_b_frame, from_=1.0, to=3.0, resolution=0.1, label='marg_perc',
                  variable=self.margin_percussive_1_b, orient=tk.HORIZONTAL).pack(fill='x')
        TickScale(self.stage_1_b_frame, from_=1, to=50, resolution=1, label='harmon_kernel',
                  variable=self.harmon_kernel_1_b, orient=tk.HORIZONTAL).pack(fill='x')
        TickScale(self.stage_1_b_frame, from_=1, to=50, resolution=1, label='percus_kernel',
                  variable=self.percus_kernel_1_b, orient=tk.HORIZONTAL).pack(fill='x')

        self.stage_2_b_frame = tk.LabelFrame(self.right_frame, text='B stage 2')
        self.stage_2_b_frame.pack(fill='x', expand=True)

        TickScale(self.stage_2_b_frame, from_=128, to=4096, resolution=128, label='n_fft',
                  variable=self.n_fft_2_b, orient=tk.HORIZONTAL).pack(fill='x')
        TickScale(self.stage_2_b_frame, from_=16, to=1024, resolution=16, label='hop_length',
                  variable=self.hop_length_2_b, orient=tk.HORIZONTAL).pack(fill='x')
        TickScale(self.stage_2_b_frame, from_=128, to=4096, resolution=128, label='win_length',
                  variable=self.win_length_2_b, orient=tk.HORIZONTAL).pack(fill='x')
        TickScale(self.stage_2_b_frame, from_=1.0, to=3.0, resolution=0.1, label='harm_marg',
                  variable=self.margin_harmonic_2_b, orient=tk.HORIZONTAL).pack(fill='x')
        TickScale(self.stage_2_b_frame, from_=1.0, to=3.0, resolution=0.1, label='marg_perc',
                  variable=self.margin_percussive_2_b, orient=tk.HORIZONTAL).pack(fill='x')
        TickScale(self.stage_2_b_frame, from_=1, to=50, resolution=1, label='harmon_kernel',
                  variable=self.harmon_kernel_2_b, orient=tk.HORIZONTAL).pack(fill='x')
        TickScale(self.stage_2_b_frame, from_=1, to=50, resolution=1, label='percus_kernel',
                  variable=self.percus_kernel_2_b, orient=tk.HORIZONTAL).pack(fill='x')

        self.stage_mix_b_frame = tk.LabelFrame(self.right_frame, text='B mix')
        self.stage_mix_b_frame.pack(fill='x', expand=True)

        TickScale(self.stage_mix_b_frame, from_=0.0, to=2, resolution=0.05, label='Background',
                  variable=self.mix_back_b, orient=tk.HORIZONTAL).pack(fill='x')
        TickScale(self.stage_mix_b_frame, from_=0.0, to=2, resolution=0.05, label='Murmur',
                  variable=self.mix_murm_b, orient=tk.HORIZONTAL).pack(fill='x')
        TickScale(self.stage_mix_b_frame, from_=0.0, to=2, resolution=0.05, label='Transient',
                  variable=self.mix_tran_b, orient=tk.HORIZONTAL).pack(fill='x')

        self.stage_display_b_frame = tk.LabelFrame(self.right_frame, text='B Display')
        self.stage_display_b_frame.pack(fill='x', expand=True)

        TickScale(self.stage_display_b_frame, from_=128, to=4096, resolution=128, label='n_fft',
                  variable=self.n_fft_display_b, orient=tk.HORIZONTAL).pack(fill='x')
        TickScale(self.stage_display_b_frame, from_=16, to=1024, resolution=16, label='hop_length',
                  variable=self.hop_length_display_b, orient=tk.HORIZONTAL).pack(fill='x')
        TickScale(self.stage_display_b_frame, from_=128, to=4096, resolution=128, label='win_length',
                  variable=self.win_length_display_b, orient=tk.HORIZONTAL).pack(fill='x')

        self.stage_controls_b_frame = tk.LabelFrame(self.right_frame, text='B Controls')
        self.stage_controls_b_frame.pack(fill='x', expand=True)

        self.selected_patient_b = tk.StringVar(master)
        self.patient_options_b = list(sorted([patient['patient_name'] for patient in self.patient_data]))
        self.selected_patient_b.set(self.patient_options_b[0])  # type: ignore

        self.patient_menu_b = ttk.Spinbox(
            self.stage_controls_b_frame,
            textvariable=self.selected_patient_b,
            values=self.patient_options_b  # type: ignore
        )
        self.patient_menu_b.pack()

        self.b_ind = tk.IntVar(value=0)
        TickScale(self.stage_controls_b_frame, label='Wav B',
                  variable=self.b_ind, from_=0, to=5, resolution=1).pack(fill='x')

        self.update_button_b = ttk.Button(self.stage_controls_b_frame, text="Update", command=self.update_plot)
        self.update_button_b.pack()

        self.b_normalise = tk.BooleanVar(value=True)
        ttk.Checkbutton(self.stage_controls_b_frame, text='Normalise B', variable=self.b_normalise).pack()
        self.play_button_B = ttk.Button(self.stage_controls_b_frame, text="Play B",
                                        command=lambda: play(standardise_signal(self.waveforms_b[self.b_ind.get()])
                                                             if self.b_normalise.get()
                                                             else self.waveforms_b[self.b_ind.get()], self.sr_b))
        self.play_button_B.pack()

        self.copy_button_b_to_a = ttk.Button(self.stage_controls_b_frame, text="Copy to A",
                                             command=lambda: self.copy_b_to_a())
        self.copy_button_b_to_a.pack()

        self.s = ttk.Style()
        self.s.theme_use('clam')

        self.update_plot()

    def destroy(self):
        print('Stopping audio')
        stop()
        print('Audio stopped')
        print('Quitting master')
        retval = self.master.quit()
        print(f'Master quat {retval=}')
        print('Destroying master')
        retval = self.master.destroy()
        print(f'Master destroyed {retval=}')
        return retval

    def copy_a_to_b(self):
        for attr in self._some_attrs:
            getattr(self, f'{attr}_b').set(getattr(self, f'{attr}_a').get())

    def copy_b_to_a(self):
        for attr in self._some_attrs:
            getattr(self, f'{attr}_a').set(getattr(self, f'{attr}_b').get())

    def update_plot(self, _=None):
        # Find the selected patient's audio data
        patient_info_b = next(
            (item for item in self.patient_data if item["patient_name"] == self.selected_patient_b.get()), None)

        def standardise_signal_a(sig):
            return sig if not self.a_normalise.get() else standardise_signal(sig)

        def standardise_signal_b(sig):
            return sig if not self.b_normalise.get() else standardise_signal(sig)

        # Clear current axes
        for ax in self.axes.flatten():
            ax.clear()

        if patient_info_b:
            self.y_b = patient_info_b['audio']
            self.sr_b = 4000  # Assuming a standard sample rate, modify if different

            self.y_b = standardise_signal(self.y_b)

            self.y_max_b = np.max(np.abs(self.y_b))

            self.D_b = librosa.stft(
                self.y_b,
                n_fft=self.n_fft_1_b.get(),
                hop_length=self.hop_length_1_b.get(),
                win_length=self.win_length_1_b.get(),
            )

            self.D_mag_dB_b = librosa.amplitude_to_db(np.abs(self.D_b), ref=self.y_max_b)
            self.vmin_b = self.D_mag_dB_b.min()
            self.vmax_b = self.D_mag_dB_b.max()

            self.background_b, self.residual_b = librosa.decompose.hpss(self.D_b,
                                                                        margin=(self.margin_harmonic_1_b.get(),
                                                                                self.margin_percussive_1_b.get()),
                                                                        kernel_size=(self.percus_kernel_1_b.get(),
                                                                                     self.harmon_kernel_1_b.get()),
                                                                        )

            self.y_background_b = librosa.istft(self.background_b,
                                                n_fft=self.n_fft_1_b.get(),
                                                hop_length=self.hop_length_1_b.get(),
                                                win_length=self.win_length_1_b.get(),
                                                )
            self.y_residual_b = librosa.istft(self.residual_b,
                                              n_fft=self.n_fft_1_b.get(),
                                              hop_length=self.hop_length_1_b.get(),
                                              win_length=self.win_length_1_b.get(),
                                              )

            self.D_residual_b = librosa.stft(self.y_residual_b,
                                             n_fft=self.n_fft_2_b.get(),
                                             hop_length=self.hop_length_2_b.get(),
                                             win_length=self.win_length_2_b.get(),
                                             )

            # Perform the second HPSS on the residual
            self.murmurs_b, self.heart_sounds_b = librosa.decompose.hpss(self.D_residual_b,
                                                                         margin=(self.margin_harmonic_2_b.get(),
                                                                                 self.margin_percussive_2_b.get()),
                                                                         kernel_size=(self.percus_kernel_2_b.get(),
                                                                                      self.harmon_kernel_2_b.get()),
                                                                         )

            self.y_murmurs_b = librosa.istft(self.murmurs_b,
                                             n_fft=self.n_fft_2_b.get(),
                                             hop_length=self.hop_length_2_b.get(),
                                             win_length=self.win_length_2_b.get(),
                                             )

            self.y_heart_sounds_b = librosa.istft(self.heart_sounds_b,
                                                  n_fft=self.n_fft_2_b.get(),
                                                  hop_length=self.hop_length_2_b.get(),
                                                  win_length=self.win_length_2_b.get(),
                                                  )

            min_len = min(len(y_i) for y_i in (self.y_background_b, self.y_murmurs_b, self.y_heart_sounds_b))

            self.y_enhanced_b = standardise_signal(
                self.mix_back_b.get() * self.y_background_b[:min_len]
                + self.mix_murm_b.get() * self.y_murmurs_b[:min_len]
                + self.mix_tran_b.get() * self.y_heart_sounds_b[:min_len]
            )

            self.waveforms_b = [self.y_b, self.y_background_b, self.y_residual_b,
                                self.y_murmurs_b, self.y_heart_sounds_b, self.y_enhanced_b]

            # combined_waveforms.extend(waveforms)

            for j, component in enumerate(self.waveforms_b):
                # Select the axis for the waveform and spectrogram
                ax_waveform_b = self.axes[j][2]
                ax_spectrogram_b = self.axes[j][3]

                # Waveform plot
                librosa.display.waveshow(standardise_signal_b(component), sr=self.sr_b, alpha=0.5, ax=ax_waveform_b,
                                         color=['red', 'green', 'blue', 'orange', 'purple', 'black'][j],
                                         linewidth=0.5, linestyle='--')
                ax_waveform_b.set_xlabel('Time (s)')
                ax_waveform_b.label_outer()  # Hide labels for shared axes

                # Spectrogram plot
                # if j < 5:  # Avoid recomputing STFT for the combined signal
                component_stft_b = librosa.stft(component,
                                                n_fft=self.n_fft_display_b.get(),
                                                hop_length=self.hop_length_display_b.get(),
                                                win_length=self.win_length_display_b.get(),
                                                )
                self.D_component_db = librosa.amplitude_to_db(np.abs(component_stft_b), ref=self.y_max_b)
                img_b = librosa.display.specshow(self.D_component_db, sr=self.sr_b, x_axis='time', y_axis='log',
                                                 ax=ax_spectrogram_b, cmap='jet',
                                                 n_fft=self.n_fft_display_b.get(),
                                                 hop_length=self.hop_length_display_b.get(),
                                                 win_length=self.win_length_display_b.get(),
                                                 vmin=self.vmin_b, vmax=self.vmax_b)
                ax_spectrogram_b.set_xlabel('Time (s)')
                ax_spectrogram_b.label_outer()  # Hide labels for shared axes

                # Colorbar is held by `cax`.
                cax_b = ax_spectrogram_b.inset_axes([1.03, 0, 0.1, 1], transform=ax_spectrogram_b.transAxes)
                cax_b.clear()
                _ = self.figure.colorbar(img_b,
                                         ax=ax_spectrogram_b,
                                         cax=cax_b,
                                         format='%+2.0f dB')

                # Ensure the plots share axes where appropriate
                if j not in [0]:
                    self.axes[j][2].sharey(self.axes[0][2])
                    self.axes[j][3].sharex(self.axes[0][3])

        patient_info_a = next(
            (item for item in self.patient_data if item["patient_name"] == self.selected_patient_a.get()), None)

        if patient_info_a:
            self.y_a = patient_info_a['audio']
            self.sr_a = 4000  # Assuming a standard sample rate, modify if different

            self.y_a = standardise_signal(self.y_a)

            self.y_max_a = np.max(np.abs(self.y_a))

            self.D_a = librosa.stft(
                self.y_a,
                n_fft=self.n_fft_1_a.get(),
                hop_length=self.hop_length_1_a.get(),
                win_length=self.win_length_1_a.get(),
            )

            self.D_mag_dB_a = librosa.amplitude_to_db(np.abs(self.D_a), ref=self.y_max_a)
            self.vmin_a = self.D_mag_dB_a.min()
            self.vmax_a = self.D_mag_dB_a.max()

            self.background_a, self.residual_a = librosa.decompose.hpss(self.D_a,
                                                                        margin=(self.margin_harmonic_1_a.get(),
                                                                                self.margin_percussive_1_a.get()),
                                                                        kernel_size=(self.percus_kernel_1_a.get(),
                                                                                     self.harmon_kernel_1_a.get()),
                                                                        )

            self.y_background_a = librosa.istft(self.background_a,
                                                n_fft=self.n_fft_1_a.get(),
                                                hop_length=self.hop_length_1_a.get(),
                                                win_length=self.win_length_1_a.get(),
                                                )
            self.y_residual_a = librosa.istft(self.residual_a,
                                              n_fft=self.n_fft_1_a.get(),
                                              hop_length=self.hop_length_1_a.get(),
                                              win_length=self.win_length_1_a.get(),
                                              )

            self.D_residual_a = librosa.stft(self.y_residual_a,
                                             n_fft=self.n_fft_2_a.get(),
                                             hop_length=self.hop_length_2_a.get(),
                                             win_length=self.win_length_2_a.get(),
                                             )

            # Perform the second HPSS on the residual
            self.murmurs_a, self.heart_sounds_a = librosa.decompose.hpss(self.D_residual_a,
                                                                         margin=(self.margin_harmonic_2_a.get(),
                                                                                 self.margin_percussive_2_a.get()),
                                                                         kernel_size=(self.percus_kernel_2_a.get(),
                                                                                      self.harmon_kernel_2_a.get()),
                                                                         )

            self.y_murmurs_a = librosa.istft(self.murmurs_a,
                                             n_fft=self.n_fft_2_a.get(),
                                             hop_length=self.hop_length_2_a.get(),
                                             win_length=self.win_length_2_a.get(),
                                             )

            self.y_heart_sounds_a = librosa.istft(self.heart_sounds_a,
                                                  n_fft=self.n_fft_2_a.get(),
                                                  hop_length=self.hop_length_2_a.get(),
                                                  win_length=self.win_length_2_a.get(),
                                                  )

            min_len = min(len(y_i) for y_i in (self.y_background_a, self.y_murmurs_a, self.y_heart_sounds_a))

            self.y_enhanced_a = standardise_signal(
                self.mix_back_a.get() * self.y_background_a[:min_len]
                + self.mix_murm_a.get() * self.y_murmurs_a[:min_len]
                + self.mix_tran_a.get() * self.y_heart_sounds_a[:min_len]
            )

            self.waveforms_a = [self.y_a, self.y_background_a, self.y_residual_a,
                                self.y_murmurs_a, self.y_heart_sounds_a, self.y_enhanced_a]

            # combined_waveforms.extend(waveforms)

            for j, component in enumerate(self.waveforms_a):
                # Select the axis for the waveform and spectrogram
                ax_waveform_a = self.axes[j][0]
                ax_spectrogram_a = self.axes[j][1]

                # Waveform plot
                librosa.display.waveshow(standardise_signal_a(component), sr=self.sr_a, alpha=0.5, ax=ax_waveform_a,
                                         color=['red', 'green', 'blue', 'orange', 'purple', 'black'][j],
                                         linewidth=0.5, linestyle='--')
                ax_waveform_a.set_ylabel(
                    ('Original' if j == 0
                     else 'Background' if j == 1
                     else 'Residual' if j == 2
                     else 'Murmurs' if j == 3
                     else 'Transient' if j == 4
                     else 'ENHANCE')
                    + ' Thang')
                ax_waveform_a.set_xlabel('Time (s)')
                ax_waveform_a.label_outer()  # Hide labels for shared axes

                # Spectrogram plot
                # if j < 5:  # Avoid recomputing STFT for the combined signal
                component_stft_a = librosa.stft(component,
                                                n_fft=self.n_fft_display_a.get(),
                                                hop_length=self.hop_length_display_a.get(),
                                                win_length=self.win_length_display_a.get(),
                                                )
                self.D_component_db = librosa.amplitude_to_db(np.abs(component_stft_a), ref=self.y_max_a)
                img_a = librosa.display.specshow(self.D_component_db, sr=self.sr_a, x_axis='time', y_axis='log',
                                                 ax=ax_spectrogram_a, cmap='jet',
                                                 n_fft=self.n_fft_display_a.get(),
                                                 hop_length=self.hop_length_display_a.get(),
                                                 win_length=self.win_length_display_a.get(),
                                                 vmin=self.vmin_a, vmax=self.vmax_a)
                ax_spectrogram_a.set_xlabel('Time (s)')
                ax_spectrogram_a.label_outer()  # Hide labels for shared axes

                # Colorbar is held by `cax`.
                cax_a = ax_spectrogram_a.inset_axes([1.03, 0, 0.1, 1], transform=ax_spectrogram_a.transAxes)
                cax_a.clear()
                _ = self.figure.colorbar(img_a,
                                         ax=ax_spectrogram_a,
                                         cax=cax_a,
                                         format='%+2.0f dB')

                # Ensure the plots share axes where appropriate
                if j not in [0]:
                    self.axes[j][0].sharey(self.axes[0][0])
                    self.axes[j][1].sharex(self.axes[0][1])

        # Refresh canvas
        self.canvas.draw()

    def mainloop(self):
        return self.master.mainloop()
