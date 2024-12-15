# Data acquisition and layout

This project uses three datasets plus two recorded-noise corpora. Only the paths need to be
supplied to the commands; nothing about the data is bundled with the code.

## Datasets

### CinC 2016 (PhysioNet Challenge 2016) — single-channel PCG and Training-A PCG+ECG
- Source: https://physionet.org/content/challenge-2016/1.0.0/
- The challenge ships six training databases `training-a` … `training-f`. Training-a additionally
  contains a synchronised ECG channel, used for the PCG+ECG setting.
- Each record is a `wfdb` record (`<id>.hea` + signal file). For Training-A, channel 0 is PCG and
  channel 1 is ECG.

### Wearable vest — multichannel PCG
- Multichannel PCG recordings from the wearable vest device described in the paper. Each recording
  is a multichannel WAV file (read with `scipy.io.wavfile`; integer PCM is scaled to [-1, 1]).
  Channel layout: PCG microphones 1-7 occupy WAV columns 0-6, ECG lead `E` is column 7, and a
  second ECG `E2` is column 8. Files are located by matching the patient id (from the reference
  CSV) against the WAV filenames. This dataset is not public; use your local copy.

### Recorded-noise corpora (for clinical-noise augmentation)
- **EPHNOGRAM** (PCG noise, AUX channels): https://physionet.org/content/ephnogram/1.0.0/
- **MIT-BIH Noise Stress Test** (ECG noise `em`/`bw`/`ma`): https://physionet.org/content/nstdb/1.0.0/

Point augmentation at these with `AugmentConfig(ephnogram_dir=..., mit_dir=...)` (or leave empty
to disable recorded-noise augmentation).

## Reference CSV

Every loader takes a reference CSV describing patients and splits:

```csv
patient,abnormality,split,split2,split3
a0001,1,train,valid,test
a0002,0,test,train,train
...
```

- `patient` — record id (matches `<data-dir>/<patient>`).
- label column — `abnormality` (or `label`/`diagnosis`); `1` = abnormal, `0`/`-1` = normal.
- `split`, `split2`, … — per-fold assignment valued `train`/`valid`/`test`. Fold `n` uses column
  `split` when `n == 1`, otherwise `split<n>`. Pass `subset='all'` (internally) to ignore splits.

Patient-level random splits are used throughout, matching the paper.

### Generating the split CSV

If you only have the CinC `REFERENCE.csv` label files, generate a stratified split CSV with:

```bash
# single database (Training-A), 5 folds, 60/20/20 patient-level stratified split
mpcg-wav2vec make-splits --data-dir <cinc>/training-a --out splits/training-a.csv --folds 5

# combine several databases into one label list (e.g. all of CinC)
mpcg-wav2vec make-splits --data-dir <cinc>/training-a --data-dir <cinc>/training-b \
    ... --out splits/cinc-all.csv --folds 5
```

This reads each directory's `REFERENCE.csv`, splits at the patient level stratified by label
(independently per fold), and writes `patient,label,split,split2,…`. For leave-source-DB-out,
generate one CSV per database.

## Expected directory layout

```
<data-dir>/
  a0001.hea  a0001.wav      # wfdb record (PCG [+ ECG] channels)
  a0002.hea  a0002.wav
  ...
<reference.csv>             # patient + label + split columns
```

Generated (synthetic) datasets produced by `mpcg-wav2vec gen-sample` follow their own manifest
format: a directory of `.wav` files plus a `REFERENCE.csv` with columns `patient,label,file`.

## Cardiac-cycle segmentation files (optional, for generator training)

Heart-cycle rearranging during generator training (`gen-train --segment-dir`) reads one JSON per
recording, `<segment-dir>/<patient>.json`, of the form:

```json
{"segments": [[i0], [i1], [i2], ...], "last_index": 120000, "fs": 1000}
```

Each group's first index is a cardiac-cycle boundary (in samples at rate `fs`); boundaries are
rescaled to the generator's sample rate on the fly. Without `--segment-dir`, training uses the
raw waveforms and skips rearranging.
