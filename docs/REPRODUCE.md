# Reproducing the experiments and ablations

This document lists the exact commands for every experiment and ablation. Set the data paths
once as environment variables; see [DATA.md](DATA.md) for how to obtain and lay out the data.

```bash
export CINC_DIR=.../challenge-2016/1.0.0            # parent of training-a … training-f
export TRAINA_DIR=$CINC_DIR/training-a
export TRAINA_CSV=.../training-a.csv
export CINC_CSV=.../cinc-all.csv
export VEST_DIR=.../vest
export VEST_CSV=.../vest.csv
export EPHNOGRAM=.../ephnogram   MIT=.../nstdb       # optional, for recorded-noise augmentation
```

Every command appends its metrics (fragment- and patient-level accuracy, UAR, sensitivity,
specificity, MCC) to the `--results-json` file. Use `--fold N` to select a cross-validation fold
and `--max-batches N` for a fast smoke check. Runs are patient-level random splits.

Convenience wrappers: `scripts/run_generators.sh` (train + sample both generators) and
`scripts/run_ablations.sh` (the classification matrix).

## 0. Prepare split CSVs

If you don't already have reference/split CSVs, generate them from the CinC `REFERENCE.csv`
label files (patient-level, label-stratified, one column per fold):

```bash
mpcg-wav2vec make-splits --data-dir "$TRAINA_DIR" --out splits/training-a.csv --folds 5
mpcg-wav2vec make-splits $(for d in a b c d e f; do echo --data-dir "$CINC_DIR/training-$d"; done) \
    --out splits/cinc-all.csv --folds 5
# leave-source-DB-out: one CSV per database
for d in a b c d e f; do
  mpcg-wav2vec make-splits --data-dir "$CINC_DIR/training-$d" --out "splits/$d.csv" --folds 1
done
```

Then point `--csv` (and the schedule JSON `split` fields) at these files.

---

## 1. Synthetic signal generation (WaveGrad + DiffWave)

Train each generator, then synthesise an augmentation dataset conditioned on real recordings.

```bash
for MODEL in diffwave wavegrad; do
  mpcg-wav2vec gen-train  --model $MODEL --data-dir "$TRAINA_DIR" --csv "$TRAINA_CSV" \
                          --segment-dir "$SEG_DIR" \
                          --output-dir modelout/$MODEL --epochs 100
  mpcg-wav2vec gen-sample --model $MODEL --weights modelout/$MODEL/weights.pt \
                          --data-dir "$TRAINA_DIR" --csv "$TRAINA_CSV" \
                          --output-dir generated/$MODEL --per-item 3
done
```

`--segment-dir` points at the per-recording cardiac-cycle segmentation files (`<patient>.json`);
with it, training rearranges heart cycles (shuffle groups of cycles and crossfade-rebuild), which
diversifies the training targets. Use `--no-rearrange` to disable, or `--prob-contiguous P` to
mix in rotated-contiguous rebuilds. `gen-train`/`gen-sample` also accept `--condition-on-ecg`
(condition on the ECG lead) and, for DiffWave, `--fast/--no-fast` sampling.

> **Legacy training regime.** The original CinC / Training-A runs used a "reference" regime that
> trains for half the requested epochs and augments the validation set. Add `--reference-train-rnn`
> to `classify-cinc` / `classify-lsdo` to reproduce it; omit it for the standard regime (full
> epochs, clean validation).

## 2. Single-channel PCG (CinC 2016, 16 kHz)

Ablation axis: pre-trained encoder vs random init, and no-augmentation vs conventional
augmentation.

```bash
mpcg-wav2vec classify-cinc --data-dir "$CINC_DIR" --csv "$CINC_CSV" --mode pcg --dataset cinc \
    --fs 16000 --epochs 20 --augment                  --results-json results/cinc.json   # augmented
mpcg-wav2vec classify-cinc --data-dir "$CINC_DIR" --csv "$CINC_CSV" --mode pcg --dataset cinc \
    --fs 16000 --epochs 20 --no-augment               --results-json results/cinc.json   # original only
mpcg-wav2vec classify-cinc --data-dir "$CINC_DIR" --csv "$CINC_CSV" --mode pcg --dataset cinc \
    --fs 16000 --epochs 20 --no-augment --random-init --results-json results/cinc.json   # no pretraining
```

## 3. Synchronised PCG + ECG (Training-A, 4.125 kHz)

Modalities: PCG-only, ECG-only, and the two-branch PCG+ECG fusion (`big_rnn:2:wav2vec`). Each with
original / augmented / random-init.

```bash
for MODE in pcg ecg pcg_ecg; do
  for FLAGS in "--augment" "--no-augment" "--no-augment --random-init"; do
    mpcg-wav2vec classify-cinc --data-dir "$TRAINA_DIR" --csv "$TRAINA_CSV" --mode $MODE \
        --dataset training-a --fs 4125 --epochs 20 $FLAGS --results-json results/training_a.json
  done
done
```

## 4. Synthetic-augmentation schedules (single-channel PCG)

Trains through a staged schedule that interleaves real Training-A data with synthetic WaveGrad or
DiffWave data (`proportion`/`augment_num` per stage). The schedule JSONs live in `data/`; edit
their `path`/`split`/`segment` fields to point at your generated datasets from step 1.

```bash
mpcg-wav2vec classify-synthetic --schedule data/gen_config_rnn_wavegrad_only.json \
    --fs 4125 --results-json results/synthetic.json      # original -> WaveGrad -> original
mpcg-wav2vec classify-synthetic --schedule data/gen_config_rnn_diffwave_only.json \
    --fs 4125 --results-json results/synthetic.json      # original -> DiffWave -> original
mpcg-wav2vec classify-synthetic --schedule data/gen_config_rnn_paper_training_a.json \
    --fs 4125 --results-json results/synthetic.json      # full mixed schedule
```

Add `--random-init` for the no-pretraining variant.

## 5. Leave-source-database-out (CinC)

Train on five CinC databases and test on the held-out one, for each held-out database.

```bash
DBS="--db a:$CINC_DIR/training-a:$TRAINA_CSV --db b:$CINC_DIR/training-b:.../b.csv \
     --db c:$CINC_DIR/training-c:.../c.csv --db d:$CINC_DIR/training-d:.../d.csv \
     --db e:$CINC_DIR/training-e:.../e.csv --db f:$CINC_DIR/training-f:.../f.csv"
for HELD in a b c d e f; do
  mpcg-wav2vec classify-lsdo $DBS --holdout $HELD --fs 4125 --epochs 20 \
      --results-json results/lsdo.json
done
```

## 6. Multichannel vest (4.125 kHz)

Two ablation axes. **Data source:** original / augmented / random-init. **Model design:** LoRA
adapters / full fine-tune / frozen encoder. Each run reports both the MLP head and an SVM fitted
on the encoder features.

```bash
# data-source axis
mpcg-wav2vec classify-vest --data-dir "$VEST_DIR" --csv "$VEST_CSV" --epochs 20 --augment    --lora    --results-json results/vest.json
mpcg-wav2vec classify-vest --data-dir "$VEST_DIR" --csv "$VEST_CSV" --epochs 20 --no-augment --lora    --results-json results/vest.json
mpcg-wav2vec classify-vest --data-dir "$VEST_DIR" --csv "$VEST_CSV" --epochs 20 --random-init --no-lora --results-json results/vest.json

# model-design axis
mpcg-wav2vec classify-vest --data-dir "$VEST_DIR" --csv "$VEST_CSV" --epochs 20 --lora            --results-json results/vest.json   # LoRA
mpcg-wav2vec classify-vest --data-dir "$VEST_DIR" --csv "$VEST_CSV" --epochs 20 --no-lora         --results-json results/vest.json   # full fine-tune
mpcg-wav2vec classify-vest --data-dir "$VEST_DIR" --csv "$VEST_CSV" --epochs 20 --freeze-encoder  --results-json results/vest.json   # frozen encoder
```

Use `--channels 1,2,3,4,5,6` to select microphones (six channels by default; the sinc beamformer
collapses them to one before the encoder).

## Cross-validation and aggregation

Repeat any command over `--fold 1 … 7` (vest uses 7 folds; CinC/Training-A use the folds present
in the reference CSV), all appending to the same `--results-json`. Then aggregate the per-fold
entries into mean/std tables per condition:

```bash
mpcg-wav2vec summarize results/training_a.json --group-by run_label --out tables/training_a.md
```

The metrics keys are `accuracy`, `uar`, `sensitivity`, `specificity`, `npv`, `precision`, `f1`,
`mcc`, under `fragment` and `patient` groupings (and `mlp`/`svm` for the vest runs).
