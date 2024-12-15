#!/usr/bin/env bash
# Reproduce the paper's classification ablations across its three settings.
#
#   * CinC single-channel PCG      (16 kHz)
#   * Training-A PCG+ECG two-branch (4.125 kHz)
#   * Multichannel vest            (4.125 kHz)
#
# Each of orig / augmented / random-init is run; results are appended to a JSON file.
set -euo pipefail

CINC_DIR=${CINC_DIR:?CinC records dir}
CINC_CSV=${CINC_CSV:?CinC reference/split CSV}
TRAINA_DIR=${TRAINA_DIR:?Training-A records dir}
TRAINA_CSV=${TRAINA_CSV:?Training-A reference/split CSV}
VEST_DIR=${VEST_DIR:?vest records dir}
VEST_CSV=${VEST_CSV:?vest reference/split CSV}
RESULTS=${RESULTS:-ablation_results.json}
EPOCHS=${EPOCHS:-20}

# --- single-channel PCG (CinC, 16 kHz) ---
mpcg-wav2vec classify-cinc --data-dir "$CINC_DIR" --csv "$CINC_CSV" --mode pcg --dataset cinc \
  --fs 16000 --epochs "$EPOCHS" --augment            --results-json "$RESULTS"
mpcg-wav2vec classify-cinc --data-dir "$CINC_DIR" --csv "$CINC_CSV" --mode pcg --dataset cinc \
  --fs 16000 --epochs "$EPOCHS" --no-augment          --results-json "$RESULTS"
mpcg-wav2vec classify-cinc --data-dir "$CINC_DIR" --csv "$CINC_CSV" --mode pcg --dataset cinc \
  --fs 16000 --epochs "$EPOCHS" --no-augment --random-init --results-json "$RESULTS"

# --- Training-A PCG+ECG two-branch (4.125 kHz) ---
for FLAGS in "--augment" "--no-augment" "--no-augment --random-init"; do
  mpcg-wav2vec classify-cinc --data-dir "$TRAINA_DIR" --csv "$TRAINA_CSV" --mode pcg_ecg \
    --dataset training-a --fs 4125 --epochs "$EPOCHS" $FLAGS --results-json "$RESULTS"
done

# --- synthetic-augmentation schedules (single-channel PCG) ---
for SCHED in wavegrad_only diffwave_only paper_training_a; do
  mpcg-wav2vec classify-synthetic --schedule "data/gen_config_rnn_${SCHED}.json" \
    --fs 4125 --results-json "$RESULTS" || true
done

# --- multichannel vest (4.125 kHz): data-source and design axes ---
mpcg-wav2vec classify-vest --data-dir "$VEST_DIR" --csv "$VEST_CSV" --epochs "$EPOCHS" \
  --augment --lora   --results-json "$RESULTS"
mpcg-wav2vec classify-vest --data-dir "$VEST_DIR" --csv "$VEST_CSV" --epochs "$EPOCHS" \
  --no-augment --lora --results-json "$RESULTS"
mpcg-wav2vec classify-vest --data-dir "$VEST_DIR" --csv "$VEST_CSV" --epochs "$EPOCHS" \
  --random-init --no-lora --results-json "$RESULTS"
mpcg-wav2vec classify-vest --data-dir "$VEST_DIR" --csv "$VEST_CSV" --epochs "$EPOCHS" \
  --no-lora --results-json "$RESULTS"          # full fine-tune
mpcg-wav2vec classify-vest --data-dir "$VEST_DIR" --csv "$VEST_CSV" --epochs "$EPOCHS" \
  --freeze-encoder --results-json "$RESULTS"

echo "Ablation results written to $RESULTS"
