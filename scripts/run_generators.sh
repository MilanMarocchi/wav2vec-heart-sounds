#!/usr/bin/env bash
# Train the DiffWave and WaveGrad generators on CinC Training-A, then generate synthetic datasets.
set -euo pipefail

DATA_DIR=${DATA_DIR:?set DATA_DIR to the CinC training-a record directory}
CSV=${CSV:?set CSV to the reference/split CSV}
OUT=${OUT:-modelout}
EPOCHS=${EPOCHS:-100}

for MODEL in diffwave wavegrad; do
  mpcg-wav2vec gen-train --model "$MODEL" --data-dir "$DATA_DIR" --csv "$CSV" \
    --output-dir "$OUT/$MODEL" --epochs "$EPOCHS"
  mpcg-wav2vec gen-sample --model "$MODEL" --weights "$OUT/$MODEL/weights.pt" \
    --data-dir "$DATA_DIR" --csv "$CSV" --output-dir "generated/$MODEL" --per-item 3
done
