#!/bin/bash
today=$(date +"%b%d" | tr '[:upper:]' '[:lower:]')
log_file="training-a-pcg-${today}.log"
models_location="models/wav2vec-4s-pcg-${log_file}"
dataset_location="data/preprocessed_audio/training-a-4s-for-wav2vec-paper"

echo "" > "${log_file}"

return
for idx in {1..10}; do
    echo "Training Trial ${idx}" >> "${log_file}"
    ./train.sh -m wav2vec -o "${models_location}-${idx}" -s data/gen_config_rnn_paper_training_a.json -b training-a-pcg -a 4 -i "${dataset_location}" -g 4125 -r -h 1 >> "${log_file}"
done
