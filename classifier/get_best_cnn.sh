#!/bin/bash
today=$(date +"%b%d" | tr '[:upper:]' '[:lower:]')
log_file="training-a-wav2vec_cnn-split-1-${today}.log"
models_location="models/wav2vec-cnn-4s-${log_file}.pt"
dataset_location="data/preprocessed_audio/training-a-4s-for-wav2vec-paper"

echo "" > "${log_file}"

for idx in {1..10}; do
    echo "Training Trial ${idx}" >> "${log_file}"
    ./train.sh -c -m big_rnn:2:wav2vec-cnn -o "${models_location}-${idx}" -s data/gen_config_rnn_paper_training_a.json -b training-a -a 4 -i "${dataset_location}" -g 4125 -r -h 1 >> "${log_file}"
done
