#!/bin/bash
today=$(date +"%b%d" | tr '[:upper:]' '[:lower:]')
log_file="cinc-wav2vec-cnn-${today}.log"
models_location="wav2vec-4s-cinc-wav2vec-cnn-${log_file}"
dataset_location="data/physionet.org/files/challenge-2016/1.0.0/"
transform="stft"
fs=4125

echo "" > "${log_file}"

for idx in {1..10}; do
    echo "Training Trial ${idx}" >> "${log_file}"
    ./pre_train.sh -m wav2vec-cnn -d "${dataset_location}" -t "${transform}" -s -r -f "${fs}" >> "${log_file}" -n "${models_location}-${idx}"
done
