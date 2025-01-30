
echo "This ain't done yet"

exit

cd /workspace
apt install gcc g++ python3-dev wget vim unzip libportaudio2 -y
wget https://lmnt.com/assets/diffwave/diffwave-ljspeech-22kHz-1000578.pt
wget https://lmnt.com/assets/wavegrad/wavegrad-24kHz.pt
unzip cinc.zip
unzip epcgdiffusion.zip
pip install -r requirements.txt

python src/main.py train-model -i home/labbo/dev/thesis/datasets/cinc/physionet.org/files/challenge-2016/1.0.0/ -d training-a-extended -s splits/CarpenterThesisHandball_2023-10Oct-29_1423.csv -t TIME_IN_MINUTES -r REF_SIG -c CON_SIG -g MODEL -w WEIGHTS

python src/main.py train-model -i home/labbo/dev/thesis/datasets/cinc/physionet.org/files/challenge-2016/1.0.0/ -d training-a-extended -s splits/CarpenterThesisHandball_2023-10Oct-29_1423.csv -t 120 -r ecg -c pcg -g DiffWave -w diffwave-ljspeech-22kHz-1000578.pt
