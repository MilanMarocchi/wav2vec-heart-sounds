#!/bin/bash
set -e
set -x

skip=0
rnn=0
m_provided=0
d_provided=0
t_provided=0
f_provided=0

model_str=""

useage() {
   echo "USEAGE: $0 -m <model> -d </path/to/data> -t <transform> <-s> <-r>"
   echo "   -m: The model type to use [resnet/vgg/inception/wav2vec/bilstm/cnn-bilstm]"
   echo "   -d: The path to the data (cinc data)"
   echo "   -t: The transform to use [stft/mel-stft/wave]"
   echo "   -s: To skip generating of the splits and segmentation data. NOTE: If this is not selected all the processed data will be deleted as well"
   echo "   -r: If the model is an rnn"
   echo "   -n: Additional string to append at the end of the save model"
   echo "   -f: Sampling rate for classification"
}

while getopts srm:d:t:n:f: flag
do
   case "${flag}" in
      s) skip=1 ;;
      r) rnn=1 ;;
      m) 
         model="$OPTARG" 
         m_provided=1
         ;;
      d) 
         data_path="$OPTARG" 
         d_provided=1
         ;;
      t) 
         transform="$OPTARG" 
         t_provided=1
         ;;
      n)
         model_str="$OPTARG"
         ;;
      f)
         sampling_rate="$OPTARG"
         f_provided=1
         ;;
   esac
done

if [ $m_provided -eq 0 ] || [ $d_provided -eq 0 ] || [ $t_provided -eq 0 ] || [ $f_provided -eq 0 ]; then
    echo "Missing required options."
    useage
    exit 1;
fi

if [ $skip -eq 0 ]; then
   if [ $rnn -eq 1 ]; then
      ## Delete all data
      rm -rf ./data/processed_audio/cinc/entire
      rm -rf ./data/segmentation/rnn/training-{a..f}
      rm -rf ./data/splits/rnn/training-{a..f}.csv

      ## First set up all the segmentation.
      for set in a b c d e f; do 
         ./src/python/segmentation.py -I "$data_path"/training-$set/ -O ./data/segmentation/rnn/training-$set/ -T 4 -D training-$set; 
      done
      ## Create the splits to be used (will be training on all of the data so no test set.)
      for set in a b c d e f; do 
         ./src/python/split.py -I "$data_path" -O ./data/splits/rnn/training-$set.csv/ -D training-$set -S 0.7:0.15:0.15; 
      done
   else
      ## Delete all data
      rm -rf ./data/images/cinc/entire
      rm -rf ./data/segmentation/training-{a..f}
      rm -rf ./data/splits/training-{a..f}

      ## First set up all the segmentation.
      for set in a b c d e f; do 
         ./src/python/segmentation.py -I "$data_path"/training-$set/ -O ./data/segmentation/rnn/training-$set/ -T 2.5 -D training-$set; 
      done
      ## Create the splits to be used (will be training on all of the data so no test set.)
      for set in a b c d e f; do 
         ./src/python/split.py -I "$data_path" -O ./data/splits/cnn/training-$set.csv/ -D training-$set -S 0.7:0.15:0.15; 
      done
   fi
fi

if [ $rnn -eq 1 ]; then
   splits=data/splits/rnn
   segments=data/segmentation/rnn
   output_dir=data/processed_audio/cinc/entire/
   model_path="data/models/cinc-wav2vec${model_str}.pth"

   fs=${sampling_rate}

   # Train the model on each dataset to produce a good pre-trained model.
   ./src/python/run_model.py train_cinc_model \
      -D "$data_path" \
      -P "$splits" \
      -Z "$segments" \
      -I "$output_dir" \
      -A time \
      -G "$fs" \
      -M "$model" \
      -L 4 \
      -O "$model_path" \
      -N 10 \
      -R "$transform" \
      -C

   ./src/python/run_model.py test_model \
      -D "$data_path" \
      -P "$splits" \
      -Z "$segments" \
      -I "$output_dir" \
      -M "$model" \
      -T data/models/cinc-wav2vec${model_str}.pth \
      -B training-a \
      -L 4 \
      -R "$transform" 
else
   splits=data/splits/
   segments=data/segmentation/
   output_dir=data/images/cinc/entire/
   model_path="data/models/cinc2${model_str}.pth"

   ./src/python/run_model.py train_cinc_model \
      -D "$data_path" \
      -P "$splits" \
      -Z "$segments" \
      -I "$output_dir" \
      -R "$transform" \
      -M "$model" \
      -O "${model_path}"  \
      -A time \
      -N 30 \
      -F 

   ./src/python/run_model.py test_model \
      -D "$data_path" \
      -P "$splits" \
      -Z "$segments" \
      -I ./data/images/cinc/training-a \
      -R "$transform" \
      -M "$model" \
      -T ./data/models/cinc2${model_str}.pth \
      -B training-a \
      -F
fi
