#!/bin/bash
####################################################
#
# train.sh
# Author: Milan Marocchi
#
#
# Purpose: Train a model with ease  
#
####################################################
set -e
set -x

rnn=0
skip_process=0
m_provided=0
d_provided=0
t_provided=0
o_provided=0
b_provided=0
i_provided=0
s_provided=0
a_provided=0
g_provided=0
c_provided=0

useage() {
   echo "USEAGE: $0 -m <model>  -t <transform> -o <outpath> -b <database> -i <processpath> -s <path/to/schedule> -a <fragment_len> -g <sampling rate> <-c> <-r>"
   echo "   -m: The model type to use [resnet/vgg/inception/wav2vec/bilstm/cnn-bilstm] or composite models eg [big:2:resnet]"
   echo "   -t: The transform to use [stft/mel-stft/wave] only required if not using an rnn"
   echo "   -o: The output path to save the trained model" 
   echo "   -b: The database being used (training-[a-b]/cinc)"
   echo "   -i: The path to store the preprocessed data."
   echo "   -s: The path to the schedule file."
   echo "   -r: If the model is an rnn."
   echo "   -a: The time each fragment will be."
   echo "   -g: The sampling rate to be used for classifcation (audio only models)."
   echo "   -c: Skip image/audio creation (only to be done if you know you created the images)."
}

# Get arguments
while getopts rm:t:o:b:i:s:a:g:c flag
do
   case "${flag}" in
      r) rnn=1 ;;
      m) 
        model="$OPTARG" 
        m_provided=1
        ;;
      t) 
        transform="$OPTARG" 
        t_provided=1
        ;;
      o) 
        output_path="$OPTARG" 
        o_provided=1
        ;;
      b) 
        database="$OPTARG" 
        b_provided=1
        ;;
      i) 
        processed_path="$OPTARG" 
        i_provided=1
        ;;
      s) 
        schedule_path="$OPTARG" 
        s_provided=1
        ;;
      a)
        segment_time="$OPTARG"
        a_provided=1
        ;;
      g)
        fs="$OPTARG"
        g_provided=1
        ;;
      c) skip_process=1 ;;
   esac
done

[ $g_provided -eq 0 ] && [ $rnn -eq 1 ] || g_provided=1
[ $t_provided -eq 0 ] && [ $rnn -eq 0 ] || t_provided=1

# Make sure arguments are provided
if [ $m_provided -eq 0 ] || 
   [ $o_provided -eq 0 ] ||
   [ $b_provided -eq 0 ] ||
   [ $i_provided -eq 0 ] ||
   [ $s_provided -eq 0 ] ||
   [ $a_provided -eq 0 ] ||
   [ $g_provided -eq 0 ] ||
   [ $t_provided -eq 0 ]; then
    echo "Missing required options."
    echo ""
    useage
    exit 1;
fi

# Get all of the paths from the schedule file
data_paths=($( ./src/python/parse_schedule.py display_data_paths -S "$schedule_path" ))
split_paths=($( ./src/python/parse_schedule.py display_split_paths -S "$schedule_path" ))
segment_paths=($( ./src/python/parse_schedule.py display_segment_paths -S "$schedule_path" ))

# Create anything that is missing
index=0
for i in ${data_paths[@]}
do 
    if ! [ -f ${split_paths[$index]} ]; then
        ./src/python/split.py -I ${data_paths[$index]} -O ${split_paths[$index]} -D $database
    fi

    if ! [ -d ${segment_paths[$index]} ]; then
        ./src/python/segmentation.py -I ${data_paths[$index]} -O ${segment_paths[$index]} -D $database -T $segment_time
    fi

    index=$index+1
done

skip_arg=""
if [ $skip_process -eq 1 ]; then
  skip_arg="-C"
fi

# if the model is an rnn
if [ $rnn -eq 1 ]; then
  ./src/python/run_model.py train_rnn_gen_model \
    -M "$model" \
    -I "$processed_path" \
    -O "$output_path" \
    -S "$schedule_path" \
    -B "$database" \
    -A "time" \
    -L "$segment_time" \
    -G "$fs" \
    $skip_arg
else
  # Otherwise check if multi cnn or single cnn
  if [[ $model = *':'* ]]; then
    ./src/python/run_model.py train_multi_gen_model \
      -M "$model" \
      -I "$processed_path" \
      -O "$output_path" \
      -S "$schedule_path" \
      -B "$database" \
      -A "time" \
      -L "$segment_time" \
      -F \
      -R "$transform" \
      $skip_arg
  else
    ./src/python/run_model.py train_gen_model \
      -M "$model" \
      -I "$processed_path" \
      -O "$output_path" \
      -S "$schedule_path" \
      -B "$database" \
      -A "time" \
      -L "$segment_time" \
      -F \
      -R "$transform" \
      $skip_arg
  fi
fi