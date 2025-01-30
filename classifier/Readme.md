# ReadMe

This repo contains frontend programs for pre processing and running classifiers, along with
various functions for signal processing and machine learning

## Setup Docker (Recommended)

NOTE: This solution will not allow python pre-processing, only MATLAB.

- The below instructions are for the docker CLI.

Building the Image
- Ensure docker is installed.
- Build the image from the contained Dockerfile.
```
docker build -t mmaro/heart_classifier:1.0 .
```
- Run the image in an interactive terminal
```
docker run -it mmaro/heart_classifier:1.0
```

Once these steps have been followed you should now be inside a docker shell and can
run the commands to run the code. 

NOTE: Depending on how docker and permissions are setup sudo may be required before docker commands.

## Setup Local


Recommended: Ubuntu 20.04 or later.
NOTE: This version allows for python pre-processing or MATLAB.

The python dependencies are managed using pipenv. So this must be installed. 
The code base also makes use of python3 and matlab. However, to get around dependency issues
the Dockerfile within the repo can be used.

### Pipenv setup (Optional)

- Ensure MATLAB 2023a is installed
- Install python3.10
- Install pipenv `python3 -m pip --upgrade install pip && python3 -m pip install pipenv`
- Install pipenv dependencies `pipenv install`
- Install pipenvshebang `python3 -m pipenv-shebang`

Note: Make sure MATLAB is installed first. As matlabengine requires MATLAB.

Then to get matlabengine installed run:
```
pipenv shell
python -m pip install matlabengine
```

*NOTE:* If using an older version of matlab (<2023b) you want to install the older version of the enginer.

```
python -m pip install matlabengine==9.14.3
```

Once these steps have been followed the code will be able to be run successfully.

### Venv Setup (Recommended)

- Ensure MATLAB 2023a is installed
- Install python3.10
- Create a venv `python3 -m venv venv`
- Activate the venv (linux) `source venv/bin/actiate` (windows) `.\venv\Scripts\activate`
- Install the dependencies `python3 -m pip install -r requirements.txt`

Note: Make sure MATLAB is installed first. As matlabengine requires MATLAB.

## Running

*NOTE:* If using venv over pipenv instead of running by ./ use python then the name of the script. Just make sure you have activated the virtual environment.

### Pre-training on all of CINC

There are two bash scripts contained within this repository which call the various scripts in order to pretrain a model on all of the CINC database. This script also gives an example of how to run each script.

To run the script you may need to first enable permission (on linux):
```
chmod +x pre_train*.sh
```

Then to run you just execute the script:
```
./pre_train.sh
```

Or run it with bash:
```
bash pre_train.sh
```

### Segmentation

The segmentation is done within it's own frontend named segmentation.py for python and
segmentation.m for matlab.

Running the python front end requires two arguments, the directory where the dataset is stored
and the output directory to write the jsons that contain the split information. An example is shown
below.

```
./src/python/segmentation.py -I ./data/cinc/training-a -O ./data/segmentation/training-a
```

To get the list of arguments and what they represent.
```
./src/python/segmentation.py --help
```

Running the matlab frontend required ... TODO

### Train/Test/Valid Split

The train/test/valid split is done through it's own frontend, this allows the training to read
the split from a file. This allows for splits to be inspected as well as reused.

Running the python front end requires three arguments, the input directory where the data is, the 
output path to write the split file and lastly the dataset, ie training-a or training-a-extended currently. An example is shown below.

```
./src/python/split.py -I ./data/cinc/training-a -O ./data/split/training-a.csv -D training-a
```


To get the list of arguments and what they represent.
```
./src/python/split.py --help
```

### Training the model

Training the model is done within a frontend that also allows for testing. This has multiple parameters which can be found from the help menu.

There are various commands to train or test different models which can be found by running:
```
./src/python/run_model.py --help
```

Running can be done as shown below. Noting that running with --help with show how to run.

```
./src/python/run_model.py train_model -D ../../data/training-a/ -P ../../data/split/training-a.csv -Z ../../data/segmentation/training-a -R stft -M resnet -I ../../data/images/training-a-O ../../data/models/test1.pth -F./src/python/run_model.py train_model
```

Running a multi input model is shown below. With an aditional change being for the model string. To have an ensemble you specify 'ensemble:\<num_models\>:\<base_model_type\>'

```
./src/python/run_model.py train_multi_model -D ../../data/training-a/ -P ../../data/split/training-a.csv -Z ../../data/segmentation/training-a -R stft -M ensemble:2:resnet -I ../../data/images/training-a-ensemble -O ../../data/models/test1-ensemble.pth -F./src/python/run_model.py train_model
```


To get the list of arguments and what they represent.
```
./src/python/run_model.py train_model --help
```
or for the multi input models
```
./src/python/run_model.py train_multi_model --help
```

### Testing the model

Testing the model requires similar arguments to training however some are ommitted. These arguments are:
- directory of the data
- Path to the train/test/valid split
- Path to the segment directory
- Model string to represent what type of model to train (resnet/vgg/inception)
- What type of image transform (stft/mel-stft/wave)
- The directory to store the output images
- The directory to read the trained model 
- The database being used (optional, default is training-a).
- Four bands to be used.

Running can be done as shown below.
```
./src/python/run_model.py test_model -D ../../data/training-a/ -P ../../data/cinc/split.csv -Z ../../data/segments-a/ -R stft -M ensemble:2:resnet -I ../../data/images/training-a-ensemble -T ../../data/models/test1-ensemble.pth -F
```

To get the list of arguments and what they represent.
```
./src/python/run_model.py test_model --help
```

## Structure 

The src folder contains all the source code as well as the front ends.

### Python folder

This folder contains the libraries and frontends that are used. There are 3 front ends, split.py segmentation.py and run_model.py, their functionality is explained above.

There are three current libraries:
- util: For util functions 
- processing: For signal processing and other intensive algorithms
- classifier: For machine learning / classifier code

### MATLAB folder

This folder contains all the MATLAB functions that are utilised by the python code plus some additional ones for running the CNN MATLAB model (Rong et al.)
