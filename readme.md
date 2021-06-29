Hello!

Below you can find a outline of how to reproduce our solution for the Hubmap-Hacking the Kidney competition.
This package will let you create a virtual environment to reproduce our solution.
If you run into any trouble with the setup/code or have any questions please contact me at ukrakim@gmail.com

# ARCHIVE CONTENTS
Hubmap-Hacking the Kidney, 3rd Place Solution.pdf : write up of methods
Makefile: makefile to set up virtual environment
requirements.txt: requirements
download_data.sh: script to download and unzip data
reproduce.sh: script to reproduce entire solution from scratch
reproduce_fast.sh: script to reproduce fast solution from scratch
data: pre-made directory to download and unzip data to
train_code: code to rebuild models from scratch
train_code_fast: code to rebuild fast models from scratch
trained_models: code/trained models with logs
trained_models_fast: code/trained version of fast models

# HARDWARE: (The following specs were used to create the original solution)
Ubuntu 18.04 LTS (512 GB boot disk)
2xRTX 3090 (24 Gb VRam each)

# SOFTWARE (python packages are detailed separately in `requirements.txt`):
Python 3.7.9
CUDA 11.1
nvidia drivers v.460.56


# IRTUAL ENVIRONMENT SETUP
1) make setup
2) bash install_additional_packages.sh

# DATA SETUP (assumes the [Kaggle API](https://github.com/Kaggle/kaggle-api) is installed)
below are the shell commands used in each step, as run from the top level directory
bash download_data.sh

# DATA PROCESSING
#The train/predict code will also call this script if it has not already been run on the relevant data.
preprocessing was done on Kaggle in this notebook: https://www.kaggle.com/shujun717/1024-reduce-4-images?scriptVersionId=59850599. download_data.sh automatically downloads the preprocessed dataset

# MODEL BUILD: There are three options to produce the solution.
retrain models:
a) expect this to run for 25 hours on 2x3090 system
b) trains all models from scratch
c) follow commands below to produce entire solution from scratch

bash reproduce.sh

retrain models fast (this uses 4x downsampled tiled images and produces lb 0.9178/0.9467):
a) expect this to run for 2 hours on 2x3090 system
b) trains 5-fold resnext50 models from scratch for 50 epochs
c) follow commands below to produce fast solution from scratch

bash reproduce_fast.sh

possible issues:
1) if you run into GPU memory issue, reduce --batch_size option in the sh script.
I suggest you try halving it.
2) --workers should be smaller than the number of cpu cores your cpu have

# INFERENCE
full solution:
1) create new Kaggle dataset of trained weights from train_code/resnext50/models and train_code/resnext101/models
2) Copy and edit this notebook: https://www.kaggle.com/shujun717/hubmap-3rd-place-inference?scriptVersionId=62198789
3) Change weights path to your own. There are 2 list variables MODELS_rsxt50, and MODELS_rsxt101 to change. They should contain paths to your trained weights. MODELS_rsxt50 should have paths to weights of resnext50 models and MODELS_rsxt101 should have paths to weights of resnext101 models
4) Commit notebook and submit with submission.csv

fast solution:
1) create new Kaggle dataset of trained weights from train_code_fast/resnext50/models
2) Copy and edit this notebook: https://www.kaggle.com/shujun717/hubmap-3rd-place-fast
3) Change weights path to your own. There are 2 list variables MODELS_rsxt50. It should contain
paths to your trained weights
4) Commit notebook and submit with submission.csv

on new data (new tiff images)
1) create new Kaggle dataset of trained weights from train_code/resnext50/models and train_code/resnext101/models
2) Copy and edit this notebook: https://www.kaggle.com/shujun717/hubmap-3rd-place-inference?scriptVersionId=62198789
3) Create an csv file with filenames without suffix as id. The csv file should have an id column. Change the variable df_sample by changing the path in the pd.read_csv function to your new csv file.
4) Change the variable DATA to the path of the folder containing new tiff images
5) run notebook and the predictions will be in submission.csv
