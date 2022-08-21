# Tumour Region Segmentation Pipeline

This repo is for training and testing brca cancer detection pipeline using Resnet-34. 
It is based on the code and concepts published in the paper: [Utilizing Automated Breast Cancer Detection to Identify Spatial Distributions of Tumor Infiltrating Lymphocytes in Invasive Breast Cancer](https://arxiv.org/abs/1905.10841)

NOTE: download the latest trained models [here](https://drive.google.com/drive/folders/1PuHSAmKTlZwCn3LYBan_Lj78ddW_ceG0?usp=sharing), extract the *.t7  files to the repository_root_folder/repo/models_cnn 

The default settings are for the latest Resnet-34 trained. To use other models, change the variable "MODEL" in conf/variables.sh to other models name downloaded from google drive above.

# Dependencies

 - [Pytorch 0.4.0](http://pytorch.org/)
 - Torchvision 0.2.0
 - cv2 (3.4.1.15)
 - [Openslide 1.1.1](https://openslide.org/api/python/)
 - [sklearn](https://scikit-learn.org/stable/)
 - [PIL](https://pillow.readthedocs.io/en/3.1.x/reference/Image.html)
 - GNU Octave (To produce Heatmaps and get Tumour Region Percentage) (Not part of the conda environment)
    - Can be installed using the command 'sudo apt-get install octave octave-control octave-image octave-io octave-optim octave-signal octave-statistics'
 
 More details are in file conda_environ.yml. Octave will have to be installed separately. 

# Running Codes Instrucstions
- Codes are in folder scripts, including training and testing
- Change the folder paths for training and prediction data in conf/variables.sh

## Setup conf/variables.sh
- Change the BASE_DIR to {path of your folder}/repo after you clone the git repository
- Change the PRED_DATA_DIR to the path of the data folder for prediction. 
- Change the SVS_QUEUE_FOLDER to the path where you wish to hold WSIs for training in small batches
- Change the TRAINING_DATA to the path where training WSIs will be stored

## Conda Environment
- Create a Conda Environment using the conda_environ.yml file by running the command *conda create --name <Env Name> --file conda_environ.yml*

## Training:
- Create folders *svs*, *patches* and *masks* in TRAINING_DATA
- Save the WSIs to be used for training in SVS_QUEUE_FOLDER. Change the value of *i* (number of WSIs in one batch of training) in scripts/move_to_training_data.sh as per space availability
- Save the binary maps for the training data in TRAINING_DATA/masks
- Run the file scripts/make_training_data/start.sh to extract patches from WSIs and label them using binary masks
- Run the file scripts/train.sh to start training. You can change the number of epochs in the shell script. 
- New Trained models are in repo/models_cnn/checkpoint

## Testing
- Create folders *svs*, *patches*, *log*, *heatmap_txt* in PRED_DATA_DIR
- Change MODEL in conf/variables.sh to your model name that is stored in repo/models_cnn
- Copy all WSI files to PRED_DATA_DIR/svs
- Run the file scripts/svs_2_tum_percent.sh
- Output are in PRED_DATA_DIR/heatmap_txt and PRED_DATA_DIR/grayscale_heatmaps and PRED_DATA_DIR/tumor_percents
- For the next batch, copy these 3 folders for storing the results and flush all the folders in PRED_DATA_DIR and start again