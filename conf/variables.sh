#!/bin/bash

# Base directories
# Change all the directories in this section, all other are derivatives
export BASE_DIR=/home/ubuntu/DP-Tumor-Segmentation/repo					# root_dir/repo
export PRED_DATA_DIR=/mnt/vol_b/pred_data								# dir to store data for prediction
export SVS_QUEUE_FOLDER=/mnt/vol_b/svs_queue							# dir to store WSI files to use in batches for training
export DOWNLOAD_FOLDER=/mnt/vol_b/tmp									# dir to download new WSIs for training/prediction
export TRAINING_DATA=/mnt/vol_b/strand-data/training_data				# dir to store data for training

# Prediction folders
# Paths of data, log, input, and output
export JSON_OUTPUT_FOLDER=${PRED_DATA_DIR}/heatmap_jsons
export HEATMAP_TXT_OUTPUT_FOLDER=${PRED_DATA_DIR}/heatmap_txt
export LOG_OUTPUT_FOLDER=${PRED_DATA_DIR}/log
export SVS_INPUT_PATH=${PRED_DATA_DIR}/svs
export PATCH_PATH=${PRED_DATA_DIR}/patches
export GRAYSCALE_HEATMAPS_PATH=${PRED_DATA_DIR}/grayscale_heatmaps
export TUMOR_PER_PATH=${PRED_DATA_DIR}/tumor_percents

# Training folders
export DATA_LIST="tumor_data_list.txt"        # Text file to contain subfolders for validating (1st line), training (the rest)
export TRAINING_SVS=${TRAINING_DATA}/svs
export TRAINING_PATCHES=${TRAINING_DATA}/patches
export TRAINING_MASKS=${TRAINING_DATA}/masks

# model is in ${LYM_NECRO_CNN_MODEL_PATH} 
export LYM_NECRO_CNN_MODEL_PATH=${BASE_DIR}/models_cnn
MODEL="RESNET_34_cancer_350px_refined_models_none_0803_0321_0.8321326627701509_5.t7"
export MODEL

if [[ -z "${CUDA_VISIBLE_DEVICES}" ]]; then
	export LYM_CNN_TRAINING_DEVICE=0
	export LYM_CNN_PRED_DEVICE=0
else
	export LYM_CNN_TRAINING_DEVICE=${CUDA_VISIBLE_DEVICES}
	export LYM_CNN_PRED_DEVICE=${CUDA_VISIBLE_DEVICES}
fi