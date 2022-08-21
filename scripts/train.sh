#!/bin/bash

source ../conf/variables.sh

EXEC_FILE=${BASE_DIR}/training/train_old_model.py

python -u ${EXEC_FILE} --data ${TRAINING_PATCHES} --data_list ${DATA_LIST} --model_folder ${LYM_NECRO_CNN_MODEL_PATH} --model_name ${MODEL} --num_epochs 6

wait;

exit 0