#!/bin/bash

cd $(dirname "$0")

source ../../conf/variables.sh

python train_valid_split.py ${TRAINING_PATCHES}

exit 0;

