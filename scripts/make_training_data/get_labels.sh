#!/bin/bash

cd $(dirname "$0")

source ../../conf/variables.sh

python get_labels.py ${TRAINING_PATCHES} ${TRAINING_SVS} ${TRAINING_MASKS} 

exit 0;

