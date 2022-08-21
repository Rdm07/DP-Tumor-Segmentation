#!/bin/bash

cd $(dirname "$0")

source ../../conf/variables.sh

WRITE_FOLDER=./grayscale_heatmaps

if [ ! -d ${WRITE_FOLDER} ]; then
    mkdir ${WRITE_FOLDER}
fi

bash get_grayscale_heatmap.sh > ${LOG_OUTPUT_FOLDER}/log.get_grayscale_heatmaps.txt

exit 0
