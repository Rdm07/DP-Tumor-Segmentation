#!/bin/bash

cd $(dirname "$0")

source ../../conf/variables.sh

for files in ${HEATMAP_TXT_OUTPUT_FOLDER}/prediction-*; do
    if [[ "$files" == *.low_res ]]; then
        python gen_json_multipleheat.py ${files} ${HEATMAP_VERSION}-low_res  ${SVS_INPUT_PATH} lym 0.5 necrosis 0.5 >> ${LOG_OUTPUT_FOLDER}/log.gen_json_multipleheat.low_res.txt 2>&1
    else
        python gen_json_multipleheat.py ${files} ${HEATMAP_VERSION}-high_res ${SVS_INPUT_PATH} lym 0.5 necrosis 0.5 >> ${LOG_OUTPUT_FOLDER}/log.gen_json_multipleheat.txt 2>&1
    fi
done

exit 0
