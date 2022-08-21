#!/bin/bash

cd $(dirname "$0")

source ../../conf/variables.sh

EXEC_FILE=${BASE_DIR}/scripts/get_tum_percent.py

for files in ${GRAYSCALE_HEATMAPS_PATH}/*.png; do
    python -u ${EXEC_FILE} ${files}
done

wait;

if [ ! -d ${TUMOR_PER_PATH} ]; then
	mkdir -p ${TUMOR_PER_PATH}
fi

mv ${GRAYSCALE_HEATMAPS_PATH}/*.txt ${TUMOR_PER_PATH}

exit 0
