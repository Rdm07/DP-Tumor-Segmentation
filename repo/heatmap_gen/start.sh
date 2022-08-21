#!/bin/bash

cd $(dirname "$0")

source ../../conf/variables.sh

rm -rf ${PRED_DATA_DIR}/json \
	${PRED_DATA_DIR}/patch-level-lym \
	${PRED_DATA_DIR}/patch-level-nec \
	${PRED_DATA_DIR}/patch-level-color \
	${PRED_DATA_DIR}/patch-level-merged
mkdir  ${PRED_DATA_DIR}/json \
	${PRED_DATA_DIR}/patch-level-lym \
	${PRED_DATA_DIR}/patch-level-nec \
	${PRED_DATA_DIR}/patch-level-color \
	${PRED_DATA_DIR}/patch-level-merged

# Copy heatmap files from lym and necrosis prediction models
# to patch-level/ and necrosis/ folders respectively.
bash cp_heatmaps_all.sh ${PATCH_PATH} > ${LOG_OUTPUT_FOLDER}/log.cp_heatmaps_all.txt 2>&1

# Combine patch-level and necrosis heatmaps into one heatmap.
# Also generate high-res and low-res version.
bash combine_lym_necrosis_all.sh > ${LOG_OUTPUT_FOLDER}/log.combine_lym_necrosis_all.txt 2>&1
rm ${HEATMAP_TXT_OUTPUT_FOLDER}/*
cp ${PRED_DATA_DIR}/patch-level-merged/* ${HEATMAP_TXT_OUTPUT_FOLDER}/     #/data/heatmap_txt
cp ${PRED_DATA_DIR}/patch-level-color/* ${HEATMAP_TXT_OUTPUT_FOLDER}/      #/data/heatmap_txt

# Generate meta and heatmap files for high-res and low-res heatmaps.
bash gen_all_json.sh > ${LOG_OUTPUT_FOLDER}/log.gen_all_json.txt 2>&1
cp ${PRED_DATA_DIR}/json/* ${JSON_OUTPUT_FOLDER}/

exit 0
